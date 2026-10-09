/**
 * pi-voice-reply — on-request spoken-summary voice replies.
 *
 * The client asks for one with `/voice-last [long|medium|short]`. This
 * extension then asks the model to rewrite an assistant reply *for listening*
 * in one of three tiers — a long listenable version, a medium ~150-word
 * summary, or a short 2–3 sentence gist — and emits it as a custom
 * `voice-reply` message. Each variant is generated on demand, so the message
 * carries only the variant just produced; the client merges it onto the
 * assistant message.
 *
 * agentchatbox (or any RPC/TUI client) renders that custom message as
 * Long/Med/Short speak buttons. The actual audio synthesis happens
 * wherever the client sends the text (agentchatbox POSTs to its /api/tts
 * → Kokoro); this extension only produces the *words*.
 *
 * Why it lives here (in pi, not in agentchatbox): deciding what to say
 * and how to say it is agent logic. agentchatbox stays a transport layer.
 *
 * Why two variants via the session model: the user picks one model for the
 * conversation; by default the rewrite reuses that same model (one auth
 * path, one bill). Set VOICE_REWRITE_MODEL="provider/modelId" (e.g.
 * "google/gemini-2.5-flash") to route JUST the spoken rewrite to a faster /
 * cheaper model — paraphrasing a reply for speech rewards speed over heavy
 * reasoning, so a flash model plus thinking disabled turns a ~40s wait into
 * a few seconds. Ignored (falls back to the session model) if the named
 * model isn't in the registry or has no configured auth.
 */

import type {
	ExtensionAPI,
	ExtensionContext,
} from "@earendil-works/pi-coding-agent";
import {
	createAgentSession,
	DefaultResourceLoader,
	ModelRuntime,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import { homedir } from "node:os";
import { resolve } from "node:path";
import {
	assistantText,
	fallbackNotice,
	keepInModelContext,
	logVoiceFailure,
	parseVoiceLastArgs,
	pickVoiceSource,
} from "./lib.js";

// ── Configuration ──────────────────────────────────────────────────

/**
 * Long variant prompt. Keeps the substance of the reply but renders it as
 * natural spoken prose — tables summarized into a sentence, code skipped
 * (with a one-phrase description of what it did), numbers and versions
 * verbalized, emoji and markdown stripped. This is the "listen to the
 * whole answer" path.
 */
const LONG_PROMPT = [
	"You are preparing an assistant's reply to be read aloud by a text-to-speech system.",
	"The reply is enclosed in quadruple backticks. Rewrite it as natural spoken prose that a person would actually say out loud.",
	"",
	"Rules:",
	"- Keep ALL the substance and detail. Do not drop information.",
	"- Verbalize numbers, versions, and identifiers the way a person would say them (e.g. \"version two point one\", not \"two dot one dot four beta three\").",
	"- Replace any table with one sentence describing what it showed (\"a table comparing X across Y\"), and read the key cells in prose only if they matter.",
	"- Skip code blocks entirely. If a code block's purpose matters to the answer, say what it does in one short phrase instead of reading the code.",
	"- Drop emoji, markdown formatting, sigils, and bare URLs.",
	"- Sound like a person explaining, not reading a document.",
	"- Output ONLY the spoken text, nothing else. No preamble, no quotes.",
].join("\n");

/**
 * Medium variant prompt. A spoken summary of at most 150 words — the main
 * points and key conclusions with enough detail to be useful, but tighter
 * than the long variant. Like long/short this is produced for listening,
 * but the client ALSO renders it as readable text below the reply.
 */
const MEDIUM_PROMPT = [
	"You are preparing an assistant's reply to be read aloud by a text-to-speech system.",
	"The reply is enclosed in quadruple backticks. Summarize it in at most 150 words of natural spoken prose.",
	"",
	"Rules:",
	"- Capture the main points and key conclusions with enough detail to be useful.",
	"- Verbalize numbers, versions, and identifiers the way a person would say them.",
	"- Replace any table with one sentence describing what it showed; skip code blocks (say what they do in one short phrase instead of reading the code).",
	"- Drop emoji, markdown formatting, sigils, and bare URLs.",
	"- Sound like a person explaining, not reading a document.",
	"- Output ONLY the summary, nothing else. No preamble, no quotes.",
].join("\n");

/**
 * Short variant prompt. 2–3 sentences — just the conclusion plus any
 * essential number. The "give me the gist" path.
 */
const SHORT_PROMPT = [
	"You are preparing an assistant's reply to be read aloud by a text-to-speech system.",
	"The reply is enclosed in quadruple backticks. Summarize it in 2 to 3 sentences of natural spoken prose.",
	"",
	"Rules:",
	"- Capture only the key conclusion and any single essential number.",
	"- No code, no tables, no emoji, no markdown.",
	"- Conversational and concise.",
	"- Output ONLY the summary, nothing else. No preamble, no quotes.",
].join("\n");

/**
 * Variant → prompt map. Each spoken tier has its own rewrite prompt.
 * /voice-last <variant> indexes into this.
 */
const VARIANT_PROMPTS = {
	long: LONG_PROMPT,
	medium: MEDIUM_PROMPT,
	short: SHORT_PROMPT,
} as const;
type VoiceVariant = keyof typeof VARIANT_PROMPTS;

// /voice-last argument parsing (variant + optional --match hint) and the
// reply-selection logic live in ./lib.ts, unit-tested there.

// ── Helpers ────────────────────────────────────────────────────────

// assistantText moved to ./lib.ts (unit-tested there).

/**
 * Resolve the model for a spoken-rewrite pass.
 *
 * Defaults to the session's current model (ctx.model). If VOICE_REWRITE_MODEL
 * is set ("provider/modelId"), prefer that model — but only if it resolves
 * in the registry AND has configured auth; otherwise warn and fall back to
 * the session model rather than failing the rewrite. See the file header
 * for the rationale.
 */
function resolveRewriteModel(ctx: ExtensionContext): ExtensionContext["model"] {
	const override = process.env.VOICE_REWRITE_MODEL?.trim();
	if (override) {
		const slash = override.indexOf("/");
		if (slash <= 0 || slash === override.length - 1) {
			console.warn(
				`[pi-voice-reply] VOICE_REWRITE_MODEL must be "provider/modelId"; ignoring "${override}".`,
			);
		} else {
			const provider = override.slice(0, slash);
			const modelId = override.slice(slash + 1);
			const m = ctx.modelRegistry.find(provider, modelId);
			if (m && ctx.modelRegistry.hasConfiguredAuth(m)) return m;
			console.warn(
				`[pi-voice-reply] VOICE_REWRITE_MODEL "${override}" not found or unconfigured; falling back to session model.`,
			);
		}
	}
	return ctx.model;
}

/** Human-readable "provider/modelId" for logs + notifications. */
function modelLabel(m: { provider?: string; modelId?: string; id?: string }): string {
	return `${m.provider ?? "?"}/${m.modelId ?? m.id ?? "?"}`;
}

// logVoiceFailure moved to ./lib.ts (bounded log — trims past 256 KB).

/** Result of a spoken-rewrite pass, including fallback metadata. */
interface RewriteResult {
	text: string | null;
	/** Why the final attempt produced nothing, when the model or API said. */
	error?: string;
	/** Present when the preferred (override) model failed and we fell back. */
	fallback?: { provider: string; modelId: string; error: string };
}

/**
 * Run one rewrite pass with call-time fallback to the session model if the
 * VOICE_REWRITE_MODEL override fails (429 / quota / auth / revoked key).
 *
 * Why call-time: resolveRewriteModel's hasConfiguredAuth only checks that a
 * key EXISTS, not that the quota is alive — free-tier limits are discovered
 * only when the call fails. On failure, retries once with ctx.model (which
 * definitely works — the main reply just used it). Returns the text plus
 * fallback metadata so the caller logs + notifies.
 */
async function rewriteForSpeech(
	ctx: ExtensionContext,
	systemPrompt: string,
	sourceText: string,
): Promise<RewriteResult> {
	const preferred = resolveRewriteModel(ctx);
	const sessionModel = ctx.model;

	if (!preferred) {
		console.warn("[pi-voice-reply] no active model; skipping rewrite");
		return { text: null };
	}

	const overrideActive =
		!!process.env.VOICE_REWRITE_MODEL?.trim() && preferred !== sessionModel;

	// No override configured, or override == session model: single attempt.
	if (!overrideActive || !sessionModel) {
		try {
			return { text: await runRewriteWithModel(ctx, systemPrompt, sourceText, preferred) };
		} catch (err) {
			console.warn("[pi-voice-reply] rewrite failed (no fallback):", err);
			return { text: null, error: err instanceof Error ? err.message : String(err) };
		}
	}

	// Override configured + differs: try preferred, fall back on any error.
	try {
		const text = await runRewriteWithModel(ctx, systemPrompt, sourceText, preferred);
		if (text) return { text };
		throw new Error("model produced no output");
	} catch (err) {
		const error = err instanceof Error ? err.message : String(err);
		const pm = preferred as { provider?: string; modelId?: string; id?: string };
		const provider = pm.provider ?? "?";
		const modelId = pm.modelId ?? pm.id ?? "?";
		console.warn(
			`[pi-voice-reply] override ${provider}/${modelId} failed (${error}); falling back to session model.`,
		);
		try {
			const text = await runRewriteWithModel(ctx, systemPrompt, sourceText, sessionModel);
			return { text, fallback: { provider, modelId, error } };
		} catch (err2) {
			console.warn("[pi-voice-reply] session-model fallback also failed:", err2);
			return {
				text: null,
				error: err2 instanceof Error ? err2.message : String(err2),
				fallback: { provider, modelId, error },
			};
		}
	}
}

/**
 * If the rewrite fell back to the session model, log it and notify. The
 * notification surfaces in the browser; the log file is the durable record for
 * deciding whether to keep VOICE_REWRITE_MODEL.
 */
function reportFallback(ctx: ExtensionContext, result: RewriteResult): void {
	const fb = result.fallback;
	if (!fb) return;

	const sm = ctx.model as { provider?: string; modelId?: string; id?: string } | undefined;
	const sessionLabel = sm ? modelLabel(sm) : "(session model)";

	logVoiceFailure({
		provider: fb.provider,
		modelId: fb.modelId,
		error: fb.error,
		fellBackTo: sessionLabel,
	});
	ctx.ui?.notify(fallbackNotice(fb, sessionLabel), "warning");
}

/**
 * Lazily-built, cached ModelRuntime for ephemeral rewrite sessions.
 *
 * createAgentSession authenticates + resolves models through a ModelRuntime,
 * but ExtensionContext only exposes a ModelRegistry (no runtime, authStorage,
 * or agentDir). Building one fresh per button press would re-read models.json
 * + auth.json every time, so cache it for the process lifetime. Reads the
 * global ~/.pi agent dir — the same place the main session loads from.
 *
 * NOTE: this replaces an earlier `AuthStorage.create()` + `ctx.modelRegistry`
 * pair. The Jul 2026 pi build dropped `AuthStorage` from the package's public
 * exports, so `AuthStorage.create()` threw "Cannot read properties of
 * undefined (reading 'create')" — which is what made every voice reply fail
 * with "model produced no output" (the GLM-5.2 fallback hit the same error).
 * `ModelRuntime` is the supported entry point and IS still exported.
 */
let cachedModelRuntime: Awaited<ReturnType<typeof ModelRuntime.create>> | undefined;
async function getModelRuntime(): Promise<Awaited<ReturnType<typeof ModelRuntime.create>>> {
	if (!cachedModelRuntime) cachedModelRuntime = await ModelRuntime.create({});
	return cachedModelRuntime;
}

/**
 * Build an ephemeral rewrite session with a SPECIFIC model and run one
 * prompt through it. Throws on model error; returns null on empty output.
 * Extracted from rewriteForSpeech so the fallback orchestrator can call it
 * twice (preferred, then session model) without duplicating the session setup.
 */
async function runRewriteWithModel(
	ctx: ExtensionContext,
	systemPrompt: string,
	sourceText: string,
	model: NonNullable<ExtensionContext["model"]>,
) {

	const loader = new DefaultResourceLoader({
		cwd: process.cwd(),
		agentDir: resolve(homedir(), ".pi"),
		systemPromptOverride: () => systemPrompt,
	});
	await loader.reload();

	const { session } = await createAgentSession({
		model,
		tools: [],
		// Paraphrasing prose for speech needs no chain-of-thought; disable
		// thinking entirely so the rewrite isn't billed for (and slowed by)
		// reasoning tokens. Applies whether we fell back to ctx.model or used
		// a VOICE_REWRITE_MODEL override.
		thinkingLevel: "off",
		sessionManager: SessionManager.inMemory(),
		// createAgentSession resolves auth + models through a ModelRuntime.
		// ctx exposes only a ModelRegistry (no runtime/authStorage/agentDir),
		// so pass a lazily-cached ModelRuntime built from the global ~/.pi dir.
		modelRuntime: await getModelRuntime(),
		resourceLoader: loader,
	});

	try {
		const userMessage =
			`The assistant's reply to rewrite for speech:\n\n\`\`\`\`${sourceText}\n\`\`\`\``;
		let out = "";
		// A failed provider call ends the turn with stopReason "error" and no
		// text; keep its message so the failure is not reported as "no output".
		let providerError = "";
		const unsub = session.subscribe((event) => {
			if (event.type === "message_end" && event.message.role === "assistant") {
				out += assistantText(event.message.content);
				const m = event.message as { stopReason?: string; errorMessage?: string };
				if (m.stopReason === "error" && m.errorMessage) providerError = m.errorMessage;
			}
		});
		await session.prompt(userMessage);
		unsub();
		const cleaned = out.trim();
		if (!cleaned && providerError) throw new Error(providerError);
		return cleaned || null;
	} finally {
		session.dispose();
	}
}

// ── Extension entry point ──────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// ── SDK-surface guard ───────────────────────────────────────────
	// A Jul 2026 pi upgrade removed `AuthStorage` from the package's public
	// exports and silently broke EVERY voice reply — the throw was swallowed
	// and resurfaced as the misleading "model produced no output". Guard the
	// exports we now depend on (ModelRuntime / createAgentSession /
	// SessionManager / DefaultResourceLoader); if a future upgrade renames or
	// drops any of them, disable the extension with a clear in-chat warning
	// that names the cause, instead of letting each button press die opaquely.
	// The voice triggers become no-ops until the extension is updated to match.
	const sdkOK =
		typeof createAgentSession === "function" &&
		typeof DefaultResourceLoader === "function" &&
		typeof ModelRuntime?.create === "function" &&
		typeof SessionManager?.inMemory === "function";
	if (!sdkOK) {
		pi.on("session_start", async (_event, ctx) => {
			ctx.ui?.notify(
				"pi-voice-reply disabled itself: this pi version removed/renamed an SDK export it needs " +
					"(ModelRuntime / createAgentSession / SessionManager / DefaultResourceLoader). " +
					"Voice buttons will do nothing until the extension is updated to match — " +
					"see ~/.pi/agent/voice-reply-failures.jsonl.",
				"warning",
			);
		});
		return;
	}

	// Backstop flag: set right before we emit the voice-reply custom
	// message (see emitVoiceReply). If a race ever lets that emit trigger a
	// spurious continuation turn, the message_end handler below blanks it so
	// the client never sees a stray reply. With the idle-defer in place this
	// rarely (never) fires, but it's cheap insurance.
	let spuriousTurnPending = false;

	/**
	 * Emit the voice-reply custom message to clients WITHOUT triggering a
	 * spurious model turn.
	 *
	 * THE BUG THIS FIXES: calling pi.sendMessage(msg, {deliverAs:"steer"})
	 * from inside the agent_end handler routes to sendCustomMessage's
	 * streaming branch — the agent is still mid-run when agent_end fires
	 * (the event is emitted from inside _runAgentPrompt, before the
	 * post-run drain loop exits). The queued steer is then drained as a
	 * CONTINUATION TURN: the model runs again on the same context and its
	 * streamed tokens reach the client as a visible second reply before
	 * message_end can blank them — exactly the "it replied twice"
	 * symptom users hit when saying "reply in voice".
	 *
	 * THE FIX: don't emit from inside the handler. Defer the emit to the
	 * macrotask queue and poll ctx.isIdle() until the run loop has fully
	 * unwound (isStreaming false). sendCustomMessage then takes its idle
	 * branch, which persists the custom_message and emits message_start /
	 * message_end but queues NOTHING — so there is no continuation and no
	 * spurious turn.
	 *
	 * We must NOT await isIdle() inside the handler itself: the run loop
	 * cannot exit (and flip isStreaming off) until the handler returns, so
	 * blocking on it would deadlock. Fire-and-forget is the only safe shape.
	 */
	const emitVoiceReply = (
		ctx: ExtensionContext | undefined,
		// Partial by design: /voice-last <variant> emits ONE variant at a
		// time, so only the generated key is present. The client merges it
		// onto the assistant message without clearing previously-generated
		// variants.
		// `match` echoes the --match hint that picked the source reply, so the
		// client merges the variant onto THAT message rather than the newest one.
		details: { long?: string; medium?: string; short?: string; match?: string },
	): void => {
		const fire = (): void => {
			spuriousTurnPending = true;
			pi.sendMessage(
				{
					customType: "voice-reply",
					content: "voice reply ready",
					display: true,
					details,
				},
				{ deliverAs: "steer", triggerTurn: false },
			);
		};
		// Deadline guards against an agent that never goes idle (another
		// extension keeping it busy, or a stuck state): after 5s, emit
		// anyway and rely on the blanking backstop instead of leaking the
		// buttons forever.
		const deadline = Date.now() + 5000;
		const poll = (): void => {
			const idle = typeof ctx?.isIdle === "function" ? ctx.isIdle() : true;
			if (idle || Date.now() >= deadline) fire();
			else setTimeout(poll, 30);
		};
		// setTimeout(0) defers out of the current handler/run-loop onto the
		// macrotask queue, so the handler returns immediately and the run
		// loop is free to unwind and flip isStreaming off.
		setTimeout(poll, 0);
	};

	pi.on("session_start", async () => {
		spuriousTurnPending = false;
	});

	/**
	 * Strip custom (non-conversational) messages from the context before
	 * each LLM call. Our voice-reply message (customType: "voice-reply")
	 * is a display-only notification for the browser; it must never enter
	 * the model's context. Without this, the steer-delivered custom
	 * message's content ("voice reply ready") leaks into the prompt.
	 */
	pi.on("context", async (event) => {
		// biome-ignore lint/suspicious/noExplicitAny: AgentMessage union is wide
		const filtered = event.messages.filter((m: any) => keepInModelContext(m));
		if (filtered.length === event.messages.length) return;
		return { messages: filtered };
	});

	/**
	 * Suppress the spurious assistant turn that sendMessage(steer)
	 * inevitably triggers. The steer delivery mechanism kicks the agent
	 * loop to process the delivered message; even though we strip custom
	 * messages from context, the model still runs and produces a brief
	 * junk reply. We blank it out here so it never reaches the client.
	 * The spuriousTurnPending flag is set just before sendMessage and
	 * consumed by the first assistant message_end after it.
	 */
	pi.on("message_end", async (event) => {
		if (spuriousTurnPending && event.message.role === "assistant") {
			spuriousTurnPending = false;
			return {
				message: {
					...event.message,
					content: [{ type: "text", text: "" }],
					stopReason: "stop",
				},
			};
		}
	});

	/**
	 * input event: never blocks or transforms the user's prompt; it only
	 * clears the spurious-turn backstop (see below).
	 */
	pi.on("input", async () => {
		// A fresh user submission starts a real, user-driven turn. Clear
		// any stale `spuriousTurnPending` flag here so it can never leak
		// out of the voice-reply turn it was meant for and blank the NEXT
		// real reply.
		//
		// Why this is needed: sendMessage(deliverAs:"steer", triggerTurn:false)
		// sets spuriousTurnPending=true expecting a spurious follow-up turn
		// to fire (whose assistant message_end we'd then blank). But with
		// triggerTurn:false against an idle agent, that spurious turn NEVER
		// fires — the steer just sits in the queue. The flag then leaks,
		// and the next real turn's assistant message_end hits it and gets
		// blanked, making that reply vanish from the client (it shows up
		// as an empty assistant message that agentchatbox drops). Clearing
		// on input guarantees the flag is dead before any real reply runs,
		// while still allowing a genuine spurious turn (if one ever fires
		// right after sendMessage, before any new input) to be blanked.
		spuriousTurnPending = false;
		return { action: "continue" };
	});

	/**
	 * /voice-last [variant] [--match "<hint>"] — retroactive voice reply. Rewrites
	 * an assistant message into ONE spoken variant — long | medium | short
	 * (default long) — and emits a voice-reply custom message carrying just that
	 * variant. The client merges it onto the assistant message so multiple
	 * presses (e.g. medium then short) accumulate without regeneration.
	 *
	 * This is what the browser's per-message LongTTS/MedTTS/ShortTTS buttons
	 * call: each press generates exactly the tier it asked for, not all three at
	 * once. Reads the assistant text straight from the session branch (no
	 * cache), so it's always accurate even across reloads.
	 *
	 * Without a hint it voices the most recent reply, as it always did. The
	 * browser adds --match "<opening words>" when the press came from an older
	 * row: the hint selects that reply, and is echoed in `details.match` so the
	 * client shows the text under the row whose button was pressed instead of
	 * the newest one.
	 */
	pi.registerCommand("voice-last", {
		description:
			"Generate a spoken voice reply (long|medium|short, default long) for the most recent assistant message, or for the reply named by --match",
		handler: async (args, ctx) => {
			const { variant, match } = parseVoiceLastArgs(args);
			// Session-branch messages, in order, for pickVoiceSource().
			const entries = ctx.sessionManager.getBranch();
			const messages = [];
			for (const entry of entries) {
				// biome-ignore lint/suspicious/noExplicitAny: entry union is wide
				if (entry && (entry as any).type === "message") messages.push((entry as any).message);
			}
			const picked = pickVoiceSource(messages, match);
			const lastText = picked.text;
			if (!lastText.trim()) {
				ctx.ui.notify("No assistant message to voice yet.", "warning");
				return;
			}

			if (ctx.ui?.setStatus) ctx.ui.setStatus("voice-reply", "preparing voice reply…");
			try {
				const result = await rewriteForSpeech(
					ctx,
					VARIANT_PROMPTS[variant],
					lastText,
				);
				reportFallback(ctx, result);
				const text = result.text;
				if (!text) {
					if (ctx.ui?.notify) {
						ctx.ui.notify(`Voice reply: ${result.error ?? "model produced no output"}.`, "warning");
					}
					return;
				}
				// Emit only the requested variant; the client merges it onto
				// the assistant message alongside any already-generated ones.
				// `match` (only when the hint actually selected the reply) tells
				// the client which message that is.
				emitVoiceReply(ctx, {
					[variant]: text,
					...(picked.matched && match ? { match } : {}),
				});
			} catch (err) {
				console.warn("[pi-voice-reply] retroactive rewrite failed:", err);
				if (ctx.ui?.notify) {
					ctx.ui.notify(
						`Voice reply failed: ${err instanceof Error ? err.message : String(err)}`,
						"warning",
					);
				}
			} finally {
				if (ctx.ui?.setStatus) ctx.ui.setStatus("voice-reply", undefined);
			}
		},
	});
}
