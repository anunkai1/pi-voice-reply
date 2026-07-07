/**
 * pi-voice-reply — on-trigger spoken-summary voice replies.
 *
 * When the user asks for a voice reply (e.g. "reply in voice", "say it
 * back", "/voice"), this extension waits for the agent's normal text reply
 * to finish, then asks the same model to rewrite that reply *for listening*
 * in two variants — a long listenable version and a short concise one —
 * and emits both as a custom `voice-reply` message.
 *
 * agentchatbox (or any RPC/TUI client) renders that custom message as two
 * speak buttons. The actual audio synthesis happens wherever the client
 * sends the text (agentchatbox POSTs to its /api/tts → Kokoro); this
 * extension only produces the *words*.
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
	AuthStorage,
	createAgentSession,
	DefaultResourceLoader,
	SessionManager,
} from "@earendil-works/pi-coding-agent";
import { homedir } from "node:os";
import { resolve } from "node:path";

// ── Configuration ──────────────────────────────────────────────────

/**
 * Phrases that trigger a voice reply. Matched case-insensitively as a
 * substring anywhere in the user's message. Kept deliberately broad — the
 * point is "any natural way the user asks for it works" rather than an
 * exhaustive list. The /voice command is the guaranteed explicit fallback.
 */
const TRIGGER_PHRASES = [
	"reply in voice",
	"reply with voice",
	"voice reply",
	"in voice",
	"say it back",
	"say it out loud",
	"read it back",
	"read it aloud",
	"read it out loud",
	"read your reply aloud",
	"talk to me",
	"speak your answer",
	"speak your reply",
	"answer out loud",
	"respond out loud",
];

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

// ── Helpers ────────────────────────────────────────────────────────

function userRequestsVoice(text: string): boolean {
	const lower = text.toLowerCase();
	return TRIGGER_PHRASES.some((p) => lower.includes(p));
}

/** Extract the text content blocks from a message's content (role-agnostic
 * — works on assistant replies AND user messages, including steered ones
 * whose content is the same [{type:"text",text}] shape). */
function assistantText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.filter(
			// biome-ignore lint/suspicious/noExplicitAny: content part shape varies
			(c: any) => c && typeof c === "object" && c.type === "text" && typeof c.text === "string",
		)
		.map((c: { text: string }) => c.text)
		.join("\n");
}

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

/**
 * Run one rewrite pass via a throwaway in-memory sub-agent using the given
 * model and system prompt. Returns the spoken-text variant, or null if the
 * model couldn't be resolved or returned nothing useful.
 *
 * Mirrors the pattern in @s1m0n38/pi-voice's generateSpeechText(): an
 * ephemeral createAgentSession with an empty tool set and an in-memory
 * session manager, so the rewrite never touches the user's real session.
 */
async function rewriteForSpeech(
	ctx: ExtensionContext,
	systemPrompt: string,
	sourceText: string,
): Promise<string | null> {
	const model = resolveRewriteModel(ctx);
	if (!model) {
		console.warn("[pi-voice-reply] no active model; skipping rewrite");
		return null;
	}

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
		authStorage: AuthStorage.create(),
		modelRegistry: ctx.modelRegistry,
		resourceLoader: loader,
	});

	try {
		const userMessage =
			`The assistant's reply to rewrite for speech:\n\n\`\`\`\`${sourceText}\n\`\`\`\``;
		let out = "";
		const unsub = session.subscribe((event) => {
			if (event.type === "message_end" && event.message.role === "assistant") {
				out += assistantText(event.message.content);
			}
		});
		await session.prompt(userMessage);
		unsub();
		const cleaned = out.trim();
		return cleaned || null;
	} finally {
		session.dispose();
	}
}

// ── Extension entry point ──────────────────────────────────────────

export default function (pi: ExtensionAPI) {
	// Turn-level flag, set by the input handler when the user asks for voice.
	let voiceRequested = false;
	let currentCtx: ExtensionContext | undefined;
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
		details: { long: string; short: string },
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

	pi.on("session_start", async (_event, ctx) => {
		currentCtx = ctx;
		voiceRequested = false;
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
		const filtered = event.messages.filter(
			// biome-ignore lint/suspicious/noExplicitAny: AgentMessage union is wide
			(m: any) => !(m.role === "custom"),
		);
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
		// Detect voice intent from a USER message delivered mid-turn (a steer).
		// The `input` hook above only fires on the prompt() path — i.e. when the
		// agent is idle and the user submits a fresh prompt. A voice phrase sent
		// as a steer while the agent is running takes a different route: it is
		// queued and drained between tool calls as a plain user message
		// (message_start + message_end, role:user), never passing through
		// emitInput(). Without this branch, "reply in voice" steered mid-run
		// would set no flag and produce no spoken variant — the exact blind
		// spot we hit. Steers are drained before the next assistant turn, so
		// setting the flag here lands it in time for the agent_end handler to
		// act on. Idempotent with the input handler: both may set the same
		// boolean for a fresh prompt (prompt also emits message_end role:user);
		// that's harmless, and agent_end consumes the flag exactly once.
		if (event.message.role === "user") {
			const text = assistantText(event.message.content);
			if (text && userRequestsVoice(text)) voiceRequested = true;
		}
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
	 * input event: detect voice intent. Always continue — we never block or
	 * transform the user's actual prompt; we just remember that they wanted
	 * a voice reply for this turn.
	 */
	pi.on("input", async (event, _ctx) => {
		const text = typeof event.text === "string" ? event.text : "";
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
		if (userRequestsVoice(text)) {
			voiceRequested = true;
		}
		return { action: "continue" };
	});

	/**
	 * Explicit /voice command — the guaranteed trigger that doesn't depend
	 * on phrase detection. Sets the flag for the NEXT reply.
	 */
	pi.registerCommand("voice", {
		description: "Request a spoken voice reply for the next answer",
		handler: async (_args, ctx) => {
			voiceRequested = true;
			ctx.ui.notify(
				"Voice reply requested — speak buttons will appear on the next answer.",
				"info",
			);
		},
	});

	/**
	 * /voice-last — retroactive voice reply. Rewrites the most recent
	 * assistant message in the session (regardless of when it was
	 * produced) into long + short spoken variants and emits the same
	 * voice-reply custom message the proactive path emits. This is what
	 * the browser's per-message 🎫 button calls: press it after reading
	 * a reply to hear a listenable version, no trigger phrase needed.
	 *
	 * Reads the last assistant text straight from the session branch
	 * (no cache), so it's always accurate even across reloads.
	 */
	pi.registerCommand("voice-last", {
		description: "Generate a spoken voice reply for the most recent assistant message",
		handler: async (_args, ctx) => {
			// Find the last assistant message in the session branch.
			const entries = ctx.sessionManager.getBranch();
			let lastText = "";
			for (let i = entries.length - 1; i >= 0; i--) {
				const entry = entries[i];
				// biome-ignore lint/suspicious/noExplicitAny: entry union is wide
				if (entry && (entry as any).type === "message") {
					const msg = (entry as any).message;
					if (msg && msg.role === "assistant") {
						lastText = assistantText(msg.content);
						if (lastText.trim()) break;
					}
				}
			}
			if (!lastText.trim()) {
				ctx.ui.notify("No assistant message to voice yet.", "warning");
				return;
			}

			if (ctx.ui?.setStatus) ctx.ui.setStatus("voice-reply", "preparing voice reply…");
			try {
				const [long, short] = await Promise.all([
					rewriteForSpeech(ctx, LONG_PROMPT, lastText),
					rewriteForSpeech(ctx, SHORT_PROMPT, lastText),
				]);
				if (!long && !short) {
					if (ctx.ui?.notify) ctx.ui.notify("Voice reply: model produced no output.", "warning");
					return;
				}
				emitVoiceReply(ctx, { long: long ?? "", short: short ?? "" });
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

	/**
	 * agent_end: the full reply is in. If voice was requested, fire two
	 * parallel rewrite passes and emit the result as a custom message.
	 */
	pi.on("agent_end", async (event, ctx) => {
		currentCtx = ctx;
		if (!voiceRequested) return;
		// Consume the flag immediately so a subsequent plain turn stays quiet.
		voiceRequested = false;

		const messages = event.messages ?? [];
		// Walk back to the last assistant message (there may be toolResult
		// messages after it).
		let lastAssistantText = "";
		for (let i = messages.length - 1; i >= 0; i--) {
			const m = messages[i];
			if (m && m.role === "assistant") {
				lastAssistantText = assistantText(m.content);
				if (lastAssistantText.trim()) break;
			}
		}
		if (!lastAssistantText.trim()) return;

		if (ctx.ui?.setStatus) {
			ctx.ui.setStatus("voice-reply", "preparing voice reply…");
		}

		try {
			const [long, short] = await Promise.all([
				rewriteForSpeech(ctx, LONG_PROMPT, lastAssistantText),
				rewriteForSpeech(ctx, SHORT_PROMPT, lastAssistantText),
			]);

			if (!long && !short) {
				if (ctx.ui?.notify) {
					ctx.ui.notify("Voice reply: model produced no output.", "warning");
				}
				return;
			}

			// Emit the voice-reply custom message. agentchatbox recognizes
			// customType "voice-reply" and renders the long/short speak
			// buttons; other clients ignore it. Deferred until idle so it
			// doesn't trigger a spurious continuation turn — see emitVoiceReply.
			emitVoiceReply(ctx, { long: long ?? "", short: short ?? "" });
		} catch (err) {
			console.warn("[pi-voice-reply] rewrite failed:", err);
			if (ctx.ui?.notify) {
				ctx.ui.notify(
					`Voice reply failed: ${err instanceof Error ? err.message : String(err)}`,
					"warning",
				);
			}
		} finally {
			if (ctx.ui?.setStatus) {
				ctx.ui.setStatus("voice-reply", undefined);
			}
		}
	});
}
