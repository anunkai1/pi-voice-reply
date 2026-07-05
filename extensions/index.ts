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
 * conversation; using the same model for the rewrite means one auth path,
 * one bill, and quality that tracks the main reply. (A future
 * VOICE_REWRITE_MODEL override could route just the rewrite to a cheaper
 * model, but that's deliberately not wired yet — keep it simple.)
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

/** Extract the text content blocks from an assistant message. */
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
	const model = ctx.model;
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
	// Set true right before we emit the voice-reply custom message, cleared
	// by the next before_agent_start. The spurious turn is the one the
	// model starts because sendMessage fed "voice reply ready" into context.
	let spuriousTurnPending = false;

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
		if (userRequestsVoice(text)) {
			voiceRequested = true;
		}
		return { action: "continue" };
	});

	/**
	 * Explicit /voice command — the guaranteed trigger that doesn't depend
	 * on phrase detection.
	 */
	pi.registerCommand("voice", {
		description: "Request a spoken voice reply for the next/current answer",
		handler: async (_args, ctx) => {
			voiceRequested = true;
			ctx.ui.notify(
				"Voice reply requested — speak buttons will appear on the next answer.",
				"info",
			);
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

			// Emit a custom message. agentchatbox recognizes customType
			// "voice-reply" and renders the long/short speak buttons; other
			// clients ignore it.
			//
			// sendMessage with steer emits the message_start/message_end
			// events immediately (so the client renders the buttons right
			// away) but also feeds the message into the model context,
			// triggering a spurious follow-up turn (the agent replies to
			// its own "voice reply ready" text). We cancel that turn via
			// ctx.abort() the moment it starts — see the before_agent_start
			// guard below, which uses the `spuriousTurnPending` flag set
			// here. nextTurn was the alternative but it defers event
			// emission entirely, so the browser would never see the buttons.
			spuriousTurnPending = true;
			pi.sendMessage(
				{
					customType: "voice-reply",
					content: "voice reply ready",
					display: true,
					details: {
						long: long ?? "",
						short: short ?? "",
					},
				},
				{ deliverAs: "steer", triggerTurn: false },
			);
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
