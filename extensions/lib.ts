/**
 * Pure helpers for pi-voice-reply — trigger detection and content extraction,
 * split out of index.ts so they can be unit-tested without standing up the
 * pi session machinery (createAgentSession / SessionManager / model calls).
 *
 * The tool/command wiring, model-rewrite orchestration, and emit/defer logic
 * stay in index.ts; only the pure functions live here.
 */

import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
// ── Voice-request detection ────────────────────────────────────────

/**
 * Phrases that trigger a voice reply. Kept deliberately broad — the point is
 * "any natural way the user asks for it works" rather than an exhaustive
 * list. The /voice command is the guaranteed explicit fallback.
 *
 * NOTE: the bare "in voice" was removed — it was the single biggest
 * false-positive source (matched "I work in voice acting", "in voice chat",
 * "in voice-over"). The real intent is fully covered by "reply in voice",
 * "reply with voice", and "voice reply" below.
 */
export const TRIGGER_PHRASES = [
	"reply in voice",
	"reply with voice",
	"voice reply",
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
] as const;

/**
 * Precompiled matcher: word-boundary alternation of all triggers, e.g.
 *   /\b(reply in voice|voice reply|...)\b/i
 *
 * Word boundaries harden against the phrase sitting inside a larger token
 * (e.g. a future single-word trigger embedded in a compound word, or a
 * trigger butted against punctuation like "voice-reply"). One regex test
 * replaces N `.includes()` scans per message.
 */
const TRIGGER_RE = new RegExp(
	`\\b(${TRIGGER_PHRASES.map((p) => p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\b`,
	"i",
);

/** True if the user's message asks for a voice reply (case-insensitive). */
export function userRequestsVoice(text: string): boolean {
	return TRIGGER_RE.test(text);
}

// ── Content extraction ─────────────────────────────────────────────

/**
 * Extract the concatenated text content blocks from a message's content
 * (role-agnostic — works on assistant replies AND user messages, including
 * steered ones whose content is the same [{type:"text",text}] shape).
 */
export function assistantText(content: unknown): string {
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

// ── Failure log (bounded) ──────────────────────────────────────────

const VOICE_FAILURE_LOG_DIR = () => resolve(process.env.HOME ?? homedir(), ".pi", "agent");
const VOICE_FAILURE_LOG = () => join(VOICE_FAILURE_LOG_DIR(), "voice-reply-failures.jsonl");

/** Soft cap. The log records voice-model fallbacks; one JSON line per event,
 *  so even a very chatty deployment stays small. Trim when it exceeds this. */
const VOICE_FAILURE_LOG_MAX_BYTES = 256 * 1024; // 256 KB

/**
 * Append a voice-model failure to the durable log so the operator can inspect
 * frequency and decide whether to keep VOICE_REWRITE_MODEL.
 * Path: ~/.pi/agent/voice-reply-failures.jsonl (one JSON object per line).
 * Bounded: if the file grows past VOICE_FAILURE_LOG_MAX_BYTES, it's trimmed to
 * its most recent lines so an unattended dotfile can't grow without limit.
 * Best-effort — never throws.
 */
export function logVoiceFailure(entry: {
	provider: string;
	modelId: string;
	error: string;
	fellBackTo: string;
}): void {
	try {
		const logPath = VOICE_FAILURE_LOG();
		try {
			const buf = readFileSync(logPath);
			// Trim to the last ~200 lines if the file has grown large. Synchronous
			// read+rewrite is fine here — the file is tiny and writes are rare.
			if (buf.byteLength > VOICE_FAILURE_LOG_MAX_BYTES) {
				const lines = buf.toString("utf8").split("\n").filter(Boolean);
				writeFileSync(logPath, `${lines.slice(-200).join("\n")}\n`);
			}
		} catch {
			/* file may not exist yet — fine, append creates it */
		}
		mkdirSync(VOICE_FAILURE_LOG_DIR(), { recursive: true });
		appendFileSync(logPath, JSON.stringify({ ts: new Date().toISOString(), ...entry }) + "\n");
	} catch {
		/* best-effort */
	}
}

// ── Fallback reporting (pure decision logic) ───────────────────────

/** One distinct model fallback that occurred during a voice reply. */
export interface VoiceFallback {
	provider: string;
	modelId: string;
	error: string;
}

/** Input shape for collectFallbacks — mirrors RewriteResult in index.ts
 *  without pulling in the SDK/ExtensionContext type. */
export type FallbackResult = { fallback?: VoiceFallback };

/**
 * Collect ALL distinct fallbacks from a long+short rewrite pair, not just
 * the first. Long + short run in parallel and can each fall back (usually
 * to the same override model). Dedup by `provider/modelId:error` so an
 * identical pair yields one entry, while two genuinely different failures
 * (e.g. a 429 then a 500) both survive instead of the second being silently
 * dropped. Returns the distinct list, in first-seen order.
 */
export function collectFallbacks(results: FallbackResult[]): VoiceFallback[] {
	const seen = new Set<string>();
	const out: VoiceFallback[] = [];
	for (const r of results) {
		const fb = r.fallback;
		if (!fb) continue;
		const key = `${fb.provider}/${fb.modelId}:${fb.error}`;
		if (seen.has(key)) continue;
		seen.add(key);
		out.push(fb);
	}
	return out;
}

/**
 * Build a single user-facing notification string for N fallbacks. One toast
 * summarizing all of them (don't spam N toasts for a pair that usually
 * failed identically). `sessionLabel` is the human-readable model the voice
 * rewrite fell back to (e.g. "minimax/MiniMax-M3").
 */
export function fallbackNotice(fallbacks: VoiceFallback[], sessionLabel: string): string {
	const lines = fallbacks.map((fb) => `${fb.provider}/${fb.modelId} (${fb.error})`);
	const tail = `used ${sessionLabel} instead. Log: ~/.pi/agent/voice-reply-failures.jsonl`;
	if (fallbacks.length === 1) return `Voice model ${lines[0]} failed; ${tail}`;
	return `${fallbacks.length} voice-model fallbacks (${lines.join("; ")}); ${tail}`;
}
