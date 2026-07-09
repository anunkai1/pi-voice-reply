/**
 * Unit tests for the pure helpers in extensions/lib.ts.
 *
 * Covers the voice-request trigger matcher (positive cases + the false-
 * positive regressions that motivated the word-boundary rewrite and dropping
 * the bare "in voice" entry), the content-text extractor, and the bounded
 * failure log. The model-rewrite orchestration and emit/defer logic in
 * index.ts are exercised in production by pi; these tests pin the logic
 * they depend on.
 */

import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assistantText, logVoiceFailure, userRequestsVoice } from "../extensions/lib.js";

// ── userRequestsVoice ──────────────────────────────────────────────

describe("userRequestsVoice", () => {
	it("matches the canonical positive phrases (case-insensitive)", () => {
		expect(userRequestsVoice("please reply in voice")).toBe(true);
		expect(userRequestsVoice("REPLY WITH VOICE")).toBe(true);
		expect(userRequestsVoice("ok, voice reply please")).toBe(true);
		expect(userRequestsVoice("can you read it aloud?")).toBe(true);
		expect(userRequestsVoice("read it out loud")).toBe(true);
		expect(userRequestsVoice("say it out loud")).toBe(true);
		expect(userRequestsVoice("speak your answer")).toBe(true);
		expect(userRequestsVoice("answer out loud")).toBe(true);
	});

	it("matches when embedded in a longer sentence with punctuation", () => {
		expect(userRequestsVoice("Great, reply in voice, thanks.")).toBe(true);
		expect(userRequestsVoice("please respond out loud.")).toBe(true);
		expect(userRequestsVoice("could you read your reply aloud?")).toBe(true);
	});

	// ── The false-positive regressions this change fixes ──
	it("does NOT match 'I work in voice acting' (bare 'in voice' was removed)", () => {
		expect(userRequestsVoice("I work in voice acting")).toBe(false);
		expect(userRequestsVoice("we're in voice chat")).toBe(false);
	});

	it("does not match a plain request with no voice intent", () => {
		expect(userRequestsVoice("fix the bug in the parser")).toBe(false);
		expect(userRequestsVoice("what time is it")).toBe(false);
		expect(userRequestsVoice("")).toBe(false);
	});

	it("keeps coverage for 'reply in voice' even though bare 'in voice' is gone", () => {
		// The removed bare entry is fully covered by these:
		expect(userRequestsVoice("reply in voice")).toBe(true);
		expect(userRequestsVoice("reply with voice")).toBe(true);
	});
});

// ── assistantText ──────────────────────────────────────────────────

describe("assistantText", () => {
	it("returns a plain string unchanged", () => {
		expect(assistantText("hello")).toBe("hello");
	});

	it("concatenates only the text parts of a content array", () => {
		const content = [
			{ type: "text", text: "first" },
			{ type: "thinking", text: "ignored" },
			{ type: "text", text: "second" },
		];
		expect(assistantText(content)).toBe("first\nsecond");
	});

	it("returns '' for non-string / non-array input", () => {
		expect(assistantText(undefined)).toBe("");
		expect(assistantText(null)).toBe("");
		expect(assistantText(42)).toBe("");
		expect(assistantText({})).toBe("");
		expect(assistantText([])).toBe("");
	});
});

// ── logVoiceFailure (bounded) ──────────────────────────────────────

describe("logVoiceFailure", () => {
	let tmpHome: string;
	const realHome = process.env.HOME;

	beforeEach(async () => {
		tmpHome = await mkdtemp(join(tmpdir(), "pvr-home-"));
		process.env.HOME = tmpHome;
	});

	afterEach(async () => {
		process.env.HOME = realHome;
		await rm(tmpHome, { recursive: true, force: true });
	});

	it("appends a JSON line and never throws", () => {
		expect(() =>
			logVoiceFailure({
				provider: "venice",
				modelId: "gemini-3-flash-preview",
				error: "boom",
				fellBackTo: "minimax/M3",
			}),
		).not.toThrow();
	});

	it("actually writes a parseable JSON line to the log file", async () => {
		logVoiceFailure({
			provider: "venice",
			modelId: "gemini-3-flash-preview",
			error: "boom",
			fellBackTo: "minimax/M3",
		});
		const logPath = join(tmpHome, ".pi", "agent", "voice-reply-failures.jsonl");
		const raw = await readFile(logPath, "utf8");
		const line = raw.trim().split("\n").at(-1)!;
		const parsed = JSON.parse(line);
		expect(parsed).toMatchObject({
			provider: "venice",
			modelId: "gemini-3-flash-preview",
			error: "boom",
			fellBackTo: "minimax/M3",
		});
		expect(typeof parsed.ts).toBe("string");
	});

	it("trims the log when it exceeds the size cap (bounded growth)", async () => {
		const logPath = join(tmpHome, ".pi", "agent", "voice-reply-failures.jsonl");
		// Pre-seed a file over the 256 KB cap with many lines.
		await mkdir(join(tmpHome, ".pi", "agent"), { recursive: true });
		const big = `${JSON.stringify({
			ts: "x",
			provider: "p",
			modelId: "m",
			error: "e",
			fellBackTo: "f",
		})}\n`.repeat(5000);
		await writeFile(logPath, big, "utf8");
		expect(Buffer.byteLength(big)).toBeGreaterThan(256 * 1024);

		logVoiceFailure({
			provider: "z",
			modelId: "marker",
			error: "trim-test",
			fellBackTo: "g",
		});
		const after = await readFile(logPath, "utf8");
		// Trimmed to ≤ ~201 lines (200 kept + the new one) — not the original 5000+1.
		expect(after.trim().split("\n").length).toBeLessThanOrEqual(201);
		expect(after).toContain("marker");
	});
});
