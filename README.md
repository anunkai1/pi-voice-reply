# pi-voice-reply

On-request **spoken-summary voice replies** for the [pi coding agent](https://github.com/earendil-works/pi).

A client asks for one with `/voice-last [long|medium|short]`. The extension asks
the **same model** to rewrite an assistant reply *for listening* and emits the
result as a custom `voice-reply` message that a client (e.g.
[agentchatbox](https://github.com/anunkai1/agentchatbox)) plays or shows as a
speak button.

This extension only produces the **words**. Audio synthesis happens wherever the
client sends the text (agentchatbox POSTs to its `/api/tts` → Kokoro). The
extension itself is TTS-engine-agnostic.

## Why

Raw agent replies aren't speech-friendly: tables read cell-by-cell, code blocks
get read aloud, big numbers and emoji get blurted out. Even with a great voice
model, that sounds like someone reading a spreadsheet. This extension fixes the
*words* so the voice model can shine.

It runs inside the pi process because deciding what to say and how to say it is
agent logic — agentchatbox (or any RPC/TUI client) stays a thin transport.

## Install

```bash
pi install /path/to/pi-voice-reply       # local
# or, if published:
pi install git:github.com/anunkai1/pi-voice-reply
```

Then restart pi (or run `/reload`). The extension is global (all sessions).

## Trigger

Only the explicit command: `/voice-last [long|medium|short] [--match "<opening words>"]`
(default `long`). Without `--match` it voices the newest assistant reply; with it,
the reply that starts with those words. There are no trigger phrases and no
`/voice` command; agentchatbox's Voice mode and speak buttons call `/voice-last`.

## How it works

1. `/voice-last` takes the chosen assistant message's text and runs ONE rewrite
   pass via an ephemeral `createAgentSession` sub-agent (the session model, or
   `VOICE_REWRITE_MODEL` if set):
   - **Long** — keeps all substance; tables→one-sentence summary; code skipped
     (with a one-phrase description of what it did); numbers/versions verbalized;
     emoji/markdown dropped.
   - **Medium** — a spoken summary of at most 150 words.
   - **Short** — 2-3 sentences: just the conclusion + any essential number.
2. Emit `pi.sendMessage({ customType: "voice-reply", details: { <variant>: text } })`.
3. Custom messages are stripped from the model's context (`context` event), and
   the spurious follow-up turn that `sendMessage(steer)` triggers is blanked, so
   the conversation stays clean.

## Why the same model (not a separate small model)?

One auth path, one bill, and quality that tracks the main reply. A big frontier
model writes a much better spoken summary than a small local model. The trade
is latency + cost on triggered turns (one extra LLM call per variant). Set
`VOICE_REWRITE_MODEL="provider/modelId"` to route just the rewrite to a faster
or cheaper model; if that model fails (quota, auth, revoked key), the rewrite
falls back to the session model, shows a warning and logs the failure to
`~/.pi/agent/voice-reply-failures.jsonl`.

## Client integration

Clients recognize the custom message by `role === "custom"` and
`customType === "voice-reply"`, then read whichever of `details.long`, `details.medium` and `details.short` is present.
agentchatbox renders each as a speak button that calls its existing TTS path.

## Development

```bash
npm install
npm test          # vitest — pure-helper unit tests (lib.ts)
```

The pure helpers (content extraction, reply selection, the bounded
failure log) live in [`extensions/lib.ts`](extensions/lib.ts) and are
unit-tested in [`tests/lib.test.ts`](tests/lib.test.ts). Tool/command wiring
and the model-rewrite orchestration stay in `extensions/index.ts` (exercised
in production by `pi`, which provides the `@earendil-works/pi-coding-agent`
types at load time).

## License

MIT.
