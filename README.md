# pi-voice-reply

On-trigger **spoken-summary voice replies** for the [pi coding agent](https://github.com/earendil-works/pi).

When the user asks for a voice reply ("reply in voice", "say it back", etc.) or
runs `/voice`, this extension waits for the agent's normal text reply to finish,
then asks the **same model** to rewrite that reply *for listening* in two
variants — a **long** listenable version and a **short** concise summary — and
emits both as a custom `voice-reply` message that a client (e.g.
[agentchatbox](https://github.com/anunkai1/agentchatbox)) renders as two speak
buttons.

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

Any of these (matched case-insensitively as **word-boundary** phrases
anywhere in the user message) request a voice reply for the current turn:

- "reply in voice", "reply with voice", "voice reply"
- "say it back", "say it out loud"
- "read it back", "read it aloud", "read it out loud", "read your reply aloud"
- "talk to me", "speak your answer", "speak your reply"
- "answer out loud", "respond out loud"

> The bare "in voice" entry was removed — it was the main false-positive
> source (matched "I work in voice acting", "in voice chat"). The intent
> is fully covered by "reply in voice" / "voice reply". Word boundaries
> also prevent matches inside compound words.

Or the explicit command: `/voice`.

When triggered, the next assistant reply gets the two speak buttons attached.

## How it works

1. `input` event → phrase detection → set a per-turn `voiceRequested` flag.
2. `agent_end` → if flagged, take the last assistant message's text and run two
   **parallel** rewrite passes via ephemeral `createAgentSession` sub-agents
   using `ctx.model` (the same model driving the conversation):
   - **Long** — keeps all substance; tables→one-sentence summary; code skipped
     (with a one-phrase description of what it did); numbers/versions verbalized;
     emoji/markdown dropped.
   - **Short** — 2-3 sentences: just the conclusion + any essential number.
3. Emit `pi.sendMessage({ customType: "voice-reply", details: { long, short } })`.
4. Custom messages are stripped from the model's context (`context` event), and
   the spurious follow-up turn that `sendMessage(steer)` triggers is blanked, so
   the conversation stays clean.

## Why the same model (not a separate small model)?

One auth path, one bill, and quality that tracks the main reply. A big frontier
model writes a much better spoken summary than a small local model. The trade
is latency + cost on triggered turns (two extra LLM calls); a future
`VOICE_REWRITE_MODEL` override could route just the rewrite to a cheaper model,
but that's deliberately not wired yet — keep it simple.

## Client integration

Clients recognize the custom message by `role === "custom"` and
`customType === "voice-reply"`, then read `details.long` and `details.short`.
agentchatbox renders each as a speak button that calls its existing TTS path.

## Development

```bash
npm install
npm test          # vitest — pure-helper unit tests (lib.ts)
```

The pure helpers (trigger detection, content extraction, the bounded
failure log) live in [`extensions/lib.ts`](extensions/lib.ts) and are
unit-tested in [`tests/lib.test.ts`](tests/lib.test.ts). Tool/command wiring
and the model-rewrite orchestration stay in `extensions/index.ts` (exercised
in production by `pi`, which provides the `@earendil-works/pi-coding-agent`
types at load time).

## License

MIT.
