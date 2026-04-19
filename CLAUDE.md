# ChimeraLang MCP — Project Memory

## Token-saving policy (always active)

Before processing large documents, conversation histories, or code blobs, always use these MCP tools to stay within token budgets:

1. **`chimera_optimize`** — Compress a raw text/code block to ~2–5% of its original size via structural + entity + frequency extraction. Use on any document > 500 chars before including it in context.

2. **`chimera_compress`** — Proportionally truncate a message history list to fit a token budget. Use when passing conversation history to a model call.

3. **`chimera_fracture`** — Full pipeline: runs `chimera_optimize` on each document AND `chimera_compress` on messages in one call, with a quality gate. Prefer this over calling the two tools separately when you have both documents and messages.

### When to apply

| Situation | Tool |
|-----------|------|
| Single large doc/code blob | `chimera_optimize` |
| Long conversation history | `chimera_compress` |
| Both docs + messages | `chimera_fracture` |
| Quick inline text (< 200 chars) | skip |
