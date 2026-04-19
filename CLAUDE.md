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

---

## Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
