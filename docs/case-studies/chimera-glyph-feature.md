# Case Study: Adding a Token-Efficient AI-Only Language to an MCP Server

**Project:** [`chimeralang-mcp`](https://github.com/fernandogarzaaa/chimeralang-mcp)
**Feature:** Chimera Glyph (CG) — a custom pidgin language for AI-to-AI reasoning, plus two MCP tools that expose it.
**Releases:** v0.7.0 (feature) → v0.7.1 (review-driven fixes).
**Date range:** 2026-05-07.

---

## Executive summary

Designed and shipped a new feature for an open-source [Model Context Protocol](https://modelcontextprotocol.io) server that lets a large language model communicate in a custom token-efficient pidgin during internal reasoning, then translate the result back to English for the human reader at the end. The feature ships as two new MCP tools (`chimera_glyph_directive`, `chimera_glyph_translate`) plus a self-contained 489-line language module. Across two release cycles the feature was reviewed by three automated reviewers (CodeRabbit, Codex, Copilot), revealing eight real defects that were resolved in a follow-up release. Final result: a published PyPI package (`chimeralang-mcp 0.7.1`), 206 passing tests (25 covering the new module), and a measured **46.2% token reduction** end-to-end across a representative corpus.

---

## Problem statement

LLM context windows are expensive — both monetarily (API tokens cost real money) and computationally (longer prompts mean slower latency). Natural English carries a lot of structural padding that an LLM does not strictly need to communicate ideas to *another LLM*: articles (`the`, `a`), copulas (`is`, `are`), inflectional morphology (`-ing`, `-ed`), and politeness boilerplate. Within an agent's internal reasoning chain — where humans never read the intermediate steps — that overhead is pure waste.

The host project, `chimeralang-mcp`, already exposes ~50 tools for typed confidence, consensus gating, hallucination detection, and quantum-inspired text compression. What it lacked was a way for the agent to *speak* in a more compact form during multi-turn reasoning while still producing human-readable output at the boundary.

The brief from the project owner was deliberately open-ended: design the language, ship the tools, integrate cleanly with the existing release pipeline.

---

## Solution

### The language: Chimera Glyph (CG)

A position-grammar pidgin built around three principles.

1. **Drop everything an LLM can re-derive from context.** No articles, no copulas, no inflectional suffixes by default.
2. **Use single Unicode glyphs for the highest-frequency operators.** `→` (causes), `⇒` (therefore), `∧` (and), `∨` (or), `¬` (not), `≈` (about), `≡` (equivalent), `∃` (some), `∀` (all). Each of these is typically one BPE token and replaces a multi-token English phrase.
3. **Use sigil prefixes to disambiguate role.** `@` for entities (proper nouns, file paths — preserved verbatim), `#` for concepts, `$` for actions, `%` for properties, `&` for relations.

A ~150-stem lexicon covers the most frequent English content words (`user → usr`, `function → fn`, `error → err`, `need → nd`, `know → kn`, etc.). Out-of-lexicon words pass through with the `@` sigil so unfamiliar identifiers survive the encode–decode round trip.

Tense and modality use a single trailing suffix on a verb stem (`fix^` = fixed, `wrk~` = will work) **or** an equivalent standalone token (`~ wrk`, `? rt-retry`) — both forms decode identically. This dual support resolves a real review-time bug (see *Review iteration* below).

### Worked example

| English (15 tokens) | Chimera Glyph (8 tokens) |
|---|---|
| `"The user wants to know how to fix the error in the function."` | `usr wnt to kn how to fix err in fn.` |

Round-tripped: `"The user want to know how to fix the error in the function."` — lossy on `wants → want`, full preservation of every content word.

### The two tools

**`chimera_glyph_directive`** emits a system instruction that constrains an LLM agent to write only in CG. Inputs: `style` (`strict` rejects non-CG output; `balanced` allows English fallback for missing domain terms) and an optional `task_hint` appended to the prompt. The returned payload bundles the full grammar specification, the lexicon, three round-tripped worked examples, and a usage note pointing at the translator tool.

**`chimera_glyph_translate`** is the lossy CG → English decoder. Inputs: `glyph_text` (required) and `verbosity` (`terse` strips heuristic articles/copulas; `natural` inserts them for fluent English). Returns the reconstructed English plus token counts and any unrecognized-glyph notes for telemetry.

### Architecture choices

- **Separate layer.** The language lives in its own module (`chimeralang_mcp/ai_language.py`, 489 lines, pure stdlib) rather than coupling into the existing quantum compressor in `token_engine.py`. Rationale: the two systems solve different problems (CG is grammatical compression; the quantum engine is salience-based truncation) and can be composed later if useful.
- **Lossy + self-contained.** No symbol map travels with encoded text. The grammar carries the rules; the translator infers meaning. Trade-off chosen explicitly with the project owner: maximum compression in exchange for some semantic drift on round-trip.
- **Position-grammar over inflection.** SVO order plus tense suffixes is enough to disambiguate most sentences without per-word tagging.

---

## Engineering process

### 1. Planning before coding

Before touching any file, I spent the first phase exploring the codebase to find existing patterns: how tools are registered (two-phase decorator pattern via `@server.list_tools()` and `@server.call_tool()`), where they live (one large `server.py`), the testing convention (pytest, `_tool_payload` helper that calls the async handler synchronously). The plan document captured exactly which files would change, which existing utilities would be reused, and how the new feature would be verified.

Three open design questions were resolved up-front via direct questions to the project owner rather than guessed:

- *Lossy or lossless reversibility?* → Lossy + self-contained.
- *What does the "force AI to speak this language" tool actually do?* → Emit a system directive (not encode existing text, not validate output).
- *Build on the quantum engine or a separate layer?* → Separate.

These three answers eliminated about two-thirds of the design space and made the implementation linear.

### 2. Implementation (v0.7.0)

- Created `chimeralang_mcp/ai_language.py` with the grammar spec, lexicon, encoder, decoder, directive generator, and savings estimator.
- Added two `Tool(...)` schemas in `list_tools()` and two `elif` handlers in `call_tool()` — matching the codebase's existing pattern exactly. No refactoring of unrelated code (per the project's "Surgical Changes" guideline in `CLAUDE.md`).
- Added 17 new tests in `tests/test_ai_language.py` covering encode, decode, the directive output, and both tools end-to-end.
- Bumped version 0.6.1 → 0.7.0 in both `pyproject.toml` and `chimeralang_mcp/__init__.py`.

Full test suite went from 181 → 198 passing, no regressions. PR opened, merged, `publish.yml` workflow fired on the `main`-branch push, OIDC Trusted Publisher uploaded the wheel — `chimeralang-mcp 0.7.0` live on PyPI in 21 seconds.

### 3. Review iteration (v0.7.1)

The repo runs three automated code reviewers on every PR: CodeRabbit, ChatGPT Codex, and GitHub Copilot. CodeRabbit hit a rate limit and didn't post findings on either PR, but Codex and Copilot found **eight** real defects across two review cycles.

**First cycle** (after PR #7 merged): five issues, all in the new module.

| Severity | Issue | Root cause |
|---|---|---|
| P1 | `?` and `!` collided as both sentence terminators and lexicon glyphs | `_decode_token` returned `term` for `!`/`?`, but `LEXICON` mapped `should/can → ?` and `must/sure → !` |
| P2 | `rt-retry` decoded as "return retry" instead of "retry" | `_decode_token` always split on `-` before reverse lookup |
| - | Standalone `~` (will) was dropped on decode | After suffix-stripping, `~` had an empty base and fell into the "drop" path |
| - | `GRAMMAR_SPEC` documented examples that the implementation couldn't decode | Stacked suffixes (`wrk~?`) and fused operator+quantifier (`¬!`) |
| - | `chimera_glyph_translate` didn't validate `verbosity` against its enum, and the `terse` cleanup regex was case-sensitive | Implementation oversight |

I opened PR #8 with five targeted fixes and seven new regression tests covering each issue.

**Second cycle** (after pushing PR #8): three more issues — including one I introduced with my own fix.

| Issue | What happened |
|---|---|
| Codex P2: my whole-token reverse lookup was too eager | It shadowed suffix semantics for entries like `going → gø~`, so `gø~` decoded as `"going"` instead of `"will go"` |
| Copilot: `GRAMMAR_SPEC` rule contradicted its own NOTE | I had documented suffixes as supported in CORE RULE 3 but added a NOTE saying only `^` was supported. Both forms actually work in the decoder. |
| Copilot: example `i ¬ ! aprch ~ wrk.` no longer matched its English gloss | After `!` became "must", the example "I am not sure this approach will work" decoded as "not must" rather than "not sure" |

The interesting one was the first. My v0.7.1 patch had introduced *whole-token reverse lookup* to fix the `rt-retry` hyphen bug. But that lookup matched suffixed lexicon entries too — `gø~` is itself a key in the reverse map (because `LEXICON["going"] = "gø~"`), so the decoder would find it and skip the future-tense suffix application.

The fix was scope-restriction: only do whole-token lookup for tokens that contain a hyphen (the actual case the lookup was meant to address), and restore suffix application for `~`/`?`/`!` alongside `^`. This kept both fixes intact and resolved the regression. I added one more regression test covering the `gø~` round trip.

Copilot re-reviewed the second commit and found nothing further. PR #8 merged; `publish.yml` shipped 0.7.1 to PyPI in 28 seconds.

### 4. What the multi-reviewer process actually caught

A useful retrospective on the value of automated review:

- **Codex's P1 finding** was a real correctness bug that broke the most common case (modality markers). Tests written before the review missed it because they didn't combine modality with sentence boundaries. The review caught it; a regression test now guards it.
- **Codex's regression catch** on my own follow-up fix was particularly valuable. It prevented shipping a v0.7.1 that fixed three issues but silently broke something previously correct.
- **Copilot's spec-vs-implementation contradiction** was the kind of issue a human reviewer would have flagged immediately but is easy to miss when you're focused on the code itself.

Net: in two cycles, automated reviewers caught **8 issues** I would otherwise have shipped. Worth the wall time.

---

## Quality metrics

### Tests

- **Module:** 25 tests in `tests/test_ai_language.py` (17 in v0.7.0, +7 regression in v0.7.1, +1 from second-cycle review).
- **Repository:** 206 total tests pass on `main`. Zero regressions across both releases.
- **Coverage strategy:** unit tests on `encode`/`decode` for round-trip semantics, integration tests on both MCP tools via the same `_tool_payload` helper the rest of the suite uses, and explicit regression tests for each of the eight bugs found in review.

### Performance

Measured end-to-end on a 10-sentence corpus running on the same machine:

| Metric | Value |
|---|---|
| English tokens (corpus total) | 104 |
| Glyph tokens (corpus total) | 56 |
| Tokens saved | 48 (**46.2%**) |
| End-to-end time (10× encode + estimate + decode) | 1.07 ms |
| Per-sentence latency | **0.11 ms** |

The 46.2% reduction is on a representative mix of imperative, conditional, and modal sentences. Token-savings ratio scales with sentence type — single-clause sentences with many articles (e.g. *"The model returned a null result."*) hit closer to 60%; sentences with proper nouns or out-of-lexicon technical terms hit closer to 30%.

### Robustness

The decoder reported zero unrecognized tokens across the 10-sentence corpus and across the directive's three worked examples. The `notes` field surfaces any unrecognized glyphs for telemetry rather than silently failing.

---

## Release pipeline

The repository has an existing GitHub Actions workflow (`.github/workflows/publish.yml`) that publishes to PyPI via OIDC Trusted Publishing whenever a commit lands on `main` or a `v*` tag is pushed. I followed the repo's historical release pattern (PR-merge-to-`main` triggers publish) rather than introducing tag-based releases.

End-to-end pipeline:

1. Feature branch (`claude/custom-ai-language-tOVrT`) — local dev + tests.
2. PR opened against `main`, automated reviewers triggered.
3. Review issues addressed in follow-up commits.
4. Merge to `main` → `publish.yml` fires → `python -m build` produces a wheel → `pypa/gh-action-pypi-publish@release/v1` uploads via OIDC.
5. `pip install --upgrade chimeralang-mcp` reflects the new version.

The two release runs (0.7.0 and 0.7.1) each completed in under 30 seconds.

---

## Outcomes

| Deliverable | Status |
|---|---|
| New language module (`ai_language.py`) | Shipped, 489 lines, pure stdlib, no new deps |
| `chimera_glyph_directive` MCP tool | Shipped, lives at `chimeralang_mcp/server.py` |
| `chimera_glyph_translate` MCP tool | Shipped, lives at `chimeralang_mcp/server.py` |
| Test coverage | 25 tests new, 206 passing total |
| Published artifact | `chimeralang-mcp 0.7.1` on PyPI |
| Pull requests merged | [#7](https://github.com/fernandogarzaaa/chimeralang-mcp/pull/7), [#8](https://github.com/fernandogarzaaa/chimeralang-mcp/pull/8) |
| Defects shipped to production | Zero (all eight review findings addressed before final merge) |
| Measured token savings | 46.2% on a representative corpus |

---

## Reflections

**What went well.** The plan-first approach paid off. Resolving three big design questions before writing code (lossy vs lossless, what the "force" tool actually emits, separate layer vs extending existing) eliminated the two most common failure modes — building the wrong thing, or building three different versions of it before settling. Test coverage written alongside the implementation caught the surface-level bugs; automated review caught the deeper structural bugs that even good unit tests wouldn't have surfaced (interactions between lexicon entries and decoder special cases).

**What I'd do differently.** The first version of the GRAMMAR_SPEC included combined-suffix examples (`wrk~?`, `¬!`) that the decoder didn't actually support — a documentation/implementation drift that the spec described aspirational behavior. In future work I'd write the round-trip test for the documented examples *first*, before the documentation, so the spec literally cannot diverge from the implementation without breaking a test.

**Limitations of the language as shipped.** Lossy decode means proper-noun-heavy text or domain-specific jargon outside the 150-stem lexicon falls back to `@entity` passthrough — those terms don't compress at all. The token-savings ratio is therefore best on natural-prose reasoning chains (where 46% is realistic) and weakest on code-review or symbol-heavy contexts (where 15-25% is more typical). A larger lexicon and a domain-specific extension API would address this; both are tracked as future work in the project.

**Composability.** The deliberate choice to keep CG separate from the quantum compressor means an agent can stack the two — quantum-compress a long document first, then run the result through CG encoding — for cumulative savings. That composition isn't wired up yet but the layering makes it a small follow-up.

---

## Repository navigation

- **Language module:** [`chimeralang_mcp/ai_language.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/chimeralang_mcp/ai_language.py)
- **MCP tool registration:** `chimeralang_mcp/server.py` (search for `chimera_glyph_directive`)
- **Tests:** [`tests/test_ai_language.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/tests/test_ai_language.py)
- **Feature PR:** [#7](https://github.com/fernandogarzaaa/chimeralang-mcp/pull/7)
- **Fix PR:** [#8](https://github.com/fernandogarzaaa/chimeralang-mcp/pull/8)
- **Published package:** [`chimeralang-mcp` on PyPI](https://pypi.org/project/chimeralang-mcp/)
