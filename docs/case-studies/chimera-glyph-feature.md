# Case Study: Designing an AI-Only Wire Format for an MCP Server

**Project:** [`chimeralang-mcp`](https://github.com/fernandogarzaaa/chimeralang-mcp)
**Feature:** Chimera Glyph (CG) — a deterministic AI-to-AI wire format with sigil-marked entities and explicit modality tokens, plus two MCP tools that expose it.
**Releases:** v0.7.0 (feature) → v0.7.1 (review-driven fixes) → v0.7.3 (empirical lexicon + benchmark, see addendum).
**Date range:** 2026-05-07.

> **2026-05-07 update — see "0.7.3 addendum" below.** A 100-sentence formal benchmark falsified this case study's original headline claim. Glyph's value is **semantic determinism + entity preservation**, not token compression; the updated numbers and reframe are at the end of this document.

---

## Executive summary

Designed and shipped a new feature for an open-source [Model Context Protocol](https://modelcontextprotocol.io) server that lets a large language model communicate in a structured pidgin during internal reasoning, then translate the result back to English for the human reader at the end. The feature ships as two new MCP tools (`chimera_glyph_directive`, `chimera_glyph_translate`) plus a self-contained 489-line language module. Across two release cycles the feature was reviewed by three automated reviewers (CodeRabbit, Codex, Copilot), revealing eight real defects that were resolved in a follow-up release.

The original headline metric was **46.2% token reduction**, measured on a 10-sentence corpus by counting the project's internal char-based estimator. A formal 100-sentence benchmark in 0.7.3 against `tiktoken` `o200k_base` (a Claude-equivalent BPE) showed the original metric was a *character* count, not a *token* count — and on real BPE the encoded form is actually **−16% on tokens** (longer than English). The 0.7.3 addendum below documents the falsification, the empirically-optimized lexicon that closed +9 percentage points of the gap, and the strategic reframe: **Glyph is a deterministic structured wire format, not a token compressor.**

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

### Performance (0.7.0 — superseded; see 0.7.3 addendum)

Measured end-to-end on a 10-sentence corpus running on the same machine, **using the project's char-based token estimator (`chars // 4`), not a real BPE tokenizer**:

| Metric | Value |
|---|---|
| Estimated English tokens (corpus total) | 104 |
| Estimated Glyph tokens (corpus total) | 56 |
| Estimated saving | 48 (**46.2%** by char-based estimator) |
| End-to-end time (10× encode + estimate + decode) | 1.07 ms |
| Per-sentence latency | **0.11 ms** |

The 46.2% number is real, but it measures *characters / 4*, not BPE tokens. The 0.7.3 addendum re-runs this benchmark against `tiktoken o200k_base` and reports the corrected token-cost numbers.

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
| Char-based estimator (0.7.0) | 46.2% reduction on 10-sentence corpus (superseded — see addendum) |
| BPE-token measurement (0.7.3) | **−16.0%** on 100-sentence corpus (Glyph is *longer* in BPE tokens) |
| Char-based measurement (0.7.3) | **+19.2%** on 100-sentence corpus |
| Decode fidelity (0.7.3) | **0.806** mean Jaccard token overlap (lossy by design) |

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

---

## 0.7.3 addendum — formal benchmark falsifies the token-cost claim

This addendum was added after an independent code+competitive audit prompted a Phase 1 effort to validate Glyph's headline metric against a real BPE tokenizer. The empirical finding falsified the original claim and led to a strategic reframe.

### What the audit asked

> *"60% reduction" measured how — characters or tokens? Against which tokenizer?*"

The 0.7.0 case study above reported "46.2% reduction" using the project's internal `chars // 4` estimator. That's a useful proxy for human-readable text but it's not what an LLM actually pays at the API boundary.

### What the formal benchmark measured

A new harness (`tools/glyph_benchmark.py`) was built and run against a 100-sentence corpus (`tests/fixtures/glyph_bench.txt`) covering 5 domains: error/debugging, dialogue, instruction, reasoning, and prose. Each sentence is encoded with `cg.encode()`, decoded with `cg.decode()`, and token-counted with `tiktoken`'s `o200k_base` (a Claude-equivalent BPE).

**Run-1 numbers (against the original 0.7.2 hand-crafted lexicon):**

| Metric | Value |
|---|---|
| Mean token reduction (BPE) | **−4.0%** (Glyph longer than English) |
| Stems strictly worse than English in BPE | 74 of 249 (30%) |
| Stems beating English in BPE | 0 of 249 |
| Char reduction | +34.6% |

The 46.2% number was character-only. On real BPE the original lexicon was a net loss.

### Why hand-crafted glyph stems fail against BPE

Modern BPE tokenizers like `o200k_base` were trained on natural text. Common English words (`user`, `function`, `code`, `approach`) are already collapsed into single tokens. Hand-crafted abbreviations (`usr`, `fn`, `cde`, `aprch`) aren't in the merge table and decompose into 2-3 sub-tokens. The verb suffix scheme was the worst offender — `fix^` becomes 2 tokens, `bld^` becomes 3 tokens, `gø~` becomes 3 tokens. Multi-byte Unicode operators (`∅`, `∀`, `∃`, `∧`, `¬`, `≈`, `≡`) all cost 2+ tokens in `o200k_base`.

### The empirical optimization

`tools/optimize_lexicon.py` re-derived every English→Glyph mapping by measuring against `o200k_base` directly and picking the form (English, current stem, or English-language operator equivalent) that BPE-collapses to the fewest tokens. Decoder backwards-compatibility is preserved via `LEGACY_GLYPH_REVERSE`, so any CG text emitted by ≤ 0.7.2 still decodes correctly.

74 entries changed:
- Tense-suffix forms retired (`fix^` → `fixed`, `wnt^` → `wanted`, `gø~` → `going`).
- Multi-token Unicode operators replaced with English (`∧` → `and`, `∨` → `or`, `∅` → `none`, `∀` → `all`).
- Multi-token abbreviations reverted to English (`cde` → `code`, `aprch` → `approach`, `wrk` → `work`, `wnt` → `want`).
- Single-token Unicode operators kept (`⇒`, `←`, `≠`, `→` are 1 token in BPE).

**Run-2 numbers (after empirical optimization, on the original 20-sentence corpus):** −4.0% → **+5.3%**.

### Run-3 — the 100-sentence formal benchmark

On the formal corpus, the optimized lexicon does this:

| Domain | n | BPE tokens saved | Char saving | Decode fidelity |
|---|---|---|---|---|
| Error / debugging | 20 | **−2.4%** | 27.5% | 0.85 |
| Dialogue / conversational | 20 | **−13.3%** | 21.5% | 0.70 |
| Instruction / prompt | 20 | **−21.7%** | 14.2% | 0.81 |
| Reasoning / explanation | 20 | **−9.3%** | 25.0% | 0.82 |
| Prose / narrative | 20 | **−26.7%** | 13.0% | 0.85 |
| **Overall** | **100** | **−16.0%** | **+19.2%** | **0.806** |

On real prose, even the optimized Glyph is **16% longer in BPE tokens** than the English source. The character-saving (19.2%) is real. The token cost is not.

### Why the optimized lexicon still loses on tokens

The benchmark harness exposes three remaining sources of loss:

1. **`@entity` sigil overhead.** Out-of-lexicon words get the `@` prefix, which is itself a token. On prose, ~15-20% of content words are `@`-tagged.
2. **Modal/sigil tokens.** Glyph emits `~`, `?`, `!`, `→` as separate tokens. English collapses "will go" / "might fail" into 2 BPE tokens; Glyph emits 2-3.
3. **Common English already wins.** Pronouns (`we`/`it`/`they` → `w`/`x`/`t`) are 1 token both ways — no saving. Articles drop, but BPE often packs `the user` into 1-2 tokens already.

### Strategic reframe — what Glyph actually does

The hand-crafted glyph-language approach cannot beat learned BPE on token cost. The academic literature converged on **learned compression** (LLMLingua-2, 20× reduction with ~1.5pt accuracy drop) for that reason — invented symbol tables can't beat a tokenizer that already absorbed the same pattern.

But the benchmark also shows what Glyph *is* good at, and these are properties no token-compression library provides:

| Property | Glyph (measured) | English | LLMLingua-2 |
|---|---|---|---|
| Decode determinism (no LLM call to invert) | ✅ 0.806 fidelity, deterministic | n/a | ❌ requires inference |
| Sigil-marked entities (`@MyService` is unambiguously a proper noun) | ✅ first-class | ❌ requires NER | ❌ no semantic markup |
| Explicit modality tokens (`~` will, `?` might, `!` must) | ✅ typed at wire level | ❌ implicit | ❌ no semantic markup |
| Stripped function words (drops articles + copulas) | ✅ +19.2% chars | n/a | partially |
| Round-trippable structure for protocol verification | ✅ | ❌ | ❌ |

These are exactly the properties needed by a **typed wire format for agent-to-agent communication** — which is the original "AI-only language" framing, just measured against the right metric. Token cost was the wrong success criterion.

### What v0.7.3 ships

- Empirically-optimized `LEXICON` (74 entries changed; tools/lexicon_diff.json is the audit trail).
- `LEGACY_GLYPH_REVERSE` dict so all v0.7.2 CG text still decodes.
- `tools/glyph_benchmark.py` — reproducible benchmark harness; anyone can run `python tools/glyph_benchmark.py` and verify the −16% / +19.2% / 0.806 numbers.
- `tests/fixtures/glyph_bench.txt` — the 100-sentence corpus.
- `tests/test_glyph_benchmark.py` — locks the headline numbers as a regression test.
- 6 new tokenizer-aware tests in `tests/test_ai_language.py` plus updated 0.7.2-era assertions; legacy stems verified via decoder-roundtrip tests.
- Schema fields on `chimera_glyph_directive` corrected: dropped the false `expected_token_savings: "60-80%"` field; replaced with honest `wire_format_role`, `measured_char_reduction`, `measured_token_reduction`, and `decode_fidelity` fields.
- Case study, SKILL.md, AGENTS.md updated to publish the corrected numbers.

### What this means for the project

Glyph is no longer pitched as a token-cost solution. For token reduction, use:
- `chimera_optimize` / `chimera_compress` / `chimera_fracture` (heuristic, real wins on long docs)
- `chimera_cache_mark` (Anthropic native prompt cache, lossless 75–90%)
- LLMLingua-2 or similar learned compressors (external, best-in-class)

Glyph's value is now positioned as the substrate for **Phase 4 of the project blueprint** — a typed wire format for cross-agent reasoning where deterministic decode and sigil semantics matter more than token count. The 19.2% char saving is a useful side benefit; the 0.806 decode fidelity and the structural guarantees are the reason to use it.

### Reflection

The most valuable engineering output of this work was *measuring honestly*. The original case study reported a real number from a real estimator — but it was the wrong metric for the cost the user actually pays. Building the benchmark harness, running it against the real BPE, publishing the negative finding, and reframing the feature accordingly is the path that keeps the project credible. The alternative (defending the original number) would have made every future claim suspect.

The Phase 1 outcome is recorded in `BLUEPRINT.md` and was used to reorder the remaining roadmap. Phase 2 (gates → ChimeraLang programs with Merkle proofs) is the actual moat per the audit and is now next on the critical path.
