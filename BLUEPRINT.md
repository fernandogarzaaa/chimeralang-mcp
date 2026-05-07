# BLUEPRINT.md — chimeralang-mcp toward AI bottleneck-solving

> **Vision:** Make chimeralang-mcp the only LLM toolkit where every reasoning step is **typed, compressed, and cryptographically verifiable** — solving three of AI's biggest bottlenecks at once: token cost, hallucination, and reproducibility.

This document tracks the four-phase plan to turn the existing ChimeraLang VM + Glyph + Merkle proofs into a defensible competitive moat. Update status checkboxes as work lands.

---

## Why this exists

Independent audit (2026-05-07) identified the project's **real moat**: it's the only stack combining (1) a typed probabilistic-computation language with gates and proofs, (2) a deterministic AI-to-AI compression DSL, and (3) Merkle-chain provenance over LLM runs. Each pillar today has a stronger external incumbent (LLMLingua-2, Patronus Lynx, DSPy), but **no incumbent owns the intersection**. The four phases below close the gaps that prevent the intersection from becoming the moat.

---

## Phase 1 — Tokenizer-aware Glyph

**Goal:** Push Glyph compression from the current self-reported 46.2% to a measured ≥60% on a public corpus, by aligning the lexicon with Claude's actual BPE token boundaries.

**Why this first:** Independent, externally measurable, and fully de-risks the headline number. If the spike fails, we know early before committing to weeks of structural work in Phase 2.

### Sub-tasks
- [x] **Spike (P1.S1):** Tokenize current Glyph stems with a Claude-equivalent BPE (tiktoken o200k_base as proxy). Measure: how many of the ~150 stems already map to single tokens? How many are 2+ token? Identify the worst offenders.
- [ ] **Go/No-go decision (P1.S2):** If headroom ≥15% additional savings is plausible from re-stemming, commit to full optimization. If <10%, pivot Phase 1 to a different angle (e.g., better operator coverage).
- [ ] **Lexicon optimization (P1.S3):** Greedy-replace 2+ token stems with 1-token alternatives, scored by `tokens_saved × decode_fidelity_kept`. Preserve all decoder semantics — every old stem must still decode if it appears in legacy CG text.
- [ ] **Public benchmark corpus (P1.S4):** Assemble a 100-sentence representative corpus across coding, reasoning, dialogue, and instruction domains. Source: existing test fixtures + hand-curated additions. Commit as `tests/fixtures/glyph_bench.txt`.
- [ ] **Measurement harness (P1.S5):** `tools/glyph_benchmark.py` — encodes corpus, counts tokens with tiktoken o200k_base + Anthropic API (when key available), reports pct_saved, decode fidelity (BLEU or token-overlap), and per-sentence breakdown.
- [ ] **Regression tests (P1.S6):** Lock the new measurements into `tests/test_glyph_benchmark.py` so future changes can't silently regress.
- [ ] **Public writeup (P1.S7):** Update `docs/case-studies/chimera-glyph-feature.md` with the new numbers + side-by-side comparison vs LLMLingua-2 (cost-only, since they require GPU inference).

### Success criteria
- [ ] Measured ≥55% reduction on the 100-sentence benchmark (60% is the stretch goal)
- [ ] Decode fidelity ≥75% token overlap with original English (lossy is fine, but not gibberish)
- [ ] Zero regressions on existing 222 tests
- [ ] Benchmark numbers reproducible by anyone running `python tools/glyph_benchmark.py`

### Risks
1. **BPE doesn't favor short Latin stems.** o200k_base groups frequent words into single tokens; "user" might already be 1 token, making "usr" a *worse* fit. The spike must measure this honestly.
2. **Anthropic tokenizer ≠ o200k_base exactly.** Mitigation: validate top-20 candidates against the real `messages.count_tokens` API before locking the lexicon.
3. **Decode fidelity drops below 75%.** Mitigation: keep the old stem as a decoder-only fallback in REVERSE_LEXICON; only swap the encoder direction.

---

## Phase 2 — Gates compile to ChimeraLang programs

**Goal:** Every `chimera_gate` / `chimera_verify` / `chimera_deliberate` / `chimera_quantum_vote` invocation emits a ChimeraLang AST that is:
1. Re-executable via `chimera_run` (deterministic outputs given the same inputs)
2. Provable via `chimera_prove` (Merkle hash of the run)
3. Type-checkable via `chimera_typecheck` (catches malformed candidate sets at the boundary)

**Why:** This is the actual moat. DSPy compiles prompts but cannot prove a run. Lynx is a single judge. **No competitor can give you a hash that says "this exact reasoning step happened with these exact inputs and outputs."**

### Sub-tasks
- [ ] **AST design spike (P2.S1):** Sketch the ChimeraLang representation for `chimera_gate` (simplest gate). Decide: do we add new opcodes or compose existing ones? Aim for ≤3 new opcodes total across all four gate tools.
- [ ] **Reference impl (P2.S2):** `chimera_gate` emits its AST as part of the response envelope. Add a new field `provenance.program: str` containing the ChimeraLang source.
- [ ] **Re-execution test (P2.S3):** Round-trip test: run `chimera_gate` → extract `program` → `chimera_run(program)` → assert outputs match.
- [ ] **Apply to remaining tools (P2.S4):** Replicate to `chimera_verify`, `chimera_deliberate`, `chimera_quantum_vote`. Each should ship with a re-execution test.
- [ ] **Merkle hash for free (P2.S5):** Wire `chimera_prove` into the gate pipeline so users get a hash without an extra call.

### Success criteria
- [ ] All four gate tools return a `provenance.program` field with a runnable AST
- [ ] Round-trip tests pass deterministically (run twice, get same Merkle hash)
- [ ] No more than 3 new ChimeraLang opcodes added
- [ ] Documentation in SKILL.md and AGENTS.md updated

### Risks
1. **Gate operations don't have clean ChimeraLang representations.** Mitigation: spike on `chimera_gate` first; if the AST is ugly, we add opcodes, not stretch existing ones.
2. **Determinism breaks under floating-point drift.** Mitigation: all gate scoring already uses integer Jaccard counts; verify no float ops sneak in.

---

## Phase 3 — ChimeraBench

**Goal:** Publish a corpus of verifiable agent tasks. Each task has:
1. A Glyph-encoded prompt (the input)
2. A ChimeraLang spec (the success criterion)
3. A canonical Merkle hash (the proof of correct execution)

This makes chimeralang-mcp a **research artifact**, not just a tool. Anyone can run their own agent and check whether it matches the canonical hash.

**Why:** Owning a benchmark is more defensible than owning an implementation. Patronus did this with HaluBench → Lynx. Stanford did this with HELM. Without ChimeraBench, every claim of "verifiable agent reasoning" is just marketing.

### Sub-tasks
- [ ] **Task taxonomy (P3.S1):** Decide the 5–10 task families (e.g., claim-verification, multi-hop reasoning, structured extraction, instruction-following, calibrated abstention).
- [ ] **Corpus authoring (P3.S2):** 50–100 tasks total, evenly distributed across families. Each task is JSON: `{glyph_prompt, chimerlang_spec, canonical_hash, metadata}`.
- [ ] **Harness (P3.S3):** `tools/chimerabench/run.py` takes a candidate agent (any callable) + the corpus, runs each task, computes Merkle hashes, reports pass rate.
- [ ] **Reference implementation (P3.S4):** Run Claude through the harness ourselves; publish the canonical hashes.
- [ ] **Publish as standalone repo or subdir (P3.S5):** Decide hosting: separate repo `chimerabench` vs. `chimeralang-mcp/benchmarks/`. Standalone is more discoverable; subdir is easier to maintain.

### Success criteria
- [ ] ≥50 tasks across ≥5 families
- [ ] Reference Claude run achieves ≥80% pass rate (sanity check the bench is achievable)
- [ ] Harness runnable in <5 minutes on a single machine
- [ ] README with reproduction instructions and Anthropic-blessed hashes

### Risks
1. **Corpus design quality.** Bad tasks → useless benchmark. Mitigation: borrow task templates from HELM and HaluBench rather than inventing from scratch.
2. **Determinism across model versions.** Mitigation: pin model + temperature + system prompt; allow versioned hashes per Claude model.

---

## Phase 4 — Cross-agent protocol demo

**Goal:** Build a reference implementation showing two agents communicating via Glyph, with ChimeraLang-verified handoffs and Merkle proofs at the boundary. Demo: 60% fewer tokens between agents *and* a cryptographic guarantee that the receiver got the sender's intent.

**Why:** Concrete proof of the "TLS for agent reasoning" positioning. One demo + writeup is more memorable than 51 disconnected tools.

### Sub-tasks
- [ ] **Reference scenario (P4.S1):** Pick a 2-agent task (e.g., orchestrator + worker resolving a coding bug). Define the protocol contract in ChimeraLang.
- [ ] **Sender agent (P4.S2):** Encodes its message via Glyph + emits the program AST + signs with `chimera_prove`.
- [ ] **Receiver agent (P4.S3):** Decodes Glyph, verifies the program AST against expected protocol shape, checks Merkle hash, produces a response under the same protocol.
- [ ] **End-to-end test (P4.S4):** `tests/test_cross_agent_protocol.py` — exercises the full handoff with assertions on token count, decode fidelity, hash agreement.
- [ ] **Demo writeup (P4.S5):** `docs/case-studies/cross-agent-protocol.md` — narrative + numbers + diagram. Aim for portfolio-quality.

### Success criteria
- [ ] Side-by-side: same 2-agent task with and without protocol → ≥40% token reduction across the handoff
- [ ] Receiver rejects forged messages (negative test)
- [ ] One-page diagram explaining the protocol fits on a single screen
- [ ] Linked from README as the headline demo

### Risks
1. **The "protocol" is too abstract to be compelling.** Mitigation: ground in a real task users care about (CI failure triage, code review handoff).
2. **Two agents in tests means CI cost / determinism issues.** Mitigation: mock the second agent with a deterministic stub for tests; reserve real Claude calls for the published demo run.

---

## Cross-cutting principles

1. **Honest measurements only.** Every claim ("60% reduction", "deterministic", "cryptographically verifiable") must be backed by a runnable test or a published artifact. No marketing without a benchmark.
2. **Prune as we add.** Each phase that lands should let us *delete* something — long-tail tools that don't serve the moat. The end-state surface is closer to 15–20 tools than 51.
3. **Branch per phase.** `feature/phase-1-tokenizer-glyph`, `feature/phase-2-gates-as-programs`, etc. PR each into main when complete.
4. **Keep skill files in sync.** `SKILL.md`, `AGENTS.md`, `CLAUDE.md`, `BLUEPRINT.md` — drift kills credibility. The integrity tests already enforce parts of this; extend them as the surface evolves.

---

## Status legend

- `[ ]` Not started
- `[~]` In progress
- `[x]` Complete
- `[!]` Blocked / pivoting (see notes)

Last updated: 2026-05-07 (Phase 1 spike in progress)
