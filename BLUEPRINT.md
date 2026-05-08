# BLUEPRINT.md — chimeralang-mcp toward AI bottleneck-solving

> **Vision:** Make chimeralang-mcp the only LLM toolkit where every reasoning step is **typed, compressed, and cryptographically verifiable** — solving three of AI's biggest bottlenecks at once: token cost, hallucination, and reproducibility.

This document tracks the four-phase plan to turn the existing ChimeraLang VM + Glyph + Merkle proofs into a defensible competitive moat. Update status checkboxes as work lands.

---

## Why this exists

Independent audit (2026-05-07) identified the project's **real moat**: it's the only stack combining (1) a typed probabilistic-computation language with gates and proofs, (2) a deterministic AI-to-AI compression DSL, and (3) Merkle-chain provenance over LLM runs. Each pillar today has a stronger external incumbent (LLMLingua-2, Patronus Lynx, DSPy), but **no incumbent owns the intersection**. The four phases below close the gaps that prevent the intersection from becoming the moat.

---

## Phase 1 — Tokenizer-aware Glyph **[COMPLETE — falsification + reframe]**

**Original goal:** Push Glyph compression from the self-reported 46.2% to ≥60% measured on a public corpus.

**Outcome:** The original 46.2% number was a **character** count, not a **token** count. The formal 100-sentence benchmark against `tiktoken o200k_base` (Claude-equivalent BPE) returned **−16.0% on tokens, +19.2% on chars, 0.806 decode fidelity**. The hand-crafted glyph approach cannot beat learned BPE — modern tokenizers already collapse common English to single tokens.

**Strategic outcome:** Glyph reframed as a **deterministic AI-to-AI wire format with semantic round-trip** (sigil-marked entities, explicit modality tokens, lossy-but-deterministic decode). Token-cost claims dropped from all surfaces. The empirical lexicon, benchmark harness, and corpus all shipped — they make every future claim measurable.

### Sub-tasks
- [x] **Spike (P1.S1):** `tools/glyph_spike.py` ran against `o200k_base` — 0 of 249 stems beat English, 74 strictly worse, corpus −4%.
- [x] **Go/No-go decision (P1.S2):** Headroom 9.3% (below 15% gate) → strategy pivoted from "push to 60%" to "publish honest numbers + reframe."
- [x] **Lexicon optimization (P1.S3):** `tools/optimize_lexicon.py` empirically rebuilt 74 entries; `LEGACY_GLYPH_REVERSE` preserves decoder backwards-compat for any 0.7.2 CG.
- [x] **Public benchmark corpus (P1.S4):** `tests/fixtures/glyph_bench.txt` — 100 sentences, 5 domains.
- [x] **Measurement harness (P1.S5):** `tools/glyph_benchmark.py` — reproducible, with optional `--validate-anthropic` flag for the real API.
- [x] **Regression tests (P1.S6):** `tests/test_glyph_benchmark.py` locks −16.0% / +19.2% / 0.806 with ±1pp / ±1pp / ≥0.75 bounds.
- [x] **Public writeup (P1.S7):** `docs/case-studies/chimera-glyph-feature.md` extended with a 0.7.3 addendum that publishes the falsification, the empirical optimization, the 100-sentence numbers, and the reframe. Token-cost claims dropped from `SKILL.md`, `AGENTS.md`, and `chimera_glyph_directive`'s response schema.

### Final measurements (v0.7.3)

| Metric | Value | Bound (regression test) |
|---|---|---|
| Token reduction (BPE) | **−16.0%** | [−17.0%, −15.0%] |
| Character reduction | **+19.2%** | [+18.0%, +21.0%] |
| Decode fidelity (Jaccard) | **0.806** | ≥ 0.75 floor |
| Worst-domain fidelity | 0.70 (dialogue) | ≥ 0.50 floor per domain |

### What the negative finding bought us
1. **Honest measurement infrastructure** — every future Glyph change has a published number to beat.
2. **Empirical lexicon** — strictly better than 0.7.2 across all measured domains, even though the absolute number is still negative on tokens.
3. **Strategic clarity** — Phase 4 (cross-agent protocol) is now the natural home for Glyph; the reframe ties Phase 1's investment directly into Phase 4 instead of competing with token-compression incumbents (LLMLingua-2, Anthropic prompt cache).

---

## Phase 2 — Gates compile to ChimeraLang programs **[COMPLETE — replay envelope]**

**Goal:** Every `chimera_gate` / `chimera_verify` / `chimera_deliberate` / `chimera_quantum_vote` invocation emits a ChimeraLang-compatible program that is re-executable, provable, and type-checkable.

**Outcome:** Shipped as a **canonical replay envelope** (`chimeralang_mcp/replay.py`). Each gate tool now returns a `provenance.program` field containing a self-contained text artifact starting with `# CHIMERA_REPLAY_v1` followed by canonical JSON of `{tool, version, args}`. The envelope:
- Round-trips through `chimera_run` (re-dispatches to the inner tool, returns identical results).
- Hashes deterministically via SHA-256 over the canonical text — `chimera_prove` returns `program_hash` directly.
- Validates via `chimera_typecheck` (recognized as `kind=replay_envelope`).
- Is whitelisted (`REPLAYABLE_TOOLS`) so only known-deterministic tools can be replayed.

**Opcode budget:** **0 new ChimeraLang opcodes** (well below the ≤3 ceiling). The envelope sits alongside existing ChimeraLang source — `chimera_run` checks for the magic header before invoking the parser/VM.

### Sub-tasks
- [x] **AST design spike (P2.S1):** Settled on the replay-envelope approach instead of adding language constructs. Rationale: gate-tool logic is already deterministic Python; the ChimeraLang parse/VM round-trip would have been ceremony around an opaque computation. The replay envelope captures the exact call site cryptographically without polluting the language surface.
- [x] **Reference impl (P2.S2):** `chimera_gate` emits `provenance.program` + `program_hash` + `replayable: true` + `tool` fields.
- [x] **Re-execution test (P2.S3):** `tests/test_replay_envelope.py::TestGateReplay` covers full round-trip + determinism + malformed-envelope rejection + whitelist enforcement.
- [x] **Apply to remaining tools (P2.S4):** `chimera_verify`, `chimera_deliberate`, `chimera_quantum_vote` all emit `provenance.program`. Each has a dedicated round-trip test.
- [x] **Merkle hash for free (P2.S5):** `chimera_prove` accepts replay envelopes natively — returns `verdict: "certified"` with the program hash as `root_hash`. No second call required.

### Final shape (v0.7.4)

| Tool | `provenance.program` emitted | Round-trip test | Deterministic |
|---|---|---|---|
| `chimera_gate` | ✅ | ✅ | ✅ |
| `chimera_verify` | ✅ | ✅ | ✅ |
| `chimera_deliberate` | ✅ | ✅ | ✅ |
| `chimera_quantum_vote` | ✅ | ✅ | ✅ |

### Cross-cutting infrastructure
- `chimeralang_mcp/replay.py` (~120 lines, pure stdlib, no new deps) — the envelope module.
- `chimera_run` — extended to dispatch replay envelopes before falling through to the parser.
- `chimera_typecheck` — extended to validate replay envelope structure.
- `chimera_prove` — extended to compute program hash + re-dispatch as a one-step Merkle proof.
- 15 new tests in `tests/test_replay_envelope.py` (envelope module + 4 round-trip suites + coverage parity test).

### What this unlocks
- **Phase 3 (ChimeraBench)** can use `provenance.program_hash` as the canonical task ID. Anyone running the bench gets the same hash for the same input → trivial reproducibility check.
- **Phase 4 (cross-agent protocol)** has its wire-format primitive: agent A's reasoning step is a hash, agent B verifies the hash matches its own re-execution, no trust required.

### Risks (resolved)
1. ~~Gate operations don't have clean ChimeraLang representations~~ → Side-stepped by the envelope approach. Future work could add a real `replay` keyword for syntactic sugar; the envelope text would auto-translate.
2. ~~Determinism breaks under floating-point drift~~ → Verified via `test_determinism_same_args_same_hash`; same args produce byte-identical envelopes.

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
