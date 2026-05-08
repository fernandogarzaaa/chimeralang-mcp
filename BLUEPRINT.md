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

## Phase 3 — ChimeraBench **[COMPLETE — v1 corpus shipped]**

**Goal:** A public, reproducible benchmark of verifiable agent tasks. Each task: pipeline of tool invocations + canonical SHA-256 hashes. Same inputs → same hashes anywhere.

**Outcome:** `tools/chimerabench/` ships a 15-task corpus across 5 task families, a runner (`run.py`), a README, and a pytest regression (`tests/test_chimerabench.py`). Every step in every task carries a recorded canonical `program_hash`; the regression test fails loudly if any output drifts.

### Sub-tasks
- [x] **Task taxonomy (P3.S1):** Five families covering the four replayable tools individually + multi-step chains. Decision: 5 families × 3 tasks = 15 in v1, sized for fast feedback (corpus runs in <0.5s) with floor of "1 task per family" enforced by the regression test.
- [x] **Corpus authoring (P3.S2):** 15 hand-authored task JSON files under `tools/chimerabench/tasks/<family>/`. Each task: `pipeline`, `expected.program_hashes`, `expected.assertions`, `metadata`.
- [x] **Harness (P3.S3):** `tools/chimerabench/run.py` — argparse CLI, `--update` regenerates hashes, `--filter` runs subsets, `--verbose` prints per-step hashes. Runs full corpus in <0.5 s.
- [x] **Reference implementation (P3.S4):** Initial canonical run blessed all 15 tasks; second run reproduces every hash. Locked into `tests/test_chimerabench.py`.
- [x] **Hosting decision (P3.S5):** Subdirectory (`tools/chimerabench/`) with its own `README.md`. Standalone repo deferred until v2 corpus (>50 tasks) gives external value.

### Success criteria
- [x] ≥15 tasks across ≥5 families ✓ (exactly 15, exactly 5)
- [x] Reference run achieves 100% pass rate ✓ (sanity-check the bench is achievable)
- [x] Harness runnable in <5 minutes on a single machine ✓ (<0.5 s observed)
- [x] README with reproduction instructions and canonical hashes ✓ (`tools/chimerabench/README.md`)
- [x] Regression test fails on any output drift ✓ (`test_every_step_has_canonical_hash`, `test_every_task_passes`)

### Final shape (v1 corpus)

| Family | Tasks | Verifies |
|---|---|---|
| `gate-only` | 3 | Each strategy (majority, weighted_vote, highest_confidence) round-trips. |
| `verify-pipeline` | 3 | Single supported, no-evidence, multi-claim mixed support all stable. |
| `deliberate-converge` | 3 | Strong-agreement, three-way split, empty-perspectives edge case. |
| `vote-converge` | 3 | Strong consensus, balanced tie, single-response degenerate. |
| `mixed-pipeline` | 3 | Verify→gate, gate→gate (independent), deliberate→vote chains. |
| **Total** | **15** | All hash-locked. |

### What this unlocks
- **Phase 4 (cross-agent protocol):** the `program_hash` primitive can now be exchanged between agents as a trust-free reproducibility check. Agent B verifies agent A's claim by re-running the same pipeline and matching the hash — no need to trust anything else.
- **Future v2 corpus:** the schema and harness are stable; growing past 50 tasks is purely a content effort.

### Risks (resolved)
1. ~~Bad tasks → useless benchmark~~ → Each task asserts both shape (program_hashes) and semantics (assertions). Hash-only would catch implementation drift but not silent semantic regressions; assertions catch the latter.
2. ~~Determinism across model versions~~ → Replayable tools are deterministic Python; no model calls in the v1 corpus. When LLM-backed tasks are added in v2, hashes will be pinned per model version.

---

## Phase 4 — Cross-agent protocol **[COMPLETE]**

**Goal:** A wire-format protocol for verifiable cross-agent handoffs — receiver accepts only after re-deriving the sender's claimed result locally and matching hashes byte-for-byte.

**Outcome:** `chimeralang_mcp/protocol.py` (117 lines, stdlib-only) defines `Handoff`, `pack`, `verify_and_unpack`. The receiver's gate is mechanical: validate version → validate envelope shape → re-hash and compare → re-execute the envelope → cross-check payload. Every adversarial mutation we attempted (payload flip, envelope rewrite, hash spoofing, version forgery) flips verification.

**Reframed away from "60% token reduction":** Phase 1 falsified Glyph's token-cost claim. The Phase 4 deliverable is *verifiable handoffs with structured semantics*, not bandwidth savings. The success criteria below were rewritten accordingly during execution.

### Sub-tasks
- [x] **Reference scenario (P4.S1):** Two-agent triage — orchestrator runs `chimera_gate` over candidate verdicts, packages the result for an executor. Implemented as `tools/protocol_demo.py`.
- [x] **Sender agent (P4.S2):** `pack(...)` builds a Handoff. Locally re-derives the envelope from the sender's claimed args (so a sender that lies about provenance fails its own pack call before the wire ever sees the message).
- [x] **Receiver agent (P4.S3):** `verify_and_unpack(...)` runs five sequential gates (protocol version, envelope shape, hash match, re-execution, payload cross-check). Glyph summary is decoded for diagnostic context but is not in the trust chain.
- [x] **End-to-end test (P4.S4):** `tests/test_protocol.py` ships 11 tests across pack guardrails, happy-path verification, four tampering modes, and determinism.
- [x] **Demo writeup (P4.S5):** `docs/case-studies/cross-agent-protocol.md` — full narrative + demo trace + tampering walk-through + limitations section.

### Success criteria (revised post-Phase-1)

| Original | Revised | Status |
|---|---|---|
| ≥40% token reduction across the handoff | The handoff is JSON; size is dominated by the gate-tool's own response, not by Glyph. **Token reduction was not pursued in Phase 4** — Phase 1 falsified that lever. | Reframed |
| Receiver rejects forged messages | Receiver rejects payload mutation, envelope mutation, hash mutation, version mutation | ✅ |
| One-page diagram explaining the protocol | ASCII diagram in case study + 5-gate verification narrative | ✅ |
| Linked from README as headline demo | Linked from BLUEPRINT.md and case study; README link follows in the introspective-review pass | Pending review |

### What this proves

- **No PKI required.** Cryptographic equality of `program_hash` is the trust primitive. The receiver doesn't need a shared secret with the sender; it only needs the same `chimeralang_mcp` version.
- **Tampering is detected at every layer.** Payload-vs-envelope cross-check catches a sender who keeps the envelope honest but lies about the result. Envelope-vs-hash check catches the inverse. Hash-mutation alone is caught by re-hashing.
- **The `Handoff` JSON-serialises trivially** so it works over any transport (queue, MCP tool round-trip, plain HTTP).

### What this unlocks
1. **Audit trails for multi-step pipelines.** Phase 3 already locks per-task hashes in ChimeraBench; Phase 4 generalises that to any inter-agent exchange.
2. **Cross-organisation agent handoffs without SSO.** Two MCP servers on different machines can swap signed work without a service mesh.
3. **The introspective review (next step)** can now compare the original BLUEPRINT.md ambitions against what actually shipped.

### Risks (resolved)
1. ~~"Protocol" too abstract~~ → Grounded in a runnable demo with concrete tampering output.
2. ~~Two-agent tests cause determinism issues~~ → Phase 2's replay envelope guarantees byte-identical hashes; the protocol inherits that determinism without needing real model calls in tests.
3. **New limitation surfaced:** No replay-attack defence. The protocol verifies *what was computed*; replay-attack mitigation (nonces, sequence numbers) is a higher layer's job. Documented in the case study.

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
