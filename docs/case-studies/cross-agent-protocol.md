# Case Study: A Verifiable Cross-Agent Handoff Protocol

**Project:** [`chimeralang-mcp`](https://github.com/fernandogarzaaa/chimeralang-mcp)
**Feature:** Phase 4 — cross-agent handoff protocol with deterministic semantics + cryptographic provenance.
**Release:** v0.7.4 (Phase 2 envelope) + v0.7.5 (Phase 4 protocol).
**Date range:** 2026-05-07 to 2026-05-08.

---

## Executive summary

Most multi-agent designs assume the agents trust each other — model A's output is taken at face value by model B. That assumption fails the moment one of the agents is compromised, drifts off-policy, or simply hallucinates. This phase delivers a wire-format protocol that makes the trust assumption unnecessary: every cross-agent handoff includes a re-runnable provenance envelope, the receiver re-executes it locally, and acceptance is gated on byte-identical hash equality. No shared secret, no signature, no model-trust required — the cryptographic equality of the program hash is the trust primitive.

Concretely: a 117-line `chimeralang_mcp/protocol.py` module, an end-to-end demo, and 11 round-trip + tampering tests, all on top of the Phase 2 replay envelope. The protocol catches every adversarial mutation we attempted (payload flip, envelope rewrite, hash spoofing, version forgery) without any need to fingerprint the sender or transmit a shared key.

---

## Problem statement

A multi-agent system has a handoff problem nobody talks about until production.

Agent A computes a result. Agent A serializes the result and ships it to agent B. Agent B receives the message and acts on it. The standard pattern — JSON over a queue, stdout-piped to stdin, or an MCP tool round-trip — assumes:

1. **The wire is honest** (no MITM rewriting fields).
2. **Agent A is honest** (no hallucinated fields, no fabricated provenance).
3. **Agent A's claimed result actually corresponds to its claimed inputs** (no swap of "I ran chimera_gate on these inputs" with "I ran something else and called it chimera_gate").

In practice, all three assumptions fail at scale: networks are flaky, agents drift off-policy, and "result" payloads are easy to forge after the fact. The standard mitigation is signed messages — a PKI bolted onto the agent runtime. That works, but it shifts the problem: now you need a key-distribution story.

The goal of this phase: do better than "trust agent A" without paying the PKI tax. Specifically, give agent B a way to **re-derive** agent A's claimed result from agent A's claimed inputs, and reject the handoff if the re-derivation doesn't match.

---

## Solution: deterministic provenance + structured semantics

The protocol piggybacks on two pieces of infrastructure shipped earlier in the project:

1. **The replay envelope** (Phase 2, `chimeralang_mcp/replay.py`). Every gate-tool invocation now emits a canonical, deterministic text artifact:

   ```text
   # CHIMERA_REPLAY_v1
   # tool: chimera_gate
   {"args":{...},"tool":"chimera_gate","version":"1"}
   ```

   The text is byte-identical for byte-identical inputs. SHA-256 over the text is the program hash. Re-running the envelope through `chimera_run` re-dispatches to the original tool with the original args.

2. **Glyph encoding** (Phase 1, `chimeralang_mcp/ai_language.py`). After Phase 1's falsification of the token-cost claim, Glyph was reframed as a deterministic semantic wire format — sigil-marked entities, explicit modality tokens, lossy-but-deterministic decode. It carries the *human-readable* part of the handoff in a form that round-trips cleanly.

The protocol module composes these two into a `Handoff` dataclass:

```python
@dataclass
class Handoff:
    protocol_version:  str
    sender:            str
    receiver:          str
    glyph_summary:     str   # human-meaningful context, lossy
    replay_envelope:   str   # the # CHIMERA_REPLAY_v1 doc
    program_hash:      str   # SHA-256 over replay_envelope
    payload:           dict  # the inner gate-tool result
    metadata:          dict
```

`pack(...)` builds a Handoff from a gate-tool result. `verify_and_unpack(...)` is the receiver's gate. Verification runs in this order:

1. Recognise the protocol version.
2. Validate the envelope is a well-formed `# CHIMERA_REPLAY_v1` doc.
3. Re-hash the envelope locally; compare to `program_hash`. Mismatch → reject.
4. Re-execute the envelope through `chimera_run` (which dispatches to the inner tool via the Phase 2 path).
5. Compare the re-executed payload to the sender's claimed payload (skipping volatile fields like timestamps/namespace). Disagreement → reject.

The Glyph summary is decoded for human review but is deliberately *not* in the trust chain. It's diagnostic context, not load-bearing.

---

## Demo trace

`python tools/protocol_demo.py` runs a triage scenario end-to-end. Agent A collapses three candidate verdicts (two `ship` at 0.92 + 0.88, one `wait` at 0.55) under `weighted_vote`. Agent B verifies and acts.

```
========================================================================
AGENT A — orchestrator
========================================================================
  scenario: chimera_gate(3 candidates)
  strategy: weighted_vote, threshold: 0.65
  gate result: value='ship'  passed=True  conf=0.766
  glyph summary: 'w nd to kn how to @ship @patch. tst pass.'
  program hash:  5f537cade2e685d4fcf9d8696e36d1cdeffa85a731c536d9615aad6e903dd5f1
  wire size:     1606 bytes

========================================================================
AGENT B — executor
========================================================================
  received from: agent_a_orchestrator
  claimed hash:  5f537cade2e685d4fcf9d8696e36d1cdeffa85a731c536d9615aad6e903dd5f1
  verification:  ACCEPTED
  tool:          chimera_gate
  decoded glyph: 'We need to know how to ship patch. the test pass.'
  payload value: 'ship'  passed=True

========================================================================
AGENT B — decision
========================================================================
  ✓ verified handoff says ship — executing.

========================================================================
BONUS — TAMPERING WALK-THROUGH
========================================================================
  payload.value flipped:   REJECTED
    reason: sender's payload disagrees with re-executed result — the envelope and payload are inconsistent
  envelope rewritten:      REJECTED
    reason: program_hash mismatch: claimed 5f537cade2e6…, computed 4cccf994c70b…
```

Every adversarial mutation flips verification. Including the payload-vs-envelope cross-check is what catches a sender who keeps the envelope honest but lies about its result — without that step, an attacker could ship `value: "ship"` while the envelope's actual computation said `wait`, and only re-execution catches it.

---

## Test coverage

`tests/test_protocol.py` (11 tests, all green):

| Concern | Tests |
|---|---|
| Pack happy-path | `test_pack_returns_handoff_with_consistent_hash`, `test_json_serialization_round_trips` |
| Pack guardrails | `test_pack_rejects_non_replayable_tool`, `test_pack_rejects_payload_with_inconsistent_envelope` |
| Verify happy-path | `test_round_trip_accepts`, `test_glyph_summary_decodes_with_no_notes` |
| Tampering | `test_payload_value_flip_rejected`, `test_envelope_mutation_with_unchanged_hash_rejected`, `test_hash_mutation_alone_rejected`, `test_unknown_protocol_version_rejected` |
| Determinism | `test_same_inputs_same_hash_two_machines` |

The protocol module imports zero third-party packages. The verifier takes `call_tool` as a parameter rather than importing `chimeralang_mcp.server`, which keeps the dependency graph clean and lets a future SDK consumer plug in any compatible dispatch.

---

## What this enables

The handoff isn't a finished product — it's a primitive. Three concrete uses unlock immediately:

1. **Audit trails for multi-step agent runs.** Each step's `program_hash` is the per-step audit record. A pipeline of N gate calls produces N hashes, chainable into a Merkle tree if a downstream system wants single-hash verification.

2. **Drop-in trust between agents that don't share infrastructure.** Two MCP servers running on different machines, possibly different organisations, can hand off work via the JSON-serialised `Handoff` and verify locally. No SSO, no shared signing key, no service mesh.

3. **A reproducibility primitive for ChimeraBench.** Phase 3 already records canonical hashes per task; the handoff protocol generalises that beyond the test corpus to any agent-to-agent exchange.

The combination — Glyph for semantics, replay envelope for provenance, handoff for verifiable transit — is what the project's pre-Phase-1 framing was reaching for but couldn't articulate. The original "AI-only language" pitch was looking at the wrong axis (token cost). The right axis was always **verifiable handoffs with structured semantics**, and the v0.7.5 protocol is the first artifact that delivers that end-to-end.

---

## Limitations

- **The protocol verifies the computation, not the intent.** A sender that computes the *wrong* `chimera_gate` call is still caught by the receiver only if the receiver knows what the *right* call would be. The protocol is an integrity primitive, not a policy primitive.
- **Replayable tools must be deterministic.** The `REPLAYABLE_TOOLS` whitelist (gate / verify / deliberate / quantum_vote) was deliberately narrow. Adding a non-deterministic tool (e.g., one that calls an external LLM) would require either pinning the model+temperature in the envelope or accepting probabilistic equality.
- **Glyph summary is lossy by design.** "We need to know how to ship the patch" → `w nd to kn how to @ship @patch.` → "We need to know how to ship patch." That round-trip drops the article and leaves "patch" un-articled. For semantic communication this is fine; for legal contract text it is not.
- **No replay-attack defence.** The same Handoff replayed twice will verify both times. Higher-level systems must add nonce or sequence-number tracking; the protocol doesn't pretend to be an authentication primitive.

---

## Repository navigation

- **Protocol library:** [`chimeralang_mcp/protocol.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/chimeralang_mcp/protocol.py)
- **Replay envelope (Phase 2 dependency):** [`chimeralang_mcp/replay.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/chimeralang_mcp/replay.py)
- **Demo:** [`tools/protocol_demo.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/tools/protocol_demo.py)
- **Tests:** [`tests/test_protocol.py`](https://github.com/fernandogarzaaa/chimeralang-mcp/blob/main/tests/test_protocol.py)
- **Phase 1 (Glyph reframe):** [`docs/case-studies/chimera-glyph-feature.md`](./chimera-glyph-feature.md)
- **Blueprint:** [`BLUEPRINT.md`](../../BLUEPRINT.md)
