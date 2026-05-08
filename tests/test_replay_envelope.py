"""tests/test_replay_envelope.py — Phase 2 of BLUEPRINT.md

Every gate-style tool (chimera_gate, chimera_verify, chimera_deliberate,
chimera_quantum_vote) emits a `provenance.program` field containing a
canonical replay envelope. The envelope:
  1. Re-executes via `chimera_run` and produces the same inner result.
  2. Is hashable via `chimera_prove` (SHA-256 over canonical text).
  3. Is recognised + validated by `chimera_typecheck`.
  4. Is deterministic — equal inputs produce byte-identical text and
     identical hashes.
"""
from __future__ import annotations

import asyncio
import json
import unittest

from chimeralang_mcp import server as srv
from chimeralang_mcp.replay import (
    REPLAYABLE_TOOLS,
    build_replay_program,
    hash_program,
    is_replay_program,
    parse_replay_program,
)


def _call(tool: str, args: dict) -> tuple[bool, dict]:
    result = asyncio.run(srv.call_tool(tool, args))
    return result.isError, json.loads(result.content[0].text)


# ── replay module unit tests ─────────────────────────────────────────────


class TestReplayModule(unittest.TestCase):
    def test_build_is_canonical_under_arg_reorder(self):
        a = build_replay_program("chimera_gate", {
            "candidates": [{"value": "X", "confidence": 0.9}],
            "strategy": "majority", "threshold": 0.5,
        })
        b = build_replay_program("chimera_gate", {
            "threshold": 0.5,
            "strategy": "majority",
            "candidates": [{"value": "X", "confidence": 0.9}],
        })
        self.assertEqual(a, b, "canonical JSON must be order-independent")
        self.assertEqual(hash_program(a), hash_program(b))

    def test_unknown_tool_rejected_at_build(self):
        with self.assertRaises(ValueError):
            build_replay_program("chimera_optimize", {})

    def test_is_replay_program_recognises_header(self):
        prog = build_replay_program("chimera_gate", {"candidates": [], "strategy": "majority"})
        self.assertTrue(is_replay_program(prog))
        self.assertFalse(is_replay_program("let x = 1\nemit x"))

    def test_parse_round_trips_args(self):
        original_args = {"candidates": [{"value": 1, "confidence": 0.8}],
                         "strategy": "weighted_vote", "threshold": 0.7}
        prog = build_replay_program("chimera_gate", original_args)
        body = parse_replay_program(prog)
        self.assertEqual(body["tool"], "chimera_gate")
        self.assertEqual(body["args"], original_args)

    def test_parse_rejects_missing_version(self):
        import json
        body = {"tool": "chimera_gate", "args": {}}  # no "version" key
        source = (
            "# CHIMERA_REPLAY_v1\n# tool: chimera_gate\n"
            + json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaises(ValueError, msg="missing version should raise"):
            parse_replay_program(source)

    def test_parse_rejects_wrong_version(self):
        import json
        body = {"tool": "chimera_gate", "args": {}, "version": "99"}
        source = (
            "# CHIMERA_REPLAY_v1\n# tool: chimera_gate\n"
            + json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n"
        )
        with self.assertRaises(ValueError, msg="wrong version should raise"):
            parse_replay_program(source)


# ── chimera_gate ─────────────────────────────────────────────────────────


class TestGateReplay(unittest.TestCase):
    ARGS = {
        "candidates": [
            {"value": "A", "confidence": 0.9},
            {"value": "B", "confidence": 0.7},
            {"value": "A", "confidence": 0.85},
        ],
        "strategy": "weighted_vote",
        "threshold": 0.6,
    }

    def setUp(self):
        is_err, self.original = _call("chimera_gate", self.ARGS)
        self.assertFalse(is_err, self.original)

    def test_provenance_present_and_well_formed(self):
        prov = self.original["provenance"]
        self.assertEqual(prov["tool"], "chimera_gate")
        self.assertTrue(prov["replayable"])
        self.assertTrue(is_replay_program(prov["program"]))
        self.assertEqual(prov["program_hash"], hash_program(prov["program"]))

    def test_replay_via_chimera_run_matches_original(self):
        program = self.original["provenance"]["program"]
        is_err, replayed = _call("chimera_run", {"source": program})
        self.assertFalse(is_err, replayed)
        self.assertTrue(replayed["replay_dispatched"])
        self.assertEqual(replayed["result"]["value"], self.original["value"])
        self.assertEqual(
            replayed["result"]["consensus_confidence"],
            self.original["consensus_confidence"],
        )
        self.assertEqual(replayed["result"]["passed"], self.original["passed"])

    def test_typecheck_accepts_envelope(self):
        program = self.original["provenance"]["program"]
        is_err, tc = _call("chimera_typecheck", {"source": program})
        self.assertFalse(is_err, tc)
        self.assertTrue(tc["ok"])
        self.assertEqual(tc["kind"], "replay_envelope")

    def test_prove_returns_program_hash(self):
        program = self.original["provenance"]["program"]
        is_err, prove = _call("chimera_prove", {"source": program})
        self.assertFalse(is_err, prove)
        self.assertEqual(prove["proof"]["verdict"], "certified")
        self.assertEqual(
            prove["proof"]["program_hash"],
            self.original["provenance"]["program_hash"],
        )

    def test_determinism_same_args_same_hash(self):
        is_err, second = _call("chimera_gate", self.ARGS)
        self.assertFalse(is_err, second)
        self.assertEqual(
            second["provenance"]["program_hash"],
            self.original["provenance"]["program_hash"],
        )

    def test_typecheck_rejects_malformed_envelope(self):
        # Magic header but unparseable body
        broken = "# CHIMERA_REPLAY_v1\n# tool: chimera_gate\n{not json"
        is_err, tc = _call("chimera_typecheck", {"source": broken})
        self.assertFalse(is_err, tc)
        self.assertFalse(tc["ok"])
        self.assertEqual(tc["kind"], "replay_envelope")

    def test_run_rejects_disallowed_tool_in_envelope(self):
        # Magic header but tool not in REPLAYABLE_TOOLS
        smuggled = (
            "# CHIMERA_REPLAY_v1\n# tool: chimera_optimize\n"
            '{"tool":"chimera_optimize","version":"1","args":{"text":"x"}}\n'
        )
        is_err, _ = _call("chimera_run", {"source": smuggled})
        self.assertTrue(is_err, "non-whitelisted tool must be rejected at chimera_run")


# ── chimera_verify ───────────────────────────────────────────────────────


class TestVerifyReplay(unittest.TestCase):
    def test_verify_round_trip(self):
        args = {
            "claims": [{"text": "the user is logged in"}],
            "evidence": ["the user logged in successfully"],
        }
        is_err, original = _call("chimera_verify", args)
        self.assertFalse(is_err, original)
        prov = original["provenance"]
        self.assertEqual(prov["tool"], "chimera_verify")
        is_err, replayed = _call("chimera_run", {"source": prov["program"]})
        self.assertFalse(is_err, replayed)
        self.assertEqual(
            replayed["result"]["verification_score"],
            original["verification_score"],
        )


# ── chimera_deliberate ───────────────────────────────────────────────────


class TestDeliberateReplay(unittest.TestCase):
    def test_deliberate_round_trip(self):
        # Deliberation expects perspectives as dicts; mode="lexical_consensus"
        # is fully deterministic given the same inputs.
        args = {
            "prompt": "Should we ship this feature?",
            "perspectives": [
                {"perspective": "engineer", "content": "the test suite is green"},
                {"perspective": "reviewer", "content": "the test suite is green"},
                {"perspective": "ops",      "content": "the change is small and reverted easily"},
            ],
            "mode": "lexical_consensus",
        }
        is_err, original = _call("chimera_deliberate", args)
        self.assertFalse(is_err, original)
        prov = original["provenance"]
        self.assertEqual(prov["tool"], "chimera_deliberate")
        is_err, replayed = _call("chimera_run", {"source": prov["program"]})
        self.assertFalse(is_err, replayed)
        # Replay produces the same deliberation shape with provenance stripped
        # (provenance is added by the outer call_tool wrapper, not the inner).
        self.assertIn("perspectives", replayed["result"])
        self.assertEqual(
            len(replayed["result"]["perspectives"]),
            len(original["perspectives"]),
        )


# ── chimera_quantum_vote ─────────────────────────────────────────────────


class TestQuantumVoteReplay(unittest.TestCase):
    def test_quantum_vote_round_trip(self):
        args = {
            "responses": [
                {"answer": "yes", "confidence": 0.9},
                {"answer": "yes", "confidence": 0.85},
                {"answer": "no",  "confidence": 0.6},
            ],
            "timeout_s": 5.0,
        }
        is_err, original = _call("chimera_quantum_vote", args)
        self.assertFalse(is_err, original)
        prov = original["provenance"]
        self.assertEqual(prov["tool"], "chimera_quantum_vote")
        is_err, replayed = _call("chimera_run", {"source": prov["program"]})
        self.assertFalse(is_err, replayed)
        # Quantum vote is deterministic given the same inputs; hash must match.
        is_err, second_call = _call("chimera_quantum_vote", args)
        self.assertFalse(is_err, second_call)
        self.assertEqual(
            second_call["provenance"]["program_hash"],
            prov["program_hash"],
        )


# ── coverage parity: every REPLAYABLE_TOOLS member has a test class above ─


class TestReplayCoverage(unittest.TestCase):
    def test_all_replayable_tools_covered_by_tests(self):
        # If you add a tool to REPLAYABLE_TOOLS, add a corresponding test class
        # above and update this assertion.
        expected = frozenset({
            "chimera_gate", "chimera_verify",
            "chimera_deliberate", "chimera_quantum_vote",
        })
        self.assertEqual(REPLAYABLE_TOOLS, expected,
                         "REPLAYABLE_TOOLS changed — update test coverage")


if __name__ == "__main__":
    unittest.main()
