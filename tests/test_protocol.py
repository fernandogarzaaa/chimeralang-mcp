"""tests/test_protocol.py — Phase 4 cross-agent handoff protocol.

Covers the full pack → verify_and_unpack round-trip, plus the four
tampering modes the protocol must catch:
  * payload mutation (sender lies about the result)
  * envelope mutation (sender lies about the inputs)
  * hash mutation alone (sender lies about both)
  * unknown protocol_version (forward compat boundary)

The protocol module is stdlib-only; we test it via the real server
call_tool so the round-trip exercises chimera_run's replay-dispatch path.
"""
from __future__ import annotations

import asyncio
import json
import unittest

from chimeralang_mcp import server as srv
from chimeralang_mcp.protocol import (
    PROTOCOL_VERSION,
    Handoff,
    pack,
    verify_and_unpack,
)


GATE_ARGS = {
    "candidates": [
        {"value": "ship", "confidence": 0.92},
        {"value": "ship", "confidence": 0.88},
        {"value": "wait", "confidence": 0.55},
    ],
    "strategy": "weighted_vote",
    "threshold": 0.65,
}


def _call(tool: str, args: dict) -> tuple[bool, dict]:
    result = asyncio.run(srv.call_tool(tool, args))
    return result.isError, json.loads(result.content[0].text)


def _make_handoff() -> Handoff:
    is_err, payload = _call("chimera_gate", GATE_ARGS)
    if is_err:
        raise AssertionError(payload)
    return pack(
        sender="agent_a", receiver="agent_b",
        tool="chimera_gate", args=GATE_ARGS,
        payload=payload,
        summary_text="we want to ship the patch",
    )


class TestPack(unittest.TestCase):
    def test_pack_returns_handoff_with_consistent_hash(self):
        h = _make_handoff()
        self.assertEqual(h.protocol_version, PROTOCOL_VERSION)
        self.assertEqual(h.sender, "agent_a")
        self.assertEqual(h.receiver, "agent_b")
        self.assertTrue(h.glyph_summary)
        self.assertTrue(h.replay_envelope.startswith("# CHIMERA_REPLAY_v1"))
        self.assertEqual(len(h.program_hash), 64)  # sha256 hex

    def test_json_serialization_round_trips(self):
        h = _make_handoff()
        wire = h.to_json()
        h2 = Handoff.from_json(wire)
        self.assertEqual(h.program_hash, h2.program_hash)
        self.assertEqual(h.replay_envelope, h2.replay_envelope)
        self.assertEqual(h.payload, h2.payload)

    def test_pack_rejects_non_replayable_tool(self):
        with self.assertRaises(ValueError):
            pack(sender="a", receiver="b", tool="chimera_optimize",
                 args={"text": "x"}, payload={}, summary_text="x")

    def test_pack_rejects_payload_with_inconsistent_envelope(self):
        is_err, payload = _call("chimera_gate", GATE_ARGS)
        self.assertFalse(is_err, payload)
        # Forge the embedded envelope to make it disagree with the args.
        payload["provenance"]["program"] = (
            "# CHIMERA_REPLAY_v1\n# tool: chimera_gate\n"
            '{"args":{"candidates":[],"strategy":"majority","threshold":0.5},'
            '"tool":"chimera_gate","version":"1"}\n'
        )
        with self.assertRaises(ValueError):
            pack(sender="a", receiver="b", tool="chimera_gate", args=GATE_ARGS,
                 payload=payload, summary_text="x")


class TestVerifyHappyPath(unittest.TestCase):
    def test_round_trip_accepts(self):
        h = _make_handoff()
        v = asyncio.run(verify_and_unpack(h, call_tool=srv.call_tool))
        self.assertTrue(v.accepted, v.failure_reason)
        self.assertEqual(v.tool, "chimera_gate")
        self.assertEqual(v.payload["value"], h.payload["value"])
        self.assertEqual(v.payload["passed"], h.payload["passed"])

    def test_glyph_summary_decodes_with_no_notes(self):
        h = _make_handoff()
        v = asyncio.run(verify_and_unpack(h, call_tool=srv.call_tool))
        self.assertTrue(v.accepted)
        self.assertEqual(v.decoded_summary_notes, [])
        # The summary text was Glyph-encoded; the decoded form should
        # still mention the human-meaningful word "ship" (or "patch").
        low = v.decoded_summary.lower()
        self.assertTrue(any(w in low for w in ("ship", "patch", "want")),
                        f"decoded summary unrecognisable: {v.decoded_summary!r}")


class TestVerifyTampering(unittest.TestCase):
    """Each tampering mode must result in accepted=False."""

    def _verify(self, h: Handoff):
        return asyncio.run(verify_and_unpack(h, call_tool=srv.call_tool))

    def test_payload_value_flip_rejected(self):
        h = _make_handoff()
        h.payload = {**h.payload, "value": "wait"}  # sender lies about result
        v = self._verify(h)
        self.assertFalse(v.accepted)
        self.assertIn("disagrees", (v.failure_reason or "").lower())

    def test_envelope_mutation_with_unchanged_hash_rejected(self):
        h = _make_handoff()
        # Rewrite a numeric in the envelope but leave the claimed hash.
        h.replay_envelope = h.replay_envelope.replace("0.92", "0.05")
        v = self._verify(h)
        self.assertFalse(v.accepted)
        self.assertIn("hash", (v.failure_reason or "").lower())

    def test_hash_mutation_alone_rejected(self):
        h = _make_handoff()
        # Flip the last hex digit; envelope stays valid.
        h.program_hash = h.program_hash[:-1] + ("a" if h.program_hash[-1] != "a" else "b")
        v = self._verify(h)
        self.assertFalse(v.accepted)
        self.assertIn("hash", (v.failure_reason or "").lower())

    def test_unknown_protocol_version_rejected(self):
        h = _make_handoff()
        h.protocol_version = "999"
        v = self._verify(h)
        self.assertFalse(v.accepted)
        self.assertIn("protocol_version", (v.failure_reason or ""))

    def test_non_dict_payload_rejected(self):
        h = _make_handoff()
        h.payload = ["not", "a", "dict"]  # type: ignore[assignment]
        v = self._verify(h)
        self.assertFalse(v.accepted)
        self.assertIn("not a dict", (v.failure_reason or "").lower())


class TestVerifyDeterminism(unittest.TestCase):
    def test_same_inputs_same_hash_two_machines(self):
        # Simulating "two machines" is just two independent pack() calls
        # — the protocol guarantees that's enough.
        h1 = _make_handoff()
        h2 = _make_handoff()
        self.assertEqual(h1.program_hash, h2.program_hash)
        self.assertEqual(h1.replay_envelope, h2.replay_envelope)


if __name__ == "__main__":
    unittest.main()
