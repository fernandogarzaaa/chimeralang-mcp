"""Cross-agent protocol demo — Phase 4 of BLUEPRINT.md.

Two simulated agents exchange a triage decision. Agent A (orchestrator)
collapses three candidate verdicts via chimera_gate and packages the
result for Agent B (executor). Agent B verifies the handoff
cryptographically, decodes the Glyph summary, and acts.

Run:
    python tools/protocol_demo.py

The demo is fully deterministic: the same inputs always produce the same
program_hash. Re-run, copy the output to another machine, re-run there —
hashes will match.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from chimeralang_mcp import server as srv  # noqa: E402
from chimeralang_mcp.protocol import (  # noqa: E402
    Handoff,
    pack,
    verify_and_unpack,
)


# ── scenario ─────────────────────────────────────────────────────────────


SCENARIO = {
    "tool": "chimera_gate",
    "args": {
        "candidates": [
            {"value": "ship", "confidence": 0.92},
            {"value": "ship", "confidence": 0.88},
            {"value": "wait", "confidence": 0.55},
        ],
        "strategy": "weighted_vote",
        "threshold": 0.65,
    },
    "summary": "We need to know how to ship the patch. The tests pass.",
    "sender": "agent_a_orchestrator",
    "receiver": "agent_b_executor",
}


def banner(text: str) -> None:
    print()
    print("=" * 72)
    print(text)
    print("=" * 72)


# ── demo ─────────────────────────────────────────────────────────────────


async def main() -> int:
    banner("AGENT A — orchestrator")
    print(f"  scenario: {SCENARIO['tool']}({len(SCENARIO['args']['candidates'])} candidates)")
    print(f"  strategy: {SCENARIO['args']['strategy']}, threshold: {SCENARIO['args']['threshold']}")

    # Agent A invokes the gate.
    a_result = await srv.call_tool(SCENARIO["tool"], SCENARIO["args"])
    a_payload = json.loads(a_result.content[0].text)
    print(f"  gate result: value={a_payload['value']!r}  "
          f"passed={a_payload['passed']}  conf={a_payload['consensus_confidence']}")

    # Agent A packages the handoff.
    handoff = pack(
        sender=SCENARIO["sender"],
        receiver=SCENARIO["receiver"],
        tool=SCENARIO["tool"],
        args=SCENARIO["args"],
        payload=a_payload,
        summary_text=SCENARIO["summary"],
        metadata={"scenario": "ship-or-wait-triage"},
    )
    print(f"  glyph summary: {handoff.glyph_summary!r}")
    print(f"  program hash:  {handoff.program_hash}")

    # Wire transit — JSON-serialise as if going over a network.
    on_the_wire = handoff.to_json()
    print(f"  wire size:     {len(on_the_wire)} bytes")

    banner("AGENT B — executor")
    received = Handoff.from_json(on_the_wire)
    print(f"  received from: {received.sender}")
    print(f"  claimed hash:  {received.program_hash}")

    verification = await verify_and_unpack(received, call_tool=srv.call_tool)
    print(f"  verification:  {'ACCEPTED' if verification.accepted else 'REJECTED'}")
    if not verification.accepted:
        print(f"  reason:        {verification.failure_reason}")
        return 1
    print(f"  tool:          {verification.tool}")
    print(f"  decoded glyph: {verification.decoded_summary!r}")
    if verification.decoded_summary_notes:
        print(f"  decode notes:  {verification.decoded_summary_notes}")
    print(f"  payload value: {verification.payload['value']!r}  "
          f"passed={verification.payload['passed']}")

    banner("AGENT B — decision")
    if verification.payload["passed"] and verification.payload["value"] == "ship":
        print("  ✓ verified handoff says ship — executing.")
        return 0
    print(f"  - verified handoff says {verification.payload['value']!r} — holding.")
    return 0


# ── tampering walk-through ───────────────────────────────────────────────


async def show_tampering() -> int:
    """Demonstrate that any wire-level mutation flips the verification."""
    banner("BONUS — TAMPERING WALK-THROUGH")
    a_result = await srv.call_tool(SCENARIO["tool"], SCENARIO["args"])
    a_payload = json.loads(a_result.content[0].text)
    handoff = pack(
        sender=SCENARIO["sender"],
        receiver=SCENARIO["receiver"],
        tool=SCENARIO["tool"],
        args=SCENARIO["args"],
        payload=a_payload,
        summary_text=SCENARIO["summary"],
    )

    # Tamper #1: flip the sender's claimed verdict in the payload.
    tampered_payload = dict(handoff.payload)
    tampered_payload["value"] = "wait"
    tampered = Handoff(
        protocol_version=handoff.protocol_version,
        sender=handoff.sender,
        receiver=handoff.receiver,
        glyph_summary=handoff.glyph_summary,
        replay_envelope=handoff.replay_envelope,
        program_hash=handoff.program_hash,
        payload=tampered_payload,
        metadata=handoff.metadata,
    )
    v = await verify_and_unpack(tampered, call_tool=srv.call_tool)
    print(f"  payload.value flipped:   {'REJECTED' if not v.accepted else 'ACCEPTED'}")
    if not v.accepted:
        print(f"    reason: {v.failure_reason}")

    # Tamper #2: rewrite the envelope to look benign but mismatch the hash.
    bogus_envelope = handoff.replay_envelope.replace("0.92", "0.10")
    tampered2 = Handoff(
        protocol_version=handoff.protocol_version,
        sender=handoff.sender,
        receiver=handoff.receiver,
        glyph_summary=handoff.glyph_summary,
        replay_envelope=bogus_envelope,
        program_hash=handoff.program_hash,  # leave the hash unchanged
        payload=handoff.payload,
        metadata=handoff.metadata,
    )
    v2 = await verify_and_unpack(tampered2, call_tool=srv.call_tool)
    print(f"  envelope rewritten:      {'REJECTED' if not v2.accepted else 'ACCEPTED'}")
    if not v2.accepted:
        print(f"    reason: {v2.failure_reason}")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    asyncio.run(show_tampering())
    sys.exit(rc)
