"""Cross-agent handoff protocol — Phase 4 of BLUEPRINT.md.

A handoff packages the result of a gate-tool call into something a peer
agent can verify *without trusting the sender*:

    +------------------------------------------------------+
    |  Handoff                                             |
    |   glyph_summary  —  human-readable context, lossy   |
    |   replay_envelope —  the canonical # CHIMERA_REPLAY |
    |   program_hash   —  SHA-256 over the envelope        |
    |   payload        —  the inner gate-tool result       |
    |   protocol_version — pinned for forward compat       |
    +------------------------------------------------------+

The receiver's verification is mechanical:
  1. Hash the embedded envelope. Must equal program_hash.
  2. Re-run the envelope through chimera_run. Must produce the same payload.
  3. (Optional) Decode glyph_summary back to English for human review.

If any of those fails, the handoff is rejected. There is no other channel —
no shared secret, no signature, no model trust. The cryptographic equality
of the program hash IS the trust primitive.

This module is deliberately stdlib-only. Glyph encode/decode for the
human-readable summary is provided by chimeralang_mcp.ai_language; the
replay primitives come from chimeralang_mcp.replay. No new dependencies.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from chimeralang_mcp.ai_language import decode as glyph_decode
from chimeralang_mcp.ai_language import encode as glyph_encode
from chimeralang_mcp.replay import (
    REPLAYABLE_TOOLS,
    build_replay_program,
    hash_program,
    is_replay_program,
    parse_replay_program,
)

PROTOCOL_VERSION = "1"


@dataclass
class Handoff:
    """A verifiable cross-agent handoff."""
    protocol_version: str
    sender: str
    receiver: str
    glyph_summary: str
    replay_envelope: str
    program_hash: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> "Handoff":
        return cls(**json.loads(data))


# ── sender side ──────────────────────────────────────────────────────────


def pack(
    *,
    sender: str,
    receiver: str,
    tool: str,
    args: dict[str, Any],
    payload: dict[str, Any],
    summary_text: str,
    metadata: dict[str, Any] | None = None,
) -> Handoff:
    """Build a Handoff from a gate-tool result.

    Caller invariant: ``payload`` is the response of ``call_tool(tool, args)``
    and must contain ``provenance.program`` matching ``build_replay_program(tool, args)``.
    Pack re-derives the envelope locally (so the sender can't lie about which
    args produced the payload).
    """
    if tool not in REPLAYABLE_TOOLS:
        raise ValueError(f"tool {tool!r} is not replayable; cannot package handoff")
    envelope = build_replay_program(tool, args)
    program_hash = hash_program(envelope)

    # Caller-supplied payload must agree with our locally-built envelope.
    # If the sender's envelope was tampered with, this catches it before the
    # handoff goes on the wire.
    sender_envelope = (payload.get("provenance") or {}).get("program")
    if sender_envelope is not None and sender_envelope != envelope:
        raise ValueError(
            "payload provenance.program disagrees with locally-built envelope"
        )

    return Handoff(
        protocol_version=PROTOCOL_VERSION,
        sender=sender,
        receiver=receiver,
        glyph_summary=glyph_encode(summary_text),
        replay_envelope=envelope,
        program_hash=program_hash,
        payload=payload,
        metadata=metadata or {},
    )


# ── receiver side ────────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    accepted: bool
    failure_reason: str | None
    tool: str | None
    program_hash: str
    decoded_summary: str
    decoded_summary_notes: list[str]
    payload: dict[str, Any] | None


async def verify_and_unpack(handoff: Handoff, *, call_tool) -> VerificationResult:
    """Verify a Handoff and return its decoded contents.

    The ``call_tool`` parameter is injected so the protocol module never
    imports the server (which would create a cycle). Pass
    ``chimeralang_mcp.server.call_tool``.

    Verification gates, in order:
      1. ``protocol_version`` recognised.
      2. ``replay_envelope`` is a well-formed ``# CHIMERA_REPLAY_v1`` doc.
      3. ``hash_program(envelope) == program_hash``.
      4. Tool is in REPLAYABLE_TOOLS.
      5. Re-running the envelope produces the same payload as ``handoff.payload``.

    The Glyph summary is always decoded — it's diagnostic context, not part
    of the trust check. If decode produces ``notes``, they surface in the
    result rather than failing verification.
    """
    decoded_text, decoded_notes = glyph_decode(handoff.glyph_summary)

    if handoff.protocol_version != PROTOCOL_VERSION:
        return VerificationResult(
            accepted=False,
            failure_reason=f"unsupported protocol_version {handoff.protocol_version!r}",
            tool=None, program_hash=handoff.program_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    if not is_replay_program(handoff.replay_envelope):
        return VerificationResult(
            accepted=False,
            failure_reason="replay_envelope is not a recognised CHIMERA_REPLAY_v1 doc",
            tool=None, program_hash=handoff.program_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    actual_hash = hash_program(handoff.replay_envelope)
    if actual_hash != handoff.program_hash:
        return VerificationResult(
            accepted=False,
            failure_reason=(
                f"program_hash mismatch: claimed "
                f"{handoff.program_hash[:12]}…, computed {actual_hash[:12]}…"
            ),
            tool=None, program_hash=actual_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    try:
        body = parse_replay_program(handoff.replay_envelope)
    except ValueError as e:
        return VerificationResult(
            accepted=False,
            failure_reason=f"envelope rejected: {e}",
            tool=None, program_hash=actual_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    # Re-execute the envelope. The server's call_tool routes the magic
    # header into the replay-dispatch path of chimera_run.
    rerun = await call_tool("chimera_run", {"source": handoff.replay_envelope})
    rerun_payload = json.loads(rerun.content[0].text)
    if rerun.isError:
        return VerificationResult(
            accepted=False,
            failure_reason=f"replay re-execution errored: {rerun_payload}",
            tool=body["tool"], program_hash=actual_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    rerun_inner = rerun_payload.get("result") or {}
    # Compare the fields that are stable across runs. Provenance fields
    # (program / program_hash) regenerate identically by construction.
    sender_payload = handoff.payload
    if not _payloads_equivalent(sender_payload, rerun_inner):
        return VerificationResult(
            accepted=False,
            failure_reason=(
                "sender's payload disagrees with re-executed result — "
                "the envelope and payload are inconsistent"
            ),
            tool=body["tool"], program_hash=actual_hash,
            decoded_summary=decoded_text,
            decoded_summary_notes=decoded_notes,
            payload=None,
        )

    return VerificationResult(
        accepted=True,
        failure_reason=None,
        tool=body["tool"],
        program_hash=actual_hash,
        decoded_summary=decoded_text,
        decoded_summary_notes=decoded_notes,
        payload=rerun_inner,
    )


# ── helpers ──────────────────────────────────────────────────────────────


# Fields that vary on every call (timestamps, namespaces, envelope-internal
# diagnostics) and so must be skipped when comparing sender/receiver payloads.
_VOLATILE_KEYS = frozenset({
    "_chimera_session_budget",
    "envelope",            # carries timestamps + per-call namespace
    "namespace",
    "storage_path",
    "provenance",          # regenerates identically; compared via hash above
})


def _payloads_equivalent(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True iff a and b agree on every key not in _VOLATILE_KEYS."""
    keys = (set(a.keys()) | set(b.keys())) - _VOLATILE_KEYS
    for k in keys:
        if a.get(k) != b.get(k):
            return False
    return True
