"""Replay envelope — canonical, deterministic program format that lets a
gate-tool invocation be re-executed via chimera_run, hashed via chimera_prove,
and validated via chimera_typecheck.

This is the Phase 2 (P2.S1/P2.S2 of BLUEPRINT.md) compromise design: rather
than adding new ChimeraLang opcodes for every gate variant, we define a
single ``# CHIMERA_REPLAY_v1`` envelope. A replay program is plain text that
chimera_run recognises and dispatches; chimera_prove SHA-256s the canonical
text; chimera_typecheck validates the JSON body against the inner tool's
known schema.

Determinism guarantees:
  * ``build_replay_program`` always emits the same string for the same args
    (canonical JSON via ``json.dumps(..., sort_keys=True)``).
  * ``hash_program`` is SHA-256 over the exact bytes of the program text.
  * The whitelist (``REPLAYABLE_TOOLS``) is the boundary; only purely
    deterministic gate-style tools are accepted.

Adding a new replayable tool: append it to ``REPLAYABLE_TOOLS`` and add a
``provenance.program`` field to its handler in ``server.py`` using
``build_replay_program(tool, args)``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

MAGIC_HEADER = "# CHIMERA_REPLAY_v1"
PROGRAM_VERSION = "1"

# Whitelist of tools whose invocations can be replayed. These must be:
#   1. Pure functions of their arguments (no namespace / persistence side-effects
#      that change the answer between runs).
#   2. Returning a deterministic envelope.
# The Phase 2 launch covers the four gate-style tools per BLUEPRINT.md.
REPLAYABLE_TOOLS: frozenset[str] = frozenset({
    "chimera_gate",
    "chimera_verify",
    "chimera_deliberate",
    "chimera_quantum_vote",
})


def build_replay_program(tool: str, args: dict[str, Any]) -> str:
    """Emit the canonical replay program text for a tool invocation.

    Format:
      # CHIMERA_REPLAY_v1
      # tool: <tool>
      <canonical JSON body>

    The body is JSON with sorted keys and stable separators, so two calls
    with equal-content arguments always produce byte-identical text.
    """
    if tool not in REPLAYABLE_TOOLS:
        raise ValueError(
            f"tool {tool!r} is not in REPLAYABLE_TOOLS; cannot build replay envelope"
        )
    body = {
        "tool": tool,
        "version": PROGRAM_VERSION,
        "args": args,
    }
    canonical_json = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return (
        f"{MAGIC_HEADER}\n"
        f"# tool: {tool}\n"
        f"{canonical_json}\n"
    )


def is_replay_program(source: str) -> bool:
    """True if ``source`` is a recognised replay envelope (cheap check)."""
    return source.startswith(MAGIC_HEADER + "\n")


def parse_replay_program(source: str) -> dict[str, Any]:
    """Extract the JSON body from a replay envelope. Raises ValueError on
    malformed inputs (missing header, malformed JSON, unknown tool)."""
    if not is_replay_program(source):
        raise ValueError(
            f"replay program must start with {MAGIC_HEADER!r} on its own line"
        )
    lines = source.splitlines()
    # Skip the magic header and any subsequent comment lines that begin with #.
    body_start = 1
    while body_start < len(lines) and lines[body_start].lstrip().startswith("#"):
        body_start += 1
    body_text = "\n".join(lines[body_start:]).strip()
    if not body_text:
        raise ValueError("replay program has empty body")
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"replay program body is not valid JSON: {e}") from e
    if not isinstance(body, dict):
        raise ValueError("replay program body must be a JSON object")
    tool = body.get("tool")
    if tool not in REPLAYABLE_TOOLS:
        raise ValueError(f"replay program targets unknown/disallowed tool: {tool!r}")
    if "args" not in body or not isinstance(body["args"], dict):
        raise ValueError("replay program must have an 'args' object")
    return body


def hash_program(source: str) -> str:
    """SHA-256 hex digest over the exact bytes of the program text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
