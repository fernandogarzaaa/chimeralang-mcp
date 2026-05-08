"""tools/cost_savings_demo.py — measure the real token-savings tools.

Runs chimera_optimize, chimera_compress, chimera_fracture,
chimera_log_compress, and chimera_cache_mark against representative
payloads and prints a compact table of measured savings. The numbers
this script produces are what `docs/cost-savings-guide.md` cites.

Re-run anytime to verify the published numbers still hold:
    python tools/cost_savings_demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from chimeralang_mcp import server as srv  # noqa: E402


# ── representative payloads ──────────────────────────────────────────────


# Long doc: the project's own CLAUDE.md (real, reproducible).
LONG_DOC = (REPO / "CLAUDE.md").read_text(encoding="utf-8")

# Conversation history (synthetic, deterministic).
CONVERSATION = [
    {"role": "user",      "content": "Can you walk me through the new feature we discussed last week?"},
    {"role": "assistant", "content": "Sure. The feature lets agents pass verifiable handoffs to each other without a shared key. The receiver re-runs the sender's claimed computation locally and compares hashes. If anything was tampered with, verification fails."},
    {"role": "user",      "content": "Got it. And how does that compare to just signing the message with a private key?"},
    {"role": "assistant", "content": "Signing requires key distribution, which is exactly the operational tax we wanted to avoid. The replay envelope hashes the canonical text of the computation itself. Same inputs always produce the same hash on any machine."},
    {"role": "user",      "content": "What if the underlying tool is non-deterministic?"},
    {"role": "assistant", "content": "Then it's not on the REPLAYABLE_TOOLS whitelist. We deliberately restricted Phase 4 to gate-style tools that are pure functions of their arguments — chimera_gate, chimera_verify, chimera_deliberate, chimera_quantum_vote. LLM-backed tools would need additional pinning."},
    {"role": "user",      "content": "Does the receiver have to know all the tool schemas in advance?"},
    {"role": "assistant", "content": "Only the schema for the tool name embedded in the envelope. The replay envelope includes the tool name and JSON-shaped args. The receiver dispatches via call_tool, which hits the server's normal handler. No new schema work."},
    {"role": "user",      "content": "And the Glyph summary is just for human-readable context?"},
    {"role": "assistant", "content": "Right. Glyph isn't in the trust chain — it's diagnostic. The lossy round-trip is fine because nothing important is decided based on the summary. The trust gate is hash equality plus payload re-execution."},
    {"role": "user",      "content": "What about replay attacks?"},
    {"role": "assistant", "content": "Out of scope at the protocol layer. The protocol verifies what was computed, not when. Higher-level systems need to add nonces or sequence numbers if replay-attack defence matters for their use case."},
    {"role": "user",      "content": "OK. So if I want to build a multi-agent triage workflow on top of this, where do I start?"},
    {"role": "assistant", "content": "Start with the demo at tools/protocol_demo.py — orchestrator, executor, full handoff with tampering walk-through. The Handoff dataclass is the primitive; pack and verify_and_unpack are the only two functions you need from chimeralang_mcp.protocol."},
]

# Build/test log — mostly INFO noise, a few real errors.
LOG = "\n".join(
    [f"INFO {i}: routine progress event"     for i in range(80)]
    + ["WARNING: deprecated configuration key 'old_path' will be removed in v2.0"]
    + [f"INFO {i}: more progress"            for i in range(80, 160)]
    + ["ERROR: Connection refused at 10.0.0.5:5432 after 3 retries"]
    + [f"INFO {i}: yet more progress"        for i in range(160, 240)]
    + ["Traceback (most recent call last):"]
    + ["  File 'pipeline.py', line 142, in run_step"]
    + ["    result = handler(payload)"]
    + ["  File 'handlers.py', line 58, in handler"]
    + ["    raise ValueError('schema mismatch on field user_id')"]
    + ["ValueError: schema mismatch on field user_id"]
    + [f"INFO {i}: noise after the failure"  for i in range(240, 320)]
)

# Stable system block — the kind of context you'd send on every API call.
SYSTEM_BLOCK = (
    "You are a careful coding assistant. " * 60
    + "Always cite file paths with line numbers. " * 30
    + "Never claim that code works without running it. " * 30
)


def pct(before: int, after: int) -> str:
    if before == 0:
        return "—"
    return f"{100 * (before - after) / before:.1f}%"


async def run() -> None:
    rows: list[tuple[str, str, str, int, int, str]] = []

    # ── chimera_optimize on a long doc ────────────────────────────────
    out = json.loads((await srv.call_tool("chimera_optimize", {
        "text": LONG_DOC,
    })).content[0].text)
    rows.append((
        "chimera_optimize",
        "long doc (CLAUDE.md)",
        "structural compression — drops filler, dedupes",
        out["estimated_tokens_before"],
        out["estimated_tokens_after"],
        pct(out["estimated_tokens_before"], out["estimated_tokens_after"]),
    ))

    # ── chimera_compress on a long text blob ──────────────────────────
    joined = "\n\n".join(m["content"] for m in CONVERSATION)
    out = json.loads((await srv.call_tool("chimera_compress", {
        "text":  joined,
        "level": "aggressive",
    })).content[0].text)
    rows.append((
        "chimera_compress",
        "12-turn dialogue text (joined)",
        "contractions + symbols + filler stripping",
        out["estimated_tokens_before"],
        out["estimated_tokens_after"],
        pct(out["estimated_tokens_before"], out["estimated_tokens_after"]),
    ))

    # ── chimera_fracture on doc + history ─────────────────────────────
    out = json.loads((await srv.call_tool("chimera_fracture", {
        "documents": [LONG_DOC],
        "messages":  CONVERSATION,
        "budget":    600,
    })).content[0].text)
    before = out["tokens_input"]
    after  = out["tokens_after_pipeline"]
    rows.append((
        "chimera_fracture",
        "long doc + 12-msg dialogue, 600-token budget",
        "single-call quality-gated mix",
        before, after, pct(before, after),
    ))

    # ── chimera_log_compress on a noisy build log ─────────────────────
    out = json.loads((await srv.call_tool("chimera_log_compress", {
        "text": LOG,
    })).content[0].text)
    rows.append((
        "chimera_log_compress",
        "320-line log with 1 warning, 1 error, 1 traceback",
        "head/tail + error/warning lines kept verbatim",
        out["lines_in"],
        out["lines_out"],
        pct(out["lines_in"], out["lines_out"]),
    ))

    # ── chimera_cache_mark on a stable system block ───────────────────
    out = json.loads((await srv.call_tool("chimera_cache_mark", {
        "blocks": [{"name": "system", "text": SYSTEM_BLOCK, "stable": True}],
        "model":  "claude-sonnet-4-6",
    })).content[0].text)
    cached    = out["cache_eligible_tokens"]
    saved_90  = out["estimated_savings_at_90pct"]
    rows.append((
        "chimera_cache_mark",
        "stable ~5kB system block, repeated across calls",
        "marks cache_control: ephemeral; lossless ~90% off cached tokens on hit",
        cached, cached - saved_90,
        pct(cached, cached - saved_90),
    ))

    # ── render ────────────────────────────────────────────────────────
    print("=" * 88)
    print("TOKEN-SAVINGS TOOLS — MEASURED ON REPRESENTATIVE PAYLOADS")
    print("=" * 88)
    print()
    print(f"  {'Tool':<24} {'Payload':<46} {'Before':>8} {'After':>8} {'Saved':>8}")
    print(f"  {'-'*24} {'-'*46} {'-'*8} {'-'*8} {'-'*8}")
    for tool, payload, _, before, after, ratio in rows:
        print(f"  {tool:<24} {payload[:46]:<46} {before:>8} {after:>8} {ratio:>8}")
    print()
    print("  Notes:")
    for tool, payload, note, *_ in rows:
        print(f"    {tool:<24} → {note}")
    print()

    # JSON dump for the doc to cite.
    out_path = Path(__file__).parent / "cost_savings_results.json"
    out_path.write_text(json.dumps([
        {"tool": r[0], "payload": r[1], "note": r[2],
         "before": r[3], "after": r[4], "saved_pct": r[5]}
        for r in rows
    ], indent=2))
    print(f"  Results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
