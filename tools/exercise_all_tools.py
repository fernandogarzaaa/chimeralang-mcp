"""Exercise every chimera_* MCP tool with a minimal valid payload.

Drives chimeralang_mcp.server.call_tool directly (same path the MCP
runtime takes) and reports pass/fail for each of the 51 tools.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from chimeralang_mcp import server as srv


PAYLOADS: dict[str, dict[str, Any]] = {
    "chimera_run": {"source": "emit 1"},
    "chimera_confident": {"value": 42, "confidence": 0.97, "label": "smoke"},
    "chimera_explore": {"value": "hypothesis", "confidence": 0.4},
    "chimera_gate": {
        "candidates": [
            {"value": "a", "confidence": 0.6},
            {"value": "a", "confidence": 0.7},
            {"value": "b", "confidence": 0.3},
        ],
        "strategy": "majority",
    },
    "chimera_detect": {"value": "alpha", "strategy": "self_consistency"},
    "chimera_constrain": {
        "tool_name": "bash",
        "output": "ok",
        "schema": {"type": "string"},
    },
    "chimera_typecheck": {"source": "let x = 1"},
    "chimera_prove": {"source": "assert true"},
    "chimera_audit": {},
    "chimera_claims": {"text": "The sky is blue. Water boils at 100C."},
    "chimera_verify": {"evidence": [{"claim": "x", "source": "y", "confidence": 0.8}]},
    "chimera_provenance_merge": {
        "envelopes": [
            {"value": "v1", "confidence": 0.8, "sources": ["a"]},
            {"value": "v1", "confidence": 0.7, "sources": ["b"]},
        ],
    },
    "chimera_policy": {},
    "chimera_trace": {},
    "chimera_materials": {},
    "chimera_fracture": {"text": "hello world " * 20},
    "chimera_optimize": {"text": "the quick brown fox " * 30, "level": "medium"},
    "chimera_compress": {"text": "alpha beta gamma " * 25},
    "chimera_budget": {},
    "chimera_score": {"messages": [{"role": "user", "content": "hi"}]},
    "chimera_cost_estimate": {"text": "estimate me"},
    "chimera_cost_track": {"tokens_before": 1000, "tokens_after": 250},
    "chimera_dashboard": {},
    "chimera_csm": {"prompt": "summarise this"},
    "chimera_budget_lock": {"max_output_tokens": 512},
    "chimera_causal": {"events": ["a", "b", "c"]},
    "chimera_deliberate": {
        "prompt": "Should we ship?",
        "perspectives": ["pro", "con"],
    },
    "chimera_metacognize": {"thought": "I think I'm right"},
    "chimera_meta_learn": {},
    "chimera_quantum_vote": {"responses": ["yes", "yes", "no"]},
    "chimera_plan_goals": {"goal": "ship the feature"},
    "chimera_world_model": {},
    "chimera_safety_check": {"content": "harmless text"},
    "chimera_ethical_eval": {"action": "donate to charity"},
    "chimera_embodied": {},
    "chimera_social": {},
    "chimera_transfer_learn": {},
    "chimera_evolve": {},
    "chimera_self_model": {},
    "chimera_knowledge": {},
    "chimera_memory": {},
    "chimera_mode": {},
    "chimera_batch": {
        "calls": [
            {"tool": "chimera_compress", "args": {"text": "foo bar " * 30}},
            {"tool": "chimera_compress", "args": {"text": "baz qux " * 30}},
        ],
    },
    "chimera_summarize": {"text": "sentence one. sentence two. sentence three." * 10},
    "chimera_cache_mark": {"blocks": [{"text": "system block", "stable": True}]},
    "chimera_log_compress": {"text": "INFO: starting\nERROR: oops\nINFO: done\n" * 30},
    "chimera_overhead_audit": {},
    "chimera_dedup_lookup": {"action": "stats"},
    "chimera_session_report": {},
    "chimera_glyph_directive": {},
    "chimera_glyph_translate": {"glyph_text": "△"},
}


async def main() -> int:
    tools = await srv.list_tools()
    names = [t.name for t in tools]
    missing = [n for n in names if n not in PAYLOADS]
    if missing:
        print(f"NO PAYLOAD FOR: {missing}", file=sys.stderr)

    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    for name in names:
        args = PAYLOADS.get(name, {})
        try:
            result = await srv.call_tool(name, args)
            content = result.content if hasattr(result, "content") else result
            if not content:
                raise RuntimeError("empty content")
            text = getattr(content[0], "text", None)
            if text is None:
                raise RuntimeError("no .text on first content item")
            # Most tools return JSON; some may return raw text. Either is fine.
            try:
                json.loads(text)
            except json.JSONDecodeError:
                pass
            passed.append(name)
        except Exception as exc:  # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))

    print(f"\n=== RESULTS: {len(passed)}/{len(names)} passed ===")
    for name in passed:
        print(f"  PASS  {name}")
    if failed:
        print(f"\n--- {len(failed)} FAILED ---")
        for name, err in failed:
            print(f"  FAIL  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
