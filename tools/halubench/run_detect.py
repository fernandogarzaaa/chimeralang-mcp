"""DetectBench harness (Phase 5.3) — measures chimera_detect's two real
signals against a task-appropriate labeled corpus:

  - certainty: does the semantic strategy flag overconfident phrasing?
  - injection: does it raise an attack flag on prompt-injection text?

chimera_detect screens phrasing and attack patterns, not evidence entailment —
so it is benchmarked on its own corpus (detect_corpus.json), not the verify
corpus. Every item is scored by calling the live chimera_detect tool.

Usage:
    python -m tools.halubench.run_detect            # score + report
    python -m tools.halubench.run_detect --verbose  # per-item predictions
    python -m tools.halubench.run_detect --update    # write detect_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from chimeralang_mcp import server as srv  # noqa: E402

CORPUS = Path(__file__).parent / "detect_corpus.json"
RESULTS = Path(__file__).parent / "detect_results.json"


async def _detect_flags(text: str) -> tuple[bool, bool]:
    """Return (certainty_flagged, injection_flagged) for one text."""
    result = await srv.call_tool("chimera_detect", {"value": text, "strategy": "semantic"})
    payload = json.loads(result.content[0].text)
    certainty = any(f.get("kind", "").startswith("SEMANTIC") for f in payload.get("flags", []))
    injection = bool(payload.get("attack_flags"))
    return certainty, injection


def _binary_metrics(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    """rows = list of (gold_flag, pred_flag)."""
    tp = sum(1 for g, p in rows if g and p)
    fp = sum(1 for g, p in rows if not g and p)
    fn = sum(1 for g, p in rows if g and not p)
    tn = sum(1 for g, p in rows if not g and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(rows), 4) if rows else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(rows),
    }


async def run(verbose: bool) -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for task_name, task in corpus["tasks"].items():
        rows: list[tuple[bool, bool]] = []
        for item in task["items"]:
            certainty, injection = await _detect_flags(item["text"])
            pred = injection if task_name == "injection" else certainty
            gold = bool(item["flag"])
            rows.append((gold, pred))
            if verbose:
                mark = "✓" if gold == pred else "✗"
                print(f"  [{task_name}] {mark} {item['id']:<24} gold={gold!s:<5} pred={pred}")
        out[task_name] = _binary_metrics(rows)
    return out


def _print_report(results: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("DetectBench — chimera_detect signal calibration")
    print("=" * 60)
    for task_name, m in results.items():
        print(f"\n  {task_name}  (n={m['n']})")
        print(f"    precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
              f"f1 {m['f1']:.3f}  acc {m['accuracy']:.3f}")
        print(f"    tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(prog="detectbench")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--update", action="store_true", help="write detect_results.json")
    args = parser.parse_args()
    results = asyncio.run(run(args.verbose))
    _print_report(results)
    if args.update:
        RESULTS.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote results -> {RESULTS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
