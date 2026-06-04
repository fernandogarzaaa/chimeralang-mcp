"""HaluBench harness — measures chimera_verify hallucination-detection quality
against a hand-labeled corpus and reports precision / recall / F1 per verdict
class plus macro and accuracy.

This is the Phase 5.1 baseline: it establishes the number that any future
semantic tier (Phase 5.2, method="nli"/"llm") must beat. Every item is scored
by calling the live chimera_verify tool, so the harness dogfoods the MCP.

Usage:
    python -m tools.halubench.run                 # score corpus, print report
    python -m tools.halubench.run --verbose       # also print per-item predictions
    python -m tools.halubench.run --method lexical # scoring method (default lexical)
    python -m tools.halubench.run --update        # write baseline_results.json
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

CORPUS = Path(__file__).parent / "corpus.json"
BASELINE = Path(__file__).parent / "baseline_results.json"
LABELS = ["supported", "contradicted", "insufficient"]


def _normalize_verdict(verdict: str) -> str:
    """Strip the method prefix (lexically_/nli_/llm_) from a verify verdict."""
    for prefix in ("lexically_", "nli_", "llm_"):
        if verdict.startswith(prefix):
            verdict = verdict[len(prefix):]
            break
    # verify uses "insufficient"; corpus label is the same
    return verdict


async def _predict(item: dict, method: str) -> str:
    args: dict[str, Any] = {"claims": [item["claim"]], "evidence": item["evidence"]}
    if method != "lexical":
        args["method"] = method
    result = await srv.call_tool("chimera_verify", args)
    if result.isError:
        try:
            msg = json.loads(result.content[0].text).get("error", result.content[0].text)
        except Exception:
            msg = "unknown chimera_verify error"
        raise RuntimeError(f"chimera_verify failed for {item.get('id')} (method={method}): {msg}")
    payload = json.loads(result.content[0].text)
    verdict = str(payload.get("verdict", "")).strip()
    if not verdict:
        raise RuntimeError(f"chimera_verify returned no verdict for {item.get('id')} (method={method})")
    return _normalize_verdict(verdict)


def _metrics(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """rows = list of (gold, predicted). Returns per-class + macro + accuracy."""
    per_class: dict[str, dict[str, float]] = {}
    correct = sum(1 for g, p in rows if g == p)
    for label in LABELS:
        tp = sum(1 for g, p in rows if g == label and p == label)
        fp = sum(1 for g, p in rows if g != label and p == label)
        fn = sum(1 for g, p in rows if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    macro_f1 = round(sum(c["f1"] for c in per_class.values()) / len(LABELS), 4)
    return {
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n": len(rows),
    }


def _confusion(rows: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for gold, pred in rows:
        if gold in matrix and pred in matrix[gold]:
            matrix[gold][pred] += 1
    return matrix


async def run(method: str, verbose: bool) -> dict[str, Any]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    items = corpus["items"]
    rows: list[tuple[str, str]] = []
    per_item: list[dict[str, str]] = []
    for item in items:
        pred = await _predict(item, method)
        rows.append((item["label"], pred))
        per_item.append({"id": item["id"], "gold": item["label"], "pred": pred,
                         "ok": item["label"] == pred})
        if verbose:
            mark = "✓" if item["label"] == pred else "✗"
            print(f"  {mark} {item['id']:<22} gold={item['label']:<13} pred={pred}")
    summary = _metrics(rows)
    summary["method"] = method
    summary["confusion"] = _confusion(rows)
    summary["per_item"] = per_item
    return summary


def _print_report(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print(f"HaluBench — method={summary['method']}  n={summary['n']}")
    print("=" * 60)
    print(f"  accuracy : {summary['accuracy']:.3f}")
    print(f"  macro F1 : {summary['macro_f1']:.3f}")
    print("\n  per-class:")
    print(f"    {'class':<14}{'prec':>7}{'recall':>8}{'f1':>7}{'support':>9}")
    for label in LABELS:
        c = summary["per_class"][label]
        print(f"    {label:<14}{c['precision']:>7.3f}{c['recall']:>8.3f}"
              f"{c['f1']:>7.3f}{c['support']:>9}")
    print("\n  confusion (rows=gold, cols=pred):")
    header = "".join(f"{label[:5]:>9}" for label in LABELS)
    print(f"    {'':<14}{header}")
    for g in LABELS:
        cells = "".join(f"{summary['confusion'][g][p]:>9}" for p in LABELS)
        print(f"    {g:<14}{cells}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(prog="halubench")
    parser.add_argument("--method", default="lexical",
                        help="verify scoring method (lexical|nli|llm); default lexical")
    parser.add_argument("--verbose", action="store_true", help="print per-item predictions")
    parser.add_argument("--update", action="store_true",
                        help="write baseline_results.json from this run")
    args = parser.parse_args()

    summary = asyncio.run(run(args.method, args.verbose))
    _print_report(summary)

    if args.update:
        snapshot = {k: summary[k] for k in ("method", "n", "accuracy", "macro_f1", "per_class")}
        out = BASELINE if args.method == "lexical" else (
            BASELINE.parent / f"results_{args.method}.json")
        out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote results -> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
