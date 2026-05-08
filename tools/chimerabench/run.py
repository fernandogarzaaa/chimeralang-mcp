"""ChimeraBench harness — runs the corpus of verifiable agent tasks and
reports per-task pass/fail plus aggregate stats.

Usage:
    python -m tools.chimerabench.run                # run all tasks, report
    python -m tools.chimerabench.run --update       # regenerate canonical hashes
    python -m tools.chimerabench.run --filter gate  # run only matching task families
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO))

from chimeralang_mcp import server as srv  # noqa: E402

TASKS_DIR = Path(__file__).parent / "tasks"


# ── data model ───────────────────────────────────────────────────────────


@dataclass
class StepResult:
    tool: str
    is_error: bool
    payload: dict[str, Any]
    expected_hash: str | None = None
    actual_hash: str | None = None
    hash_match: bool = False


@dataclass
class TaskResult:
    task_id: str
    family: str
    passed: bool
    steps: list[StepResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@dataclass
class BenchSummary:
    total: int = 0
    passed: int = 0
    failed: int = 0
    by_family: dict[str, dict[str, int]] = field(default_factory=dict)
    results: list[TaskResult] = field(default_factory=list)


# ── helpers ──────────────────────────────────────────────────────────────


def _path_get(payload: dict, path: str) -> Any:
    """Resolve a dotted path like `foo.bar.0` against a dict/list payload."""
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _load_tasks(filter_substr: str | None = None) -> list[dict]:
    tasks: list[dict] = []
    for path in sorted(TASKS_DIR.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["_path"] = str(path.relative_to(REPO))
        if filter_substr and filter_substr not in data.get("family", "") \
                and filter_substr not in data.get("id", ""):
            continue
        tasks.append(data)
    return tasks


async def _run_step(step: dict) -> tuple[bool, dict]:
    result = await srv.call_tool(step["tool"], step["args"])
    payload = json.loads(result.content[0].text)
    return result.isError, payload


# ── runner ───────────────────────────────────────────────────────────────


async def run_task(task: dict, *, update_canonical: bool = False) -> TaskResult:
    """Execute one task. If `update_canonical`, regenerates expected hashes."""
    tr = TaskResult(task_id=task["id"], family=task["family"], passed=True)
    expected_hashes = list(task.get("expected", {}).get("program_hashes", []))

    for idx, step in enumerate(task["pipeline"]):
        is_error, payload = await _run_step(step)
        prov = payload.get("provenance") or {}
        actual_hash = prov.get("program_hash")
        expected_hash = expected_hashes[idx] if idx < len(expected_hashes) else None

        sr = StepResult(
            tool=step["tool"],
            is_error=is_error,
            payload=payload,
            expected_hash=expected_hash,
            actual_hash=actual_hash,
            hash_match=(actual_hash == expected_hash) if expected_hash else False,
        )
        tr.steps.append(sr)

        if is_error:
            tr.passed = False
            tr.failures.append(f"step {idx} ({step['tool']}) returned error")
            continue
        if not actual_hash:
            tr.passed = False
            tr.failures.append(f"step {idx} ({step['tool']}) missing provenance.program_hash")
            continue
        if not update_canonical and expected_hash and not sr.hash_match:
            tr.passed = False
            tr.failures.append(
                f"step {idx} ({step['tool']}) hash mismatch: "
                f"got {actual_hash[:12]}…, expected {expected_hash[:12]}…"
            )

    # Run assertions against step payloads.
    for assertion in task.get("expected", {}).get("assertions", []):
        step_idx = int(assertion["step"])
        if step_idx >= len(tr.steps):
            tr.passed = False
            tr.failures.append(f"assertion targets non-existent step {step_idx}")
            continue
        actual_value = _path_get(tr.steps[step_idx].payload, assertion["path"])
        expected_value = assertion["equals"]
        if actual_value != expected_value:
            tr.passed = False
            tr.failures.append(
                f"assertion failed at step {step_idx} path={assertion['path']!r}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )

    if update_canonical:
        # Rewrite the task file with regenerated canonical hashes.
        task_path = REPO / task["_path"]
        original = json.loads(task_path.read_text(encoding="utf-8"))
        original["expected"] = original.get("expected", {})
        original["expected"]["program_hashes"] = [s.actual_hash for s in tr.steps]
        task_path.write_text(json.dumps(original, indent=2) + "\n", encoding="utf-8")

    return tr


async def run_all(*, filter_substr: str | None = None,
                  update_canonical: bool = False) -> BenchSummary:
    summary = BenchSummary()
    tasks = _load_tasks(filter_substr=filter_substr)
    for task in tasks:
        tr = await run_task(task, update_canonical=update_canonical)
        summary.results.append(tr)
        summary.total += 1
        fam = summary.by_family.setdefault(tr.family, {"total": 0, "passed": 0, "failed": 0})
        fam["total"] += 1
        if tr.passed:
            summary.passed += 1
            fam["passed"] += 1
        else:
            summary.failed += 1
            fam["failed"] += 1
    return summary


def print_report(summary: BenchSummary, verbose: bool = False) -> None:
    print("=" * 72)
    print(f"CHIMERABENCH — {summary.total} tasks, "
          f"{summary.passed} passed, {summary.failed} failed")
    print("=" * 72)
    print()
    if summary.by_family:
        print("BY FAMILY")
        for fam, stats in sorted(summary.by_family.items()):
            print(f"  {fam:<24} {stats['passed']:>3}/{stats['total']:>3}")
        print()
    if summary.failed or verbose:
        print("PER-TASK")
        for tr in summary.results:
            mark = "✓" if tr.passed else "✗"
            print(f"  [{mark}] {tr.family}/{tr.task_id}")
            for failure in tr.failures:
                print(f"        - {failure}")
            if verbose:
                for idx, step in enumerate(tr.steps):
                    h = (step.actual_hash or "—")[:16]
                    print(f"        step {idx} {step.tool}: {h}…")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Regenerate canonical hashes in task JSON files")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run tasks whose family or id contains this substring")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-step hashes")
    args = parser.parse_args()

    summary = asyncio.run(run_all(filter_substr=args.filter,
                                   update_canonical=args.update))
    print_report(summary, verbose=args.verbose)
    sys.exit(0 if summary.failed == 0 else 1)


if __name__ == "__main__":
    main()
