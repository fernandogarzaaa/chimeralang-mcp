"""tests/test_chimerabench.py — pytest regression around ChimeraBench.

ChimeraBench (tools/chimerabench/) is a corpus of verifiable agent tasks
where every gate-tool invocation's `provenance.program_hash` is the
canonical identity. This test runs the full corpus and asserts:
  1. Every task passes its assertions.
  2. Every step's actual program_hash matches the recorded canonical hash.
  3. Every family declared in tools/chimerabench/__init__.py has at least
     one task in the corpus.

If you change the gate-tool semantics (encoder, lexicon, scoring), this
test will fail. Re-run `python -m tools.chimerabench.run --update` to
regenerate canonical hashes intentionally; do NOT silently update the
JSON to pass the test.
"""
from __future__ import annotations

import asyncio
import unittest

from tools.chimerabench.run import run_all


EXPECTED_FAMILIES = frozenset({
    "gate-only",
    "verify-pipeline",
    "deliberate-converge",
    "vote-converge",
    "mixed-pipeline",
})


class TestChimeraBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = asyncio.run(run_all(update_canonical=False))

    def test_all_expected_families_have_tasks(self):
        present = set(self.summary.by_family.keys())
        missing = EXPECTED_FAMILIES - present
        self.assertEqual(missing, set(),
                         f"families with no tasks in corpus: {missing}")

    def test_corpus_size_floor(self):
        # The Phase 3 launch shipped 15 tasks (3 per family). Don't let
        # the corpus shrink below 15 without a deliberate update here.
        self.assertGreaterEqual(self.summary.total, 15,
                                f"corpus shrank to {self.summary.total} tasks")

    def test_every_task_passes(self):
        failures = [tr for tr in self.summary.results if not tr.passed]
        if failures:
            lines = []
            for tr in failures:
                lines.append(f"\n  {tr.family}/{tr.task_id}:")
                for failure in tr.failures:
                    lines.append(f"    - {failure}")
            self.fail(
                f"{len(failures)} ChimeraBench tasks failed:" + "".join(lines)
                + "\n\nRun `python -m tools.chimerabench.run --update` ONLY if "
                  "the change is intentional and you want to bless new hashes."
            )

    def test_every_step_has_canonical_hash(self):
        # No task may ship without a recorded canonical hash for every
        # step — that would make hash regressions invisible.
        for tr in self.summary.results:
            for idx, step in enumerate(tr.steps):
                with self.subTest(task=f"{tr.family}/{tr.task_id}", step=idx):
                    self.assertIsNotNone(step.expected_hash,
                                         f"missing canonical hash for step {idx}")
                    self.assertTrue(step.hash_match,
                                    f"hash mismatch on step {idx}")


if __name__ == "__main__":
    unittest.main()
