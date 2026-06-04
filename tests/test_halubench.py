"""tests/test_halubench.py — pytest regression around HaluBench.

HaluBench (tools/halubench/) measures chimera_verify's hallucination-detection
quality against a hand-labeled corpus. This test locks the v1 lexical baseline:

  1. The corpus stays balanced (10 per verdict class) and leakage-free.
  2. The lexical method's published metrics stay within tolerance bounds.
  3. The known weakness (contradiction recall == 0.0 for lexical) is asserted
     explicitly, so when the Phase 5.2 semantic tier lifts it, that change is a
     deliberate, visible event — not a silent drift.

If you change the verify scoring engine, this test will move. Re-run
`python -m tools.halubench.run --update` to regenerate the baseline
intentionally; do NOT silently edit baseline_results.json to pass.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from tools.halubench.run import LABELS, run
from tools.halubench import run_detect

CORPUS = Path(__file__).parent.parent / "tools" / "halubench" / "corpus.json"


def _gold_claims_lower() -> set[str]:
    # The built-in verification_gold pack — these must not leak into HaluBench.
    from chimeralang_mcp.materials.builders import _VERIFICATION_GOLD
    return {str(r["claim"]).strip().lower() for r in _VERIFICATION_GOLD}


class TestHaluBench(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        cls.summary = asyncio.run(run("lexical", verbose=False))

    # ── corpus integrity ──────────────────────────────────────────────

    def test_corpus_balanced_across_classes(self):
        counts = {label: 0 for label in LABELS}
        for item in self.corpus["items"]:
            counts[item["label"]] += 1
        self.assertEqual(counts, {"supported": 10, "contradicted": 10, "insufficient": 10})

    def test_corpus_has_no_gold_leakage(self):
        gold = _gold_claims_lower()
        leaked = [it["id"] for it in self.corpus["items"]
                  if it["claim"].strip().lower() in gold]
        self.assertEqual(leaked, [], f"claims leaked from verification_gold: {leaked}")

    def test_corpus_ids_unique(self):
        ids = [it["id"] for it in self.corpus["items"]]
        self.assertEqual(len(ids), len(set(ids)))

    # ── locked baseline (lexical method) ──────────────────────────────

    def test_lexical_macro_f1_baseline(self):
        # v1 baseline 0.5253; lock within ±0.03 so engine drift is caught.
        self.assertAlmostEqual(self.summary["macro_f1"], 0.525, delta=0.03)

    def test_lexical_accuracy_baseline(self):
        self.assertAlmostEqual(self.summary["accuracy"], 0.633, delta=0.03)

    def test_lexical_contradiction_recall_is_zero(self):
        """Documented weakness: lexical scoring catches no contradictions.

        This is the headline finding that motivates the Phase 5.2 semantic
        tier. If a future change lifts this above zero, update this test
        deliberately — the lift is the whole point of 5.2.
        """
        self.assertEqual(self.summary["per_class"]["contradicted"]["recall"], 0.0)

    def test_lexical_insufficient_detection_strong(self):
        # Off-topic evidence is handled well by lexical overlap.
        self.assertGreaterEqual(self.summary["per_class"]["insufficient"]["f1"], 0.85)


from chimeralang_mcp import semantic  # noqa: E402


@unittest.skipUnless(semantic.available("nli"),
                     "nli method unavailable ([semantic] extra not installed)")
class TestHaluBenchNLI(unittest.TestCase):
    """Phase 5.2: the semantic tier must beat the lexical baseline's headline
    weakness. Skips cleanly where the optional model isn't installed (e.g. CI)."""

    @classmethod
    def setUpClass(cls):
        cls.summary = asyncio.run(run("nli", verbose=False))

    def test_nli_lifts_contradiction_recall_off_zero(self):
        # Lexical baseline is 0.0; the whole point of 5.2 is to fix that.
        recall = self.summary["per_class"]["contradicted"]["recall"]
        self.assertGreaterEqual(recall, 0.8,
                                f"nli contradiction recall {recall} should beat lexical 0.0")

    def test_nli_macro_f1_beats_lexical(self):
        # Lexical baseline macro-F1 is 0.525.
        self.assertGreater(self.summary["macro_f1"], 0.80)


@unittest.skipUnless(semantic.available("nli"),
                     "nli method unavailable ([semantic] extra not installed)")
class TestHaluBenchRAG(unittest.TestCase):
    """Phase 5.3 RAG: grounded verify retrieves evidence from a shared pool
    instead of being handed the exact snippet. Retrieval costs some accuracy vs
    oracle nli (0.933) but must still clear the lexical baseline (0.633)."""

    @classmethod
    def setUpClass(cls):
        cls.summary = asyncio.run(run("nli", verbose=False, rag=True))

    def test_rag_beats_lexical_baseline(self):
        self.assertGreater(self.summary["accuracy"], 0.70)

    def test_rag_keeps_contradiction_recall_high(self):
        self.assertGreaterEqual(self.summary["per_class"]["contradicted"]["recall"], 0.8)


class TestDetectBench(unittest.TestCase):
    """Phase 5.3: calibrate chimera_detect's two signals. Deterministic — no
    optional deps. Locks both the strength (high-precision certainty flagging)
    and the documented weakness (narrow injection recall)."""

    @classmethod
    def setUpClass(cls):
        cls.results = asyncio.run(run_detect.run(verbose=False))

    def test_certainty_high_precision(self):
        # No false positives on hedged/measured phrasing.
        self.assertEqual(self.results["certainty"]["precision"], 1.0)

    def test_certainty_recall_solid_but_imperfect(self):
        # Catches listed markers; misses synonyms outside the substring list.
        recall = self.results["certainty"]["recall"]
        self.assertGreaterEqual(recall, 0.7)
        self.assertLess(recall, 1.0)

    def test_injection_precision_no_false_positives(self):
        # Benign requests must never be flagged as attacks.
        self.assertEqual(self.results["injection"]["precision"], 1.0)

    def test_injection_recall_is_narrow_documented_weakness(self):
        """Headline 5.3 finding: detect's attack-pattern matching is narrow —
        it catches the canonical 'ignore all previous instructions' phrasing but
        misses most variants. Locked so a future attack-pattern expansion that
        lifts recall is a deliberate, visible event."""
        recall = self.results["injection"]["recall"]
        self.assertGreater(recall, 0.0)   # catches at least the canonical case
        self.assertLess(recall, 0.5)      # but most variants slip through today


if __name__ == "__main__":
    unittest.main()
