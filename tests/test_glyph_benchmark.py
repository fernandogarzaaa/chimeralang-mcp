"""tests/test_glyph_benchmark.py — locks the 0.7.3 Glyph benchmark numbers.

After the Phase 1 falsification (BLUEPRINT.md + docs/case-studies), the
project ships an empirically-optimized lexicon and publishes -16% / +19.2%
/ 0.806 as the headline numbers. This test ensures any future change to
the encoder, decoder, or LEXICON cannot silently regress those numbers.

If you change the lexicon and these tests fail, that's expected: re-run
`python tools/optimize_lexicon.py` and `python tools/glyph_benchmark.py`,
then update the bounds below to the new measurements. The point is that
the bench numbers move *visibly*, with intent, not by accident.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Skip the file entirely if tiktoken is not installed (e.g. on a minimal
# CI image). The bench is measurement-only; production code does not
# depend on tiktoken.
try:
    import tiktoken  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("tiktoken not installed; benchmark test skipped")

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from glyph_benchmark import load_corpus, run_benchmark  # noqa: E402


class TestGlyphBenchmark(unittest.TestCase):
    """Headline-number regression test. Bounds are intentionally tight
    (±0.5pp) so any encoder/lexicon change must update them deliberately."""

    @classmethod
    def setUpClass(cls):
        sentences = load_corpus()
        cls.report = run_benchmark(sentences, validate_anthropic=False)
        cls.overall = cls.report["overall"]

    def test_corpus_size_locked(self):
        # Corpus shape is part of the benchmark identity; changing it
        # changes the meaning of all the numbers below.
        self.assertEqual(self.report["n_sentences"], 100)
        self.assertEqual(self.report["n_domains"], 5)

    def test_token_pct_within_published_range(self):
        # 0.7.3 published number: -16.0%. Bound to ±1pp so we catch
        # silent regressions without being so tight that benign refactors
        # of unrelated code break the test.
        pct = self.overall["pct_saved_tokens"]
        self.assertGreaterEqual(pct, -17.0,
                                f"token reduction worse than published: {pct}%")
        self.assertLessEqual(pct, -15.0,
                             f"token reduction better than published — please "
                             f"update the published number in docs/case-studies "
                             f"and AGENTS.md / SKILL.md: {pct}%")

    def test_char_pct_within_published_range(self):
        # 0.7.3 published number: +19.2%.
        pct = self.overall["pct_saved_chars"]
        self.assertGreaterEqual(pct, 18.0, f"char reduction regressed: {pct}%")
        self.assertLessEqual(pct, 21.0, f"char reduction improved — update docs: {pct}%")

    def test_decode_fidelity_above_floor(self):
        # 0.7.3 published number: 0.806. The decoder is lossy by design
        # but should not slip below 0.75 — that's where it stops being
        # useful as a wire format.
        fid = self.overall["avg_decode_fidelity"]
        self.assertGreaterEqual(fid, 0.75,
                                f"decode fidelity dropped below published floor: {fid}")

    def test_no_domain_decode_fidelity_collapses(self):
        # No single domain should ever drop below 0.50 — that's where
        # the wire-format claim breaks. (Dialogue is the weakest domain
        # at 0.70 today; we leave a real margin.)
        for domain, agg in self.report["by_domain"].items():
            with self.subTest(domain=domain):
                self.assertGreaterEqual(
                    agg["avg_decode_fidelity"], 0.50,
                    f"{domain!r} fidelity collapsed: {agg['avg_decode_fidelity']}",
                )


if __name__ == "__main__":
    unittest.main()
