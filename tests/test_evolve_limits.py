"""Regression tests: chimera_evolve inputs are clamped so one call cannot exhaust memory."""
from __future__ import annotations

import asyncio
import json
import unittest

from chimeralang_mcp.server import _EvolutionEngine, _clamp_num, call_tool


def _cands(n: int) -> list[dict]:
    return [{"id": f"c{i}", "value": i, "fitness_score": i / max(1, n)} for i in range(n)]


class TestEvolveLimits(unittest.TestCase):
    def test_survival_ratio_one_no_longer_doubles_without_bound(self):
        # Before the fix: 10 * 2**40 candidates -> MemoryError. Now capped.
        r = _EvolutionEngine().run(_cands(10), generations=40, mutation_rate=0.1, survival_ratio=1.0)
        self.assertEqual(len(r["generations"]), 40)
        self.assertLessEqual(r["survivors"], _EvolutionEngine.MAX_POPULATION)
        self.assertEqual(r["survivors"], _EvolutionEngine.MAX_POPULATION)
        self.assertTrue(all(h["population"] <= 1000 for h in r["generations"]))

    def test_population_cap_keeps_the_fittest(self):
        r = _EvolutionEngine().run(_cands(10), generations=20, mutation_rate=0.0, survival_ratio=1.0)
        scores = [c["fitness_score"] for c in r["ranked"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(r["best"]["id"], "c9")

    def test_generations_clamped(self):
        e = _EvolutionEngine()
        self.assertEqual(len(e.run(_cands(3), 10**9, 0.1, 0.5)["generations"]), 100)
        self.assertEqual(len(e.run(_cands(3), 0, 0.1, 0.5)["generations"]), 1)
        self.assertEqual(len(e.run(_cands(3), -5, 0.1, 0.5)["generations"]), 1)

    def test_survival_ratio_and_mutation_clamped(self):
        r = _EvolutionEngine().run(_cands(4), 2, 50.0, 7.0)
        self.assertEqual(r["effective_params"]["survival_ratio"], 1.0)
        self.assertEqual(r["effective_params"]["mutation_rate"], 1.0)
        r = _EvolutionEngine().run(_cands(4), 2, -1.0, -3.0)
        self.assertEqual(r["effective_params"]["survival_ratio"], 0.01)
        self.assertEqual(r["effective_params"]["mutation_rate"], 0.0)

    def test_candidates_truncated(self):
        r = _EvolutionEngine().run(_cands(600), 1, 0.0, 0.5)
        self.assertTrue(r["candidates_truncated"])
        self.assertEqual(r["initial_count"], 600)
        self.assertEqual(r["survivors"], 500)  # 500 kept * 0.5 survive, then + 250 mutated
        r = _EvolutionEngine().run(_cands(500), 1, 0.0, 0.5)
        self.assertFalse(r["candidates_truncated"])

    def test_default_ratio_keeps_population_steady(self):
        r = _EvolutionEngine().run(_cands(8), 5, 0.1, 0.5)
        self.assertEqual([h["population"] for h in r["generations"]], [8] * 5)

    def test_clamp_num(self):
        self.assertEqual(_clamp_num("abc", 0, 1, 0.5), 0.5)
        self.assertEqual(_clamp_num(None, 0, 1, 0.5), 0.5)
        self.assertEqual(_clamp_num(float("nan"), 0, 1, 0.5), 0.5)
        self.assertEqual(_clamp_num(float("inf"), 0, 1, 0.5), 0.5)
        self.assertEqual(_clamp_num(2, 0, 1, 0.5), 1)
        self.assertEqual(_clamp_num(-2, 0, 1, 0.5), 0)
        self.assertEqual(_clamp_num("0.25", 0, 1, 0.5), 0.25)
        self.assertEqual(_clamp_num(0, 0, 1, 0.5), 0)
        self.assertEqual(_clamp_num(1, 0, 1, 0.5), 1)


class TestEvolveTool(unittest.TestCase):
    def _call(self, args):
        return asyncio.run(call_tool("chimera_evolve", args))

    def test_tool_huge_args_are_clamped(self):
        res = self._call({"action": "run", "candidates": _cands(5), "generations": 10**6,
                          "survival_ratio": 1, "mutation_rate": 0.1})
        self.assertFalse(res.is_error, res.content[0].text)
        body = json.loads(res.content[0].text)
        self.assertEqual(len(body["generations"]), 100)
        self.assertLessEqual(body["survivors"], 1000)

    def test_tool_non_numeric_args_fall_back(self):
        res = self._call({"action": "run", "candidates": _cands(4), "generations": "lots", "survival_ratio": "x"})
        self.assertFalse(res.is_error, res.content[0].text)
        body = json.loads(res.content[0].text)
        self.assertEqual(body["effective_params"]["generations"], 5)
        self.assertEqual(body["effective_params"]["survival_ratio"], 0.5)

    def test_tool_rejects_bad_candidates(self):
        self.assertTrue(self._call({"action": "run", "candidates": [{"id": "a"}]}).is_error)
        self.assertTrue(self._call({"action": "run", "candidates": "nope"}).is_error)
        self.assertTrue(self._call({"action": "run"}).is_error)


if __name__ == "__main__":
    unittest.main()
