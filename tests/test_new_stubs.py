"""test_new_stubs.py — tests for chimera_embodied, chimera_social, chimera_transfer_learn, chimera_evolve."""
from __future__ import annotations

import unittest

from chimeralang_mcp.server import (
    _EmbodiedState, _SocialCognition, _TransferLearner, _EvolutionEngine,
    _get_embodied, _get_social, _get_transfer, _get_evolve,
)


class TestEmbodied(unittest.TestCase):
    def setUp(self):
        self.eb = _EmbodiedState()

    def test_status_defaults(self):
        s = self.eb.status()
        self.assertEqual(s["energy"], 1.0)
        self.assertIsNone(s["last_action"])

    def test_perceive(self):
        r = self.eb.perceive(["tree", "rock"], "forest")
        self.assertTrue(r["perceived"])
        self.assertEqual(r["objects"], ["tree", "rock"])
        self.assertEqual(self.eb.perception["environment"], "forest")

    def test_act_reduces_energy(self):
        r = self.eb.act("move", {"direction": "north"})
        self.assertTrue(r["executed"])
        self.assertLess(r["energy_after"], 1.0)

    def test_act_log_capped_at_20(self):
        for i in range(25):
            self.eb.act(f"action_{i}", {})
        self.assertLessEqual(len(self.eb.action_log), 20)

    def test_reset(self):
        self.eb.act("move", {})
        r = self.eb.reset()
        self.assertTrue(r["reset"])
        self.assertEqual(self.eb.energy, 1.0)
        self.assertEqual(len(self.eb.action_log), 0)


class TestSocial(unittest.TestCase):
    def setUp(self):
        self.sc = _SocialCognition()

    def test_record_interaction(self):
        r = self.sc.record_interaction("alice", "work", 0.8)
        self.assertEqual(r["agent"], "alice")
        self.assertEqual(r["interaction_count"], 1)
        self.assertAlmostEqual(r["sentiment_avg"], 0.8)

    def test_relationship_strengthens(self):
        for _ in range(5):
            self.sc.record_interaction("bob", "project", 0.9)
        r = self.sc.query("bob")
        self.assertGreater(r["relationship_strength"], 0.5)

    def test_query_unknown_agent(self):
        r = self.sc.query("nobody")
        self.assertFalse(r["found"])

    def test_list_agents(self):
        self.sc.record_interaction("carol", "chat", 0.0)
        r = self.sc.list_agents()
        self.assertIn("carol", r["agents"])
        self.assertEqual(r["count"], 1)

    def test_sentiment_clamped(self):
        r = self.sc.record_interaction("dave", "x", 5.0)
        self.assertLessEqual(r["sentiment_avg"], 1.0)


class TestTransferLearner(unittest.TestCase):
    def setUp(self):
        self.tl = _TransferLearner()

    def test_add_and_query(self):
        self.tl.add_mapping("physics", "finance", "momentum", "market momentum", 0.9)
        r = self.tl.query("physics", "finance")
        self.assertEqual(r["count"], 1)
        self.assertEqual(r["matches"][0]["analogy"], "market momentum")

    def test_query_no_results(self):
        r = self.tl.query("biology", "cooking")
        self.assertEqual(r["count"], 0)

    def test_list_all(self):
        self.tl.add_mapping("A", "B", "concept", "analogy", 0.7)
        r = self.tl.list_all()
        self.assertEqual(r["total_mappings"], 1)

    def test_confidence_clamped(self):
        self.tl.add_mapping("x", "y", "c", "a", 5.0)
        r = self.tl.query("x", "y")
        self.assertLessEqual(r["matches"][0]["confidence"], 1.0)


class TestEvolutionEngine(unittest.TestCase):
    def setUp(self):
        self.ev = _EvolutionEngine()
        self.candidates = [
            {"id": "A", "value": "option_a", "fitness_score": 0.9},
            {"id": "B", "value": "option_b", "fitness_score": 0.5},
            {"id": "C", "value": "option_c", "fitness_score": 0.3},
        ]

    def test_run_returns_best(self):
        r = self.ev.run(self.candidates, generations=3, mutation_rate=0.05, survival_ratio=0.5)
        self.assertIn("best", r)
        self.assertIn("ranked", r)
        self.assertGreater(len(r["ranked"]), 0)

    def test_run_has_generation_history(self):
        r = self.ev.run(self.candidates, generations=5, mutation_rate=0.1, survival_ratio=0.5)
        self.assertEqual(len(r["generations"]), 5)

    def test_best_fitness_in_range(self):
        r = self.ev.run(self.candidates, generations=3, mutation_rate=0.1, survival_ratio=0.5)
        self.assertGreaterEqual(r["best"]["fitness_score"], 0.0)
        self.assertLessEqual(r["best"]["fitness_score"], 1.0)

    def test_info_before_run(self):
        r = self.ev.info()
        self.assertIn("note", r)

    def test_info_after_run(self):
        self.ev.run(self.candidates, generations=2, mutation_rate=0.1, survival_ratio=0.5)
        r = self.ev.info()
        self.assertIn("last_run_best", r)


if __name__ == "__main__":
    unittest.main()
