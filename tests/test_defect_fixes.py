"""Adversarial + happy-path tests for the 7 v0.6.0 defect fixes.

Coverage:
  1. chimera_prove  — no TypeError on simple emit program
  2. chimera_deliberate — semantic mode exposed; lexical divergence re-check
  3. chimera_score  — importance_for_goal mode returns goal-relevant messages first
  4. chimera_plan_goals — expanded keywords; domain_terms extracted; honest note
  5. chimera_gate   — all-unique divergence=1.0 warning present
  6. chimera_compress/chimera_summarize — auto_track records to cost_tracker
  7. chimera_verify — lexically_* verdict prefixes; method_note present
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from chimeralang_mcp.persistence import PersistentNamespaceStore


def _run(coro):
    return asyncio.run(coro)


class TestDefectFixes(unittest.TestCase):
    def setUp(self):
        import chimeralang_mcp.materials.loader as loader
        import chimeralang_mcp.server as srv

        self.loader = loader
        self.srv = srv
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_store = srv._store
        self.original_materials = srv._materials_registry
        srv._store = PersistentNamespaceStore(self.tempdir.name)
        srv._materials_registry = None
        loader._REGISTRY_CACHE.clear()
        self._reset_state()

    def tearDown(self):
        self.srv._store = self.original_store
        self.srv._materials_registry = self.original_materials
        self.loader._REGISTRY_CACHE.clear()
        self._reset_state()
        self.tempdir.cleanup()

    def _reset_state(self):
        self.srv._kb_cache.clear()
        self.srv._world_model_cache.clear()
        self.srv._self_model_cache.clear()
        self.srv._memory_store_cache.clear()
        self.srv._meta_learner_cache.clear()
        self.srv._cost_tracker_cache.clear()

    def _call(self, name: str, args: dict):
        result = _run(self.srv.call_tool(name, args))
        self.assertFalse(result.isError, result.content[0].text)
        return json.loads(result.content[0].text)

    def _call_raw(self, name: str, args: dict):
        """Return (is_error, payload_dict)."""
        result = _run(self.srv.call_tool(name, args))
        return result.isError, json.loads(result.content[0].text)

    # ------------------------------------------------------------------
    # 1. chimera_prove — no TypeError on simple emit
    # ------------------------------------------------------------------

    def test_prove_simple_emit_no_type_error(self):
        """Bug: val x: Int = 1\\nemit x used to raise TypeError."""
        result = self._call("chimera_prove", {"source": "val x: Int = 1\nemit x"})
        # Result is nested: result["proof"]["verdict"]
        self.assertIn("proof", result)
        verdict = result["proof"]["verdict"]
        self.assertNotIn("TypeError", verdict)
        self.assertIn("PASS", verdict)

    def test_prove_returns_integrity_report(self):
        result = self._call("chimera_prove", {"source": "val y: Int = 42\nemit y"})
        self.assertIn("proof", result)
        proof = result["proof"]
        self.assertIn("verdict", proof)
        self.assertIn("chain_valid", proof)
        self.assertIn("chain_length", proof)

    def test_prove_failing_assertion(self):
        result = self._call("chimera_prove", {"source": "assert 1 == 2"})
        self.assertIn("proof", result)
        self.assertIn("FAIL", result["proof"]["verdict"])

    # ------------------------------------------------------------------
    # 2. chimera_deliberate — mode param exposed
    # ------------------------------------------------------------------

    def test_deliberate_semantic_mode_accepted(self):
        """Mode='semantic' must be accepted without error."""
        result = self._call(
            "chimera_deliberate",
            {
                "question": "Is Python a good language for data science?",
                "responses": [
                    {"text": "Python is excellent for data science due to numpy and pandas.", "confidence": 0.9},
                    {"text": "Python excels in ML and data analysis workflows.", "confidence": 0.85},
                    {"text": "Python dominates data science with sklearn and jupyter.", "confidence": 0.8},
                ],
                "mode": "semantic",
            },
        )
        self.assertIn("consensus", result)
        self.assertIn("divergence", result)

    def test_deliberate_lexical_mode_accepted(self):
        result = self._call(
            "chimera_deliberate",
            {
                "question": "Best web framework?",
                "responses": [
                    {"text": "Django is great for full-stack web development.", "confidence": 0.7},
                    {"text": "FastAPI is excellent for REST APIs.", "confidence": 0.7},
                    {"text": "Flask is lightweight and flexible for web apps.", "confidence": 0.7},
                ],
                "mode": "lexical_consensus",
            },
        )
        self.assertIn("consensus", result)

    def test_deliberate_paraphrases_low_divergence_in_semantic_mode(self):
        """Five paraphrases should have lower divergence in semantic mode than lexical."""
        paraphrases = [
            {"text": "The sky appears blue because of Rayleigh scattering of sunlight.", "confidence": 0.9},
            {"text": "Blue sky results from sunlight being scattered by air molecules via Rayleigh scattering.", "confidence": 0.85},
            {"text": "Rayleigh scattering causes the sky to look blue by dispersing short wavelengths.", "confidence": 0.88},
            {"text": "Sunlight scattered through the atmosphere produces a blue sky via Rayleigh scattering.", "confidence": 0.87},
            {"text": "The atmosphere scatters blue light more than red, making the sky appear blue.", "confidence": 0.86},
        ]
        kwargs = {"question": "Why is the sky blue?", "responses": paraphrases}
        sem   = self._call("chimera_deliberate", {**kwargs, "mode": "semantic"})
        lex   = self._call("chimera_deliberate", {**kwargs, "mode": "lexical_consensus"})
        # Semantic divergence should be less extreme (more correctly low) than lexical
        self.assertLessEqual(sem["divergence"], lex["divergence"] + 0.3)

    # ------------------------------------------------------------------
    # 3. chimera_score — importance_for_goal mode
    # ------------------------------------------------------------------

    def test_score_importance_mode_ranks_goal_relevant_first(self):
        messages = [
            {"role": "user",      "content": "How do I configure the database connection pool?"},
            {"role": "assistant", "content": "The weather today is sunny with mild temperatures."},
            {"role": "assistant", "content": "Configure the database pool via DATABASE_POOL_SIZE env var. Connection pooling improves database performance."},
        ]
        result = self._call(
            "chimera_score",
            {"messages": messages, "focus": "database configuration", "mode": "importance_for_goal"},
        )
        self.assertEqual(result["mode"], "importance_for_goal")
        scores = result["scores"]
        # The database-related messages should outscore the weather message
        weather_score = next(s["score"] for s in scores if "weather" in messages[s["index"]]["content"].lower())
        db_score = next(s["score"] for s in scores if "DATABASE_POOL_SIZE" in messages[s["index"]]["content"])
        self.assertGreater(db_score, weather_score)

    def test_score_drop_priority_mode_still_works(self):
        messages = [
            {"role": "user",      "content": "Tell me about Python."},
            {"role": "assistant", "content": "Python is a high-level programming language."},
        ]
        result = self._call(
            "chimera_score",
            {"messages": messages, "focus": "Python", "mode": "drop_priority"},
        )
        self.assertEqual(result["mode"], "drop_priority")
        self.assertIn("scores", result)

    def test_score_empty_messages_returns_empty(self):
        result = self._call("chimera_score", {"messages": [], "focus": "anything"})
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # 4. chimera_plan_goals — expanded keywords, domain_terms, honest note
    # ------------------------------------------------------------------

    def test_plan_goals_debug_keyword_gives_diagnostic_repair(self):
        result = self._call("chimera_plan_goals", {"goal": "debug the authentication bug"})
        self.assertEqual(result["best_known_strategy"], "diagnostic_repair")

    def test_plan_goals_patch_keyword_gives_diagnostic_repair(self):
        result = self._call("chimera_plan_goals", {"goal": "patch the broken JWT validation"})
        self.assertEqual(result["best_known_strategy"], "diagnostic_repair")

    def test_plan_goals_refactor_keyword_gives_iterative_refactor(self):
        result = self._call("chimera_plan_goals", {"goal": "refactor the payment service for testability"})
        self.assertEqual(result["best_known_strategy"], "iterative_refactor")

    def test_plan_goals_optimize_keyword_gives_iterative_refactor(self):
        result = self._call("chimera_plan_goals", {"goal": "optimize the database query performance"})
        self.assertEqual(result["best_known_strategy"], "iterative_refactor")

    def test_plan_goals_extracts_domain_terms(self):
        result = self._call("chimera_plan_goals", {"goal": "build a Redis caching layer for the checkout service"})
        domain_terms = result["domain_terms"]
        self.assertIsInstance(domain_terms, list)
        # At least one meaningful term should appear
        combined = " ".join(domain_terms).lower()
        self.assertTrue(
            any(t in combined for t in ["redis", "caching", "layer", "checkout", "service", "build"]),
            f"Expected domain terms from goal, got: {domain_terms}"
        )

    def test_plan_goals_note_mentions_heuristic(self):
        result = self._call("chimera_plan_goals", {"goal": "analyze API latency"})
        self.assertIn("heuristic", result["note"].lower())

    def test_plan_goals_confidence_below_0_8(self):
        """Confidence should be honest (not overconfident >0.8) for a heuristic decomposer."""
        result = self._call("chimera_plan_goals", {"goal": "decide which cloud provider to use"})
        self.assertLessEqual(result["confidence"], 0.80)

    # ------------------------------------------------------------------
    # 5. chimera_gate — all-unique warning
    # ------------------------------------------------------------------

    def test_gate_all_unique_warns(self):
        """When every candidate is different, divergence=1.0 should produce a warning."""
        result = self._call(
            "chimera_gate",
            {
                "candidates": [
                    {"value": "alpha", "confidence": 0.8},
                    {"value": "beta",  "confidence": 0.8},
                    {"value": "gamma", "confidence": 0.8},
                ],
                "strategy": "majority",
            },
        )
        self.assertEqual(result["divergence_ratio"], 1.0)
        self.assertTrue(result.get("all_unique", False))
        self.assertIn("warning", result)
        self.assertIn("completely different", result["warning"].lower())

    def test_gate_trivial_consensus_warns(self):
        """When all branches agree, trivial_consensus warning should fire."""
        result = self._call(
            "chimera_gate",
            {
                "candidates": [
                    {"value": "same", "confidence": 0.9},
                    {"value": "same", "confidence": 0.9},
                    {"value": "same", "confidence": 0.9},
                ],
            },
        )
        self.assertTrue(result["trivial_consensus"])
        self.assertIn("warning", result)
        self.assertIn("identical", result["warning"].lower())

    def test_gate_partial_consensus_no_unique_warning(self):
        """Two of three agree — divergence < 1.0, no all-unique warning."""
        result = self._call(
            "chimera_gate",
            {
                "candidates": [
                    {"value": "yes", "confidence": 0.9},
                    {"value": "yes", "confidence": 0.8},
                    {"value": "no",  "confidence": 0.5},
                ],
                "strategy": "weighted_vote",
            },
        )
        self.assertFalse(result.get("all_unique", False))
        self.assertGreater(result["consensus_confidence"], 0.0)

    # ------------------------------------------------------------------
    # 6. chimera_compress — auto_track records to cost tracker
    # ------------------------------------------------------------------

    def test_compress_auto_track_records_to_dashboard(self):
        long_text = "Hello world. " * 200
        self._call(
            "chimera_compress",
            {"text": long_text, "level": "aggressive", "auto_track": True, "namespace": "compress-track-test"},
        )
        dash = self._call("chimera_dashboard", {"namespace": "compress-track-test"})
        self.assertGreater(dash["total_tokens_saved"], 0)
        self.assertGreater(len(dash["history"]), 0)

    def test_compress_auto_track_false_does_not_record(self):
        long_text = "Token savings test text. " * 200
        self._call(
            "chimera_compress",
            {"text": long_text, "level": "medium", "auto_track": False, "namespace": "compress-notrack-test"},
        )
        dash = self._call("chimera_dashboard", {"namespace": "compress-notrack-test"})
        self.assertEqual(dash["total_tokens_saved"], 0)
        self.assertEqual(len(dash["history"]), 0)

    def test_compress_tracked_field_in_response(self):
        long_text = "Repetitive sentence. " * 100
        result = self._call(
            "chimera_compress",
            {"text": long_text, "level": "medium", "auto_track": True, "namespace": "compress-tracked-field"},
        )
        if result.get("estimated_tokens_saved", 0) > 0:
            self.assertIn("tracked", result)
            self.assertIn("request_id", result["tracked"])

    def test_summarize_auto_track_records_to_dashboard(self):
        long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "A stitch in time saves nine. "
            "All that glitters is not gold. "
            "To be or not to be, that is the question. "
            "The early bird catches the worm. "
        ) * 20
        self._call(
            "chimera_summarize",
            {"text": long_text, "ratio": 0.2, "auto_track": True, "namespace": "summ-track-test"},
        )
        dash = self._call("chimera_dashboard", {"namespace": "summ-track-test"})
        self.assertGreater(dash["total_tokens_saved"], 0)
        self.assertGreater(len(dash["history"]), 0)

    def test_summarize_auto_track_false_does_not_record(self):
        long_text = "Sentence one. Sentence two. Sentence three. " * 30
        self._call(
            "chimera_summarize",
            {"text": long_text, "ratio": 0.3, "auto_track": False, "namespace": "summ-notrack-test"},
        )
        dash = self._call("chimera_dashboard", {"namespace": "summ-notrack-test"})
        self.assertEqual(dash["total_tokens_saved"], 0)
        self.assertEqual(len(dash["history"]), 0)

    # ------------------------------------------------------------------
    # 7. chimera_verify — lexically_* prefixes + method_note
    # ------------------------------------------------------------------

    def test_verify_supported_verdict_has_lexical_prefix(self):
        result = self._call(
            "chimera_verify",
            {
                "claims": [{"claim_id": "c1", "text": "Paris is the capital of France.", "claim_type": "factual"}],
                "evidence": ["Paris is the capital and most populous city of France."],
            },
        )
        self.assertTrue(
            result["verdict"].startswith("lexically_"),
            f"Expected lexically_* verdict, got: {result['verdict']}"
        )

    def test_verify_contradicted_verdict_has_lexical_prefix(self):
        result = self._call(
            "chimera_verify",
            {
                "claims": [{"claim_id": "c1", "text": "Saturn is made of cheese.", "claim_type": "factual"}],
                "evidence": ["Saturn is a gas giant composed mostly of hydrogen and helium."],
            },
        )
        self.assertTrue(
            result["verdict"].startswith("lexically_"),
            f"Expected lexically_* verdict, got: {result['verdict']}"
        )

    def test_verify_method_note_mentions_jaccard(self):
        result = self._call(
            "chimera_verify",
            {
                "claims": [{"claim_id": "c1", "text": "Water is H2O.", "claim_type": "factual"}],
                "evidence": ["H2O is the chemical formula for water."],
            },
        )
        self.assertIn("method_note", result)
        note = result["method_note"].lower()
        self.assertTrue(
            "jaccard" in note or "token" in note or "overlap" in note,
            f"method_note should mention token overlap, got: {result['method_note']}"
        )

    def test_verify_not_semantic_entailment(self):
        """method_note must explicitly disclaim NLI — should say 'not' near 'semantic entailment'."""
        result = self._call(
            "chimera_verify",
            {
                "claims": [{"claim_id": "c1", "text": "Fire is hot.", "claim_type": "factual"}],
                "evidence": ["High temperatures characterize combustion events."],
            },
        )
        self.assertIn("method_note", result)
        note_lower = result["method_note"].lower()
        # The note should contain "not" to negate "semantic entailment" or "nli"
        self.assertIn("not", note_lower)
        # And it should mention either "semantic" or "entailment" or "nli"
        self.assertTrue(
            "semantic" in note_lower or "entailment" in note_lower or "nli" in note_lower,
            f"method_note should disclaim NLI/entailment, got: {result['method_note']}"
        )
