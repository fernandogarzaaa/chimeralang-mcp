from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from chimeralang_mcp.persistence import PersistentNamespaceStore


def _run(coro):
    return asyncio.run(coro)


class TestEnvelopeAndPersistenceFeatures(unittest.TestCase):
    def setUp(self):
        import chimeralang_mcp.server as srv

        self.srv = srv
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_store = srv._store
        srv._store = PersistentNamespaceStore(self.tempdir.name)
        self._reset_state()

    def tearDown(self):
        self.srv._store = self.original_store
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

    def test_claims_extracts_atomic_statements(self):
        result = self._call(
            "chimera_claims",
            {
                "namespace": "claims-test",
                "text": "Paris is the capital of France. The Seine runs through Paris. Maybe this is enough.",
            },
        )
        self.assertGreaterEqual(result["claim_count"], 2)
        self.assertEqual(len(result["envelope"]["claims"]), result["claim_count"])

    def test_verify_splits_supported_and_unsupported_claims(self):
        result = self._call(
            "chimera_verify",
            {
                "namespace": "verify-test",
                "text": "Paris is the capital of France. Saturn is made of cheese.",
                "evidence": ["Paris is the capital of France and its metro area is large."],
            },
        )
        self.assertEqual(len(result["verified_claims"]), 1)
        self.assertEqual(len(result["unsupported_claims"]), 1)
        self.assertFalse(result["supported"])

    def test_provenance_merge_combines_multiple_envelopes(self):
        left = self._call("chimera_confident", {"namespace": "merge-test", "value": {"city": "Paris"}, "confidence": 0.99})
        right = self._call("chimera_explore", {"namespace": "merge-test", "value": {"idea": "draft"}, "confidence": 0.4})
        merged = self._call(
            "chimera_provenance_merge",
            {
                "namespace": "merge-test",
                "envelopes": [left["envelope"], right["envelope"]],
                "strategy": "mean",
            },
        )
        self.assertEqual(merged["merged_from"], 2)
        self.assertEqual(len(merged["envelope"]["metadata"]["merged_from"]), 2)

    def test_policy_apply_requires_sources_for_strict_factual(self):
        result = self._call(
            "chimera_policy",
            {
                "namespace": "policy-test",
                "action": "apply",
                "policy": "strict_factual",
                "value": {"answer": "Paris"},
            },
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("sources" in warning.lower() for warning in result["warnings"]))

    def test_trace_returns_latest_envelope(self):
        confident = self._call(
            "chimera_confident",
            {"namespace": "trace-test", "value": {"city": "Paris"}, "confidence": 0.99},
        )
        trace = self._call("chimera_trace", {"namespace": "trace-test", "action": "latest"})
        self.assertEqual(trace["latest_trace"]["envelope_id"], confident["envelope"]["envelope_id"])

    def test_knowledge_persists_after_cache_reset(self):
        namespace = "knowledge-persist"
        added = self._call(
            "chimera_knowledge",
            {
                "namespace": namespace,
                "action": "add",
                "content": "Paris is the capital of France.",
                "category": "geography",
                "tags": ["capital"],
            },
        )
        self.assertTrue(added["added"])
        self._reset_state()
        result = self._call(
            "chimera_knowledge",
            {"namespace": namespace, "action": "search", "query": "Paris"},
        )
        self.assertEqual(len(result["results"]), 1)

    def test_world_model_persists_after_cache_reset(self):
        namespace = "world-persist"
        updated = self._call(
            "chimera_world_model",
            {
                "namespace": namespace,
                "action": "update",
                "key": "capital_france",
                "value": "Paris",
                "confidence": 0.95,
            },
        )
        self.assertEqual(updated["updated"], "capital_france")
        self._reset_state()
        result = self._call(
            "chimera_world_model",
            {"namespace": namespace, "action": "query", "key": "capital_france"},
        )
        self.assertEqual(result["value"], "Paris")


if __name__ == "__main__":
    unittest.main()
