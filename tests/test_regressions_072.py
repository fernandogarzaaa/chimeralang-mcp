"""Regression coverage for the bug fixes shipped in 0.7.2."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest

from chimeralang_mcp import server as srv
from chimeralang_mcp.persistence import PersistentNamespaceStore


def _call(tool: str, args: dict) -> tuple[bool, dict]:
    result = asyncio.run(srv.call_tool(tool, args))
    return result.isError, json.loads(result.content[0].text)


class TestCostTrackUnboundLocal(unittest.TestCase):
    """Issue 1: a local `log: list[str]` in the chimera_optimize handler
    shadowed the module-level logger, raising UnboundLocalError when other
    handlers (e.g. chimera_cost_track) called `log.info(...)`."""

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self._original_store = srv._store
        srv._store = PersistentNamespaceStore(self._tempdir.name)
        srv._cost_tracker_cache.clear()

    def tearDown(self):
        srv._store = self._original_store
        srv._cost_tracker_cache.clear()
        self._tempdir.cleanup()

    def test_cost_track_succeeds_without_optimize_first(self):
        is_error, payload = _call("chimera_cost_track", {
            "tokens_before": 1000,
            "tokens_after":  300,
            "label":         "regression-072",
            "namespace":     "regression-072",
        })
        self.assertFalse(is_error, payload)
        self.assertEqual(payload["tokens_before"], 1000)
        self.assertEqual(payload["tokens_after"], 300)
        self.assertGreater(payload["pct_saved"], 0)

    def test_cost_track_then_dashboard_consistency(self):
        _, _ = _call("chimera_cost_track", {
            "tokens_before": 500, "tokens_after": 100,
            "namespace": "regression-072-dash",
        })
        is_error, dash = _call("chimera_dashboard", {"namespace": "regression-072-dash"})
        self.assertFalse(is_error, dash)
        self.assertEqual(dash["request_count"], 1)
        self.assertEqual(dash["total_tokens_saved"], 400)


class TestModeParameterName(unittest.TestCase):
    """Issue 2: confirm chimera_mode reads `task_description` (the actual
    schema parameter), and that `task_type` is silently ignored as MCP
    drops unknown args. The 0.7.1 skill 'known issue' was misdocumented."""

    def test_task_description_drives_auto_mode(self):
        _, payload = _call("chimera_mode", {"task_description": "compress this long doc to save tokens"})
        self.assertEqual(payload.get("mode"), "token")

    def test_unknown_task_type_falls_through_to_default(self):
        _, payload = _call("chimera_mode", {"task_type": "compress this long doc"})
        self.assertEqual(payload.get("mode"), "minimal")


if __name__ == "__main__":
    unittest.main()
