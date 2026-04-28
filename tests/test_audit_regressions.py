from __future__ import annotations

import asyncio
import json
import unittest


def _run(coro):
    return asyncio.run(coro)


class TestAuditRegressions(unittest.TestCase):
    def setUp(self):
        from chimeralang_mcp.server import call_tool, list_tools

        self.call_tool = call_tool
        self.list_tools = list_tools

    def _call(self, name: str, args: dict):
        result = _run(self.call_tool(name, args))
        self.assertFalse(result.isError, result.content[0].text)
        return json.loads(result.content[0].text)

    def test_confident_preserves_structured_value(self):
        payload = {"city": "Paris", "country": "France"}
        result = self._call("chimera_confident", {"value": payload, "confidence": 0.99})
        self.assertEqual(result["value"], payload)

    def test_explore_preserves_structured_value(self):
        payload = {"idea": "draft", "score": 0.4}
        result = self._call("chimera_explore", {"value": payload, "confidence": 0.4})
        self.assertEqual(result["value"], payload)

    def test_constrain_preserves_structured_value(self):
        payload = {"answer": "Paris", "sources": ["a", "b"]}
        result = self._call("chimera_constrain", {"tool_name": "search", "output": payload})
        self.assertEqual(result["value"], payload)

    def test_cost_estimate_counts_multimodal_text_messages(self):
        result = self._call(
            "chimera_cost_estimate",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello world"}],
                    }
                ],
                "model": "claude-sonnet-4-6",
            },
        )
        self.assertGreater(result["input_tokens"], 0)

    def test_fracture_lossy_stops_after_first_budget_satisfying_drop(self):
        messages = [
            {"role": "user", "content": "Q " + "x" * 120},
            {"role": "assistant", "content": "A " + "y" * 120},
            {"role": "user", "content": "Followup " + "z" * 120},
            {"role": "assistant", "content": "Response " + "w" * 120},
        ]
        result = self._call(
            "chimera_fracture",
            {"messages": messages, "documents": [], "token_budget": 120, "allow_lossy": True},
        )
        self.assertTrue(result["quality_passed"])
        self.assertEqual(result["lossy_dropped_count"], 1)

    def test_mode_full_reports_real_tool_inventory(self):
        result = self._call("chimera_mode", {"mode": "full"})
        tool_names = [tool.name for tool in _run(self.list_tools())]
        self.assertEqual(result["tool_count_active"], len(tool_names))
        self.assertEqual(result["recommended_tools"], tool_names)


if __name__ == "__main__":
    unittest.main()
