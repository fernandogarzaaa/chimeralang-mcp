"""test_batch_summarize.py — tests for chimera_batch and chimera_summarize."""
from __future__ import annotations

import asyncio
import json
import unittest


def _run(coro):
    return asyncio.run(coro)


class TestChimeraBatch(unittest.TestCase):
    def setUp(self):
        from chimeralang_mcp.server import call_tool
        self.call = call_tool

    def _batch(self, calls, stop_on_error=False):
        return _run(self.call("chimera_batch", {"calls": calls, "stop_on_error": stop_on_error}))

    def _parse(self, result):
        return json.loads(result.content[0].text)

    def test_single_call(self):
        r = self._parse(self._batch([{"tool": "chimera_explore", "args": {"value": "hello"}}]))
        self.assertEqual(r["total"], 1)
        self.assertEqual(r["succeeded"], 1)
        self.assertEqual(r["failed"], 0)

    def test_multiple_calls_in_order(self):
        calls = [
            {"tool": "chimera_explore", "args": {"value": "a"}},
            {"tool": "chimera_explore", "args": {"value": "b"}},
            {"tool": "chimera_audit", "args": {}},
        ]
        r = self._parse(self._batch(calls))
        self.assertEqual(r["total"], 3)
        self.assertEqual(r["executed"], 3)
        self.assertEqual(r["results"][0]["tool"], "chimera_explore")
        self.assertEqual(r["results"][1]["tool"], "chimera_explore")
        self.assertEqual(r["results"][2]["tool"], "chimera_audit")

    def test_unknown_tool_counted_as_failure(self):
        r = self._parse(self._batch([{"tool": "chimera_nonexistent", "args": {}}]))
        self.assertEqual(r["failed"], 1)
        self.assertEqual(r["succeeded"], 0)

    def test_stop_on_error_halts_execution(self):
        calls = [
            {"tool": "chimera_nonexistent", "args": {}},
            {"tool": "chimera_explore", "args": {"value": "should not run"}},
        ]
        r = self._parse(self._batch(calls, stop_on_error=True))
        self.assertEqual(r["executed"], 1)

    def test_continue_on_error_by_default(self):
        calls = [
            {"tool": "chimera_nonexistent", "args": {}},
            {"tool": "chimera_explore", "args": {"value": "should run"}},
        ]
        r = self._parse(self._batch(calls, stop_on_error=False))
        self.assertEqual(r["executed"], 2)
        self.assertEqual(r["succeeded"], 1)

    def test_result_indices_correct(self):
        calls = [{"tool": "chimera_explore", "args": {"value": str(i)}} for i in range(4)]
        r = self._parse(self._batch(calls))
        for i, res in enumerate(r["results"]):
            self.assertEqual(res["index"], i)


class TestChimeraSummarize(unittest.TestCase):
    def setUp(self):
        from chimeralang_mcp.server import call_tool
        self.call = call_tool

    def _summarize(self, text, ratio=0.25, min_sentences=2):
        return json.loads(
            _run(self.call("chimera_summarize", {
                "text": text, "ratio": ratio, "min_sentences": min_sentences,
            })).content[0].text
        )

    LONG_TEXT = (
        "The sun rose over the mountains, casting golden light across the valley. "
        "Birds began to sing their morning songs as the mist slowly lifted. "
        "Farmers were already in their fields, preparing for the harvest. "
        "Children ran to school along the winding paths. "
        "The market opened with the smell of fresh bread and coffee. "
        "Merchants arranged their goods under colorful awnings. "
        "By midday the town was buzzing with activity and conversation. "
        "The afternoon brought clouds and a welcome breeze from the west. "
        "As evening fell, families gathered around their tables for dinner. "
        "The stars appeared one by one as night settled over the town."
    )

    def test_returns_fewer_sentences(self):
        r = self._summarize(self.LONG_TEXT, ratio=0.3, min_sentences=2)
        self.assertLess(r["sentences_out"], r["sentences_in"])

    def test_summary_is_string(self):
        r = self._summarize(self.LONG_TEXT)
        self.assertIsInstance(r["summary"], str)
        self.assertGreater(len(r["summary"]), 0)

    def test_token_savings_positive(self):
        r = self._summarize(self.LONG_TEXT, ratio=0.3)
        self.assertGreaterEqual(r["savings_pct"], 0.0)

    def test_min_sentences_respected(self):
        r = self._summarize(self.LONG_TEXT, ratio=0.01, min_sentences=3)
        self.assertGreaterEqual(r["sentences_out"], 3)

    def test_empty_text(self):
        r = self._summarize("", ratio=0.5)
        self.assertEqual(r["sentences_in"], 0)

    def test_single_sentence_passthrough(self):
        r = self._summarize("Only one sentence here.", ratio=0.25, min_sentences=3)
        self.assertEqual(r["sentences_in"], r["sentences_out"])

    def test_ratio_achieved_in_range(self):
        r = self._summarize(self.LONG_TEXT, ratio=0.3)
        self.assertGreater(r["ratio_achieved"], 0.0)
        self.assertLessEqual(r["ratio_achieved"], 1.0)

    def test_tokens_before_gte_tokens_after(self):
        r = self._summarize(self.LONG_TEXT, ratio=0.3)
        self.assertGreaterEqual(r["tokens_before"], r["tokens_after"])


if __name__ == "__main__":
    unittest.main()
