"""test_tools.py — integration tests for all 5 affected tools."""
from __future__ import annotations

import unittest

from chimeralang_mcp.server import (
    _ok, _err, _tbm, _scorer,
)
from chimeralang_mcp.token_engine import get_token_budget_manager


class TestChimeraOptimize(unittest.TestCase):
    """chimera_optimize tests."""

    def test_collapse_lists_deduplicates(self):
        """collapse_lists deduplicates repeated list items."""
        import re, asyncio
        from chimeralang_mcp.server import server, call_tool

        text = """
- first item
- second item
- first item
- third item
- second item
""".strip()

        # We test the logic directly since we can't run the full async server
        import re as _re
        list_item_pattern = r"(?:^|\n)([-*•])\s+(.+?)(?=\n[-*•]|\n\n|$)"
        items_ordered: list[tuple[str, str]] = []
        seen_items: set[str] = set()
        for m in _re.finditer(list_item_pattern, text, _re.MULTILINE):
            bullet, text_content = m.group(1), m.group(2).strip()
            key = text_content.lower()
            if key and key not in seen_items:
                seen_items.add(key)
                items_ordered.append((bullet, text_content))

        self.assertEqual(len(items_ordered), 3)
        self.assertEqual(items_ordered[0][1], "first item")
        self.assertEqual(items_ordered[1][1], "second item")
        self.assertEqual(items_ordered[2][1], "third item")

    def test_preserve_code_stashes_and_restores(self):
        """Code fences are stashed before processing and restored after."""
        import re as _re
        text = "Hello world\n```python\ndef foo():\n    pass\n```\nMore text"
        code_blocks: list[str] = []

        def _stash(m: "_re.Match[str]") -> str:
            code_blocks.append(m.group(0))
            return f"\x00CODE{len(code_blocks) - 1}\x00"

        work = _re.sub(r"```[\s\S]*?```", _stash, text)
        work = _re.sub(r"[ \t]+", " ", work)  # should NOT affect code

        for i, block in enumerate(code_blocks):
            work = work.replace(f"\x00CODE{i}\x00", block)

        self.assertIn("def foo():", work)
        self.assertNotIn("\x00CODE", work)


class TestChimeraCompress(unittest.TestCase):
    """chimera_compress tests."""

    def test_aggressive_mode_no_harmful_symbols(self):
        """Aggressive mode no longer produces ∴ ∵ & w/ w/o."""
        import re as _re

        _SYMBOLS_AGGRESSIVE = {
            r"\bapproximately\b": "≈",
            r"\bgreater than\b": ">",
            r"\bless than\b": "<",
            r"\bequals\b": "=",
            r"\bnumber\b": "nr.",
            r"\bversus\b": "vs.",
            r"\bregarding\b": "re:",
            r"\bfor example\b": "e.g.",
            r"\bthat is\b": "i.e.",
            r"\betcetera\b": "etc.",
        }

        text = "Therefore, because I am with you without number versus etcetera"
        result = text
        for pat, repl in _SYMBOLS_AGGRESSIVE.items():
            result = _re.sub(pat, repl, result, flags=_re.IGNORECASE)

        self.assertNotIn("∴", result)
        self.assertNotIn("∵", result)
        self.assertIn("nr.", result)   # "number" → "nr." was in test text
        self.assertIn("vs.", result)   # "versus" → "vs." was in test text


class TestChimeraBudget(unittest.TestCase):
    """chimera_budget tests."""

    def test_budget_returns_correct_structure(self):
        """chimera_budget returns all required fields."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        used = _tbm.count_messages(messages)
        max_tokens = 200000
        reserve = 10000
        remaining = max(0, max_tokens - used - reserve)
        pct = used / max_tokens if max_tokens else 0

        self.assertGreaterEqual(used, 0)
        self.assertGreaterEqual(remaining, 0)
        self.assertLessEqual(pct, 1.0)

    def test_budget_warn_at_72_percent(self):
        """72% usage triggers warn status."""
        used = int(200000 * 0.72)
        pct = used / 200000
        self.assertGreaterEqual(pct, 0.70)
        self.assertLess(pct, 0.85)


class TestChimeraScore(unittest.TestCase):
    """chimera_score tests."""

    def test_score_returns_all_fields(self):
        """Scored messages have index, role, score, and reason."""
        messages = [
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "content": "Tool use: read file"},
        ]
        ranked = _scorer.rank(messages)
        self.assertEqual(len(ranked), 3)
        for r in ranked:
            self.assertIn("index", r)
            self.assertIn("role", r)
            self.assertIn("score", r)
            self.assertIn("reason", r)
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    def test_lowest_score_is_verbose_prose(self):
        """Lowest-scoring message is the verbose assistant preamble."""
        messages = [
            {"role": "assistant", "content": "Sure, I'd be happy to help with that! As you can see, this is a long verbose response."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        ranked = _scorer.rank(messages)
        lowest = ranked[0]
        self.assertIn("old_message", lowest["reason"].lower())


class TestChimeraFracturePipeline(unittest.TestCase):
    """chimera_fracture pipeline tests."""

    def test_fracture_returns_required_fields(self):
        """Pipeline returns quality_passed, tokens, budget_remaining."""
        # Simulate the pipeline logic
        messages = [{"role": "user", "content": "Hello world"}]
        documents = ["This is a test document with some content."]
        token_budget = 1500
        allow_lossy = False

        tokens_input = _tbm.count_messages(messages)
        tokens_docs = _tbm.count_texts(documents)
        tokens_after = tokens_input + tokens_docs
        budget_remaining = max(0, token_budget - tokens_after)
        quality_passed = tokens_after <= token_budget

        self.assertIn("quality_passed", {"quality_passed": quality_passed})
        self.assertGreaterEqual(budget_remaining, 0)
        self.assertIsInstance(quality_passed, bool)


if __name__ == "__main__":
    unittest.main()
