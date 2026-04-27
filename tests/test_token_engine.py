"""test_token_engine.py — unit tests for TokenBudgetManager + MessageImportanceScorer."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestTokenBudgetManager(unittest.TestCase):
    """TokenBudgetManager tests."""

    def test_count_tokens_fallback_no_api_key(self):
        """When ANTHROPIC_API_KEY is absent, falls back to len//4."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            # Force fresh instance
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            self.assertEqual(tbm._token_count_method, "estimate")
            self.assertEqual(tbm.count_tokens("hello world"), 2)  # 11 chars // 4 = 2

    def test_count_tokens_empty_string(self):
        """Empty string returns 0."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            self.assertEqual(tbm.count_tokens(""), 0)

    def test_count_messages_empty_list(self):
        """Empty messages list returns 0."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            self.assertEqual(tbm.count_messages([]), 0)

    def test_cache_avoids_double_counting(self):
        """Identical text is counted once and cached."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            text = "some repeated text"
            r1 = tbm.count_tokens(text)
            r2 = tbm.count_tokens(text)
            self.assertEqual(r1, r2)
            self.assertEqual(tbm._cache_hits, 1)
            self.assertEqual(tbm._cache_misses, 1)

    def test_get_stats(self):
        """get_stats returns cache and method info."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            tbm.count_tokens("hello")
            stats = tbm.get_stats()
            self.assertIn("token_count_method", stats)
            self.assertIn("cache_size", stats)
            self.assertEqual(stats["cache_size"], 1)
            self.assertIn("cache_max_entries", stats)
            self.assertIn("fallback_count", stats)

    def test_cache_has_bounded_size(self):
        """Cache evicts oldest values when max size is reached."""
        with patch.dict("os.environ", {"CHIMERA_TOKEN_CACHE_MAX_ENTRIES": "2"}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            tbm.count_tokens("one")
            tbm.count_tokens("two")
            tbm.count_tokens("three")
            self.assertEqual(len(tbm._cache), 2)

    def test_api_error_increments_fallback_count(self):
        """API errors should increment fallback counters and preserve estimates."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake"}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            tbm._initialized = False
            tbm.__init__()
            tbm._client = MagicMock()
            tbm._token_count_method = "api"
            tbm._client.beta.messages.count_tokens.side_effect = RuntimeError("boom")

            result = tbm.count_tokens("abcdef")

            self.assertEqual(result, 1)  # len("abcdef") // 4
            self.assertEqual(tbm._fallback_count, 1)
            self.assertEqual(tbm._last_fallback_reason, "count_tokens_api_error")


class TestMessageImportanceScorer(unittest.TestCase):
    """MessageImportanceScorer tests."""

    def test_tool_result_scores_highest(self):
        """tool_result messages score high on type axis (tested at recent position)."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        tool_msg = {"role": "tool", "content": "Tool use: file written"}
        score = scorer.score(tool_msg, 4, 5)
        self.assertGreaterEqual(score, 0.75)

    def test_user_question_scores_high(self):
        """User questions score high on type axis (tested at recent position)."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        user_msg = {"role": "user", "content": "How do I fix the auth bug?"}
        score = scorer.score(user_msg, 4, 5)
        self.assertGreaterEqual(score, 0.7)

    def test_old_message_scores_lower(self):
        """Older messages (lower index) score lower on recency."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        msg = {"role": "assistant", "content": "Some reply"}
        # First message (oldest) should score lower than last
        old_score = scorer.score(msg, 0, 5)
        recent_score = scorer.score(msg, 4, 5)
        self.assertLess(old_score, recent_score)

    def test_rank_orders_by_score(self):
        """rank() returns messages sorted ascending by score."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        messages = [
            {"role": "assistant", "content": "Sure, I'd be happy to help with that!"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "tool", "content": "Tool use: read file auth.py"},
            {"role": "assistant", "content": "def authenticate():\n    pass"},
        ]
        ranked = scorer.rank(messages)
        scores = [r["score"] for r in ranked]
        self.assertEqual(scores, sorted(scores))

    def test_rank_on_empty_list(self):
        """rank() on empty list returns empty list, not error."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        ranked = scorer.rank([])
        self.assertEqual(ranked, [])

    def test_code_fence_is_preserved(self):
        """Code fences are scored high on type axis (tested at recent position)."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        code_msg = {"role": "assistant", "content": "```python\ndef foo():\n    pass\n```"}
        score = scorer.score(code_msg, 4, 5)
        self.assertGreaterEqual(score, 0.75)

    def test_verbose_prose_scores_low(self):
        """Long verbose assistant prose scores low on type axis."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer
        scorer = MessageImportanceScorer()
        prose = {"role": "assistant", "content": "Sure, I'd be happy to help you with that! As you can see, this is a very long verbose response that goes on and on about something that could be said much more concisely. Let me walk you through each step in detail." * 3}
        score = scorer.score(prose, 0, 5)
        self.assertLess(score, 0.5)


if __name__ == "__main__":
    unittest.main()
