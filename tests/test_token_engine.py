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

    def test_singleton_reuses_same_instance(self):
        """TokenBudgetManager should reuse the same singleton instance."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            first = TokenBudgetManager()
            second = TokenBudgetManager()
            self.assertIs(first, second)

    def test_count_messages_normalizes_text_parts(self):
        """List-based multimodal content should count text parts instead of zero."""
        with patch.dict("os.environ", {}, clear=True):
            from chimeralang_mcp.token_engine import TokenBudgetManager
            TokenBudgetManager._instance = None
            tbm = TokenBudgetManager()
            messages = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]
            self.assertGreater(tbm.count_messages(messages), 0)


class TestQuantumCompressionEngine(unittest.TestCase):
    """Quantum compression engine tests."""

    def test_optimize_keeps_focus_and_drops_irrelevant_filler(self):
        """Focused optimization should keep relevant release facts and shrink fluff."""
        from chimeralang_mcp.token_engine import get_quantum_compression_engine

        engine = get_quantum_compression_engine()
        text = (
            "Marketing overview: this project is very exciting and clearly has many possibilities.\n\n"
            "Release checklist: bump pyproject.toml to 0.6.0, update __init__.py, and confirm the publish workflow runs on push to main.\n\n"
            "Additional narrative: as you can see, the branding story is basically unrelated to the release mechanics."
        )
        result = engine.optimize_text(
            text,
            focus="release version bump publish workflow",
            strategies=["whitespace", "dedup_sentences", "strip_filler"],
            level="aggressive",
        )
        self.assertIn("pyproject.toml", result.text)
        self.assertIn("publish workflow", result.text.lower())
        self.assertNotIn("branding story", result.text.lower())
        self.assertLess(result.compressed_tokens, result.original_tokens)

    def test_optimize_preserves_code_blocks(self):
        """Quantum optimization should preserve fenced code exactly."""
        from chimeralang_mcp.token_engine import get_quantum_compression_engine

        engine = get_quantum_compression_engine()
        text = (
            "Context before.\n\n"
            "```python\n"
            "def release_version():\n"
            "    return '0.6.0'\n"
            "```\n\n"
            "A very verbose explanation that should shrink aggressively."
        )
        result = engine.optimize_text(text, focus="release version", level="aggressive")
        self.assertIn("def release_version():", result.text)
        self.assertIn("0.6.0", result.text)

    def test_compress_messages_respects_budget_pressure(self):
        """Focused message compression should keep critical release facts under pressure."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer, get_quantum_compression_engine

        engine = get_quantum_compression_engine()
        scorer = MessageImportanceScorer()
        messages = [
            {"role": "user", "content": "Please ship version 0.6.0 and make sure PyPI publish still works."},
            {"role": "assistant", "content": "Sure, I'd be happy to help with that. " * 25},
            {"role": "tool", "content": "pyproject.toml version = 0.6.0; workflow: publish.yml triggers on push to main"},
            {"role": "assistant", "content": "We should also remember the key release steps and verify tests before pushing."},
        ]
        result = engine.compress_messages(
            messages,
            focus="version 0.6.0 pypi publish workflow",
            scorer=scorer,
            token_budget=55,
            allow_lossy=True,
        )
        flattened = "\n".join(m["content"] for m in result.messages)
        self.assertIn("0.6.0", flattened)
        self.assertIn("publish", flattened.lower())
        self.assertLess(result.compressed_tokens, result.original_tokens)


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

    def test_focus_overlap_scores_higher(self):
        """Messages matching the active task focus should outrank unrelated prose."""
        from chimeralang_mcp.token_engine import MessageImportanceScorer

        scorer = MessageImportanceScorer()
        focused = {"role": "assistant", "content": "Bump pyproject.toml to 0.6.0 and verify publish.yml."}
        unrelated = {"role": "assistant", "content": "We can also think broadly about future branding possibilities."}
        focus = "version bump publish workflow"
        self.assertGreater(
            scorer.score(focused, 3, 4, focus=focus),
            scorer.score(unrelated, 3, 4, focus=focus),
        )


if __name__ == "__main__":
    unittest.main()
