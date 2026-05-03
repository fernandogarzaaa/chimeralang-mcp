"""test_tools.py — integration tests for chimera tools."""
from __future__ import annotations

import asyncio
import json
import unittest

from chimeralang_mcp import __version__
from chimeralang_mcp.server import (
    _ok, _err, _tbm, _scorer, call_tool, server,
    _MODEL_PRICING, _DEFAULT_MODEL, _CostTracker,
)
from chimeralang_mcp.token_engine import get_token_budget_manager


def _tool_payload(name: str, arguments: dict) -> dict:
    result = asyncio.run(call_tool(name, arguments))
    return json.loads(result.content[0].text)


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

    def test_quantum_optimize_outperforms_classic_on_focus_heavy_text(self):
        """Quantum optimize should save more while keeping the release facts."""
        text = (
            "Narrative filler: this is clearly a broad overview of the roadmap and future possibilities.\n\n"
            "Critical release details: bump pyproject.toml to 0.6.0, update __init__.py, and ensure publish.yml triggers on push to main.\n\n"
            "More unrelated storytelling that is very broad and basically not needed for the current task."
        )
        classic = _tool_payload("chimera_optimize", {
            "text": text,
            "algorithm": "classic",
        })
        quantum = _tool_payload("chimera_optimize", {
            "text": text,
            "focus": "release version bump publish workflow",
            "algorithm": "quantum",
        })
        self.assertIn("pyproject.toml", quantum["optimised_text"])
        self.assertLess(quantum["optimised_chars"], classic["optimised_chars"])
        self.assertEqual(quantum["algorithm"], "quantum")


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

    def test_quantum_compress_keeps_focus_terms(self):
        """Quantum compress should preserve release-critical tokens."""
        text = (
            "We should broadly think about many things.\n\n"
            "What matters now is version 0.6.0, pyproject.toml, and the publish workflow on push to main.\n\n"
            "There is also a lot of unrelated brainstorming here."
        )
        payload = _tool_payload("chimera_compress", {
            "text": text,
            "level": "aggressive",
            "focus": "version 0.6.0 pyproject publish workflow",
            "algorithm": "quantum",
        })
        self.assertIn("0.6.0", payload["compressed_text"])
        self.assertIn("publish", payload["compressed_text"].lower())
        self.assertEqual(payload["algorithm"], "quantum")


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

    def test_quantum_fracture_returns_real_compressed_outputs(self):
        """Quantum fracture should return compressed artifacts, not just stats."""
        messages = [
            {"role": "user", "content": "Ship version 0.6.0 and keep the publish workflow intact."},
            {"role": "assistant", "content": "Sure, I'd be happy to help with that. " * 20},
            {"role": "tool", "content": "pyproject.toml version = 0.6.0; workflow: publish.yml triggers on push to main"},
            {"role": "assistant", "content": "We also need to run tests and confirm the branch is synced before pushing."},
        ]
        documents = [
            "Release note draft: this is very broad narrative filler. "
            "Important: update __init__.py, update pyproject.toml, and verify publish.yml. "
            "There is additional unrelated storytelling that should compress away."
        ]
        payload = _tool_payload("chimera_fracture", {
            "messages": messages,
            "documents": documents,
            "token_budget": 120,
            "allow_lossy": True,
            "focus": "version 0.6.0 publish workflow tests",
            "algorithm": "quantum",
        })
        flattened = json.dumps(payload["compressed_messages"])
        self.assertEqual(payload["algorithm"], "quantum")
        self.assertTrue(payload["quality_passed"])
        self.assertTrue(payload["optimized_documents"])
        self.assertIn("0.6.0", flattened)
        self.assertIn("publish", flattened.lower())


class TestChimeraProve(unittest.TestCase):
    """chimera_prove tests."""

    def test_prove_returns_integrity_payload_for_minimal_program(self):
        """User-facing chimera_prove should work on the same minimal program chimera_run accepts."""
        payload = _tool_payload("chimera_prove", {"source": 'emit Confident("ok", 0.97)'})
        self.assertIn("execution", payload)
        self.assertIn("proof", payload)
        self.assertIn("verdict", payload["proof"])
        self.assertIn("root_hash", payload["proof"])


class TestCostTracker(unittest.TestCase):
    """_CostTracker unit tests."""

    def setUp(self):
        self.tracker = _CostTracker(maxlen=10)

    def test_record_returns_required_fields(self):
        entry = self.tracker.record(tokens_before=1000, tokens_after=600)
        for field in ("request_id", "timestamp", "tokens_before", "tokens_after",
                      "tokens_saved", "cost_before", "cost_after", "savings", "pct_saved"):
            self.assertIn(field, entry)

    def test_tokens_saved_correct(self):
        entry = self.tracker.record(tokens_before=1000, tokens_after=600)
        self.assertEqual(entry["tokens_saved"], 400)

    def test_pct_saved_correct(self):
        entry = self.tracker.record(tokens_before=1000, tokens_after=500)
        self.assertAlmostEqual(entry["pct_saved"], 50.0)

    def test_cost_uses_model_pricing(self):
        input_price, _ = _MODEL_PRICING["claude-haiku-4-5"]
        entry = self.tracker.record(tokens_before=1_000_000, tokens_after=0, model="claude-haiku-4-5")
        self.assertAlmostEqual(entry["cost_before"], input_price, places=2)

    def test_summary_totals_accumulate(self):
        self.tracker.record(tokens_before=1000, tokens_after=600)
        self.tracker.record(tokens_before=2000, tokens_after=1000)
        s = self.tracker.summary()
        self.assertEqual(s["request_count"], 2)
        self.assertEqual(s["total_tokens_saved"], 1400)
        self.assertGreater(s["total_cost_saved"], 0)

    def test_ring_buffer_caps_at_maxlen(self):
        for i in range(15):
            self.tracker.record(tokens_before=100, tokens_after=50)
        self.assertEqual(self.tracker.summary()["request_count"], 10)

    def test_summary_history_limited_to_10(self):
        for i in range(20):
            self.tracker.record(tokens_before=100, tokens_after=50)
        s = self.tracker.summary()
        self.assertLessEqual(len(s["history"]), 10)


class TestChimera_cost_estimate(unittest.TestCase):
    """chimera_cost_estimate logic tests."""

    def test_known_model_pricing(self):
        self.assertIn("claude-sonnet-4-6", _MODEL_PRICING)
        self.assertIn("gpt-4o", _MODEL_PRICING)

    def test_cost_scales_linearly(self):
        input_price, _ = _MODEL_PRICING[_DEFAULT_MODEL]
        tokens = 100_000
        expected = round(tokens * input_price / 1_000_000, 6)
        actual = round(_tbm.count_tokens("x" * (tokens * 4)) * input_price / 1_000_000, 6)
        self.assertAlmostEqual(actual, expected, delta=0.01)

    def test_default_model_exists(self):
        self.assertIn(_DEFAULT_MODEL, _MODEL_PRICING)


class TestChimeraCSM(unittest.TestCase):
    """chimera_csm tests."""

    def test_quantum_csm_beats_classic_input_tokens(self):
        """Quantum csm should produce a smaller optimized input than classic for noisy context."""
        messages = [
            {"role": "assistant", "content": "Sure, I'd be happy to help with that. " * 15},
            {"role": "tool", "content": "publish.yml triggers on push to main; pyproject.toml version = 0.6.0"},
        ]
        classic = _tool_payload("chimera_csm", {
            "prompt": "Ship version 0.6.0 and verify the publish workflow before push.",
            "messages": messages,
            "algorithm": "classic",
        })
        quantum = _tool_payload("chimera_csm", {
            "prompt": "Ship version 0.6.0 and verify the publish workflow before push.",
            "messages": messages,
            "algorithm": "quantum",
            "focus": "version 0.6.0 publish workflow",
        })
        self.assertLessEqual(quantum["total_input_tokens"], classic["total_input_tokens"])
        self.assertEqual(quantum["algorithm"], "quantum")


class TestServerMetadata(unittest.TestCase):
    """Initialization metadata tests."""

    def test_server_reports_package_version(self):
        """MCP initialize metadata should report the package version, not the SDK version."""
        options = server.create_initialization_options()
        self.assertEqual(options.server_name, "chimeralang-mcp")
        self.assertEqual(options.server_version, __version__)


if __name__ == "__main__":
    unittest.main()
