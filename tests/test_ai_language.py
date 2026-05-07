"""test_ai_language.py — tests for the Chimera Glyph language and its tools."""
from __future__ import annotations

import asyncio
import json
import unittest

from chimeralang_mcp import ai_language as cg
from chimeralang_mcp.server import call_tool


def _tool_payload(name: str, arguments: dict) -> dict:
    result = asyncio.run(call_tool(name, arguments))
    return json.loads(result.content[0].text)


class TestEncode(unittest.TestCase):
    def test_drops_articles(self):
        glyph = cg.encode("The user wants the result.")
        self.assertNotIn("the", glyph.lower().split())
        self.assertIn("usr", glyph)
        self.assertIn("wnt", glyph)

    def test_compresses_known_sentences(self):
        cases = [
            "The user wants to know how to fix the error in the function.",
            "If the test fails, then we should retry it.",
            "I think this approach will work.",
            "The model returned a null result.",
            "We need to check the data and run the tests.",
        ]
        for s in cases:
            glyph = cg.encode(s)
            self.assertTrue(len(glyph) < len(s),
                            f"glyph not shorter than english: {s!r} -> {glyph!r}")

    def test_savings_estimator_reports_reduction(self):
        s = "The user wants to know how to fix the error in the function."
        info = cg.estimate_savings(s)
        self.assertGreater(info["tokens_saved"], 0)
        self.assertGreater(info["savings_ratio"], 0.0)
        self.assertEqual(info["english_tokens"], max(0, len(s) // 4))

    def test_unknown_word_becomes_entity(self):
        glyph = cg.encode("The Foobarbaz library is great.")
        self.assertIn("@Foobarbaz", glyph)


class TestDecode(unittest.TestCase):
    def test_recovers_key_content_words(self):
        english, notes = cg.decode("usr wnt kn fix err in fn.")
        low = english.lower()
        for needle in ("user", "want", "know", "fix", "error", "function"):
            self.assertIn(needle, low, f"missing {needle!r} in {english!r}")
        self.assertEqual(notes, [])

    def test_pronoun_does_not_get_the_prefix(self):
        english, _ = cg.decode("i thk x wrk~.")
        self.assertNotIn("the I", english)
        self.assertIn("I", english)

    def test_unknown_glyph_logged_in_notes(self):
        english, notes = cg.decode("zzz123nope usr.")
        self.assertTrue(any("unrecognized" in n for n in notes))
        self.assertIn("user", english.lower())

    def test_entity_passthrough(self):
        english, _ = cg.decode("usr cl @MyService.")
        self.assertIn("MyService", english)


class TestDirective(unittest.TestCase):
    def test_directive_contains_grammar_markers(self):
        d = cg.directive("strict")
        self.assertIn("→", d)
        self.assertIn("@", d)
        self.assertIn("LEXICON", d)
        self.assertIn("STRICT MODE", d)

    def test_balanced_mode_has_softer_phrasing(self):
        d = cg.directive("balanced")
        self.assertIn("BALANCED MODE", d)
        self.assertNotIn("STRICT MODE", d)

    def test_task_hint_is_appended(self):
        d = cg.directive("strict", task_hint="debug a failing pytest run")
        self.assertIn("debug a failing pytest run", d)


class TestGlyphDirectiveTool(unittest.TestCase):
    def test_default_strict(self):
        payload = _tool_payload("chimera_glyph_directive", {})
        self.assertIn("directive", payload)
        self.assertEqual(payload["style"], "strict")
        self.assertIn("STRICT MODE", payload["directive"])
        self.assertEqual(payload["translator_tool"], "chimera_glyph_translate")
        self.assertGreaterEqual(len(payload["examples"]), 3)

    def test_balanced_with_hint(self):
        payload = _tool_payload(
            "chimera_glyph_directive",
            {"style": "balanced", "task_hint": "code review"},
        )
        self.assertEqual(payload["style"], "balanced")
        self.assertIn("code review", payload["directive"])

    def test_invalid_style_falls_back_to_strict(self):
        payload = _tool_payload("chimera_glyph_directive", {"style": "wild"})
        self.assertEqual(payload["style"], "strict")


class TestGlyphTranslateTool(unittest.TestCase):
    def test_translates_known_glyph(self):
        payload = _tool_payload(
            "chimera_glyph_translate",
            {"glyph_text": "usr wnt kn fix err in fn."},
        )
        self.assertTrue(payload["lossy"])
        self.assertIn("user", payload["english"].lower())
        self.assertIn("fix", payload["english"].lower())

    def test_terse_strips_articles_and_copulas(self):
        payload = _tool_payload(
            "chimera_glyph_translate",
            {"glyph_text": "usr wnt rs.", "verbosity": "terse"},
        )
        self.assertNotIn(" the ", " " + payload["english"] + " ")

    def test_unrecognized_logged(self):
        payload = _tool_payload(
            "chimera_glyph_translate",
            {"glyph_text": "zqzq usr."},
        )
        self.assertTrue(any("unrecognized" in n for n in payload["notes"]))


if __name__ == "__main__":
    unittest.main()
