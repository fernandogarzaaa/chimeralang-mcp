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
        # 0.7.3+: "want/wants" encodes to itself (BPE-aligned). Was "wnt".
        self.assertIn("wants", glyph)

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


class TestRegressions071(unittest.TestCase):
    """Regression coverage for the five issues fixed in 0.7.1."""

    # Issue 1 — only "." terminates; "?" / "!" are modality markers.
    def test_question_glyph_does_not_split_sentence(self):
        english, _ = cg.decode("w ? rt-retry x.")
        self.assertEqual(english.count("."), 1, f"unexpected sentence break: {english!r}")
        self.assertIn("might", english.lower())

    def test_bang_glyph_renders_as_must(self):
        english, _ = cg.decode("u ! chk dt.")
        self.assertIn("must", english.lower())
        self.assertEqual(english.count("."), 1)

    # Issue 2 — canonical hyphenated stems decode as a single term.
    def test_hyphenated_stem_uses_full_lookup(self):
        english, _ = cg.decode("w rt-retry.")
        self.assertIn("retry", english.lower())
        self.assertNotIn("return retry", english.lower())

    # Issue 3 — standalone "~" decodes to "will" instead of being dropped.
    def test_standalone_tilde_is_will(self):
        english, _ = cg.decode("aprch ~ wrk.")
        self.assertIn("will", english.lower())
        self.assertIn("work", english.lower())

    # 0.7.3+: the suffix scheme was retired (BPE-multi-token). Encoder now
    # emits the English past/progressive form directly; decoder still
    # accepts legacy suffix forms via LEGACY_GLYPH_REVERSE.
    def test_progressive_verb_encodes_as_english(self):
        cg_text = cg.encode("I am going.")
        # "am" drops (auxiliary), "going" stays as "going" — both 1 BPE token.
        self.assertIn("going", cg_text)
        self.assertNotIn("gø~", cg_text)
        english, notes = cg.decode(cg_text)
        self.assertIn("going", english.lower())
        self.assertEqual(notes, [])

    def test_legacy_suffix_form_still_decodes(self):
        # Old CG text in the wild (e.g. logged from 0.7.2) must still decode.
        # The decoder splits "gø~" into the legacy "gø" stem (-> "go") and
        # the standalone "~" modal (-> "will"), reproducing the 0.7.2
        # semantics of "will go" rather than the 0.7.3 form "going".
        english, notes = cg.decode("i gø~.")
        self.assertEqual(notes, [])
        self.assertIn("go", english.lower())
        self.assertIn("will", english.lower())

    # Issue 4 — directive examples must round-trip without unrecognized tokens.
    def test_directive_examples_round_trip(self):
        d = cg.directive("strict")
        for line in d.splitlines():
            line = line.strip()
            if not line or not line.startswith(("usr ", "if ", "i ", "mdl ")):
                continue
            _, notes = cg.decode(line)
            self.assertEqual(notes, [], f"example failed to decode cleanly: {line!r} → {notes}")

    # Issue 5 — verbosity is validated; terse strips capitalized "The".
    def test_translate_invalid_verbosity_falls_back_to_natural(self):
        payload = _tool_payload(
            "chimera_glyph_translate",
            {"glyph_text": "usr nd rs.", "verbosity": "loud"},
        )
        self.assertEqual(payload["verbosity"], "natural")

    def test_translate_terse_strips_leading_capitalized_the(self):
        payload = _tool_payload(
            "chimera_glyph_translate",
            {"glyph_text": "usr nd rs.", "verbosity": "terse"},
        )
        self.assertNotIn("The ", payload["english"])
        self.assertNotIn(" the ", payload["english"])


class TestTokenizerAwareLexicon(unittest.TestCase):
    """0.7.3 — empirical lexicon optimization (Phase 1 of BLUEPRINT.md).

    The hand-crafted Glyph stems were measured against tiktoken o200k_base
    (Claude-equivalent BPE) and 74 entries were found to cost MORE tokens
    than the English source. Those entries now encode as English; the
    decoder retains LEGACY_GLYPH_REVERSE for backwards compatibility.
    """

    # ── encoder side: BPE-friendly output ────────────────────────────
    def test_multitoken_stems_revert_to_english(self):
        # These stems were >=2 BPE tokens; encoder must emit the English form.
        cases = {
            "code":     "code",      # was "cde"
            "approach": "approach",  # was "aprch"
            "person":   "person",    # was "psn"
            "build":    "build",     # was "bld"
            "find":     "find",      # was "fnd"
            "think":    "think",     # was "thk"
            "work":     "work",      # was "wrk"
            "going":    "going",     # was "gø~"
            "fixed":    "fixed",     # was "fix^"
            "wanted":   "wanted",    # was "wnt^"
        }
        for english, expected in cases.items():
            glyph = cg.encode(f"the {english}")
            self.assertIn(expected, glyph.split(),
                          f"{english!r} should encode to {expected!r}, got {glyph!r}")

    def test_unicode_operators_demoted_when_multitoken(self):
        # ∅, ∀, ∃, ∧, ∨, ¬, ≈, ≡ all cost 2 BPE tokens; encoder uses English.
        self.assertIn("none", cg.encode("the result is null").split())
        self.assertIn("all", cg.encode("all users").split())
        self.assertIn("some", cg.encode("some answer").split())
        self.assertIn("and", cg.encode("user and system").split())
        self.assertIn("not", cg.encode("not the same").split())
        self.assertIn("about", cg.encode("about ten").split())

    def test_unicode_operators_kept_when_single_token(self):
        # ⇒, ←, ≠, → are 1 BPE token in o200k_base — keep them.
        self.assertIn("⇒", cg.encode("if x then y").split())
        self.assertIn("←", cg.encode("because x").split())
        self.assertIn("≠", cg.encode("a different thing").split())

    # ── decoder side: legacy stems still decode ──────────────────────
    def test_all_retired_stems_decode(self):
        # Sample from LEGACY_GLYPH_REVERSE — every key must round-trip.
        legacy_samples = [
            ("usr cde.",       "code"),
            ("usr aprch.",     "approach"),
            ("usr psn.",       "person"),
            ("usr fix^.",      "fixed"),
            ("usr wnt.",       "want"),
            ("usr wnt^.",      "wanted"),
            ("usr ∅.",         "empty"),
            ("usr ∀.",         "all"),
            ("usr ∃.",         "some"),
            ("usr ∧ sys.",     "and"),
            ("usr ¬ wrk.",     "not"),
            ("usr wrk^.",      "worked"),
            ("usr rt-retry.",  "retry"),
        ]
        for cg_text, must_contain in legacy_samples:
            english, notes = cg.decode(cg_text)
            self.assertIn(must_contain, english.lower(),
                          f"legacy {cg_text!r} did not decode to contain {must_contain!r}: {english!r}")
            self.assertEqual(notes, [], f"unexpected notes on {cg_text!r}: {notes}")

    def test_corpus_token_savings_are_positive(self):
        # Phase 1 acceptance gate: on a representative corpus the new
        # lexicon must produce strictly fewer characters AND no negative
        # token outcomes per sentence. This is a sanity floor — the formal
        # benchmark in tools/glyph_benchmark.py owns the headline number.
        corpus = [
            "The user wants to know how to fix the error in the function.",
            "The model returned a null result.",
            "We need to check the data and run the tests.",
            "Maybe this approach will work.",
            "The user reported that the build is failing.",
        ]
        for sentence in corpus:
            glyph = cg.encode(sentence)
            self.assertLess(len(glyph), len(sentence),
                            f"glyph not shorter than english: {sentence!r} -> {glyph!r}")


if __name__ == "__main__":
    unittest.main()
