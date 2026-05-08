"""Phase 1 spike — measure how the current Chimera Glyph lexicon performs
against a Claude-equivalent BPE tokenizer.

We use tiktoken's o200k_base as a proxy for Anthropic's tokenizer (they
share BPE properties on English; the top-N candidates we pick will be
re-validated against the real Anthropic API in P1.S5).

Three things we want to know before committing to the full optimization:
  1. Per-stem: how many Glyph stems are 1 token vs 2+ tokens?
  2. Per-stem: when is the Glyph stem WORSE (more tokens) than the original
     English word? (a sign the lexicon was hand-tuned without tokenizer awareness)
  3. Corpus-level: on a representative corpus, what's the realised token
     reduction today, and what's the theoretical headroom if every multi-token
     glyph stem could be replaced with a 1-token alternative?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import tiktoken

# Make the project package importable when run from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from chimeralang_mcp.ai_language import LEXICON, encode  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")


def n_tokens(s: str) -> int:
    return len(ENC.encode(s))


# ── Per-stem analysis ────────────────────────────────────────────────────


def per_stem_report() -> dict:
    """For every (english -> glyph) pair, count tokens both ways."""
    rows = []
    for eng, glyph in LEXICON.items():
        # Skip auxiliaries that map to "" (decoder-only, not encoded)
        if not glyph:
            continue
        # Whitespace-prefixed forms reflect how words appear mid-sentence.
        # BPE often gives different counts for "user" vs " user".
        eng_tok = n_tokens(" " + eng)
        gly_tok = n_tokens(" " + glyph)
        rows.append({
            "english": eng,
            "glyph": glyph,
            "eng_tokens": eng_tok,
            "glyph_tokens": gly_tok,
            "delta": eng_tok - gly_tok,  # positive = glyph saves tokens
        })

    total = len(rows)
    one_tok_glyph = sum(1 for r in rows if r["glyph_tokens"] == 1)
    multi_tok_glyph = sum(1 for r in rows if r["glyph_tokens"] >= 2)
    glyph_worse = [r for r in rows if r["delta"] < 0]
    glyph_neutral = [r for r in rows if r["delta"] == 0]
    glyph_better = [r for r in rows if r["delta"] > 0]

    return {
        "total_pairs": total,
        "glyph_one_token": one_tok_glyph,
        "glyph_multi_token": multi_tok_glyph,
        "glyph_worse_than_english": len(glyph_worse),
        "glyph_neutral": len(glyph_neutral),
        "glyph_better_than_english": len(glyph_better),
        "worst_offenders": sorted(glyph_worse, key=lambda r: r["delta"])[:15],
        "biggest_wins": sorted(glyph_better, key=lambda r: -r["delta"])[:10],
        "all_rows": rows,
    }


# ── Corpus-level analysis ────────────────────────────────────────────────

# Representative corpus across 4 domains: code/error, dialogue, instruction, reasoning.
# This is the seed for the formal P1.S4 benchmark; for now it's enough to ground
# the spike numbers in something realistic.
CORPUS = [
    # error / debugging
    "The user wants to know how to fix the error in the function.",
    "If the test fails, then we should retry it.",
    "I think this approach will work.",
    "The model returned a null result.",
    "We need to check the data and run the tests.",
    "The user reported that the build is failing on the new branch.",
    "If the file does not exist, the function returns an empty string.",
    "We should add a regression test before deleting the old code.",
    # dialogue
    "Can you tell me what the function does?",
    "I am not sure if this is the right approach.",
    "Maybe the bug is in the parser.",
    "Yes, the model passes the tests now.",
    # instruction
    "Read the file, then write the result to a new file.",
    "Compress the message history before passing it to the model.",
    "Mark the system prompt as cacheable.",
    "Track the tokens saved after every compression step.",
    # reasoning
    "If all the tests pass, then we can ship the change to production.",
    "The user is asking a question that requires reasoning across multiple files.",
    "Because the cache was empty, the model had to recompute the answer.",
    "The error is not in the parser; it is in the type checker.",
]


def corpus_report() -> dict:
    """Encode each sentence with the current LEXICON; measure tokens before/after."""
    per_sentence = []
    total_eng_tok = 0
    total_gly_tok = 0
    total_eng_chars = 0
    total_gly_chars = 0

    for sent in CORPUS:
        gly = encode(sent)
        eng_tok = n_tokens(sent)
        gly_tok = n_tokens(gly)
        per_sentence.append({
            "english": sent,
            "glyph": gly,
            "eng_tokens": eng_tok,
            "glyph_tokens": gly_tok,
            "pct_saved": round(100 * (eng_tok - gly_tok) / max(1, eng_tok), 1),
        })
        total_eng_tok += eng_tok
        total_gly_tok += gly_tok
        total_eng_chars += len(sent)
        total_gly_chars += len(gly)

    return {
        "n_sentences": len(CORPUS),
        "total_english_tokens": total_eng_tok,
        "total_glyph_tokens": total_gly_tok,
        "tokens_saved": total_eng_tok - total_gly_tok,
        "pct_saved_tokens": round(100 * (total_eng_tok - total_gly_tok) / max(1, total_eng_tok), 1),
        "pct_saved_chars": round(100 * (total_eng_chars - total_gly_chars) / max(1, total_eng_chars), 1),
        "per_sentence": per_sentence,
    }


# ── Headroom estimate ────────────────────────────────────────────────────


def headroom_report(stem_report: dict) -> dict:
    """If every multi-token glyph stem could be swapped to a 1-token alternative,
    what's the additional tokens-saved on the corpus?"""
    # Build a per-stem extra-token cost table: glyph_tokens - 1.
    # Then re-tokenize the corpus, scaling the saving by occurrences.
    extra_cost = {
        r["glyph"]: r["glyph_tokens"] - 1
        for r in stem_report["all_rows"]
        if r["glyph_tokens"] >= 2
    }
    # Count occurrences of each multi-token stem in the encoded corpus.
    occurrences: dict[str, int] = {}
    for sent in CORPUS:
        gly = encode(sent)
        for tok in gly.split():
            # Strip trailing punctuation for the lookup
            clean = tok.rstrip(".,!?")
            if clean in extra_cost:
                occurrences[clean] = occurrences.get(clean, 0) + 1

    additional_tokens_savable = sum(
        extra_cost[stem] * count for stem, count in occurrences.items()
    )
    return {
        "multi_token_stems_used_in_corpus": len(occurrences),
        "occurrences": dict(sorted(occurrences.items(), key=lambda kv: -kv[1])[:15]),
        "additional_tokens_savable_if_optimized": additional_tokens_savable,
    }


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    stems = per_stem_report()
    corpus = corpus_report()
    headroom = headroom_report(stems)

    additional_pct = round(
        100 * headroom["additional_tokens_savable_if_optimized"]
        / max(1, corpus["total_english_tokens"]),
        1,
    )

    print("=" * 72)
    print("CHIMERA GLYPH — PHASE 1 SPIKE (tiktoken o200k_base, proxy for Claude)")
    print("=" * 72)
    print()
    print("--- Per-stem analysis ---")
    print(f"  Total stems analysed (excluding null/decoder-only): {stems['total_pairs']}")
    print(f"  Glyph stems that are 1 token:    {stems['glyph_one_token']:>3} "
          f"({100*stems['glyph_one_token']/stems['total_pairs']:.0f}%)")
    print(f"  Glyph stems that are 2+ tokens:  {stems['glyph_multi_token']:>3} "
          f"({100*stems['glyph_multi_token']/stems['total_pairs']:.0f}%)")
    print()
    print(f"  Glyph BETTER than English (saves tokens):  {stems['glyph_better_than_english']:>3}")
    print(f"  Glyph NEUTRAL (same token count):          {stems['glyph_neutral']:>3}")
    print(f"  Glyph WORSE than English (costs tokens!):  {stems['glyph_worse_than_english']:>3}")
    print()
    if stems['worst_offenders']:
        print("  Top 15 worst offenders (Glyph more expensive than English):")
        for r in stems['worst_offenders']:
            print(f"    {r['english']!r:<14} -> {r['glyph']!r:<10} "
                  f"({r['eng_tokens']} -> {r['glyph_tokens']} tokens, "
                  f"delta={r['delta']:+d})")
    print()
    if stems['biggest_wins']:
        print("  Top 10 biggest wins:")
        for r in stems['biggest_wins']:
            print(f"    {r['english']!r:<14} -> {r['glyph']!r:<10} "
                  f"({r['eng_tokens']} -> {r['glyph_tokens']} tokens, "
                  f"delta={r['delta']:+d})")
    print()
    print("--- Corpus analysis (20-sentence representative corpus) ---")
    print(f"  Total English tokens: {corpus['total_english_tokens']}")
    print(f"  Total Glyph tokens:   {corpus['total_glyph_tokens']}")
    print(f"  Tokens saved:         {corpus['tokens_saved']}")
    print(f"  Pct saved (tokens):   {corpus['pct_saved_tokens']}%")
    print(f"  Pct saved (chars):    {corpus['pct_saved_chars']}% (for reference)")
    print()
    print("--- Headroom analysis ---")
    print(f"  Multi-token stems hit in this corpus: "
          f"{headroom['multi_token_stems_used_in_corpus']}")
    print(f"  Top occurrences:")
    for stem, count in headroom['occurrences'].items():
        print(f"    {stem!r:<10} hit {count:>2}x")
    print(f"  Additional tokens savable if every multi-token "
          f"stem -> 1 token: {headroom['additional_tokens_savable_if_optimized']}")
    print(f"  That's an additional {additional_pct}% of total English tokens.")
    print()
    print("=" * 72)
    print("DECISION INPUT")
    print("=" * 72)
    current = corpus['pct_saved_tokens']
    projected = current + additional_pct
    print(f"  Current measured savings (this corpus, this tokenizer): {current}%")
    print(f"  Projected savings if Phase 1 fully executes:            {projected}%")
    print(f"  Phase 1 go/no-go threshold (BLUEPRINT.md):              ≥15% headroom")
    print(f"  Headroom: {additional_pct}%   "
          f"=> Phase 1 = {'GO' if additional_pct >= 15 else 'PIVOT'}")
    print()

    # Also dump JSON for machine-consumption / future regression tests.
    out = {
        "tokenizer": "tiktoken o200k_base (Claude proxy)",
        "stems": {k: v for k, v in stems.items() if k != "all_rows"},
        "corpus": {k: v for k, v in corpus.items() if k != "per_sentence"},
        "headroom": headroom,
        "decision": "GO" if additional_pct >= 15 else "PIVOT",
        "current_pct_saved": current,
        "projected_pct_saved": projected,
        "headroom_pct": additional_pct,
    }
    out_path = Path(__file__).parent / "glyph_spike_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
