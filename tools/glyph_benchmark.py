"""Phase 1 — formal benchmark harness for Chimera Glyph.

Runs the encoder against a fixed 100-sentence corpus across 5 domains
(error, dialogue, instruction, reasoning, prose), measures BPE token
counts via tiktoken's o200k_base (Claude-equivalent), and reports
per-domain + overall token reduction plus decode fidelity.

Outputs:
  - tools/glyph_benchmark_results.json — machine-readable totals + per-sentence
  - stdout — human-readable summary suitable for the case study

Usage:
  python tools/glyph_benchmark.py
  python tools/glyph_benchmark.py --validate-anthropic   # also queries the
      real Anthropic count_tokens API for the headline numbers (requires
      ANTHROPIC_API_KEY in the environment).

The harness is deterministic: re-running gives identical numbers, which
lets tests/test_glyph_benchmark.py lock the headline as a regression gate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
from chimeralang_mcp.ai_language import decode, encode  # noqa: E402

ENC = tiktoken.get_encoding("o200k_base")
CORPUS_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "glyph_bench.txt"
RESULTS_PATH = Path(__file__).parent / "glyph_benchmark_results.json"


# ── corpus loading ───────────────────────────────────────────────────────


class Sentence(NamedTuple):
    domain: str
    text: str


def load_corpus(path: Path = CORPUS_PATH) -> list[Sentence]:
    sentences: list[Sentence] = []
    current_domain = "uncategorized"
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # Domain header: "## DOMAIN: error / debugging"
        m = re.match(r"^##\s*DOMAIN:\s*(.+)$", line)
        if m:
            current_domain = m.group(1).strip()
            continue
        if line.startswith("#"):
            continue
        sentences.append(Sentence(domain=current_domain, text=line))
    return sentences


# ── token + fidelity measurement ─────────────────────────────────────────


def n_tokens(s: str) -> int:
    return len(ENC.encode(s))


def decode_fidelity(original: str, decoded: str) -> float:
    """Token-overlap (Jaccard-like, content-word level) between the
    original English and the decoded English. Range 0.0 – 1.0; lossy is
    fine, but sustained <0.5 means the decoder is producing gibberish."""
    norm = lambda s: {  # noqa: E731
        w.lower().strip(".,!?;:")
        for w in s.split()
        if w.strip(".,!?;:")
    } - {  # ignore function words from the fidelity score
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "at", "for", "and", "or", "with", "by",
    }
    orig = norm(original)
    dec = norm(decoded)
    if not orig:
        return 1.0
    return len(orig & dec) / len(orig | dec) if (orig | dec) else 1.0


# ── benchmark runner ─────────────────────────────────────────────────────


def run_benchmark(sentences: list[Sentence], validate_anthropic: bool = False) -> dict:
    per_sentence = []
    by_domain: dict[str, list[dict]] = {}

    anth_client = None
    if validate_anthropic:
        try:
            import anthropic  # type: ignore
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("WARNING: --validate-anthropic requires ANTHROPIC_API_KEY; skipping.",
                      file=sys.stderr)
            else:
                anth_client = anthropic.Anthropic()
        except ImportError:
            print("WARNING: anthropic package not installed; skipping API validation.",
                  file=sys.stderr)

    for sent in sentences:
        glyph = encode(sent.text)
        decoded, decode_notes = decode(glyph)

        eng_tok = n_tokens(sent.text)
        gly_tok = n_tokens(glyph)
        eng_chars = len(sent.text)
        gly_chars = len(glyph)

        anth_eng = anth_gly = None
        if anth_client:
            try:
                anth_eng = anth_client.beta.messages.count_tokens(
                    model="claude-sonnet-4-5",
                    messages=[{"role": "user", "content": sent.text}],
                ).input_tokens
                anth_gly = anth_client.beta.messages.count_tokens(
                    model="claude-sonnet-4-5",
                    messages=[{"role": "user", "content": glyph}],
                ).input_tokens
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: Anthropic call failed for {sent.text!r}: {e}",
                      file=sys.stderr)
                anth_client = None

        row = {
            "domain": sent.domain,
            "english": sent.text,
            "glyph": glyph,
            "decoded": decoded,
            "decode_notes": decode_notes,
            "decode_fidelity": round(decode_fidelity(sent.text, decoded), 3),
            "eng_tokens_o200k": eng_tok,
            "glyph_tokens_o200k": gly_tok,
            "tokens_saved_o200k": eng_tok - gly_tok,
            "pct_saved_tokens_o200k": round(100 * (eng_tok - gly_tok) / max(1, eng_tok), 1),
            "eng_chars": eng_chars,
            "glyph_chars": gly_chars,
            "pct_saved_chars": round(100 * (eng_chars - gly_chars) / max(1, eng_chars), 1),
            "eng_tokens_anthropic": anth_eng,
            "glyph_tokens_anthropic": anth_gly,
        }
        per_sentence.append(row)
        by_domain.setdefault(sent.domain, []).append(row)

    # ── aggregations ────────────────────────────────────────────────────

    def aggregate(rows: list[dict]) -> dict:
        total_eng_tok = sum(r["eng_tokens_o200k"] for r in rows)
        total_gly_tok = sum(r["glyph_tokens_o200k"] for r in rows)
        total_eng_chars = sum(r["eng_chars"] for r in rows)
        total_gly_chars = sum(r["glyph_chars"] for r in rows)
        avg_fid = sum(r["decode_fidelity"] for r in rows) / max(1, len(rows))
        agg = {
            "n": len(rows),
            "eng_tokens": total_eng_tok,
            "glyph_tokens": total_gly_tok,
            "tokens_saved": total_eng_tok - total_gly_tok,
            "pct_saved_tokens": round(
                100 * (total_eng_tok - total_gly_tok) / max(1, total_eng_tok), 1
            ),
            "pct_saved_chars": round(
                100 * (total_eng_chars - total_gly_chars) / max(1, total_eng_chars), 1
            ),
            "avg_decode_fidelity": round(avg_fid, 3),
        }
        # Anthropic side, if validated
        anth_eng = [r["eng_tokens_anthropic"] for r in rows if r["eng_tokens_anthropic"]]
        anth_gly = [r["glyph_tokens_anthropic"] for r in rows if r["glyph_tokens_anthropic"]]
        if anth_eng and anth_gly and len(anth_eng) == len(anth_gly):
            agg["anthropic_eng_tokens"] = sum(anth_eng)
            agg["anthropic_glyph_tokens"] = sum(anth_gly)
            agg["anthropic_pct_saved"] = round(
                100 * (sum(anth_eng) - sum(anth_gly)) / max(1, sum(anth_eng)), 1
            )
        return agg

    return {
        "tokenizer_proxy": "tiktoken o200k_base",
        "anthropic_validated": anth_client is not None,
        "n_sentences": len(per_sentence),
        "n_domains": len(by_domain),
        "overall": aggregate(per_sentence),
        "by_domain": {
            domain: aggregate(rows) for domain, rows in by_domain.items()
        },
        "per_sentence": per_sentence,
    }


# ── reporting ────────────────────────────────────────────────────────────


def print_report(report: dict) -> None:
    print("=" * 72)
    print("CHIMERA GLYPH — FORMAL BENCHMARK")
    print(f"Tokenizer: {report['tokenizer_proxy']}"
          + (" + Anthropic API validation" if report['anthropic_validated'] else ""))
    print(f"Corpus: {report['n_sentences']} sentences across "
          f"{report['n_domains']} domains")
    print("=" * 72)
    overall = report["overall"]
    print()
    print("OVERALL")
    print(f"  Sentences:            {overall['n']}")
    print(f"  English tokens:       {overall['eng_tokens']}")
    print(f"  Glyph tokens:         {overall['glyph_tokens']}")
    print(f"  Tokens saved:         {overall['tokens_saved']}")
    print(f"  Pct saved (tokens):   {overall['pct_saved_tokens']}%")
    print(f"  Pct saved (chars):    {overall['pct_saved_chars']}%")
    print(f"  Avg decode fidelity:  {overall['avg_decode_fidelity']:.3f}")
    if "anthropic_pct_saved" in overall:
        print(f"  Anthropic API saved:  {overall['anthropic_pct_saved']}% "
              f"({overall['anthropic_eng_tokens']} -> "
              f"{overall['anthropic_glyph_tokens']} tokens)")
    print()
    print("BY DOMAIN")
    for domain, agg in report["by_domain"].items():
        print(f"  {domain:<30} n={agg['n']:>3}  "
              f"tokens: {agg['pct_saved_tokens']:>5}% saved   "
              f"chars: {agg['pct_saved_chars']:>5}% saved   "
              f"fid: {agg['avg_decode_fidelity']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-anthropic", action="store_true",
                        help="Also query the real Anthropic count_tokens API")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH,
                        help="Path to the benchmark corpus")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH,
                        help="Path for the JSON results")
    args = parser.parse_args()

    sentences = load_corpus(args.corpus)
    report = run_benchmark(sentences, validate_anthropic=args.validate_anthropic)
    print_report(report)

    args.out.write_text(json.dumps(report, indent=2))
    print()
    print(f"Full results written to {args.out}")


if __name__ == "__main__":
    main()
