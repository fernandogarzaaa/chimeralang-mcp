# HaluBench — hallucination-detection benchmark

HaluBench measures how well `chimera_verify` classifies a claim against supplied
evidence as **supported**, **contradicted**, or **insufficient**. It exists so
the hallucination pillar has a *published number* — the same discipline Phase 1
applied to Glyph compression.

## Why it exists

`chimera_verify`'s default scoring is **lexical** (Jaccard token overlap +
negation heuristics), not semantic entailment. That is fast, deterministic, and
transparent — but it cannot catch a contradiction that reuses the claim's
vocabulary. HaluBench quantifies exactly where that breaks, and gives any future
semantic tier (`method="nli"` / `method="llm"`, Phase 5.2) a baseline to beat.

## Corpus

`corpus.json` — 30 hand-authored, human-labeled `{claim, evidence, label}`
triples, balanced 10 / 10 / 10 across the three verdict classes and spread over
five domains (geography, science, history, health, tech). The corpus is offline
and deterministic (no network, no external datasets). **None of these triples
appear in the built-in `verification_gold` pack**, so scores reflect
generalization, not memorization.

## Running

```bash
python -m tools.halubench.run            # score + report
python -m tools.halubench.run --verbose  # also print per-item predictions
python -m tools.halubench.run --update   # rewrite baseline_results.json
python -m tools.halubench.run --method nli   # (Phase 5.2) semantic tier
```

Every item is scored by calling the live `chimera_verify` MCP tool.

## Baseline (v1, lexical method)

| Metric | Value |
|---|---|
| Accuracy | **0.633** |
| Macro F1 | **0.525** |
| Supported — P / R / F1 | 0.529 / 0.900 / 0.667 |
| Contradicted — P / R / F1 | **0.000 / 0.000 / 0.000** |
| Insufficient — P / R / F1 | 0.833 / 1.000 / 0.909 |

Confusion (rows = gold, cols = predicted):

|              | supported | contradicted | insufficient |
|--------------|-----------|--------------|--------------|
| supported    | 9         | 1            | 0            |
| contradicted | 8         | 0            | 2            |
| insufficient | 0         | 0            | 10           |

### Honest reading of the baseline

- **Contradiction recall is 0.0.** The lexical method catches none of the 10
  contradictions; it labels 8 of them "supported" because the claim and the
  evidence share tokens (e.g. *"The Nile River flows through Asia"* vs *"The Nile
  River flows through northeastern Africa"*). This is the headline weakness the
  semantic tier must fix.
- **Insufficient detection is strong (F1 0.909).** Off-topic evidence has low
  token overlap, which lexical scoring handles well.
- **Supported recall is high but precision is only 0.53**, dragged down by
  contradictions leaking into the "supported" bucket.

The numbers above are locked by `tests/test_halubench.py`; any change to the
scoring engine that moves them will fail the regression test loudly.
