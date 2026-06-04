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

## Semantic tier (Phase 5.2, `method="nli"`)

The `nli` method routes each pair through a local cross-encoder NLI model
(`cross-encoder/nli-deberta-v3-xsmall`, deterministic given fixed weights). It
requires the optional `[semantic]` extra and is measured on the **same corpus**:

| Metric | lexical (baseline) | **nli** |
|---|---|---|
| Accuracy | 0.633 | **0.933** |
| Macro F1 | 0.525 | **0.935** |
| Contradicted — recall | **0.000** | **1.000** |
| Contradicted — F1 | 0.000 | 0.909 |
| Supported — F1 | 0.667 | 0.947 |
| Insufficient — F1 | 0.909 | 0.947 |

The NLI tier catches **all 10 contradictions** the lexical method missed — the
headline weakness, closed and measured. It is not free: it pulls in
`sentence-transformers`/`torch` and is ~100 ms/pair vs sub-millisecond for
lexical, so lexical remains the default. Full nli numbers are in
`results_nli.json`; the lift is regression-guarded by
`tests/test_halubench.py::TestHaluBenchNLI` (which skips where the extra isn't
installed, e.g. CI).

A third method, `llm` (Anthropic judge, `[llm]` extra + `ANTHROPIC_API_KEY`),
is also available; it is non-deterministic and therefore not hash-replayable.

## Grounded verify / RAG (Phase 5.3, `corpus`)

Instead of hand-picking the exact evidence, you can pass a `corpus` (document
pool) and let `chimera_verify` retrieve the top `retrieve_k` snippets per claim
(deterministic token-overlap) and verify against those. Measured on the same 30
claims, with the pool set to **all 30 evidence snippets** (so each claim's
correct snippet is buried among 29 distractors):

| Metric | lexical baseline | oracle nli | **nli + rag** |
|---|---|---|---|
| Accuracy | 0.633 | 0.933 | **0.800** |
| Macro F1 | 0.525 | 0.935 | **0.805** |
| Contradicted — recall | 0.000 | 1.000 | **1.000** |

Honest reading: retrieval keeps most of the NLI lift (0.80 vs 0.93 oracle, well
above the 0.633 lexical baseline) and preserves perfect contradiction recall,
but costs ~13 points of accuracy — the retriever sometimes surfaces a
*contradicting* neighbour for a claim whose true evidence is supportive or
absent (3 supported→contradicted, 3 insufficient→contradicted). Reproduce with
`python -m tools.halubench.run --method nli --rag`; numbers in
`results_nli_rag.json`, guarded by `tests/test_halubench.py::TestHaluBenchRAG`.

## DetectBench (Phase 5.3, `chimera_detect` calibration)

`chimera_detect` screens phrasing and attack patterns, not evidence entailment,
so it is benchmarked on its own task-appropriate corpus (`detect_corpus.json`),
not the verify corpus. Two signals are measured (deterministic, no extra deps):

```bash
python -m tools.halubench.run_detect            # score + report
python -m tools.halubench.run_detect --verbose  # per-item predictions
```

| Signal | precision | recall | F1 |
|---|---|---|---|
| certainty (overconfident phrasing) | 1.000 | 0.800 | 0.889 |
| injection (prompt-injection text) | 1.000 | **0.125** | 0.222 |

### Honest reading

- **Certainty:** perfect precision (never flags hedged statements), recall 0.8 —
  it catches the marker words in its list but misses synonyms outside it
  (`absolutely`, `undoubtedly`). It is a transparent substring matcher, not a
  calibrated classifier.
- **Injection: precision 1.0 but recall only 0.125** — `chimera_detect` reliably
  flags the canonical *"ignore all previous instructions"* phrasing but misses
  most variants (*"override your safety rules"*, *"developer mode"*, etc.). Its
  prompt-injection screening is **narrow**: trustworthy when it fires, but far
  from complete coverage. Expanding the `attack_patterns` pack is the natural
  follow-up. Locked by `tests/test_halubench.py::TestDetectBench` so any future
  recall lift is a deliberate, visible change.
