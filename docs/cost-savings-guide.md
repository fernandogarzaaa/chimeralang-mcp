# Token Cost Savings — what works, in plain terms

This is the project's "what to use when you actually want to save tokens" guide.
After Phase 1 of `BLUEPRINT.md` falsified Chimera Glyph as a token compressor,
the open question was: *can chimeralang-mcp reduce token costs at all?* The
answer is yes — through a different set of tools. This doc lists them with
real measured numbers, in priority order.

All numbers below come from `python tools/cost_savings_demo.py`, which is
deterministic and reproducible. Re-run the script to verify.

---

## Measured savings on representative payloads

| Tool | Payload | Before | After | Saved |
|---|---|---:|---:|---:|
| `chimera_cache_mark` | stable ~5 kB system block, repeated across calls | 1215 | 122 | **90.0%** |
| `chimera_compress` | 12-turn dialogue text, level=aggressive | 509 | 143 | **71.9%** |
| `chimera_optimize` | long doc (the project's own `CLAUDE.md`) | 992 | 287 | **71.1%** |
| `chimera_log_compress` | 320-line log with one warning, one error, one traceback | 328 | 163 | **50.3%** |
| `chimera_fracture` | long doc + 12-msg dialogue, 600-token budget | 497 | 432 | **13.1%** |

`chimera_fracture` looks "weak" because the input was already small; on bigger
mixed payloads it does the work of `optimize` + `compress` in a single call.

`chimera_cache_mark`'s 90% is **lossless** — it relies on Anthropic's native
prompt cache pricing (cached tokens ~10% of normal cost on reuse). The other
four are heuristic/lossy compressors.

---

## Decision tree — pick one, don't stack

```
Do you have a stable block (system prompt, tool defs) you'll send on every call?
├── Yes →  chimera_cache_mark      (lossless, biggest win, sets cache_control: ephemeral)
│
└── No →  Is the payload …
    ├── A single long document or code blob (>500 chars)
    │     →  chimera_optimize       (60–75% savings, drops filler/dedupes/normalizes)
    │
    ├── A long conversation/text, you want max squash
    │     →  chimera_compress       (60–80% with level=aggressive)
    │
    ├── A build/test/install log
    │     →  chimera_log_compress   (50–80%; errors/warnings kept verbatim)
    │
    ├── Mixed: documents AND message history, with a token budget
    │     →  chimera_fracture       (one call, quality gate, budget-aware)
    │
    └── Short prompt (<200 chars)
          →  skip — chimera tools cost more than they save here
```

---

## What about Glyph?

Don't use Chimera Glyph for cost reduction. Phase 1's formal benchmark
measured it at **−16% on tokens** (the encoded form is *longer* than English)
on a 100-sentence corpus against a Claude-equivalent BPE. The 0.7.0 case
study's "46.2%" number was character savings, not token savings.

Glyph is now positioned as a deterministic AI-to-AI **wire format** with
sigil-marked entities and explicit modality tokens — useful for verifiable
agent-to-agent handoffs (see `docs/case-studies/cross-agent-protocol.md`),
not for cutting API spend.

---

## What we don't ship — but you can pair with

- **LLMLingua-2** (Microsoft): learned compression, ~20× reduction with
  ~1.5pt accuracy drop. Best-in-class for one-shot prompts. Requires GPU
  inference at runtime. We don't reimplement it; pair it with chimeralang-mcp
  if you need that compression tier.
- **Anthropic prompt cache** (native, no extra tooling): handled by
  `chimera_cache_mark`, which generates the right `cache_control` markers.

---

## Hooks already do this for you on long inputs

`.claude/settings.json` ships hooks that fire automatically:

- **UserPromptSubmit** — auto-runs the optimizer on any prompt > 800 chars.
- **PostToolUse** — flags any tool result > 2000 chars with a recommendation
  to compress.
- **Stop** — emits a session-totals summary with tokens saved.

So most of these tools are invoked behind the scenes. The list above tells
you which one to call manually when you want to be explicit.

---

## Reproducing the numbers

```bash
python tools/cost_savings_demo.py
```

Writes `tools/cost_savings_results.json`. The exact numbers depend on the
project's `CLAUDE.md` content (used as the "long doc" payload), so they may
drift over time as the file evolves. Re-run any time to refresh the table
above.
