# Codex Agent Guide — chimeralang-mcp

> **chimera_version: 0.7.5** — keep this in sync with `pyproject.toml` and `.claude/skills/chimera/SKILL.md`.

This file is the Codex-readable adapter of the **chimera skill**. The canonical version lives at `.claude/skills/chimera/SKILL.md`; this is the same routing matrix in the location Codex actually walks.

When Codex is operating inside this repository, follow these rules.

---

## When to invoke chimeralang-mcp

The chimeralang-mcp server (currently `0.7.5`) exposes 51 tools. Pick the smallest correct subset for the user's actual intent — never invoke all of them.

**Skip chimera tools entirely** if:
- Prompt is `< 200` chars and has no attached document/log/history.
- The user is debugging chimeralang-mcp itself (don't pre-compress the thing you're trying to read).
- The user is asking a meta question about tools (definitions, schemas).
- The user said "skip chimera" or "raw output only".

**Engage chimera tools** otherwise — the cost-saving compounds quickly.

---

## Routing matrix — prompt pattern → tool

| Trigger | First-choice tool | Notes |
|---|---|---|
| Single doc/code blob > 500 chars | `chimera_optimize` | 60–75% reduction; pass `preserve_code=true` to keep fenced blocks. |
| Long conversation history | `chimera_compress` | Proportional truncation. |
| Both docs + history | `chimera_fracture` | One call, quality gate, prefer over chaining. |
| Build/test/install log | `chimera_log_compress` | Errors/warnings verbatim, body abridged. |
| Stable system block reused across SDK calls | `chimera_cache_mark` | Lossless 75–90% via Anthropic prompt cache. |
| Multiple chimera calls at once | `chimera_batch` | Saves round-trips. |
| Recurring per-turn cost surfacing | `chimera_overhead_audit` | Quantifies system-prompt + tool-def overhead. |
| Inspect repeated tool calls | `chimera_dedup_lookup` | Populated by the PostToolUse hook. |
| End-of-session totals | `chimera_session_report` | Already auto-fired by Stop hook. |
| LLM-free extractive summary | `chimera_summarize` | Deterministic, no API call. |

---

## Calling conventions — exact arg names and types

| Tool | Correct call | Wrong call |
|---|---|---|
| `chimera_log_compress` | `text="<log content>"` | `log="..."` — param is `text`, not `log` |
| `chimera_optimize` | `preserve_code=true` (JSON bool) | `preserve_code="true"` (string rejected) |
| `chimera_cost_track` | `tokens_saved=265` (integer) | `tokens_saved="265"` (string rejected) |
| `chimera_mode` | `task_description="..."` | `task_type="..."` (unknown args silently dropped) |
| `chimera_glyph_translate` | `verbosity="terse"` or `"natural"` | any other string → falls back to `"natural"` |
| `chimera_batch` | `operations=[{"tool": "chimera_optimize", "arguments": {...}}]` | flat args |

---

## Reasoning lane — verification, deliberation, claims

| Trigger | First-choice tool |
|---|---|
| Extract claims from text | `chimera_claims` |
| Verify claims against evidence | `chimera_verify` |
| Multi-perspective analysis | `chimera_deliberate` |
| Confidence-weighted vote across N answers | `chimera_quantum_vote` |
| Collapse candidates into one consensus | `chimera_gate` |
| Hallucination / MCP attack detection | `chimera_detect` |
| Type-check ChimeraLang program | `chimera_typecheck` |
| Execute ChimeraLang program | `chimera_run` |
| Merkle integrity proof of a run | `chimera_prove` |
| Causal graph from claims | `chimera_causal` |
| Calibration error (ECE) | `chimera_metacognize` |

---

## Glyph lane — AI-to-AI compression

For internal reasoning the user will not directly read:

1. `chimera_glyph_directive(style="strict", task_hint=<the task>)` → emit the directive into the system slot.
2. The agent produces output in Chimera Glyph (CG).
3. `chimera_glyph_translate(glyph_text=<output>, verbosity="natural")` → recover English at the boundary.

**Worked example:**
- English: *"The user wants to know how to fix the error returned by the function."* (14 tokens)
- CG: `usr wnt kn fix err $rt fn.` (≈8 tokens)
- Decoded: *"User wants know fix error return function."*

**Measured (0.7.3, 100-sentence benchmark, tiktoken o200k_base):** Glyph is **−16% on tokens** (encoded form is *longer* in BPE tokens) and **+19.2% on characters**, with **0.806 decode fidelity**. The hand-crafted glyph approach cannot beat learned BPE on token cost — modern tokenizers already collapse common English to single tokens. Glyph's value is **semantic determinism**, not compression: deterministic decode, sigil-marked entities, explicit modality tokens. Use `chimera_optimize` / `chimera_fracture` / `chimera_cache_mark` for actual token reduction. See `docs/case-studies/chimera-glyph-feature.md` for the full benchmark.

---

## Mode selection

`chimera_mode` returns canonical subsets. Pick one early and stick with it.

| Mode | Tools | When |
|---|---|---|
| `minimal` (6) | csm, budget_lock, gate, confident, memory, policy | Short Q&A, no payload. |
| `token` (10) | + optimize, compress, fracture, budget, score, cost_estimate, cost_track, dashboard | Long doc / log / conversation. |
| `agi` (19) | + causal, deliberate, metacognize, quantum_vote, plan_goals, world_model, safety_check, ethical_eval, knowledge, claims, verify, provenance_merge, trace, materials | Multi-step reasoning, planning, verification. |
| `full` (51) | All | Only if the task spans every domain (rare). |

Auto-detect via `chimera_mode(task_description="...")`. Keywords *compress/budget/token/cost* → `token`; *reason/analyze/causal/plan/deliberate* → `agi`.

---

## Worked examples — prompt → tool sequence

**1. Summarize a large design doc**
```
chimera_csm → chimera_mode(task_description="summarize design doc")
→ chimera_optimize(text=<doc>, preserve_code=false) → chimera_summarize(text=<compressed>)
```

**2. Diagnose a failing CI log**
```
chimera_csm → chimera_log_compress(text=<log>) → answer from compressed output
```

**3. Multi-perspective analysis**
```
chimera_csm → chimera_mode(task_description="analyze competing designs")
→ chimera_deliberate(...) → chimera_gate(candidates=[...])
```

**4. Claim verification**
```
chimera_csm → chimera_claims(text=<claim>) → chimera_verify(claims=[...], evidence=<source>)
→ chimera_detect(text=<claim>)
```

**5. AI-to-AI compressed reasoning**
```
chimera_glyph_directive(style="strict", task_hint=<task>)
→ [agent emits CG] → chimera_glyph_translate(glyph_text=<CG>, verbosity="natural")
```

---

## Enforcement checklist (start of any non-trivial task)

1. `chimera_csm` — first call on every message with a payload.
2. `chimera_mode(task_description=<prompt>)` — pick the subset.
3. Route into the matrix above for the actual work.
4. Sub-tasks → wrap in `chimera_batch`.
5. Before returning long output → `chimera_score` to rank, `chimera_summarize` if compressed is acceptable.
6. End of work → `chimera_session_report` (or let the Stop hook fire it).

Skip any step that would cost more tokens than it saves.

---

## Telemetry reaction matrix

| Signal | Threshold | Action |
|---|---|---|
| `avg_pct_saved` | < 30% | Inputs too short — check skip conditions; avoid chimera on sub-200-char prompts |
| `avg_pct_saved` | > 70% | Healthy — add `chimera_cache_mark` on stable system blocks |
| Dedup hits | > 2 in session | Wrap repeated calls in `chimera_batch` |
| Session savings | > $0.01 | Record for ROI evidence via `chimera_dashboard` |
| `chimera_cost_track` errors | — | Fixed in `0.7.2`; on older versions the entry persists despite the error |

---

## Project-specific Codex notes

- **Branch convention.** All new development for AI coding agents in this repo lives on `claude/<task-slug>` or `docs/<task-slug>` branches. Never push directly to `main`. Open a PR.
- **Tests.** `python -m pytest -q` from repo root. 263 tests pass on `main` as of `0.7.5`. Don't ship a change that drops that count.
- **Release pipeline.** Merging to `main` triggers `.github/workflows/publish.yml` which uploads to PyPI via OIDC Trusted Publishing. Bump `pyproject.toml` AND `chimeralang_mcp/__init__.py` together.
- **Code style.** No new dependencies in the language module (`chimeralang_mcp/ai_language.py` is stdlib-only). Match existing patterns in `server.py` rather than refactoring.

---

## Pointer back to canonical skill

The full version of these rules — including verified behavior measurements from a live tool-pipeline run, decision tree, and known-issue list — lives at:

```
.claude/skills/chimera/SKILL.md
```

Both files are kept in sync. If you edit one, edit the other.
