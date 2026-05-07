# Codex Agent Guide — chimeralang-mcp

This file is the Codex-readable adapter of the **chimera skill**. The canonical version lives at `.claude/skills/chimera/SKILL.md`; this is the same routing matrix in the location Codex actually walks.

When Codex is operating inside this repository, follow these rules.

---

## When to invoke chimeralang-mcp

The chimeralang-mcp server (currently `0.7.1`) exposes 51 tools. Pick the smallest correct subset for the user's actual intent — never invoke all of them.

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

Measured: 46.2% token reduction on a 10-sentence corpus, 0.11 ms/sentence end-to-end. See `docs/case-studies/chimera-glyph-feature.md` for the benchmark.

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

## Enforcement checklist (start of any non-trivial task)

1. `chimera_csm` — first call on every message with a payload.
2. `chimera_mode(task_description=<prompt>)` — pick the subset.
3. Route into the matrix above for the actual work.
4. Sub-tasks → wrap in `chimera_batch`.
5. Before returning long output → `chimera_score` to rank, `chimera_summarize` if compressed is acceptable.
6. End of work → `chimera_session_report` (or let the Stop hook fire it).

Skip any step that would cost more tokens than it saves.

---

## Project-specific Codex notes

- **Branch convention.** All new development for AI coding agents in this repo lives on `claude/<task-slug>` or `docs/<task-slug>` branches. Never push directly to `main`. Open a PR.
- **Tests.** `python -m pytest -q` from repo root. 206 tests pass on `main` as of `0.7.1`. Don't ship a change that drops that count.
- **Release pipeline.** Merging to `main` triggers `.github/workflows/publish.yml` which uploads to PyPI via OIDC Trusted Publishing. Bump `pyproject.toml` AND `chimeralang_mcp/__init__.py` together.
- **Code style.** No new dependencies in the language module (`chimeralang_mcp/ai_language.py` is stdlib-only). Match existing patterns in `server.py` rather than refactoring.

---

## Pointer back to canonical skill

The full version of these rules — including verified behavior measurements from a live tool-pipeline run, decision tree, and known-issue list — lives at:

```
.claude/skills/chimera/SKILL.md
```

Both files are kept in sync. If you edit one, edit the other.
