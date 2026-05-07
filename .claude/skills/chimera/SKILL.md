---
name: chimera
description: Use when a prompt involves a long document or log, a conversation history that needs trimming, AI-to-AI reasoning with token cost concerns, claim verification or hallucination detection, multi-tool batched work, or any task whose intent maps onto the chimeralang-mcp tool surface. Enforces use of the right `chimera_*` tool for the job and skips chimera tools entirely for trivial inputs.
---

# Chimera skill — smart routing into chimeralang-mcp

Works for both Claude Code (via this `SKILL.md`) and Codex (via the sibling `AGENTS.md` adapter at repo root). Both agents end up running the same routing rules.

The chimeralang-mcp server (currently `0.7.1`) exposes 51 tools. Routing all of them through every prompt is wasteful. The goal of this skill is: pick the smallest correct tool subset for the user's actual intent and execute the work through it.

---

## Decision tree (run in order)

1. **Trivial input?** If the prompt is `< 200` chars and there are no documents/logs/messages attached, skip every chimera tool and answer directly. The token-saving hooks in `.claude/settings.json` already handle short prompts.
2. **Has structured payload?** If the prompt or attached artifact is a document (`> 500` chars), a build/test log, a long message history, or a corpus of multiple docs — go to the **Routing matrix** below.
3. **Reasoning / verification request?** If the user is asking for analysis, claim checking, multi-perspective deliberation, or hallucination detection — go to the **Reasoning lane**.
4. **AI-to-AI reasoning?** If the user wants the agent to reason internally (without producing prose for them yet), or wants to compress a system prompt for another LLM — use the **Glyph lane**.
5. **End-of-task?** Always finish with `chimera_session_report` so totals are surfaced. The Stop hook does this automatically — only call manually if you want telemetry mid-session.

---

## Routing matrix — prompt pattern → chimera tool

| Trigger | First-choice tool | Notes |
|---|---|---|
| Single doc/code blob > 500 chars | `chimera_optimize` | 60–75% reduction typical; preserves fenced code with `preserve_code=true`. |
| Long conversation history | `chimera_compress` | Proportional truncation to a token budget. |
| Both docs + history | `chimera_fracture` | Single call with quality gate; prefer this over chaining the two above. |
| Build / test / install / install log | `chimera_log_compress` | Errors and warnings preserved verbatim, body abridged. |
| Stable system block reused across SDK calls | `chimera_cache_mark` | Lossless 75–90% via Anthropic prompt cache markers. |
| Multiple chimera calls in one go | `chimera_batch` | Saves round-trips when the input is fan-out parallel. |
| Recurring per-turn cost surfacing | `chimera_overhead_audit` | Quantifies "ghost tokens" (system prompt + tool defs). |
| Inspect repeated tool calls | `chimera_dedup_lookup` | The PostToolUse hook populates the cache. |
| End-of-session totals | `chimera_session_report` | Bundles cost summary + budget + dedup. Stop hook auto-fires. |
| LLM-free extractive summary | `chimera_summarize` | Fast and deterministic; no Claude API call. |

---

## Reasoning lane — verification, deliberation, claims

| Trigger | First-choice tool | Notes |
|---|---|---|
| Extract claims from text | `chimera_claims` | Returns atomic claims with hedge/abstention tags. |
| Verify claims against evidence | `chimera_verify` | Lexical token-overlap scoring. |
| Multi-perspective analysis | `chimera_deliberate` | Multiple perspective passes. |
| Confidence-weighted vote across N answers | `chimera_quantum_vote` | Use when you have multiple candidate responses. |
| Collapse candidates into one consensus | `chimera_gate` | Lighter than `quantum_vote`. |
| Hallucination / MCP attack pattern detection | `chimera_detect` | Pattern-based safety check. |
| Type-check a ChimeraLang program | `chimera_typecheck` | Static check, no execution. |
| Execute a ChimeraLang program | `chimera_run` | Returns full execution envelope. |
| Need a Merkle-chain integrity proof of a run | `chimera_prove` | Heaviest of the run options. |
| Causal graph from claims | `chimera_causal` | Graph builder; pair with `chimera_claims` upstream. |
| Calibration error (ECE) from predictions | `chimera_metacognize` | Needs `[{predicted_confidence, was_correct}]`. |

---

## Glyph lane — AI-to-AI compression via Chimera Glyph

| Trigger | Tool sequence |
|---|---|
| Internal reasoning step the user will not read | 1. `chimera_glyph_directive(style="strict", task_hint=...)` 2. Agent emits CG  3. `chimera_glyph_translate(verbosity="natural")` to recover English at the boundary. |
| User asks for a system prompt that forces CG | `chimera_glyph_directive` only — paste the returned `directive` into the system slot. |
| User pastes CG and wants the English | `chimera_glyph_translate(verbosity="natural")`; use `terse` to strip heuristic articles/copulas. |

Measured savings on a 10-sentence representative corpus: 46.2% token reduction, 0.11 ms/sentence end-to-end. See `docs/case-studies/chimera-glyph-feature.md` for the full benchmark.

---

## Mode selection (which subset of tools to surface)

`chimera_mode` returns four canonical subsets. Pick one early and stick with it for the duration of the task:

| Mode | Tools active | When to use |
|---|---|---|
| `minimal` (6) | csm, budget_lock, gate, confident, memory, policy | Short Q&A, no large payload. Default. |
| `token` (10) | + optimize, compress, fracture, budget, score, cost_estimate, cost_track, dashboard | User has a long doc / log / conversation history. |
| `agi` (19) | + causal, deliberate, metacognize, quantum_vote, plan_goals, world_model, safety_check, ethical_eval, knowledge, claims, verify, provenance_merge, trace, materials | Multi-step reasoning, planning, verification chains. |
| `full` (51) | All | Only when the task genuinely spans every domain (rare). |

Use `chimera_mode(task_description="...")` for auto-detection — keywords like *compress / budget / token / cost* steer it to `token`, while *reason / analyze / causal / plan / deliberate* steers it to `agi`.

---

## Enforcement checklist (run at the start of any non-trivial task)

The skill is **enforcing** when the agent does these in order:

1. `chimera_csm` — call first on every message that has a payload (the description literally says *"CALL FIRST on every message"*).
2. `chimera_mode(task_description=<the user's prompt>)` — pick the subset.
3. Route to the matrix above for the actual work.
4. If multiple sub-tasks: wrap them in `chimera_batch` to save round-trips.
5. Before returning a long response: `chimera_score` to rank what to keep, `chimera_summarize` if a compressed answer is acceptable.
6. After the work is done: `chimera_session_report` — only manually if telemetry is needed mid-session, otherwise the Stop hook handles it.

If any of these steps would consume more tokens than they save (e.g. running `chimera_optimize` on a 100-char prompt), skip that step. The skill is enforcing the **intent** of using chimeralang-mcp where appropriate, not blanket coverage.

---

## Skip conditions (do not invoke chimera tools)

- Prompt is `< 200` chars and has no attachments.
- The user is debugging the chimeralang-mcp project itself (don't compress the very thing you're trying to read).
- The user is asking a meta question about the tools (definitions, schemas, "what does X do") — answer directly.
- The user has explicitly said "skip chimera" or "raw output only".

---

## Verified behavior (real measurements from this session)

These numbers came from running the actual tools while writing this skill:

- `chimera_optimize` on the project's `CLAUDE.md` token-policy section: 1463 → 400 chars, **265 tokens saved (72.6%)**, 4 of 21 sentence units retained, focus terms preserved (`chimera_compress`, `chimera_optimize`, `conversation`, `history`, ...).
- `chimera_fracture` on a mixed payload (650 chars docs + 4-message history, budget 200 tokens): **99 tokens saved**, quality gate passed in **3.1 ms**.
- `chimera_log_compress` on a focused failure log: 24/24 lines kept verbatim because every line carried error/warning/percentage signals — correct behavior for short failure-heavy logs.
- `chimera_glyph_translate` round trip on `w nd kn how usr wnt fix err in fn. mb aprch ~ wrk. if tst fail ⇒ w ? rt-retry x.` → 20 glyph tokens decode to 34 English tokens, modality (`maybe`, `will`, `might`) preserved, zero unrecognized notes.
- `chimera_dashboard` after one logged event: 1 request, **265 tokens saved**, **$0.000795 saved** at Sonnet 4.6 pricing, 72.6% avg reduction.

---

## Known issues observed while writing this skill (file as follow-ups)

1. `chimera_cost_track` returns an `UnboundLocalError: cannot access local variable 'log'` even though the entry IS persisted (visible in `chimera_dashboard`). The error response is a false negative.
2. `chimera_mode` accepts `mode` and `task_description` — `task_type` is ignored and falls through to `minimal`. Schema and docs say `task_type`, handler reads `task_description`. Either the schema or the handler needs updating.

---

## Codex compatibility

Codex does not auto-discover files inside `.claude/skills/`. The repo-root `AGENTS.md` (created alongside this file) contains the same routing matrix in Codex's preferred location so a Codex agent walking up from cwd will find it.

If you are inside Codex, treat this file's matrix and decision tree as binding and call the chimeralang-mcp tools through the same routing logic.
