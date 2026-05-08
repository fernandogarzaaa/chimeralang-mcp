---
name: chimera
chimera_version: "0.7.5"
description: "Trigger: large document (>500 chars), build/test log, long conversation history, claim-checking, hallucination detection, multi-tool batch, or token-cost concerns. Routes each request to the smallest correct chimera_* tool subset and skips chimera entirely for prompts <200 chars with no attachments."
---

# Chimera skill — smart routing into chimeralang-mcp

Works for both Claude Code (via this `SKILL.md`) and Codex (via the sibling `AGENTS.md` adapter at repo root). Both agents end up running the same routing rules.

The chimeralang-mcp server (currently `0.7.5`) exposes 51 tools. Routing all of them through every prompt is wasteful. The goal of this skill is: pick the smallest correct tool subset for the user's actual intent and execute the work through it.

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

## Calling conventions — exact arg names and types

Common mistakes that cause tool errors:

| Tool | Correct call | Wrong call |
|---|---|---|
| `chimera_log_compress` | `text="<log content>"` | `log="..."` — param is `text`, not `log` |
| `chimera_optimize` | `preserve_code=true` (JSON bool) | `preserve_code="true"` (string rejected) |
| `chimera_cost_track` | `tokens_saved=265` (integer) | `tokens_saved="265"` (string rejected) |
| `chimera_mode` | `task_description="..."` | `task_type="..."` (unknown args are silently dropped) |
| `chimera_glyph_translate` | `verbosity="terse"` or `"natural"` | any other string → silently falls back to `"natural"` |
| `chimera_batch` | `operations=[{"tool": "chimera_optimize", "arguments": {...}}]` | flat args — must be an array of `{tool, arguments}` objects |

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

### Glyph worked example

English (14 tokens): *"The user wants to know how to fix the error returned by the function."*
CG (≈8 tokens): `usr wnt kn fix err $rt fn.`
Decoded English: *"User wants know fix error return function."*

**Measured (0.7.3, 100-sentence benchmark, tiktoken o200k_base):**

| Metric | Value |
|---|---|
| Token reduction | **−16.0%** (Glyph is longer in BPE tokens than English) |
| Character reduction | +19.2% |
| Decode fidelity | 0.806 (lossy by design) |

The hand-crafted glyph approach cannot beat learned BPE on token cost — modern tokenizers already collapse common English to single tokens. **Glyph's value is semantic determinism, not compression:** deterministic decode (no LLM call needed), sigil-marked entities (`@MyService` is unambiguously a proper noun), explicit modality tokens (`~`/`?`/`!` for will/might/must). For actual token reduction use `chimera_optimize`, `chimera_fracture`, or `chimera_cache_mark`. See `docs/case-studies/chimera-glyph-feature.md` for the full benchmark.

**Grammar cheat-sheet:**
- No articles (`a/an/the`) — drop them.
- No copulas (`is/are/was`) — implied by juxtaposition.
- Tense: suffix on verb — `^` past, `~` future (or standalone `~` = "will"), `?` conditional/uncertain ("might"), `!` certain ("must"). No suffix = present.
- Pronouns: `i`, `u`, `w`, `t`, `x` (it/this).
- Sigils: `@` entity (proper noun/identifier preserved verbatim), `#` concept, `$` action.
- Unknown words pass through as `@token`.

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

## Worked examples — prompt → tool sequence

**1. Summarize a large design doc**
```
User: "Here's a 2000-word design doc. Summarize it."
→ chimera_csm
→ chimera_mode(task_description="summarize design doc")           # → token mode
→ chimera_optimize(text=<doc>, preserve_code=false)               # 60-75% reduction
→ chimera_summarize(text=<compressed_doc>)                        # deterministic extractive
```

**2. Diagnose a failing CI log**
```
User: "Why did my CI fail?" + 500-line log attached
→ chimera_csm
→ chimera_log_compress(text=<log>)                                # errors verbatim, body abridged
→ answer from compressed log (no additional chimera call needed)
```

**3. Multi-perspective analysis**
```
User: "Analyze these 3 competing API designs"
→ chimera_csm
→ chimera_mode(task_description="analyze competing API designs")  # → agi mode
→ chimera_deliberate(topic=<question>, perspectives=[<A>,<B>,<C>])
→ chimera_gate(candidates=[<A_analysis>,<B_analysis>,<C_analysis>])
```

**4. Claim verification / hallucination check**
```
User: "Is this claim accurate?" + source text
→ chimera_csm
→ chimera_claims(text=<claim_text>)                               # extract atomic claims
→ chimera_verify(claims=[...], evidence=<source_text>)            # token-overlap scoring
→ chimera_detect(text=<claim_text>)                               # MCP attack / injection check
```

**5. AI-to-AI compressed internal reasoning**
```
User: "Reason through this internally then explain your answer"
→ chimera_glyph_directive(style="strict", task_hint=<task>)       # get CG system directive
→ [agent emits CG internally]
→ chimera_glyph_translate(glyph_text=<CG_output>, verbosity="natural")  # recover English
```

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

## Telemetry reaction matrix

Check `chimera_dashboard` after a run and respond to these signals:

| Signal | Threshold | Action |
|---|---|---|
| `avg_pct_saved` | < 30% | Inputs may be too short — review whether skip conditions apply; avoid firing chimera on sub-200-char prompts |
| `avg_pct_saved` | > 70% | Healthy — consider `chimera_cache_mark` on any stable system blocks for additional lossless savings |
| Dedup hits (`chimera_dedup_lookup`) | > 2 in a session | Wrap repeated identical calls in `chimera_batch`; same inputs should not re-fire |
| Session cost (`chimera_session_report`) | savings > $0.01 | Record for ROI evidence; the `dashboard` shows per-tool breakdown |
| `chimera_mode` returned `minimal` | — | Only 6 tools active; if the task grows, re-call `chimera_mode` with an updated description |
| `chimera_cost_track` errors | — | Fixed in `0.7.2`; on `< 0.7.2` the entry is persisted despite the error — verify via `chimera_dashboard` |

---

## Skip conditions (do not invoke chimera tools)

- Prompt is `< 200` chars and has no attachments.
- The user is debugging the chimeralang-mcp project itself (don't compress the very thing you're trying to read).
- The user is asking a meta question about the tools (definitions, schemas, "what does X do") — answer directly.
- The user has explicitly said "skip chimera" or "raw output only".

---

## Long tail — tools not in the main routing matrix

These 24 tools are real but situational. Use them when the primary matrix doesn't cover the need:

| Tool | When to reach for it |
|---|---|
| `chimera_audit` | Audit a ChimeraLang program for policy violations before running it. |
| `chimera_budget_lock` | Hard-cap the token budget for a session so no call can exceed it. |
| `chimera_confident` | Emit a confidence score for a single answer before committing to it. |
| `chimera_constrain` | Apply type constraints to a value (probabilistic type checking at runtime). |
| `chimera_cost_estimate` | Estimate token cost of a planned operation before executing it. |
| `chimera_cost_track` | Log actual tokens saved after a compression step (for dashboard). Note: returns a false `UnboundLocalError` but DOES persist the entry. |
| `chimera_dashboard` | Retrieve the session cost/savings summary mid-session (Stop hook shows totals automatically). |
| `chimera_embodied` | Model the agent's own physical/environmental context for situated reasoning. |
| `chimera_ethical_eval` | Run a structured ethical evaluation pass on a proposed action or output. |
| `chimera_evolve` | Iteratively refine a candidate answer through successive improvement passes. |
| `chimera_explore` | Breadth-first search across a knowledge/solution space before committing to one path. |
| `chimera_knowledge` | Query the embedded knowledge graph for factual grounding. |
| `chimera_materials` | Retrieve relevant reference materials (docs, examples) from the internal corpus. |
| `chimera_memory` | Persist a key fact across turns within the session (key-value store). |
| `chimera_meta_learn` | Apply few-shot meta-learning patterns from prior task examples. |
| `chimera_plan_goals` | Decompose a high-level goal into a ranked, dependency-ordered task list. |
| `chimera_policy` | Check an action against the active safety/behavior policy before executing. |
| `chimera_provenance_merge` | Merge multiple provenanced sources while preserving attribution chains. |
| `chimera_safety_check` | Run a targeted safety analysis on a code diff or LLM output. |
| `chimera_self_model` | Query the agent's own capability/uncertainty model for a given task type. |
| `chimera_social` | Model social context (roles, norms, relationship history) for conversational tasks. |
| `chimera_trace` | Emit a structured execution trace for debugging a multi-step chimera pipeline. |
| `chimera_transfer_learn` | Apply transfer-learning heuristics from a source task to the current target task. |
| `chimera_world_model` | Query or update the agent's world-state model (used in planning/`plan_goals` chains). |

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

Both issues from `0.7.1` were resolved in `0.7.2`:

1. ~~`chimera_cost_track` returns an `UnboundLocalError`~~ — **fixed in 0.7.2**: a local `log: list[str]` in the `chimera_optimize` handler shadowed the module-level logger; renamed to `passes_log` to free the name.
2. ~~`chimera_mode` accepts `task_type` but handler reads `task_description`~~ — **not actually a bug**: the schema declares `task_description`, matching the handler. The original `0.7.1` skill note was wrong.

---

## Codex compatibility

Codex does not auto-discover files inside `.claude/skills/`. The repo-root `AGENTS.md` (created alongside this file) contains the same routing matrix in Codex's preferred location so a Codex agent walking up from cwd will find it.

If you are inside Codex, treat this file's matrix and decision tree as binding and call the chimeralang-mcp tools through the same routing logic.
