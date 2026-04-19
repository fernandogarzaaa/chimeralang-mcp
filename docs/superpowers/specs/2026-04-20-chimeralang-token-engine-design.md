# ChimeraLang Token Engine Redesign

**Date:** 2026-04-20
**Status:** Approved — pending implementation
**Target:** `chimeralang-mcp-v2` (local) → PyPI release

---

## Goal

Directly influence how Claude uses tokens and cut inference cost by:

1. Fixing silent bugs in existing tools
2. Replacing char-based token estimation with exact Anthropic API counts
3. Adding a shared internal engine (`token_engine.py`) for budget management and importance scoring
4. Shipping two new tools: `chimera_budget` (proactive monitoring) and `chimera_score` (importance audit)
5. Adding opt-in lossy compression via `MessageImportanceScorer`

---

## Architecture

### New file: `chimeralang_mcp/token_engine.py`

Single shared module — not exposed as MCP tools. Two classes:

#### `TokenBudgetManager` (singleton)

- Lazy-initializes `anthropic.Anthropic` client from `ANTHROPIC_API_KEY` env var
- `count_tokens(text: str) -> int` — calls `client.beta.messages.count_tokens`
- `count_messages(messages: list[dict]) -> int` — counts full message list in one API call
- Caches results keyed on SHA-256 of content — identical text never counted twice per session
- Fallback: if API key absent or call times out (>2s), returns `len(text) // 4` and sets `token_count_method: "estimate"` in the response

#### `MessageImportanceScorer` (stateless, pure)

Scores each message on 4 axes (0.0–1.0 each, averaged):

| Axis | High score | Low score |
|---|---|---|
| Recency | Last 3 messages | First messages |
| Content type | `tool_result`, code fence | Long assistant prose |
| Information density | Low tokens-per-unique-word (diverse vocabulary) | High tokens-per-unique-word (repetitive filler) |
| Replaceability | User question (irreplaceable) | Assistant preamble |

Returns list of `{index, role, score, reason}` sorted ascending — lowest score dropped first in lossy mode.

### Module layout after redesign

```
chimeralang_mcp/
├── server.py          # tool handlers — thinner, delegates to token_engine
├── token_engine.py    # TokenBudgetManager + MessageImportanceScorer (new)
```

---

## Tool Changes

### `chimera_optimize` — 3 fixes

**Bug fix — `collapse_lists`:**
Current implementation passes `lambda m: m.group(0)` as regex replacement — a no-op that returns the original match unchanged. Fix: collect bullet items within each matched block, deduplicate by normalized text, re-emit as inline `• item1 • item2` string. If only one unique item remains, emit it without the bullet prefix.

**New param — `preserve_code: bool = True`:**
Stash all ` ``` ` fenced blocks before any regex pass, restore after. Identical to the pattern already used in `chimera_compress`. Prevents optimization passes from mangling code syntax.

**Real token count:**
`estimated_tokens_saved` = `TokenBudgetManager.count_tokens(original_text) - count_tokens(result_text)`. Falls back to `saved // 4` if API unavailable.

**Input schema additions:**
```json
"preserve_code": { "type": "boolean", "default": true }
```

---

### `chimera_compress` — 2 fixes + 1 addition

**Fix — remove harmful aggressive symbols:**
Drop these substitutions from `_SYMBOLS_AGGRESSIVE` — they save ~1 char each but corrupt Claude's ability to parse its own compressed history:
- `therefore → ∴`
- `because → ∵`
- `approximately → ≈`
- `and → &`
- `with → w/`
- `without → w/o`
- `number → nr.`

Keep only unambiguous, universally readable abbreviations: `versus → vs.`, `for example → e.g.`, `etcetera → etc.`

**Fix — real token count:**
`estimated_tokens_saved` and `compression_ratio` computed via `TokenBudgetManager`.

**Addition — `allow_lossy: bool = False`:**
When `True` and a `messages` list is provided alongside `text`, invokes `MessageImportanceScorer` to rank and drop lowest-scoring messages until the `max_tokens` budget is met. Each dropped message is replaced with a tombstone:
```json
{"role": "system", "content": "[3 messages omitted — importance scores: 0.21, 0.18, 0.14]"}
```
Stops dropping if fewer than 2 messages would remain; returns `budget_not_met: true` with explanation.

**Input schema additions:**
```json
"allow_lossy": { "type": "boolean", "default": false },
"messages": {
  "type": "array",
  "description": "Optional message list for lossy importance-scored dropping",
  "items": { "type": "object", "properties": { "role": {"type":"string"}, "content": {"type":"string"} } }
},
"max_tokens": { "type": "integer", "default": 3000 }
```

---

### `chimera_fracture` — true pipeline (v2 canonical)

The local v2 implementation currently just splits text into fragments — diverging from the PyPI version. This redesign makes v2 the canonical full-pipeline implementation.

**New interface:**
```
messages: list[{role, content}]     # conversation to compress
documents: list[str]                # docs to optimize (optional)
token_budget: int = 3000            # target total token budget
optimize_ratio: float = 0.05        # per-doc target ratio
allow_lossy: bool = False           # enable importance-scored message dropping
query: str = ""                     # metadata only
```

**Pipeline:**
1. `chimera_optimize` each document (`preserve_code=True`)
2. `chimera_compress` messages losslessly
3. `TokenBudgetManager.count_messages(compressed_messages)` — exact count
4. Budget gate: if under `token_budget` → `quality_passed=True`, done
5. If over budget and `allow_lossy=False` → `quality_passed=False`, suggest `allow_lossy=True`
6. If over budget and `allow_lossy=True` → `MessageImportanceScorer` drops lowest messages → recount → loop until met or min 2 messages

**Output additions:**
```json
"lossy_dropped_count": 0,
"budget_not_met": false,
"token_count_method": "exact"
```

Old fragment-splitting behavior (text → numbered fragments) moves to an internal helper. No external tool exposes it — it had no direct callers.

---

## New Tools

### `chimera_budget`

**Purpose:** Proactive token position monitor. Claude calls this at the start of heavy tasks to know exactly where it stands and receive an action recommendation.

**Input:**
```json
{
  "messages": [{"role": "...", "content": "..."}],
  "max_tokens": 200000,
  "reserve_tokens": 10000
}
```

**Output:**
```json
{
  "used_tokens": 42300,
  "remaining_tokens": 147700,
  "pct_used": 21.2,
  "status": "ok",
  "recommendation": "ok",
  "token_count_method": "exact",
  "thresholds": { "warn": 0.70, "critical": 0.85 }
}
```

**Status rules:**
| `pct_used` | `status` | `recommendation` |
|---|---|---|
| < 70% | `ok` | `"ok"` |
| 70–85% | `warn` | `"call chimera_compress"` |
| > 85% | `critical` | `"call chimera_fracture with allow_lossy=true"` |

---

### `chimera_score`

**Purpose:** Expose `MessageImportanceScorer` directly. Transparency tool — shows Claude (and the user) exactly which messages would be dropped in lossy mode and why, before committing.

**Input:**
```json
{
  "messages": [{"role": "...", "content": "..."}]
}
```

**Output:**
```json
[
  { "index": 0, "role": "user", "score": 0.91, "reason": "user_question" },
  { "index": 1, "role": "assistant", "score": 0.31, "reason": "verbose_prose" },
  { "index": 2, "role": "assistant", "score": 0.87, "reason": "code_fence" }
]
```

Sorted ascending by score. Lowest scores are what lossy mode drops first.

---

## Data Flow

### Lossless (default)
```
chimera_fracture(messages, documents, token_budget)
  → optimize each doc (preserve_code=True)
  → lossless compress messages
  → TokenBudgetManager.count_messages → exact count
  → under budget? → quality_passed=True, return
  → over budget? → quality_passed=False, suggest allow_lossy=True
```

### Lossy (opt-in)
```
chimera_fracture(..., allow_lossy=True)
  → [same lossless pipeline]
  → over budget →
      MessageImportanceScorer.rank(messages)
      drop lowest-score → insert tombstone → recount
      loop until budget met OR messages < 2
  → return with lossy_dropped_count > 0
```

### Error handling

| Condition | Behavior |
|---|---|
| `ANTHROPIC_API_KEY` absent | Fallback `len//4`, `token_count_method: "estimate"` |
| Anthropic API timeout >2s | Same fallback |
| `allow_lossy=True` but dropping leaves < 2 messages | Stop, return `budget_not_met: true` |
| Malformed message (missing `role`/`content`) | Skip, include `skipped_count` in stats |
| `chimera_score` on empty list | Return `[]`, not an error |

---

## Testing

### `tests/test_token_engine.py`
- Cache hit: second identical call does not invoke Anthropic client (mock)
- Fallback: `ANTHROPIC_API_KEY` unset → returns `len//4`, method is `"estimate"`
- Scorer: `tool_result` message scores higher than same-length verbose prose
- Scorer: most recent message scores higher than first message of equal content type

### `tests/test_tools.py`
- `chimera_optimize`: `collapse_lists` deduplicates; code fence survives with `preserve_code=True`; estimated tokens use real count
- `chimera_compress`: aggressive mode emits no `∴` or `&`; lossless preserves all messages; lossy tombstone present with correct omission count
- `chimera_fracture`: `quality_passed=True` under budget; `lossy_dropped_count > 0` when forced lossy; `budget_not_met=True` when impossible
- `chimera_budget`: returns `warn` at 72% usage, `critical` at 87%, `ok` at 50%
- `chimera_score`: lowest-score entry is verbose assistant reply; user question has score > 0.85

### `tests/test_pypi_parity.py`
Smoke test: install current PyPI version in a subprocess, run identical inputs against v2, confirm output shape keys match. Guards against regressions for existing callers after PyPI release.

---

## PyPI Release Plan

1. Bump version in `pyproject.toml` / `setup.cfg`
2. Add `anthropic` to `install_requires` (already a transitive dep — make it explicit)
3. `python -m build && twine upload dist/*`
4. Verify `pip install --upgrade chimeralang-mcp` picks up new version
5. Run `test_pypi_parity.py` against freshly installed package
