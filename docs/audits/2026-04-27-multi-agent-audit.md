# ChimeraLang MCP Multi-Agent Audit

Date: 2026-04-27
Scope: `chimera/`, `chimeralang_mcp/`, `tests/`, packaging metadata.

## Agent Topology

This audit was executed as a coordinated 4-agent pass:

1. **Agent A — Test & Runtime Validation**
   - Goal: verify baseline behavior and test health.
   - Command: `PYTHONPATH=. pytest -q`

2. **Agent B — Dependency & Packaging Integrity**
   - Goal: detect broken dependencies and packaging drift.
   - Command: `python -m pip check`

3. **Agent C — Security Static Analysis**
   - Goal: identify suspicious patterns and known risky code constructs.
   - Command: `bandit -q -r chimera chimeralang_mcp -f txt`

4. **Agent D — Manual Code Risk Review**
   - Goal: review design hotspots not fully captured by automated scanners.
   - Inputs: token counting engine, VM randomness, server token usage paths.

## Results Summary

### 1) Test & Runtime Validation (Agent A)

- Result: **PASS**
- Details: all tests passed when run with repository root on `PYTHONPATH`.
- Observed caveat: plain `pytest -q` fails collection unless the package is installed or `PYTHONPATH` includes project root.

### 2) Dependency & Packaging Integrity (Agent B)

- Result: **PASS**
- Details: `pip check` reported no broken requirements.

### 3) Security Static Analysis (Agent C)

- Result: **LOW-RISK FINDINGS ONLY**
- Bandit reported 7 low-severity issues:
  - `B311` for deterministic PRNG usage in `chimera/vm.py` (likely intentional for reproducibility).
  - Multiple `B105` flags in `chimeralang_mcp/token_engine.py` caused by string literals (`"api"`, `"estimate"`) that are not credentials in context.

Interpretation: no medium/high findings; current output is mostly scanner noise + one design-choice warning.

### 4) Manual Code Risk Review (Agent D)

#### Finding D1: Unbounded token-count cache growth

- `TokenBudgetManager` stores token counts in an in-memory dictionary keyed by SHA-256 hashes.
- No TTL or maximum size exists.
- Long-running MCP sessions or high-cardinality prompts can cause unbounded memory growth.

Risk: **Operational reliability risk (memory pressure over uptime).**

Recommendation:
- Add a bounded LRU cache (e.g., max entries configurable via env var).
- Optionally add periodic metrics/logging and cache reset hooks.

#### Finding D2: Broad exception fallbacks hide failure modes

- `count_tokens` and `count_messages` catch broad exceptions and silently switch to estimate mode.
- This improves resilience but can mask persistent upstream API failures.

Risk: **Observability and correctness drift risk.**

Recommendation:
- Keep fallback, but emit structured warning logs with reason and cooldown/once-per-window behavior.
- Track fallback activation counters exposed through diagnostics.

#### Finding D3: Test entrypoint ergonomics

- Tests depend on `PYTHONPATH=.`, indicating local invocation can fail in fresh environments.

Risk: **Developer friction / CI portability risk.**

Recommendation:
- Prefer `python -m pytest -q` after editable install, or add a test runner target (e.g., `make test`) that sets environment correctly.

## Prioritized Actions

1. **P1:** Add bounded cache policy to `TokenBudgetManager`.
2. **P2:** Improve fallback observability for token counting API errors.
3. **P3:** Standardize test command in docs/CI to avoid import-path surprises.

## Overall Assessment

- **Security posture:** acceptable for current alpha stage; no high-severity static findings.
- **Reliability posture:** moderate risk due to unbounded cache and silent fallback behavior.
- **Maintainability posture:** good test coverage signal (30 passing tests), with minor invocation ergonomics issue.
