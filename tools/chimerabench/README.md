# ChimeraBench

A reproducible benchmark for the chimeralang-mcp gate-tool stack. Every task is a JSON file; every step's `provenance.program_hash` is recorded as the canonical identity. Anyone running the harness on any machine should get the same hashes.

## Why

After Phase 1 of `BLUEPRINT.md` falsified Glyph's token-cost claim, the project's headline became reproducibility, not compression. Phase 2 made every gate-tool call cryptographically replayable via the `# CHIMERA_REPLAY_v1` envelope. ChimeraBench is what turns that into a public benchmark: a corpus of agent tasks whose canonical answers are SHA-256 hashes, not opaque pass/fail.

## Run

```bash
# Run the full corpus
python -m tools.chimerabench.run

# Verbose: print per-step hashes
python -m tools.chimerabench.run --verbose

# Run only one family
python -m tools.chimerabench.run --filter mixed-pipeline

# Regenerate canonical hashes (use after intentional gate-tool changes)
python -m tools.chimerabench.run --update
```

The harness exits with code 0 on full pass, code 1 on any failure. `tests/test_chimerabench.py` runs the full corpus from pytest and asserts every step's hash matches.

## Task families

Five families, three tasks each (15 total in the v1 corpus):

| Family | Purpose |
|---|---|
| `gate-only` | Pure consensus over pre-supplied candidates. Smallest unit. |
| `verify-pipeline` | Claim verification against evidence. Lexical overlap scoring. |
| `deliberate-converge` | Multi-perspective deliberation. Lexical-consensus mode. |
| `vote-converge` | Confidence-weighted vote across candidate answers. |
| `mixed-pipeline` | Multi-step chains (gate → verify, verify → gate, etc.). |

Every task uses tools from `chimeralang_mcp.replay.REPLAYABLE_TOOLS` so each step has a `provenance.program_hash`.

## Task schema

```json
{
  "id":          "unique-slug",
  "family":      "<one of the families above>",
  "description": "human-readable purpose",
  "pipeline": [
    {"tool": "chimera_gate", "args": { ... }}
  ],
  "expected": {
    "program_hashes": ["<sha256>", "..."],
    "assertions": [
      {"step": 0, "path": "value",  "equals": "..."},
      {"step": 0, "path": "passed", "equals": true}
    ]
  },
  "metadata": { "author": "...", "purpose": "..." }
}
```

`path` uses dotted notation (`foo.bar.0`) to address nested values.

## Adding a task

1. Pick (or add) a family directory under `tasks/`.
2. Write the task JSON with empty `program_hashes` and your assertions.
3. Run `python -m tools.chimerabench.run --update` to bless the canonical hashes.
4. Run `python -m tools.chimerabench.run` to confirm reproducibility.
5. Commit the JSON. Future hash changes will fail the regression test loudly.

## What this proves

For every task in this corpus, **the same `(tool, args)` produces the same SHA-256 hash on any machine running the same chimeralang-mcp version**. That's the reproducibility primitive Phase 4 (cross-agent protocol) builds on: agent A and agent B don't need to trust each other, only the hash.

If a future change to a gate-tool's logic alters its output for the same input, the corresponding `program_hash` changes and this benchmark fails — surfacing the change visibly instead of silently shifting downstream behavior.
