# Axiom-TTT alongside chimeralang-mcp

This repo is configured to run [Axiom-TTT](https://github.com/fernandogarzaaa/AXIOM-AETHER)
as a second MCP server next to chimeralang-mcp, so a host LLM sees both tool
providers at once.

| Server | Command (`.mcp.json`) | Tools |
|--------|-----------------------|-------|
| `chimeralang` | `python -m chimeralang_mcp.server` | 51 `chimera_*` tools |
| `axiom` | `bash scripts/axiom_mcp.sh` | `axiom_compress_path`, `axiom_evaluate_drift` |

Axiom is a Rust **test-time-training** engine; in `--mode mcp` it exposes:
- **`axiom_compress_path`** — absorbs a directory/file through local TTT and
  returns an `<axiom_context_fingerprint>` (compress-then-reference).
- **`axiom_evaluate_drift`** — cross-entropy of code vs the current fast-weights;
  a spike past the drift gate (default `7.03`) returns `isError: true`.

## Provisioning (required once per environment)

This Claude Code environment is **ephemeral** — the compiled Axiom binary does
not survive a container reclaim. Provision it with:

```bash
scripts/setup_axiom.sh        # clones + builds Axiom into ~/AXIOM-AETHER
```

`setup_axiom.sh` is idempotent, needs Rust/cargo, and applies a small upstream
build fix (the `axiom_engine` binary's `main.rs` is missing `mod model_meta;`,
which `inference.rs` references — it builds the library but not the binary
without it). For a durable setup, add `scripts/setup_axiom.sh` to the
environment's **setup-script** configuration so it runs on container start.

The launcher (`scripts/axiom_mcp.sh`) finds the binary via `$AXIOM_BIN`,
`$AXIOM_HOME` (default `~/AXIOM-AETHER`), or `/tmp/AXIOM-AETHER`, and runs from
`~/.axiom-state` so Axiom's runtime files (vibe memory, checkpoints) never land
in the repo working tree.

## Notes

- **No checkpoint = baseline weights.** With no trained model present, Axiom
  boots on random weights and a hash tokenizer; the MCP tools work but the drift
  gate is **uncalibrated**. For calibrated drift, train a model
  (`train_tokenizer` + `train_semantic`) and set `AXIOM_PRODUCTION_BPE=1` plus a
  `--checkpoint`.
- **Proxy mode is not used here.** Axiom's context-compression proxy points at
  `api.anthropic.com`, which this environment's network policy blocks. The MCP
  (stdio) integration above is the relevant one.
- **Env knobs:** `AXIOM_BIN`, `AXIOM_HOME`, `AXIOM_STATE`, `AXIOM_VIBE=0`
  (disable vibe persistence), `AXIOM_DRIFT_THRESHOLD`.
