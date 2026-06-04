#!/usr/bin/env bash
# Launch the Axiom-TTT native MCP server (JSON-RPC stdio) so it can run as a
# tool provider alongside chimeralang-mcp. Referenced by .mcp.json.
#
# stdout MUST stay a pure JSON-RPC channel; Axiom sends all diagnostics to
# stderr, so we do not print anything to stdout here.
set -euo pipefail

AXIOM_HOME="${AXIOM_HOME:-$HOME/AXIOM-AETHER}"

# Resolve the built binary: explicit override first, then known locations.
BIN="${AXIOM_BIN:-}"
if [[ -z "$BIN" ]]; then
  for cand in \
    "$AXIOM_HOME/axiom_engine_rs/target/release/axiom_engine" \
    "$HOME/AXIOM-AETHER/axiom_engine_rs/target/release/axiom_engine" \
    "/tmp/AXIOM-AETHER/axiom_engine_rs/target/release/axiom_engine"; do
    if [[ -x "$cand" ]]; then BIN="$cand"; break; fi
  done
fi

if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "[axiom_mcp] axiom_engine binary not found. Run scripts/setup_axiom.sh first." >&2
  exit 1
fi

# Keep Axiom's runtime state (vibe memory, checkpoints) out of the repo working
# tree by running from a dedicated state directory.
STATE="${AXIOM_STATE:-$HOME/.axiom-state}"
mkdir -p "$STATE"
cd "$STATE"

# --mode mcp exposes axiom_compress_path + axiom_evaluate_drift. With no
# checkpoint present it boots on baseline weights (drift gate uncalibrated);
# set AXIOM_PRODUCTION_BPE=1 + a trained checkpoint for calibrated drift.
exec "$BIN" --mode mcp "$@"
