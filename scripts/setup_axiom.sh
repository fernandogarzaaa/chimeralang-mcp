#!/usr/bin/env bash
# Provision the Axiom-TTT engine (https://github.com/fernandogarzaaa/AXIOM-AETHER)
# for use as an MCP server alongside chimeralang-mcp.
#
# This clones + builds the Rust binary into $AXIOM_HOME (default ~/AXIOM-AETHER).
# Idempotent: skips the build if the binary already exists. Because this Claude
# Code environment is ephemeral, run this once per fresh container (ideally from
# the environment's setup-script config) before launching the MCP servers.
set -euo pipefail

AXIOM_HOME="${AXIOM_HOME:-$HOME/AXIOM-AETHER}"
AXIOM_REPO="${AXIOM_REPO:-https://github.com/fernandogarzaaa/AXIOM-AETHER}"
BIN="$AXIOM_HOME/axiom_engine_rs/target/release/axiom_engine"

if [[ -x "$BIN" ]]; then
  echo "[setup_axiom] already built: $BIN"
  exit 0
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "[setup_axiom] ERROR: Rust/cargo not found. Install via https://rustup.rs" >&2
  exit 1
fi

if [[ ! -d "$AXIOM_HOME/.git" ]]; then
  echo "[setup_axiom] cloning $AXIOM_REPO -> $AXIOM_HOME"
  git clone --depth 1 "$AXIOM_REPO" "$AXIOM_HOME"
fi

# NOTE: a fresh AXIOM-AETHER build of the `axiom_engine` binary currently fails
# with E0433 because axiom_engine_rs/src/main.rs is missing `mod model_meta;`
# (inference.rs references crate::model_meta; the library declares it, the binary
# does not). That fix belongs UPSTREAM in AXIOM-AETHER, not here — see
# docs/axiom-integration.md. This script does not patch upstream source; if the
# build fails on that error, apply the upstream one-liner first.
echo "[setup_axiom] building release binary (this can take several minutes)..."
if ! ( cd "$AXIOM_HOME/axiom_engine_rs" && cargo build --release --bin axiom_engine ); then
  echo "[setup_axiom] build failed. If the error is E0433 'crate::model_meta', the" >&2
  echo "  AXIOM-AETHER repo needs 'mod model_meta;' added to axiom_engine_rs/src/main.rs" >&2
  echo "  (see docs/axiom-integration.md)." >&2
  exit 1
fi
echo "[setup_axiom] done: $BIN"
