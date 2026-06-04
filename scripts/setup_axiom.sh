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

# Upstream build fix: the axiom_engine binary's module tree (main.rs) is missing
# `mod model_meta;`, which inference.rs references via crate::model_meta. The
# library declares it but the binary does not, so a fresh `cargo build` fails
# with E0433. Inject it if absent (mirrors lib.rs). Remove once fixed upstream.
MAIN="$AXIOM_HOME/axiom_engine_rs/src/main.rs"
if [[ -f "$MAIN" ]] && ! grep -q '^mod model_meta;' "$MAIN"; then
  sed -i 's/^mod model;$/mod model;\nmod model_meta;/' "$MAIN"
  echo "[setup_axiom] applied model_meta module fix to main.rs"
fi

echo "[setup_axiom] building release binary (this can take several minutes)..."
( cd "$AXIOM_HOME/axiom_engine_rs" && cargo build --release --bin axiom_engine )
echo "[setup_axiom] done: $BIN"
