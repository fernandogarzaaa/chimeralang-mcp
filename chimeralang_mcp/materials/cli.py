from __future__ import annotations

import argparse
import json
from typing import Sequence

from .builders import build_external_packs, sync_source_metadata
from .loader import MaterialRegistry


def run_materials_cli(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="chimeralang-mcp", description="ChimeraLang MCP materials commands")
    parser.add_argument("command", choices=["sync", "build", "status", "licenses"])
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional base directory for generated material artifacts. Defaults to CHIMERA_MCP_DATA_DIR or ~/.chimeralang_mcp.",
    )
    args = parser.parse_args(list(argv))

    registry = MaterialRegistry(args.output_dir)
    if args.command == "sync":
        payload = sync_source_metadata(registry.base_dir)
    elif args.command == "build":
        payload = build_external_packs(registry.base_dir)
    elif args.command == "licenses":
        payload = registry.licenses()
    else:
        payload = registry.status()

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0
