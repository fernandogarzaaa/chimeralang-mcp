from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


class PersistentNamespaceStore:
    def __init__(self, base_dir: str | None = None) -> None:
        root = base_dir or os.environ.get("CHIMERA_MCP_DATA_DIR")
        self._base_dir = Path(root) if root else Path.home() / ".chimeralang_mcp"
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def load(self, kind: str, namespace: str, default: Any) -> Any:
        path = self._path(kind, namespace)
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def save(self, kind: str, namespace: str, payload: Any) -> str:
        path = self._path(kind, namespace)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        tmp.replace(path)
        return str(path)

    def append(self, kind: str, namespace: str, entry: Any, max_items: int = 200) -> str:
        items = self.load(kind, namespace, [])
        if not isinstance(items, list):
            items = []
        items.append(entry)
        if len(items) > max_items:
            items = items[-max_items:]
        return self.save(kind, namespace, items)

    def path_for(self, kind: str, namespace: str) -> str:
        return str(self._path(kind, namespace))

    def _path(self, kind: str, namespace: str) -> Path:
        safe_kind = self._sanitize(kind)
        safe_namespace = self._sanitize(namespace or "default")
        folder = self._base_dir / safe_kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{safe_namespace}.json"

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
        return cleaned or "default"
