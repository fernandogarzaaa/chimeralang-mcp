from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .builders import (
    build_core_pack,
    build_license_report,
    build_source_manifest,
    build_status_report,
)


class MaterialRegistry:
    def __init__(self, base_dir: str | None = None) -> None:
        root = base_dir or os.environ.get("CHIMERA_MCP_DATA_DIR")
        self.base_dir = Path(root) if root else Path.home() / ".chimeralang_mcp"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = build_source_manifest()
        self._core_pack = build_core_pack(self._manifest)
        self._license_report = build_license_report(self._manifest, self._core_pack)

    @property
    def pack_version(self) -> str:
        return str(self._core_pack["pack_version"])

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def core_pack(self) -> dict[str, Any]:
        return self._core_pack

    def list_packs(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for pack_type, records in self._core_pack["packs"].items():
            source_ids = sorted({sid for record in records for sid in record.get("source_ids", [])})
            items.append(
                {
                    "pack_type": pack_type,
                    "pack_version": self.pack_version,
                    "record_count": len(records),
                    "runtime_record_count": sum(
                        1 for record in records if record.get("bundle_scope") == "runtime"
                    ),
                    "source_ids": source_ids,
                }
            )
        return items

    def status(self) -> dict[str, Any]:
        return build_status_report(self.base_dir, self._manifest, self._core_pack)

    def licenses(self) -> dict[str, Any]:
        return self._license_report

    def source_manifest(self) -> dict[str, Any]:
        return self._manifest

    def pack(self, pack_type: str) -> list[dict[str, Any]]:
        return list(self._core_pack["packs"].get(pack_type, []))

    def policy_pattern(self, policy_name: str) -> dict[str, Any] | None:
        for item in self.pack("policy_patterns"):
            if item.get("name") == policy_name:
                return item
        return None

    def classify_claim(self, text: str) -> dict[str, Any]:
        lowered = text.lower().strip()
        risk_tags: list[str] = []
        hedged = any(marker in lowered for marker in self._core_pack["lexicons"]["hedge_markers"])
        abstained = any(marker in lowered for marker in self._core_pack["lexicons"]["abstention_markers"])
        has_citation = any(marker in lowered for marker in self._core_pack["lexicons"]["citation_markers"])
        has_temporal = bool(re.search(r"\b\d{4}\b|\btoday\b|\byesterday\b|\btomorrow\b|\bthis year\b", lowered))
        has_numeric = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", lowered))
        attack_flags = self.find_attack_matches(text)
        security_sensitive = bool(attack_flags) or any(
            marker in lowered for marker in self._core_pack["lexicons"]["security_terms"]
        )

        if has_citation:
            claim_type = "citation"
        elif security_sensitive:
            claim_type = "security_sensitive"
        elif has_temporal:
            claim_type = "temporal"
        elif has_numeric:
            claim_type = "numeric"
        else:
            claim_type = "factual"

        if hedged:
            risk_tags.append("hedged")
        if abstained:
            risk_tags.append("abstention")
        if has_temporal:
            risk_tags.append("temporal")
        if has_numeric:
            risk_tags.append("numeric")
        if has_citation:
            risk_tags.append("citation")
        if security_sensitive:
            risk_tags.append("security_sensitive")
        for flag in attack_flags:
            category = str(flag.get("category", "")).strip()
            if category and category not in risk_tags:
                risk_tags.append(category)

        atomic_parts = self.atomic_claim_parts(text)
        return {
            "claim_type": claim_type,
            "risk_tags": risk_tags,
            "hedged": hedged,
            "abstained": abstained,
            "atomic_parts": atomic_parts,
            "attack_flags": attack_flags,
            "pack_version": self.pack_version,
        }

    def atomic_claim_parts(self, text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text.strip())
        parts = [
            part.strip(" ,;")
            for part in re.split(r"\b(?:and|but|while)\b|;", cleaned, flags=re.IGNORECASE)
            if part.strip(" ,;")
        ]
        return parts or [cleaned]

    def find_attack_matches(self, text: str) -> list[dict[str, Any]]:
        lowered = text.lower()
        matches: list[dict[str, Any]] = []
        for record in self.pack("attack_patterns"):
            hits = [term for term in record.get("match_terms", []) if term.lower() in lowered]
            if not hits:
                continue
            matches.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "severity": record["severity"],
                    "description": record["description"],
                    "matched_terms": hits[:5],
                    "owasp_refs": record.get("owasp_refs", []),
                    "source_ids": record.get("source_ids", []),
                }
            )
        return matches

    def security_category_counts(self, flags: list[dict[str, Any]]) -> dict[str, int]:
        counts = Counter(str(flag.get("category", "unknown")) for flag in flags)
        return dict(counts)

    def material_usage(
        self,
        pack_types: list[str],
        *,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        resolved_sources = set(source_ids or [])
        materials_used: list[dict[str, Any]] = []
        pack_versions: dict[str, str] = {}
        for pack_type in pack_types:
            records = self.pack(pack_type)
            for record in records:
                resolved_sources.update(record.get("source_ids", []))
            materials_used.append(
                {
                    "pack_type": pack_type,
                    "pack_version": self.pack_version,
                    "record_count": len(records),
                }
            )
            pack_versions[pack_type] = self.pack_version
        return {
            "materials_used": materials_used,
            "pack_versions": pack_versions,
            "source_ids": sorted(resolved_sources),
        }


_REGISTRY_CACHE: dict[str, MaterialRegistry] = {}


def get_material_registry(base_dir: str | None = None, refresh: bool = False) -> MaterialRegistry:
    root = base_dir or os.environ.get("CHIMERA_MCP_DATA_DIR") or str(Path.home() / ".chimeralang_mcp")
    if refresh or root not in _REGISTRY_CACHE:
        _REGISTRY_CACHE[root] = MaterialRegistry(root)
    return _REGISTRY_CACHE[root]
