from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ResultEnvelope:
    envelope_version: str = "1.0"
    envelope_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    kind: str = "generic"
    value: Any = None
    confidence: float = 0.0
    confidence_source: str = ""
    provenance: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    transform_history: list[dict[str, Any]] = field(default_factory=list)
    constraints_applied: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResultEnvelope":
        return cls(
            envelope_version=str(payload.get("envelope_version", "1.0")),
            envelope_id=str(payload.get("envelope_id", str(uuid.uuid4())[:12])),
            kind=str(payload.get("kind", "generic")),
            value=payload.get("value"),
            confidence=float(payload.get("confidence", 0.0)),
            confidence_source=str(payload.get("confidence_source", "")),
            provenance=list(payload.get("provenance", [])),
            sources=list(payload.get("sources", [])),
            transform_history=list(payload.get("transform_history", [])),
            constraints_applied=list(payload.get("constraints_applied", [])),
            claims=list(payload.get("claims", [])),
            warnings=list(payload.get("warnings", [])),
            metadata=dict(payload.get("metadata", {})),
            created_at=float(payload.get("created_at", time.time())),
        )

    @classmethod
    def coerce(
        cls,
        payload: Any,
        *,
        kind: str = "generic",
        confidence: float = 0.0,
        confidence_source: str = "",
        provenance: list[dict[str, Any]] | None = None,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ResultEnvelope":
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict) and "envelope_id" in payload and "value" in payload:
            return cls.from_dict(payload)
        return cls(
            kind=kind,
            value=payload,
            confidence=confidence,
            confidence_source=confidence_source,
            provenance=list(provenance or []),
            sources=list(sources or []),
            metadata=dict(metadata or {}),
        )

    def add_provenance(self, step: str, **details: Any) -> None:
        self.provenance.append({"step": step, "timestamp": time.time(), **details})

    def add_transform(self, step: str, **details: Any) -> None:
        self.transform_history.append({"step": step, "timestamp": time.time(), **details})

    def add_constraint(self, constraint: str, passed: bool, **details: Any) -> None:
        self.constraints_applied.append(
            {"constraint": constraint, "passed": passed, "timestamp": time.time(), **details}
        )

    def with_claims(self, claims: list[dict[str, Any]]) -> "ResultEnvelope":
        self.claims = claims
        return self


def merge_envelopes(
    envelopes: list[ResultEnvelope],
    *,
    strategy: str = "weighted",
    merge_value_mode: str = "list",
) -> ResultEnvelope:
    if not envelopes:
        return ResultEnvelope(kind="merged", value=[], confidence=0.0)

    if strategy == "max":
        confidence = max(env.confidence for env in envelopes)
    elif strategy == "mean":
        confidence = sum(env.confidence for env in envelopes) / len(envelopes)
    else:
        total_weight = sum(max(env.confidence, 0.001) for env in envelopes)
        confidence = (
            sum(env.confidence * max(env.confidence, 0.001) for env in envelopes) / total_weight
            if total_weight
            else 0.0
        )

    if merge_value_mode == "first":
        value: Any = envelopes[0].value
    elif merge_value_mode == "consensus":
        counts: dict[str, int] = {}
        for env in envelopes:
            key = repr(env.value)
            counts[key] = counts.get(key, 0) + 1
        winner = max(counts, key=counts.get)
        value = next(env.value for env in envelopes if repr(env.value) == winner)
    else:
        value = [env.value for env in envelopes]

    merged = ResultEnvelope(
        kind="merged",
        value=value,
        confidence=round(confidence, 4),
        confidence_source=f"merge:{strategy}",
        provenance=[p for env in envelopes for p in env.provenance],
        sources=[s for env in envelopes for s in env.sources],
        transform_history=[t for env in envelopes for t in env.transform_history],
        constraints_applied=[c for env in envelopes for c in env.constraints_applied],
        claims=[c for env in envelopes for c in env.claims],
        warnings=[w for env in envelopes for w in env.warnings],
        metadata={"merged_from": [env.envelope_id for env in envelopes], "merge_value_mode": merge_value_mode},
    )
    merged.add_transform("provenance_merge", strategy=strategy, envelope_count=len(envelopes))
    return merged
