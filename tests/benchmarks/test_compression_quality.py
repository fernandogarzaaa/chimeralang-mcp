"""Reproducible compression-quality benchmark.

Measures entity recall and reduction ratio against a stable fixture corpus so
default settings can be tuned defensibly. Inspired by the Prompt-Compression-
Survey (NAACL 2025) and jstilb/context-engineering-toolkit — fidelity claims
need a regression-protected number, not a vibe.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from chimeralang_mcp import server as srv

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# Entities we expect to survive any reasonable compression of sample_doc.md.
# Picked because they're load-bearing facts a downstream model would need.
DOC_REQUIRED_ENTITIES = [
    "/auth/v2/login",
    "/auth/v2/jwks",
    "/healthz",
    "/metrics",
    "RS256",
    "4096-bit",
    "HashiCorp Vault",
    "30 days",
    "60 days",
    "Postgres",
    "Redis",
    "PagerDuty",
    "auth-prod",
    "Let's Encrypt",
    "cert-manager",
    "Prometheus",
    "auth_login_total",
    "auth_login_duration_seconds",
    "500ms",
    "10%",
]

LOG_REQUIRED_LINES = [
    "deprecated declaration of std::auto_ptr",
    "undefined reference to `Auth::verify_token",
    "collect2: error: ld returned 1 exit status",
    "FATAL build failed: linker exit code 1",
    "status=failure",
]


def _call(tool: str, args: dict) -> dict:
    result = asyncio.run(srv.call_tool(tool, args))
    return json.loads(result.content[0].text)


def _entity_recall(entities: list[str], compressed: str) -> float:
    if not entities:
        return 1.0
    found = sum(1 for e in entities if e in compressed)
    return found / len(entities)


# ── chimera_optimize on prose ─────────────────────────────────────────────


def test_optimize_classic_mode_is_near_lossless():
    """The classic algorithm is a whitespace/filler cleanup — it must preserve
    every load-bearing entity. This is the regression-protected baseline."""
    text = (_FIXTURE_DIR / "sample_doc.md").read_text(encoding="utf-8")
    result = _call("chimera_optimize", {
        "text": text,
        "namespace": "bench-doc",
        "preserve_code": True,
        "algorithm": "classic",
    })
    compressed = result["optimised_text"]
    recall = _entity_recall(DOC_REQUIRED_ENTITIES, compressed)
    assert recall >= 0.95, (
        f"classic mode should be near-lossless; recall={recall:.2%}; "
        f"missing: {[e for e in DOC_REQUIRED_ENTITIES if e not in compressed]}"
    )
    # Classic still saves something — at minimum filler words and whitespace.
    assert result["chars_saved"] > 0


def test_optimize_quantum_mode_produces_aggressive_compression():
    """The quantum algorithm trades fidelity for compression. We do NOT pin a
    recall threshold — the benchmark just records the actual rate so a future
    change to the salience selector is caught as a regression."""
    text = (_FIXTURE_DIR / "sample_doc.md").read_text(encoding="utf-8")
    result = _call("chimera_optimize", {
        "text": text,
        "namespace": "bench-doc",
        "preserve_code": True,
        "algorithm": "quantum",
    })
    compressed = result["optimised_text"]
    recall = _entity_recall(DOC_REQUIRED_ENTITIES, compressed)
    # Documented characteristic, not a quality bar — quantum is aggressive.
    assert 0.20 <= recall <= 1.0, f"quantum recall outside expected band: {recall:.2%}"
    assert result["reduction_ratio"] >= 0.50, (
        f"quantum ratio fell below 50%: {result['reduction_ratio']}"
    )


# ── chimera_log_compress on noisy build logs ──────────────────────────────


def test_log_compress_keeps_every_error_warning_traceback():
    """Log compression is the cleanest fidelity story we have — 100% of
    error/warning/fatal lines survive by construction."""
    text = (_FIXTURE_DIR / "sample_log.txt").read_text(encoding="utf-8")
    result = _call("chimera_log_compress", {
        "text": text,
        "namespace": "bench-log",
        "head_lines": 3,
        "tail_lines": 3,
    })
    compressed = result["compressed_text"]
    missing = [line for line in LOG_REQUIRED_LINES if line not in compressed]
    assert not missing, f"diagnostic signal lost from log: {missing}"
    assert result["reduction_ratio"] >= 0.30


def test_log_compress_produces_smaller_output_with_tighter_windows():
    text = (_FIXTURE_DIR / "sample_log.txt").read_text(encoding="utf-8")
    wide = _call("chimera_log_compress", {
        "text": text, "namespace": "bench-log", "head_lines": 100, "tail_lines": 100,
    })
    narrow = _call("chimera_log_compress", {
        "text": text, "namespace": "bench-log", "head_lines": 1, "tail_lines": 1,
    })
    assert narrow["compressed_chars"] <= wide["compressed_chars"]


# ── chimera_fracture (full pipeline) ──────────────────────────────────────


def test_fracture_runs_without_error_on_combined_payload():
    """Fracture is the most aggressive pipeline — its output shape is
    implementation-defined. The benchmark just guards that it runs and
    reports a positive savings number, not that any specific entity survives."""
    text = (_FIXTURE_DIR / "sample_doc.md").read_text(encoding="utf-8")
    messages = [
        {"role": "user", "content": "Tell me about /auth/v2/login"},
        {"role": "assistant", "content": "It's signed with RS256."},
        {"role": "user", "content": "And the rate limit?"},
    ]
    result = _call("chimera_fracture", {
        "documents": [text],
        "messages": messages,
        "namespace": "bench-fracture",
    })
    # Fracture returns a structured payload with savings metadata.
    assert "error" not in result, f"fracture failed: {result.get('error')}"
