"""Tests for the session-wide token tools: budget advisory injection,
selective oversized-response compression, and the PostToolUse hook."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from chimeralang_mcp import server as srv


def _call(tool: str, args: dict) -> dict:
    result = asyncio.run(srv.call_tool(tool, args))
    return json.loads(result.content[0].text)


class TestBudgetAdvisoryInjection:
    """Every chimera tool response should now carry a session budget snapshot,
    making the token policy visible on every turn instead of only the first."""

    def test_advisory_attached_to_normal_tool(self):
        payload = _call("chimera_explore", {"value": "x", "confidence": 0.5, "namespace": "advisory-test"})
        adv = payload.get("_chimera_session_budget")
        assert adv, "missing _chimera_session_budget on chimera_explore"
        assert adv["tool"] == "chimera_explore"
        assert adv["namespace"] == "advisory-test"
        assert "advisory" in adv

    def test_advisory_skipped_for_native_budget_tools(self):
        # Tools that already report budget data inline must not be double-tagged.
        payload = _call("chimera_dashboard", {"namespace": "advisory-skip"})
        assert "_chimera_session_budget" not in payload

    def test_advisory_reflects_active_budget_lock(self):
        ns = "advisory-locked"
        _call("chimera_budget_lock", {"action": "lock", "max_output_tokens": 1000, "namespace": ns})
        payload = _call("chimera_explore", {"value": "x", "confidence": 0.5, "namespace": ns})
        adv = payload["_chimera_session_budget"]
        assert adv["lock_active"] is True
        assert adv["lock_max_output_tokens"] == 1000
        assert adv["lock_tokens_remaining"] == 1000
        assert adv["advisory"] == "lock_ok"

    def test_advisory_does_not_break_error_path(self):
        # Errors return _err which is a different code path — no advisory expected,
        # but more importantly nothing crashes.
        payload = _call("chimera_gate", {"candidates": [], "namespace": "advisory-err"})
        assert "error" in payload


class TestSelectiveCompression:
    """Oversized responses should auto-compress long narrative fields while
    preserving identifiers/hashes/JSON-shaped values."""

    def test_compression_triggered_on_oversized_response(self):
        token = srv._call_context.set(("chimera_explore", "compress-test"))
        try:
            big_narrative = (
                "Lorem ipsum dolor sit amet consectetur adipiscing elit "
                "sed do eiusmod tempor incididunt ut labore. " * 50
            )
            result = srv._ok({
                "narrative": big_narrative,
                "envelope_id": "preserve-me-exactly",
                "value": "short",
            })
        finally:
            srv._call_context.reset(token)
        payload = json.loads(result.content[0].text)
        assert "_chimera_compressed_fields" in payload
        compressed = payload["_chimera_compressed_fields"]
        assert any(entry["path"] == "narrative" for entry in compressed)
        # Identifiers must never be compressed
        assert payload["envelope_id"] == "preserve-me-exactly"
        # Narrative shrunk
        assert len(payload["narrative"]) < len(big_narrative)

    def test_compression_skipped_for_text_output_tools(self):
        token = srv._call_context.set(("chimera_optimize", "compress-skip"))
        try:
            result = srv._ok({"optimised_text": "x" * 6000, "estimated_tokens_saved": 1})
        finally:
            srv._call_context.reset(token)
        payload = json.loads(result.content[0].text)
        # chimera_optimize output should never be re-compressed
        assert "_chimera_compressed_fields" not in payload
        assert len(payload["optimised_text"]) == 6000

    def test_compression_preserves_short_responses(self):
        token = srv._call_context.set(("chimera_explore", "compress-small"))
        try:
            result = srv._ok({"value": "short", "note": "tiny"})
        finally:
            srv._call_context.reset(token)
        payload = json.loads(result.content[0].text)
        assert "_chimera_compressed_fields" not in payload


class TestPostToolUseHook:
    """The PostToolUse hook nudges the model when non-chimera tools return
    large payloads — the missing piece that makes token tools fire mid-turn."""

    def _run(self, stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "chimeralang_mcp.cli", "hook", "--event", "post-tool-use"],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=20,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_large_non_chimera_response_emits_advisory(self):
        body = "a long file body with lots of content " * 100
        proc = self._run(json.dumps({
            "tool_name": "Read",
            "tool_input": {"file_path": "/foo"},
            "tool_response": body,
        }))
        assert proc.returncode == 0
        payload = json.loads(proc.stdout)
        ctx = payload["hookSpecificOutput"]["additionalContext"]
        assert "[chimera-token-saver]" in ctx
        assert "Read" in ctx
        assert "chimera_optimize" in ctx

    def test_chimera_tool_response_is_skipped(self):
        proc = self._run(json.dumps({
            "tool_name": "chimera_explore",
            "tool_response": "x" * 5000,
        }))
        assert proc.returncode == 0
        # Empty stdout = no nudge, since chimera tools self-report budget
        assert proc.stdout.strip() == ""

    def test_chimeralang_namespaced_tool_is_skipped(self):
        proc = self._run(json.dumps({
            "tool_name": "mcp__chimeralang__chimera_run",
            "tool_response": "x" * 5000,
        }))
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    def test_short_response_is_silent(self):
        proc = self._run(json.dumps({
            "tool_name": "Bash",
            "tool_response": "ok",
        }))
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""
