"""Tests for the second wave of token tooling: prompt-cache markers,
log compression, overhead audit, tool-call dedup cache, and session report.
Plus PreToolUse + Stop hooks."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from chimeralang_mcp import server as srv


def _call(tool: str, args: dict) -> dict:
    result = asyncio.run(srv.call_tool(tool, args))
    return json.loads(result.content[0].text)


# ── chimera_cache_mark ────────────────────────────────────────────────────


class TestCacheMark:
    """Anthropic prompt-cache markers — lossless win on stable content."""

    def test_marks_block_above_min_tokens(self):
        big = "You are a helpful assistant. " * 200  # ~1400+ tokens
        result = _call("chimera_cache_mark", {
            "blocks": [{"name": "system", "text": big, "stable": True}],
            "model": "claude-sonnet-4-6",
        })
        assert result["breakpoints_used"] == 1
        assert result["cache_eligible_tokens"] > 0
        assert result["blocks"][0]["cache_control"] == {"type": "ephemeral"}
        assert result["estimated_savings_at_90pct"] >= int(result["cache_eligible_tokens"] * 0.9) - 1

    def test_skips_blocks_below_min_tokens(self):
        result = _call("chimera_cache_mark", {
            "blocks": [{"name": "tiny", "text": "hi", "stable": True}],
            "model": "claude-sonnet-4-6",
        })
        assert result["breakpoints_used"] == 0
        assert result["skipped_too_small"][0]["name"] == "tiny"
        assert "cache_control" not in result["blocks"][0]

    def test_never_marks_unstable_blocks(self):
        big = "x " * 5000
        result = _call("chimera_cache_mark", {
            "blocks": [{"name": "user", "text": big, "stable": False}],
            "model": "claude-sonnet-4-6",
        })
        assert result["breakpoints_used"] == 0
        assert "cache_control" not in result["blocks"][0]

    def test_respects_max_breakpoints(self):
        big = "system reminder text " * 800  # ~1600+ tokens, comfortably above 1024 min
        blocks = [{"name": f"b{i}", "text": big, "stable": True} for i in range(8)]
        result = _call("chimera_cache_mark", {
            "blocks": blocks, "model": "claude-sonnet-4-6", "max_breakpoints": 4,
        })
        assert result["breakpoints_used"] == 4
        marked = [b for b in result["blocks"] if "cache_control" in b]
        assert len(marked) == 4

    def test_haiku_min_threshold_higher_than_sonnet(self):
        text = "x " * 800  # ~1600 tokens — eligible for sonnet, not haiku
        sonnet = _call("chimera_cache_mark", {
            "blocks": [{"name": "b", "text": text}],
            "model": "claude-sonnet-4-6",
        })
        haiku = _call("chimera_cache_mark", {
            "blocks": [{"name": "b", "text": text}],
            "model": "claude-haiku-4-5",
        })
        assert sonnet["breakpoints_used"] >= haiku["breakpoints_used"]


# ── chimera_log_compress ──────────────────────────────────────────────────


class TestLogCompress:
    """Build/test/install logs — preserve all error lines verbatim."""

    def test_preserves_error_lines_verbatim(self):
        lines = (
            [f"INFO setup step {i}" for i in range(500)]
            + ["ERROR: tests/test_foo.py::test_bar FAILED at line 42"]
            + [f"INFO teardown step {i}" for i in range(500)]
        )
        text = "\n".join(lines)
        result = _call("chimera_log_compress", {"text": text, "namespace": "log-test"})
        assert "ERROR: tests/test_foo.py::test_bar FAILED at line 42" in result["compressed_text"]
        assert result["matches_kept"] >= 1
        assert result["lines_in"] == 1001
        assert result["reduction_ratio"] > 0.5

    def test_preserves_warnings_and_tracebacks(self):
        text = "\n".join([
            *(["INFO mundane"] * 200),
            "WARNING: deprecated API used",
            *(["INFO mundane"] * 200),
            "Traceback (most recent call last):",
            "  File 'x.py', line 1, in <module>",
            "    boom()",
            *(["INFO mundane"] * 200),
        ])
        result = _call("chimera_log_compress", {"text": text, "namespace": "log-test"})
        assert "WARNING: deprecated API" in result["compressed_text"]
        assert "Traceback" in result["compressed_text"]

    def test_records_savings_to_cost_tracker(self):
        text = "\n".join([f"INFO {i}" for i in range(2000)] + ["ERROR boom"])
        result = _call("chimera_log_compress", {
            "text": text, "namespace": "log-track", "auto_track": True,
        })
        assert "tracked" in result
        assert result["estimated_tokens_saved"] > 0

    def test_custom_keep_patterns(self):
        text = "alpha\nbravo\ncharlie\ndelta\necho\n" * 100
        result = _call("chimera_log_compress", {
            "text": text,
            "keep_patterns": ["delta"],
            "head_lines": 0, "tail_lines": 0,
            "namespace": "log-pat",
        })
        assert "delta" in result["compressed_text"]
        # other distinctive non-context lines should be dropped
        assert result["matches_kept"] == 100


# ── chimera_overhead_audit ────────────────────────────────────────────────


class TestOverheadAudit:
    def test_categorizes_lean_moderate_heavy(self):
        lean = _call("chimera_overhead_audit", {"system_prompt": "short"})
        assert lean["advisory"].startswith("lean")
        heavy = _call("chimera_overhead_audit", {
            "system_prompt": "x" * 30000,
            "tool_definitions": [
                {"name": f"t{i}", "description": "d" * 500, "schema": {"x": 1}}
                for i in range(40)
            ],
            "mcp_servers": [{"name": "s", "tool_count": 50, "avg_tokens_per_tool": 250}],
        })
        assert heavy["advisory"].startswith("heavy"), (
            f"expected 'heavy' advisory, got {heavy['advisory']!r} "
            f"at grand_total={heavy['grand_total_tokens']}"
        )
        assert heavy["grand_total_tokens"] > lean["grand_total_tokens"]

    def test_breakdown_sorted_by_tokens(self):
        result = _call("chimera_overhead_audit", {
            "tool_definitions": [
                {"name": "small", "description": "x", "schema": {}},
                {"name": "big", "description": "d" * 1000, "schema": {}},
            ],
        })
        top = result["tool_breakdown_top"]
        assert top[0]["name"] == "big"
        assert top[0]["tokens"] >= top[1]["tokens"]


# ── tool-call dedup cache ─────────────────────────────────────────────────


class TestDedupCache:
    NS = "dedup-unit"

    def setup_method(self):
        srv._dedup_clear(self.NS)

    def test_first_record_then_hit(self):
        srv._dedup_record(self.NS, "Read", {"file_path": "/x"}, "first response body")
        entries = srv._dedup_load(self.NS)
        assert len(entries) == 1
        assert entries[0]["hit_count"] == 0  # first call is not a hit

        srv._dedup_record(self.NS, "Read", {"file_path": "/x"}, "first response body")
        entries = srv._dedup_load(self.NS)
        assert len(entries) == 1
        assert entries[0]["hit_count"] == 1

    def test_different_args_get_different_keys(self):
        srv._dedup_record(self.NS, "Read", {"file_path": "/a"}, "A")
        srv._dedup_record(self.NS, "Read", {"file_path": "/b"}, "B")
        entries = srv._dedup_load(self.NS)
        assert len({e["key"] for e in entries}) == 2

    def test_lookup_tool_get_clear(self):
        srv._dedup_record(self.NS, "Read", {"file_path": "/c"}, "body")
        key = srv._dedup_key("Read", {"file_path": "/c"})

        result = _call("chimera_dedup_lookup", {
            "namespace": self.NS, "action": "get", "key": key,
        })
        assert result["found"] is True
        assert result["entry"]["tool_name"] == "Read"

        listed = _call("chimera_dedup_lookup", {"namespace": self.NS})
        assert listed["entry_count"] >= 1

        cleared = _call("chimera_dedup_lookup", {"namespace": self.NS, "action": "clear"})
        assert cleared["cleared_entries"] >= 1
        assert _call("chimera_dedup_lookup", {"namespace": self.NS})["entry_count"] == 0

    def test_lookup_get_requires_key(self):
        result = _call("chimera_dedup_lookup", {"namespace": self.NS, "action": "get"})
        assert "error" in result

    def test_canonical_key_is_argument_order_independent(self):
        k1 = srv._dedup_key("Bash", {"command": "ls", "timeout": 5})
        k2 = srv._dedup_key("Bash", {"timeout": 5, "command": "ls"})
        assert k1 == k2

    def test_previews_are_off_by_default(self, monkeypatch):
        monkeypatch.delenv("CHIMERA_DEDUP_STORE_PREVIEWS", raising=False)
        srv._dedup_record(self.NS, "Bash", {"command": "echo secret"}, "secret-output")
        entry = srv._dedup_load(self.NS)[-1]
        assert "tool_input_preview" not in entry
        assert "response_preview" not in entry
        # Hashes + lengths still present so dedup works.
        assert entry["response_hash"]
        assert entry["response_chars"] == len("secret-output")

    def test_previews_opt_in_via_env(self, monkeypatch):
        monkeypatch.setenv("CHIMERA_DEDUP_STORE_PREVIEWS", "1")
        srv._dedup_record(self.NS, "Bash", {"command": "echo hello"}, "hello")
        entry = srv._dedup_load(self.NS)[-1]
        assert entry["response_preview"] == "hello"
        assert "echo hello" in entry["tool_input_preview"]

    def test_lru_moves_hit_entry_to_end(self, monkeypatch):
        monkeypatch.delenv("CHIMERA_DEDUP_STORE_PREVIEWS", raising=False)
        srv._dedup_record(self.NS, "Read", {"file_path": "/a"}, "A")
        srv._dedup_record(self.NS, "Read", {"file_path": "/b"}, "B")
        srv._dedup_record(self.NS, "Read", {"file_path": "/c"}, "C")
        # Hit /a — it should move to the end.
        srv._dedup_record(self.NS, "Read", {"file_path": "/a"}, "A")
        entries = srv._dedup_load(self.NS)
        assert entries[-1]["key"] == srv._dedup_key("Read", {"file_path": "/a"})
        assert entries[-1]["hit_count"] == 1


class TestLogCompressCorrectness:
    """Regressions for two Copilot review findings."""

    def test_lines_out_includes_abridgement_markers(self):
        text = "\n".join(["INFO {}".format(i) for i in range(200)] + ["ERROR boom"])
        result = _call("chimera_log_compress", {
            "text": text, "namespace": "log-fix", "head_lines": 2, "tail_lines": 2,
        })
        emitted = result["compressed_text"].count("\n") + 1
        assert result["lines_out"] == emitted
        assert result["lines_kept_verbatim"] <= result["lines_out"]

    def test_reduction_ratio_clamped_to_unit_interval(self):
        # Very short input where the abridgement marker can outweigh savings.
        result = _call("chimera_log_compress", {
            "text": "ERROR boom\nINFO ok",
            "namespace": "log-fix",
            "head_lines": 0, "tail_lines": 0,
        })
        assert 0.0 <= result["reduction_ratio"] <= 1.0


# ── chimera_session_report ────────────────────────────────────────────────


class TestSessionReport:
    NS = "session-report-unit"

    def setup_method(self):
        srv._dedup_clear(self.NS)

    def test_report_includes_dedup_when_requested(self):
        srv._dedup_record(self.NS, "Read", {"file_path": "/r"}, "body")
        srv._dedup_record(self.NS, "Read", {"file_path": "/r"}, "body")
        srv._dedup_record(self.NS, "Bash", {"command": "ls"}, "out")
        report = _call("chimera_session_report", {"namespace": self.NS})
        assert report["dedup"]["tracked_calls"] == 2
        assert report["dedup"]["total_hits"] >= 1

    def test_report_can_omit_dedup(self):
        report = _call("chimera_session_report", {"namespace": self.NS, "include_dedup": False})
        assert "dedup" not in report

    def test_report_carries_budget_snapshot(self):
        report = _call("chimera_session_report", {"namespace": self.NS})
        assert "budget" in report
        assert "advisory" in report["budget"]


# ── PreToolUse + Stop hooks ────────────────────────────────────────────────


class TestPreToolUseHook:
    def _run(self, stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "chimeralang_mcp.cli", "hook", "--event", "pre-tool-use"],
            input=stdin, capture_output=True, text=True, timeout=20,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_oversized_edit_input_emits_advisory(self):
        body = "a " * 3000
        proc = self._run(json.dumps({
            "tool_name": "Edit",
            "tool_input": {"file_path": "/f", "old_string": body, "new_string": "y"},
        }))
        assert proc.returncode == 0
        ctx = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "[chimera-token-saver]" in ctx
        assert "Edit" in ctx

    def test_small_input_is_silent(self):
        proc = self._run(json.dumps({
            "tool_name": "Edit",
            "tool_input": {"file_path": "/f", "old_string": "a", "new_string": "b"},
        }))
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""

    def test_dedup_hit_emits_advisory(self):
        srv._dedup_clear("claude-code-hook:default")
        srv._dedup_record("claude-code-hook:default", "Read", {"file_path": "/dedup-pre"}, "body")
        proc = self._run(json.dumps({
            "tool_name": "Read",
            "tool_input": {"file_path": "/dedup-pre"},
        }))
        assert proc.returncode == 0
        ctx = json.loads(proc.stdout)["hookSpecificOutput"]["additionalContext"]
        assert "already called" in ctx
        assert "chimera_dedup_lookup" in ctx

    def test_chimera_tool_is_skipped(self):
        proc = self._run(json.dumps({
            "tool_name": "chimera_explore", "tool_input": {"value": "x" * 10000},
        }))
        assert proc.returncode == 0
        assert proc.stdout.strip() == ""


class TestStopHook:
    def _run(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "chimeralang_mcp.cli", "hook", "--event", "stop"],
            input="{}", capture_output=True, text=True, timeout=20,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_stop_emits_summary_when_dedup_present(self):
        srv._dedup_clear("claude-code-hook:default")
        srv._dedup_record("claude-code-hook:default", "Read", {"file_path": "/stop-test"}, "body")
        proc = self._run()
        assert proc.returncode == 0
        out = proc.stdout.strip()
        assert out, "stop hook should emit output when dedup entries are tracked"
        ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        assert "Session totals" in ctx
        assert "Dedup" in ctx


# ── PostToolUse now records into dedup ─────────────────────────────────────


class TestPostToolUseRecordsDedup:
    def _run(self, stdin: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "chimeralang_mcp.cli", "hook", "--event", "post-tool-use"],
            input=stdin, capture_output=True, text=True, timeout=20,
            cwd=str(Path(__file__).resolve().parent.parent),
        )

    def test_post_tool_use_records_dedup_entry(self):
        srv._dedup_clear("claude-code-hook:default")
        self._run(json.dumps({
            "tool_name": "Read",
            "tool_input": {"file_path": "/post-records-test"},
            "tool_response": "hello",
        }))
        expected_key = srv._dedup_key("Read", {"file_path": "/post-records-test"})
        entries = srv._dedup_load("claude-code-hook:default")
        # Match by key (previews are off by default for privacy).
        assert any(
            e.get("tool_name") == "Read" and e.get("key") == expected_key
            for e in entries
        )

    def test_chimera_tool_is_not_recorded(self):
        srv._dedup_clear("claude-code-hook:default")
        self._run(json.dumps({
            "tool_name": "chimera_explore",
            "tool_input": {"value": "x"},
            "tool_response": "...",
        }))
        entries = srv._dedup_load("claude-code-hook:default")
        assert not any(e.get("tool_name", "").startswith("chimera_") for e in entries)
