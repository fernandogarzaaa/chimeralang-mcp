"""ChimeraLang MCP Server — full implementation.

Tools exposed to Claude:
  chimera_run          Execute a .chimera program string
  chimera_confident    Enforce >= 0.95 confidence threshold on a value
  chimera_explore      Wrap a value as exploratory (hallucination permitted)
  chimera_gate         Collapse multiple candidates via quantum consensus
  chimera_detect       Hallucination detection — 5 strategies
  chimera_constrain    Full constraint middleware on any tool result
  chimera_typecheck    Static type-check a .chimera program
  chimera_prove        Execute + generate Merkle-chain integrity proof
  chimera_audit        Session call-log summary
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from chimera.lexer import Lexer
from chimera.parser import Parser
from chimera.vm import ChimeraVM
from chimera.type_checker import TypeChecker
from chimera.detect import HallucinationDetector, DetectionReport
from chimera.integrity import IntegrityEngine
from chimera.types import ConfidenceViolation
from chimera.claude_adapter import ClaudeConstraintMiddleware, ToolCallSpec

# ── session-scoped singletons ─────────────────────────────────────────────
_middleware = ClaudeConstraintMiddleware(confidence_threshold=0.7)
_detector   = HallucinationDetector()
server      = Server("chimeralang-mcp")


# ── helpers ───────────────────────────────────────────────────────────────

def _ok(data: Any) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(data, indent=2))],
        isError=False,
    )

def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )

def _run(source: str) -> dict[str, Any]:
    tokens = Lexer(source).tokenize()
    ast    = Parser(tokens).parse()
    vm     = ChimeraVM()
    r      = vm.execute(ast)
    return {
        "emitted": [
            {
                "value":      str(v.raw),
                "confidence": round(v.confidence.value, 4),
                "type":       type(v).__name__,
                "trace":      v.trace[-3:],
            }
            for v in r.emitted
        ],
        "assertions_passed":  r.assertions_passed,
        "assertions_failed":  r.assertions_failed,
        "errors":             r.errors,
        "duration_ms":        round(r.duration_ms, 3),
        "gate_logs": [
            {k: gl[k] for k in (
                "gate", "branches", "collapse",
                "result_value", "result_confidence",
                "divergence_ratio", "unique_branch_values",
            ) if k in gl}
            for gl in r.gate_logs
        ],
        "trace_tail": r.trace[-8:],
    }


# ── tool registry ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="chimera_run",
            description=(
                "Execute a ChimeraLang program string and return emitted values with "
                "confidence scores, gate consensus logs, assertion results, and execution "
                "trace. Use to run probabilistic reasoning pipelines, validate values "
                "through consensus gates, or chain hallucination-detection logic."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "ChimeraLang source code"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="chimera_confident",
            description=(
                "Assert that a value meets ChimeraLang's Confident<> threshold (>= 0.95). "
                "Returns the wrapped ConfidentValue on success. "
                "Returns a ConfidenceViolation error with a suggestion if confidence is too low. "
                "Use before passing a value to a critical downstream tool or decision."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "value":      {"description": "The value to assert confidence on"},
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0–1.0. Must be >= 0.95 to pass.",
                        "minimum": 0.0, "maximum": 1.0,
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label for trace identification",
                    },
                },
                "required": ["value", "confidence"],
            },
        ),
        Tool(
            name="chimera_explore",
            description=(
                "Wrap a value as Explore<> — explicitly marking it as exploratory where "
                "hallucination is permitted and expected. "
                "Use for hypotheses, creative outputs, brainstorms, and anything that "
                "must not be treated as verified fact."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "value":      {"description": "The exploratory value"},
                    "confidence": {
                        "type": "number",
                        "description": "Confidence 0.0–1.0 (typically low for explore values)",
                        "minimum": 0.0, "maximum": 1.0, "default": 0.5,
                    },
                    "label": {"type": "string", "description": "Optional label"},
                },
                "required": ["value"],
            },
        ),
        Tool(
            name="chimera_gate",
            description=(
                "Collapse multiple candidate values into a single consensus result using "
                "ChimeraLang's quantum consensus gate. "
                "Detects and flags trivial consensus (all branches identical — no real agreement). "
                "Strategies: majority, weighted_vote, highest_confidence. "
                "Use when reconciling multiple model outputs, tool results, or reasoning branches."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "description": "Candidate values with optional confidence scores",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value":      {"description": "Candidate value"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            },
                            "required": ["value"],
                        },
                        "minItems": 2,
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["majority", "weighted_vote", "highest_confidence"],
                        "default": "weighted_vote",
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum consensus confidence to pass (default 0.80)",
                        "default": 0.80,
                    },
                },
                "required": ["candidates"],
            },
        ),
        Tool(
            name="chimera_detect",
            description=(
                "Run ChimeraLang hallucination detection on a value. Five strategies:\n"
                "  range            — numeric value must fall within [min, max]\n"
                "  dictionary       — value must be in an allowed set\n"
                "  semantic         — forbidden patterns / absolute-certainty markers\n"
                "  cross_reference  — value must not deviate from a reference set\n"
                "  temporal         — value timestamp must not be stale\n"
                "  confidence_threshold — confidence must be >= threshold\n"
                "Returns flags with severity scores and evidence."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "value":      {"description": "Value to scan"},
                    "confidence": {"type": "number", "default": 0.8},
                    "strategy": {
                        "type": "string",
                        "enum": ["range", "dictionary", "semantic",
                                 "cross_reference", "temporal", "confidence_threshold"],
                    },
                    "params": {
                        "type": "object",
                        "description": (
                            "Strategy params:\n"
                            "  range:            {valid_range: [min, max]}\n"
                            "  dictionary:       {allowed_values: [...]}\n"
                            "  semantic:         {forbidden_patterns: [...]} "
                            "                    (omit for default absolute-certainty scan)\n"
                            "  cross_reference:  {reference_values: [...], tolerance: 0.1}\n"
                            "  temporal:         {max_age_seconds: 3600, reference_time: <unix_ts>}\n"
                            "  confidence_threshold: {threshold: 0.7}"
                        ),
                    },
                },
                "required": ["value", "strategy"],
            },
        ),
        Tool(
            name="chimera_constrain",
            description=(
                "Apply ChimeraLang's full constraint middleware to any tool result. "
                "Pipeline: confidence gate → must-constraint checks → "
                "forbidden capability logging → hallucination detection → ephemeral scope cleanup. "
                "Returns pass/fail with violations, warnings, confidence, and audit trace. "
                "Primary integration point for wrapping Claude's own tool calls."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_name":        {"type": "string"},
                    "output":           {"description": "Raw tool output to constrain"},
                    "input_confidence": {"type": "number", "default": 1.0,
                                        "description": "Caller confidence this was the right call"},
                    "min_confidence":   {"type": "number", "default": 0.0,
                                        "description": "Minimum confidence to accept output"},
                    "output_forbidden": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Forbidden capability markers (e.g. file_write, pii)",
                    },
                    "detect_strategy": {
                        "type": "string",
                        "enum": ["confidence_threshold", "range", "semantic",
                                 "cross_reference", "temporal"],
                        "default": "confidence_threshold",
                    },
                    "detect_threshold": {"type": "number", "default": 0.5},
                    "strict": {
                        "type": "boolean", "default": False,
                        "description": "True = violations raise exceptions, not just warnings",
                    },
                },
                "required": ["tool_name", "output"],
            },
        ),
        Tool(
            name="chimera_typecheck",
            description=(
                "Statically type-check a ChimeraLang program without executing it. "
                "Validates confidence boundaries, memory scope rules, and illegal "
                "Explore→Confident promotions outside a gate. "
                "Returns errors and warnings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "ChimeraLang source to check"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="chimera_prove",
            description=(
                "Execute a ChimeraLang program and generate a Merkle-chain integrity proof. "
                "Every reasoning step is SHA-256 hashed and chained — tamper-evident derivation. "
                "Returns execution results + proof with root hash, chain length, verdict, "
                "gate certificates, and hallucination scan summary."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "ChimeraLang source to prove"},
                },
                "required": ["source"],
            },
        ),
        Tool(
            name="chimera_audit",
            description=(
                "Return the constraint audit summary for this session. "
                "Shows total calls, pass/fail counts, average output confidence, "
                "flagged warnings, and tools used. "
                "Use at the end of a multi-tool workflow to assess overall reliability."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


# ── tool handlers ─────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    try:

        # ── chimera_run ───────────────────────────────────────────────────
        if name == "chimera_run":
            return _ok(_run(arguments["source"]))

        # ── chimera_confident ─────────────────────────────────────────────
        elif name == "chimera_confident":
            value      = arguments["value"]
            confidence = float(arguments["confidence"])
            label      = arguments.get("label", str(value)[:40])

            if confidence < 0.95:
                return _ok({
                    "passed":     False,
                    "error":      f"ConfidenceViolation: {confidence:.3f} < required 0.95",
                    "suggestion": (
                        "Use chimera_explore for uncertain values, or route through "
                        "chimera_gate to build consensus before asserting confidence."
                    ),
                    "value":      str(value),
                    "confidence": confidence,
                })
            return _ok({
                "passed":     True,
                "type":       "ConfidentValue",
                "value":      str(value),
                "confidence": confidence,
                "label":      label,
                "trace":      [f"confident({label})", f"score={confidence:.4f}"],
            })

        # ── chimera_explore ───────────────────────────────────────────────
        elif name == "chimera_explore":
            value      = arguments["value"]
            confidence = min(max(float(arguments.get("confidence", 0.5)), 0.0), 1.0)
            label      = arguments.get("label", str(value)[:40])
            return _ok({
                "type":              "ExploreValue",
                "value":             str(value),
                "confidence":        confidence,
                "label":             label,
                "exploration_budget": 1.0,
                "note":              (
                    "Hallucination is explicitly permitted in Explore<> space. "
                    "Gate this value before treating it as fact."
                ),
                "trace": [f"explore({label})", f"score={confidence:.4f}"],
            })

        # ── chimera_gate ──────────────────────────────────────────────────
        elif name == "chimera_gate":
            candidates = arguments["candidates"]
            strategy   = arguments.get("strategy", "weighted_vote")
            threshold  = float(arguments.get("threshold", 0.80))

            if len(candidates) < 2:
                return _err("chimera_gate requires at least 2 candidates")

            branches = [
                {"value": c["value"], "str": str(c["value"]),
                 "confidence": float(c.get("confidence", 0.8))}
                for c in candidates
            ]

            unique_vals    = set(b["str"] for b in branches)
            divergence     = (len(unique_vals) - 1) / max(len(branches) - 1, 1)
            trivial        = divergence == 0.0

            if strategy == "highest_confidence":
                winner       = max(branches, key=lambda b: b["confidence"])
                consensus_conf = winner["confidence"]

            elif strategy == "weighted_vote":
                weights: dict[str, float] = {}
                for b in branches:
                    weights[b["str"]] = weights.get(b["str"], 0.0) + b["confidence"]
                total        = sum(weights.values())
                winner_key   = max(weights, key=weights.__getitem__)
                winner       = next(b for b in branches if b["str"] == winner_key)
                consensus_conf = weights[winner_key] / total if total else 0.0

            else:  # majority
                groups: dict[str, list] = {}
                for b in branches:
                    groups.setdefault(b["str"], []).append(b)
                winner_key   = max(groups, key=lambda k: len(groups[k]))
                group        = groups[winner_key]
                winner       = group[0]
                consensus_conf = sum(b["confidence"] for b in group) / len(group)

            passed = consensus_conf >= threshold
            result: dict[str, Any] = {
                "passed":            passed,
                "type":              "ConvergeValue",
                "value":             winner["value"],
                "consensus_confidence": round(consensus_conf, 4),
                "threshold":         threshold,
                "strategy":          strategy,
                "branches":          len(branches),
                "unique_values":     len(unique_vals),
                "divergence_ratio":  round(divergence, 4),
                "trivial_consensus": trivial,
            }
            if trivial:
                result["warning"] = (
                    "All branches returned identical values — trivial consensus. "
                    "No genuine divergence detected. Use independent reasoning paths "
                    "for real consensus signal."
                )
            if not passed:
                result["warning"] = (
                    f"Consensus confidence {consensus_conf:.3f} below threshold {threshold}. "
                    "Result is unreliable — consider more branches or lower threshold."
                )
            return _ok(result)

        # ── chimera_detect ────────────────────────────────────────────────
        elif name == "chimera_detect":
            value      = arguments["value"]
            confidence = float(arguments.get("confidence", 0.8))
            strategy   = arguments["strategy"]
            params     = arguments.get("params") or {}
            flags: list[dict[str, Any]] = []
            passed = True

            if strategy == "range":
                vr = params.get("valid_range")
                if vr and len(vr) == 2:
                    lo, hi = float(vr[0]), float(vr[1])
                    try:
                        v = float(value)
                        if not (lo <= v <= hi):
                            passed = False
                            flags.append({
                                "kind":        "RANGE_VIOLATION",
                                "severity":    0.9,
                                "description": f"Value {v} outside valid range [{lo}, {hi}]",
                            })
                    except (TypeError, ValueError):
                        flags.append({
                            "kind":        "TYPE_ERROR",
                            "severity":    0.5,
                            "description": f"Cannot range-check non-numeric: {value!r}",
                        })

            elif strategy == "dictionary":
                allowed = params.get("allowed_values", [])
                if value not in allowed:
                    passed = False
                    flags.append({
                        "kind":        "DICTIONARY_VIOLATION",
                        "severity":    0.85,
                        "description": f"{value!r} not in allowed set",
                        "allowed":     allowed,
                    })

            elif strategy == "semantic":
                forbidden = params.get("forbidden_patterns", [])
                val_str   = str(value).lower()
                if forbidden:
                    for pat in forbidden:
                        if pat.lower() in val_str:
                            passed = False
                            flags.append({
                                "kind":        "SEMANTIC_VIOLATION",
                                "severity":    0.85,
                                "description": f"Forbidden pattern '{pat}' found",
                            })
                else:
                    # Default: flag absolute-certainty markers
                    markers = ["always", "never", "definitely", "100%",
                               "impossible", "certain", "guaranteed", "never fails"]
                    hits = [m for m in markers if m in val_str]
                    if hits:
                        flags.append({
                            "kind":        "SEMANTIC_WARNING",
                            "severity":    0.6,
                            "description": f"Absolute-certainty markers detected: {hits}",
                            "note":        "May indicate hallucination in uncertain domains",
                        })

            elif strategy == "cross_reference":
                refs      = params.get("reference_values", [])
                tolerance = float(params.get("tolerance", 0.1))
                if refs:
                    try:
                        target = float(value)
                        ref_fs = [float(r) for r in refs]
                        avg    = sum(ref_fs) / len(ref_fs)
                        dev    = abs(target - avg) / (abs(avg) + 1e-9)
                        if dev > tolerance:
                            passed = False
                            flags.append({
                                "kind":        "CROSS_REFERENCE_VIOLATION",
                                "severity":    min(dev, 1.0),
                                "description": (
                                    f"Value {target} deviates {dev:.3f} "
                                    f"from reference avg {avg:.3f} (tolerance {tolerance})"
                                ),
                            })
                    except (TypeError, ValueError):
                        if value not in refs:
                            passed = False
                            flags.append({
                                "kind":        "CROSS_REFERENCE_VIOLATION",
                                "severity":    0.8,
                                "description": f"{value!r} not in reference set",
                            })

            elif strategy == "temporal":
                import time as _t
                max_age  = float(params.get("max_age_seconds", 3600))
                ref_time = float(params.get("reference_time", _t.time()))
                try:
                    age = ref_time - float(value)
                    if age > max_age:
                        passed = False
                        flags.append({
                            "kind":        "TEMPORAL_VIOLATION",
                            "severity":    min(age / max_age, 1.0),
                            "description": f"Timestamp age {age:.1f}s exceeds max {max_age}s",
                        })
                except (TypeError, ValueError):
                    flags.append({
                        "kind":        "TEMPORAL_SKIP",
                        "severity":    0.0,
                        "description": "Not a timestamp — temporal check skipped",
                    })

            elif strategy == "confidence_threshold":
                threshold = float(params.get("threshold", 0.7))
                if confidence < threshold:
                    passed = False
                    flags.append({
                        "kind":        "CONFIDENCE_BELOW_THRESHOLD",
                        "severity":    round(1.0 - confidence, 3),
                        "description": f"Confidence {confidence:.3f} < threshold {threshold}",
                    })

            return _ok({
                "passed":     passed,
                "strategy":   strategy,
                "value":      str(value),
                "confidence": confidence,
                "clean":      len(flags) == 0,
                "flag_count": len(flags),
                "flags":      flags,
            })

        # ── chimera_constrain ─────────────────────────────────────────────
        elif name == "chimera_constrain":
            spec = ToolCallSpec(
                tool_name        = arguments["tool_name"],
                min_confidence   = float(arguments.get("min_confidence", 0.0)),
                output_forbidden = arguments.get("output_forbidden", []),
                detect_strategy  = arguments.get("detect_strategy", "confidence_threshold"),
                detect_threshold = float(arguments.get("detect_threshold", 0.5)),
                strict           = bool(arguments.get("strict", False)),
            )
            r = _middleware.call(
                spec,
                raw_output        = arguments["output"],
                input_confidence  = float(arguments.get("input_confidence", 1.0)),
            )
            return _ok({
                "tool_name":  r.tool_name,
                "passed":     r.passed,
                "value":      str(r.value),
                "confidence": round(r.confidence, 4),
                "violations": r.violations,
                "warnings":   r.warnings,
                "trace":      r.trace,
                "duration_ms": round(r.duration_ms, 3),
                "detection": {
                    "clean":      r.detection_report.clean if r.detection_report else True,
                    "flag_count": len(r.detection_report.flags) if r.detection_report else 0,
                    "flags": [
                        {"kind": f.kind.name, "severity": f.severity, "description": f.description}
                        for f in (r.detection_report.flags if r.detection_report else [])
                    ],
                },
            })

        # ── chimera_typecheck ─────────────────────────────────────────────
        elif name == "chimera_typecheck":
            source  = arguments["source"]
            tokens  = Lexer(source).tokenize()
            ast     = Parser(tokens).parse()
            result  = TypeChecker().check(ast)
            return _ok({
                "ok":            result.ok,
                "error_count":   len(result.errors),
                "warning_count": len(result.warnings),
                "errors":        result.errors,
                "warnings":      result.warnings,
            })

        # ── chimera_prove ─────────────────────────────────────────────────
        elif name == "chimera_prove":
            source  = arguments["source"]
            tokens  = Lexer(source).tokenize()
            ast     = Parser(tokens).parse()
            vm      = ChimeraVM()
            result  = vm.execute(ast)

            detection = DetectionReport()
            for v in result.emitted:
                _detector.scan_value(v, detection)
            for gl in result.gate_logs:
                _detector.scan_gate_log(gl, detection)

            report = IntegrityEngine().certify(result, detection, source)
            proof  = report.to_dict()

            return _ok({
                "execution": {
                    "emitted": [
                        {"value": str(v.raw), "confidence": round(v.confidence.value, 4)}
                        for v in result.emitted
                    ],
                    "errors":             result.errors,
                    "assertions_passed":  result.assertions_passed,
                    "assertions_failed":  result.assertions_failed,
                },
                "proof": {
                    "verdict":              proof["verdict"],
                    "chain_length":         proof["chain"]["length"],
                    "root_hash":            proof["chain"]["root_hash"],
                    "chain_valid":          proof["chain"]["valid"],
                    "program_hash":         proof["program_hash"],
                    "hallucination_clean":  proof["hallucination"]["clean"],
                    "hallucination_flags":  proof["hallucination"]["flags"],
                    "gate_certificates":    proof["gates"],
                    "note": (
                        f"Merkle chain of {proof['chain']['length']} steps. "
                        "Every reasoning step SHA-256 hashed and chained — tamper-evident."
                    ),
                },
            })

        # ── chimera_audit ─────────────────────────────────────────────────
        elif name == "chimera_audit":
            summary = _middleware.audit_summary()
            log     = _middleware.call_log()
            return _ok({
                **summary,
                "recent_calls": [
                    {
                        "tool":       r.tool_name,
                        "passed":     r.passed,
                        "confidence": round(r.confidence, 4),
                        "violations": len(r.violations),
                        "warnings":   len(r.warnings),
                    }
                    for r in log[-10:]
                ],
            })

        else:
            return _err(f"Unknown tool: {name}")

    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ── entrypoint ────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())
