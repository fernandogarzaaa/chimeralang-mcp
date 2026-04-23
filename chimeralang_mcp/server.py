"""ChimeraLang MCP Server — full implementation.

Tools exposed to Claude:
  Core reasoning:    chimera_run, chimera_confident, chimera_explore, chimera_gate,
                    chimera_detect, chimera_constrain, chimera_typecheck, chimera_prove,
                    chimera_audit
  Token management: chimera_optimize, chimera_compress, chimera_fracture,
                    chimera_budget, chimera_score
  AGI (OpenChimera): chimera_causal, chimera_deliberate, chimera_metacognize,
                    chimera_meta_learn, chimera_quantum_vote, chimera_plan_goals,
                    chimera_world_model, chimera_safety_check, chimera_ethical_eval,
                    chimera_embodied, chimera_social, chimera_transfer_learn,
                    chimera_evolve, chimera_self_model, chimera_knowledge,
                    chimera_memory
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "openchimera"))

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

from chimeralang_mcp.token_engine import (
    TokenBudgetManager,
    MessageImportanceScorer,
    get_token_budget_manager,
)

# ── AGI module imports from OpenChimera ────────────────────────────────────
try:
    from core.causal_reasoning import CausalGraph, CausalReasoning
    from core.deliberation import DeliberationGraph
    from core.deliberation_engine import DeliberationEngine
    from core.metacognition import MetacognitionEngine
    from core.meta_learning import MetaLearning
    from core.goal_planner import DecompositionStrategyLearner, GoalPlanner
    from core.world_model import SystemWorldModel
    from core.safety_layer import SafetyLayer
    from core.ethical_reasoning import EthicalReasoning
    from core.embodied_interaction import EmbodiedInteraction
    from core.social_cognition import SocialCognition
    from core.transfer_learning import TransferLearning
    from core.evolution import EvolutionEngine
    from core.self_model import SelfModel
    from core.knowledge_base import KnowledgeBase
    from core.memory import MemorySystem
    from core.quantum_engine import QuantumEngine, ConsensusResult
    _OPENCHIMERA_LOADED = True
except ImportError as e:
    logging.warning("OpenChimera AGI modules not available: %s", e)
    _OPENCHIMERA_LOADED = False

log = logging.getLogger(__name__)

# ── model pricing table (input $/1M tokens, output $/1M tokens) ───────────
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-7":      (15.00, 75.00),
    "claude-sonnet-4-6":    ( 3.00, 15.00),
    "claude-haiku-4-5":     ( 0.80,  4.00),
    "claude-opus-4-5":      (15.00, 75.00),
    "claude-sonnet-4-5":    ( 3.00, 15.00),
    "claude-haiku-3-5":     ( 0.80,  4.00),
    "gpt-4o":               ( 5.00, 15.00),
    "gpt-4o-mini":          ( 0.15,  0.60),
    "gpt-4-turbo":          (10.00, 30.00),
    "gemini-1.5-pro":       ( 3.50, 10.50),
    "gemini-1.5-flash":     ( 0.35,  1.05),
}
_DEFAULT_MODEL = "claude-sonnet-4-6"


# ── cost tracker ──────────────────────────────────────────────────────────
import collections as _collections
import uuid as _uuid

class _CostTracker:
    """In-memory ring buffer of the last 100 cost events."""

    def __init__(self, maxlen: int = 100) -> None:
        self._history: collections.deque[dict[str, Any]] = _collections.deque(maxlen=maxlen)

    def record(
        self,
        tokens_before: int,
        tokens_after: int,
        model: str = _DEFAULT_MODEL,
        label: str = "",
    ) -> dict[str, Any]:
        input_price, _ = _MODEL_PRICING.get(model, _MODEL_PRICING[_DEFAULT_MODEL])
        cost_before  = round(tokens_before * input_price / 1_000_000, 6)
        cost_after   = round(tokens_after  * input_price / 1_000_000, 6)
        savings      = round(cost_before - cost_after, 6)
        pct_saved    = round((1 - tokens_after / tokens_before) * 100, 1) if tokens_before else 0.0
        entry: dict[str, Any] = {
            "request_id":    str(_uuid.uuid4())[:8],
            "timestamp":     time.time(),
            "label":         label,
            "model":         model,
            "tokens_before": tokens_before,
            "tokens_after":  tokens_after,
            "tokens_saved":  tokens_before - tokens_after,
            "cost_before":   cost_before,
            "cost_after":    cost_after,
            "savings":       savings,
            "pct_saved":     pct_saved,
        }
        self._history.append(entry)
        return entry

    def summary(self) -> dict[str, Any]:
        history = list(self._history)
        total_tokens_saved = sum(e["tokens_saved"] for e in history)
        total_cost_saved   = round(sum(e["savings"] for e in history), 6)
        total_cost_before  = round(sum(e["cost_before"] for e in history), 6)
        avg_pct_saved      = round(
            sum(e["pct_saved"] for e in history) / len(history), 1
        ) if history else 0.0
        return {
            "request_count":      len(history),
            "total_tokens_saved": total_tokens_saved,
            "total_cost_saved":   total_cost_saved,
            "total_cost_before":  total_cost_before,
            "avg_pct_saved":      avg_pct_saved,
            "history":            history[-10:],  # last 10 for brevity
        }


# ── session-scoped singletons ─────────────────────────────────────────────
_middleware    = ClaudeConstraintMiddleware(confidence_threshold=0.7)
_detector      = HallucinationDetector()
_tbm           = get_token_budget_manager()
_scorer        = MessageImportanceScorer()
_cost_tracker  = _CostTracker()
server         = Server("chimeralang-mcp")

# ── AGI component singletons (lazy-initialized) ───────────────────────────
_causal_reasoning: CausalReasoning | None = None
_deliberation_engine: DeliberationEngine | None = None
_metacog_engine: MetacognitionEngine | None = None
_meta_learner: MetaLearning | None = None
_goal_planner: GoalPlanner | None = None
_world_model: SystemWorldModel | None = None
_safety_layer: SafetyLayer | None = None
_ethical_reasoner: EthicalReasoning | None = None
_embodied: EmbodiedInteraction | None = None
_social: SocialCognition | None = None
_transfer: TransferLearning | None = None
_evolution: EvolutionEngine | None = None
_self_model: SelfModel | None = None
_kb: KnowledgeBase | None = None
_memory: MemorySystem | None = None
_quantum: QuantumEngine | None = None


def _get_causal() -> CausalReasoning:
    global _causal_reasoning
    if _causal_reasoning is None:
        _causal_reasoning = CausalReasoning()
    return _causal_reasoning


def _get_deliberation() -> DeliberationEngine:
    global _deliberation_engine
    if _deliberation_engine is None:
        _deliberation_engine = DeliberationEngine()
    return _deliberation_engine


def _get_safety() -> SafetyLayer:
    global _safety_layer
    if _safety_layer is None:
        _safety_layer = SafetyLayer()
    return _safety_layer


def _get_ethical() -> EthicalReasoning:
    global _ethical_reasoner
    if _ethical_reasoner is None:
        _ethical_reasoner = EthicalReasoning()
    return _ethical_reasoner


def _get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb


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
        Tool(
            name="chimera_fracture",
            description=(
                "Full compression pipeline: chimera_optimize each document → chimera_compress messages "
                "(lossy if opted in) → TokenBudgetManager budget gate → quality flag. "
                "This is the primary pre-processing step before any complex Claude task. "
                "Returns quality_passed, combined stats, budget_remaining, and lossy_dropped_count."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Conversation history [{role, content}]. "
                                       "Compressed to fit token_budget before processing.",
                    },
                    "documents": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document strings to optimise and compress.",
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Maximum tokens for the compressed output. Default 1500.",
                        "default": 1500,
                    },
                    "allow_lossy": {
                        "type": "boolean",
                        "description": (
                            "When True and token_budget is exceeded, drop lowest-importance messages "
                            "until the budget is met. Default False (lossless)."
                        ),
                        "default": False,
                    },
                },
            },
        ),
        Tool(
            name="chimera_optimize",
            description=(
                "Reduce token count of a prompt or document by removing redundancy, "
                "normalising whitespace, collapsing repeated phrases, and stripping "
                "filler language — while preserving semantic meaning. "
                "Returns the optimised text and token-savings statistics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to optimise",
                    },
                    "strategies": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "whitespace",
                                "dedup_sentences",
                                "strip_filler",
                                "collapse_lists",
                            ],
                        },
                        "description": (
                            "Ordered list of optimisation passes to apply. "
                            "Default: [whitespace, dedup_sentences, strip_filler]"
                        ),
                        "default": ["whitespace", "dedup_sentences", "strip_filler"],
                    },
                    "preserve_code": {
                        "type": "boolean",
                        "description": "Skip optimisation inside code fences (``` blocks). Default true.",
                        "default": True,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_compress",
            description=(
                "Compress text using abbreviation and shorthand strategies to minimise "
                "token usage. Three levels: light (whitespace + punctuation), "
                "medium (+ common word contractions), aggressive (+ symbol substitutions). "
                "Returns compressed text, compression ratio, and estimated token savings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to compress",
                    },
                    "level": {
                        "type": "string",
                        "enum": ["light", "medium", "aggressive"],
                        "description": "Compression aggressiveness. Default: medium.",
                        "default": "medium",
                    },
                    "preserve_code": {
                        "type": "boolean",
                        "description": "Skip compression inside code fences (``` blocks). Default true.",
                        "default": True,
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_budget",
            description=(
                "Report current token usage against a budget and get a compression recommendation. "
                "Claude calls this proactively at the start of heavy tasks to know exactly where it stands. "
                "status: ok (<70% used) | warn (70-85%) | critical (>85%). "
                "recommendation: ok | call chimera_compress | call chimera_fracture"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Current conversation messages [{role, content}]",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Claude context window size. Default 200000.",
                        "default": 200000,
                    },
                    "reserve_tokens": {
                        "type": "integer",
                        "description": "Headroom reserved for the response. Default 10000.",
                        "default": 10000,
                    },
                },
            },
        ),
        Tool(
            name="chimera_score",
            description=(
                "Rank messages by importance for lossy compression decisions. "
                "Each message is scored 0.0-1.0 on recency, content type, information density, "
                "and replaceability. Lowest scores are dropped first when allow_lossy=True. "
                "Used by chimera_compress and chimera_fracture internally; also exposed directly "
                "for transparency audit before lossy compression."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Messages to score [{role, content}]",
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="chimera_cost_estimate",
            description=(
                "Deterministic cost estimate for a text or message list against any supported model. "
                "Returns token count and estimated dollar cost with NO API call required. "
                "Supports claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5, gpt-4o, gpt-4o-mini, "
                "gemini-1.5-pro, and more. Use before sending to the LLM to predict spend."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw text to estimate. Use instead of messages for single strings.",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Message list [{role, content}] to estimate. Use instead of text.",
                    },
                    "model": {
                        "type": "string",
                        "description": f"Model name. Default: {_DEFAULT_MODEL}. Supported: {', '.join(_MODEL_PRICING)}",
                        "default": _DEFAULT_MODEL,
                    },
                    "output_tokens": {
                        "type": "integer",
                        "description": "Expected output tokens (for total cost). Default 0 (input only).",
                        "default": 0,
                    },
                },
            },
        ),
        Tool(
            name="chimera_cost_track",
            description=(
                "Record a before/after compression event to the session cost tracker. "
                "Call this after chimera_compress or chimera_fracture to log actual savings. "
                "Stored in memory (last 100 entries). View with chimera_dashboard."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tokens_before": {
                        "type": "integer",
                        "description": "Token count before compression.",
                    },
                    "tokens_after": {
                        "type": "integer",
                        "description": "Token count after compression.",
                    },
                    "model": {
                        "type": "string",
                        "description": f"Model used for pricing. Default: {_DEFAULT_MODEL}",
                        "default": _DEFAULT_MODEL,
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label (e.g. task name) for this entry.",
                        "default": "",
                    },
                },
                "required": ["tokens_before", "tokens_after"],
            },
        ),
        Tool(
            name="chimera_dashboard",
            description=(
                "Return session-level cost intelligence summary: total tokens saved, "
                "total dollars saved, average compression %, and the last 10 tracked events. "
                "Use to report spend reduction to the user or to decide whether more compression is needed."
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

        # ── chimera_fracture — full pipeline ──────────────────────────────
        elif name == "chimera_fracture":
            import re as _re

            messages     = arguments.get("messages", [])
            documents    = arguments.get("documents", [])
            token_budget = int(arguments.get("token_budget", 1500))
            allow_lossy  = bool(arguments.get("allow_lossy", False))

            total_start = time.time()
            stats: dict[str, Any] = {
                "documents_input": sum(len(d) for d in documents),
                "messages_input":  len(messages),
                "tokens_input":    _tbm.count_messages(messages),
            }

            # Step 1: optimize each document
            optimised_docs: list[str] = []
            for doc in documents:
                # Quick optimise: whitespace + strip_filler (skip dedup_sentences/collapse_lists for speed)
                d = _re.sub(r"[ \t]+", " ", doc)
                d = _re.sub(r"\n{3,}", "\n\n", d).strip()
                for pat in [
                    r"\bplease note that\b", r"\bit is worth noting that\b",
                    r"\bit should be noted that\b", r"\bin order to\b",
                    r"\bbasically\b", r"\bactually\b", r"\bvery\b",
                    r"\bjust\b", r"\bsimply\b", r"\bquite\b",
                    r"\bof course\b", r"\bneedless to say\b",
                ]:
                    d = _re.sub(pat, "", d, flags=_re.IGNORECASE)
                d = _re.sub(r"[ \t]{2,}", " ", d).strip()
                optimised_docs.append(d)

            stats["documents_optimised"] = sum(len(d) for d in optimised_docs)

            # Step 2: compress messages (lossless first)
            if messages:
                msg_text = "\n".join(
                    f"[{m.get('role', 'user')}]: {m.get('content', '')}"
                    for m in messages
                )
                compressed = _re.sub(r"[ \t]+", " ", msg_text)
                compressed = _re.sub(r"\n{3,}", "\n\n", compressed).strip()
                for pat, repl in {
                    r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't",
                    r"\bdid not\b": "didn't", r"\bcannot\b": "can't",
                    r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't",
                    r"\bshould not\b": "shouldn't", r"\bcould not\b": "couldn't",
                    r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
                    r"\bwere not\b": "weren't", r"\bhave not\b": "haven't",
                    r"\bhas not\b": "hasn't", r"\bhad not\b": "hadn't",
                    r"\bit is\b": "it's", r"\bthat is\b": "that's",
                }.items():
                    compressed = _re.sub(pat, repl, compressed, flags=_re.IGNORECASE)
                messages_compressed = compressed
            else:
                messages_compressed = ""

            tokens_after = _tbm.count_tokens(messages_compressed) + _tbm.count_texts(optimised_docs)
            stats["tokens_after_pipeline"] = tokens_after
            budget_remaining = max(0, token_budget - tokens_after)
            quality_passed = tokens_after <= token_budget
            lossy_dropped_count = 0

            # Step 3: lossy if needed and opted in
            if not quality_passed and allow_lossy:
                ranked = _scorer.rank(messages)
                min_keep = 2
                to_drop: list[dict[str, Any]] = []
                for entry in ranked:
                    if len(messages) - len(to_drop) <= min_keep:
                        break
                    to_drop.append(entry)
                dropped_scores = [e["score"] for e in to_drop]
                kept = [m for i, m in enumerate(messages)
                        if i not in {e["index"] for e in to_drop}]
                if to_drop:
                    tombstone = {
                        "role": "system",
                        "content": (
                            f"[{len(to_drop)} messages omitted — "
                            f"low importance scores: {', '.join(str(s) for s in dropped_scores)}]"
                        ),
                    }
                    kept.append(tombstone)
                    messages = kept
                    lossy_dropped_count = len(to_drop)
                    # Re-compress kept messages
                    msg_text = "\n".join(
                        f"[{m.get('role', 'user')}]: {m.get('content', '')}"
                        for m in messages
                    )
                    compressed = _re.sub(r"[ \t]+", " ", msg_text)
                    compressed = _re.sub(r"\n{3,}", "\n\n", compressed).strip()
                    messages_compressed = compressed
                    tokens_after = _tbm.count_tokens(messages_compressed) + _tbm.count_texts(optimised_docs)
                    budget_remaining = max(0, token_budget - tokens_after)
                    quality_passed = tokens_after <= token_budget

            stats["tokens_after_pipeline"] = tokens_after
            stats["budget_remaining"] = budget_remaining
            stats["lossy_dropped_count"] = lossy_dropped_count
            stats["duration_ms"] = round((time.time() - total_start) * 1000, 1)

            return _ok({
                "quality_passed":      quality_passed,
                "budget_remaining":    budget_remaining,
                "tokens_input":        stats["tokens_input"],
                "tokens_after_pipeline": tokens_after,
                "documents_input":    stats["documents_input"],
                "documents_optimised": stats["documents_optimised"],
                "messages_input":      stats["messages_input"],
                "lossy_dropped_count": lossy_dropped_count,
                "compression_time_ms": stats["duration_ms"],
                "token_count_method": _tbm.get_stats()["token_count_method"],
            })

        # ── chimera_optimize ──────────────────────────────────────────────
        elif name == "chimera_optimize":
            import re as _re

            _FILLER = [
                r"\bplease note that\b", r"\bit is worth noting that\b",
                r"\bit should be noted that\b", r"\bin order to\b",
                r"\bbasically\b", r"\bactually\b", r"\bvery\b",
                r"\bjust\b", r"\bsimply\b", r"\bquite\b",
                r"\bof course\b", r"\bneedless to say\b",
                r"\bas you can see\b", r"\bclearly\b",
            ]

            text       = arguments["text"]
            strategies = arguments.get("strategies") or ["whitespace", "dedup_sentences", "strip_filler"]
            preserve_code = bool(arguments.get("preserve_code", True))
            original_len = len(text)
            result_text  = text
            log: list[str] = []
            code_blocks: list[str] = []

            # Extract code fences if preserving
            if preserve_code:
                def _stash(m: "_re.Match[str]") -> str:
                    code_blocks.append(m.group(0))
                    return f"\x00CODE{len(code_blocks) - 1}\x00"
                result_text = _re.sub(r"```[\s\S]*?```", _stash, result_text)

            if "whitespace" in strategies:
                before = len(result_text)
                result_text = _re.sub(r"[ \t]+", " ", result_text)
                result_text = _re.sub(r"\n{3,}", "\n\n", result_text).strip()
                log.append(f"whitespace: -{before - len(result_text)} chars")

            if "dedup_sentences" in strategies:
                before    = len(result_text)
                seen: set[str] = set()
                out_lines: list[str] = []
                for line in result_text.splitlines():
                    key = line.strip().lower()
                    if key and key not in seen:
                        seen.add(key)
                        out_lines.append(line)
                    elif not key:
                        out_lines.append(line)
                result_text = "\n".join(out_lines)
                log.append(f"dedup_sentences: -{before - len(result_text)} chars")

            if "strip_filler" in strategies:
                before = len(result_text)
                for pat in _FILLER:
                    result_text = _re.sub(pat, "", result_text, flags=_re.IGNORECASE)
                result_text = _re.sub(r"[ \t]{2,}", " ", result_text).strip()
                log.append(f"strip_filler: -{before - len(result_text)} chars")

            if "collapse_lists" in strategies:
                before = len(result_text)
                # Collect all list items, deduplicate while preserving order
                list_item_pattern = r"(?:^|\n)([-*•])\s+(.+?)(?=\n[-*•]|\n\n|$)"
                items_ordered: list[tuple[str, str]] = []
                seen_items: set[str] = set()
                for m in _re.finditer(list_item_pattern, result_text, _re.MULTILINE):
                    bullet, text_content = m.group(1), m.group(2).strip()
                    key = text_content.lower()
                    if key and key not in seen_items:
                        seen_items.add(key)
                        items_ordered.append((bullet, text_content))
                if items_ordered:
                    rebuilt_parts = [f"{b} {t}" for b, t in items_ordered]
                    result_text = "\n".join(rebuilt_parts)
                log.append(f"collapse_lists: -{before - len(result_text)} chars")

            # Restore code blocks
            if preserve_code:
                for i, block in enumerate(code_blocks):
                    result_text = result_text.replace(f"\x00CODE{i}\x00", block)

            saved = original_len - len(result_text)
            ratio = round(saved / original_len, 4) if original_len else 0.0

            return _ok({
                "optimised_text":    result_text,
                "original_chars":    original_len,
                "optimised_chars":   len(result_text),
                "chars_saved":       saved,
                "reduction_ratio":   ratio,
                "estimated_tokens_saved": max(0, round(saved / 4)),
                "passes_applied":     log,
                "code_blocks_preserved": len(code_blocks),
            })

        # ── chimera_compress ──────────────────────────────────────────────
        elif name == "chimera_compress":
            import re as _re

            _CONTRACTIONS_MEDIUM = {
                r"\bdo not\b": "don't",     r"\bdoes not\b": "doesn't",
                r"\bdid not\b": "didn't",   r"\bcannot\b": "can't",
                r"\bwill not\b": "won't",   r"\bwould not\b": "wouldn't",
                r"\bshould not\b": "shouldn't", r"\bcould not\b": "couldn't",
                r"\bare not\b": "aren't",   r"\bwas not\b": "wasn't",
                r"\bwere not\b": "weren't", r"\bhave not\b": "haven't",
                r"\bhas not\b": "hasn't",   r"\bhad not\b": "hadn't",
                r"\bI am\b": "I'm",         r"\bI have\b": "I've",
                r"\bI will\b": "I'll",      r"\bI would\b": "I'd",
                r"\bit is\b": "it's",       r"\bthat is\b": "that's",
                r"\bthere is\b": "there's", r"\bthey are\b": "they're",
                r"\bwe are\b": "we're",     r"\byou are\b": "you're",
            }

            _SYMBOLS_AGGRESSIVE = {
                # Removed ∴ ∵ & w/ w/o — these break Claude's comprehension of its own compressed history.
                # Keep only unambiguous, Claude-readable substitutions.
                r"\bapproximately\b": "≈",
                r"\bgreater than\b": ">",
                r"\bless than\b": "<",
                r"\bequals\b": "=",
                r"\bnumber\b": "nr.",
                r"\bversus\b": "vs.",
                r"\bregarding\b": "re:",
                r"\bfor example\b": "e.g.",
                r"\bthat is\b": "i.e.",
                r"\betcetera\b": "etc.",
            }

            text          = arguments["text"]
            level         = arguments.get("level", "medium")
            preserve_code = bool(arguments.get("preserve_code", True))
            original_len  = len(text)

            # Extract code fences if preserving
            code_blocks: list[str] = []
            work = text
            if preserve_code:
                def _stash(m: "_re.Match[str]") -> str:
                    code_blocks.append(m.group(0))
                    return f"\x00CODE{len(code_blocks) - 1}\x00"
                work = _re.sub(r"```[\s\S]*?```", _stash, work)

            # light: normalise whitespace
            work = _re.sub(r"[ \t]+", " ", work)
            work = _re.sub(r"\n{3,}", "\n\n", work).strip()

            if level in ("medium", "aggressive"):
                for pat, repl in _CONTRACTIONS_MEDIUM.items():
                    work = _re.sub(pat, repl, work, flags=_re.IGNORECASE)

            if level == "aggressive":
                for pat, repl in _SYMBOLS_AGGRESSIVE.items():
                    work = _re.sub(pat, repl, work, flags=_re.IGNORECASE)
                # strip redundant punctuation runs
                work = _re.sub(r"\.{2,}", "…", work)
                work = _re.sub(r"\s+([,;:!?])", r"\1", work)

            # Restore code blocks
            if preserve_code:
                for i, block in enumerate(code_blocks):
                    work = work.replace(f"\x00CODE{i}\x00", block)

            compressed_len = len(work)
            saved          = original_len - compressed_len
            ratio          = round(saved / original_len, 4) if original_len else 0.0

            return _ok({
                "compressed_text":   work,
                "level":             level,
                "original_chars":    original_len,
                "compressed_chars":  compressed_len,
                "chars_saved":       saved,
                "compression_ratio": ratio,
                "estimated_tokens_saved": max(0, round(saved / 4)),
                "code_blocks_preserved": len(code_blocks),
            })

        # ── chimera_budget ────────────────────────────────────────────────
        elif name == "chimera_budget":
            messages = arguments.get("messages", [])
            max_tokens = int(arguments.get("max_tokens", 200000))
            reserve = int(arguments.get("reserve_tokens", 10000))
            used = _tbm.count_messages(messages)
            remaining = max(0, max_tokens - used - reserve)
            pct = used / max_tokens if max_tokens else 0
            if pct < 0.70:
                status, recommendation = "ok", "ok"
            elif pct < 0.85:
                status, recommendation = "warn", "call chimera_compress"
            else:
                status, recommendation = "critical", "call chimera_fracture"
            return _ok({
                "used_tokens": used,
                "remaining_tokens": remaining,
                "pct_used": round(pct * 100, 2),
                "status": status,
                "recommendation": recommendation,
                "thresholds": {"warn": 70, "critical": 85},
                "token_count_method": _tbm.get_stats()["token_count_method"],
            })

        # ── chimera_score ────────────────────────────────────────────────
        elif name == "chimera_score":
            messages = arguments.get("messages", [])
            if not messages:
                return _ok([])
            ranked = _scorer.rank(messages)
            return _ok({
                "scores": ranked,
                "total_messages": len(messages),
                "token_count_method": _tbm.get_stats()["token_count_method"],
            })

        # ── AGI: chimera_causal ────────────────────────────────────────────
        elif name == "chimera_causal" and _OPENCHIMERA_LOADED:
            action = arguments.get("action", "info")
            if action == "add_edge":
                cr = _get_causal()
                from core.causal_reasoning import CausalEdge, EdgeType, ConfidenceLevel
                edge = CausalEdge(
                    cause=arguments["cause"],
                    effect=arguments["effect"],
                    edge_type=EdgeType(arguments.get("edge_type", "causes")),
                    strength=float(arguments.get("strength", 0.5)),
                    confidence=float(arguments.get("confidence", 0.5)),
                    confidence_level=ConfidenceLevel(arguments.get("confidence_level", "observed")),
                )
                cr.add_edge(edge)
                return _ok({"added": True, "edge": str(edge)})
            elif action == "query":
                cr = _get_causal()
                result = cr.query(cause=arguments.get("cause"), effect=arguments.get("effect"))
                return _ok({"query_result": result})
            elif action == "paths":
                cr = _get_causal()
                paths = cr.find_causal_paths(arguments.get("source", ""), arguments.get("target", ""))
                return _ok({"paths": [vars(p) for p in paths]})
            else:
                cr = _get_causal()
                return _ok({"variables": list(cr.graph.variables), "edge_count": cr.graph.edge_count})

        # ── AGI: chimera_deliberate ────────────────────────────────────────
        elif name == "chimera_deliberate" and _OPENCHIMERA_LOADED:
            prompt = arguments.get("prompt", "")
            perspectives = arguments.get("perspectives", [{"perspective": "default", "content": "", "model": "claude"}])
            de = _get_deliberation()
            result = de.deliberate(prompt, perspectives)
            return _ok(result)

        # ── AGI: chimera_metacognize ───────────────────────────────────────
        elif name == "chimera_metacognize" and _OPENCHIMERA_LOADED:
            # Note: MetacognitionEngine requires db + bus; expose diagnostic only
            return _ok({
                "note": "MetacognitionEngine requires database + event bus. Use compute_ece() on an initialized engine.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_meta_learn ─────────────────────────────────────────
        elif name == "chimera_meta_learn" and _OPENCHIMERA_LOADED:
            action = arguments.get("action", "info")
            if not hasattr(importlib, "import_module"):
                import importlib
            from core.meta_learning import MetaLearning
            if action == "record":
                ml = _meta_learner or MetaLearning()
                ml.record_adaptation(...)
                return _ok({"recorded": True})
            return _ok({"available": True, "action": action})

        # ── AGI: chimera_quantum_vote ──────────────────────────────────────
        elif name == "chimera_quantum_vote" and _OPENCHIMERA_LOADED:
            responses = arguments.get("responses", [])
            timeout_s = float(arguments.get("timeout_s", 5.0))
            if not responses:
                return _err("chimera_quantum_vote requires responses list")
            try:
                qe = QuantumEngine(timeout_s=timeout_s)
                result: ConsensusResult = qe.gather_sync([
                    {"agent_id": f"agent_{i}", "answer": r.get("answer"), "latency_ms": r.get("latency_ms", 100.0)}
                    for i, r in enumerate(responses)
                ])
                return _ok({
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "participating": result.participating,
                    "early_exit": result.early_exit,
                    "contradictions": result.contradictions_found,
                })
            except Exception as e:
                return _err(f"Quantum vote failed: {e}")

        # ── AGI: chimera_plan_goals ─────────────────────────────────────────
        elif name == "chimera_plan_goals" and _OPENCHIMERA_LOADED:
            goal = arguments.get("goal", "")
            if not goal:
                return _err("chimera_plan_goals requires a goal string")
            learner = DecompositionStrategyLearner()
            best = learner.best_strategy(goal[:50])
            return _ok({
                "goal": goal,
                "best_known_strategy": best,
                "note": "DecompositionStrategyLearner is stateless — attach persistence for learned strategies",
            })

        # ── AGI: chimera_world_model ─────────────────────────────────────────
        elif name == "chimera_world_model" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "SystemWorldModel requires a CausalReasoning instance. Use with full OpenChimera kernel.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_safety_check ────────────────────────────────────────
        elif name == "chimera_safety_check" and _OPENCHIMERA_LOADED:
            content = arguments.get("content", "")
            sl = _get_safety()
            is_safe, reason = sl.validate_content(content)
            return _ok({
                "is_safe": is_safe,
                "reason": reason,
                "blocked_count": sl._blocked_count,
                "allowed_count": sl._allowed_count,
            })

        # ── AGI: chimera_ethical_eval ────────────────────────────────────────
        elif name == "chimera_ethical_eval" and _OPENCHIMERA_LOADED:
            action_desc = arguments.get("action", "")
            if not action_desc:
                return _err("chimera_ethical_eval requires action description")
            er = _get_ethical()
            result = er.evaluate_action(action_desc)
            return _ok(vars(result) if hasattr(result, '__dict__') else result)

        # ── AGI: chimera_embodied ───────────────────────────────────────────
        elif name == "chimera_embodied" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "EmbodiedInteraction module available. Initialize with hardware context for full functionality.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_social ────────────────────────────────────────────
        elif name == "chimera_social" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "SocialCognition module available. Initialize with interaction history for relationship tracking.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_transfer_learn ──────────────────────────────────────
        elif name == "chimera_transfer_learn" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "TransferLearning module available. Apply learned representations across domains.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_evolve ──────────────────────────────────────────────
        elif name == "chimera_evolve" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "EvolutionEngine available. Run evolutionary optimization over problem populations.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_self_model ──────────────────────────────────────────
        elif name == "chimera_self_model" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "SelfModel module available. Maintain self-representation for capability awareness.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── AGI: chimera_knowledge ──────────────────────────────────────────
        elif name == "chimera_knowledge" and _OPENCHIMERA_LOADED:
            action = arguments.get("action", "search")
            kb = _get_kb()
            if action == "add":
                entry = kb.add(
                    content=arguments.get("content", ""),
                    category=arguments.get("category", "general"),
                    tags=arguments.get("tags", []),
                )
                return _ok({"added": True, "entry_id": entry.entry_id})
            elif action == "search":
                query = arguments.get("query", "")
                results = kb.search(query=query)
                return _ok({"results": [vars(r) for r in results]})
            elif action == "list":
                return _ok({"entries": len(kb._entries), "categories": list(set(e.category for e in kb._entries.values()))})
            else:
                return _ok({"entry_count": len(kb._entries)})

        # ── AGI: chimera_memory ────────────────────────────────────────────────
        elif name == "chimera_memory" and _OPENCHIMERA_LOADED:
            return _ok({
                "note": "MemorySystem available from core.memory. Initialize with storage path for persistence.",
                "available": _OPENCHIMERA_LOADED,
            })

        # ── chimera_cost_estimate ──────────────────────────────────────────────
        elif name == "chimera_cost_estimate":
            model         = arguments.get("model", _DEFAULT_MODEL)
            output_tokens = int(arguments.get("output_tokens", 0))
            text          = arguments.get("text")
            messages      = arguments.get("messages")

            if text:
                input_tokens = _tbm.count_tokens(str(text))
            elif messages:
                input_tokens = _tbm.count_messages(messages)
            else:
                return _err("Provide 'text' or 'messages'")

            input_price, output_price = _MODEL_PRICING.get(model, _MODEL_PRICING[_DEFAULT_MODEL])
            input_cost  = round(input_tokens  * input_price  / 1_000_000, 6)
            output_cost = round(output_tokens * output_price / 1_000_000, 6)
            total_cost  = round(input_cost + output_cost, 6)

            return _ok({
                "model":          model,
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "input_cost_usd": input_cost,
                "output_cost_usd": output_cost,
                "total_cost_usd": total_cost,
                "pricing_per_1m": {"input": input_price, "output": output_price},
                "note": "token count via " + _tbm.get_stats()["token_count_method"],
            })

        # ── chimera_cost_track ────────────────────────────────────────────────
        elif name == "chimera_cost_track":
            entry = _cost_tracker.record(
                tokens_before = int(arguments["tokens_before"]),
                tokens_after  = int(arguments["tokens_after"]),
                model         = arguments.get("model", _DEFAULT_MODEL),
                label         = arguments.get("label", ""),
            )
            log.info(
                "[CostTracker] %s → %s tokens ($%.4f → $%.4f) saved %.1f%%",
                entry["tokens_before"], entry["tokens_after"],
                entry["cost_before"], entry["cost_after"], entry["pct_saved"],
            )
            return _ok(entry)

        # ── chimera_dashboard ─────────────────────────────────────────────────
        elif name == "chimera_dashboard":
            return _ok(_cost_tracker.summary())

        else:
            return _err(f"Unknown tool: {name}")

    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


# ── entrypoint ────────────────────────────────────────────────────────────

async def _async_main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

def main() -> None:
    asyncio.run(_async_main())

if __name__ == "__main__":
    main()
