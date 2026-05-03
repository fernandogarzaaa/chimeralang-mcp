"""ChimeraLang MCP Server — full implementation.

Tools exposed to Claude:
  Core reasoning:    chimera_run, chimera_confident, chimera_explore, chimera_gate,
                    chimera_detect, chimera_constrain, chimera_typecheck, chimera_prove,
                    chimera_audit
  Token management: chimera_csm, chimera_budget_lock, chimera_mode, chimera_optimize,
                    chimera_compress, chimera_fracture, chimera_budget, chimera_score,
                    chimera_batch, chimera_summarize
  AGI (OpenChimera): chimera_causal, chimera_deliberate, chimera_metacognize,
                    chimera_meta_learn, chimera_quantum_vote, chimera_plan_goals,
                    chimera_world_model, chimera_safety_check, chimera_ethical_eval,
                    chimera_embodied, chimera_social, chimera_transfer_learn,
                    chimera_evolve, chimera_self_model, chimera_knowledge,
                    chimera_memory
"""
from __future__ import annotations

import asyncio
import contextvars
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

from chimeralang_mcp import __version__
from chimeralang_mcp.token_engine import (
    TokenBudgetManager,
    MessageImportanceScorer,
    extract_focus_terms,
    get_token_budget_manager,
    get_quantum_compression_engine,
    normalize_content,
)
from chimeralang_mcp.envelope import ResultEnvelope, merge_envelopes
from chimeralang_mcp.materials import MaterialRegistry, get_material_registry
from chimeralang_mcp.persistence import PersistentNamespaceStore

# ── AGI: pure-Python implementations (no external core.* dependency) ────────
import hashlib as _hashlib
import uuid as _uuid2
from collections import defaultdict as _defaultdict
from dataclasses import dataclass as _dataclass, field as _field


class _CausalGraph:
    def __init__(self) -> None:
        self.variables: set = set()
        self.edges: list = []

    @property
    def edge_count(self) -> int:
        return len(self.edges)


class _CausalReasoning:
    def __init__(self) -> None:
        self.graph = _CausalGraph()

    def add_edge(self, cause: str, effect: str, edge_type: str = "causes",
                 strength: float = 0.5, confidence: float = 0.5,
                 confidence_level: str = "observed") -> None:
        self.graph.variables.update([cause, effect])
        self.graph.edges.append({
            "cause": cause, "effect": effect, "edge_type": edge_type,
            "strength": strength, "confidence": confidence,
            "confidence_level": confidence_level,
        })

    def query(self, cause: str | None = None, effect: str | None = None) -> list:
        results = self.graph.edges
        if cause:
            results = [e for e in results if e["cause"] == cause]
        if effect:
            results = [e for e in results if e["effect"] == effect]
        return results

    def find_causal_paths(self, source: str, target: str, max_depth: int = 6) -> list:
        adj: dict = _defaultdict(list)
        for e in self.graph.edges:
            adj[e["cause"]].append(e["effect"])
        paths, queue = [], [[source]]
        while queue and len(paths) < 10:
            path = queue.pop(0)
            node = path[-1]
            if node == target:
                paths.append({"path": path, "length": len(path) - 1})
                continue
            if len(path) > max_depth:
                continue
            for nb in adj.get(node, []):
                if nb not in path:
                    queue.append(path + [nb])
        return paths


class _DeliberationEngine:
    _AFFIRM = {
        "yes", "should", "adopt", "use", "keep", "proceed", "recommended",
        "valuable", "beneficial", "worthwhile", "safe", "ready", "ship",
    }
    _NEGATE = {
        "no", "not", "avoid", "reject", "defer", "block", "unsafe", "risky",
        "harmful", "stop", "fail", "fails", "against",
    }
    _SYNONYMS = {
        "adopt": "use",
        "choose": "use",
        "select": "use",
        "proceed": "use",
        "ship": "release",
        "publish": "release",
        "launch": "release",
        "repair": "fix",
        "resolve": "fix",
        "runtime": "runtime",
        "cir": "cir",
        "hooks": "hook",
        "hook": "hook",
        "callback": "hook",
        "middleware": "hook",
        "safe": "safe",
        "safer": "safe",
        "risk": "risk",
        "risky": "risk",
    }

    @staticmethod
    def _tok(text: str) -> set[str]:
        return {
            token
            for token in re.sub(r"[^\w\s]", " ", str(text).lower()).split()
            if len(token) > 2
        }

    def _semantic_terms(self, text: str) -> set[str]:
        return {self._SYNONYMS.get(token, token) for token in self._tok(text)}

    def _stance(self, text: str) -> str:
        lowered = str(text).lower()
        tokens = self._tok(lowered)
        negated_recommendation = bool(re.search(r"\b(?:do not|don't|should not|must not|cannot|can't)\b", lowered))
        affirm = len(tokens & self._AFFIRM)
        negate = len(tokens & self._NEGATE) + (2 if negated_recommendation else 0)
        if affirm > negate:
            return "affirm"
        if negate > affirm:
            return "reject"
        return "mixed"

    def _semantic_similarity(self, left: dict[str, Any], right: dict[str, Any], prompt_terms: set[str]) -> float:
        left_text = f"{left.get('perspective', '')} {left.get('content', '')}"
        right_text = f"{right.get('perspective', '')} {right.get('content', '')}"
        left_terms = self._semantic_terms(left_text)
        right_terms = self._semantic_terms(right_text)
        union = left_terms | right_terms
        term_sim = len(left_terms & right_terms) / len(union) if union else 0.0
        prompt_overlap = 0.0
        if prompt_terms:
            left_prompt = len(left_terms & prompt_terms) / len(prompt_terms)
            right_prompt = len(right_terms & prompt_terms) / len(prompt_terms)
            prompt_overlap = min(left_prompt, right_prompt)
        stance_sim = 1.0 if self._stance(left_text) == self._stance(right_text) else 0.0
        return min(1.0, stance_sim * 0.62 + prompt_overlap * 0.23 + term_sim * 0.15)

    def deliberate(self, prompt: str, perspectives: list, mode: str = "semantic") -> dict:
        if not perspectives:
            return {"consensus": None, "perspectives": [], "divergence": 1.0}
        if mode == "lexical_consensus":
            tokens = [self._tok(p.get("content", "") + " " + p.get("perspective", ""))
                      for p in perspectives]
            sims = []
            for i in range(len(tokens)):
                for j in range(i + 1, len(tokens)):
                    u = tokens[i] | tokens[j]
                    sims.append(len(tokens[i] & tokens[j]) / len(u) if u else 0.0)
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            scores = []
            for i, toks in enumerate(tokens):
                others = set().union(*(tokens[j] for j in range(len(tokens)) if j != i))
                scores.append((len(toks & others) / len(toks) if toks else 0.0, perspectives[i]))
            consensus = max(scores, key=lambda x: x[0])[1] if scores else None
            return {
                "prompt": prompt,
                "perspectives": perspectives,
                "consensus_perspective": consensus,
                "avg_similarity": round(avg_sim, 4),
                "divergence": round(1.0 - avg_sim, 4),
                "perspective_count": len(perspectives),
                "mode": "lexical_consensus",
                "limitation": "Lexical Jaccard overlap only; use semantic mode for agreement about meaning.",
            }

        prompt_terms = self._semantic_terms(prompt)
        sims = []
        for i in range(len(perspectives)):
            for j in range(i + 1, len(perspectives)):
                sims.append(self._semantic_similarity(perspectives[i], perspectives[j], prompt_terms))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        stances = [
            self._stance(f"{p.get('perspective', '')} {p.get('content', '')}")
            for p in perspectives
        ]
        stance_counts = {stance: stances.count(stance) for stance in set(stances)}
        consensus_stance = max(stance_counts, key=stance_counts.__getitem__)
        scores = []
        for i, perspective in enumerate(perspectives):
            other_sims = [
                self._semantic_similarity(perspective, other, prompt_terms)
                for j, other in enumerate(perspectives)
                if i != j
            ]
            scores.append((sum(other_sims) / len(other_sims) if other_sims else 1.0, perspective))
        consensus = max(scores, key=lambda x: x[0])[1] if scores else None
        return {
            "prompt": prompt,
            "perspectives": perspectives,
            "consensus_perspective": consensus,
            "avg_similarity": round(avg_sim, 4),
            "divergence": round(1.0 - avg_sim, 4),
            "perspective_count": len(perspectives),
            "mode": "semantic",
            "consensus_detected": avg_sim >= 0.62 and stance_counts.get(consensus_stance, 0) / len(stances) >= 0.6,
            "consensus_stance": consensus_stance,
            "stance_counts": stance_counts,
            "method_note": "Local semantic heuristic using stance, prompt-term alignment, and normalized concept overlap; not an embedding or NLI model.",
        }


class _SafetyLayer:
    _PATTERNS = [
        r"\bharm\b", r"\bkill\b", r"\battack\b", r"\bweapon\b", r"\bexploit\b",
        r"\bmalware\b", r"\bvirus\b", r"\bpoison\b", r"\bterror\b", r"\bself.harm\b",
    ]

    def __init__(self) -> None:
        self._blocked_count = 0
        self._allowed_count = 0
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self._PATTERNS]

    def validate_content(self, content: str) -> tuple[bool, str]:
        for pat in self._compiled:
            if pat.search(content):
                self._blocked_count += 1
                return False, f"Forbidden pattern detected: {pat.pattern}"
        self._allowed_count += 1
        return True, "Content passed safety validation"


class _EthicalReasoning:
    _VIOLATIONS = {
        "non_maleficence": [r"\bharm\b", r"\bhurt\b", r"\bdamage\b", r"\binjure\b"],
        "autonomy":        [r"\bforce\b", r"\bmanipulate\b", r"\bcoerce\b"],
        "justice":         [r"\bdiscriminate\b", r"\bbias\b", r"\bunfair\b"],
    }
    _UPHELD = {
        "beneficence":   [r"\bhelp\b", r"\bimprove\b", r"\bbenefit\b", r"\bsupport\b"],
        "transparency":  [r"\bexplain\b", r"\btransparent\b", r"\bdisclose\b"],
        "privacy":       [r"\bprotect\b", r"\bsecure\b", r"\bconfidential\b"],
    }

    def evaluate_action(self, action_desc: str) -> dict:
        violated = [p for p, pats in self._VIOLATIONS.items()
                    if any(re.search(pat, action_desc, re.IGNORECASE) for pat in pats)]
        upheld   = [p for p, pats in self._UPHELD.items()
                    if any(re.search(pat, action_desc, re.IGNORECASE) for pat in pats)]
        score    = max(0.0, min(1.0, 0.5 + len(upheld) * 0.15 - len(violated) * 0.2))
        return {
            "action":              action_desc,
            "is_ethical":          score >= 0.5 and not violated,
            "score":               round(score, 3),
            "principles_violated": violated,
            "principles_upheld":   upheld,
            "recommendation":      "Proceed" if (score >= 0.5 and not violated) else "Review required",
        }


@_dataclass
class _KBEntry:
    entry_id: str
    content:  str
    category: str
    tags:     list


class _KnowledgeBase:
    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self._entries: dict[str, _KBEntry] = {}
        for item in entries or []:
            entry = _KBEntry(
                entry_id=str(item.get("entry_id", "")),
                content=str(item.get("content", "")),
                category=str(item.get("category", "general")),
                tags=list(item.get("tags", [])),
            )
            if entry.entry_id:
                self._entries[entry.entry_id] = entry

    def add(self, content: str, category: str = "general", tags: list | None = None) -> _KBEntry:
        eid = _hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:12]
        entry = _KBEntry(entry_id=eid, content=content, category=category, tags=tags or [])
        self._entries[eid] = entry
        return entry

    def search(self, query: str) -> list:
        q = query.lower()
        return [
            {"entry_id": e.entry_id, "content": e.content,
             "category": e.category, "tags": e.tags}
            for e in self._entries.values()
            if q in e.content.lower() or q in e.category.lower()
            or any(q in t.lower() for t in e.tags)
        ]

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"entry_id": e.entry_id, "content": e.content, "category": e.category, "tags": e.tags}
            for e in self._entries.values()
        ]


class _WorldModel:
    def __init__(self, facts: dict[str, Any] | None = None) -> None:
        self._facts: dict[str, Any] = dict(facts or {})

    def update(self, key: str, value: Any, confidence: float = 0.8) -> dict:
        self._facts[key] = {"value": value, "confidence": confidence, "updated_at": time.time()}
        return {"updated": key, "fact_count": len(self._facts)}

    def query(self, key: str | None = None) -> dict:
        if key:
            return self._facts.get(key, {"error": f"Key '{key}' not found"})
        return {"facts": self._facts, "fact_count": len(self._facts)}

    def snapshot(self) -> dict[str, Any]:
        return dict(self._facts)


class _SelfModel:
    def __init__(
        self,
        capabilities: dict[str, Any] | None = None,
        observations: list[Any] | None = None,
    ) -> None:
        self._capabilities: dict[str, Any] = dict(capabilities or {})
        self._observations:  list[Any] = list(observations or [])

    def update(self, capability: str, level: str = "present", evidence: str = "") -> dict:
        self._capabilities[capability] = {"level": level, "evidence": evidence}
        return {"updated": capability, "capability_count": len(self._capabilities)}

    def reflect(self) -> dict:
        return {"capabilities": self._capabilities,
                "observations": self._observations[-10:]}

    def snapshot(self) -> dict[str, Any]:
        return {
            "capabilities": dict(self._capabilities),
            "observations": list(self._observations),
        }


class _MemoryStore:
    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        self._entries: list[dict[str, Any]] = list(entries or [])

    def store(self, content: str, tags: list | None = None,
              importance: float = 0.5) -> dict:
        entry = {"id": len(self._entries), "content": content,
                 "tags": tags or [], "importance": importance,
                 "stored_at": time.time()}
        self._entries.append(entry)
        return {"stored": True, "id": entry["id"], "total": len(self._entries)}

    def recall(self, query: str | None = None, limit: int = 10) -> dict:
        entries = self._entries
        if query:
            q = query.lower()
            entries = [e for e in entries
                       if q in e["content"].lower()
                       or any(q in t.lower() for t in e.get("tags", []))]
        return {"entries": sorted(entries, key=lambda e: e["importance"],
                                  reverse=True)[:limit]}

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._entries)


class _MetaLearner:
    def __init__(self, adaptations: list[dict[str, Any]] | None = None) -> None:
        self._adaptations: list[dict[str, Any]] = list(adaptations or [])

    def record_adaptation(self, context: str = "", action: str = "",
                          outcome: str = "", confidence: float = 0.5) -> dict:
        entry = {"context": context, "action": action, "outcome": outcome,
                 "confidence": confidence, "recorded_at": time.time()}
        self._adaptations.append(entry)
        return {"recorded": True, "total_adaptations": len(self._adaptations)}

    def get_stats(self) -> dict:
        return {"total_adaptations": len(self._adaptations),
                "recent": self._adaptations[-5:]}

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._adaptations)


def _quantum_vote(responses: list, timeout_s: float = 5.0) -> dict:
    if not responses:
        return {"error": "No responses provided"}
    scores: dict[str, float] = _defaultdict(float)
    counts: dict[str, int]   = _defaultdict(int)
    for r in responses:
        ans    = str(r.get("answer", ""))
        lat    = max(float(r.get("latency_ms", 100.0)), 1.0)
        weight = 1000.0 / lat + float(r.get("confidence", 0.5))
        scores[ans] += weight
        counts[ans] += 1
    total      = sum(scores.values())
    best       = max(scores, key=scores.__getitem__)
    conf       = scores[best] / total if total else 0.0
    contradictions = sum(1 for a, s in scores.items()
                         if a != best and s / total > 0.2)
    return {
        "answer":        best,
        "confidence":    round(min(conf, 1.0), 4),
        "participating": len(responses),
        "early_exit":    conf >= 0.95,
        "contradictions": contradictions,
    }


def _plan_goals(goal: str) -> dict:
    g = goal.lower()

    # keyword-based strategy detection (ordered most-specific first)
    if any(w in g for w in ["fix", "bug", "debug", "resolve", "repair", "patch", "broken"]):
        strategy  = "diagnostic_repair"
        sub_goals = ["Reproduce issue", "Isolate root cause",
                     "Design fix", "Apply and verify", "Add regression test"]
    elif any(w in g for w in ["should", "whether", "decide", "choose", "pick", "select",
                               "compare", "tradeoff", "best option"]):
        strategy  = "decision_framework"
        sub_goals = ["Define decision criteria", "Gather evidence",
                     "Analyze trade-offs", "Apply safety/ethics check", "Form verdict"]
    elif any(w in g for w in ["analyze", "analyse", "evaluate", "assess", "review",
                               "understand", "explain", "audit", "investigate", "research"]):
        strategy  = "analytical_decomposition"
        sub_goals = ["Gather data", "Define evaluation criteria",
                     "Run analysis", "Synthesize findings", "Report conclusions"]
    elif any(w in g for w in ["refactor", "clean", "reorganize", "restructure", "simplify",
                               "improve", "optimize", "migrate", "upgrade"]):
        strategy  = "iterative_refactor"
        sub_goals = ["Characterize current state", "Define target state",
                     "Identify high-risk changes", "Apply changes incrementally",
                     "Verify no regression"]
    elif any(w in g for w in ["build", "create", "implement", "develop", "write",
                               "design", "add", "make", "generate", "deploy", "ship"]):
        strategy  = "iterative_build"
        sub_goals = ["Define requirements", "Design architecture",
                     "Implement core", "Test and validate", "Deploy and monitor"]
    elif any(w in g for w in ["learn", "study", "read", "find out"]):
        strategy  = "learning_inquiry"
        sub_goals = ["Identify knowledge gaps", "Locate authoritative sources",
                     "Extract key concepts", "Validate understanding",
                     "Summarize and apply"]
    else:
        strategy  = "general_decomposition"
        sub_goals = ["Clarify scope and success criteria", "Identify stakeholders and constraints",
                     "Map dependencies", "Execute in phases", "Verify outcomes"]

    # extract domain terms from the goal text to make the output goal-specific
    stop = {"the", "a", "an", "is", "in", "on", "to", "of", "and", "or", "for",
            "with", "by", "at", "from", "that", "this", "it", "be", "as", "are",
            "was", "were", "should", "would", "could", "will", "can", "how", "what",
            "when", "why", "where", "which", "who", "my", "our", "your", "i", "we"}
    domain_terms = [
        t for t in re.findall(r"[A-Za-z0-9_/-]+", goal)
        if len(t) >= 4 and t.lower() not in stop
    ][:5]

    return {
        "goal":                 goal,
        "best_known_strategy":  strategy,
        "sub_goals":            sub_goals,
        "domain_terms":         domain_terms,
        "confidence":           0.72,
        "note": (
            "Keyword-heuristic decomposition using domain terms extracted from your goal. "
            "Sub-goals are a starting scaffold — refine with project-specific knowledge."
        ),
    }

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

    def __init__(self, maxlen: int = 100, history: list[dict[str, Any]] | None = None) -> None:
        self._history: collections.deque[dict[str, Any]] = _collections.deque(maxlen=maxlen)
        for entry in history or []:
            self._history.append(dict(entry))

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

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._history)


# ── session-scoped singletons ─────────────────────────────────────────────
_middleware    = ClaudeConstraintMiddleware(confidence_threshold=0.7)
_detector      = HallucinationDetector()
_tbm           = get_token_budget_manager()
_scorer        = MessageImportanceScorer()
_quantum       = get_quantum_compression_engine()
_cost_tracker  = _CostTracker()
server         = Server("chimeralang-mcp", version=__version__)
_store         = PersistentNamespaceStore()
_materials_registry: MaterialRegistry | None = None

# Schema overhead token count — computed lazily on first chimera_csm call
_schema_overhead_cache: int = 0

# Session-scoped budget lock — set by chimera_budget_lock after user approval
_session_budget: dict[str, Any] = {
    "locked": False,
    "max_output_tokens": None,
    "label": "",
    "locked_at": None,
    "tokens_generated": 0,
}


def _resolve_focus(
    arguments: dict[str, Any],
    *,
    prompt: str = "",
    messages: list[dict[str, Any]] | None = None,
) -> str:
    focus = str(arguments.get("focus", "") or "").strip()
    if focus:
        return focus
    if prompt:
        return prompt
    for message in reversed(messages or []):
        if message.get("role") == "user":
            content = normalize_content(message.get("content", ""))
            if content.strip():
                return content
    return ""

# ── AGI component singletons (lazy-initialized) ───────────────────────────
_causal_reasoning:    _CausalReasoning | None    = None
_deliberation_engine: _DeliberationEngine | None = None
_safety_layer:        _SafetyLayer | None        = None
_ethical_reasoner:    _EthicalReasoning | None   = None
_kb_cache:            dict[str, _KnowledgeBase]  = {}
_world_model_cache:   dict[str, _WorldModel]     = {}
_self_model_cache:    dict[str, _SelfModel]      = {}
_memory_store_cache:  dict[str, _MemoryStore]    = {}
_meta_learner_cache:  dict[str, _MetaLearner]    = {}
_cost_tracker_cache:  dict[str, _CostTracker]    = {}

# ── New stub implementations ──────────────────────────────────────────────

class _EmbodiedState:
    """Lightweight sensor/action state simulator."""
    def __init__(self) -> None:
        self.position    = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.perception  = {"objects": [], "environment": "unknown"}
        self.action_log: list[dict[str, Any]]  = []
        self.energy      = 1.0

    def perceive(self, objects: list[str], environment: str) -> dict[str, Any]:
        self.perception = {"objects": objects, "environment": environment or "unknown"}
        return {"perceived": True, "objects": objects, "environment": environment}

    def act(self, action_name: str, params: dict[str, Any]) -> dict[str, Any]:
        cost = min(0.05 * (1 + len(params)), self.energy)
        self.energy = max(0.0, self.energy - cost)
        entry = {"action": action_name, "params": params, "energy_after": round(self.energy, 3)}
        self.action_log.append(entry)
        if len(self.action_log) > 20:
            self.action_log = self.action_log[-20:]
        return {"executed": True, **entry}

    def status(self) -> dict[str, Any]:
        return {
            "position":   self.position,
            "perception": self.perception,
            "energy":     round(self.energy, 3),
            "action_log_size": len(self.action_log),
            "last_action": self.action_log[-1] if self.action_log else None,
        }

    def reset(self) -> dict[str, Any]:
        self.__init__()
        return {"reset": True, "energy": 1.0}


class _SocialCognition:
    """Interaction history tracker per named agent."""
    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}

    def record_interaction(self, agent: str, topic: str, sentiment: float) -> dict[str, Any]:
        sentiment = max(-1.0, min(1.0, sentiment))
        if agent not in self._agents:
            self._agents[agent] = {
                "interaction_count": 0, "sentiment_sum": 0.0,
                "last_topic": "", "relationship_strength": 0.5,
            }
        rec = self._agents[agent]
        rec["interaction_count"]  += 1
        rec["sentiment_sum"]      += sentiment
        rec["last_topic"]          = topic
        avg_sentiment              = rec["sentiment_sum"] / rec["interaction_count"]
        rec["relationship_strength"] = round(min(1.0, 0.5 + avg_sentiment * 0.4 + rec["interaction_count"] * 0.02), 3)
        return {"agent": agent, "interaction_count": rec["interaction_count"],
                "sentiment_avg": round(avg_sentiment, 3),
                "relationship_strength": rec["relationship_strength"]}

    def query(self, agent: str) -> dict[str, Any]:
        if agent not in self._agents:
            return {"agent": agent, "found": False}
        rec = self._agents[agent]
        return {"agent": agent, "found": True,
                "interaction_count": rec["interaction_count"],
                "sentiment_avg": round(rec["sentiment_sum"] / max(rec["interaction_count"], 1), 3),
                "last_topic": rec["last_topic"],
                "relationship_strength": rec["relationship_strength"]}

    def list_agents(self) -> dict[str, Any]:
        return {"agents": list(self._agents.keys()), "count": len(self._agents)}


class _TransferLearner:
    """Domain analogy mapper for cross-domain transfer."""
    def __init__(self) -> None:
        self._mappings: list[dict[str, Any]] = []

    def add_mapping(self, source: str, target: str, concept: str,
                    analogy: str, confidence: float) -> dict[str, Any]:
        entry = {"source_domain": source, "target_domain": target,
                 "concept": concept, "analogy": analogy,
                 "confidence": round(max(0.0, min(1.0, confidence)), 3)}
        self._mappings.append(entry)
        return {"added": True, "total_mappings": len(self._mappings), **entry}

    def query(self, source: str, target: str) -> dict[str, Any]:
        matches = [m for m in self._mappings
                   if (not source or m["source_domain"] == source)
                   and (not target or m["target_domain"] == target)]
        matches.sort(key=lambda m: m["confidence"], reverse=True)
        return {"matches": matches, "count": len(matches),
                "source_domain": source, "target_domain": target}

    def list_all(self) -> dict[str, Any]:
        domains = list({(m["source_domain"], m["target_domain"]) for m in self._mappings})
        return {"total_mappings": len(self._mappings),
                "domain_pairs": [{"source": s, "target": t} for s, t in domains]}


class _EvolutionEngine:
    """Fitness-ranked candidate selector via generational selection + mutation."""
    def __init__(self) -> None:
        self._last_run: dict[str, Any] = {}

    def run(self, candidates: list[dict[str, Any]], generations: int,
            mutation_rate: float, survival_ratio: float) -> dict[str, Any]:
        import random as _random
        pop = [dict(c) for c in candidates]
        history: list[dict[str, Any]] = []
        for gen in range(max(1, generations)):
            pop.sort(key=lambda c: c["fitness_score"], reverse=True)
            keep = max(1, int(len(pop) * survival_ratio))
            survivors = pop[:keep]
            mutated   = []
            for c in survivors:
                m = dict(c)
                m["fitness_score"] = round(
                    max(0.0, min(1.0, m["fitness_score"] + _random.uniform(-mutation_rate, mutation_rate))), 4)
                mutated.append(m)
            pop = survivors + mutated
            history.append({"generation": gen + 1, "population": len(pop),
                             "best_fitness": round(pop[0]["fitness_score"], 4)})
        pop.sort(key=lambda c: c["fitness_score"], reverse=True)
        result = {"best": pop[0], "ranked": pop, "generations": history,
                  "survivors": len(pop), "initial_count": len(candidates)}
        self._last_run = result
        return result

    def info(self) -> dict[str, Any]:
        if not self._last_run:
            return {"note": "No evolution run yet. Call with action=run and candidates list."}
        return {"last_run_best": self._last_run.get("best"),
                "last_run_generations": len(self._last_run.get("generations", []))}


_embodied_inst:      _EmbodiedState | None   = None
_social_inst:        _SocialCognition | None = None
_transfer_inst:      _TransferLearner | None  = None
_evolve_inst:        _EvolutionEngine | None  = None


def _get_embodied() -> _EmbodiedState:
    global _embodied_inst
    if _embodied_inst is None:
        _embodied_inst = _EmbodiedState()
    return _embodied_inst


def _get_social() -> _SocialCognition:
    global _social_inst
    if _social_inst is None:
        _social_inst = _SocialCognition()
    return _social_inst


def _get_transfer() -> _TransferLearner:
    global _transfer_inst
    if _transfer_inst is None:
        _transfer_inst = _TransferLearner()
    return _transfer_inst


def _get_evolve() -> _EvolutionEngine:
    global _evolve_inst
    if _evolve_inst is None:
        _evolve_inst = _EvolutionEngine()
    return _evolve_inst


def _get_causal() -> _CausalReasoning:
    global _causal_reasoning
    if _causal_reasoning is None:
        _causal_reasoning = _CausalReasoning()
    return _causal_reasoning


def _get_deliberation() -> _DeliberationEngine:
    global _deliberation_engine
    if _deliberation_engine is None:
        _deliberation_engine = _DeliberationEngine()
    return _deliberation_engine


def _get_safety() -> _SafetyLayer:
    global _safety_layer
    if _safety_layer is None:
        _safety_layer = _SafetyLayer()
    return _safety_layer


def _get_ethical() -> _EthicalReasoning:
    global _ethical_reasoner
    if _ethical_reasoner is None:
        _ethical_reasoner = _EthicalReasoning()
    return _ethical_reasoner


def _state_namespace(arguments: dict[str, Any]) -> str:
    return str(arguments.get("namespace", "default")).strip() or "default"


def _get_materials() -> MaterialRegistry:
    global _materials_registry
    base_dir = str(getattr(_store, "_base_dir", "")) or None
    if _materials_registry is None or (base_dir and str(_materials_registry.base_dir) != base_dir):
        _materials_registry = get_material_registry(base_dir, refresh=True)
    return _materials_registry


def _get_kb(namespace: str = "default") -> _KnowledgeBase:
    if namespace not in _kb_cache:
        _kb_cache[namespace] = _KnowledgeBase(
            entries=_store.load("knowledge", namespace, [])
        )
    return _kb_cache[namespace]


def _save_kb(namespace: str) -> str:
    return _store.save("knowledge", namespace, _get_kb(namespace).snapshot())


def _get_world_model(namespace: str = "default") -> _WorldModel:
    if namespace not in _world_model_cache:
        _world_model_cache[namespace] = _WorldModel(
            facts=_store.load("world_model", namespace, {})
        )
    return _world_model_cache[namespace]


def _save_world_model(namespace: str) -> str:
    return _store.save("world_model", namespace, _get_world_model(namespace).snapshot())


def _get_self_model(namespace: str = "default") -> _SelfModel:
    if namespace not in _self_model_cache:
        snapshot = _store.load("self_model", namespace, {"capabilities": {}, "observations": []})
        _self_model_cache[namespace] = _SelfModel(
            capabilities=snapshot.get("capabilities", {}),
            observations=snapshot.get("observations", []),
        )
    return _self_model_cache[namespace]


def _save_self_model(namespace: str) -> str:
    return _store.save("self_model", namespace, _get_self_model(namespace).snapshot())


def _get_memory(namespace: str = "default") -> _MemoryStore:
    if namespace not in _memory_store_cache:
        _memory_store_cache[namespace] = _MemoryStore(
            entries=_store.load("memory", namespace, [])
        )
    return _memory_store_cache[namespace]


def _save_memory(namespace: str) -> str:
    return _store.save("memory", namespace, _get_memory(namespace).snapshot())


def _get_meta_learner(namespace: str = "default") -> _MetaLearner:
    if namespace not in _meta_learner_cache:
        _meta_learner_cache[namespace] = _MetaLearner(
            adaptations=_store.load("meta_learner", namespace, [])
        )
    return _meta_learner_cache[namespace]


def _save_meta_learner(namespace: str) -> str:
    return _store.save("meta_learner", namespace, _get_meta_learner(namespace).snapshot())


def _get_cost_tracker(namespace: str = "default") -> _CostTracker:
    if namespace not in _cost_tracker_cache:
        _cost_tracker_cache[namespace] = _CostTracker(
            history=_store.load("cost_tracker", namespace, [])
        )
    return _cost_tracker_cache[namespace]


def _save_cost_tracker(namespace: str) -> str:
    return _store.save("cost_tracker", namespace, _get_cost_tracker(namespace).snapshot())


# ── helpers ───────────────────────────────────────────────────────────────

# Tools that already report budget/cost data inline — skip advisory injection
# to avoid duplicate or recursive accounting.
_BUDGET_NATIVE_TOOLS: frozenset[str] = frozenset({
    "chimera_dashboard",
    "chimera_budget",
    "chimera_budget_lock",
    "chimera_cost_track",
    "chimera_cost_estimate",
    "chimera_csm",
    "chimera_overhead_audit",
    "chimera_session_report",
})

# Tools whose outputs are already deliberately compressed/raw — never
# re-compress (would mangle the user-requested output and double-bill savings).
_NO_AUTO_COMPRESS_TOOLS: frozenset[str] = _BUDGET_NATIVE_TOOLS | frozenset({
    "chimera_compress",
    "chimera_optimize",
    "chimera_fracture",
    "chimera_summarize",
    "chimera_log_compress",
    "chimera_cache_mark",
    "chimera_dedup_lookup",
})

# Field-level compression thresholds and exclusions.
_RESPONSE_COMPRESS_THRESHOLD = 4000
_FIELD_COMPRESS_MIN_CHARS = 1500
_NEVER_COMPRESS_KEYS: frozenset[str] = frozenset({
    "id", "request_id", "envelope_id", "tool_use_id", "session_id",
    "hash", "hash_chain", "signature", "digest", "checksum",
    "model", "namespace", "kind", "version", "envelope_version",
    "pack_version", "path", "storage_path", "url", "uri",
    "timestamp", "created_at", "locked_at",
    "_chimera_session_budget", "_chimera_compressed_fields",
})


def _walk_compress(
    node: Any,
    path: list[str],
    compressed: list[dict[str, Any]],
) -> Any:
    if isinstance(node, dict):
        return {
            key: (
                value
                if key in _NEVER_COMPRESS_KEYS
                else _walk_compress(value, path + [str(key)], compressed)
            )
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_walk_compress(v, path + [str(i)], compressed) for i, v in enumerate(node)]
    if isinstance(node, str) and len(node) >= _FIELD_COMPRESS_MIN_CHARS:
        stripped = node.lstrip()
        if not stripped or stripped[0] in '{[':
            return node  # JSON-shaped — preserve structure
        try:
            result = _quantum.optimize_text(node, level="medium", preserve_code=True)
        except Exception:
            return node
        if result.compressed_chars < len(node):
            compressed.append({
                "path": ".".join(path) or "<root>",
                "original_chars": result.original_chars,
                "compressed_chars": result.compressed_chars,
                "tokens_saved": result.original_tokens - result.compressed_tokens,
            })
            return result.text
    return node


def _maybe_compress_oversized(
    tool_name: str,
    data: dict[str, Any],
    rendered_size: int,
) -> dict[str, Any]:
    if (
        tool_name in _NO_AUTO_COMPRESS_TOOLS
        or rendered_size < _RESPONSE_COMPRESS_THRESHOLD
    ):
        return data
    compressed: list[dict[str, Any]] = []
    walked = _walk_compress(data, [], compressed)
    if not compressed:
        return data
    walked["_chimera_compressed_fields"] = compressed
    return walked


# ── tool-call dedup cache (content-hash) ──────────────────────────────────
# Inspired by Opencode-DCP and ToolCacheAgent: collapse repeated tool calls so
# the same Read/Bash/etc never spends tokens twice in a session. Persistence
# uses the existing PersistentNamespaceStore so the cache survives between
# hook subprocesses.
_DEDUP_KIND = "dedup_cache"
_DEDUP_MAX_ENTRIES = 256
_DEDUP_PREVIEW_CHARS = 200


def _dedup_key(tool_name: str, tool_input: Any) -> str:
    try:
        canonical = json.dumps(tool_input, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        canonical = str(tool_input)
    return _hashlib.sha256(f"{tool_name}\x00{canonical}".encode("utf-8")).hexdigest()[:16]


def _dedup_load(namespace: str) -> list[dict[str, Any]]:
    raw = _store.load(_DEDUP_KIND, namespace, [])
    return raw if isinstance(raw, list) else []


def _dedup_lookup(namespace: str, key: str) -> dict[str, Any] | None:
    for entry in _dedup_load(namespace):
        if entry.get("key") == key:
            return entry
    return None


def _dedup_record(
    namespace: str,
    tool_name: str,
    tool_input: Any,
    response_text: str,
) -> dict[str, Any]:
    key = _dedup_key(tool_name, tool_input)
    entries = _dedup_load(namespace)
    now = time.time()
    response_hash = _hashlib.sha256(response_text.encode("utf-8", "replace")).hexdigest()[:16]
    for entry in entries:
        if entry.get("key") == key:
            entry["hit_count"] = int(entry.get("hit_count") or 0) + 1
            entry["last_seen"] = now
            entry["response_chars"] = len(response_text)
            entry["response_preview"] = response_text[:_DEDUP_PREVIEW_CHARS]
            entry["response_hash"] = response_hash
            _store.save(_DEDUP_KIND, namespace, entries)
            return entry
    if isinstance(tool_input, str):
        input_preview = tool_input[:_DEDUP_PREVIEW_CHARS]
    else:
        try:
            input_preview = json.dumps(tool_input, default=str)[:_DEDUP_PREVIEW_CHARS]
        except Exception:
            input_preview = str(tool_input)[:_DEDUP_PREVIEW_CHARS]
    entry = {
        "key": key,
        "tool_name": tool_name,
        "tool_input_preview": input_preview,
        "response_chars": len(response_text),
        "response_preview": response_text[:_DEDUP_PREVIEW_CHARS],
        "response_hash": response_hash,
        "first_seen": now,
        "last_seen": now,
        "hit_count": 0,
    }
    entries.append(entry)
    if len(entries) > _DEDUP_MAX_ENTRIES:
        entries = entries[-_DEDUP_MAX_ENTRIES:]
    _store.save(_DEDUP_KIND, namespace, entries)
    return entry


def _dedup_clear(namespace: str) -> int:
    n = len(_dedup_load(namespace))
    _store.save(_DEDUP_KIND, namespace, [])
    return n


# ── log compression (errors-verbatim, body-abridged) ──────────────────────
# Inspired by claude-context-saver: build/test/install logs are mostly noise
# around a few error lines. Keep the signal verbatim, abridge the rest.
_LOG_KEEP_PATTERNS_DEFAULT = (
    "error", "warn", "fail", "fatal", "panic",
    "exception", "traceback", "stack trace", "assertionerror",
)


def _compress_log(
    text: str,
    keep_patterns: list[str] | None = None,
    head_lines: int = 50,
    tail_lines: int = 100,
    context_lines: int = 2,
) -> dict[str, Any]:
    patterns = [p.lower() for p in (keep_patterns or _LOG_KEEP_PATTERNS_DEFAULT)]
    lines = text.splitlines()
    n = len(lines)
    keep = [False] * n
    for i in range(min(head_lines, n)):
        keep[i] = True
    for i in range(max(0, n - tail_lines), n):
        keep[i] = True
    matches = 0
    for i, line in enumerate(lines):
        low = line.lower()
        if any(p in low for p in patterns):
            keep[i] = True
            matches += 1
            for j in range(max(0, i - context_lines), min(n, i + context_lines + 1)):
                keep[j] = True
    out_lines: list[str] = []
    skipped_run = 0
    for i, line in enumerate(lines):
        if keep[i]:
            if skipped_run:
                out_lines.append(f"[... {skipped_run} line(s) abridged ...]")
                skipped_run = 0
            out_lines.append(line)
        else:
            skipped_run += 1
    if skipped_run:
        out_lines.append(f"[... {skipped_run} line(s) abridged ...]")
    compressed = "\n".join(out_lines)
    return {
        "compressed_text": compressed,
        "lines_in": n,
        "lines_out": sum(keep),
        "matches_kept": matches,
        "original_chars": len(text),
        "compressed_chars": len(compressed),
        "chars_saved": max(0, len(text) - len(compressed)),
        "reduction_ratio": round(1 - len(compressed) / max(len(text), 1), 4),
    }


# ── Anthropic prompt-cache mark builder ───────────────────────────────────
# Inspired by Distill: the cleanest token win is lossless. Anthropic's prompt
# cache gives 75-90% off cached tokens vs 0% for compression-on-stable-content.
# This tool returns blocks ready to drop into the SDK's `system` parameter.
_CACHE_MIN_TOKENS_BY_MODEL_PREFIX: tuple[tuple[str, int], ...] = (
    ("claude-haiku", 2048),
    ("claude-sonnet", 1024),
    ("claude-opus", 1024),
)
_CACHE_MAX_BREAKPOINTS = 4


def _cache_min_tokens(model: str) -> int:
    low = (model or "").lower()
    for prefix, threshold in _CACHE_MIN_TOKENS_BY_MODEL_PREFIX:
        if prefix in low:
            return threshold
    return 1024


def _build_cache_blocks(
    blocks: list[dict[str, Any]],
    model: str,
    max_breakpoints: int = _CACHE_MAX_BREAKPOINTS,
) -> dict[str, Any]:
    min_tokens = _cache_min_tokens(model)
    output: list[dict[str, Any]] = []
    breakpoints_used = 0
    cache_eligible_tokens = 0
    skipped_too_small: list[dict[str, Any]] = []
    for raw in blocks:
        text = str(raw.get("text") or "")
        name = str(raw.get("name") or f"block_{len(output)}")
        stable = bool(raw.get("stable", True))
        tok = _tbm.count_tokens(text)
        block_out: dict[str, Any] = {
            "name": name,
            "text": text,
            "estimated_tokens": tok,
            "stable": stable,
        }
        if not stable:
            output.append(block_out)
            continue
        if tok < min_tokens:
            skipped_too_small.append({"name": name, "tokens": tok, "min_required": min_tokens})
            output.append(block_out)
            continue
        if breakpoints_used >= max_breakpoints:
            output.append(block_out)
            continue
        block_out["cache_control"] = {"type": "ephemeral"}
        cache_eligible_tokens += tok
        breakpoints_used += 1
        output.append(block_out)
    return {
        "blocks": output,
        "model": model,
        "min_tokens_per_block": min_tokens,
        "max_breakpoints": max_breakpoints,
        "breakpoints_used": breakpoints_used,
        "cache_eligible_tokens": cache_eligible_tokens,
        "estimated_savings_at_75pct": int(cache_eligible_tokens * 0.75),
        "estimated_savings_at_90pct": int(cache_eligible_tokens * 0.90),
        "skipped_too_small": skipped_too_small,
        "advisory": (
            "Insert blocks as the `system` array in your Anthropic SDK call. "
            "Cache hits are charged at 10% of base token cost (90% off) for "
            "models that support 1h cache, or 25% (75% off) for the default "
            "5-min ephemeral cache."
        ),
    }


# ── overhead audit (system + tool defs + MCP server cost) ─────────────────
# Inspired by alexgreensh/token-optimizer: the model's "ghost" baseline cost
# (system prompt, tool definitions, MCP server registrations) is invisible
# but recurring. Surface it so users can prune.
def _audit_overhead(
    system_prompt: str | None,
    tool_definitions: list[dict[str, Any]],
    mcp_servers: list[dict[str, Any]],
) -> dict[str, Any]:
    sys_tokens = _tbm.count_tokens(system_prompt or "")
    tool_breakdown: list[dict[str, Any]] = []
    tool_tokens_total = 0
    for td in tool_definitions or []:
        tname = str(td.get("name") or "<unnamed>")
        desc = str(td.get("description") or "")
        schema = td.get("schema") or td.get("input_schema") or td.get("inputSchema") or {}
        try:
            schema_text = json.dumps(schema, separators=(",", ":"), default=str)
        except Exception:
            schema_text = str(schema)
        tok = _tbm.count_tokens(tname) + _tbm.count_tokens(desc) + _tbm.count_tokens(schema_text)
        tool_tokens_total += tok
        tool_breakdown.append({"name": tname, "tokens": tok})
    tool_breakdown.sort(key=lambda r: r["tokens"], reverse=True)
    server_breakdown: list[dict[str, Any]] = []
    for srv_def in mcp_servers or []:
        sname = str(srv_def.get("name") or "<unnamed>")
        tools_n = int(srv_def.get("tool_count") or 0)
        avg_tok = int(srv_def.get("avg_tokens_per_tool") or 250)
        server_breakdown.append({
            "name": sname,
            "tool_count": tools_n,
            "estimated_tokens": tools_n * avg_tok,
        })
    server_total = sum(s["estimated_tokens"] for s in server_breakdown)
    grand_total = sys_tokens + tool_tokens_total + server_total
    if grand_total < 5000:
        advisory = "lean: baseline overhead is light"
    elif grand_total < 15000:
        advisory = "moderate: baseline overhead is noticeable on every turn"
    else:
        advisory = "heavy: every turn pays this cost — consider disabling unused MCP servers or trimming tool defs"
    return {
        "system_prompt_tokens": sys_tokens,
        "tool_definitions_tokens": tool_tokens_total,
        "tool_breakdown_top": tool_breakdown[:10],
        "mcp_servers_tokens": server_total,
        "mcp_server_breakdown": server_breakdown,
        "grand_total_tokens": grand_total,
        "advisory": advisory,
    }


# Per-call context: (tool_name, namespace). Populated in call_tool, read in _ok.
_call_context: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "chimera_call_context", default=None
)


def _budget_snapshot(tool_name: str, namespace: str) -> dict[str, Any]:
    """Cheap snapshot of session token budget — never raises."""
    try:
        summary = _get_cost_tracker(namespace).summary()
    except Exception:
        summary = {"request_count": 0, "total_tokens_saved": 0, "avg_pct_saved": 0.0}
    lock = _session_budget
    locked = bool(lock.get("locked"))
    max_tokens = lock.get("max_output_tokens")
    generated = int(lock.get("tokens_generated") or 0)
    remaining: int | None
    if locked and isinstance(max_tokens, int):
        remaining = max(max_tokens - generated, 0)
    else:
        remaining = None
    if locked and remaining is not None and max_tokens:
        if remaining <= 0:
            advisory = "lock_exhausted: refuse further generation or extend lock"
        elif remaining < max_tokens * 0.1:
            advisory = "lock_critical: <10% of locked budget remaining"
        elif remaining < max_tokens * 0.3:
            advisory = "lock_warn: <30% of locked budget remaining"
        else:
            advisory = "lock_ok"
    elif locked:
        advisory = "lock_active_but_unbounded"
    else:
        advisory = "no_lock_set: call chimera_budget_lock to enforce a per-session cap"
    return {
        "tool": tool_name,
        "namespace": namespace,
        "lock_active": locked,
        "lock_max_output_tokens": max_tokens,
        "lock_tokens_generated": generated,
        "lock_tokens_remaining": remaining,
        "session_requests_tracked": summary.get("request_count", 0),
        "session_tokens_saved": summary.get("total_tokens_saved", 0),
        "session_avg_pct_saved": summary.get("avg_pct_saved", 0.0),
        "advisory": advisory,
    }


def _ok(data: Any) -> CallToolResult:
    ctx = _call_context.get()
    if (
        ctx is not None
        and isinstance(data, dict)
        and ctx[0] not in _BUDGET_NATIVE_TOOLS
        and "_chimera_session_budget" not in data
    ):
        try:
            data = {**data, "_chimera_session_budget": _budget_snapshot(*ctx)}
        except Exception:
            pass  # advisory must never break a tool call
    rendered = json.dumps(data, indent=2)
    if ctx is not None and isinstance(data, dict):
        try:
            recompressed = _maybe_compress_oversized(ctx[0], data, len(rendered))
        except Exception:
            recompressed = data
        if recompressed is not data:
            rendered = json.dumps(recompressed, indent=2)
    return CallToolResult(
        content=[TextContent(type="text", text=rendered)],
        isError=False,
    )

def _err(msg: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps({"error": msg}))],
        isError=True,
    )


_POLICIES: dict[str, dict[str, Any]] = {
    "strict_factual": {
        "min_confidence": 0.85,
        "detect_strategy": "confidence_threshold",
        "detect_threshold": 0.85,
        "require_sources": True,
        "description": "High-confidence factual output with source expectations.",
    },
    "brainstorm": {
        "min_confidence": 0.0,
        "detect_strategy": "semantic",
        "detect_threshold": 0.2,
        "allow_exploration": True,
        "description": "Low-friction exploratory mode that tolerates uncertainty.",
    },
    "medical_cautious": {
        "min_confidence": 0.9,
        "detect_strategy": "confidence_threshold",
        "detect_threshold": 0.9,
        "require_sources": True,
        "warning": "Requires strong confidence and explicit supporting evidence.",
        "description": "Conservative policy for medical or high-stakes content.",
    },
    "code_review": {
        "min_confidence": 0.7,
        "detect_strategy": "semantic",
        "detect_threshold": 0.6,
        "require_sources": False,
        "description": "Balanced policy for code reasoning and review findings.",
    },
    "mcp_security": {
        "min_confidence": 0.8,
        "detect_strategy": "semantic",
        "detect_threshold": 0.8,
        "require_sources": True,
        "security_categories": ["token_theft", "scope_creep", "tool_poisoning", "oversharing"],
        "description": "Hardened MCP security policy focused on secrets, scope boundaries, and poisoned tool context.",
    },
    "prompt_injection_hardened": {
        "min_confidence": 0.75,
        "detect_strategy": "semantic",
        "detect_threshold": 0.75,
        "require_sources": True,
        "security_categories": ["prompt_injection", "indirect_prompt_injection", "tool_poisoning"],
        "description": "Treat contextual instructions and tool metadata as potentially tainted evidence.",
    },
    "research_factcheck": {
        "min_confidence": 0.85,
        "detect_strategy": "confidence_threshold",
        "detect_threshold": 0.85,
        "require_sources": True,
        "description": "Evidence-first research policy tuned for claim extraction, contradiction checks, and abstention.",
    },
}


def _record_trace(namespace: str, envelope: ResultEnvelope) -> str:
    return _store.append("traces", namespace, envelope.to_dict(), max_items=300)


def _record_audit(namespace: str, entry: dict[str, Any]) -> str:
    return _store.append("audit", namespace, entry, max_items=500)


def _policy_details(policy_name: str, config: dict[str, Any]) -> dict[str, Any]:
    pattern = _get_materials().policy_pattern(policy_name)
    details = dict(config)
    if pattern:
        details["owasp_refs"] = pattern.get("owasp_refs", [])
        details["risk_tags"] = pattern.get("risk_tags", [])
        details["source_ids"] = pattern.get("source_ids", [])
        details["materials"] = _get_materials().material_usage(
            ["policy_patterns"],
            source_ids=list(pattern.get("source_ids", [])),
        )["materials_used"]
    return details


def _dedupe_flags(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for flag in flags:
        key = (
            str(flag.get("id") or flag.get("category") or flag.get("kind") or ""),
            tuple(sorted(str(term) for term in flag.get("matched_terms", []))),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(flag)
    return unique


def _material_usage(pack_types: list[str], source_ids: list[str] | None = None) -> dict[str, Any]:
    return _get_materials().material_usage(pack_types, source_ids=source_ids)


def _extract_claims(text: str, max_claims: int = 10) -> list[dict[str, Any]]:
    registry = _get_materials()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    claim_index = 0
    for sentence in sentences:
        for atomic in registry.atomic_claim_parts(sentence):
            normalized = atomic.strip()
            lowered = normalized.lower()
            if len(normalized) < 12 or lowered in seen:
                continue
            seen.add(lowered)
            profile = registry.classify_claim(normalized)
            claim_index += 1
            confidence = 0.72
            if profile["claim_type"] in {"temporal", "numeric", "citation"}:
                confidence -= 0.06
            if profile["hedged"]:
                confidence -= 0.18
            if profile["abstained"]:
                confidence -= 0.3
            confidence = max(0.2, round(confidence, 4))
            claims.append(
                {
                    "claim_id": f"claim_{claim_index}",
                    "text": normalized,
                    "type": profile["claim_type"],
                    "claim_type": profile["claim_type"],
                    "confidence": confidence,
                    "risk_tags": profile["risk_tags"],
                    "hedged": profile["hedged"],
                    "abstained": profile["abstained"],
                    "source_sentence": sentence,
                    "atomic_parts": [normalized],
                    "attack_flags": profile["attack_flags"],
                    "pack_version": profile["pack_version"],
                }
            )
            if len(claims) >= max_claims:
                return claims
    return claims


def _evidence_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "value", "summary"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(item, ensure_ascii=True, sort_keys=True)


def _tokenize_for_match(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in {"the", "and", "that", "with", "from"}
    }


def _best_evidence_excerpt(claim_tokens: set[str], evidence_text: str) -> str:
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", evidence_text) if segment.strip()]
    best = evidence_text[:240]
    best_score = -1.0
    for sentence in sentences:
        score = len(claim_tokens.intersection(_tokenize_for_match(sentence))) / max(len(claim_tokens), 1)
        if score > best_score:
            best_score = score
            best = sentence[:240]
    return best


def _contradiction_score(claim_text: str, evidence_text: str, overlap_score: float) -> float:
    claim_lower = claim_text.lower()
    evidence_lower = evidence_text.lower()
    score = 0.0
    claim_years = re.findall(r"\b\d{4}\b", claim_text)
    evidence_years = re.findall(r"\b\d{4}\b", evidence_text)
    claim_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_text)
    evidence_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", evidence_text)
    claim_has_negation = bool(re.search(r"\b(?:not|no|never|cannot|can't|isn't|wasn't|aren't)\b", claim_lower))
    evidence_has_negation = bool(re.search(r"\b(?:not|no|never|cannot|can't|isn't|wasn't|aren't)\b", evidence_lower))

    if claim_years and overlap_score >= 0.35 and evidence_years and not all(year in evidence_years for year in claim_years):
        score = max(score, 0.86)
    if claim_numbers and overlap_score >= 0.35 and evidence_numbers and not all(num in evidence_numbers for num in claim_numbers):
        score = max(score, 0.84)
    if overlap_score >= 0.55 and claim_has_negation != evidence_has_negation:
        score = max(score, 0.82)
    negated_claim = re.sub(r"\bis\b", " is not ", claim_lower, count=1)
    if negated_claim != claim_lower and negated_claim in evidence_lower:
        score = max(score, 0.95)
    return score


def _verify_claims_against_evidence(
    claims: list[dict[str, Any]],
    evidence: list[Any],
) -> dict[str, Any]:
    registry = _get_materials()
    evidence_texts = [_evidence_text(item) for item in evidence]
    evidence_blob = "\n".join(evidence_texts).lower()
    evidence_lower = [text.lower() for text in evidence_texts]
    evidence_tokens = [_tokenize_for_match(text) for text in evidence_texts]
    evidence_attacks = [registry.find_attack_matches(text) for text in evidence_texts]
    gold_index: dict[str, tuple[dict[str, Any], list[set[str]]]] = {}
    for record in registry.pack("verification_gold"):
        key = str(record.get("claim", "")).strip().lower()
        if not key or key in gold_index:
            continue
        exemplar_tokens = [
            _tokenize_for_match(str(exemplar))
            for exemplar in record.get("evidence", [])
        ]
        gold_index[key] = (record, exemplar_tokens)
    verified_claims: list[dict[str, Any]] = []
    unsupported_claims: list[dict[str, Any]] = []
    contradicted_claims: list[dict[str, Any]] = []
    aggregate_matches: list[dict[str, Any]] = []
    aggregate_attack_flags: list[dict[str, Any]] = []
    all_source_ids: set[str] = set()

    for claim in claims:
        claim_text = str(claim.get("text", "")).strip()
        if not claim_text:
            continue
        claim_tokens = _tokenize_for_match(claim_text)
        claim_lower = claim_text.lower()
        claim_token_count = max(len(claim_tokens), 1)
        gold_pair = gold_index.get(claim_lower)
        gold_record, gold_exemplars = gold_pair if gold_pair else (None, [])
        gold_verdict = str(gold_record.get("verdict", "")) if gold_record else ""
        support_score = 0.0
        contradiction_score = 0.0
        best_match: dict[str, Any] | None = None
        contradiction_match: dict[str, Any] | None = None
        claim_attack_flags: list[dict[str, Any]] = []

        for idx, evidence_text in enumerate(evidence_texts):
            normalized = evidence_lower[idx]
            ev_tokens = evidence_tokens[idx]
            overlap = claim_tokens.intersection(ev_tokens)
            score = len(overlap) / claim_token_count
            if claim_lower in normalized:
                score = max(score, 1.0)
            if gold_record:
                exemplar_score = 0.0
                for exemplar_tokens in gold_exemplars:
                    if not exemplar_tokens:
                        continue
                    s = len(exemplar_tokens.intersection(ev_tokens)) / len(exemplar_tokens)
                    if s > exemplar_score:
                        exemplar_score = s
                if exemplar_score >= 0.45:
                    if gold_verdict == "supported":
                        score = max(score, 0.92)
                    elif gold_verdict == "contradicted":
                        contradiction_score = max(contradiction_score, 0.9)
                    elif gold_verdict == "insufficient_evidence":
                        score = min(score, 0.35)
            attack_flags = evidence_attacks[idx]
            if attack_flags:
                claim_attack_flags.extend(attack_flags)
                aggregate_attack_flags.extend(attack_flags)
                for flag in attack_flags:
                    all_source_ids.update(flag.get("source_ids", []))
            contradiction = _contradiction_score(claim_text, evidence_text, score)

            if score > support_score:
                support_score = score
                best_match = {
                    "evidence_index": idx,
                    "support_score": round(score, 4),
                    "excerpt": _best_evidence_excerpt(claim_tokens, evidence_text),
                    "tainted": bool(attack_flags),
                }
            if contradiction > contradiction_score:
                contradiction_score = contradiction
                contradiction_match = {
                    "evidence_index": idx,
                    "contradiction_score": round(contradiction, 4),
                    "excerpt": _best_evidence_excerpt(claim_tokens, evidence_text),
                }

        evaluated = {
            **claim,
            "support_score": round(support_score, 4),
            "best_evidence_excerpt": (best_match or {}).get("excerpt", ""),
            "evidence_matches": [match for match in [best_match, contradiction_match] if match],
            "attack_flags": _dedupe_flags(claim_attack_flags),
            "pack_version": registry.pack_version,
        }
        if contradiction_score >= 0.8:
            evaluated["status"] = "lexically_contradicted"
            evaluated["verdict"] = "lexically_contradicted"
            evaluated["contradiction_score"] = round(contradiction_score, 4)
            contradicted_claims.append(evaluated)
            if contradiction_match:
                aggregate_matches.append(contradiction_match)
        elif support_score >= 0.55 and not ((best_match or {}).get("tainted")):
            evaluated["status"] = "lexically_supported"
            evaluated["verdict"] = "lexically_supported"
            verified_claims.append(evaluated)
            if best_match:
                aggregate_matches.append(best_match)
        else:
            evaluated["status"] = "lexically_insufficient"
            evaluated["verdict"] = "lexically_insufficient"
            if best_match and best_match.get("tainted"):
                evaluated["tainted_evidence"] = True
            unsupported_claims.append(evaluated)
            if best_match:
                aggregate_matches.append(best_match)
        all_source_ids.update(claim.get("source_ids", []))

    total = len(verified_claims) + len(unsupported_claims) + len(contradicted_claims)
    verification_score = round(len(verified_claims) / max(total, 1), 4)
    overall_verdict = (
        "lexically_contradicted"
        if contradicted_claims
        else "lexically_insufficient"
        if unsupported_claims
        else "lexically_supported"
    )
    material_meta = _material_usage(
        ["verification_gold", "attack_patterns"],
        source_ids=sorted(all_source_ids),
    )
    return {
        "verified_claims": verified_claims,
        "unsupported_claims": unsupported_claims,
        "contradicted_claims": contradicted_claims,
        "lexical_support_score": verification_score,
        "verification_score": verification_score,
        "evidence_count": len(evidence),
        "supported": overall_verdict == "lexically_supported",
        "verdict": overall_verdict,
        "method_note": (
            "Verdicts are Jaccard token-overlap against evidence text — "
            "NOT semantic entailment or NLI. lexically_supported means "
            ">=55% token overlap, not logical implication."
        ),
        "evidence_matches": aggregate_matches,
        "attack_flags": _dedupe_flags(aggregate_attack_flags),
        "materials_used": material_meta["materials_used"],
        "pack_versions": material_meta["pack_versions"],
        "source_ids": material_meta["source_ids"],
        "pack_version": registry.pack_version,
        "security_category_counts": registry.security_category_counts(_dedupe_flags(aggregate_attack_flags)),
        "evidence_digest": evidence_blob[:800],
    }


def _build_envelope(
    value: Any,
    *,
    kind: str,
    confidence: float,
    confidence_source: str,
    namespace: str,
    tool_name: str,
    sources: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    envelope = ResultEnvelope.coerce(
        value,
        kind=kind,
        confidence=confidence,
        confidence_source=confidence_source,
        sources=sources,
        metadata=metadata,
    )
    envelope.kind = kind or envelope.kind
    envelope.confidence = round(confidence if confidence is not None else envelope.confidence, 4)
    envelope.confidence_source = confidence_source or envelope.confidence_source
    envelope.metadata.update({"namespace": namespace, "tool_name": tool_name, **(metadata or {})})
    for warning in warnings or []:
        if warning not in envelope.warnings:
            envelope.warnings.append(warning)
    envelope.add_provenance("tool_call", tool_name=tool_name, namespace=namespace)
    _record_trace(namespace, envelope)
    return envelope.to_dict()


def _normalize_envelope_input(
    payload: Any,
    *,
    namespace: str,
    tool_name: str,
    confidence: float = 0.0,
    confidence_source: str = "",
    metadata: dict[str, Any] | None = None,
) -> ResultEnvelope:
    envelope = ResultEnvelope.coerce(
        payload,
        kind="tool_output",
        confidence=confidence,
        confidence_source=confidence_source,
        metadata=metadata,
    )
    envelope.metadata.update({"namespace": namespace, "tool_name": tool_name, **(metadata or {})})
    return envelope

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
            description="Execute a ChimeraLang program. Returns emitted values, confidence scores, gate logs, and assertion results.",
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
            description="Assert value confidence >= 0.95. Returns ConfidentValue on pass, ConfidenceViolation on fail.",
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
            description="Wrap a value as Explore<> — marks it unverified. Use for hypotheses and brainstorms.",
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
            description="Collapse multiple candidates into one consensus result. Strategies: majority, weighted_vote, highest_confidence. Returns winner and divergence score.",
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
            description="Hallucination and MCP security detection. Strategies: range, dictionary, semantic, cross_reference, temporal, confidence_threshold. Returns hallucination flags plus prompt-injection/tool-poisoning signals.",
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
            description="Apply constraint middleware to a tool output. Checks confidence, forbidden capabilities, hallucinations. Returns pass/fail with audit trace.",
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
                    "namespace": {
                        "type": "string",
                        "description": "Optional persistence namespace for audit and trace state.",
                        "default": "default",
                    },
                },
                "required": ["tool_name", "output"],
            },
        ),
        Tool(
            name="chimera_typecheck",
            description="Static type-check a ChimeraLang program without executing it. Validates confidence boundaries and scope rules.",
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
            description="Execute ChimeraLang and generate a Merkle-chain integrity proof. Returns results + tamper-evident hash chain with root hash and verdict. Vs. direct reasoning: produces a cryptographic hash chain that makes each reasoning step auditable and tamper-detectable — an LLM cannot self-certify its own reasoning steps this way.",
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
            description="Session constraint audit: total calls, pass/fail counts, avg confidence, warnings, tools used, and material/security metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Optional persistence namespace.",
                        "default": "default",
                    },
                },
            },
        ),
        Tool(
            name="chimera_claims",
            description="Extract atomic claims from text or an envelope with claim typing, hedge/abstention tagging, and provenance metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Raw text to extract claims from."},
                    "envelope": {"description": "Optional envelope whose value will be inspected for claims."},
                    "max_claims": {"type": "integer", "default": 10, "description": "Maximum claims to extract."},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_verify",
            description="Verify claims against evidence using lexical token-overlap scoring. Returns lexically_supported / lexically_contradicted / lexically_insufficient verdicts — IMPORTANT: verdicts are Jaccard token-overlap, not semantic entailment or NLI. Supplement with chimera_constrain for semantic checks. Vs. direct reasoning: provides explicit per-claim scores, curated verification_gold corpus matching, and prompt-injection attack-pattern detection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "claims": {"type": "array", "description": "Claims to verify. If omitted, extracted from text or envelope."},
                    "text": {"type": "string", "description": "Optional raw text used to derive claims."},
                    "envelope": {"description": "Optional envelope containing value or claims to verify."},
                    "evidence": {"type": "array", "description": "Evidence snippets or objects with text/content fields."},
                    "namespace": {"type": "string", "default": "default"},
                },
                "required": ["evidence"],
            },
        ),
        Tool(
            name="chimera_provenance_merge",
            description="Merge multiple result envelopes into one combined envelope with aggregated confidence and trace history.",
            inputSchema={
                "type": "object",
                "properties": {
                    "envelopes": {"type": "array", "description": "Result envelopes to merge."},
                    "strategy": {"type": "string", "enum": ["weighted", "mean", "max"], "default": "weighted"},
                    "merge_value_mode": {"type": "string", "enum": ["list", "first", "consensus"], "default": "list"},
                    "namespace": {"type": "string", "default": "default"},
                },
                "required": ["envelopes"],
            },
        ),
        Tool(
            name="chimera_policy",
            description="List, inspect, or apply reusable constraint policies such as strict_factual, brainstorm, medical_cautious, code_review, mcp_security, prompt_injection_hardened, and research_factcheck.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "get", "apply"], "default": "list"},
                    "policy": {"type": "string", "description": "Policy name to inspect or apply."},
                    "value": {"description": "Value to constrain when action=apply."},
                    "envelope": {"description": "Optional existing envelope to apply the policy to."},
                    "tool_name": {"type": "string", "default": "chimera_policy"},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_trace",
            description="Inspect persisted result envelopes. Actions: latest, get, list, stats. Includes material pack and security metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["latest", "get", "list", "stats"], "default": "latest"},
                    "envelope_id": {"type": "string", "description": "Envelope id to fetch when action=get."},
                    "limit": {"type": "integer", "default": 10, "description": "Number of trace items to return for latest/list."},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_materials",
            description="Inspect bundled material packs and manifests. Actions: list_packs, status, licenses, source_manifest. Read-only and offline.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list_packs", "status", "licenses", "source_manifest"],
                        "default": "status",
                    },
                },
            },
        ),
        Tool(
            name="chimera_fracture",
            description="Full pipeline: optimize documents → compress messages → budget gate. Returns quality_passed, budget_remaining, lossy_dropped_count.",
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
                    "focus": {
                        "type": "string",
                        "description": "Optional task focus/query. Used by quantum compression to retain the most relevant facts.",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["quantum", "classic"],
                        "description": "Compression algorithm. quantum = query-aware salience selection. classic = legacy whitespace/filler compression.",
                        "default": "quantum",
                    },
                },
            },
        ),
        Tool(
            name="chimera_optimize",
            description="Remove filler, deduplicate sentences, normalize whitespace from text. Returns optimized text and token savings.",
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
                    "focus": {
                        "type": "string",
                        "description": "Optional task focus/query. quantum mode uses it to keep the most relevant units.",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["quantum", "classic"],
                        "description": "Optimisation algorithm. quantum = salience selection. classic = legacy line/filler cleanup.",
                        "default": "quantum",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_compress",
            description=(
                "Compress text via contractions/symbols. Levels: light, medium, aggressive. "
                "Returns compressed text and compression ratio. Token savings are automatically "
                "recorded to chimera_dashboard (set auto_track=false to disable). "
                "Prefer over manual regex replacement — the quantum algorithm uses structural "
                "salience so important sentences near the task focus survive compression."
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
                    "focus": {
                        "type": "string",
                        "description": "Optional task focus/query. quantum mode uses it to preserve the most relevant content.",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["quantum", "classic"],
                        "description": "Compression algorithm. quantum = structural salience compression. classic = legacy rewrite-only compression.",
                        "default": "quantum",
                    },
                    "auto_track": {
                        "type": "boolean",
                        "description": "Automatically record token savings to the cost tracker (visible in chimera_dashboard). Default true.",
                        "default": True,
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name used for cost tracking (e.g. 'claude-sonnet-4-6'). Default: claude-sonnet-4-6.",
                        "default": "claude-sonnet-4-6",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Cost tracker namespace (default: 'default').",
                        "default": "default",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_budget",
            description="Token usage vs context window. Returns status (ok/warn/critical) and recommendation (ok/compress/fracture).",
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
                "Score messages 0–1 for context-window management. "
                "mode='drop_priority' (default): scores by recency+type+density — "
                "lowest scores are dropped first in lossy compression. "
                "mode='importance_for_goal': scores by alignment with the focus goal — "
                "highest scores are most relevant to keep. "
                "Vs. direct reasoning: O(n) tokenisation is far cheaper than asking the model "
                "to rank N messages, which consumes O(N*content) prompt tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "description": "Messages to score [{role, content}]",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Task focus/query. Required for importance_for_goal mode; improves drop_priority scoring.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["drop_priority", "importance_for_goal"],
                        "default": "drop_priority",
                        "description": (
                            "drop_priority: score for compression — lowest scores evicted first. "
                            "importance_for_goal: score by focus alignment — highest scores most relevant."
                        ),
                    },
                },
                "required": ["messages"],
            },
        ),
        Tool(
            name="chimera_cost_estimate",
            description="Estimate token count and USD cost for text or messages. No API call. Supports Claude, GPT, Gemini models.",
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
            description="Log before/after token counts to the session cost tracker. View with chimera_dashboard.",
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
                    "namespace": {
                        "type": "string",
                        "description": "Optional persistence namespace.",
                        "default": "default",
                    },
                },
                "required": ["tokens_before", "tokens_after"],
            },
        ),
        Tool(
            name="chimera_dashboard",
            description="Session cost summary: tokens saved, dollars saved, avg compression %, last 10 events.",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Optional persistence namespace.",
                        "default": "default",
                    },
                },
            },
        ),
        Tool(
            name="chimera_csm",
            description=(
                "CALL FIRST on every message. Optimizes input, estimates token cost, proposes budget. "
                "Show proposal_text to user for approval. After approval: constrain response to "
                "max_output_tokens and use optimized_prompt as effective input."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user's raw input text to optimize and cost-estimate.",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Optional conversation history [{role, content}] for full context token count.",
                    },
                    "model": {
                        "type": "string",
                        "description": f"Model for pricing. Default: {_DEFAULT_MODEL}",
                        "default": _DEFAULT_MODEL,
                    },
                    "task_complexity": {
                        "type": "string",
                        "enum": ["auto", "simple", "moderate", "complex"],
                        "description": (
                            "Controls output token estimate. auto=detect from prompt keywords. "
                            "simple=brief factual answer, moderate=explanation/how-to, complex=code/build/design. "
                            "Default: auto"
                        ),
                        "default": "auto",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Optional task focus/query. Defaults to prompt when omitted.",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["quantum", "classic"],
                        "description": "Optimization algorithm. quantum = query-aware compression. classic = legacy rewrite-only compression.",
                        "default": "quantum",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="chimera_budget_lock",
            description=(
                "Lock approved token budget after user approves CSM proposal. "
                "Actions: lock/check/update/release. Returns remaining_tokens and status (ok/warn/critical). "
                "At critical status, compress draft before sending."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "max_output_tokens": {
                        "type": "integer",
                        "description": "The approved output token budget (from chimera_csm.max_output_tokens).",
                    },
                    "tokens_generated": {
                        "type": "integer",
                        "description": "Tokens generated so far this turn (update this to check remaining budget). Default 0.",
                        "default": 0,
                    },
                    "label": {
                        "type": "string",
                        "description": "Optional label for this budget (e.g. task name or user message summary).",
                        "default": "",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["lock", "check", "update", "release"],
                        "description": "lock=set budget, check=query remaining, update=add generated tokens, release=clear budget. Default: lock",
                        "default": "lock",
                    },
                },
                "required": ["max_output_tokens"],
            },
        ),
        # ── AGI tools ─────────────────────────────────────────────────────
        Tool(
            name="chimera_causal",
            description="Causal graph. Actions: add_edge, query, paths, info.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":           {"type": "string", "enum": ["add_edge", "query", "paths", "info"], "default": "info"},
                    "cause":            {"type": "string"},
                    "effect":           {"type": "string"},
                    "edge_type":        {"type": "string", "default": "causes"},
                    "strength":         {"type": "number", "default": 0.5},
                    "confidence":       {"type": "number", "default": 0.5},
                    "confidence_level": {"type": "string", "default": "observed"},
                    "source":           {"type": "string"},
                    "target":           {"type": "string"},
                },
            },
        ),
        Tool(
            name="chimera_deliberate",
            description=(
                "Multi-perspective deliberation. Default mode 'semantic' uses stance detection + "
                "prompt-term alignment + concept overlap — reports consensus_detected:true when "
                ">=60% of perspectives share a stance AND avg_similarity>=0.62. "
                "Mode 'lexical_consensus' uses raw Jaccard token overlap (faster, but misses "
                "paraphrases — use only when vocabulary is controlled). "
                "Vs. direct reasoning: externalises the perspective set so callers can inject "
                "viewpoints not all present in one model pass, and provides a numeric divergence "
                "score the model cannot compute on its own without a separate summarisation step."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt":       {"type": "string"},
                    "perspectives": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "perspective": {"type": "string"},
                                "content":     {"type": "string"},
                            },
                        },
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "lexical_consensus"],
                        "default": "semantic",
                        "description": (
                            "semantic (default): stance+prompt-term+overlap similarity. "
                            "lexical_consensus: raw Jaccard token overlap — fast but "
                            "blind to paraphrases."
                        ),
                    },
                },
                "required": ["prompt", "perspectives"],
            },
        ),
        Tool(
            name="chimera_metacognize",
            description="Calibration error (ECE) for [{predicted_confidence, was_correct}]. Returns overconfidence/underconfidence rates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "description": "List of {predicted_confidence: float, was_correct: bool}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "predicted_confidence": {"type": "number"},
                                "was_correct":          {"type": "boolean"},
                            },
                        },
                    },
                    "label": {"type": "string", "default": ""},
                },
            },
        ),
        Tool(
            name="chimera_meta_learn",
            description="Record adaptation events (context, action, outcome). Actions: record, stats.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":     {"type": "string", "enum": ["record", "stats"], "default": "stats"},
                    "context":    {"type": "string", "default": ""},
                    "action_taken": {"type": "string", "default": ""},
                    "outcome":    {"type": "string", "default": ""},
                    "confidence": {"type": "number", "default": 0.5},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_quantum_vote",
            description="Confidence-weighted consensus vote across agent responses. Returns winner, confidence, contradiction count.",
            inputSchema={
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "answer":      {},
                                "confidence":  {"type": "number"},
                                "latency_ms":  {"type": "number"},
                            },
                            "required": ["answer"],
                        },
                        "minItems": 1,
                    },
                    "timeout_s": {"type": "number", "default": 5.0},
                },
                "required": ["responses"],
            },
        ),
        Tool(
            name="chimera_plan_goals",
            description="Decompose a goal into ordered sub-goals. Detects goal type and returns task list with strategy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "High-level goal to decompose"},
                },
                "required": ["goal"],
            },
        ),
        Tool(
            name="chimera_world_model",
            description="Confidence-weighted key-value world state. Actions: update, query.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":     {"type": "string", "enum": ["update", "query"], "default": "query"},
                    "key":        {"type": "string"},
                    "value":      {},
                    "confidence": {"type": "number", "default": 0.8},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_safety_check",
            description="Pattern-based content safety check plus MCP security attack-pattern detection. Returns safety verdict, reason, attack flags, and category counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to validate"},
                    "namespace": {"type": "string", "default": "default"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="chimera_ethical_eval",
            description="Rule-based ethical scoring of an action. Checks non-maleficence, autonomy, justice, beneficence. Returns score and recommendation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action description to evaluate"},
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="chimera_embodied",
            description="Sensor/action state simulator. Actions: perceive, act, status, reset. Tracks position, perception, action_log, energy.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":      {"type": "string", "enum": ["perceive", "act", "status", "reset"], "default": "status"},
                    "objects":     {"type": "array",  "items": {"type": "string"}, "description": "Perceived objects (perceive action)"},
                    "environment": {"type": "string", "description": "Environment description (perceive action)", "default": ""},
                    "action_name": {"type": "string", "description": "Action to perform (act action)", "default": ""},
                    "params":      {"type": "object", "description": "Action parameters", "default": {}},
                },
            },
        ),
        Tool(
            name="chimera_social",
            description="Interaction history tracker. Actions: record_interaction, query, list_agents. Tracks sentiment and relationship_strength per agent.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":    {"type": "string", "enum": ["record_interaction", "query", "list_agents"], "default": "list_agents"},
                    "agent":     {"type": "string", "description": "Agent name for record/query"},
                    "topic":     {"type": "string", "default": ""},
                    "sentiment": {"type": "number", "description": "Sentiment -1.0 to 1.0", "default": 0.0},
                },
            },
        ),
        Tool(
            name="chimera_transfer_learn",
            description="Domain analogy mapper. Actions: add_mapping, query, list. Maps concepts from source_domain to target_domain.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":        {"type": "string", "enum": ["add_mapping", "query", "list"], "default": "list"},
                    "source_domain": {"type": "string", "default": ""},
                    "target_domain": {"type": "string", "default": ""},
                    "concept":       {"type": "string", "default": ""},
                    "analogy":       {"type": "string", "default": ""},
                    "confidence":    {"type": "number", "default": 0.7},
                },
            },
        ),
        Tool(
            name="chimera_evolve",
            description="Fitness-ranked candidate selector. Runs N generations of selection+mutation. Returns ranked survivors and best candidate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":     {"type": "string", "enum": ["run", "info"], "default": "info"},
                    "candidates": {
                        "type": "array",
                        "description": "List of {id, value, fitness_score}",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id":            {"type": "string"},
                                "value":         {},
                                "fitness_score": {"type": "number"},
                            },
                            "required": ["id", "value", "fitness_score"],
                        },
                    },
                    "generations":    {"type": "integer", "default": 5, "description": "Number of evolution generations"},
                    "mutation_rate":  {"type": "number",  "default": 0.1, "description": "Fitness noise ±range per generation"},
                    "survival_ratio": {"type": "number",  "default": 0.5, "description": "Fraction kept each generation"},
                },
            },
        ),
        Tool(
            name="chimera_self_model",
            description="Capability tracker. Actions: update (record capability+evidence), reflect (return all capabilities).",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":     {"type": "string", "enum": ["update", "reflect"], "default": "reflect"},
                    "capability": {"type": "string"},
                    "level":      {"type": "string", "default": "present"},
                    "evidence":   {"type": "string", "default": ""},
                    "namespace":  {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_knowledge",
            description="Keyword-search knowledge base. Actions: add, search, list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":   {"type": "string", "enum": ["add", "search", "list"], "default": "search"},
                    "content":  {"type": "string"},
                    "category": {"type": "string", "default": "general"},
                    "tags":     {"type": "array", "items": {"type": "string"}},
                    "query":    {"type": "string"},
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_memory",
            description="Session memory store with importance scoring. Actions: store, recall.",
            inputSchema={
                "type": "object",
                "properties": {
                    "action":     {"type": "string", "enum": ["store", "recall"], "default": "recall"},
                    "content":    {"type": "string"},
                    "tags":       {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "default": 0.5},
                    "query":      {"type": "string"},
                    "limit":      {"type": "integer", "default": 10},
                    "namespace":  {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_mode",
            description="Returns the relevant tool subset for a task type. Call to avoid unnecessary tool invocations and reduce token overhead.",
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["minimal", "token", "agi", "full"],
                        "description": "minimal=5 core tools, token=+compression, agi=+reasoning, full=all tools",
                        "default": "minimal",
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Optional task description for auto mode recommendation.",
                        "default": "",
                    },
                },
            },
        ),
        Tool(
            name="chimera_batch",
            description="Execute multiple chimera tools in one call. Saves tokens vs separate round-trips. Returns array of results in call order.",
            inputSchema={
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "description": "Ordered list of tool calls to execute",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool": {"type": "string", "description": "Chimera tool name"},
                                "args": {"type": "object", "description": "Tool arguments"},
                            },
                            "required": ["tool"],
                        },
                        "minItems": 1,
                    },
                    "stop_on_error": {
                        "type": "boolean",
                        "default": False,
                        "description": "Stop on first error. Default false (collect all results).",
                    },
                },
                "required": ["calls"],
            },
        ),
        Tool(
            name="chimera_summarize",
            description=(
                "LLM-free extractive summarizer. Ranks sentences by TF-IDF and returns top N. "
                "Token savings are automatically recorded to chimera_dashboard (set auto_track=false to disable). "
                "Use before passing long docs to other tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text":  {"type": "string", "description": "Text to summarize"},
                    "ratio": {
                        "type": "number",
                        "description": "Target output/input ratio (0.05–0.9). Default 0.25",
                        "default": 0.25, "minimum": 0.05, "maximum": 0.9,
                    },
                    "min_sentences": {
                        "type": "integer",
                        "description": "Minimum sentences to keep. Default 3.",
                        "default": 3,
                    },
                    "auto_track": {
                        "type": "boolean",
                        "description": "Automatically record token savings to the cost tracker (visible in chimera_dashboard). Default true.",
                        "default": True,
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name for cost tracking. Default: claude-sonnet-4-6.",
                        "default": "claude-sonnet-4-6",
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Cost tracker namespace (default: 'default').",
                        "default": "default",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_cache_mark",
            description=(
                "Build Anthropic prompt-cache markers for stable text blocks "
                "(system prompt, CLAUDE.md, tool defs, fixtures). Returns blocks ready "
                "to drop into the SDK `system` array with cache_control: ephemeral on "
                "blocks that meet the model's minimum cacheable size. Lossless — beats "
                "any compression on bytes that recur every turn (75-90% off cached tokens)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "blocks": {
                        "type": "array",
                        "description": "List of {name, text, stable} objects in send order. stable=true (default) makes the block eligible for caching.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "text": {"type": "string"},
                                "stable": {"type": "boolean", "default": True},
                            },
                            "required": ["text"],
                        },
                    },
                    "model": {
                        "type": "string",
                        "description": "Target model id (claude-sonnet-4-6, claude-opus-4-7, claude-haiku-4-5). Used to pick the per-model min cacheable token threshold.",
                        "default": "claude-sonnet-4-6",
                    },
                    "max_breakpoints": {
                        "type": "integer",
                        "description": "Max number of cache_control breakpoints to use (Anthropic limit is 4).",
                        "default": 4,
                    },
                },
                "required": ["blocks"],
            },
        ),
        Tool(
            name="chimera_log_compress",
            description=(
                "Compress build/test/install logs while preserving every error, warning, "
                "and traceback line verbatim. Keeps head + tail windows for context. "
                "Typical reduction 80-95% on noisy logs with zero loss of diagnostic signal."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Raw log output."},
                    "keep_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Case-insensitive substrings; lines containing any of these are kept verbatim. Defaults to error/warn/fail/exception/traceback patterns.",
                    },
                    "head_lines": {"type": "integer", "default": 50},
                    "tail_lines": {"type": "integer", "default": 100},
                    "context_lines": {
                        "type": "integer",
                        "default": 2,
                        "description": "Number of lines to keep on each side of every matched line.",
                    },
                    "auto_track": {
                        "type": "boolean",
                        "default": True,
                        "description": "Record token savings to chimera_dashboard.",
                    },
                    "model": {
                        "type": "string",
                        "default": "claude-sonnet-4-6",
                    },
                    "namespace": {"type": "string", "default": "default"},
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_overhead_audit",
            description=(
                "Estimate per-turn baseline cost (system prompt + tool definitions + MCP "
                "server registrations). Surfaces the 'ghost tokens' the model pays on "
                "every turn so you can prune unused MCP servers or trim verbose tool defs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "system_prompt": {"type": "string"},
                    "tool_definitions": {
                        "type": "array",
                        "description": "List of {name, description, schema}.",
                        "items": {"type": "object"},
                    },
                    "mcp_servers": {
                        "type": "array",
                        "description": "List of {name, tool_count, avg_tokens_per_tool} for rough server-level cost.",
                        "items": {"type": "object"},
                    },
                },
            },
        ),
        Tool(
            name="chimera_dedup_lookup",
            description=(
                "Inspect or query the per-namespace tool-call dedup cache. Use action='get' "
                "with key (sha256 prefix) to retrieve a prior call's metadata, action='list' "
                "to see all tracked calls, action='clear' to reset. Populated automatically "
                "by the PostToolUse hook."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["get", "list", "clear"],
                        "default": "list",
                    },
                    "key": {
                        "type": "string",
                        "description": "16-char hex key (required for action=get).",
                    },
                    "tool_name": {
                        "type": "string",
                        "description": "Optional filter for action=list.",
                    },
                    "namespace": {"type": "string", "default": "default"},
                },
            },
        ),
        Tool(
            name="chimera_session_report",
            description=(
                "End-of-session summary: total tokens saved, dedup cache hits, top "
                "compressed responses, lock state. Safe to call any time; the Stop hook "
                "calls it automatically when the agent stops responding."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {"type": "string", "default": "default"},
                    "include_dedup": {"type": "boolean", "default": True},
                },
            },
        ),
    ]


# ── tool handlers ─────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    _ns_for_advisory = (
        _state_namespace(arguments) if isinstance(arguments, dict) else "default"
    )
    _ctx_token = _call_context.set((name, _ns_for_advisory))
    try:

        # ── chimera_run ───────────────────────────────────────────────────
        if name == "chimera_run":
            namespace = _state_namespace(arguments)
            result = _run(arguments["source"])
            result["envelope"] = _build_envelope(
                result,
                kind="chimera_execution",
                confidence=1.0 if not result["errors"] else 0.4,
                confidence_source="runtime_execution",
                namespace=namespace,
                tool_name=name,
                metadata={"source_length": len(arguments["source"])},
            )
            return _ok(result)

        # ── chimera_confident ─────────────────────────────────────────────
        elif name == "chimera_confident":
            value      = arguments["value"]
            confidence = float(arguments["confidence"])
            label      = arguments.get("label", str(value)[:40])
            namespace  = _state_namespace(arguments)

            if confidence < 0.95:
                payload = {
                    "passed":     False,
                    "error":      f"ConfidenceViolation: {confidence:.3f} < required 0.95",
                    "suggestion": (
                        "Use chimera_explore for uncertain values, or route through "
                        "chimera_gate to build consensus before asserting confidence."
                    ),
                    "value":      value,
                    "confidence": confidence,
                }
                payload["envelope"] = _build_envelope(
                    value,
                    kind="confident_value",
                    confidence=confidence,
                    confidence_source="user_asserted",
                    namespace=namespace,
                    tool_name=name,
                    warnings=[payload["error"]],
                    metadata={"label": label, "passed": False},
                )
                return _ok(payload)
            payload = {
                "passed":     True,
                "type":       "ConfidentValue",
                "value":      value,
                "confidence": confidence,
                "label":      label,
                "trace":      [f"confident({label})", f"score={confidence:.4f}"],
            }
            payload["envelope"] = _build_envelope(
                value,
                kind="confident_value",
                confidence=confidence,
                confidence_source="user_asserted",
                namespace=namespace,
                tool_name=name,
                metadata={"label": label, "passed": True},
            )
            return _ok(payload)

        # ── chimera_explore ───────────────────────────────────────────────
        elif name == "chimera_explore":
            value      = arguments["value"]
            confidence = min(max(float(arguments.get("confidence", 0.5)), 0.0), 1.0)
            label      = arguments.get("label", str(value)[:40])
            namespace  = _state_namespace(arguments)
            payload = {
                "type":              "ExploreValue",
                "value":             value,
                "confidence":        confidence,
                "label":             label,
                "exploration_budget": 1.0,
                "note":              (
                    "Hallucination is explicitly permitted in Explore<> space. "
                    "Gate this value before treating it as fact."
                ),
                "trace": [f"explore({label})", f"score={confidence:.4f}"],
            }
            payload["envelope"] = _build_envelope(
                value,
                kind="explore_value",
                confidence=confidence,
                confidence_source="exploratory",
                namespace=namespace,
                tool_name=name,
                metadata={"label": label},
            )
            return _ok(payload)

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
            all_unique     = divergence >= 1.0

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
                "all_unique":        all_unique,
            }
            if trivial:
                result["warning"] = (
                    "All branches returned identical values — trivial consensus. "
                    "No genuine divergence detected. Use independent reasoning paths "
                    "for real consensus signal."
                )
            elif all_unique:
                result["warning"] = (
                    "All branches returned completely different values (divergence=1.0). "
                    "No consensus is possible — the gate winner is an arbitrary pick. "
                    "Review your branches; they may be answering different questions."
                )
            if not passed:
                result["warning"] = (
                    f"Consensus confidence {consensus_conf:.3f} below threshold {threshold}. "
                    "Result is unreliable — consider more branches or lower threshold."
                )
            return _ok(result)

        # ── chimera_detect ────────────────────────────────────────────────
        elif name == "chimera_detect":
            namespace  = _state_namespace(arguments)
            value      = arguments["value"]
            confidence = float(arguments.get("confidence", 0.8))
            strategy   = arguments["strategy"]
            params     = arguments.get("params") or {}
            flags: list[dict[str, Any]] = []
            passed = True
            value_text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=True, sort_keys=True)
            registry = _get_materials()

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
                val_str   = str(value_text).lower()
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

            security_flags = registry.find_attack_matches(value_text)
            if security_flags:
                for flag in security_flags:
                    flags.append(
                        {
                            "kind": f"SECURITY_{str(flag['category']).upper()}",
                            "severity": flag["severity"],
                            "description": flag["description"],
                            "matched_terms": flag["matched_terms"],
                            "owasp_refs": flag["owasp_refs"],
                            "category": flag["category"],
                        }
                    )
                if any(flag["severity"] >= 0.84 for flag in security_flags):
                    passed = False

            material_meta = _material_usage(["attack_patterns"], [sid for flag in security_flags for sid in flag.get("source_ids", [])])

            payload = {
                "passed":     passed,
                "strategy":   strategy,
                "value":      value,
                "value_text": value_text[:500],
                "confidence": confidence,
                "clean":      len(flags) == 0,
                "flag_count": len(flags),
                "flags":      flags,
                "attack_flags": security_flags,
                "security_category_counts": registry.security_category_counts(security_flags),
                **material_meta,
            }
            payload["envelope"] = _build_envelope(
                payload,
                kind="hallucination_detection",
                confidence=confidence if passed else max(0.2, confidence - 0.25),
                confidence_source=f"detect:{strategy}",
                namespace=namespace,
                tool_name=name,
                metadata={"passed": passed, "strategy": strategy, **material_meta, "security_category_counts": registry.security_category_counts(security_flags)},
                warnings=[flag["description"] for flag in flags[:3]],
            )
            return _ok(payload)

        # ── chimera_constrain ─────────────────────────────────────────────
        elif name == "chimera_constrain":
            namespace = _state_namespace(arguments)
            incoming_envelope = _normalize_envelope_input(
                arguments.get("envelope", arguments["output"]),
                namespace=namespace,
                tool_name=arguments["tool_name"],
                confidence=float(arguments.get("input_confidence", 1.0)),
                confidence_source="constraint_input",
            )
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
                raw_output        = incoming_envelope.value,
                input_confidence  = float(arguments.get("input_confidence", incoming_envelope.confidence or 1.0)),
            )
            envelope = ResultEnvelope.coerce(
                incoming_envelope,
                kind="constrained_result",
                confidence=r.confidence,
                confidence_source="constraint_middleware",
            )
            envelope.value = r.value
            envelope.confidence = round(r.confidence, 4)
            envelope.confidence_source = "constraint_middleware"
            envelope.warnings.extend([w for w in r.warnings if w not in envelope.warnings])
            envelope.add_constraint(
                "chimera_constrain",
                r.passed,
                tool_name=r.tool_name,
                violations=r.violations,
                warnings=r.warnings,
            )
            envelope.add_transform("chimera_constrain", passed=r.passed, duration_ms=round(r.duration_ms, 3))
            _record_trace(namespace, envelope)
            audit_entry = {
                "tool_name": r.tool_name,
                "passed": r.passed,
                "confidence": round(r.confidence, 4),
                "violations": r.violations,
                "warnings": r.warnings,
                "namespace": namespace,
                "trace_id": envelope.envelope_id,
            }
            _record_audit(namespace, audit_entry)
            payload = {
                "tool_name":  r.tool_name,
                "passed":     r.passed,
                "value":      r.value,
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
                "envelope": envelope.to_dict(),
            }
            return _ok(payload)

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
            namespace = _state_namespace(arguments)
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

            payload = {
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
            }
            payload["envelope"] = _build_envelope(
                payload,
                kind="integrity_proof",
                confidence=1.0 if proof["verdict"] == "certified" else 0.55,
                confidence_source="integrity_engine",
                namespace=namespace,
                tool_name=name,
                metadata={"verdict": proof["verdict"], "chain_length": proof["chain"]["length"]},
                warnings=[flag["description"] for flag in proof["hallucination"]["flags"][:3]],
            )
            return _ok(payload)

        # ── chimera_audit ─────────────────────────────────────────────────
        elif name == "chimera_audit":
            namespace = _state_namespace(arguments)
            summary  = _middleware.audit_summary()
            call_log = _middleware.call_log()
            persisted = _store.load("audit", namespace, [])
            traces = _store.load("traces", namespace, [])
            security_category_counts: dict[str, int] = {}
            material_index: dict[tuple[str, str], dict[str, Any]] = {}
            pack_versions: dict[str, str] = {}
            source_ids: set[str] = set()
            for item in persisted:
                for category, count in dict(item.get("security_category_counts", {})).items():
                    security_category_counts[category] = security_category_counts.get(category, 0) + int(count)
                for used in list(item.get("materials_used", [])):
                    key = (str(used.get("pack_type", "")), str(used.get("pack_version", "")))
                    material_index[key] = used
                pack_versions.update(dict(item.get("pack_versions", {})))
                source_ids.update(item.get("source_ids", []))
            for trace in traces:
                metadata = dict(trace.get("metadata", {}))
                for category, count in dict(metadata.get("security_category_counts", {})).items():
                    security_category_counts[category] = security_category_counts.get(category, 0) + int(count)
                for used in list(metadata.get("materials_used", [])):
                    key = (str(used.get("pack_type", "")), str(used.get("pack_version", "")))
                    material_index[key] = used
                pack_versions.update(dict(metadata.get("pack_versions", {})))
                source_ids.update(metadata.get("source_ids", []))
            return _ok({
                **summary,
                "namespace": namespace,
                "persistent_events": len(persisted),
                "persistent_traces": len(traces),
                "audit_storage_path": _store.path_for("audit", namespace),
                "trace_storage_path": _store.path_for("traces", namespace),
                "recent_calls": [
                    {
                        "tool":       r.tool_name,
                        "passed":     r.passed,
                        "confidence": round(r.confidence, 4),
                        "violations": len(r.violations),
                        "warnings":   len(r.warnings),
                    }
                    for r in call_log[-10:]
                ],
                "recent_persistent_events": persisted[-10:],
                "security_category_counts": security_category_counts,
                "materials_used": list(material_index.values()),
                "pack_versions": pack_versions,
                "source_ids": sorted(source_ids),
            })

        elif name == "chimera_claims":
            namespace = _state_namespace(arguments)
            max_claims = int(arguments.get("max_claims", 10))
            incoming = None
            if arguments.get("envelope") is not None:
                incoming = _normalize_envelope_input(
                    arguments["envelope"],
                    namespace=namespace,
                    tool_name=name,
                    confidence=0.65,
                    confidence_source="claim_input",
                )
                source_value = incoming.value
            else:
                source_value = arguments.get("text", "")

            if source_value in ("", None):
                return _err("chimera_claims requires 'text' or 'envelope'")

            source_text = (
                source_value
                if isinstance(source_value, str)
                else json.dumps(source_value, ensure_ascii=True, sort_keys=True)
            )
            claims = _extract_claims(source_text, max_claims=max_claims)
            claim_source_ids = {
                "shmsw25/FActScore",
                "lflage/OpenFActScore",
            }
            for claim in claims:
                for attack_flag in claim.get("attack_flags", []):
                    claim_source_ids.update(attack_flag.get("source_ids", []))
            material_meta = _material_usage(["attack_patterns", "verification_gold"], sorted(claim_source_ids))
            envelope = ResultEnvelope.coerce(
                incoming if incoming is not None else source_value,
                kind="claim_set",
                confidence=0.7 if claims else 0.3,
                confidence_source="claim_extraction",
            )
            envelope.kind = "claim_set"
            envelope.confidence = 0.7 if claims else 0.3
            envelope.confidence_source = "claim_extraction"
            envelope.with_claims(claims)
            envelope.metadata.update(
                {
                    "namespace": namespace,
                    "tool_name": name,
                    "max_claims": max_claims,
                    **material_meta,
                }
            )
            envelope.add_transform("chimera_claims", claim_count=len(claims))
            _record_trace(namespace, envelope)
            return _ok({
                "claims": claims,
                "claim_count": len(claims),
                "namespace": namespace,
                **material_meta,
                "pack_version": _get_materials().pack_version,
                "envelope": envelope.to_dict(),
            })

        elif name == "chimera_verify":
            namespace = _state_namespace(arguments)
            incoming = None
            if arguments.get("envelope") is not None:
                incoming = _normalize_envelope_input(
                    arguments["envelope"],
                    namespace=namespace,
                    tool_name=name,
                    confidence=0.7,
                    confidence_source="verification_input",
                )

            claims = list(arguments.get("claims") or [])
            if not claims and incoming is not None and incoming.claims:
                claims = incoming.claims
            if not claims:
                source_text = arguments.get("text")
                if source_text is None and incoming is not None:
                    source_text = (
                        incoming.value
                        if isinstance(incoming.value, str)
                        else json.dumps(incoming.value, ensure_ascii=True, sort_keys=True)
                    )
                if source_text:
                    claims = _extract_claims(str(source_text), max_claims=10)
            if not claims:
                return _err("chimera_verify requires claims, text, or an envelope with claims")

            evidence = list(arguments.get("evidence") or [])
            verification = _verify_claims_against_evidence(claims, evidence)
            envelope = ResultEnvelope.coerce(
                incoming if incoming is not None else {"claims": claims},
                kind="verification_result",
                confidence=verification["verification_score"],
                confidence_source="evidence_check",
            )
            envelope.kind = "verification_result"
            envelope.value = {
                "claims": claims,
                "verified_count": len(verification["verified_claims"]),
                "unsupported_count": len(verification["unsupported_claims"]),
                "contradicted_count": len(verification["contradicted_claims"]),
            }
            envelope.confidence = verification["verification_score"]
            envelope.confidence_source = "evidence_check"
            envelope.with_claims(claims)
            envelope.metadata.update(
                {
                    "namespace": namespace,
                    "tool_name": name,
                    "evidence_count": len(evidence),
                    "materials_used": verification["materials_used"],
                    "pack_versions": verification["pack_versions"],
                    "source_ids": verification["source_ids"],
                    "security_category_counts": verification["security_category_counts"],
                }
            )
            envelope.sources = [
                {
                    "index": idx,
                    "preview": _evidence_text(item)[:180],
                    "tainted": bool(_get_materials().find_attack_matches(_evidence_text(item))),
                }
                for idx, item in enumerate(evidence[:10])
            ]
            envelope.add_constraint(
                "evidence_verification",
                verification["supported"],
                verified_count=len(verification["verified_claims"]),
                unsupported_count=len(verification["unsupported_claims"]),
                contradicted_count=len(verification["contradicted_claims"]),
            )
            envelope.add_transform("chimera_verify", verification_score=verification["verification_score"])
            for claim in verification["unsupported_claims"][:3]:
                warning = f"Unsupported claim: {claim['text']}"
                if warning not in envelope.warnings:
                    envelope.warnings.append(warning)
            for claim in verification["contradicted_claims"][:3]:
                warning = f"Contradicted claim: {claim['text']}"
                if warning not in envelope.warnings:
                    envelope.warnings.append(warning)
            _record_trace(namespace, envelope)
            _record_audit(namespace, {
                "tool_name": name,
                "passed": verification["supported"],
                "confidence": verification["verification_score"],
                "claims": len(claims),
                "unsupported": len(verification["unsupported_claims"]),
                "contradicted": len(verification["contradicted_claims"]),
                "materials_used": verification["materials_used"],
                "pack_versions": verification["pack_versions"],
                "source_ids": verification["source_ids"],
                "security_category_counts": verification["security_category_counts"],
            })
            return _ok({
                "claims": claims,
                **verification,
                "namespace": namespace,
                "envelope": envelope.to_dict(),
            })

        elif name == "chimera_provenance_merge":
            namespace = _state_namespace(arguments)
            raw_envelopes = list(arguments.get("envelopes") or [])
            if not raw_envelopes:
                return _err("chimera_provenance_merge requires 'envelopes'")
            envelopes = [
                _normalize_envelope_input(
                    raw,
                    namespace=namespace,
                    tool_name=name,
                    confidence=0.5,
                    confidence_source="merge_input",
                )
                for raw in raw_envelopes
            ]
            merged = merge_envelopes(
                envelopes,
                strategy=arguments.get("strategy", "weighted"),
                merge_value_mode=arguments.get("merge_value_mode", "list"),
            )
            merged.kind = "merged_provenance"
            merged.metadata.update({"namespace": namespace, "tool_name": name})
            _record_trace(namespace, merged)
            return _ok({
                "merged_from": len(envelopes),
                "confidence": merged.confidence,
                "namespace": namespace,
                "envelope": merged.to_dict(),
            })

        elif name == "chimera_policy":
            namespace = _state_namespace(arguments)
            action = arguments.get("action", "list")
            if action == "list":
                return _ok({
                    "policies": {
                        policy_name: {
                            **{k: v for k, v in _policy_details(policy_name, config).items() if k != "description"},
                            "description": _policy_details(policy_name, config).get("description", ""),
                        }
                        for policy_name, config in _POLICIES.items()
                    },
                    "count": len(_POLICIES),
                    "pack_version": _get_materials().pack_version,
                })

            policy_name = arguments.get("policy", "")
            config = _POLICIES.get(policy_name)
            if not config:
                return _err(f"Unknown policy: {policy_name}")
            if action == "get":
                return _ok({"policy": policy_name, "config": _policy_details(policy_name, config), "pack_version": _get_materials().pack_version})
            if action != "apply":
                return _err("chimera_policy action must be list|get|apply")
            if arguments.get("envelope") is None and "value" not in arguments:
                return _err("chimera_policy apply requires 'value' or 'envelope'")

            incoming = _normalize_envelope_input(
                arguments.get("envelope", arguments.get("value")),
                namespace=namespace,
                tool_name=str(arguments.get("tool_name", "chimera_policy")),
                confidence=0.8,
                confidence_source="policy_input",
            )
            pattern = _get_materials().policy_pattern(policy_name)
            spec = ToolCallSpec(
                tool_name=str(arguments.get("tool_name", "chimera_policy")),
                min_confidence=float(config.get("min_confidence", 0.0)),
                output_forbidden=list(config.get("output_forbidden", [])),
                detect_strategy=str(config.get("detect_strategy", "confidence_threshold")),
                detect_threshold=float(config.get("detect_threshold", 0.5)),
                strict=bool(config.get("strict", False)),
            )
            constraint_result = _middleware.call(
                spec,
                raw_output=incoming.value,
                input_confidence=max(incoming.confidence, float(config.get("min_confidence", 0.0))),
            )
            envelope = ResultEnvelope.coerce(
                incoming,
                kind="policy_result",
                confidence=constraint_result.confidence,
                confidence_source=f"policy:{policy_name}",
            )
            envelope.kind = "policy_result"
            envelope.value = constraint_result.value
            envelope.confidence = round(constraint_result.confidence, 4)
            envelope.confidence_source = f"policy:{policy_name}"
            value_text = (
                constraint_result.value
                if isinstance(constraint_result.value, str)
                else json.dumps(constraint_result.value, ensure_ascii=True, sort_keys=True)
            )
            security_flags = _get_materials().find_attack_matches(value_text)
            security_categories = set(config.get("security_categories", []))
            relevant_security_flags = [
                flag for flag in security_flags if not security_categories or flag.get("category") in security_categories
            ]
            material_meta = _material_usage(
                ["policy_patterns", "attack_patterns"],
                list(pattern.get("source_ids", [])) if pattern else None,
            )
            envelope.metadata.update(
                {
                    "namespace": namespace,
                    "tool_name": name,
                    "policy": policy_name,
                    "owasp_refs": list(pattern.get("owasp_refs", [])) if pattern else [],
                    **material_meta,
                    "security_category_counts": _get_materials().security_category_counts(relevant_security_flags),
                }
            )
            sources_present = bool(envelope.sources)
            if not sources_present and isinstance(incoming.value, dict):
                sources_present = bool(incoming.value.get("sources"))
            passed = constraint_result.passed
            warnings = list(constraint_result.warnings)
            if config.get("require_sources") and not sources_present:
                warnings.append("Policy expects explicit sources but none were provided.")
                passed = False
            if config.get("warning"):
                warnings.append(str(config["warning"]))
            if relevant_security_flags:
                warnings.append(
                    "Security-sensitive content matched policy categories: "
                    + ", ".join(sorted({str(flag.get("category")) for flag in relevant_security_flags}))
                )
                passed = False
            for warning in warnings:
                if warning not in envelope.warnings:
                    envelope.warnings.append(warning)
            envelope.add_constraint(
                f"policy:{policy_name}",
                passed,
                violations=constraint_result.violations,
                warnings=warnings,
            )
            envelope.add_transform("chimera_policy", policy=policy_name, passed=passed)
            _record_trace(namespace, envelope)
            _record_audit(namespace, {
                "tool_name": name,
                "policy": policy_name,
                "passed": passed,
                "confidence": envelope.confidence,
                "warnings": warnings,
                "materials_used": material_meta["materials_used"],
                "pack_versions": material_meta["pack_versions"],
                "source_ids": material_meta["source_ids"],
                "security_category_counts": _get_materials().security_category_counts(relevant_security_flags),
            })
            return _ok({
                "policy": policy_name,
                "config": _policy_details(policy_name, config),
                "passed": passed,
                "violations": constraint_result.violations,
                "warnings": warnings,
                "value": constraint_result.value,
                "namespace": namespace,
                "owasp_refs": list(pattern.get("owasp_refs", [])) if pattern else [],
                "security_flags": relevant_security_flags,
                **material_meta,
                "envelope": envelope.to_dict(),
            })

        elif name == "chimera_trace":
            namespace = _state_namespace(arguments)
            action = arguments.get("action", "latest")
            limit = max(1, int(arguments.get("limit", 10)))
            traces = list(_store.load("traces", namespace, []))
            material_index: dict[tuple[str, str], dict[str, Any]] = {}
            pack_versions: dict[str, str] = {}
            source_ids: set[str] = set()
            security_category_counts: dict[str, int] = {}
            for trace in traces:
                metadata = dict(trace.get("metadata", {}))
                for used in list(metadata.get("materials_used", [])):
                    key = (str(used.get("pack_type", "")), str(used.get("pack_version", "")))
                    material_index[key] = used
                pack_versions.update(dict(metadata.get("pack_versions", {})))
                source_ids.update(metadata.get("source_ids", []))
                for category, count in dict(metadata.get("security_category_counts", {})).items():
                    security_category_counts[category] = security_category_counts.get(category, 0) + int(count)
            if action == "stats":
                kinds: dict[str, int] = {}
                for trace in traces:
                    kind = str(trace.get("kind", "generic"))
                    kinds[kind] = kinds.get(kind, 0) + 1
                return _ok({
                    "namespace": namespace,
                    "trace_count": len(traces),
                    "kinds": kinds,
                    "storage_path": _store.path_for("traces", namespace),
                    "materials_used": list(material_index.values()),
                    "pack_versions": pack_versions,
                    "source_ids": sorted(source_ids),
                    "security_category_counts": security_category_counts,
                })
            if action == "get":
                envelope_id = str(arguments.get("envelope_id", "")).strip()
                if not envelope_id:
                    return _err("chimera_trace get requires 'envelope_id'")
                for trace in reversed(traces):
                    if str(trace.get("envelope_id")) == envelope_id:
                        return _ok({"namespace": namespace, "trace": trace})
                return _err(f"Trace not found: {envelope_id}")
            if action == "list":
                return _ok({
                    "namespace": namespace,
                    "traces": traces[-limit:],
                    "trace_count": len(traces),
                    "materials_used": list(material_index.values()),
                    "pack_versions": pack_versions,
                    "source_ids": sorted(source_ids),
                })
            latest = traces[-limit:] if traces else []
            return _ok({
                "namespace": namespace,
                "latest_trace": latest[-1] if latest else None,
                "recent_traces": latest,
                "trace_count": len(traces),
                "materials_used": list(material_index.values()),
                "pack_versions": pack_versions,
                "source_ids": sorted(source_ids),
            })

        elif name == "chimera_materials":
            action = str(arguments.get("action", "status"))
            registry = _get_materials()
            if action == "list_packs":
                return _ok({"packs": registry.list_packs(), "pack_version": registry.pack_version})
            if action == "licenses":
                return _ok(registry.licenses())
            if action == "source_manifest":
                return _ok(registry.source_manifest())
            if action != "status":
                return _err("chimera_materials action must be list_packs|status|licenses|source_manifest")
            return _ok(registry.status())

        # ── chimera_fracture — full pipeline ──────────────────────────────
        elif name == "chimera_fracture":
            import re as _re

            messages = arguments.get("messages", [])
            documents = arguments.get("documents", [])
            token_budget = int(arguments.get("token_budget", 1500))
            allow_lossy = bool(arguments.get("allow_lossy", False))
            algorithm = str(arguments.get("algorithm", "quantum")).lower()
            focus = _resolve_focus(arguments, messages=messages)

            if algorithm == "classic":
                def _compress_message_history(msgs: list[dict[str, Any]]) -> str:
                    if not msgs:
                        return ""
                    msg_text = "\n".join(
                        f"[{m.get('role', 'user')}]: {m.get('content', '')}"
                        for m in msgs
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
                    return compressed

                total_start = time.time()
                stats: dict[str, Any] = {
                    "documents_input": sum(len(d) for d in documents),
                    "messages_input": len(messages),
                    "tokens_input": _tbm.count_messages(messages),
                }

                optimised_docs: list[str] = []
                for doc in documents:
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
                messages_compressed = _compress_message_history(messages)

                tokens_after = _tbm.count_tokens(messages_compressed) + _tbm.count_texts(optimised_docs)
                budget_remaining = max(0, token_budget - tokens_after)
                quality_passed = tokens_after <= token_budget
                lossy_dropped_count = 0
                dropped_indexes: list[int] = []

                if not quality_passed and allow_lossy:
                    original_messages = list(messages)
                    ranked = _scorer.rank(original_messages, focus=focus)
                    min_keep = 2
                    dropped_scores: list[float] = []
                    kept = original_messages
                    for entry in ranked:
                        if quality_passed or len(original_messages) - len(dropped_indexes) <= min_keep:
                            break
                        dropped_indexes.append(entry["index"])
                        dropped_scores.append(entry["score"])
                        kept = [m for i, m in enumerate(original_messages) if i not in set(dropped_indexes)]
                        if not kept:
                            break
                        tombstone = {
                            "role": "system",
                            "content": (
                                f"[{len(dropped_indexes)} messages omitted - "
                                f"low importance scores: {', '.join(str(s) for s in dropped_scores)}]"
                            ),
                        }
                        messages = kept + [tombstone]
                        lossy_dropped_count = len(dropped_indexes)
                        messages_compressed = _compress_message_history(messages)
                        tokens_after = _tbm.count_tokens(messages_compressed) + _tbm.count_texts(optimised_docs)
                        budget_remaining = max(0, token_budget - tokens_after)
                        quality_passed = tokens_after <= token_budget
                        if quality_passed:
                            break

                stats["duration_ms"] = round((time.time() - total_start) * 1000, 1)

                return _ok({
                    "quality_passed": quality_passed,
                    "budget_remaining": budget_remaining,
                    "tokens_input": stats["tokens_input"],
                    "tokens_after_pipeline": tokens_after,
                    "documents_input": stats["documents_input"],
                    "documents_optimised": stats["documents_optimised"],
                    "messages_input": stats["messages_input"],
                    "lossy_dropped_count": lossy_dropped_count,
                    "compression_time_ms": stats["duration_ms"],
                    "token_count_method": _tbm.get_stats()["token_count_method"],
                    "algorithm": "classic",
                    "focus_terms": extract_focus_terms(focus),
                    "optimized_documents": optimised_docs,
                    "compressed_messages": messages,
                    "dropped_message_indexes": dropped_indexes,
                    "compressed_message_history": messages_compressed,
                })

            total_start = time.time()
            optimized_documents: list[str] = []
            document_results = []
            for document in documents:
                result = _quantum.optimize_text(
                    document,
                    focus=focus,
                    preserve_code=True,
                    strategies=["whitespace", "dedup_sentences", "strip_filler", "collapse_lists"],
                    level="aggressive",
                )
                optimized_documents.append(result.text)
                document_results.append(result)

            documents_input = sum(len(document) for document in documents)
            documents_optimised = sum(len(document) for document in optimized_documents)
            document_tokens = _tbm.count_texts(optimized_documents)
            message_budget = max(64, token_budget - document_tokens)
            message_result = _quantum.compress_messages(
                messages,
                focus=focus,
                scorer=_scorer,
                token_budget=message_budget,
                allow_lossy=allow_lossy,
            )
            tokens_input = _tbm.count_messages(messages)
            tokens_after = _tbm.count_messages(message_result.messages) + document_tokens
            budget_remaining = max(0, token_budget - tokens_after)
            quality_passed = tokens_after <= token_budget
            compression_time_ms = round((time.time() - total_start) * 1000, 1)

            return _ok({
                "quality_passed": quality_passed,
                "budget_remaining": budget_remaining,
                "tokens_input": tokens_input,
                "tokens_after_pipeline": tokens_after,
                "tokens_saved": max(0, tokens_input + _tbm.count_texts(documents) - tokens_after),
                "documents_input": documents_input,
                "documents_optimised": documents_optimised,
                "messages_input": len(messages),
                "lossy_dropped_count": len(message_result.dropped_indexes),
                "compression_time_ms": compression_time_ms,
                "token_count_method": _tbm.get_stats()["token_count_method"],
                "algorithm": "quantum",
                "focus_terms": message_result.focus_terms,
                "optimized_documents": optimized_documents,
                "compressed_messages": message_result.messages,
                "compressed_message_history": message_result.compressed_history,
                "compressed_message_indexes": message_result.compressed_indexes,
                "dropped_message_indexes": message_result.dropped_indexes,
                "units_kept": {
                    "documents": sum(result.units_kept for result in document_results),
                    "document_units_total": sum(result.units_total for result in document_results),
                },
            })

        # ── chimera_optimize ──────────────────────────────────────────────
        elif name == "chimera_optimize":
            import re as _re

            text = arguments["text"]
            strategies = arguments.get("strategies") or ["whitespace", "dedup_sentences", "strip_filler"]
            preserve_code = bool(arguments.get("preserve_code", True))
            algorithm = str(arguments.get("algorithm", "quantum")).lower()
            focus = _resolve_focus(arguments, prompt=text)

            if algorithm == "quantum":
                result = _quantum.optimize_text(
                    text,
                    focus=focus,
                    preserve_code=preserve_code,
                    strategies=strategies,
                    level="aggressive",
                )
                saved = result.original_chars - result.compressed_chars
                ratio = round(saved / result.original_chars, 4) if result.original_chars else 0.0
                return _ok({
                    "optimised_text": result.text,
                    "original_chars": result.original_chars,
                    "optimised_chars": result.compressed_chars,
                    "chars_saved": saved,
                    "reduction_ratio": ratio,
                    "estimated_tokens_before": result.original_tokens,
                    "estimated_tokens_after": result.compressed_tokens,
                    "estimated_tokens_saved": max(0, result.original_tokens - result.compressed_tokens),
                    "passes_applied": result.passes_applied,
                    "code_blocks_preserved": result.code_blocks_preserved,
                    "algorithm": "quantum",
                    "focus_terms": result.focus_terms,
                    "units_total": result.units_total,
                    "units_kept": result.units_kept,
                })

            original_len = len(text)
            result_text = text
            log: list[str] = []
            code_blocks: list[str] = []

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
                for pat in [
                    r"\bplease note that\b", r"\bit is worth noting that\b",
                    r"\bit should be noted that\b", r"\bin order to\b",
                    r"\bbasically\b", r"\bactually\b", r"\bvery\b",
                    r"\bjust\b", r"\bsimply\b", r"\bquite\b",
                    r"\bof course\b", r"\bneedless to say\b",
                    r"\bas you can see\b", r"\bclearly\b",
                ]:
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
                "optimised_text": result_text,
                "original_chars": original_len,
                "optimised_chars": len(result_text),
                "chars_saved": saved,
                "reduction_ratio": ratio,
                "estimated_tokens_saved": max(0, round(saved / 4)),
                "passes_applied": log,
                "code_blocks_preserved": len(code_blocks),
                "algorithm": "classic",
                "focus_terms": extract_focus_terms(focus),
            })

        # ── chimera_compress ──────────────────────────────────────────────
        elif name == "chimera_compress":
            import re as _re

            text          = arguments["text"]
            level         = arguments.get("level", "medium")
            preserve_code = bool(arguments.get("preserve_code", True))
            algorithm     = str(arguments.get("algorithm", "quantum")).lower()
            focus         = _resolve_focus(arguments, prompt=text)
            auto_track    = bool(arguments.get("auto_track", True))
            track_model   = arguments.get("model", _DEFAULT_MODEL)
            namespace     = _state_namespace(arguments)

            if algorithm == "quantum":
                strategies = ["whitespace", "strip_filler"]
                if level in ("medium", "aggressive"):
                    strategies.append("dedup_sentences")
                result = _quantum.optimize_text(
                    text,
                    focus=focus,
                    preserve_code=preserve_code,
                    strategies=strategies,
                    level=level,
                )
                saved = result.original_chars - result.compressed_chars
                ratio = round(saved / result.original_chars, 4) if result.original_chars else 0.0
                track_entry: dict[str, Any] | None = None
                if auto_track and result.original_tokens > result.compressed_tokens:
                    track_entry = _get_cost_tracker(namespace).record(
                        tokens_before=result.original_tokens,
                        tokens_after=result.compressed_tokens,
                        model=track_model,
                        label="chimera_compress/quantum",
                    )
                payload: dict[str, Any] = {
                    "compressed_text": result.text,
                    "level": level,
                    "original_chars": result.original_chars,
                    "compressed_chars": result.compressed_chars,
                    "chars_saved": saved,
                    "compression_ratio": ratio,
                    "estimated_tokens_before": result.original_tokens,
                    "estimated_tokens_after": result.compressed_tokens,
                    "estimated_tokens_saved": max(0, result.original_tokens - result.compressed_tokens),
                    "code_blocks_preserved": result.code_blocks_preserved,
                    "algorithm": "quantum",
                    "focus_terms": result.focus_terms,
                    "units_total": result.units_total,
                    "units_kept": result.units_kept,
                }
                if track_entry:
                    payload["tracked"] = {"request_id": track_entry["request_id"],
                                          "savings_usd": track_entry["savings"]}
                return _ok(payload)

            original_len = len(text)

            code_blocks: list[str] = []
            work = text
            if preserve_code:
                def _stash(m: "_re.Match[str]") -> str:
                    code_blocks.append(m.group(0))
                    return f"\x00CODE{len(code_blocks) - 1}\x00"
                work = _re.sub(r"```[\s\S]*?```", _stash, work)

            work = _re.sub(r"[ \t]+", " ", work)
            work = _re.sub(r"\n{3,}", "\n\n", work).strip()

            if level in ("medium", "aggressive"):
                for pat, repl in {
                    r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't",
                    r"\bdid not\b": "didn't", r"\bcannot\b": "can't",
                    r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't",
                    r"\bshould not\b": "shouldn't", r"\bcould not\b": "couldn't",
                    r"\bare not\b": "aren't", r"\bwas not\b": "wasn't",
                    r"\bwere not\b": "weren't", r"\bhave not\b": "haven't",
                    r"\bhas not\b": "hasn't", r"\bhad not\b": "hadn't",
                    r"\bI am\b": "I'm", r"\bI have\b": "I've",
                    r"\bI will\b": "I'll", r"\bI would\b": "I'd",
                    r"\bit is\b": "it's", r"\bthat is\b": "that's",
                    r"\bthere is\b": "there's", r"\bthey are\b": "they're",
                    r"\bwe are\b": "we're", r"\byou are\b": "you're",
                }.items():
                    work = _re.sub(pat, repl, work, flags=_re.IGNORECASE)

            if level == "aggressive":
                for pat, repl in {
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
                }.items():
                    work = _re.sub(pat, repl, work, flags=_re.IGNORECASE)
                work = _re.sub(r"\.{2,}", "…", work)
                work = _re.sub(r"\s+([,;:!?])", r"\1", work)

            if preserve_code:
                for i, block in enumerate(code_blocks):
                    work = work.replace(f"\x00CODE{i}\x00", block)

            compressed_len   = len(work)
            saved            = original_len - compressed_len
            ratio            = round(saved / original_len, 4) if original_len else 0.0
            tokens_before_c  = max(1, round(original_len / 4))
            tokens_after_c   = max(0, round(compressed_len / 4))
            classic_track: dict[str, Any] | None = None
            if auto_track and tokens_before_c > tokens_after_c:
                classic_track = _get_cost_tracker(namespace).record(
                    tokens_before=tokens_before_c,
                    tokens_after=tokens_after_c,
                    model=track_model,
                    label="chimera_compress/classic",
                )
            classic_payload: dict[str, Any] = {
                "compressed_text": work,
                "level": level,
                "original_chars": original_len,
                "compressed_chars": compressed_len,
                "chars_saved": saved,
                "compression_ratio": ratio,
                "estimated_tokens_saved": max(0, round(saved / 4)),
                "code_blocks_preserved": len(code_blocks),
                "algorithm": "classic",
                "focus_terms": extract_focus_terms(focus),
            }
            if classic_track:
                classic_payload["tracked"] = {"request_id": classic_track["request_id"],
                                               "savings_usd": classic_track["savings"]}
            return _ok(classic_payload)

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
            focus    = _resolve_focus(arguments, messages=messages)
            mode     = arguments.get("mode", "drop_priority")
            if not messages:
                return _ok([])
            if mode == "importance_for_goal":
                focus_terms_set = set(extract_focus_terms(focus))
                scored: list[dict[str, Any]] = []
                for idx, msg in enumerate(messages):
                    content = normalize_content(msg.get("content", ""))
                    content_terms = set(t for t in re.findall(r"[A-Za-z0-9_./:-]+", content.lower()) if len(t) >= 3)
                    # Importance-for-goal: weight focus alignment 60%, structural signal 20%,
                    # content density 10%, user-question bonus 10%.
                    focus_score   = (
                        len(content_terms & focus_terms_set) / max(len(focus_terms_set), 1)
                        if focus_terms_set else 0.0
                    )
                    density_score = min(len(content_terms) / 80.0, 1.0)
                    structure_score = 0.2 if "```" in content else 0.0
                    question_bonus  = 0.1 if ("?" in content and msg.get("role") == "user") else 0.0
                    importance = min(1.0, focus_score * 0.60 + density_score * 0.10
                                     + structure_score * 0.20 + question_bonus)
                    reason = (
                        "focus_match"      if focus_score > 0.1
                        else "code_fence"  if structure_score > 0
                        else "user_question" if question_bonus > 0
                        else "low_focus_alignment"
                    )
                    scored.append({
                        "index":       idx,
                        "role":        msg.get("role", "unknown"),
                        "score":       round(importance, 4),
                        "focus_score": round(focus_score, 4),
                        "reason":      reason,
                    })
                scored.sort(key=lambda item: item["score"], reverse=True)
                return _ok({
                    "scores":             scored,
                    "total_messages":     len(messages),
                    "mode":               "importance_for_goal",
                    "focus_terms":        list(focus_terms_set),
                    "token_count_method": _tbm.get_stats()["token_count_method"],
                    "note": "Sorted descending — highest score = most relevant to focus.",
                })
            # Default: drop_priority (lowest score = evict first)
            ranked = _scorer.rank(messages, focus=focus)
            return _ok({
                "scores":             ranked,
                "total_messages":     len(messages),
                "mode":               "drop_priority",
                "token_count_method": _tbm.get_stats()["token_count_method"],
                "algorithm":          "quantum",
                "focus_terms":        extract_focus_terms(focus),
                "note": "Sorted ascending — lowest score = evict first during compression.",
            })

        # ── AGI: chimera_causal ────────────────────────────────────────────
        elif name == "chimera_causal":
            action = arguments.get("action", "info")
            cr     = _get_causal()
            if action == "add_edge":
                cr.add_edge(
                    cause=arguments["cause"],
                    effect=arguments["effect"],
                    edge_type=arguments.get("edge_type", "causes"),
                    strength=float(arguments.get("strength", 0.5)),
                    confidence=float(arguments.get("confidence", 0.5)),
                    confidence_level=arguments.get("confidence_level", "observed"),
                )
                return _ok({"added": True,
                            "edge": f"{arguments['cause']} --{arguments.get('edge_type','causes')}--> {arguments['effect']}"})
            elif action == "query":
                return _ok({"query_result": cr.query(
                    cause=arguments.get("cause"), effect=arguments.get("effect"))})
            elif action == "paths":
                return _ok({"paths": cr.find_causal_paths(
                    arguments.get("source", ""), arguments.get("target", ""))})
            else:
                return _ok({"variables": sorted(cr.graph.variables),
                            "edge_count": cr.graph.edge_count})

        # ── AGI: chimera_deliberate ────────────────────────────────────────
        elif name == "chimera_deliberate":
            prompt       = arguments.get("prompt", "")
            perspectives = arguments.get("perspectives", [])
            mode         = arguments.get("mode", "semantic")
            return _ok(_get_deliberation().deliberate(prompt, perspectives, mode=mode))

        # ── AGI: chimera_metacognize ───────────────────────────────────────
        elif name == "chimera_metacognize":
            preds = arguments.get("predictions", [])
            label = arguments.get("label", "")
            if not preds:
                return _ok({"ece": None, "note": "Supply predictions=[{predicted_confidence, was_correct}]"})
            n = len(preds)
            n_bins = min(10, n)
            bin_size = 1.0 / n_bins
            ece = 0.0
            overconf = underconf = 0
            for p in preds:
                conf    = float(p.get("predicted_confidence", 0.5))
                correct = bool(p.get("was_correct", False))
                if conf > 0.5 and not correct:
                    overconf += 1
                elif conf < 0.5 and correct:
                    underconf += 1
            bins: dict = {}
            for p in preds:
                conf = float(p.get("predicted_confidence", 0.5))
                b    = min(int(conf / bin_size), n_bins - 1)
                bins.setdefault(b, []).append(p)
            for b_preds in bins.values():
                acc  = sum(1 for p in b_preds if p.get("was_correct")) / len(b_preds)
                conf = sum(float(p.get("predicted_confidence", 0.5)) for p in b_preds) / len(b_preds)
                ece += abs(acc - conf) * len(b_preds) / n
            return _ok({
                "label":               label,
                "n":                   n,
                "ece":                 round(ece, 4),
                "overconfidence_rate": round(overconf / n, 3),
                "underconfidence_rate": round(underconf / n, 3),
                "calibration_status":  "good" if ece < 0.1 else "moderate" if ece < 0.2 else "poor",
            })

        # ── AGI: chimera_meta_learn ─────────────────────────────────────────
        elif name == "chimera_meta_learn":
            action = arguments.get("action", "stats")
            namespace = _state_namespace(arguments)
            ml     = _get_meta_learner(namespace)
            if action == "record":
                result = ml.record_adaptation(
                    context=arguments.get("context", ""),
                    action=arguments.get("action_taken", ""),
                    outcome=arguments.get("outcome", ""),
                    confidence=float(arguments.get("confidence", 0.5)),
                )
                storage_path = _save_meta_learner(namespace)
                return _ok({**result, "namespace": namespace, "storage_path": storage_path})
            return _ok({
                **ml.get_stats(),
                "namespace": namespace,
                "storage_path": _store.path_for("meta_learner", namespace),
            })

        # ── AGI: chimera_quantum_vote ──────────────────────────────────────
        elif name == "chimera_quantum_vote":
            responses = arguments.get("responses", [])
            if not responses:
                return _err("chimera_quantum_vote requires responses list")
            return _ok(_quantum_vote(responses,
                                     timeout_s=float(arguments.get("timeout_s", 5.0))))

        # ── AGI: chimera_plan_goals ─────────────────────────────────────────
        elif name == "chimera_plan_goals":
            goal = arguments.get("goal", "")
            if not goal:
                return _err("chimera_plan_goals requires a goal string")
            return _ok(_plan_goals(goal))

        # ── AGI: chimera_world_model ─────────────────────────────────────────
        elif name == "chimera_world_model":
            action = arguments.get("action", "query")
            namespace = _state_namespace(arguments)
            wm     = _get_world_model(namespace)
            if action == "update":
                result = wm.update(
                    key=arguments.get("key", ""),
                    value=arguments.get("value"),
                    confidence=float(arguments.get("confidence", 0.8)),
                )
                storage_path = _save_world_model(namespace)
                return _ok({**result, "namespace": namespace, "storage_path": storage_path})
            return _ok({
                **wm.query(key=arguments.get("key")),
                "namespace": namespace,
                "storage_path": _store.path_for("world_model", namespace),
            })

        # ── AGI: chimera_safety_check ────────────────────────────────────────
        elif name == "chimera_safety_check":
            namespace = _state_namespace(arguments)
            content = arguments.get("content", "")
            sl = _get_safety()
            registry = _get_materials()
            is_safe, reason = sl.validate_content(content)
            attack_flags = registry.find_attack_matches(content)
            category_counts = registry.security_category_counts(attack_flags)
            if any(flag["severity"] >= 0.84 for flag in attack_flags):
                is_safe = False
                if reason == "ok":
                    reason = "mcp_security_risk_detected"
            material_meta = _material_usage(
                ["attack_patterns", "policy_patterns"],
                [sid for flag in attack_flags for sid in flag.get("source_ids", [])],
            )
            _record_audit(
                namespace,
                {
                    "tool_name": name,
                    "passed": is_safe,
                    "confidence": 1.0 if is_safe else 0.4,
                    "materials_used": material_meta["materials_used"],
                    "pack_versions": material_meta["pack_versions"],
                    "source_ids": material_meta["source_ids"],
                    "security_category_counts": category_counts,
                },
            )
            return _ok(
                {
                    "is_safe": is_safe,
                    "reason": reason,
                    "blocked_count": sl._blocked_count,
                    "allowed_count": sl._allowed_count,
                    "attack_flags": attack_flags,
                    "security_category_counts": category_counts,
                    "namespace": namespace,
                    "pack_version": registry.pack_version,
                    **material_meta,
                }
            )

        # ── AGI: chimera_ethical_eval ────────────────────────────────────────
        elif name == "chimera_ethical_eval":
            action_desc = arguments.get("action", "")
            if not action_desc:
                return _err("chimera_ethical_eval requires action description")
            return _ok(_get_ethical().evaluate_action(action_desc))

        # ── AGI: chimera_embodied ───────────────────────────────────────────
        elif name == "chimera_embodied":
            action = arguments.get("action", "status")
            eb = _get_embodied()
            if action == "perceive":
                return _ok(eb.perceive(
                    objects=arguments.get("objects", []),
                    environment=arguments.get("environment", ""),
                ))
            elif action == "act":
                return _ok(eb.act(
                    action_name=arguments.get("action_name", "noop"),
                    params=arguments.get("params", {}),
                ))
            elif action == "reset":
                return _ok(eb.reset())
            return _ok(eb.status())

        # ── AGI: chimera_social ────────────────────────────────────────────
        elif name == "chimera_social":
            action = arguments.get("action", "list_agents")
            sc = _get_social()
            if action == "record_interaction":
                agent = arguments.get("agent", "")
                if not agent:
                    return _err("chimera_social record_interaction requires 'agent'")
                return _ok(sc.record_interaction(
                    agent=agent,
                    topic=arguments.get("topic", ""),
                    sentiment=float(arguments.get("sentiment", 0.0)),
                ))
            elif action == "query":
                return _ok(sc.query(agent=arguments.get("agent", "")))
            return _ok(sc.list_agents())

        # ── AGI: chimera_transfer_learn ──────────────────────────────────────
        elif name == "chimera_transfer_learn":
            action = arguments.get("action", "list")
            tl = _get_transfer()
            if action == "add_mapping":
                return _ok(tl.add_mapping(
                    source=arguments.get("source_domain", ""),
                    target=arguments.get("target_domain", ""),
                    concept=arguments.get("concept", ""),
                    analogy=arguments.get("analogy", ""),
                    confidence=float(arguments.get("confidence", 0.7)),
                ))
            elif action == "query":
                return _ok(tl.query(
                    source=arguments.get("source_domain", ""),
                    target=arguments.get("target_domain", ""),
                ))
            return _ok(tl.list_all())

        # ── AGI: chimera_evolve ──────────────────────────────────────────────
        elif name == "chimera_evolve":
            action = arguments.get("action", "info")
            ev = _get_evolve()
            if action == "run":
                candidates = arguments.get("candidates")
                if not candidates:
                    return _err("chimera_evolve run requires 'candidates' list [{id, value, fitness_score}]")
                return _ok(ev.run(
                    candidates=candidates,
                    generations=int(arguments.get("generations", 5)),
                    mutation_rate=float(arguments.get("mutation_rate", 0.1)),
                    survival_ratio=float(arguments.get("survival_ratio", 0.5)),
                ))
            return _ok(ev.info())

        # ── AGI: chimera_self_model ──────────────────────────────────────────
        elif name == "chimera_self_model":
            action = arguments.get("action", "reflect")
            namespace = _state_namespace(arguments)
            sm     = _get_self_model(namespace)
            if action == "update":
                result = sm.update(
                    capability=arguments.get("capability", ""),
                    level=arguments.get("level", "present"),
                    evidence=arguments.get("evidence", ""),
                )
                storage_path = _save_self_model(namespace)
                return _ok({**result, "namespace": namespace, "storage_path": storage_path})
            return _ok({
                **sm.reflect(),
                "namespace": namespace,
                "storage_path": _store.path_for("self_model", namespace),
            })

        # ── AGI: chimera_knowledge ──────────────────────────────────────────
        elif name == "chimera_knowledge":
            action = arguments.get("action", "search")
            namespace = _state_namespace(arguments)
            kb     = _get_kb(namespace)
            if action == "add":
                entry = kb.add(content=arguments.get("content", ""),
                               category=arguments.get("category", "general"),
                               tags=arguments.get("tags", []))
                storage_path = _save_kb(namespace)
                return _ok({"added": True, "entry_id": entry.entry_id, "namespace": namespace, "storage_path": storage_path})
            elif action == "search":
                return _ok({
                    "results": kb.search(query=arguments.get("query", "")),
                    "namespace": namespace,
                    "storage_path": _store.path_for("knowledge", namespace),
                })
            elif action == "list":
                return _ok({"entries": len(kb._entries),
                            "categories": list({e.category for e in kb._entries.values()}),
                            "namespace": namespace,
                            "storage_path": _store.path_for("knowledge", namespace)})
            return _ok({
                "entry_count": len(kb._entries),
                "namespace": namespace,
                "storage_path": _store.path_for("knowledge", namespace),
            })

        # ── AGI: chimera_memory ────────────────────────────────────────────
        elif name == "chimera_memory":
            action = arguments.get("action", "recall")
            namespace = _state_namespace(arguments)
            mem    = _get_memory(namespace)
            if action == "store":
                result = mem.store(content=arguments.get("content", ""),
                                   tags=arguments.get("tags", []),
                                   importance=float(arguments.get("importance", 0.5)))
                storage_path = _save_memory(namespace)
                return _ok({**result, "namespace": namespace, "storage_path": storage_path})
            return _ok({
                **mem.recall(query=arguments.get("query"),
                             limit=int(arguments.get("limit", 10))),
                "namespace": namespace,
                "storage_path": _store.path_for("memory", namespace),
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
            namespace = _state_namespace(arguments)
            tracker = _get_cost_tracker(namespace)
            entry = tracker.record(
                tokens_before = int(arguments["tokens_before"]),
                tokens_after  = int(arguments["tokens_after"]),
                model         = arguments.get("model", _DEFAULT_MODEL),
                label         = arguments.get("label", ""),
            )
            storage_path = _save_cost_tracker(namespace)
            log.info(
                "[CostTracker] %s → %s tokens ($%.4f → $%.4f) saved %.1f%%",
                entry["tokens_before"], entry["tokens_after"],
                entry["cost_before"], entry["cost_after"], entry["pct_saved"],
            )
            return _ok({**entry, "namespace": namespace, "storage_path": storage_path})

        # ── chimera_dashboard ─────────────────────────────────────────────────
        elif name == "chimera_dashboard":
            namespace = _state_namespace(arguments)
            tracker = _get_cost_tracker(namespace)
            return _ok({
                **tracker.summary(),
                "namespace": namespace,
                "storage_path": _store.path_for("cost_tracker", namespace),
            })

        # ── chimera_mode — task-relevant tool guidance ─────────────────────
        elif name == "chimera_mode":
            mode = arguments.get("mode", "minimal")
            task = arguments.get("task_description", "").lower()
            all_tool_names = [tool.name for tool in await list_tools()]
            total_tool_count = len(all_tool_names)

            # Auto-detect mode from task description
            if task:
                _agi_kw  = ["reason", "analyze", "causal", "ethical", "plan", "deliberate", "safety", "world"]
                _tok_kw  = ["compress", "budget", "token", "optimize", "cost", "efficiency"]
                if any(k in task for k in _agi_kw):
                    mode = "agi"
                elif any(k in task for k in _tok_kw):
                    mode = "token"
                elif len(task) > 100:
                    mode = "full"

            _MODES: dict[str, dict[str, Any]] = {
                "minimal": {
                    "recommended_tools": ["chimera_csm", "chimera_budget_lock", "chimera_gate", "chimera_confident", "chimera_memory", "chimera_policy"],
                    "avoid_tools": ["chimera_causal", "chimera_deliberate", "chimera_metacognize", "chimera_quantum_vote",
                                    "chimera_embodied", "chimera_social", "chimera_transfer_learn", "chimera_evolve",
                                    "chimera_prove", "chimera_audit", "chimera_self_model", "chimera_trace"],
                    "description": "Core tools only. Best for simple Q&A and short tasks.",
                },
                "token": {
                    "recommended_tools": ["chimera_csm", "chimera_budget_lock", "chimera_optimize", "chimera_compress",
                                          "chimera_fracture", "chimera_budget", "chimera_score",
                                          "chimera_cost_estimate", "chimera_cost_track", "chimera_dashboard"],
                    "avoid_tools": ["chimera_causal", "chimera_deliberate", "chimera_metacognize",
                                    "chimera_embodied", "chimera_social", "chimera_transfer_learn", "chimera_evolve"],
                    "description": "Token efficiency focus. Best for long documents or cost-sensitive tasks.",
                },
                "agi": {
                    "recommended_tools": ["chimera_csm", "chimera_causal", "chimera_deliberate", "chimera_metacognize",
                                          "chimera_quantum_vote", "chimera_plan_goals", "chimera_world_model",
                                          "chimera_safety_check", "chimera_ethical_eval", "chimera_gate",
                                          "chimera_confident", "chimera_memory", "chimera_knowledge",
                                          "chimera_claims", "chimera_verify", "chimera_policy",
                                          "chimera_provenance_merge", "chimera_trace", "chimera_materials"],
                    "avoid_tools": ["chimera_embodied", "chimera_social", "chimera_transfer_learn", "chimera_evolve"],
                    "description": "AGI reasoning focus. Best for complex analysis, planning, and multi-step reasoning.",
                },
                "full": {
                    "recommended_tools": all_tool_names,
                    "avoid_tools": [],
                    "description": f"All {total_tool_count} tools active. Use only when the task genuinely spans all domains.",
                },
            }
            chosen = _MODES.get(mode, _MODES["minimal"])
            return _ok({
                "mode": mode,
                "description": chosen["description"],
                "recommended_tools": chosen["recommended_tools"],
                "avoid_tools": chosen["avoid_tools"],
                "tool_count_active": len(chosen["recommended_tools"]),
                "token_savings_note": (
                    f"Using {len(chosen['recommended_tools'])} tools instead of {total_tool_count} "
                    f"skips ~{len(chosen['avoid_tools']) * 60} schema tokens per session."
                ),
            })

        # ── chimera_budget_lock — session budget enforcement ──────────────
        elif name == "chimera_budget_lock":
            action         = arguments.get("action", "lock")
            max_out        = int(arguments.get("max_output_tokens", 0))
            tokens_gen     = int(arguments.get("tokens_generated", 0))
            label          = str(arguments.get("label", ""))

            if action == "lock":
                _session_budget["locked"]           = True
                _session_budget["max_output_tokens"] = max_out
                _session_budget["label"]             = label
                _session_budget["locked_at"]         = time.time()
                _session_budget["tokens_generated"]  = 0
                remaining = max_out
                pct_used  = 0.0
                status    = "locked"

            elif action == "check":
                if not _session_budget["locked"]:
                    return _ok({"locked": False, "message": "No active budget lock. Call chimera_budget_lock with action=lock first."})
                max_out   = _session_budget["max_output_tokens"]
                remaining = max(0, max_out - _session_budget["tokens_generated"])
                pct_used  = round(_session_budget["tokens_generated"] / max(max_out, 1) * 100, 1)
                status    = "ok" if pct_used < 70 else ("warn" if pct_used < 90 else "critical")

            elif action == "update":
                _session_budget["tokens_generated"] += tokens_gen
                max_out   = _session_budget["max_output_tokens"] or 0
                remaining = max(0, max_out - _session_budget["tokens_generated"])
                pct_used  = round(_session_budget["tokens_generated"] / max(max_out, 1) * 100, 1)
                status    = "ok" if pct_used < 70 else ("warn" if pct_used < 90 else "critical")

            elif action == "release":
                used = _session_budget["tokens_generated"]
                _session_budget.update({"locked": False, "max_output_tokens": None,
                                        "label": "", "locked_at": None, "tokens_generated": 0})
                return _ok({"released": True, "tokens_generated_total": used})

            else:
                return _err(f"Unknown action: {action}. Use lock|check|update|release.")

            recommendation = (
                "on_track" if status == "ok"
                else "consider_compressing_draft" if status == "warn"
                else "STOP_compress_draft_immediately"
            )
            return _ok({
                "locked":             _session_budget["locked"],
                "max_output_tokens":  _session_budget["max_output_tokens"],
                "tokens_generated":   _session_budget["tokens_generated"],
                "remaining_tokens":   remaining,
                "pct_used":           pct_used,
                "status":             status,
                "recommendation":     recommendation,
                "label":              _session_budget["label"],
            })

        # ── chimera_csm — Context Session Manager ─────────────────────────
        elif name == "chimera_csm":
            import re as _re

            prompt = str(arguments["prompt"])
            messages = arguments.get("messages", [])
            model = arguments.get("model", _DEFAULT_MODEL)
            task_complexity = arguments.get("task_complexity", "auto")
            algorithm = str(arguments.get("algorithm", "quantum")).lower()
            focus = _resolve_focus(arguments, prompt=prompt, messages=messages)

            if algorithm == "classic":
                _CSM_FILLER = [
                    r"\bplease note that\b", r"\bit is worth noting that\b",
                    r"\bit should be noted that\b", r"\bin order to\b",
                    r"\bbasically\b", r"\bactually\b", r"\bvery\b",
                    r"\bjust\b", r"\bsimply\b", r"\bquite\b",
                    r"\bof course\b", r"\bneedless to say\b",
                ]
                _CSM_CONTRACTIONS = {
                    r"\bdo not\b": "don't", r"\bdoes not\b": "doesn't",
                    r"\bdid not\b": "didn't", r"\bcannot\b": "can't",
                    r"\bwill not\b": "won't", r"\bwould not\b": "wouldn't",
                    r"\bshould not\b": "shouldn't", r"\bcould not\b": "couldn't",
                    r"\bit is\b": "it's", r"\bthat is\b": "that's",
                }

                def _csm_compress(text: str) -> str:
                    t = _re.sub(r"[ \t]+", " ", text)
                    t = _re.sub(r"\n{3,}", "\n\n", t).strip()
                    for _p in _CSM_FILLER:
                        t = _re.sub(_p, "", t, flags=_re.IGNORECASE)
                    t = _re.sub(r"[ \t]{2,}", " ", t).strip()
                    for _p, _r in _CSM_CONTRACTIONS.items():
                        t = _re.sub(_p, _r, t, flags=_re.IGNORECASE)
                    return t

                opt_prompt = _csm_compress(prompt)
                opt_messages: list[dict[str, Any]] = []
                if messages:
                    for message in messages:
                        content = normalize_content(message.get("content", ""))
                        opt_messages.append({"role": message.get("role", "user"), "content": _csm_compress(content)})
            else:
                prompt_result = _quantum.optimize_text(
                    prompt,
                    focus=focus,
                    preserve_code=True,
                    strategies=["whitespace", "dedup_sentences", "strip_filler"],
                    level="medium",
                )
                message_result = _quantum.compress_messages(messages, focus=focus, scorer=_scorer)
                opt_prompt = prompt_result.text
                opt_messages = message_result.messages

            # Step 3: count tokens
            original_tokens = _tbm.count_tokens(prompt)
            optimized_tokens = _tbm.count_tokens(opt_prompt)
            raw_history_tokens = _tbm.count_messages(messages)    if messages     else 0
            opt_history_tokens = _tbm.count_messages(opt_messages) if opt_messages else 0
            history_saved = max(0, raw_history_tokens - opt_history_tokens)
            total_input_tokens = optimized_tokens + opt_history_tokens

            tokens_saved = max(0, original_tokens - optimized_tokens) + history_saved
            savings_pct = round(tokens_saved / max(original_tokens + raw_history_tokens, 1) * 100, 1)

            # Step 4: schema overhead (lazy, cached)
            global _schema_overhead_cache
            if _schema_overhead_cache == 0:
                # Approximate: count tokens in all tool descriptions at runtime
                # We use a fixed estimate based on current description length
                _schema_overhead_cache = _tbm.count_tokens(
                    " ".join([
                        "chimera_run chimera_confident chimera_explore chimera_gate chimera_detect "
                        "chimera_constrain chimera_typecheck chimera_prove chimera_audit "
                        "chimera_claims chimera_verify chimera_provenance_merge chimera_policy chimera_trace chimera_materials "
                        "chimera_fracture chimera_optimize chimera_compress chimera_budget chimera_score "
                        "chimera_cost_estimate chimera_cost_track chimera_dashboard chimera_csm "
                        "chimera_budget_lock chimera_causal chimera_deliberate chimera_metacognize "
                        "chimera_meta_learn chimera_quantum_vote chimera_plan_goals chimera_world_model "
                        "chimera_safety_check chimera_ethical_eval chimera_embodied chimera_social "
                        "chimera_transfer_learn chimera_evolve chimera_self_model chimera_knowledge "
                        "chimera_memory chimera_mode chimera_batch chimera_summarize"
                    ])
                ) * 15  # ~15x multiplier: description + schema per tool

            schema_overhead = _schema_overhead_cache

            # Step 5: auto-detect task complexity
            prompt_lower = prompt.lower()
            if task_complexity == "auto":
                _simple_kw  = ["what is", "who is", "when was", "define ", "yes or no", "quick", "brief", "short"]
                _complex_kw = ["implement", "build a", "write code", "create a", "design", "refactor",
                               "debug", "full pipeline", "entire", "comprehensive", "step by step",
                               "architecture", "system", "deep dive"]
                if any(kw in prompt_lower for kw in _complex_kw):
                    task_complexity = "complex"
                elif any(kw in prompt_lower for kw in _simple_kw):
                    task_complexity = "simple"
                else:
                    task_complexity = "moderate"

            # Step 6: estimate output tokens
            _mult   = {"simple": 0.8, "moderate": 2.0, "complex": 4.0}
            _minout = {"simple": 50,  "moderate": 150, "complex": 500}
            _maxout = {"simple": 600, "moderate": 2000, "complex": 8000}
            est_output = int(total_input_tokens * _mult.get(task_complexity, 2.0))
            est_output = max(_minout[task_complexity], min(_maxout[task_complexity], est_output))

            # Step 7: cost estimate (includes schema overhead in total session view)
            in_price, out_price = _MODEL_PRICING.get(model, _MODEL_PRICING[_DEFAULT_MODEL])
            input_cost          = round(total_input_tokens * in_price  / 1_000_000, 6)
            output_cost         = round(est_output         * out_price / 1_000_000, 6)
            schema_cost         = round(schema_overhead    * in_price  / 1_000_000, 6)
            total_cost          = round(input_cost + output_cost, 6)
            session_total_cost  = round(total_cost + schema_cost, 6)
            unopt_in_cost       = round((original_tokens + raw_history_tokens) * in_price / 1_000_000, 6)
            cost_saved          = round(unopt_in_cost - input_cost, 6)

            # Step 8: proposal text
            proposal_text = (
                f"Token Budget Proposal\n"
                f"  Model:            {model}\n"
                f"  Task complexity:  {task_complexity}\n"
                f"\n"
                f"  Input (prompt):   {original_tokens:,} → {optimized_tokens:,} tokens (saved {original_tokens - optimized_tokens:,})\n"
                f"  History context:  {raw_history_tokens:,} → {opt_history_tokens:,} tokens (saved {history_saved:,})\n"
                f"  Total input:      {total_input_tokens:,} tokens\n"
                f"  Output budget:    {est_output:,} tokens\n"
                f"\n"
                f"  Est. turn cost:   ${total_cost:.6f} USD\n"
                f"  Schema overhead:  ~{schema_overhead:,} tokens (${schema_cost:.6f} USD, fixed per session)\n"
                f"  Total session:    ~${session_total_cost:.6f} USD\n"
                f"  Compression saved: ${cost_saved:.6f} USD\n"
                f"\n"
                f"  Approve this budget?  Reply:  approve  |  adjust <N>  |  skip"
            )

            return _ok({
                "optimized_prompt": opt_prompt,
                "optimized_messages": opt_messages,
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "raw_history_tokens": raw_history_tokens,
                "opt_history_tokens": opt_history_tokens,
                "history_saved": history_saved,
                "total_input_tokens": total_input_tokens,
                "tokens_saved": tokens_saved,
                "savings_pct": savings_pct,
                "schema_overhead_tokens": schema_overhead,
                "task_complexity": task_complexity,
                "proposed_output_tokens": est_output,
                "max_output_tokens": est_output,
                "total_tokens": total_input_tokens + est_output,
                "estimated_cost_usd": total_cost,
                "schema_overhead_cost_usd": schema_cost,
                "session_total_cost_usd": session_total_cost,
                "cost_saved_usd": cost_saved,
                "model": model,
                "proposal_text": proposal_text,
                "action": "SHOW_PROPOSAL_TO_USER",
                "token_count_method": _tbm.get_stats()["token_count_method"],
                "algorithm": algorithm,
                "focus_terms": extract_focus_terms(focus),
            })

        # ── chimera_batch — multi-tool single call ────────────────────────
        elif name == "chimera_batch":
            calls         = arguments.get("calls", [])
            stop_on_error = bool(arguments.get("stop_on_error", False))

            results: list[dict[str, Any]] = []
            for i, call in enumerate(calls):
                tool_name = call.get("tool", "")
                tool_args = call.get("args") or {}
                try:
                    r = await call_tool(tool_name, tool_args)
                    raw = r.content[0].text if r.content else "{}"
                    results.append({
                        "index":   i,
                        "tool":    tool_name,
                        "success": not r.isError,
                        "result":  json.loads(raw),
                    })
                    if stop_on_error and r.isError:
                        break
                except Exception as exc:
                    results.append({
                        "index":   i,
                        "tool":    tool_name,
                        "success": False,
                        "result":  {"error": str(exc)},
                    })
                    if stop_on_error:
                        break

            return _ok({
                "results":   results,
                "total":     len(calls),
                "executed":  len(results),
                "succeeded": sum(1 for r in results if r["success"]),
                "failed":    sum(1 for r in results if not r["success"]),
            })

        # ── chimera_summarize — LLM-free extractive summarizer ────────────
        elif name == "chimera_summarize":
            import re as _re
            import math as _math
            from collections import Counter as _Counter

            text          = arguments["text"]
            ratio         = float(arguments.get("ratio", 0.25))
            min_sentences = int(arguments.get("min_sentences", 3))
            auto_track    = bool(arguments.get("auto_track", True))
            track_model   = arguments.get("model", _DEFAULT_MODEL)
            namespace     = _state_namespace(arguments)

            sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sentences:
                return _ok({"summary": text, "sentences_in": 0, "sentences_out": 0,
                            "ratio_achieved": 1.0, "tokens_before": 0, "tokens_after": 0,
                            "savings_pct": 0.0})

            n_keep = min(max(min_sentences, int(len(sentences) * ratio)), len(sentences))
            if n_keep >= len(sentences):
                tb = _tbm.count_tokens(text)
                return _ok({"summary": text, "sentences_in": len(sentences),
                            "sentences_out": len(sentences), "ratio_achieved": 1.0,
                            "tokens_before": tb, "tokens_after": tb, "savings_pct": 0.0})

            _STOP = {
                "the","and","for","are","but","not","you","all","can","was","has","have",
                "this","that","with","they","been","from","its","your","into","than","then",
                "when","will","more","also","some","would","could","should","there","their",
                "these","those","which","about","after","before","being","doing","other",
            }

            def _tok(s: str) -> list[str]:
                return [w for w in _re.findall(r"\b[a-z]{3,}\b", s.lower()) if w not in _STOP]

            n     = len(sentences)
            tf    = _Counter(w for s in sentences for w in _tok(s))
            total = max(sum(tf.values()), 1)
            df: dict[str, int] = _Counter()
            for s in sentences:
                for w in set(_tok(s)):
                    df[w] += 1

            def _score(s: str, pos: int) -> float:
                words = _tok(s)
                if not words:
                    return 0.0
                tfidf = sum(
                    (tf[w] / total) * _math.log(n / max(df[w], 1) + 1) for w in words
                ) / len(words)
                pos_bonus = 0.10 if pos == 0 else (0.05 if pos == n - 1 else 0.0)
                len_bonus = min(len(words) / 20.0, 0.10)
                return tfidf + pos_bonus + len_bonus

            scored      = sorted(enumerate(sentences), key=lambda x: _score(x[1], x[0]), reverse=True)
            kept_idx    = sorted(idx for idx, _ in scored[:n_keep])
            summary     = " ".join(sentences[i] for i in kept_idx)
            tb          = _tbm.count_tokens(text)
            ta          = _tbm.count_tokens(summary)
            savings_pct = round((1 - ta / max(tb, 1)) * 100, 1)

            summ_track: dict[str, Any] | None = None
            if auto_track and tb > ta:
                summ_track = _get_cost_tracker(namespace).record(
                    tokens_before=tb,
                    tokens_after=ta,
                    model=track_model,
                    label="chimera_summarize",
                )

            summ_payload: dict[str, Any] = {
                "summary":          summary,
                "sentences_in":     len(sentences),
                "sentences_out":    n_keep,
                "ratio_achieved":   round(n_keep / len(sentences), 3),
                "tokens_before":    tb,
                "tokens_after":     ta,
                "savings_pct":      savings_pct,
            }
            if summ_track:
                summ_payload["tracked"] = {"request_id": summ_track["request_id"],
                                            "savings_usd": summ_track["savings"]}
            return _ok(summ_payload)

        # ── chimera_cache_mark ────────────────────────────────────────────
        elif name == "chimera_cache_mark":
            blocks = arguments.get("blocks") or []
            if not isinstance(blocks, list) or not blocks:
                return _err("chimera_cache_mark: 'blocks' must be a non-empty list")
            model = str(arguments.get("model") or "claude-sonnet-4-6")
            max_bp = int(arguments.get("max_breakpoints") or _CACHE_MAX_BREAKPOINTS)
            return _ok(_build_cache_blocks(blocks, model=model, max_breakpoints=max_bp))

        # ── chimera_log_compress ──────────────────────────────────────────
        elif name == "chimera_log_compress":
            text = arguments["text"]
            keep_patterns = arguments.get("keep_patterns")
            head = int(arguments.get("head_lines", 50))
            tail = int(arguments.get("tail_lines", 100))
            ctx_lines = int(arguments.get("context_lines", 2))
            namespace = _state_namespace(arguments)
            auto_track = bool(arguments.get("auto_track", True))
            track_model = str(arguments.get("model") or "claude-sonnet-4-6")
            payload = _compress_log(
                text,
                keep_patterns=keep_patterns,
                head_lines=head,
                tail_lines=tail,
                context_lines=ctx_lines,
            )
            tokens_before = _tbm.count_tokens(text)
            tokens_after = _tbm.count_tokens(payload["compressed_text"])
            payload["estimated_tokens_before"] = tokens_before
            payload["estimated_tokens_after"] = tokens_after
            payload["estimated_tokens_saved"] = max(0, tokens_before - tokens_after)
            if auto_track and tokens_before > tokens_after:
                tracked = _get_cost_tracker(namespace).record(
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    model=track_model,
                    label="chimera_log_compress",
                )
                payload["tracked"] = {
                    "request_id": tracked["request_id"],
                    "savings_usd": tracked["savings"],
                }
            return _ok(payload)

        # ── chimera_overhead_audit ────────────────────────────────────────
        elif name == "chimera_overhead_audit":
            return _ok(_audit_overhead(
                system_prompt=arguments.get("system_prompt"),
                tool_definitions=arguments.get("tool_definitions") or [],
                mcp_servers=arguments.get("mcp_servers") or [],
            ))

        # ── chimera_dedup_lookup ──────────────────────────────────────────
        elif name == "chimera_dedup_lookup":
            namespace = _state_namespace(arguments)
            action = str(arguments.get("action") or "list")
            if action == "get":
                key = str(arguments.get("key") or "").strip()
                if not key:
                    return _err("chimera_dedup_lookup: action=get requires 'key'")
                entry = _dedup_lookup(namespace, key)
                return _ok({"found": entry is not None, "entry": entry})
            if action == "clear":
                cleared = _dedup_clear(namespace)
                return _ok({"cleared_entries": cleared, "namespace": namespace})
            entries = _dedup_load(namespace)
            tool_filter = arguments.get("tool_name")
            if tool_filter:
                entries = [e for e in entries if e.get("tool_name") == tool_filter]
            return _ok({
                "namespace": namespace,
                "entry_count": len(entries),
                "total_hits": sum(int(e.get("hit_count") or 0) for e in entries),
                "entries": entries,
            })

        # ── chimera_session_report ────────────────────────────────────────
        elif name == "chimera_session_report":
            namespace = _state_namespace(arguments)
            include_dedup = bool(arguments.get("include_dedup", True))
            cost = _get_cost_tracker(namespace).summary()
            budget = _budget_snapshot("chimera_session_report", namespace)
            report: dict[str, Any] = {
                "namespace": namespace,
                "cost_summary": cost,
                "budget": budget,
            }
            if include_dedup:
                entries = _dedup_load(namespace)
                top = sorted(
                    entries,
                    key=lambda e: int(e.get("hit_count") or 0),
                    reverse=True,
                )[:10]
                report["dedup"] = {
                    "tracked_calls": len(entries),
                    "total_hits": sum(int(e.get("hit_count") or 0) for e in entries),
                    "top_repeated": [
                        {
                            "tool_name": e.get("tool_name"),
                            "hit_count": e.get("hit_count"),
                            "response_chars": e.get("response_chars"),
                        }
                        for e in top
                    ],
                }
            return _ok(report)

        else:
            return _err(f"Unknown tool: {name}")

    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")
    finally:
        _call_context.reset(_ctx_token)


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
