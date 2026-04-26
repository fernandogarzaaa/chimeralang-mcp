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
    @staticmethod
    def _tok(text: str) -> set:
        return set(re.sub(r"[^\w\s]", "", str(text).lower()).split())

    def deliberate(self, prompt: str, perspectives: list) -> dict:
        if not perspectives:
            return {"consensus": None, "perspectives": [], "divergence": 1.0}
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
    def __init__(self) -> None:
        self._entries: dict[str, _KBEntry] = {}

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


class _WorldModel:
    def __init__(self) -> None:
        self._facts: dict = {}

    def update(self, key: str, value: Any, confidence: float = 0.8) -> dict:
        self._facts[key] = {"value": value, "confidence": confidence, "updated_at": time.time()}
        return {"updated": key, "fact_count": len(self._facts)}

    def query(self, key: str | None = None) -> dict:
        if key:
            return self._facts.get(key, {"error": f"Key '{key}' not found"})
        return {"facts": self._facts, "fact_count": len(self._facts)}


class _SelfModel:
    def __init__(self) -> None:
        self._capabilities: dict = {}
        self._observations:  list = []

    def update(self, capability: str, level: str = "present", evidence: str = "") -> dict:
        self._capabilities[capability] = {"level": level, "evidence": evidence}
        return {"updated": capability, "capability_count": len(self._capabilities)}

    def reflect(self) -> dict:
        return {"capabilities": self._capabilities,
                "observations": self._observations[-10:]}


class _MemoryStore:
    def __init__(self) -> None:
        self._entries: list = []

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


class _MetaLearner:
    def __init__(self) -> None:
        self._adaptations: list = []

    def record_adaptation(self, context: str = "", action: str = "",
                          outcome: str = "", confidence: float = 0.5) -> dict:
        entry = {"context": context, "action": action, "outcome": outcome,
                 "confidence": confidence, "recorded_at": time.time()}
        self._adaptations.append(entry)
        return {"recorded": True, "total_adaptations": len(self._adaptations)}

    def get_stats(self) -> dict:
        return {"total_adaptations": len(self._adaptations),
                "recent": self._adaptations[-5:]}


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
    if any(w in g for w in ["build", "create", "implement", "develop", "write"]):
        strategy  = "iterative_build"
        sub_goals = ["Define requirements", "Design architecture",
                     "Implement core", "Test and validate", "Deploy and monitor"]
    elif any(w in g for w in ["analyze", "evaluate", "assess", "review"]):
        strategy  = "analytical_decomposition"
        sub_goals = ["Gather data", "Define evaluation criteria",
                     "Run analysis", "Synthesize findings", "Report conclusions"]
    elif any(w in g for w in ["fix", "debug", "resolve", "repair"]):
        strategy  = "diagnostic_repair"
        sub_goals = ["Reproduce issue", "Isolate root cause",
                     "Design fix", "Apply and verify", "Add regression test"]
    elif any(w in g for w in ["should", "whether", "decide", "choose"]):
        strategy  = "decision_framework"
        sub_goals = ["Define decision criteria", "Gather evidence",
                     "Analyze trade-offs", "Apply safety/ethics check", "Form verdict"]
    else:
        strategy  = "general_decomposition"
        sub_goals = ["Clarify scope", "Identify stakeholders",
                     "Map dependencies", "Execute in phases", "Verify outcomes"]
    return {
        "goal":                 goal,
        "best_known_strategy":  strategy,
        "sub_goals":            sub_goals,
        "confidence":           0.75,
        "note":                 "Heuristic decomposition — refine with domain knowledge",
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

# ── AGI component singletons (lazy-initialized) ───────────────────────────
_causal_reasoning:    _CausalReasoning | None    = None
_deliberation_engine: _DeliberationEngine | None = None
_safety_layer:        _SafetyLayer | None        = None
_ethical_reasoner:    _EthicalReasoning | None   = None
_kb:                  _KnowledgeBase | None      = None
_world_model_inst:    _WorldModel | None         = None
_self_model_inst:     _SelfModel | None          = None
_memory_store:        _MemoryStore | None        = None
_meta_learner_inst:   _MetaLearner | None        = None

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


def _get_kb() -> _KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = _KnowledgeBase()
    return _kb


def _get_world_model() -> _WorldModel:
    global _world_model_inst
    if _world_model_inst is None:
        _world_model_inst = _WorldModel()
    return _world_model_inst


def _get_self_model() -> _SelfModel:
    global _self_model_inst
    if _self_model_inst is None:
        _self_model_inst = _SelfModel()
    return _self_model_inst


def _get_memory() -> _MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = _MemoryStore()
    return _memory_store


def _get_meta_learner() -> _MetaLearner:
    global _meta_learner_inst
    if _meta_learner_inst is None:
        _meta_learner_inst = _MetaLearner()
    return _meta_learner_inst


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
            description="Hallucination detection. Strategies: range, dictionary, semantic, cross_reference, temporal, confidence_threshold. Returns flags with severity.",
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
            description="Execute ChimeraLang and generate a Merkle-chain integrity proof. Returns results + tamper-evident hash chain with root hash and verdict.",
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
            description="Session constraint audit: total calls, pass/fail counts, avg confidence, warnings, tools used.",
            inputSchema={
                "type": "object",
                "properties": {},
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
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="chimera_compress",
            description="Compress text via contractions/symbols. Levels: light, medium, aggressive. Returns compressed text and compression ratio.",
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
            description="Score messages 0–1 by importance (recency, type, density, replaceability). Lowest scores dropped first in lossy compression.",
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
                },
                "required": ["tokens_before", "tokens_after"],
            },
        ),
        Tool(
            name="chimera_dashboard",
            description="Session cost summary: tokens saved, dollars saved, avg compression %, last 10 events.",
            inputSchema={
                "type": "object",
                "properties": {},
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
            description="Multi-perspective deliberation. Returns Jaccard consensus, divergence score, and most-consensus perspective.",
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
                },
            },
        ),
        Tool(
            name="chimera_safety_check",
            description="Pattern-based content safety check. Returns is_safe, reason, blocked/allowed counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to validate"},
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
            description="LLM-free extractive summarizer. Ranks sentences by TF-IDF and returns top N. Use before passing long docs to other tools.",
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
                },
                "required": ["text"],
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
            summary  = _middleware.audit_summary()
            call_log = _middleware.call_log()
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
                    for r in call_log[-10:]
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
            return _ok(_get_deliberation().deliberate(prompt, perspectives))

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
            ml     = _get_meta_learner()
            if action == "record":
                return _ok(ml.record_adaptation(
                    context=arguments.get("context", ""),
                    action=arguments.get("action_taken", ""),
                    outcome=arguments.get("outcome", ""),
                    confidence=float(arguments.get("confidence", 0.5)),
                ))
            return _ok(ml.get_stats())

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
            wm     = _get_world_model()
            if action == "update":
                return _ok(wm.update(
                    key=arguments.get("key", ""),
                    value=arguments.get("value"),
                    confidence=float(arguments.get("confidence", 0.8)),
                ))
            return _ok(wm.query(key=arguments.get("key")))

        # ── AGI: chimera_safety_check ────────────────────────────────────────
        elif name == "chimera_safety_check":
            content  = arguments.get("content", "")
            sl       = _get_safety()
            is_safe, reason = sl.validate_content(content)
            return _ok({"is_safe": is_safe, "reason": reason,
                        "blocked_count": sl._blocked_count,
                        "allowed_count": sl._allowed_count})

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
            sm     = _get_self_model()
            if action == "update":
                return _ok(sm.update(
                    capability=arguments.get("capability", ""),
                    level=arguments.get("level", "present"),
                    evidence=arguments.get("evidence", ""),
                ))
            return _ok(sm.reflect())

        # ── AGI: chimera_knowledge ──────────────────────────────────────────
        elif name == "chimera_knowledge":
            action = arguments.get("action", "search")
            kb     = _get_kb()
            if action == "add":
                entry = kb.add(content=arguments.get("content", ""),
                               category=arguments.get("category", "general"),
                               tags=arguments.get("tags", []))
                return _ok({"added": True, "entry_id": entry.entry_id})
            elif action == "search":
                return _ok({"results": kb.search(query=arguments.get("query", ""))})
            elif action == "list":
                return _ok({"entries": len(kb._entries),
                            "categories": list({e.category for e in kb._entries.values()})})
            return _ok({"entry_count": len(kb._entries)})

        # ── AGI: chimera_memory ────────────────────────────────────────────
        elif name == "chimera_memory":
            action = arguments.get("action", "recall")
            mem    = _get_memory()
            if action == "store":
                return _ok(mem.store(content=arguments.get("content", ""),
                                     tags=arguments.get("tags", []),
                                     importance=float(arguments.get("importance", 0.5))))
            return _ok(mem.recall(query=arguments.get("query"),
                                  limit=int(arguments.get("limit", 10))))

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

        # ── chimera_mode — task-relevant tool guidance ─────────────────────
        elif name == "chimera_mode":
            mode = arguments.get("mode", "minimal")
            task = arguments.get("task_description", "").lower()

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
                    "recommended_tools": ["chimera_csm", "chimera_budget_lock", "chimera_gate", "chimera_confident", "chimera_memory"],
                    "avoid_tools": ["chimera_causal", "chimera_deliberate", "chimera_metacognize", "chimera_quantum_vote",
                                    "chimera_embodied", "chimera_social", "chimera_transfer_learn", "chimera_evolve",
                                    "chimera_prove", "chimera_audit", "chimera_self_model"],
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
                                          "chimera_confident", "chimera_memory", "chimera_knowledge"],
                    "avoid_tools": ["chimera_embodied", "chimera_social", "chimera_transfer_learn", "chimera_evolve"],
                    "description": "AGI reasoning focus. Best for complex analysis, planning, and multi-step reasoning.",
                },
                "full": {
                    "recommended_tools": ["all tools active"],
                    "avoid_tools": [],
                    "description": "All 37 tools active. Use only when the task genuinely spans all domains.",
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
                    f"Using {len(chosen['recommended_tools'])} tools instead of 37 "
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

            prompt          = str(arguments["prompt"])
            messages        = arguments.get("messages", [])
            model           = arguments.get("model", _DEFAULT_MODEL)
            task_complexity = arguments.get("task_complexity", "auto")

            _CSM_FILLER = [
                r"\bplease note that\b", r"\bit is worth noting that\b",
                r"\bit should be noted that\b", r"\bin order to\b",
                r"\bbasically\b", r"\bactually\b", r"\bvery\b",
                r"\bjust\b", r"\bsimply\b", r"\bquite\b",
                r"\bof course\b", r"\bneedless to say\b",
            ]
            _CSM_CONTRACTIONS = {
                r"\bdo not\b": "don't",   r"\bdoes not\b": "doesn't",
                r"\bdid not\b": "didn't", r"\bcannot\b": "can't",
                r"\bwill not\b": "won't",  r"\bwould not\b": "wouldn't",
                r"\bshould not\b": "shouldn't", r"\bcould not\b": "couldn't",
                r"\bit is\b": "it's",     r"\bthat is\b": "that's",
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

            # Step 1: optimize prompt
            opt_prompt = _csm_compress(prompt)

            # Step 2: compress message history (returns optimized_messages)
            opt_messages: list[dict[str, Any]] = []
            if messages:
                for m in messages:
                    content = str(m.get("content", ""))
                    opt_messages.append({"role": m.get("role", "user"), "content": _csm_compress(content)})

            # Step 3: count tokens
            original_tokens    = _tbm.count_tokens(prompt)
            optimized_tokens   = _tbm.count_tokens(opt_prompt)
            raw_history_tokens = _tbm.count_messages(messages)    if messages     else 0
            opt_history_tokens = _tbm.count_messages(opt_messages) if opt_messages else 0
            history_saved      = max(0, raw_history_tokens - opt_history_tokens)
            total_input_tokens = optimized_tokens + opt_history_tokens

            tokens_saved = max(0, original_tokens - optimized_tokens) + history_saved
            savings_pct  = round(tokens_saved / max(original_tokens + raw_history_tokens, 1) * 100, 1)

            # Step 4: schema overhead (lazy, cached)
            global _schema_overhead_cache
            if _schema_overhead_cache == 0:
                # Approximate: count tokens in all tool descriptions at runtime
                # We use a fixed estimate based on current description length
                _schema_overhead_cache = _tbm.count_tokens(
                    " ".join([
                        "chimera_run chimera_confident chimera_explore chimera_gate chimera_detect "
                        "chimera_constrain chimera_typecheck chimera_prove chimera_audit "
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
                "optimized_prompt":      opt_prompt,
                "optimized_messages":    opt_messages,
                "original_tokens":       original_tokens,
                "optimized_tokens":      optimized_tokens,
                "raw_history_tokens":    raw_history_tokens,
                "opt_history_tokens":    opt_history_tokens,
                "history_saved":         history_saved,
                "total_input_tokens":    total_input_tokens,
                "tokens_saved":          tokens_saved,
                "savings_pct":           savings_pct,
                "schema_overhead_tokens": schema_overhead,
                "task_complexity":       task_complexity,
                "proposed_output_tokens": est_output,
                "max_output_tokens":     est_output,
                "total_tokens":          total_input_tokens + est_output,
                "estimated_cost_usd":    total_cost,
                "schema_overhead_cost_usd": schema_cost,
                "session_total_cost_usd": session_total_cost,
                "cost_saved_usd":        cost_saved,
                "model":                 model,
                "proposal_text":         proposal_text,
                "action":                "SHOW_PROPOSAL_TO_USER",
                "token_count_method":    _tbm.get_stats()["token_count_method"],
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

            return _ok({
                "summary":          summary,
                "sentences_in":     len(sentences),
                "sentences_out":    n_keep,
                "ratio_achieved":   round(n_keep / len(sentences), 3),
                "tokens_before":    tb,
                "tokens_after":     ta,
                "savings_pct":      savings_pct,
            })

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
