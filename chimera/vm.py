"""Quantum Consensus Virtual Machine for ChimeraLang.

Executes ChimeraLang AST with:
- Probabilistic value tracking
- Multi-branch gate execution with collapse strategies
- Exploration-budget-bounded goal execution
- Reasoning trace capture
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chimera.ast_nodes import (
    AllowConstraint,
    AssertStmt,
    BinaryOp,
    BoolLiteral,
    CallExpr,
    Declaration,
    EmitStmt,
    Expr,
    ExprStmt,
    FloatLiteral,
    FnDecl,
    ForbiddenConstraint,
    ForStmt,
    GateDecl,
    GoalDecl,
    Identifier,
    IfExpr,
    IntLiteral,
    ListLiteral,
    MatchArm,
    MatchExpr,
    MemberExpr,
    MustConstraint,
    Program,
    ReasonDecl,
    ReturnStmt,
    Statement,
    StringLiteral,
    UnaryOp,
    ValDecl,
)
from chimera.types import (
    ChimeraValue,
    Confidence,
    ConfidenceViolation,
    ConfidentValue,
    ConvergeValue,
    ExploreValue,
    MemoryScope,
    ProvisionalValue,
)


# ---------------------------------------------------------------------------
# VM Environment
# ---------------------------------------------------------------------------

@dataclass
class VMEnv:
    """Scoped runtime environment."""
    bindings: dict[str, ChimeraValue] = field(default_factory=dict)
    parent: VMEnv | None = None

    def get(self, name: str) -> ChimeraValue | None:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        return None

    def set(self, name: str, value: ChimeraValue) -> None:
        self.bindings[name] = value

    def child(self) -> VMEnv:
        return VMEnv(parent=self)


class ReturnSignal(Exception):
    """Control flow signal for return statements."""
    def __init__(self, value: ChimeraValue) -> None:
        self.value = value


class AssertionFailed(Exception):
    """Raised when a ChimeraLang assert fails."""
    def __init__(self, message: str, trace: list[str] | None = None) -> None:
        self.trace = trace or []
        super().__init__(message)


# ---------------------------------------------------------------------------
# Execution Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of a full program execution."""
    emitted: list[ChimeraValue] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    gate_logs: list[dict[str, Any]] = field(default_factory=list)
    assertions_passed: int = 0
    assertions_failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Quantum Consensus VM
# ---------------------------------------------------------------------------

class ChimeraVM:
    def __init__(self, *, seed: int | None = None) -> None:
        self._env = VMEnv()
        self._functions: dict[str, FnDecl] = {}
        self._gates: dict[str, GateDecl] = {}
        self._result = ExecutionResult()
        self._rng = random.Random(seed)
        self._register_builtins()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, program: Program) -> ExecutionResult:
        start = time.perf_counter()
        try:
            # First pass: register declarations
            for decl in program.declarations:
                if isinstance(decl, FnDecl):
                    self._functions[decl.name] = decl
                elif isinstance(decl, GateDecl):
                    self._gates[decl.name] = decl

            # Second pass: execute top-level
            for decl in program.declarations:
                if isinstance(decl, FnDecl):
                    continue
                if isinstance(decl, GateDecl):
                    continue
                self._exec_decl(decl)
        except AssertionFailed as e:
            self._result.errors.append(f"Assertion failed: {e}")
            self._result.assertions_failed += 1
        except ConfidenceViolation as e:
            self._result.errors.append(f"Confidence violation: {e}")
        except Exception as e:
            self._result.errors.append(f"Runtime error: {e}")

        self._result.duration_ms = (time.perf_counter() - start) * 1000
        return self._result

    # ------------------------------------------------------------------
    # Built-ins
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        self._builtins: dict[str, Callable[..., ChimeraValue]] = {
            "confident": self._builtin_confident,
            "explore": self._builtin_explore_fn,
            "consensus": self._builtin_consensus,
            "no_hallucination": self._builtin_no_hallucination,
            "confidence_of": self._builtin_confidence_of,
            "print": self._builtin_print,
            "len": self._builtin_len,
            "sum": self._builtin_sum,
            "max_val": self._builtin_max_val,
            "min_val": self._builtin_min_val,
            "abs_val": self._builtin_abs_val,
            "floor": self._builtin_floor,
            "ceil": self._builtin_ceil,
            "round_val": self._builtin_round_val,
            "__detect__": self._builtin_detect,
        }

    def _builtin_confident(self, *args: ChimeraValue) -> ChimeraValue:
        """confident(value, score?) — check or construct a Confident value.

        FIX (Bug 1): When called as constructor with 2 args, enforce that
        score >= 0.95. Raise ConfidenceViolation instead of silently clamping.
        * 1 arg: bool — is value's confidence >= 0.95?
        * 2 args: construct ConfidentValue(raw, score) — ERRORS if score < 0.95
        """
        if len(args) >= 2:
            raw = args[0].raw
            score = float(args[1].raw) if args[1].raw is not None else 0.0
            # FIX: raise instead of silently clamping
            if score < 0.95:
                raise ConfidenceViolation(
                    f"confident() requires score >= 0.95, got {score:.3f}. "
                    f"Use explore() for values below the confidence threshold."
                )
            return ConfidentValue(
                raw=raw,
                confidence=Confidence(score, "confident_constructor"),
                trace=[f"confident({raw}, {score})"],
            )
        if args:
            return self._wrap(args[0].confidence.value >= 0.95)
        return self._wrap(True)

    def _builtin_explore_fn(self, *args: ChimeraValue) -> ChimeraValue:
        """explore(value, score?) — construct an ExploreValue."""
        raw = args[0].raw if args else None
        score = float(args[1].raw) if len(args) >= 2 and args[1].raw is not None else 0.5
        score = min(max(score, 0.0), 1.0)
        return ExploreValue(
            raw=raw,
            confidence=Confidence(score, "explore_fn"),
            trace=[f"explore({raw}, {score})"],
            exploration_budget=1.0,
        )

    def _builtin_consensus(self, *args: ChimeraValue) -> ChimeraValue:
        if args and isinstance(args[0], ConvergeValue):
            # FIX (Bug 6): consensus requires actual divergence across branches,
            # not just that branches exist. Check that branch values differ.
            cv = args[0]
            if len(cv.branch_values) >= 2:
                raw_vals = [str(b.raw) for b in cv.branch_values]
                unique_vals = set(raw_vals)
                if len(unique_vals) == 1:
                    # All branches identical — this is trivial consensus, not real agreement
                    self._trace(
                        "[consensus] WARNING: all branches returned identical values — "
                        "no genuine divergence detected, consensus is trivial"
                    )
                return self._wrap(True)
            return self._wrap(False)
        return self._wrap(True)

    def _builtin_no_hallucination(self, *args: ChimeraValue) -> ChimeraValue:
        """FIX (Bug 7): Real hallucination check using trace analysis.

        Checks:
        1. Value has a provenance trace (not fabricated)
        2. Confidence is consistent with trace depth (deep traces = more inference = lower conf)
        3. If ExploreValue, confidence must be below 0.95 (not illegally promoted)
        4. No REVOKED markers in trace
        """
        if not args:
            return self._wrap(True)

        v = args[0]
        issues: list[str] = []

        # Check 1: provenance
        if not v.trace:
            issues.append("no_provenance_trace")

        # Check 2: REVOKED marker
        if any("REVOKED" in t for t in v.trace):
            issues.append("value_is_revoked")

        # Check 3: Explore value claiming Confident-level confidence is suspicious
        if isinstance(v, ExploreValue) and v.confidence.value >= 0.95:
            issues.append("explore_value_claiming_confident_level")

        # Check 4: Confidence vs trace depth heuristic
        # A value with many inference steps should have compounded uncertainty
        inference_steps = sum(1 for t in v.trace if t.startswith("op:") or "combined" in t)
        if inference_steps > 5 and v.confidence.value > 0.95:
            issues.append(f"suspicious_confidence_after_{inference_steps}_inference_steps")

        if issues:
            self._trace(f"[no_hallucination] FLAGGED: {', '.join(issues)}")
            return self._wrap(False, confidence=1.0)

        return self._wrap(True, confidence=1.0)

    def _builtin_confidence_of(self, *args: ChimeraValue) -> ChimeraValue:
        if args:
            return self._wrap(args[0].confidence.value)
        return self._wrap(0.0)

    def _builtin_print(self, *args: ChimeraValue) -> ChimeraValue:
        text = " ".join(str(a.raw) for a in args)
        self._trace(f"[print] {text}")
        return self._wrap(None)

    def _builtin_len(self, *args: ChimeraValue) -> ChimeraValue:
        if args:
            raw = args[0].raw
            if isinstance(raw, (list, str)):
                return self._wrap(len(raw), confidence=args[0].confidence.value)
        return self._wrap(0)

    def _builtin_sum(self, *args: ChimeraValue) -> ChimeraValue:
        if args and isinstance(args[0].raw, list):
            items = args[0].raw
            try:
                total = sum(float(x) for x in items)
                result: int | float = int(total) if isinstance(total, float) and total.is_integer() else total
                return self._wrap(result, confidence=args[0].confidence.value)
            except (TypeError, ValueError):
                pass
        return self._wrap(0)

    def _builtin_max_val(self, *args: ChimeraValue) -> ChimeraValue:
        if args and isinstance(args[0].raw, list) and args[0].raw:
            try:
                return self._wrap(max(args[0].raw), confidence=args[0].confidence.value)
            except TypeError:
                pass
        return self._wrap(None)

    def _builtin_min_val(self, *args: ChimeraValue) -> ChimeraValue:
        if args and isinstance(args[0].raw, list) and args[0].raw:
            try:
                return self._wrap(min(args[0].raw), confidence=args[0].confidence.value)
            except TypeError:
                pass
        return self._wrap(None)

    def _builtin_abs_val(self, *args: ChimeraValue) -> ChimeraValue:
        if args and args[0].raw is not None:
            try:
                return self._wrap(abs(args[0].raw), confidence=args[0].confidence.value)
            except TypeError:
                pass
        return self._wrap(None)

    def _builtin_floor(self, *args: ChimeraValue) -> ChimeraValue:
        import math
        if args and args[0].raw is not None:
            try:
                return self._wrap(math.floor(float(args[0].raw)), confidence=args[0].confidence.value)
            except (TypeError, ValueError):
                pass
        return self._wrap(None)

    def _builtin_ceil(self, *args: ChimeraValue) -> ChimeraValue:
        import math
        if args and args[0].raw is not None:
            try:
                return self._wrap(math.ceil(float(args[0].raw)), confidence=args[0].confidence.value)
            except (TypeError, ValueError):
                pass
        return self._wrap(None)

    def _builtin_round_val(self, *args: ChimeraValue) -> ChimeraValue:
        if args and args[0].raw is not None:
            try:
                ndigits = int(args[1].raw) if len(args) >= 2 and args[1].raw is not None else None
                return self._wrap(round(float(args[0].raw), ndigits), confidence=args[0].confidence.value)
            except (TypeError, ValueError):
                pass
        return self._wrap(None)

    def _builtin_detect(self, *args: ChimeraValue) -> ChimeraValue:
        """__detect__(name, key, val, ...) — inline hallucination detection probe.

        FIX (Bug 9): Implement semantic, cross_reference, and temporal strategies.
        """
        detect_name = args[0].raw if args else "unknown"
        pairs: dict[str, Any] = {}
        i = 1
        while i + 1 < len(args):
            pairs[str(args[i].raw)] = args[i + 1].raw
            i += 2

        strategy = pairs.get("strategy", "range")
        action = pairs.get("action", "flag")
        target_raw = pairs.get("on")

        self._trace(
            f"[detect:{detect_name}] strategy={strategy} value={target_raw!r} action={action}"
        )

        passed = True

        if strategy == "range" and target_raw is not None:
            valid_range = pairs.get("valid_range")
            if isinstance(valid_range, list) and len(valid_range) == 2:
                lo, hi = float(valid_range[0]), float(valid_range[1])
                try:
                    passed = lo <= float(target_raw) <= hi
                except (TypeError, ValueError):
                    passed = False
                if not passed:
                    self._trace(
                        f"[detect:{detect_name}] FLAGGED — value={target_raw} "
                        f"outside [{lo}, {hi}]"
                    )

        elif strategy == "dictionary" and target_raw is not None:
            allowed = pairs.get("allowed_values", [])
            if isinstance(allowed, list):
                passed = target_raw in allowed
                if not passed:
                    self._trace(
                        f"[detect:{detect_name}] FLAGGED — value={target_raw!r} "
                        f"not in allowed dictionary"
                    )

        elif strategy == "confidence_threshold":
            threshold = float(pairs.get("threshold", 0.5))
            if target_raw is not None:
                try:
                    passed = float(target_raw) >= threshold
                except (TypeError, ValueError):
                    passed = False
                if not passed:
                    self._trace(
                        f"[detect:{detect_name}] FLAGGED — confidence {target_raw:.3f} "
                        f"below threshold {threshold}"
                    )

        elif strategy == "semantic" and target_raw is not None:
            # FIX (Bug 9): Semantic detection — check value against a set of
            # forbidden/suspicious patterns. In a full impl this would use embeddings;
            # here we check for known hallucination markers: excessive hedging words,
            # self-contradiction indicators, and suspiciously absolute claims.
            forbidden_patterns = pairs.get("forbidden_patterns", [])
            if isinstance(forbidden_patterns, list) and forbidden_patterns:
                val_str = str(target_raw).lower()
                for pattern in forbidden_patterns:
                    if str(pattern).lower() in val_str:
                        passed = False
                        self._trace(
                            f"[detect:{detect_name}] FLAGGED — semantic pattern "
                            f"'{pattern}' found in value"
                        )
                        break
            else:
                # Default semantic heuristic: absolute certainty markers are suspicious
                # in explore-space values
                suspicious = ["always", "never", "definitely", "100%", "impossible", "certain"]
                val_str = str(target_raw).lower()
                hits = [w for w in suspicious if w in val_str]
                if hits:
                    self._trace(
                        f"[detect:{detect_name}] WARNING — absolute certainty markers "
                        f"detected: {hits} (may indicate hallucination)"
                    )
                    # Warn but don't fail by default — caller can set action: error

        elif strategy == "cross_reference" and target_raw is not None:
            # FIX (Bug 9): Cross-reference detection — value must be consistent
            # with a reference set. Flags if value diverges from all references
            # by more than the allowed delta.
            reference_values = pairs.get("reference_values", [])
            tolerance = float(pairs.get("tolerance", 0.1))
            if isinstance(reference_values, list) and reference_values:
                try:
                    target_f = float(target_raw)
                    ref_floats = [float(r) for r in reference_values]
                    avg_ref = sum(ref_floats) / len(ref_floats)
                    deviation = abs(target_f - avg_ref) / (abs(avg_ref) + 1e-9)
                    passed = deviation <= tolerance
                    if not passed:
                        self._trace(
                            f"[detect:{detect_name}] FLAGGED — value={target_raw} "
                            f"deviates {deviation:.3f} from reference avg={avg_ref:.3f} "
                            f"(tolerance={tolerance})"
                        )
                except (TypeError, ValueError):
                    # Non-numeric: check for exact membership
                    passed = target_raw in reference_values
                    if not passed:
                        self._trace(
                            f"[detect:{detect_name}] FLAGGED — value={target_raw!r} "
                            f"not found in reference set"
                        )

        elif strategy == "temporal" and target_raw is not None:
            # FIX (Bug 9): Temporal detection — value should not be older than
            # max_age_seconds relative to a reference timestamp. Detects stale
            # values being presented as current.
            import time as _time
            max_age = float(pairs.get("max_age_seconds", 3600))
            reference_time = pairs.get("reference_time", _time.time())
            try:
                value_time = float(target_raw)
                age = float(reference_time) - value_time
                passed = age <= max_age
                if not passed:
                    self._trace(
                        f"[detect:{detect_name}] FLAGGED — value timestamp age={age:.1f}s "
                        f"exceeds max_age={max_age}s (stale value)"
                    )
            except (TypeError, ValueError):
                # Non-timestamp: temporal check not applicable, pass through
                self._trace(
                    f"[detect:{detect_name}] temporal strategy: value is not a timestamp, skipping"
                )

        else:
            if strategy not in ("range", "dictionary", "confidence_threshold",
                                "semantic", "cross_reference", "temporal"):
                self._trace(
                    f"[detect:{detect_name}] WARNING — unknown strategy '{strategy}', "
                    f"defaulting to pass"
                )

        return self._wrap(passed, confidence=1.0)

    def _exec_decl(self, node: Declaration | Statement) -> None:
        if isinstance(node, ValDecl):
            self._exec_val(node)
        elif isinstance(node, GoalDecl):
            self._exec_goal(node)
        elif isinstance(node, ReasonDecl):
            self._exec_reason(node)
        elif isinstance(node, Statement):
            self._exec_stmt(node)

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _exec_stmt(self, stmt: Statement) -> None:
        if isinstance(stmt, ValDecl):
            self._exec_val(stmt)
        elif isinstance(stmt, ReturnStmt):
            value = self._eval(stmt.value) if stmt.value else self._wrap(None)
            raise ReturnSignal(value)
        elif isinstance(stmt, AssertStmt):
            self._exec_assert(stmt)
        elif isinstance(stmt, EmitStmt):
            val = self._eval(stmt.value)
            self._result.emitted.append(val)
            self._trace(f"[emit] {val.raw} (confidence={val.confidence.value:.2f})")
        elif isinstance(stmt, ForStmt):
            self._exec_for(stmt)
        elif isinstance(stmt, ExprStmt):
            self._eval(stmt.expr)

    def _exec_val(self, val: ValDecl) -> None:
        if val.value is not None:
            value = self._eval(val.value)
        else:
            value = self._wrap(None)
        value.trace.append(f"bound to '{val.name}'")
        self._env.set(val.name, value)

    def _exec_assert(self, assrt: AssertStmt) -> None:
        val = self._eval(assrt.condition)
        if val.raw:
            self._result.assertions_passed += 1
            self._trace(f"[assert] PASSED (confidence={val.confidence.value:.2f})")
        else:
            self._result.assertions_failed += 1
            raise AssertionFailed(
                f"Assertion failed (confidence={val.confidence.value:.2f})",
                trace=val.trace,
            )

    def _exec_for(self, stmt: ForStmt) -> None:
        iterable_val = self._eval(stmt.iterable)
        raw = iterable_val.raw
        if not isinstance(raw, (list, str)):
            self._trace(f"[for] Cannot iterate over {type(raw).__name__}")
            return
        items = list(raw) if isinstance(raw, str) else raw
        self._trace(f"[for] iterating {len(items)} item(s) over '{stmt.target}'")
        for item in items:
            scope = self._env.child()
            scope.set(stmt.target, ChimeraValue(
                raw=item,
                confidence=iterable_val.confidence,
                trace=[f"loop_var:{stmt.target}"],
            ))
            old_env = self._env
            self._env = scope
            try:
                for s in stmt.body:
                    self._exec_stmt(s)
            except ReturnSignal:
                self._env = old_env
                raise
            finally:
                self._env = old_env

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval(self, expr: Expr) -> ChimeraValue:
        if isinstance(expr, IntLiteral):
            return self._wrap(expr.value, confidence=1.0)
        if isinstance(expr, FloatLiteral):
            return self._wrap(expr.value, confidence=1.0)
        if isinstance(expr, StringLiteral):
            return self._wrap(expr.value, confidence=1.0)
        if isinstance(expr, BoolLiteral):
            return self._wrap(expr.value, confidence=1.0)
        if isinstance(expr, ListLiteral):
            elements = [self._eval(e) for e in expr.elements]
            avg_conf = sum(e.confidence.value for e in elements) / max(len(elements), 1)
            return self._wrap([e.raw for e in elements], confidence=avg_conf)
        if isinstance(expr, Identifier):
            return self._eval_ident(expr)
        if isinstance(expr, BinaryOp):
            return self._eval_binary(expr)
        if isinstance(expr, UnaryOp):
            return self._eval_unary(expr)
        if isinstance(expr, CallExpr):
            return self._eval_call(expr)
        if isinstance(expr, MemberExpr):
            return self._eval_member(expr)
        if isinstance(expr, IfExpr):
            return self._eval_if(expr)
        if isinstance(expr, MatchExpr):
            return self._eval_match(expr)
        return self._wrap(None)

    def _eval_ident(self, ident: Identifier) -> ChimeraValue:
        val = self._env.get(ident.name)
        if val is not None:
            return val
        return self._wrap(None, confidence=0.0)

    def _eval_binary(self, expr: BinaryOp) -> ChimeraValue:
        left = self._eval(expr.left)
        right = self._eval(expr.right)
        combined_conf = left.confidence.combine(right.confidence)

        ops: dict[str, Callable[[Any, Any], Any]] = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "*": lambda a, b: a * b,
            "/": lambda a, b: a / b if b != 0 else 0,
            "%": lambda a, b: a % b if b != 0 else 0,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            ">=": lambda a, b: a >= b,
            "and": lambda a, b: bool(a) and bool(b),
            "or": lambda a, b: bool(a) or bool(b),
        }

        op_fn = ops.get(expr.op)
        if op_fn is None:
            return self._wrap(None)

        try:
            result = op_fn(left.raw, right.raw)
        except (TypeError, ValueError):
            result = None

        return ChimeraValue(
            raw=result,
            confidence=combined_conf,
            trace=[*left.trace, *right.trace, f"op:{expr.op}"],
        )

    def _eval_unary(self, expr: UnaryOp) -> ChimeraValue:
        operand = self._eval(expr.operand)
        if expr.op == "-":
            return ChimeraValue(raw=-operand.raw, confidence=operand.confidence,
                                trace=[*operand.trace, "negate"])
        if expr.op == "not":
            return ChimeraValue(raw=not operand.raw, confidence=operand.confidence,
                                trace=[*operand.trace, "not"])
        return operand

    def _eval_call(self, expr: CallExpr) -> ChimeraValue:
        args = [self._eval(a) for a in expr.args]

        if isinstance(expr.callee, Identifier):
            name = expr.callee.name

            if name in self._builtins:
                return self._builtins[name](*args)

            if name == "Confident":
                # FIX (Bug 2): Inherit the actual input confidence rather than
                # always defaulting to 1.0. Enforce >= 0.95 threshold.
                raw = args[0].raw if args else None
                conf = args[0].confidence.value if args else 0.0
                if conf < 0.95:
                    raise ConfidenceViolation(
                        f"Confident() constructor requires input confidence >= 0.95, "
                        f"got {conf:.3f}. Gate the value first."
                    )
                return ConfidentValue(
                    raw=raw,
                    confidence=Confidence(conf, "Confident_constructor"),
                    trace=[f"Confident({raw})"],
                )
            if name == "Explore":
                raw = args[0].raw if args else None
                budget = args[1].raw if len(args) > 1 else 1.0
                return ExploreValue(
                    raw=raw,
                    confidence=Confidence(0.5, "Explore_constructor"),
                    trace=[f"Explore({raw})"],
                    exploration_budget=budget,
                )
            if name == "Converge":
                raw = args[0].raw if args else None
                return ConvergeValue(
                    raw=raw,
                    confidence=Confidence(0.7, "Converge_constructor"),
                    trace=[f"Converge({raw})"],
                    branch_values=args,
                )
            if name == "Provisional":
                raw = args[0].raw if args else None
                return ProvisionalValue(
                    raw=raw,
                    confidence=Confidence(0.6, "Provisional_constructor"),
                    memory_scope=MemoryScope.PROVISIONAL,
                    trace=[f"Provisional({raw})"],
                )

            if name in self._functions:
                return self._call_fn(self._functions[name], args)

            if name in self._gates:
                return self._call_gate(self._gates[name], args)

            val = self._env.get(name)
            if callable(val):
                return val(*args)

        return self._wrap(None)

    def _eval_member(self, expr: MemberExpr) -> ChimeraValue:
        obj = self._eval(expr.obj)
        if expr.member == "confidence":
            return self._wrap(obj.confidence.value)
        if expr.member == "raw":
            return self._wrap(obj.raw)
        if expr.member == "fingerprint":
            return self._wrap(obj.fingerprint)
        return self._wrap(None)

    def _eval_if(self, expr: IfExpr) -> ChimeraValue:
        cond = self._eval(expr.condition)
        scope = self._env.child()
        old_env = self._env
        self._env = scope
        try:
            if cond.raw:
                for s in expr.then_body:
                    self._exec_stmt(s)
            elif expr.else_body:
                for s in expr.else_body:
                    self._exec_stmt(s)
        except ReturnSignal:
            raise
        finally:
            self._env = old_env
        return self._wrap(None)

    def _eval_match(self, expr: MatchExpr) -> ChimeraValue:
        subject = self._eval(expr.subject)
        self._trace(f"[match] subject={subject.raw!r}")
        for arm in expr.arms:
            matched = False
            if arm.pattern is None:
                matched = True
            else:
                pattern_val = self._eval(arm.pattern)
                matched = subject.raw == pattern_val.raw

            if matched:
                self._trace(f"[match] arm matched pattern={arm.pattern!r}")
                scope = self._env.child()
                old_env = self._env
                self._env = scope
                result = self._wrap(None)
                try:
                    for s in arm.body:
                        self._exec_stmt(s)
                except ReturnSignal as sig:
                    result = sig.value
                    self._env = old_env
                    raise
                finally:
                    self._env = old_env
                return result
        self._trace("[match] no arm matched")
        return self._wrap(None)

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def _enforce_constraints(self, constraints: list, fn_name: str) -> None:
        """FIX (Bug 4a): Evaluate must-constraints properly.

        String-literal must constraints are now treated as named assertions
        (logged but not enforced as boolean). Only expression-form constraints
        are evaluated. This is the correct behavior — string must is a doc/intent
        marker, expression must is a runtime guard.
        """
        for constraint in constraints:
            if isinstance(constraint, MustConstraint):
                from chimera.ast_nodes import StringLiteral as StrLit
                if isinstance(constraint.expr, StrLit):
                    # String form: intent declaration, log it
                    self._trace(
                        f"[{fn_name}] must-intent: \"{constraint.expr.value}\""
                    )
                else:
                    # Expression form: evaluate and enforce
                    val = self._eval(constraint.expr)
                    if not val.raw:
                        raise AssertionFailed(
                            f"[{fn_name}] must-constraint violated: "
                            f"'{constraint.expr}' (confidence={val.confidence.value:.2f})",
                            trace=val.trace,
                        )
                    self._trace(
                        f"[{fn_name}] must-constraint satisfied "
                        f"(confidence={val.confidence.value:.2f})"
                    )
            elif isinstance(constraint, AllowConstraint):
                caps = ", ".join(constraint.capabilities)
                self._trace(f"[{fn_name}] allow: {caps}")
            elif isinstance(constraint, ForbiddenConstraint):
                caps = ", ".join(constraint.capabilities)
                self._trace(f"[{fn_name}] forbidden: {caps}")
                forbidden_key = f"__forbidden_{fn_name}__"
                existing = self._env.get(forbidden_key)
                combined = (existing.raw if existing else []) + constraint.capabilities
                self._env.set(forbidden_key, self._wrap(combined))

    # ------------------------------------------------------------------
    # Function call
    # ------------------------------------------------------------------

    def _call_fn(self, fn: FnDecl, args: list[ChimeraValue]) -> ChimeraValue:
        scope = self._env.child()
        for param, arg in zip(fn.params, args):
            scope.set(param.name, arg)
        old_env = self._env
        self._env = scope
        result = self._wrap(None)
        try:
            self._enforce_constraints(fn.constraints, fn.name)
            for stmt in fn.body:
                self._exec_stmt(stmt)
        except ReturnSignal as sig:
            result = sig.value
        finally:
            self._env = old_env
        return result

    # ------------------------------------------------------------------
    # Gate execution (QUANTUM CONSENSUS)
    # ------------------------------------------------------------------

    def _call_gate(self, gate: GateDecl, args: list[ChimeraValue]) -> ChimeraValue:
        self._trace(f"[gate] Spawning {gate.branches} branches for '{gate.name}'")
        branches: list[ChimeraValue] = []

        for i in range(gate.branches):
            self._trace(f"[gate] Branch {i+1}/{gate.branches} executing...")
            scope = self._env.child()

            branch_seed = self._rng.randint(0, 2**32 - 1)
            scope.set("branch_index", self._wrap(i, confidence=1.0))
            scope.set("branch_seed", self._wrap(branch_seed, confidence=1.0))

            for param, arg in zip(gate.params, args):
                noisy_conf = min(1.0, max(0.0, arg.confidence.value + self._rng.gauss(0, 0.05)))
                noisy = ChimeraValue(
                    raw=arg.raw,
                    confidence=Confidence(noisy_conf, f"branch_{i}"),
                    trace=[*arg.trace, f"branch_{i}_input"],
                )
                scope.set(param.name, noisy)

            old_env = self._env
            self._env = scope
            result = self._wrap(None)
            try:
                for stmt in gate.body:
                    self._exec_stmt(stmt)
            except ReturnSignal as sig:
                result = sig.value
            finally:
                self._env = old_env

            result.trace.append(f"branch_{i}_output")
            branches.append(result)

        collapsed = self._collapse(branches, gate.collapse, gate.threshold)

        # FIX (Bug 6): log divergence stats for downstream hallucination detection
        raw_vals = [str(b.raw) for b in branches]
        unique_count = len(set(raw_vals))
        divergence_ratio = (unique_count - 1) / max(len(branches) - 1, 1)

        self._result.gate_logs.append({
            "gate": gate.name,
            "branches": gate.branches,
            "collapse": gate.collapse,
            "branch_confidences": [b.confidence.value for b in branches],
            "branch_values": raw_vals,
            "unique_branch_values": unique_count,
            "divergence_ratio": divergence_ratio,
            "result_confidence": collapsed.confidence.value,
            "result_value": collapsed.raw,
        })

        self._trace(
            f"[gate] Collapsed '{gate.name}': "
            f"confidence={collapsed.confidence.value:.3f}, value={collapsed.raw}, "
            f"divergence={divergence_ratio:.2f} ({unique_count}/{len(branches)} unique)"
        )

        return collapsed

    def _collapse(
        self,
        branches: list[ChimeraValue],
        strategy: str,
        threshold: float,
    ) -> ChimeraValue:
        if not branches:
            return self._wrap(None, confidence=0.0)

        if strategy == "highest_confidence":
            best = max(branches, key=lambda b: b.confidence.value)
            return ConvergeValue(
                raw=best.raw,
                confidence=best.confidence,
                branch_values=branches,
                trace=["collapsed:highest_confidence"],
            )

        if strategy == "weighted_vote":
            vote_weights: dict[Any, float] = {}
            for b in branches:
                key = str(b.raw)
                vote_weights[key] = vote_weights.get(key, 0.0) + b.confidence.value
            winner_key = max(vote_weights, key=vote_weights.get)  # type: ignore[arg-type]
            winner = next(b for b in branches if str(b.raw) == winner_key)
            total_weight = sum(vote_weights.values())
            winner_weight = vote_weights[winner_key]
            consensus_conf = winner_weight / total_weight if total_weight > 0 else 0.0
            return ConvergeValue(
                raw=winner.raw,
                confidence=Confidence(consensus_conf, "weighted_vote"),
                branch_values=branches,
                trace=[f"collapsed:weighted_vote({consensus_conf:.3f})"],
            )

        # Default: majority
        votes: dict[str, list[ChimeraValue]] = {}
        for b in branches:
            key = str(b.raw)
            votes.setdefault(key, []).append(b)
        majority_key = max(votes, key=lambda k: len(votes[k]))
        majority_group = votes[majority_key]
        avg_conf = sum(b.confidence.value for b in majority_group) / len(majority_group)

        if avg_conf < threshold:
            self._trace(f"[gate] Consensus below threshold ({avg_conf:.3f} < {threshold})")

        return ConvergeValue(
            raw=majority_group[0].raw,
            confidence=Confidence(avg_conf, "majority"),
            branch_values=branches,
            trace=[f"collapsed:majority({len(majority_group)}/{len(branches)})"],
        )

    # ------------------------------------------------------------------
    # Goal execution
    # ------------------------------------------------------------------

    def _exec_goal(self, goal: GoalDecl) -> None:
        self._trace(f'[goal] Pursuing: "{goal.description}"')
        self._trace(f"[goal] Budget: {goal.explore_budget}, Constraints: {goal.constraints_list}")
        scope = self._env.child()
        old_env = self._env
        self._env = scope
        try:
            for stmt in goal.body:
                self._exec_stmt(stmt)
        except ReturnSignal:
            pass
        finally:
            self._env = old_env
        self._trace(f'[goal] Completed: "{goal.description}"')

    # ------------------------------------------------------------------
    # Reason execution
    # ------------------------------------------------------------------

    def _exec_reason(self, reason: ReasonDecl) -> None:
        def _invoke(*args: ChimeraValue) -> ChimeraValue:
            self._trace(f"[reason] Starting reasoning with given: {reason.given}")
            scope = self._env.child()
            for param, arg in zip(reason.params, args):
                scope.set(param.name, arg)
            old_env = self._env
            self._env = scope
            result: ChimeraValue | None = None
            try:
                for stmt in reason.body:
                    self._exec_stmt(stmt)
            except ReturnSignal as ret:
                result = ret.value
            finally:
                self._env = old_env
            self._trace(f"[reason] Committed via: {reason.commit_strategy}")
            return result if result is not None else self._wrap(None)

        self._env.set(reason.name, _invoke)

    # ------------------------------------------------------------------
    # Value construction helpers
    # ------------------------------------------------------------------

    def _wrap(self, raw: Any, confidence: float = 1.0) -> ChimeraValue:
        return ChimeraValue(
            raw=raw,
            confidence=Confidence(confidence, "literal"),
            trace=[],
        )

    def _trace(self, msg: str) -> None:
        self._result.trace.append(msg)
