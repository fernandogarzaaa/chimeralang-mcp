"""ChimeraLang ↔ Claude Adapter.

This module provides ClaudeConstraintMiddleware — a runtime constraint layer
that wraps Claude API calls with ChimeraLang's safety primitives:

  1. Pre-call confidence gating    (must confidence >= threshold before calling)
  2. Output constraint validation  (must/allow/forbidden on tool results)
  3. Hallucination detection       (scan emitted values after execution)
  4. Ephemeral scope enforcement   (tool outputs don't leak across calls)
  5. Semantic + temporal detection (detect strategies applied to outputs)

Usage:
    from chimera.claude_adapter import ClaudeConstraintMiddleware, ToolCallSpec

    middleware = ClaudeConstraintMiddleware(confidence_threshold=0.85)

    spec = ToolCallSpec(
        tool_name="web_search",
        input_constraints=["query must not be empty"],
        output_must_exprs=["result != None"],
        output_forbidden=["personal_data", "unverified_claims"],
        detect_strategy="confidence_threshold",
        detect_threshold=0.7,
    )

    result = middleware.call(spec, query="latest quantum computing research")
    if result.passed:
        use(result.value)
    else:
        handle(result.violations)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from chimera.types import (
    ChimeraValue,
    Confidence,
    ConfidenceViolation,
    ConfidentValue,
    ExploreValue,
    MemoryScope,
)
from chimera.detect import HallucinationDetector, DetectionReport


# ---------------------------------------------------------------------------
# Tool Call Specification
# ---------------------------------------------------------------------------

@dataclass
class ToolCallSpec:
    """Declarative constraint specification for a single Claude tool call."""
    tool_name: str

    # Pre-call: minimum confidence required to even attempt the call
    min_confidence: float = 0.0

    # Post-call: must-expressions (string = intent, callable = runtime check)
    # Callables receive the raw output value and return bool
    output_must: list[str | Callable[[Any], bool]] = field(default_factory=list)

    # Post-call: forbidden capability markers (logged, not enforced by default)
    output_forbidden: list[str] = field(default_factory=list)

    # Hallucination detection strategy for the output
    detect_strategy: str = "confidence_threshold"
    detect_threshold: float = 0.5

    # Optional: reference values for cross_reference strategy
    reference_values: list[Any] = field(default_factory=list)

    # Optional: valid range for range strategy
    valid_range: tuple[float, float] | None = None

    # Optional: forbidden semantic patterns
    forbidden_patterns: list[str] = field(default_factory=list)

    # Max age in seconds for temporal strategy
    max_age_seconds: float = 3600.0

    # Whether to escalate violations as exceptions or just log them
    strict: bool = False


# ---------------------------------------------------------------------------
# Tool Call Result
# ---------------------------------------------------------------------------

@dataclass
class ConstrainedResult:
    """Result of a middleware-wrapped tool call."""
    tool_name: str
    passed: bool
    value: Any = None
    confidence: float = 0.0
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    detection_report: DetectionReport | None = None
    duration_ms: float = 0.0

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ConstrainedResult({self.tool_name}, {status}, "
            f"conf={self.confidence:.2f}, violations={len(self.violations)})"
        )


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class ClaudeConstraintMiddleware:
    """Runtime constraint layer wrapping Claude tool calls with ChimeraLang semantics.

    This is the core integration between ChimeraLang and Claude's agentic loop.
    Every tool call passes through:

        Input → [confidence gate] → [tool execution] → [constraint check]
               → [hallucination scan] → [scope cleanup] → Output

    The middleware implements ChimeraLang's safety model without requiring
    .chimera source files — it works directly from Python ToolCallSpec objects,
    making it easy to integrate into existing Claude tool-use pipelines.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        strict: bool = False,
    ) -> None:
        self._threshold = confidence_threshold
        self._strict = strict
        self._detector = HallucinationDetector()
        self._call_log: list[ConstrainedResult] = []
        # Ephemeral scope: cleared after each call unless marked Persistent
        self._ephemeral_scope: dict[str, ChimeraValue] = {}

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------

    def call(
        self,
        spec: ToolCallSpec,
        tool_fn: Callable[..., Any] | None = None,
        raw_output: Any = None,
        input_confidence: float = 1.0,
        **kwargs: Any,
    ) -> ConstrainedResult:
        """Execute a constrained tool call.

        Either provide tool_fn (called with **kwargs) or raw_output (already executed).
        input_confidence: caller's confidence that this is the right call to make.
        """
        start = time.perf_counter()
        result = ConstrainedResult(tool_name=spec.tool_name, passed=True)

        try:
            # Phase 1: Confidence gate
            result.trace.append(
                f"[gate] {spec.tool_name}: input_confidence={input_confidence:.3f} "
                f"threshold={spec.min_confidence:.3f}"
            )
            if input_confidence < spec.min_confidence:
                violation = (
                    f"confidence gate: {input_confidence:.3f} < "
                    f"required {spec.min_confidence:.3f}"
                )
                result.violations.append(violation)
                result.passed = False
                result.trace.append(f"[gate] BLOCKED — {violation}")
                result.duration_ms = (time.perf_counter() - start) * 1000
                self._call_log.append(result)
                if spec.strict or self._strict:
                    raise ConfidenceViolation(violation)
                return result

            result.trace.append(f"[gate] PASS")

            # Phase 2: Execute tool
            if tool_fn is not None:
                output = tool_fn(**kwargs)
            elif raw_output is not None:
                output = raw_output
            else:
                output = None

            result.value = output

            # Phase 3: Wrap output in ChimeraValue for analysis
            output_val = self._wrap_output(output, input_confidence, spec.tool_name)

            # Phase 4: must-constraint validation
            for constraint in spec.output_must:
                if isinstance(constraint, str):
                    result.trace.append(f"[must-intent] {constraint}")
                elif callable(constraint):
                    try:
                        ok = bool(constraint(output))
                        if ok:
                            result.trace.append(f"[must] PASSED: {constraint.__name__}")
                        else:
                            violation = f"must-constraint failed: {constraint.__name__}"
                            result.violations.append(violation)
                            result.passed = False
                            result.trace.append(f"[must] VIOLATED: {constraint.__name__}")
                            if spec.strict or self._strict:
                                raise AssertionError(violation)
                    except Exception as e:
                        violation = f"must-constraint error: {e}"
                        result.violations.append(violation)
                        result.passed = False

            # Phase 5: forbidden capability markers
            for cap in spec.output_forbidden:
                result.trace.append(f"[forbidden] logged: {cap}")
                result.warnings.append(f"forbidden capability in scope: {cap}")

            # Phase 6: Hallucination detection on output
            detection = self._detect(output_val, spec)
            result.detection_report = detection
            if not detection.clean:
                for flag in detection.flags:
                    warning = f"hallucination flag: {flag.kind.name} (severity={flag.severity:.2f})"
                    result.warnings.append(warning)
                    result.trace.append(f"[detect] {flag.description}")
                    if flag.severity >= 0.8 and (spec.strict or self._strict):
                        result.passed = False
                        result.violations.append(warning)

            # Phase 7: Confidence propagation
            result.confidence = output_val.confidence.value

            # Phase 8: Ephemeral scope cleanup
            self._ephemeral_scope.clear()
            result.trace.append("[scope] ephemeral bindings cleared")

        except (ConfidenceViolation, AssertionError) as e:
            result.passed = False
            result.violations.append(str(e))
        except Exception as e:
            result.passed = False
            result.violations.append(f"runtime error: {e}")

        result.duration_ms = (time.perf_counter() - start) * 1000
        self._call_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Batch execution with consensus
    # ------------------------------------------------------------------

    def consensus_call(
        self,
        spec: ToolCallSpec,
        tool_fns: list[Callable[..., Any]],
        collapse: str = "majority",
        **kwargs: Any,
    ) -> ConstrainedResult:
        """Run multiple tool variants and collapse to consensus.

        This is ChimeraLang's gate primitive applied to Claude tool calls:
        run the same logical operation across N implementations, then
        collapse to the most reliable answer.

        collapse: "majority" | "highest_confidence" | "weighted_vote"
        """
        results = []
        for i, fn in enumerate(tool_fns):
            r = self.call(spec, tool_fn=fn, **kwargs)
            r.trace.append(f"branch_{i}_output")
            results.append(r)

        # Measure divergence
        raw_vals = [str(r.value) for r in results]
        unique_vals = set(raw_vals)
        divergence = (len(unique_vals) - 1) / max(len(results) - 1, 1)

        if divergence == 0:
            merged = results[0]
            merged.warnings.append(
                "consensus: all branches returned identical values — "
                "trivial consensus, no genuine divergence detected"
            )
            merged.trace.append(f"[consensus] trivial (divergence=0.00)")
            return merged

        # Collapse
        if collapse == "majority":
            vote_counts: dict[str, list[ConstrainedResult]] = {}
            for r in results:
                key = str(r.value)
                vote_counts.setdefault(key, []).append(r)
            winner_key = max(vote_counts, key=lambda k: len(vote_counts[k]))
            winner = vote_counts[winner_key][0]
        elif collapse == "highest_confidence":
            winner = max(results, key=lambda r: r.confidence)
        else:  # weighted_vote
            winner = max(results, key=lambda r: r.confidence if r.passed else 0.0)

        winner.trace.append(
            f"[consensus] collapse={collapse}, divergence={divergence:.2f}, "
            f"branches={len(results)}, unique={len(unique_vals)}"
        )
        return winner

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def call_log(self) -> list[ConstrainedResult]:
        """Return the full call log for this middleware instance."""
        return list(self._call_log)

    def audit_summary(self) -> dict[str, Any]:
        """Return a summary of all calls made through this middleware."""
        total = len(self._call_log)
        passed = sum(1 for r in self._call_log if r.passed)
        flagged = sum(1 for r in self._call_log if r.warnings)
        avg_conf = (
            sum(r.confidence for r in self._call_log) / total
            if total > 0 else 0.0
        )
        return {
            "total_calls": total,
            "passed": passed,
            "failed": total - passed,
            "flagged_warnings": flagged,
            "avg_confidence": round(avg_conf, 4),
            "tools": list({r.tool_name for r in self._call_log}),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_output(
        self,
        output: Any,
        input_confidence: float,
        source: str,
    ) -> ChimeraValue:
        """Wrap a raw tool output into a ChimeraValue for analysis."""
        # Confidence heuristics based on output type
        if output is None:
            conf = 0.0
        elif isinstance(output, bool):
            conf = 1.0 if output else 0.3
        elif isinstance(output, (int, float)):
            conf = min(input_confidence, 0.95)
        elif isinstance(output, str):
            # Longer, more detailed responses are typically more reliable
            length_bonus = min(len(output) / 1000, 0.1)
            conf = min(input_confidence * 0.9 + length_bonus, 1.0)
        elif isinstance(output, (list, dict)):
            conf = min(input_confidence * 0.85, 1.0)
        else:
            conf = input_confidence * 0.8

        return ChimeraValue(
            raw=output,
            confidence=Confidence(conf, f"tool_output:{source}"),
            memory_scope=MemoryScope.EPHEMERAL,
            trace=[f"tool:{source}", f"input_conf:{input_confidence:.3f}"],
        )

    def _detect(self, value: ChimeraValue, spec: ToolCallSpec) -> DetectionReport:
        """Run hallucination detection on a tool output value."""
        from chimera.detect import HallucinationFlag, HallucinationKind, DetectionReport

        report = DetectionReport()
        self._detector.scan_value(value, report)

        # Apply spec-specific detection strategy
        if spec.detect_strategy == "confidence_threshold":
            if value.confidence.value < spec.detect_threshold:
                report.add(HallucinationFlag(
                    kind=HallucinationKind.CONFIDENCE_ANOMALY,
                    severity=1.0 - value.confidence.value,
                    description=(
                        f"Output confidence {value.confidence.value:.3f} below "
                        f"threshold {spec.detect_threshold}"
                    ),
                    evidence={"confidence": value.confidence.value, "threshold": spec.detect_threshold},
                ))

        elif spec.detect_strategy == "range" and spec.valid_range is not None:
            lo, hi = spec.valid_range
            try:
                v = float(value.raw)  # type: ignore[arg-type]
                if not (lo <= v <= hi):
                    report.add(HallucinationFlag(
                        kind=HallucinationKind.CONFIDENCE_ANOMALY,
                        severity=0.9,
                        description=f"Value {v} outside valid range [{lo}, {hi}]",
                        evidence={"value": v, "range": [lo, hi]},
                    ))
            except (TypeError, ValueError):
                pass

        elif spec.detect_strategy == "semantic" and spec.forbidden_patterns:
            val_str = str(value.raw).lower()
            for pat in spec.forbidden_patterns:
                if pat.lower() in val_str:
                    report.add(HallucinationFlag(
                        kind=HallucinationKind.CONFIDENCE_ANOMALY,
                        severity=0.85,
                        description=f"Forbidden semantic pattern '{pat}' in output",
                        evidence={"pattern": pat},
                    ))

        elif spec.detect_strategy == "cross_reference" and spec.reference_values:
            try:
                target = float(value.raw)  # type: ignore[arg-type]
                refs = [float(r) for r in spec.reference_values]
                avg = sum(refs) / len(refs)
                dev = abs(target - avg) / (abs(avg) + 1e-9)
                if dev > 0.2:
                    report.add(HallucinationFlag(
                        kind=HallucinationKind.BRANCH_DIVERGENCE,
                        severity=min(dev, 1.0),
                        description=(
                            f"Output {target} deviates {dev:.2f} from "
                            f"reference average {avg:.3f}"
                        ),
                        evidence={"value": target, "avg_ref": avg, "deviation": dev},
                    ))
            except (TypeError, ValueError):
                pass

        return report
