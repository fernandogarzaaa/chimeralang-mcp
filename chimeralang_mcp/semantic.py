"""Optional semantic verification backends for chimera_verify (Phase 5.2).

chimera_verify's default `method="lexical"` (Jaccard token overlap, in
server.py) needs none of this and stays the always-available default. Two
opt-in semantic methods sit behind the same `method` parameter:

  - "nli": a local cross-encoder NLI model. Deterministic given fixed weights.
           Requires the [semantic] extra (sentence-transformers + torch).
  - "llm": an Anthropic model acting as judge. Non-deterministic; carries its
           model_id. Requires the [llm] extra (anthropic) and ANTHROPIC_API_KEY.

Both backends are lazy-loaded, so importing this module — and running the core
server — costs nothing until a caller actually requests a semantic method.

Every backend returns the same shape:
    {"label": "supported"|"contradicted"|"insufficient",
     "method": "nli"|"llm", "model_id": str, "deterministic": bool, ...}
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
NLI_LABELS = ("contradiction", "entailment", "neutral")
NLI_THRESHOLD = 0.5
LLM_MODEL_ID = "claude-haiku-4-5-20251001"

_LLM_JUDGE_SYSTEM = (
    "You are a strict fact-verification judge. Given a CLAIM and EVIDENCE, "
    "classify the claim as exactly one of: supported (the evidence entails the "
    "claim), contradicted (the evidence contradicts the claim), or insufficient "
    "(the evidence neither entails nor contradicts the claim). Judge only from "
    "the supplied evidence, not outside knowledge. Respond with JSON only: "
    '{"verdict": "supported|contradicted|insufficient", "rationale": "<one sentence>"}.'
)


class SemanticUnavailable(RuntimeError):
    """Raised when a requested semantic method's dependencies are missing."""


def available(method: str) -> bool:
    if method == "nli":
        return _nli_available()
    if method == "llm":
        return _llm_available()
    return False


def _nli_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _llm_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def _nli_model():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SemanticUnavailable(
            "method='nli' requires the [semantic] extra: "
            "pip install 'chimeralang-mcp[semantic]'"
        ) from exc
    return CrossEncoder(NLI_MODEL_ID)


def _softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    total = sum(exps)
    return [e / total for e in exps] if total else [0.0 for _ in row]


def _classify_nli(claim: str, evidence_texts: list[str]) -> dict[str, Any]:
    if not evidence_texts:
        return {"label": "insufficient", "method": "nli", "model_id": NLI_MODEL_ID,
                "deterministic": True, "scores": {}}
    model = _nli_model()
    raw = model.predict([(ev, claim) for ev in evidence_texts])  # premise, hypothesis
    best_c = best_e = best_n = 0.0
    for logits in raw:
        c, e, n = _softmax([float(x) for x in logits])
        best_c, best_e, best_n = max(best_c, c), max(best_e, e), max(best_n, n)
    if best_c >= best_e and best_c >= NLI_THRESHOLD:
        label = "contradicted"
    elif best_e >= NLI_THRESHOLD:
        label = "supported"
    else:
        label = "insufficient"
    return {
        "label": label,
        "method": "nli",
        "model_id": NLI_MODEL_ID,
        "deterministic": True,
        "scores": {"contradiction": round(best_c, 4),
                   "entailment": round(best_e, 4),
                   "neutral": round(best_n, 4)},
    }


def _classify_llm(claim: str, evidence_texts: list[str]) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise SemanticUnavailable(
            "method='llm' requires the [llm] extra: "
            "pip install 'chimeralang-mcp[llm]'"
        ) from exc
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SemanticUnavailable("method='llm' requires ANTHROPIC_API_KEY to be set")

    client = anthropic.Anthropic()
    evidence_block = "\n".join(f"- {e}" for e in evidence_texts) or "(no evidence provided)"
    message = client.messages.create(
        model=LLM_MODEL_ID,
        max_tokens=256,
        system=[{"type": "text", "text": _LLM_JUDGE_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user",
                   "content": f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence_block}"}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    verdict, rationale = _parse_llm_verdict(text)
    return {
        "label": verdict,
        "method": "llm",
        "model_id": LLM_MODEL_ID,
        "deterministic": False,
        "rationale": rationale,
    }


def _parse_llm_verdict(text: str) -> tuple[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict in {"supported", "contradicted", "insufficient"}:
                return verdict, str(obj.get("rationale", ""))
        except json.JSONDecodeError:
            pass
    # Defensive fallback if the model didn't return clean JSON.
    low = text.lower()
    for verdict in ("contradicted", "insufficient", "supported"):
        if verdict in low:
            return verdict, text.strip()[:200]
    return "insufficient", text.strip()[:200]


def classify(claim: str, evidence_texts: list[str], method: str) -> dict[str, Any]:
    """Classify one claim against evidence using the requested semantic method."""
    if method == "nli":
        return _classify_nli(claim, evidence_texts)
    if method == "llm":
        return _classify_llm(claim, evidence_texts)
    raise ValueError(f"unknown semantic method: {method!r}")
