"""token_engine.py — ChimeraLang Token Budget Manager + Message Importance Scorer.

Internal module (not exposed as MCP tools directly). Used by server.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TokenBudgetManager — Anthropic API counting with SHA-256 cache
# ---------------------------------------------------------------------------

class TokenBudgetManager:
    """Singleton token counter using Anthropic count_tokens API.

    Thread-safe. Falls back to len//4 when ANTHROPIC_API_KEY is absent.
    Results are cached per content hash to avoid double-counting.
    """

    _instance: TokenBudgetManager | None = None

    def __new__(cls) -> TokenBudgetManager:
        # Singleton
        if cls._instance is None:
            obj = object.__new__(cls)
            obj._initialized = False
            cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None
        self._cache_max_entries = self._read_cache_size()
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._fallback_count = 0
        self._last_fallback_reason = ""
        self._last_fallback_at = 0.0
        self._last_fallback_log_at = 0.0
        self._fallback_log_interval_s = self._read_fallback_log_interval()
        self._token_count_method = "estimate"
        if self._api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
                self._token_count_method = "api"
                log.info("TokenBudgetManager: Anthropic API token counting enabled")
            except ImportError:
                log.warning("TokenBudgetManager: anthropic package not found, using len//4 fallback")
        else:
            log.info("TokenBudgetManager: ANTHROPIC_API_KEY not set, using len//4 fallback")

    # ------------------------------------------------------------------
    # Public counting API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens for a single text string."""
        if not text:
            return 0
        h = self._hash(text)
        cached = self._cache_get(h)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        if self._client and self._token_count_method == "api":
            try:
                result = self._client.beta.messages.count_tokens(
                    model="claude-opus-4-7",
                    messages=[{"role": "user", "content": text}],
                )
                count = result.usage.input_tokens
            except Exception as e:
                count = len(text) // 4
                self._record_fallback("count_tokens_api_error", e)
        else:
            count = len(text) // 4
        self._cache_put(h, count)
        return count

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a full message list in one API call."""
        if not messages:
            return 0
        # Build cache key from all content
        content_hash = self._hash(str(messages))
        cached = self._cache_get(content_hash)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        if self._client and self._token_count_method == "api":
            try:
                # Convert to Anthropic message format
                anthropic_msgs = []
                for m in messages:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        # Handle multimodal content
                        text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
                        content = " ".join(text_parts)
                    anthropic_msgs.append({"role": role, "content": str(content)})
                result = self._client.beta.messages.count_tokens(
                    model="claude-opus-4-7",
                    messages=anthropic_msgs,
                )
                count = result.usage.input_tokens
            except Exception as e:
                count = sum(self._estimate_message_tokens(m) for m in messages)
                self._record_fallback("count_messages_api_error", e)
        else:
            count = sum(self._estimate_message_tokens(m) for m in messages)
        self._cache_put(content_hash, count)
        return count

    def count_texts(self, texts: list[str]) -> int:
        """Count tokens for multiple text strings."""
        return sum(self.count_tokens(t) for t in texts)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "token_count_method": self._token_count_method,
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_max_entries": self._cache_max_entries,
            "fallback_count": self._fallback_count,
            "last_fallback_reason": self._last_fallback_reason,
            "last_fallback_at": self._last_fallback_at,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> int | None:
        value = self._cache.get(key)
        if value is None:
            return None
        self._cache.move_to_end(key)
        return value

    def _cache_put(self, key: str, value: int) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max_entries:
            self._cache.popitem(last=False)

    def _record_fallback(self, reason: str, exc: Exception | None = None) -> None:
        self._token_count_method = "estimate"
        self._fallback_count += 1
        self._last_fallback_reason = reason
        self._last_fallback_at = time.time()
        now = self._last_fallback_at
        should_log = (now - self._last_fallback_log_at) >= self._fallback_log_interval_s
        if self._fallback_count == 1 or should_log:
            self._last_fallback_log_at = now
            if exc:
                log.warning("TokenBudgetManager fallback to len//4 (%s): %s", reason, exc)
            else:
                log.warning("TokenBudgetManager fallback to len//4 (%s)", reason)

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = [str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text"]
            return len(" ".join(text_parts)) // 4
        return len(str(content or "")) // 4

    @staticmethod
    def _read_cache_size() -> int:
        raw = os.environ.get("CHIMERA_TOKEN_CACHE_MAX_ENTRIES", "").strip()
        if not raw:
            return 2048
        try:
            parsed = int(raw)
            return max(parsed, 1)
        except ValueError:
            log.warning("Invalid CHIMERA_TOKEN_CACHE_MAX_ENTRIES=%r; using default 2048", raw)
            return 2048

    @staticmethod
    def _read_fallback_log_interval() -> float:
        raw = os.environ.get("CHIMERA_TOKEN_FALLBACK_LOG_INTERVAL_S", "").strip()
        if not raw:
            return 60.0
        try:
            parsed = float(raw)
            return max(parsed, 0.0)
        except ValueError:
            log.warning("Invalid CHIMERA_TOKEN_FALLBACK_LOG_INTERVAL_S=%r; using default 60.0", raw)
            return 60.0


# ---------------------------------------------------------------------------
# MessageImportanceScorer — 4-axis importance ranking for lossy compression
# ---------------------------------------------------------------------------

class MessageImportanceScorer:
    """Score each message by importance for lossy compression decisions.

    Thread-safe, stateless. Called by chimera_compress and chimera_fracture
    when allow_lossy=True.

    Axes:
      - recency:      position from end (recent = higher score)
      - content_type: tool_result > code fence > user question > assistant prose
      - info_density: tokens per unique word (higher = better)
      - replaceability: irreplaceable content scores higher
    """

    # Content-type weights (higher = more important to keep)
    _TYPE_WEIGHTS: dict[str, float] = {
        "tool_result":  1.0,
        "code_fence":    0.95,
        "user_question": 0.85,
        "assistant_code": 0.80,
        "assistant_prose": 0.40,
        "system":        0.20,
        "preamble":      0.10,
    }

    def score(self, message: dict[str, Any], index: int, total: int) -> float:
        """Return importance score 0.0–1.0 for a single message."""
        recency_score = self._recency(index, total)
        type_score = self._content_type_score(message)
        density_score = self._info_density(message)
        replace_score = self._replaceability(message)
        # Weighted average — recency and type dominate
        score = (
            recency_score * 0.30
            + type_score * 0.35
            + density_score * 0.20
            + replace_score * 0.15
        )
        return round(score, 4)

    def rank(
        self,
        messages: list[dict[str, Any]],
        min_messages: int = 2,
    ) -> list[dict[str, Any]]:
        """Return messages sorted ascending by importance score (lowest first).

        Used by lossy compression to determine what to drop first.
        """
        total = len(messages)
        scored: list[dict[str, Any]] = []
        for i, msg in enumerate(messages):
            score = self.score(msg, i, total)
            reason = self._dominant_reason(msg, i, total, score)
            scored.append({
                "index": i,
                "role": msg.get("role", "unknown"),
                "score": score,
                "reason": reason,
            })
        scored.sort(key=lambda x: (x["score"], x["index"]))
        return scored

    # ------------------------------------------------------------------
    # Private scoring axes
    # ------------------------------------------------------------------

    def _recency(self, index: int, total: int) -> float:
        """Score based on position — newer (higher index) = higher score."""
        if total <= 1:
            return 1.0
        return index / (total - 1)

    def _content_type_score(self, message: dict[str, Any]) -> float:
        content = str(message.get("content", ""))
        role = message.get("role", "")
        # Detect code fences
        if "```" in content or content.startswith("```"):
            return self._TYPE_WEIGHTS["code_fence"]
        # Detect tool results
        if role == "tool" or content.startswith("Tool use"):
            return self._TYPE_WEIGHTS["tool_result"]
        # Detect user questions
        if role == "user" and "?" in content:
            return self._TYPE_WEIGHTS["user_question"]
        # Detect assistant code
        if role == "assistant" and any(kw in content for kw in ["def ", "class ", "import ", "fn ", "pub ", "func "]):
            return self._TYPE_WEIGHTS["assistant_code"]
        # Detect verbose prose (long assistant messages with no code)
        if role == "assistant" and len(content) > 300:
            return self._TYPE_WEIGHTS["assistant_prose"]
        # Detect preamble / greeting
        if role == "assistant" and any(p in content.lower() for p in ["sure,", "of course", "happy to", "i'd be happy"]):
            return self._TYPE_WEIGHTS["preamble"]
        # Default
        if role == "system":
            return self._TYPE_WEIGHTS["system"]
        return self._TYPE_WEIGHTS["assistant_prose"]

    def _info_density(self, message: dict[str, Any]) -> float:
        """Tokens per unique word — higher density = more information."""
        content = str(message.get("content", ""))
        words = content.lower().split()
        if not words:
            return 0.0
        unique_words = set(words)
        if len(unique_words) == 0:
            return 0.0
        # Ratio of unique words to total words
        unique_ratio = len(unique_words) / len(words)
        # Heuristic: dense text has many unique words relative to length
        token_estimate = len(content) // 4
        if token_estimate == 0:
            return 0.0
        density = unique_ratio * min(token_estimate / 50.0, 1.0)
        return min(density, 1.0)

    def _replaceability(self, message: dict[str, Any]) -> float:
        """How hard is it to recover this content from context?"""
        role = message.get("role", "")
        content = str(message.get("content", ""))
        if role == "user":
            # User questions are irreplaceable
            return 1.0
        if role == "tool" or "```" in content:
            # Tool results and code are somewhat irreplaceable
            return 0.8
        if role == "assistant" and len(content) < 100:
            # Short assistant responses are easily regenerated
            return 0.3
        return 0.5

    def _dominant_reason(
        self,
        message: dict[str, Any],
        index: int,
        total: int,
        score: float,
    ) -> str:
        """Return the single most impactful reason for the score."""
        type_scores = {
            "tool_result": self._content_type_score(message),
            "code_fence": self._content_type_score(message),
            "user_question": self._content_type_score(message),
            "assistant_prose": self._content_type_score(message),
            "preamble": self._content_type_score(message),
            "system": self._content_type_score(message),
        }
        top_type = max(type_scores, key=type_scores.get)
        if type_scores[top_type] >= 0.95:
            return top_type
        if self._recency(index, total) < 0.3:
            return "old_message"
        if self._replaceability(message) < 0.4:
            return "replaceable"
        return top_type


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

def get_token_budget_manager() -> TokenBudgetManager:
    """Lazy singleton for TokenBudgetManager."""
    return TokenBudgetManager()
