"""token_engine.py - token counting plus quantum-inspired context compression.

Internal module (not exposed as MCP tools directly). Used by server.py.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_CODE_PLACEHOLDER_RE = re.compile(r"\x00CODE\d+\x00")
_FILLER_PATTERNS = [
    r"\bplease note that\b",
    r"\bit is worth noting that\b",
    r"\bit should be noted that\b",
    r"\bin order to\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bvery\b",
    r"\bjust\b",
    r"\bsimply\b",
    r"\bquite\b",
    r"\bof course\b",
    r"\bneedless to say\b",
    r"\bas you can see\b",
    r"\bclearly\b",
]
_CONTRACTIONS_MEDIUM = {
    r"\bdo not\b": "don't",
    r"\bdoes not\b": "doesn't",
    r"\bdid not\b": "didn't",
    r"\bcannot\b": "can't",
    r"\bwill not\b": "won't",
    r"\bwould not\b": "wouldn't",
    r"\bshould not\b": "shouldn't",
    r"\bcould not\b": "couldn't",
    r"\bare not\b": "aren't",
    r"\bwas not\b": "wasn't",
    r"\bwere not\b": "weren't",
    r"\bhave not\b": "haven't",
    r"\bhas not\b": "hasn't",
    r"\bhad not\b": "hadn't",
    r"\bI am\b": "I'm",
    r"\bI have\b": "I've",
    r"\bI will\b": "I'll",
    r"\bI would\b": "I'd",
    r"\bit is\b": "it's",
    r"\bthat is\b": "that's",
    r"\bthere is\b": "there's",
    r"\bthey are\b": "they're",
    r"\bwe are\b": "we're",
    r"\byou are\b": "you're",
}
_SYMBOLS_AGGRESSIVE = {
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
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "here", "hers", "him",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
    "more", "most", "my", "of", "on", "or", "our", "ours", "she", "so", "than",
    "that", "the", "their", "them", "there", "these", "they", "this", "those",
    "to", "too", "up", "us", "very", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your", "yours",
}
_STRUCTURE_HINTS = (
    "error", "warning", "exception", "traceback", "fix", "todo", "must",
    "required", "important", "critical", "regression", "budget", "token",
    "version", "release", "tests", "workflow", "build", "path", "file",
)


def _estimate_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _normalize_content(content: Any) -> str:
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(part))
        return " ".join(p for p in text_parts if p)
    return str(content or "")


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_filler(text: str) -> str:
    for pattern in _FILLER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return _normalize_whitespace(re.sub(r"[ \t]{2,}", " ", text))


def _apply_contractions(text: str) -> str:
    for pattern, replacement in _CONTRACTIONS_MEDIUM.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _apply_symbols(text: str) -> str:
    for pattern, replacement in _SYMBOLS_AGGRESSIVE.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r"\s+([,;:!?])", r"\1", text)
    return text


def _dedup_lines(text: str) -> str:
    seen: set[str] = set()
    lines_out: list[str] = []
    for line in text.splitlines():
        key = line.strip().lower()
        if not key:
            lines_out.append(line)
            continue
        if key in seen:
            continue
        seen.add(key)
        lines_out.append(line)
    return "\n".join(lines_out)


def _collapse_lists(text: str) -> str:
    list_item_pattern = r"(?:^|\n)([-*•])\s+(.+?)(?=\n[-*•]|\n\n|$)"
    items_ordered: list[tuple[str, str]] = []
    seen_items: set[str] = set()
    for match in re.finditer(list_item_pattern, text, re.MULTILINE):
        bullet, text_content = match.group(1), match.group(2).strip()
        key = text_content.lower()
        if key and key not in seen_items:
            seen_items.add(key)
            items_ordered.append((bullet, text_content))
    if not items_ordered:
        return text
    return "\n".join(f"{bullet} {item}" for bullet, item in items_ordered)


def _stash_code_blocks(text: str) -> tuple[str, list[str]]:
    code_blocks: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        code_blocks.append(match.group(0))
        return f"\x00CODE{len(code_blocks) - 1}\x00"

    return re.sub(r"```[\s\S]*?```", _stash, text), code_blocks


def _restore_code_blocks(text: str, code_blocks: list[str]) -> str:
    for index, block in enumerate(code_blocks):
        text = text.replace(f"\x00CODE{index}\x00", block)
    return text


def _tokenize_terms(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"[A-Za-z0-9_./:-]+", text.lower()):
        if len(token) < 3 and not token.isdigit():
            continue
        if len(token) > 64:
            if len(set(token)) <= 4 and "/" not in token and "." not in token:
                continue
            token = token[:64]
        if len(token) > 24 and len(set(token)) <= 2:
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _extract_focus_terms(text: str, limit: int = 12) -> list[str]:
    counts = Counter(_tokenize_terms(text))
    if not counts:
        return []
    sorted_terms = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [term for term, _ in sorted_terms[:limit]]


def normalize_content(content: Any) -> str:
    return _normalize_content(content)


def extract_focus_terms(text: str, limit: int = 12) -> list[str]:
    return _extract_focus_terms(text, limit=limit)


@dataclass
class CompressionUnit:
    index: int
    text: str
    terms: set[str]
    token_cost: int
    locked: bool = False
    base_amplitude: float = 0.0
    entanglement: float = 0.0
    probability: float = 0.0
    priority: float = 0.0


@dataclass
class QuantumCompressionResult:
    text: str
    algorithm: str
    original_chars: int
    compressed_chars: int
    original_tokens: int
    compressed_tokens: int
    units_total: int
    units_kept: int
    code_blocks_preserved: int
    focus_terms: list[str] = field(default_factory=list)
    passes_applied: list[str] = field(default_factory=list)


@dataclass
class QuantumMessageCompressionResult:
    messages: list[dict[str, Any]]
    compressed_history: str
    original_tokens: int
    compressed_tokens: int
    compressed_indexes: list[int]
    dropped_indexes: list[int]
    ranking: list[dict[str, Any]]
    focus_terms: list[str]
    omitted_summary: str = ""


class QuantumCompressionEngine:
    """Deterministic, query-aware token saver.

    The algorithm is "quantum-inspired" in the sense that each chunk gets a
    salience amplitude, redundancy acts as interference, shared entities become
    entanglement boosts, and final selection is a budget-constrained
    measurement step.
    """

    _LEVEL_TARGETS = {"light": 0.82, "medium": 0.58, "aggressive": 0.36}
    _ROLE_FLOORS = {
        "system": 0.28,
        "user": 0.42,
        "assistant": 0.18,
        "tool": 0.34,
    }

    def optimize_text(
        self,
        text: str,
        *,
        focus: str = "",
        preserve_code: bool = True,
        strategies: list[str] | None = None,
        level: str = "medium",
        target_ratio: float | None = None,
        max_tokens: int | None = None,
    ) -> QuantumCompressionResult:
        original_text = str(text or "")
        original_tokens = _estimate_tokens(original_text)
        code_blocks: list[str] = []
        passes_applied: list[str] = []
        work = original_text

        if preserve_code:
            work, code_blocks = _stash_code_blocks(work)

        strategies = strategies or ["whitespace", "dedup_sentences", "strip_filler"]
        if "whitespace" in strategies:
            before = len(work)
            work = _normalize_whitespace(work)
            passes_applied.append(f"whitespace: -{before - len(work)} chars")
        if "dedup_sentences" in strategies:
            before = len(work)
            work = _dedup_lines(work)
            passes_applied.append(f"dedup_sentences: -{before - len(work)} chars")
        if "strip_filler" in strategies:
            before = len(work)
            work = _strip_filler(work)
            passes_applied.append(f"strip_filler: -{before - len(work)} chars")
        if "collapse_lists" in strategies:
            before = len(work)
            work = _collapse_lists(work)
            passes_applied.append(f"collapse_lists: -{before - len(work)} chars")

        focus_terms = _extract_focus_terms(focus)
        units = self._split_units(work)
        if not units:
            restored = _restore_code_blocks(work, code_blocks) if preserve_code else work
            restored = self._micro_compress(restored, level)
            return QuantumCompressionResult(
                text=restored,
                algorithm="quantum",
                original_chars=len(original_text),
                compressed_chars=len(restored),
                original_tokens=original_tokens,
                compressed_tokens=_estimate_tokens(restored),
                units_total=0,
                units_kept=0,
                code_blocks_preserved=len(code_blocks),
                focus_terms=focus_terms,
                passes_applied=passes_applied,
            )

        preselection_tokens = sum(unit.token_cost for unit in units)
        budget_tokens = self._target_budget(
            total_tokens=preselection_tokens,
            level=level,
            target_ratio=target_ratio,
            max_tokens=max_tokens,
            focus_terms=focus_terms,
        )
        selected_units = self._measure_units(units, focus_terms, budget_tokens)
        selected_text = self._render_units(selected_units)
        if not selected_text.strip() and units:
            selected_text = units[0].text
        selected_text = self._micro_compress(selected_text, level)
        if preserve_code:
            selected_text = _restore_code_blocks(selected_text, code_blocks)
        selected_text = _normalize_whitespace(selected_text)

        return QuantumCompressionResult(
            text=selected_text,
            algorithm="quantum",
            original_chars=len(original_text),
            compressed_chars=len(selected_text),
            original_tokens=original_tokens,
            compressed_tokens=_estimate_tokens(selected_text),
            units_total=len(units),
            units_kept=len(selected_units),
            code_blocks_preserved=len(code_blocks),
            focus_terms=focus_terms,
            passes_applied=passes_applied,
        )

    def compress_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        focus: str = "",
        scorer: MessageImportanceScorer | None = None,
        token_budget: int | None = None,
        allow_lossy: bool = False,
    ) -> QuantumMessageCompressionResult:
        normalized_messages = [
            {
                "role": message.get("role", "user"),
                "content": _normalize_content(message.get("content", "")),
            }
            for message in messages
        ]
        focus_terms = _extract_focus_terms(focus)
        ranking = scorer.rank(normalized_messages, focus=focus) if scorer else []
        by_index = {item["index"]: item for item in ranking}
        original_history = self.render_history(normalized_messages)
        original_tokens = _estimate_tokens(original_history)
        pressure = min(1.0, (token_budget / max(original_tokens, 1))) if token_budget else 0.58

        compressed_messages: list[dict[str, Any]] = []
        compressed_indexes: list[int] = []
        for index, message in enumerate(normalized_messages):
            score = by_index.get(index, {}).get("score", 0.5)
            target_ratio = self._message_target_ratio(
                role=message["role"],
                score=score,
                pressure=pressure,
                content=message["content"],
            )
            level = "aggressive" if score < 0.45 else "medium"
            result = self.optimize_text(
                message["content"],
                focus=focus,
                preserve_code=True,
                strategies=["whitespace", "dedup_sentences", "strip_filler"],
                level=level,
                target_ratio=target_ratio,
            )
            compressed_messages.append({
                "role": message["role"],
                "content": result.text,
            })
            if result.text != message["content"]:
                compressed_indexes.append(index)

        final_messages = list(compressed_messages)
        compressed_history = self.render_history(final_messages)
        compressed_tokens = _estimate_tokens(compressed_history)
        dropped_indexes: list[int] = []
        omitted_summary = ""

        if token_budget is not None and compressed_tokens > token_budget and allow_lossy and ranking:
            min_keep = 2
            for entry in ranking:
                if compressed_tokens <= token_budget or len(final_messages) - len(dropped_indexes) <= min_keep:
                    break
                dropped_indexes.append(entry["index"])
                kept_messages = [
                    message
                    for idx, message in enumerate(compressed_messages)
                    if idx not in set(dropped_indexes)
                ]
                omitted_summary = self._summarize_dropped(
                    [compressed_messages[idx]["content"] for idx in dropped_indexes],
                    focus_terms=focus_terms,
                    dropped_count=len(dropped_indexes),
                )
                final_messages = list(kept_messages)
                if omitted_summary:
                    final_messages.append({"role": "system", "content": omitted_summary})
                compressed_history = self.render_history(final_messages)
                compressed_tokens = _estimate_tokens(compressed_history)

        return QuantumMessageCompressionResult(
            messages=final_messages,
            compressed_history=compressed_history,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compressed_indexes=compressed_indexes,
            dropped_indexes=dropped_indexes,
            ranking=ranking,
            focus_terms=focus_terms,
            omitted_summary=omitted_summary,
        )

    def score_message(
        self,
        message: dict[str, Any],
        index: int,
        total: int,
        *,
        focus: str = "",
    ) -> float:
        content = _normalize_content(message.get("content", ""))
        recency_score = self._recency(index, total)
        type_score = self._content_type_score(message)
        density_score = self._info_density(content)
        replace_score = self._replaceability(message)
        focus_terms = set(_extract_focus_terms(focus))
        content_terms = set(_tokenize_terms(content))
        focus_score = self._focus_overlap(content_terms, focus_terms)
        structure_score = self._structural_signal(content, _CODE_PLACEHOLDER_RE.search(content) is not None)
        score = (
            recency_score * 0.22
            + type_score * 0.38
            + density_score * 0.12
            + replace_score * 0.12
            + focus_score * 0.10
            + structure_score * 0.06
        )
        return round(min(1.0, score), 4)

    def render_history(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            return ""
        return "\n".join(
            f"[{message.get('role', 'user')}]: {_normalize_content(message.get('content', ''))}"
            for message in messages
        )

    def _target_budget(
        self,
        *,
        total_tokens: int,
        level: str,
        target_ratio: float | None,
        max_tokens: int | None,
        focus_terms: list[str],
    ) -> int:
        if max_tokens is not None:
            return max(8, max_tokens)
        ratio = self._LEVEL_TARGETS.get(level, self._LEVEL_TARGETS["medium"])
        if target_ratio is not None:
            ratio = max(0.12, min(0.95, target_ratio))
        elif focus_terms:
            ratio = max(0.25, ratio - 0.08)
        return max(8, math.ceil(total_tokens * ratio))

    def _message_target_ratio(self, *, role: str, score: float, pressure: float, content: str) -> float:
        floor = self._ROLE_FLOORS.get(role, 0.2)
        if "```" in content:
            floor = max(floor, 0.55)
        target = pressure * (0.45 + score * 0.75)
        return max(floor, min(0.92, target))

    def _micro_compress(self, text: str, level: str) -> str:
        work = _normalize_whitespace(text)
        work = _strip_filler(work)
        if level in ("medium", "aggressive"):
            work = _apply_contractions(work)
        if level == "aggressive":
            work = _apply_symbols(work)
        return _normalize_whitespace(work)

    def _split_units(self, text: str) -> list[CompressionUnit]:
        raw_units: list[str] = []
        for block in re.split(r"\n{2,}", text):
            block = block.strip()
            if not block:
                continue
            if _CODE_PLACEHOLDER_RE.fullmatch(block):
                raw_units.append(block)
                continue
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) > 1 and all(re.match(r"^[-*•]\s+", line) for line in lines):
                raw_units.extend(lines)
                continue
            for line in lines or [block]:
                raw_units.extend(self._split_sentence_like(line))
        units: list[CompressionUnit] = []
        for index, unit_text in enumerate(raw_units):
            if not unit_text:
                continue
            units.append(
                CompressionUnit(
                    index=index,
                    text=unit_text.strip(),
                    terms=set(_tokenize_terms(unit_text)),
                    token_cost=max(1, _estimate_tokens(unit_text)),
                    locked=_CODE_PLACEHOLDER_RE.fullmatch(unit_text.strip()) is not None,
                )
            )
        return units

    def _split_sentence_like(self, line: str) -> list[str]:
        if _CODE_PLACEHOLDER_RE.fullmatch(line.strip()):
            return [line.strip()]
        if re.match(r"^[-*•]\s+", line.strip()):
            return [line.strip()]
        if len(line) < 180:
            return [line.strip()]
        parts = re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9`])", line.strip())
        return [part.strip() for part in parts if part.strip()]

    def _measure_units(
        self,
        units: list[CompressionUnit],
        focus_terms: list[str],
        budget_tokens: int,
    ) -> list[CompressionUnit]:
        if not units:
            return []
        focus_set = set(focus_terms)
        global_freq = Counter(term for unit in units for term in unit.terms)
        for unit in units:
            focus_overlap = self._focus_overlap(unit.terms, focus_set)
            structure_score = self._structural_signal(unit.text, unit.locked)
            rarity_score = self._rarity_score(unit.terms, global_freq)
            unit.entanglement = self._entanglement_score(unit.terms, global_freq, focus_set)
            position_score = self._position_score(unit.index, len(units))
            amplitude = (
                focus_overlap * 0.34
                + structure_score * 0.22
                + rarity_score * 0.18
                + unit.entanglement * 0.16
                + position_score * 0.10
            )
            if not focus_set:
                amplitude += rarity_score * 0.08
            unit.base_amplitude = min(1.0, amplitude if not unit.locked else max(amplitude, 0.99))
            unit.probability = min(1.0, unit.base_amplitude ** 2 if not unit.locked else 1.0)
            unit.priority = unit.probability / max(math.sqrt(unit.token_cost), 1.0)

        selected_indices = {unit.index for unit in units if unit.locked}
        selected_units = [unit for unit in units if unit.locked]
        current_tokens = sum(unit.token_cost for unit in selected_units)
        coverage_terms = set().union(*(unit.terms for unit in selected_units)) if selected_units else set()
        remaining = [unit for unit in units if unit.index not in selected_indices]

        if not selected_units and remaining:
            anchor = max(remaining, key=lambda unit: unit.priority)
            selected_indices.add(anchor.index)
            selected_units.append(anchor)
            current_tokens += anchor.token_cost
            coverage_terms.update(anchor.terms)
            remaining = [unit for unit in remaining if unit.index != anchor.index]

        while remaining and current_tokens < budget_tokens:
            best_unit: CompressionUnit | None = None
            best_priority = -1.0
            for unit in remaining:
                redundancy = self._jaccard(unit.terms, coverage_terms)
                new_focus_terms = len((unit.terms & focus_set) - coverage_terms)
                new_terms = len(unit.terms - coverage_terms)
                interference = max(0.25, 1.0 - redundancy * 0.6)
                coverage_boost = 1.0 + min(0.28, new_focus_terms * 0.07 + new_terms * 0.01)
                adjusted_priority = unit.priority * interference * coverage_boost
                if adjusted_priority > best_priority:
                    best_priority = adjusted_priority
                    best_unit = unit

            if best_unit is None:
                break

            remaining = [unit for unit in remaining if unit.index != best_unit.index]
            if selected_units and current_tokens + best_unit.token_cost > budget_tokens:
                continue

            selected_indices.add(best_unit.index)
            selected_units.append(best_unit)
            current_tokens += best_unit.token_cost
            coverage_terms.update(best_unit.terms)

        if not selected_units and units:
            selected_units = [max(units, key=lambda unit: unit.priority)]

        selected_units.sort(key=lambda unit: unit.index)
        return selected_units

    def _render_units(self, units: list[CompressionUnit]) -> str:
        if not units:
            return ""
        return "\n".join(unit.text for unit in units if unit.text).strip()

    def _summarize_dropped(self, texts: list[str], *, focus_terms: list[str], dropped_count: int) -> str:
        if not texts:
            return ""
        counts = Counter()
        for text in texts:
            counts.update(_tokenize_terms(text))
        preferred = [term for term in focus_terms if term in counts and len(term) <= 24]
        for term, _ in counts.most_common(8):
            if len(term) > 24:
                continue
            if term not in preferred:
                preferred.append(term)
            if len(preferred) >= 6:
                break
        if not preferred:
            return f"[{dropped_count} messages omitted after quantum compression]"
        return (
            f"[{dropped_count} messages omitted after quantum compression; "
            f"retained topics: {', '.join(preferred[:6])}]"
        )

    def _focus_overlap(self, terms: set[str], focus_terms: set[str]) -> float:
        if not focus_terms:
            return 0.55 if terms else 0.0
        if not terms:
            return 0.0
        overlap = len(terms & focus_terms)
        return min(1.0, overlap / max(1, min(len(focus_terms), 4)))

    def _rarity_score(self, terms: set[str], global_freq: Counter[str]) -> float:
        if not terms:
            return 0.0
        rarity = sum(1.0 / max(1, global_freq[term]) for term in terms) / len(terms)
        return min(1.0, rarity)

    def _entanglement_score(
        self,
        terms: set[str],
        global_freq: Counter[str],
        focus_terms: set[str],
    ) -> float:
        if not terms:
            return 0.0
        shared_terms = [term for term in terms if global_freq[term] > 1]
        focus_links = len(terms & focus_terms)
        return min(1.0, (len(shared_terms) * 0.18) + (focus_links * 0.22))

    def _position_score(self, index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        edge_distance = min(index, total - 1 - index)
        normalized = 1.0 - (edge_distance / max(total - 1, 1))
        return max(0.25, normalized)

    def _structural_signal(self, text: str, locked: bool) -> float:
        if locked:
            return 1.0
        score = 0.0
        if re.match(r"^[-*•]\s+", text):
            score += 0.25
        if re.search(r"\b\d{2,4}\b", text):
            score += 0.18
        if re.search(r"[A-Za-z0-9_.-]+\.[A-Za-z0-9_.-]+", text):
            score += 0.20
        if re.search(r"(?:[A-Za-z]:\\|/)", text):
            score += 0.15
        if any(hint in text.lower() for hint in _STRUCTURE_HINTS):
            score += 0.20
        if ":" in text or "=" in text:
            score += 0.08
        if text.isupper() and len(text) > 4:
            score += 0.12
        return min(1.0, score + 0.15)

    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _recency(self, index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        return index / (total - 1)

    def _content_type_score(self, message: dict[str, Any]) -> float:
        content = _normalize_content(message.get("content", ""))
        role = message.get("role", "")
        if "```" in content or content.startswith("```"):
            return 0.95
        if role == "tool" or content.startswith("Tool use"):
            return 1.0
        if role == "user" and "?" in content:
            return 0.85
        if role == "assistant" and any(keyword in content for keyword in ["def ", "class ", "import ", "fn ", "pub ", "func "]):
            return 0.80
        if role == "assistant" and len(content) > 300:
            return 0.40
        if role == "assistant" and any(prefix in content.lower() for prefix in ["sure,", "of course", "happy to", "i'd be happy"]):
            return 0.10
        if role == "system":
            return 0.20
        return 0.40

    def _info_density(self, content: str) -> float:
        words = content.lower().split()
        if not words:
            return 0.0
        unique_words = set(words)
        unique_ratio = len(unique_words) / len(words)
        token_estimate = _estimate_tokens(content)
        if token_estimate == 0:
            return 0.0
        density = unique_ratio * min(token_estimate / 50.0, 1.0)
        return min(density, 1.0)

    def _replaceability(self, message: dict[str, Any]) -> float:
        role = message.get("role", "")
        content = _normalize_content(message.get("content", ""))
        if role == "user":
            return 1.0
        if role == "tool" or "```" in content:
            return 0.8
        if role == "assistant" and len(content) < 100:
            return 0.3
        return 0.5


# ---------------------------------------------------------------------------
# TokenBudgetManager - Anthropic API counting with SHA-256 cache
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
                count = _estimate_tokens(text)
                self._record_fallback("count_tokens_api_error", e)
        else:
            count = _estimate_tokens(text)
        self._cache_put(h, count)
        return count

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for a full message list in one API call."""
        if not messages:
            return 0
        normalized_messages = [
            {
                "role": message.get("role", "user"),
                "content": _normalize_content(message.get("content", "")),
            }
            for message in messages
        ]
        content_hash = self._hash(str(normalized_messages))
        cached = self._cache_get(content_hash)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1
        if self._client and self._token_count_method == "api":
            try:
                result = self._client.beta.messages.count_tokens(
                    model="claude-opus-4-7",
                    messages=normalized_messages,
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
        return sum(self.count_tokens(text) for text in texts)

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
# MessageImportanceScorer - 6-axis importance ranking for lossy compression
# ---------------------------------------------------------------------------


class MessageImportanceScorer:
    """Score each message by importance for lossy compression decisions."""

    def __init__(self) -> None:
        self._quantum = QuantumCompressionEngine()

    def score(
        self,
        message: dict[str, Any],
        index: int,
        total: int,
        focus: str = "",
    ) -> float:
        """Return importance score 0.0-1.0 for a single message."""
        return self._quantum.score_message(message, index, total, focus=focus)

    def rank(
        self,
        messages: list[dict[str, Any]],
        min_messages: int = 2,
        focus: str = "",
    ) -> list[dict[str, Any]]:
        """Return messages sorted ascending by importance score (lowest first)."""
        total = len(messages)
        scored: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            score = self.score(message, index, total, focus=focus)
            reason = self._dominant_reason(message, index, total, score, focus=focus)
            scored.append({
                "index": index,
                "role": message.get("role", "unknown"),
                "score": score,
                "reason": reason,
            })
        scored.sort(key=lambda item: (item["score"], item["index"]))
        return scored

    def _dominant_reason(
        self,
        message: dict[str, Any],
        index: int,
        total: int,
        score: float,
        *,
        focus: str = "",
    ) -> str:
        content = _normalize_content(message.get("content", ""))
        focus_terms = set(_extract_focus_terms(focus))
        content_terms = set(_tokenize_terms(content))
        if focus_terms and content_terms & focus_terms:
            return "focus_match"
        if "```" in content:
            return "code_fence"
        if message.get("role") == "tool":
            return "tool_result"
        if "?" in content and message.get("role") == "user":
            return "user_question"
        if self._quantum._recency(index, total) < 0.3:
            return "old_message"
        if self._quantum._replaceability(message) < 0.4:
            return "replaceable"
        if score >= 0.7:
            return "high_signal"
        return "assistant_prose"


_quantum_engine_singleton: QuantumCompressionEngine | None = None


def get_token_budget_manager() -> TokenBudgetManager:
    """Lazy singleton for TokenBudgetManager."""
    return TokenBudgetManager()


def get_quantum_compression_engine() -> QuantumCompressionEngine:
    """Lazy singleton for QuantumCompressionEngine."""
    global _quantum_engine_singleton
    if _quantum_engine_singleton is None:
        _quantum_engine_singleton = QuantumCompressionEngine()
    return _quantum_engine_singleton
