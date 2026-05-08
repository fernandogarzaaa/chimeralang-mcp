"""Chimera Glyph (CG) — deterministic AI-to-AI wire format.

Designed by the AI for AI-to-AI reasoning. Lossy and self-contained: the
translator reconstructs *meaning*, not surface form. No symbol map needs to
travel with the encoded text.

Note: CG is *not* a token-compression scheme. BPE benchmarks (v0.7.3,
tiktoken o200k_base) show CG is ≈16% *longer* in tokens than plain English
on representative corpora — modern tokenisers already collapse common English
to single tokens. CG's value is semantic determinism: sigil-marked entities,
explicit modality tokens, and stable round-trip decode. Use chimera_optimize /
chimera_fracture / chimera_cache_mark for actual token reduction.

Public surface:
    GRAMMAR_SPEC, LEXICON, OPERATORS
    encode(text) -> str
    decode(glyph_text) -> tuple[str, list[str]]
    directive(style="strict", task_hint=None) -> str
    estimate_savings(english) -> dict
"""
from __future__ import annotations

import re


# ── grammar spec (shipped to the AI as part of the directive) ────────────────

GRAMMAR_SPEC = """\
CHIMERA GLYPH (CG) — deterministic AI-to-AI wire format, BPE-aware.

Note: CG produces output that is ~16% *longer* in BPE tokens than plain
English (tiktoken o200k_base benchmark, v0.7.3). Its value is semantic
determinism and verifiable handoffs, not token compression. Use
chimera_optimize/chimera_fracture/chimera_cache_mark for cost reduction.

CORE RULES
  1. No articles: drop "a", "an", "the".
  2. No copulas: "is/are/was" implicit by juxtaposition. ("x red" = "x is red")
  3. SVO order. Tense / modality emitted as a standalone token before the
     verb (the suffix scheme `wrk^` was retired in 0.7.3 because most
     suffix forms cost +1-2 BPE tokens vs the plain English past tense):
       ~ future / will        ? might / uncertain
       ! must / certain       (no marker) = present
     Past tense uses the English form directly: "fixed", "ran", "built".
     Do NOT stack markers (`~?`) or fuse them with operators (`¬!`) —
     emit each as its own token.
  4. Pronouns collapse: i, u, w (we), t (they), x (it/this).
  5. Logical operators (each verified 1 BPE token in o200k_base):
       ⇒  therefore / then     ←  because
       ≠  not-equal            →  to / into
     Connectives that cost 2+ tokens as Unicode were demoted to English:
       and, or, not, all, some, about, equal, none.
  6. Sigil prefixes disambiguate roles:
       @  entity (proper noun, file path, identifier — preserved verbatim)
       #  concept/topic         $  action
       %  property              &  relation
  7. Quantifiers: + more, - less, * many, ? unknown, ! certain.
  8. Sentence terminator: "." (newline also terminates).
  9. Out-of-lexicon words pass through prefixed with @ (entity sigil).

LEXICON HINTS (English -> CG stem; only listing the substitutions —
unlisted words encode to themselves):
  user=usr  system=sys  function=fn  variable=var  value=val
  context=ctx  model=mdl  data=dt  text=txt  message=msg
  error=err  result=rs  question=q  answer=ans
  file=fl  test=tst  bug=bg  delete=del  check=chk  return=rt
  call=cl  need=nd  know=kn  make=mk  get=gt  give=gv
  take=tk  use=us  come=cm  say=sd  see=sw  have=hv
  is=  are=  was=  the=  a=  an=
  yes=y  no=n  maybe=mb  good=gd  bad=bd  small=sm  fast=fst
  hot=ht  cold=cd  new=nw  old=ol  true=T  false=F
  while=wh  for=fr  from=fm  when=wn  where=wr

EXAMPLES (round-trip safe under decode())
  EN: "The user wants to know how to fix the error in the function."
  CG: "usr wants kn fix err in fn."

  EN: "If the test fails, then we should retry it."
  CG: "if tst fail ⇒ w ? retry x."

  EN: "Maybe this approach will work."
  CG: "mb x approach ~ work."

  EN: "The model returned a null result."
  CG: "mdl returned none rs."
"""


# ── lexicon: English content-word -> CG stem ─────────────────────────────────

LEXICON: dict[str, str] = {
    # nouns — most common English words are 1-token in BPE; only short
    # abbreviations that are also 1-token actually save anything.
    "user": "usr", "users": "usr",
    "system": "sys", "systems": "sys",
    "function": "fn", "functions": "fn",
    "variable": "var", "variables": "var",
    "value": "val", "values": "val",
    "context": "ctx",
    "model": "mdl", "models": "mdl",
    "data": "dt",
    "text": "txt",
    "message": "msg", "messages": "msg",
    "error": "err", "errors": "err",
    "result": "rs", "results": "rs",
    "question": "q", "questions": "q",
    "answer": "ans", "answers": "ans",
    "code": "code",                                       # was "cde" (2 BPE tokens)
    "file": "fl", "files": "fl",
    "test": "tst", "tests": "tst",
    "bug": "bg", "bugs": "bg",
    "approach": "approach",                               # was "aprch" (2 BPE tokens)
    "way": "way",
    "thing": "th", "things": "th",
    "person": "person", "people": "people",               # was "psn" (2 BPE tokens)
    "time": "tm",
    "place": "pl",
    # verbs — present/lemma forms (suffix scheme dropped, see past-tense block)
    "fix": "fix", "fixes": "fix", "fixing": "fix",
    "add": "add", "adds": "add", "adding": "add",
    "delete": "del", "deletes": "del", "deleting": "del",
    "remove": "del", "removes": "del", "removing": "del",
    "check": "chk", "checks": "chk", "checking": "chk",
    "return": "rt", "returns": "rt", "returning": "rt",
    "call": "cl", "calls": "cl", "calling": "cl",
    "run": "run", "runs": "run", "running": "run",
    "build": "build", "builds": "builds", "building": "building",  # was "bld" (2 BPE tokens)
    "need": "nd", "needs": "nd", "needing": "nd",
    "want": "want", "wants": "wants", "wanting": "wanting",        # was "wnt" (2 BPE tokens)
    "know": "kn", "knows": "kn", "knowing": "kn",
    "make": "mk", "makes": "mk", "making": "mk",
    "get": "gt", "gets": "gt", "getting": "gt",
    "give": "gv", "gives": "gv", "giving": "gv",
    "take": "tk", "takes": "tk", "taking": "tk",
    "use": "us", "uses": "us", "using": "us",
    "find": "find", "finds": "finds", "finding": "finding",        # was "fnd" (2 BPE tokens)
    "think": "think", "thinks": "thinks", "thinking": "thinking",  # was "thk" (2 BPE tokens)
    "do": "do", "does": "dø", "doing": "dø",
    "say": "sd", "says": "sd", "saying": "sd",
    "see": "sw", "sees": "sw", "seeing": "sw",
    "have": "hv", "has": "hv", "having": "hv",
    "go": "go", "goes": "goes", "going": "going",                  # was "gø/gø~" (multi-token)
    "come": "cm", "comes": "cm", "coming": "cm",
    "fail": "fail", "fails": "fail", "failing": "fail",
    "pass": "pass", "passes": "pass",
    "work": "work", "works": "works", "working": "working",        # was "wrk" (2 BPE tokens)
    "try": "try", "tries": "try", "trying": "try",
    "retry": "retry",                                              # was "rt-retry" (3 BPE tokens)
    # past-tense — English wins; the suffix scheme (`fix^`, `wnt^`) cost +1-2 tokens.
    "fixed": "fixed", "added": "added", "deleted": "deleted",
    "removed": "removed", "checked": "checked", "returned": "returned",
    "called": "called", "ran": "ran", "built": "built",
    "needed": "needed", "wanted": "wanted", "knew": "knew",
    "made": "made", "got": "got", "gave": "gave", "took": "took",
    "used": "used", "found": "found", "thought": "thought",
    "did": "did", "said": "said", "saw": "saw", "had": "had",
    "went": "went", "came": "came", "failed": "failed",
    "passed": "passed", "worked": "worked", "tried": "tried",
    # adjectives
    "good": "gd", "bad": "bd",
    "big": "big", "small": "sm",
    "fast": "fst", "slow": "slow",
    "hot": "ht", "cold": "cd",
    "new": "nw", "old": "ol",
    "true": "T", "false": "F",
    "empty": "none", "null": "none", "none": "none",   # ∅ was 2 BPE tokens
    "sure": "!",
    # answers / discourse
    "yes": "y", "no": "n", "maybe": "mb", "ok": "y",
    # quantifiers
    "all": "all", "every": "all",                      # ∀ was 2 BPE tokens
    "some": "some", "any": "some",                     # ∃ was 2 BPE tokens
    "many": "*", "much": "*",
    "more": "+", "less": "-",
    # connectives — English wins for the bigrams; ⇒ / ← / ≠ stay (1-token Unicode).
    "and": "and", "or": "or", "not": "not",            # ∧/∨/¬ were 2 BPE tokens
    "if": "if", "then": "⇒", "else": "el",
    "therefore": "⇒", "because": "←", "so": "⇒",
    "while": "wh", "for": "fr",
    "about": "about", "approximately": "about",        # ≈ was 2 BPE tokens
    "equal": "eq", "equals": "eq", "same": "eq",       # ≡ was 2 BPE tokens
    "different": "≠",
    # pronouns
    "i": "i", "me": "i", "my": "i",
    "you": "u", "your": "u",
    "we": "w", "us": "w", "our": "w",
    "they": "t", "them": "t", "their": "t",
    "it": "x", "its": "x", "this": "x", "that": "x",
    "he": "p", "she": "p", "him": "p", "her": "p",
    # auxiliaries (dropped in CG; mapped to "" so encoder skips them)
    "is": "", "are": "", "was": "", "were": "", "be": "", "been": "", "being": "",
    "am": "",
    "will": "~", "shall": "~",
    "should": "?", "would": "?", "could": "?", "may": "?", "might": "?",
    "must": "!",
    "can": "?",
    # articles (dropped)
    "a": "", "an": "", "the": "",
    # prepositions
    "in": "in", "on": "on", "at": "at", "to": "to", "from": "fm",
    "with": "with", "without": "without",              # were "w/", "w/o" (multi-token)
    "by": "by", "of": "of",
    "into": "→", "onto": "→",
    # misc
    "how": "how", "what": "what", "why": "why",        # "wht" was 2 BPE tokens
    "when": "wn", "where": "wr",
    "who": "who", "which": "which",                    # "wch" was 2 BPE tokens
    "very": "*", "really": "*",
}


# Legacy stems retained for decoder backwards compatibility. CG text emitted
# by chimeralang-mcp ≤ 0.7.2 may contain these forms; the decoder must still
# map them to plausible English. The encoder no longer emits them — see
# tools/lexicon_diff.json for the full migration record.
LEGACY_GLYPH_REVERSE: dict[str, str] = {
    # noun stems retired (BPE multi-token)
    "cde": "code", "aprch": "approach", "psn": "person",
    # past-tense suffix forms retired
    "fix^": "fixed", "add^": "added", "del^": "deleted",
    "chk^": "checked", "rt^": "returned", "cl^": "called",
    "run^": "ran", "bld": "build", "bld^": "built",
    "nd^": "needed", "wnt": "want", "wnt^": "wanted",
    "kn^": "knew", "mk^": "made", "gt^": "got",
    "gv^": "gave", "tk^": "took", "us^": "used",
    "fnd": "find", "fnd^": "found",
    "thk": "think", "thk^": "thought",
    "dø^": "did", "sd^": "said", "sw^": "saw", "hv^": "had",
    "gø": "go", "gø^": "went", "gø~": "going",
    "cm^": "came", "fail^": "failed", "pass^": "passed",
    "wrk": "work", "wrk^": "worked",
    "try^": "tried", "rt-retry": "retry",
    # adjective stems retired
    "bg2": "big", "slw": "slow",
    # operator stems retired (Unicode multi-token)
    "∅": "empty", "∀": "all", "∃": "some",
    "∧": "and", "∨": "or", "¬": "not",
    "≈": "about", "≡": "equal",
    # misc retired
    "wht": "what", "wch": "which",
    "w/": "with", "w/o": "without",
}


# operator glyph -> rough English gloss (used by decoder)
OPERATORS: dict[str, str] = {
    "→": "leads to",
    "⇒": "therefore",
    "∧": "and",
    "∨": "or",
    "¬": "not",
    "≈": "about",
    "≠": "not equal to",
    "≡": "equivalent to",
    "∃": "some",
    "∀": "all",
    "∅": "empty",
    "←": "because",
}


# auto-built reverse lexicon: CG stem -> English (first English mapping wins)
# Order matters: we layer the current LEXICON on top of LEGACY_GLYPH_REVERSE so
# new mappings win for any stem the encoder still emits, while legacy stems
# (no longer produced by encode()) remain decodable.
def _build_reverse_lexicon() -> dict[str, str]:
    rev: dict[str, str] = dict(LEGACY_GLYPH_REVERSE)
    for eng, cg in LEXICON.items():
        if not cg:
            continue
        # Prefer the shortest / lemma-form English when multiple English words
        # collide on the same stem.
        if cg not in rev or len(eng) < len(rev[cg]):
            rev[cg] = eng
    return rev


REVERSE_LEXICON: dict[str, str] = _build_reverse_lexicon()


# ── encoder: English -> CG ───────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'_-]*|[.!?,;:]|\S")


def encode(text: str) -> str:
    """Lossy English -> Chimera Glyph encoder.

    Word-level lookup against LEXICON. Out-of-lexicon words are preserved as
    @entity tokens. Punctuation collapses to "." for sentence terminators only.
    """
    if not text:
        return ""
    out: list[str] = []
    for tok in _WORD_RE.findall(text):
        low = tok.lower()
        if tok in (".", "!", "?"):
            out.append(".")
            continue
        if tok in (",", ";", ":"):
            continue  # drop minor punctuation
        if low in LEXICON:
            cg = LEXICON[low]
            if cg:  # skip empty (articles, copulas)
                out.append(cg)
            continue
        if tok.isdigit() or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", tok):
            out.append(tok)
            continue
        # out-of-lexicon: entity sigil
        out.append(f"@{tok}")
    # collapse multiple spaces and "." sequences
    encoded = " ".join(out)
    encoded = re.sub(r"\s*\.\s*", ". ", encoded)
    encoded = re.sub(r"(\.\s*)+", ". ", encoded)
    return encoded.strip()


# ── decoder: CG -> English (lossy reconstruction) ────────────────────────────

# tokens that act as connectives where we should NOT prepend "the".
# Note: `?` and `!` are intentionally NOT in here — they are modality markers
# (some/certain), and `.` is the sole sentence terminator.
_CONNECTIVE_TOKENS = set(OPERATORS.keys()) | {
    "if", "el", "wh", "fr", "in", "on", "at", "to", "fm", "w/", "w/o",
    "by", "of", "how", "wht", "why", "wn", "wr", "who", "wch",
    "y", "n", "mb", "T", "F",
    ".",
    "i", "u", "w", "t", "x", "p",
}

_PRONOUN_TOKENS = {"i", "u", "w", "t", "x", "p"}

# Standalone modality / tense glyphs (when not used as a verb suffix).
_STANDALONE_MODAL: dict[str, str] = {
    "~": "will",
    "?": "might",
    "!": "must",
}


def _decode_token(tok: str, notes: list[str]) -> tuple[str, str]:
    """Return (english, kind) for a single CG token. kind ∈ {noun, verb,
    adj, op, ent, num, conn, term, drop, modal}.
    """
    if tok == ".":
        return ".", "term"
    if tok in _STANDALONE_MODAL:
        return _STANDALONE_MODAL[tok], "modal"
    if tok in OPERATORS:
        return OPERATORS[tok], "op"
    if tok.startswith("@"):
        return tok[1:], "ent"
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", tok):
        return tok, "num"
    # Hyphenated stems: try the whole token in REVERSE_LEXICON first.
    # Catches canonical compounds like "rt-retry" → "retry" without
    # splitting into ["return", "retry"]. Restricted to hyphenated
    # tokens so it doesn't shadow suffix semantics for entries like
    # "going" -> "gø~" (the ~ must still be applied as future).
    if "-" in tok and tok in REVERSE_LEXICON:
        return REVERSE_LEXICON[tok], _classify(tok)
    # Legacy suffix forms (e.g. dø^→did, hv^→had) are stored verbatim in
    # REVERSE_LEXICON (from LEGACY_GLYPH_REVERSE). Check for an exact match
    # for past-tense (^) tokens *before* stripping the suffix so these legacy
    # entries take priority over the stem-lookup + _past_tense() path, which
    # would produce wrong results for stems like dø (does → doesed) or hv
    # (has → hased). Modal suffixes (~/?/!) are intentionally excluded so that
    # entries like gø~ continue to decode as "will go" per the 0.7.2 contract.
    if tok.endswith("^") and tok in REVERSE_LEXICON:
        return REVERSE_LEXICON[tok], _classify(tok[:-1] or tok)
    # Strip a single trailing tense/modal suffix. Bare-glyph standalone
    # modals (just "~"/"?"/"!") were already handled above; here we only
    # strip when the token has a stem in front of the suffix.
    base, suffix = tok, ""
    if len(tok) > 1 and tok[-1] in {"^", "~", "?", "!"}:
        base, suffix = tok[:-1], tok[-1]
    # full base in lexicon (post-suffix-strip)
    if base in REVERSE_LEXICON:
        word = REVERSE_LEXICON[base]
    elif base in _PRONOUN_TOKENS:
        word = {"i": "I", "u": "you", "w": "we", "t": "they",
                "x": "it", "p": "they"}[base]
    elif base in _CONNECTIVE_TOKENS:
        word = base
    elif "-" in base:
        # fallback: split a non-canonical hyphenated stem
        parts = [REVERSE_LEXICON.get(p, p) for p in base.split("-")]
        word = " ".join(p for p in parts if p)
    elif not base:
        return "", "drop"
    else:
        notes.append(f"unrecognized:{tok}")
        return f"[{tok}]", "ent"
    if suffix == "^":
        word = _past_tense(word)
    elif suffix == "~":
        word = "will " + word
    elif suffix == "?":
        word = "might " + word
    elif suffix == "!":
        word = "must " + word
    return word, _classify(base)


def _classify(stem: str) -> str:
    if stem in _PRONOUN_TOKENS:
        return "pron"
    if stem in _CONNECTIVE_TOKENS:
        return "conn"
    # crude: lexicon stems mapped from verb-table words are verbs
    if stem in {"fix", "add", "del", "chk", "rt", "cl", "run", "bld",
                "nd", "wnt", "kn", "mk", "gt", "gv", "tk", "us", "fnd",
                "thk", "dø", "sd", "sw", "hv", "gø", "cm", "fail", "pass",
                "wrk", "try"}:
        return "verb"
    if stem in {"gd", "bd", "bg2", "sm", "fst", "slw", "ht", "cd", "nw", "ol", "T", "F"}:
        return "adj"
    return "noun"


def _past_tense(word: str) -> str:
    irregular = {
        "go": "went", "do": "did", "have": "had", "make": "made",
        "say": "said", "see": "saw", "get": "got", "give": "gave",
        "take": "took", "come": "came", "find": "found", "think": "thought",
        "run": "ran", "build": "built",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith("e"):
        return word + "d"
    return word + "ed"


def decode(glyph_text: str) -> tuple[str, list[str]]:
    """Lossy CG -> English. Returns (english_text, notes).

    The reconstruction is deliberately simple: token-by-token expansion,
    sentence boundaries on ".", and a minimal "the"-insertion heuristic
    so the output reads as English rather than as a stem soup.
    """
    if not glyph_text:
        return "", []
    notes: list[str] = []
    raw_tokens = glyph_text.split()
    sentences: list[list[tuple[str, str]]] = [[]]
    for tok in raw_tokens:
        # strip trailing "." and emit as separate terminator
        if tok.endswith(".") and tok != ".":
            inner = tok[:-1]
            if inner:
                sentences[-1].append(_decode_token(inner, notes))
            sentences[-1].append((".", "term"))
            sentences.append([])
            continue
        if tok == ".":
            sentences[-1].append((".", "term"))
            sentences.append([])
            continue
        sentences[-1].append(_decode_token(tok, notes))

    pieces: list[str] = []
    for sent in sentences:
        if not sent or all(k == "term" for _, k in sent):
            continue
        pieces.append(_render_sentence(sent))
    text = " ".join(pieces).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    if text and text[-1] not in ".!?":
        text += "."
    if text:
        text = text[0].upper() + text[1:]
    return text, notes


def _render_sentence(words: list[tuple[str, str]]) -> str:
    """Tiny SVO heuristic: insert "the" before nouns that aren't already
    preceded by a determiner/pronoun/preposition; insert "is" between a
    bare noun and a trailing adjective."""
    out: list[str] = []
    prev_kind = "start"
    for w, kind in words:
        if kind == "term":
            out.append(".")
            prev_kind = "start"
            continue
        if kind == "noun":
            if prev_kind in {"start", "verb", "conn", "op"}:
                out.append("the")
            out.append(w)
        elif kind == "adj" and prev_kind == "noun":
            out.append("is")
            out.append(w)
        elif kind == "drop" or w == "":
            continue
        else:
            out.append(w)
        prev_kind = kind
    return " ".join(out)


# ── directive: system instruction that forces an AI to use CG ────────────────

def directive(style: str = "strict", task_hint: str | None = None) -> str:
    """Build the system instruction that constrains an LLM to write only in CG.

    style="strict"   — adds an explicit "REJECT non-CG output" line.
    style="balanced" — softer phrasing; allows brief English fallback when CG
                       cannot express a domain term.
    """
    header = (
        "You MUST respond using Chimera Glyph (CG), an AI-only language "
        "designed for token efficiency. Do not use ordinary English prose."
    )
    if style == "strict":
        rule = (
            "STRICT MODE: every output token must be a CG token. If you cannot "
            "express something in CG, mark it with @ as an entity. REJECT any "
            "internal urge to write English prose — the human will translate "
            "your output later."
        )
    else:
        rule = (
            "BALANCED MODE: prefer CG for all reasoning. Brief English fallback "
            "is allowed only for domain terms with no CG mapping; flag those "
            "with @ as entities."
        )
    examples = (
        "EXAMPLES OF VALID OUTPUT:\n"
        "  usr wnt kn fix err in fn.\n"
        "  if tst fail ⇒ w ? rt-retry x.\n"
        "  i thk aprch ~ wrk ∧ rs gd."
    )
    hint = f"\n\nTASK CONTEXT: {task_hint}" if task_hint else ""
    return f"{header}\n\n{rule}\n\n{GRAMMAR_SPEC}\n{examples}{hint}"


# ── savings estimator ────────────────────────────────────────────────────────

def _estimate_tokens(s: str) -> int:
    return max(0, len(s) // 4)


def estimate_savings(english: str) -> dict:
    glyph = encode(english)
    en_tok = _estimate_tokens(english)
    cg_tok = _estimate_tokens(glyph)
    saved = max(0, en_tok - cg_tok)
    ratio = (saved / en_tok) if en_tok else 0.0
    return {
        "english_tokens": en_tok,
        "glyph_tokens": cg_tok,
        "tokens_saved": saved,
        "savings_ratio": round(ratio, 3),
        "glyph": glyph,
    }
