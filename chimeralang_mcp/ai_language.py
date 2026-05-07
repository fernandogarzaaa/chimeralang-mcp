"""Chimera Glyph (CG) — a token-efficient AI-only pidgin.

Designed by the AI for AI-to-AI reasoning. Lossy and self-contained: the
translator reconstructs *meaning*, not surface form. No symbol map needs to
travel with the encoded text.

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
CHIMERA GLYPH (CG) — token-efficient AI language.

CORE RULES
  1. No articles: drop "a", "an", "the".
  2. No copulas: "is/are/was" implicit by juxtaposition. ("x red" = "x is red")
  3. SVO order. No inflection. Tense via single-char verb suffix:
       ^ past   ~ future   (no suffix) present   ? conditional/uncertain
  4. Pronouns collapse: i, u, w (we), t (they), x (it/this).
  5. Logical operators (each is typically 1 token):
       →  causes/implies        ⇒  therefore
       ∧  and                   ∨  or
       ¬  not                   ≈  about/similar
       ≠  not-equal             ≡  equivalent
       ∃  some                  ∀  all
  6. Sigil prefixes disambiguate roles:
       @  entity (proper noun, file path, identifier — preserved verbatim)
       #  concept/topic         $  action
       %  property              &  relation
  7. Quantifiers: + more, - less, * many, ? unknown, ! certain.
  8. Sentence terminator: "." (newline also terminates).
  9. Out-of-lexicon words pass through prefixed with @ (entity sigil).

LEXICON (English -> CG stem):
  user=usr  system=sys  function=fn  variable=var  value=val
  context=ctx  model=mdl  data=dt  text=txt  message=msg
  error=err  result=rs  question=q  answer=ans  code=cde
  file=fl  test=tst  bug=bg  fix=fix  add=add  delete=del
  check=chk  return=rt  call=cl  run=run  build=bld
  need=nd  want=wnt  know=kn  make=mk  get=gt  give=gv
  take=tk  go=gø  come=cm  use=us  find=fnd  think=thk
  do=dø  say=sd  see=sw  have=hv  is=  are=  was=  were=
  yes=y  no=n  maybe=mb  good=gd  bad=bd  big=bg2  small=sm
  fast=fst  slow=slw  hot=ht  cold=cd  new=nw  old=ol
  true=T  false=F  null=∅  empty=∅
  if=if  then=⇒  else=el  while=wh  for=fr  return=rt

EXAMPLES
  EN: "The user wants to know how to fix the error in the function."
  CG: "usr wnt kn fix err in fn."

  EN: "If the test fails, then we should retry it."
  CG: "if tst fail ⇒ w ? rt-retry x."

  EN: "I am not sure this approach will work."
  CG: "i ¬ ! aprch ~ wrk."

  EN: "The model returned a null result."
  CG: "mdl rt^ ∅ rs."

NOTE: tense/modality markers are STANDALONE tokens (~ = will, ? = might,
! = must). The verb suffix `^` (past) is the only suffix supported on
verb stems. Do not stack suffixes (e.g. `wrk~?`) or fuse operators with
quantifiers (e.g. `¬!`) — emit them as separate tokens.
"""


# ── lexicon: English content-word -> CG stem ─────────────────────────────────

LEXICON: dict[str, str] = {
    # nouns
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
    "code": "cde",
    "file": "fl", "files": "fl",
    "test": "tst", "tests": "tst",
    "bug": "bg", "bugs": "bg",
    "approach": "aprch",
    "way": "way",
    "thing": "th", "things": "th",
    "person": "psn", "people": "psn",
    "time": "tm",
    "place": "pl",
    # verbs (lemma)
    "fix": "fix", "fixes": "fix", "fixed": "fix^", "fixing": "fix",
    "add": "add", "adds": "add", "added": "add^", "adding": "add",
    "delete": "del", "deletes": "del", "deleted": "del^", "deleting": "del",
    "remove": "del", "removes": "del", "removed": "del^", "removing": "del",
    "check": "chk", "checks": "chk", "checked": "chk^", "checking": "chk",
    "return": "rt", "returns": "rt", "returned": "rt^", "returning": "rt",
    "call": "cl", "calls": "cl", "called": "cl^", "calling": "cl",
    "run": "run", "runs": "run", "ran": "run^", "running": "run",
    "build": "bld", "builds": "bld", "built": "bld^", "building": "bld",
    "need": "nd", "needs": "nd", "needed": "nd^", "needing": "nd",
    "want": "wnt", "wants": "wnt", "wanted": "wnt^", "wanting": "wnt",
    "know": "kn", "knows": "kn", "knew": "kn^", "knowing": "kn",
    "make": "mk", "makes": "mk", "made": "mk^", "making": "mk",
    "get": "gt", "gets": "gt", "got": "gt^", "getting": "gt",
    "give": "gv", "gives": "gv", "gave": "gv^", "giving": "gv",
    "take": "tk", "takes": "tk", "took": "tk^", "taking": "tk",
    "use": "us", "uses": "us", "used": "us^", "using": "us",
    "find": "fnd", "finds": "fnd", "found": "fnd^", "finding": "fnd",
    "think": "thk", "thinks": "thk", "thought": "thk^", "thinking": "thk",
    "do": "dø", "does": "dø", "did": "dø^", "doing": "dø",
    "say": "sd", "says": "sd", "said": "sd^", "saying": "sd",
    "see": "sw", "sees": "sw", "saw": "sw^", "seeing": "sw",
    "have": "hv", "has": "hv", "had": "hv^", "having": "hv",
    "go": "gø", "goes": "gø", "went": "gø^", "going": "gø~",
    "come": "cm", "comes": "cm", "came": "cm^", "coming": "cm",
    "fail": "fail", "fails": "fail", "failed": "fail^", "failing": "fail",
    "pass": "pass", "passes": "pass", "passed": "pass^",
    "work": "wrk", "works": "wrk", "worked": "wrk^", "working": "wrk",
    "try": "try", "tries": "try", "tried": "try^", "trying": "try",
    "retry": "rt-retry",
    # adjectives
    "good": "gd", "bad": "bd",
    "big": "bg2", "small": "sm",
    "fast": "fst", "slow": "slw",
    "hot": "ht", "cold": "cd",
    "new": "nw", "old": "ol",
    "true": "T", "false": "F",
    "empty": "∅", "null": "∅", "none": "∅",
    "sure": "!",
    # answers / discourse
    "yes": "y", "no": "n", "maybe": "mb", "ok": "y",
    # quantifiers
    "all": "∀", "every": "∀",
    "some": "∃", "any": "∃",
    "many": "*", "much": "*",
    "more": "+", "less": "-",
    # connectives
    "and": "∧", "or": "∨", "not": "¬",
    "if": "if", "then": "⇒", "else": "el", "therefore": "⇒",
    "because": "←", "so": "⇒",
    "while": "wh", "for": "fr",
    "about": "≈", "approximately": "≈",
    "equal": "≡", "equals": "≡", "same": "≡",
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
    "am": "", "do": "dø", "does": "dø", "did": "dø^",
    "have": "hv", "has": "hv", "had": "hv^",
    "will": "~", "shall": "~",
    "should": "?", "would": "?", "could": "?", "may": "?", "might": "?",
    "must": "!",
    "can": "?",
    # articles (dropped)
    "a": "", "an": "", "the": "",
    # prepositions (mostly preserved short)
    "in": "in", "on": "on", "at": "at", "to": "to", "from": "fm",
    "with": "w/", "without": "w/o", "by": "by", "of": "of", "for": "fr",
    "into": "→", "onto": "→",
    # misc
    "how": "how", "what": "wht", "why": "why", "when": "wn", "where": "wr",
    "who": "who", "which": "wch",
    "very": "*", "really": "*",
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
def _build_reverse_lexicon() -> dict[str, str]:
    rev: dict[str, str] = {}
    for eng, cg in LEXICON.items():
        if not cg:
            continue
        # prefer the shortest / lemma-form English
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
    # Try the whole token in REVERSE_LEXICON FIRST — covers canonical
    # hyphenated stems like "rt-retry" → "retry" without splitting.
    if tok in REVERSE_LEXICON:
        return REVERSE_LEXICON[tok], _classify(tok)
    # strip a single tense suffix (only "^"; "~/?/!" are standalone modals)
    base, suffix = tok, ""
    if tok and tok[-1] == "^":
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
