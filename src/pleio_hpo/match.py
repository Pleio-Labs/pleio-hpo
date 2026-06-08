"""Lexical Aho-Corasick matching over HPO labels and synonyms.

"""

from __future__ import annotations

import re
from dataclasses import dataclass

import ahocorasick

from pleio_hpo.hpo import HPOTerm

# Short terms that are meaningful medical abbreviations and should be matched
# even though they are < 3 characters.
DEFAULT_SHORT_ALLOWLIST: frozenset[str] = frozenset({"id", "gdd", "ofc"})


@dataclass(frozen=True)
class LexicalHit:
    start: int
    end: int
    text: str
    hpo_id: str


def _is_word_bounded(text: str, start: int, end: int) -> bool:
    """True when ``text[start:end]`` is not glued to a longer word on either side.

    Aho-Corasick matches substrings anywhere, so without this an HPO term like
    "Tics" fires inside "gene*tics*" and the root term "All" inside "bilater*all*y".
    A match is accepted only when the adjacent characters are non-alphanumeric
    (or the span sits at a text edge) — i.e. regex ``\\b`` semantics.
    """
    left_ok = start == 0 or not text[start - 1].isalnum()
    right_ok = end >= len(text) or not text[end].isalnum()
    return left_ok and right_ok


def build_automaton(
    terms: list[HPOTerm],
    short_allowlist: frozenset[str] = DEFAULT_SHORT_ALLOWLIST,
) -> ahocorasick.Automaton:
    """Build a pyahocorasick Automaton from HPO term labels and synonyms.

    Keys are lowercase. Values are (hpo_id, original_label_or_synonym) pairs.
    When multiple terms map to the same key the first one wins (arbitrary but
    stable for a given ordered input list).
    """
    A: ahocorasick.Automaton = ahocorasick.Automaton()

    def _add(key: str, hpo_id: str, surface: str) -> None:
        lkey = key.lower()
        if len(lkey) < 3 and lkey not in short_allowlist:
            return
        if lkey not in A:
            A.add_word(lkey, (hpo_id, surface))

    for term in terms:
        _add(term.label, term.id, term.label)
        for syn in term.synonyms:
            _add(syn, term.id, syn)

    A.make_automaton()
    return A


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]*")
_CLAUSE_BREAK = re.compile(r"[.;:]")

# Order-free (token-set) matching config. A contiguous span whose singularized
# content-token multiset equals an HPO term's matches that term regardless of word
# order ("nervous system tumors" -> "Neoplasm of the nervous system"). Connective
# stopwords and "type" are dropped from the multiset; spans crossing clause/comma
# punctuation or shorter than two content tokens are rejected. The window caps the
# raw token span so the multiset stays a tight local phrase, not a sentence bag.
_TOKENSET_STOP: frozenset[str] = frozenset(
    {"of", "the", "a", "an", "and", "or", "with", "to", "type"}
)
_TOKENSET_WINDOW = 5
_TOKENSET_MIN_TOKENS = 2
_TOKENSET_BREAK = re.compile(r"[.;:,]")


def _singular(word: str) -> str:
    """Naive, deterministic singularizer (handles the common English plural forms).

    Bridges GSC+'s plural phrasing ("ear anomalies", "preauricular pits") to the
    singular HPO surface forms. Not a full lemmatizer — covers regular -s/-es/-ies.
    """
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def build_lemma_index(
    terms: list[HPOTerm],
    short_allowlist: frozenset[str] = DEFAULT_SHORT_ALLOWLIST,
) -> tuple[dict[tuple[str, ...], str], int]:
    """Map singularized token-tuples of labels/synonyms to HPO ids (+ max key length).

    A morphology-aware companion to the exact automaton: keys are tuples of
    singularized word tokens, so "ear anomaly" also matches "ear anomalies".
    """
    index: dict[tuple[str, ...], str] = {}
    max_len = 1
    for term in terms:
        for surface in (term.label, *term.synonyms):
            toks = tuple(_singular(m.group(0)) for m in _WORD_RE.finditer(surface))
            if not toks:
                continue
            if len(toks) == 1 and len(toks[0]) < 3 and toks[0] not in short_allowlist:
                continue
            index.setdefault(toks, term.id)
            max_len = max(max_len, len(toks))
    return index, max_len


def lemma_match(
    text: str,
    lemma_index: tuple[dict[tuple[str, ...], str], int],
) -> list[LexicalHit]:
    """Longest-match over singularized word tokens; rejects spans crossing a clause break."""
    index, max_len = lemma_index
    toks = [(m.start(), m.end(), _singular(m.group(0))) for m in _WORD_RE.finditer(text)]
    hits: list[LexicalHit] = []
    i = 0
    while i < len(toks):
        matched = False
        for j in range(min(max_len, len(toks) - i), 0, -1):
            key = tuple(t[2] for t in toks[i : i + j])
            hpo_id = index.get(key)
            if hpo_id is None:
                continue
            start, end = toks[i][0], toks[i + j - 1][1]
            if _CLAUSE_BREAK.search(text[start:end]):
                continue  # don't span a clause/sentence boundary
            hits.append(LexicalHit(start=start, end=end, text=text[start:end], hpo_id=hpo_id))
            i += j
            matched = True
            break
        if not matched:
            i += 1
    return hits


def _content_multiset(tokens: list[str]) -> tuple[str, ...]:
    """Sorted tuple of singularized content tokens (drops stopwords + short tokens)."""
    return tuple(sorted(
        t for t in tokens
        if t not in _TOKENSET_STOP and len(t) >= 2
    ))


def build_token_set_index(
    terms: list[HPOTerm],
) -> dict[tuple[str, ...], str]:
    """Map each label/synonym's content-token multiset to an HPO id.

    Order-free companion to the exact automaton and the morphology (lemma) index:
    keys are the sorted multiset of singularized content tokens, so reordered or
    scattered phrasings of the same term collide on one key. Only multi-token
    surfaces (>=2 content tokens) are indexed — single-token order-free matching
    would just duplicate the exact automaton while inviting noise.
    """
    index: dict[tuple[str, ...], str] = {}
    for term in terms:
        for surface in (term.label, *term.synonyms):
            toks = [_singular(m.group(0)) for m in _WORD_RE.finditer(surface.lower())]
            key = _content_multiset(toks)
            if len(key) >= _TOKENSET_MIN_TOKENS:
                index.setdefault(key, term.id)
    return index


def token_set_match(
    text: str,
    index: dict[tuple[str, ...], str],
) -> list[LexicalHit]:
    """Order-free matches: a contiguous token window whose content multiset is a key.

    Slides windows of up to ``_TOKENSET_WINDOW`` raw tokens; a window matches when
    its singularized content-token multiset (>=2 tokens) exactly equals an index
    key and no clause/comma punctuation falls inside the span. Less certain than an
    exact match, so callers route these through the validator (as for lemma hits).
    """
    toks = [(m.start(), m.end(), _singular(m.group(0).lower())) for m in _WORD_RE.finditer(text)]
    hits: list[LexicalHit] = []
    seen: set[tuple[str, int, int]] = set()
    n = len(toks)
    for i in range(n):
        for j in range(i + 1, min(i + _TOKENSET_WINDOW, n) + 1):
            start, end = toks[i][0], toks[j - 1][1]
            if _TOKENSET_BREAK.search(text[start:end]):
                continue  # don't span a clause/list boundary
            key = _content_multiset([t for _, _, t in toks[i:j]])
            if len(key) < _TOKENSET_MIN_TOKENS:
                continue
            hpo_id = index.get(key)
            if hpo_id is not None and (hpo_id, start, end) not in seen:
                seen.add((hpo_id, start, end))
                hits.append(LexicalHit(start=start, end=end, text=text[start:end], hpo_id=hpo_id))
    return hits


def match(
    text: str,
    automaton: ahocorasick.Automaton,
    short_allowlist: frozenset[str] = DEFAULT_SHORT_ALLOWLIST,
) -> list[LexicalHit]:
    """Run the automaton over *text* (case-insensitive), return longest-match hits.

    Overlapping matches are resolved by longest-match-wins: shorter spans
    entirely contained within a longer span are dropped.
    """
    lower = text.lower()
    raw: list[tuple[int, int, str, str]] = []  # (start, end, hpo_id, surface)

    for end_idx, (hpo_id, surface) in automaton.iter(lower):
        key_len = len(surface.lower())
        start_idx = end_idx - key_len + 1
        if not _is_word_bounded(lower, start_idx, end_idx + 1):
            continue
        raw.append((start_idx, end_idx + 1, hpo_id, surface))

    # Longest-match resolution: sort by length desc, then start asc.
    raw.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

    accepted: list[tuple[int, int, str, str]] = []
    for cand in raw:
        cs, ce = cand[0], cand[1]
        # Check if this candidate is contained within any already-accepted span.
        dominated = any(
            (a[0] <= cs and ce <= a[1]) for a in accepted
        )
        if not dominated:
            accepted.append(cand)

    # Re-sort by start offset for a predictable output order.
    accepted.sort(key=lambda x: x[0])

    return [
        LexicalHit(start=s, end=e, text=text[s:e], hpo_id=hid)
        for s, e, hid, _ in accepted
    ]
