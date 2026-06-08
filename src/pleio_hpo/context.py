"""Negation, family-history, and uncertainty detection (NegEx-style).

Hand-rolled rather than wrapping
negspacy so the trigger sets are transparent and auditable (Task 6.2). For a
mention span, we look at the surrounding clause: a window of characters before
the mention (and, for some cues, after it) within the same sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Cues before the mention, within the sentence, that negate the finding.
NEGATION_PRE = (
    "no evidence of",
    "without evidence of",
    "no sign of",
    "no signs of",
    "no abnormality of",
    "negative for",
    "free of",
    "absence of",
    "lack of",
    "ruled out",
    "without the",
    "without",
    "denies",
    "denied",
    "deny",
    "absent",
    "never",
    "no",
    "not",
    "neither",
    "nor",
)

# Cues after the mention that negate it ("X was not seen").
NEGATION_POST = (
    "was not seen",
    "were not seen",
    "not seen",
    "not identified",
    "not detected",
    "not present",
    "not appreciated",
    "not observed",
    "was absent",
    "were absent",
    "is absent",
    "ruled out",
)

FAMILY_PRE = (
    "family history",
    "familial",
    "mother",
    "maternal",
    "father",
    "paternal",
    "sister",
    "brother",
    "sibling",
    "siblings",
    "cousin",
    "aunt",
    "uncle",
    "grandmother",
    "grandfather",
    "grandparent",
    "grandparents",
    "relative",
    "relatives",
)

UNCERTAIN_PRE = (
    "differential diagnosis",
    "concerning for",
    "concern for",
    "suspicious for",
    "suggestive of",
    "consistent with",
    "question of",
    "questionable",
    "cannot exclude",
    "cannot rule out",
    "raising the possibility",
    "raising the question",
    "rule out",
    "evaluate for",
    "possible",
    "possibly",
    "probable",
    "probably",
    "presumed",
    "suspected",
    "suspect",
    "apparent",
    "versus",
)

UNCERTAIN_POST = (
    "could not be confirmed",
    "could not be excluded",
    "could not be ruled out",
    "cannot be excluded",
    "cannot be confirmed",
    "remains a concern",
    "remains a consideration",
    "remains a possibility",
    "is suspected",
    "was suspected",
    "is uncertain",
)

# How many characters of context to scan, capped at the sentence boundary.
PRE_WINDOW_CHARS = 70
POST_WINDOW_CHARS = 45
_SENT_BOUNDARY = re.compile(r"[.!?\n]")


@dataclass(frozen=True)
class ContextFlags:
    negated: bool = False
    family_history: bool = False
    uncertain: bool = False


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return (sent_start, sent_end) for the sentence containing [start, end)."""
    left = 0
    for m in _SENT_BOUNDARY.finditer(text, 0, start):
        left = m.end()
    right_match = _SENT_BOUNDARY.search(text, end)
    right = right_match.start() if right_match else len(text)
    return left, right


def _contains_phrase(window: str, phrases: tuple[str, ...]) -> bool:
    """True if any phrase appears in window as a whole-word match."""
    return any(
        re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", window) for phrase in phrases
    )


def detect(text: str, span: tuple[int, int]) -> ContextFlags:
    """Detect negation / family-history / uncertainty around a mention span."""
    start, end = span
    sent_start, sent_end = _sentence_bounds(text, start, end)

    pre = text[max(sent_start, start - PRE_WINDOW_CHARS) : start].lower()
    post = text[end : min(sent_end, end + POST_WINDOW_CHARS)].lower()
    # Gold spans often swallow the negation cue ("no webbed neck"), so the
    # negation scan extends through the mention itself.
    neg_pre = text[max(sent_start, start - PRE_WINDOW_CHARS) : end].lower()

    negated = _contains_phrase(neg_pre, NEGATION_PRE) or _contains_phrase(
        post, NEGATION_POST
    )
    family = _contains_phrase(pre, FAMILY_PRE)
    uncertain = _contains_phrase(pre, UNCERTAIN_PRE) or _contains_phrase(post, UNCERTAIN_POST)

    return ContextFlags(negated=negated, family_history=family, uncertain=uncertain)
