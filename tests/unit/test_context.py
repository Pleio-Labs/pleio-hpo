"""Unit tests for context.py — >=12 patterns."""

from __future__ import annotations

import pytest

from pleio_hpo.context import detect


def _span(text: str, mention: str) -> tuple[int, int]:
    i = text.index(mention)
    return (i, i + len(mention))


# (text, mention, expect_negated, expect_family, expect_uncertain)
CASES = [
    # --- Negation ---
    ("Patient has no seizures.", "seizures", True, False, False),
    ("She denies headache.", "headache", True, False, False),
    ("Examined without polydactyly.", "polydactyly", True, False, False),
    ("There is absence of macrocephaly.", "macrocephaly", True, False, False),
    ("No evidence of webbed neck on exam.", "webbed neck", True, False, False),
    # --- Family history ---
    ("The mother had ataxia.", "ataxia", False, True, False),
    ("A sibling with hypotonia was noted.", "hypotonia", False, True, False),
    ("Family history of GDD is reported.", "GDD", False, True, False),
    ("A paternal cousin with short foot.", "short foot", False, True, False),
    # --- Uncertainty ---
    ("Findings show possible Marfan syndrome.", "Marfan syndrome", False, False, True),
    ("This is consistent with hypotonia.", "hypotonia", False, False, True),
    ("Features suggestive of ataxia.", "ataxia", False, False, True),
    ("Imaging is concerning for medulloblastoma.", "medulloblastoma", False, False, True),
    # --- Present (counter-examples: no flags) ---
    ("The patient has macrocephaly.", "macrocephaly", False, False, False),
    ("On exam, hypotonia is evident.", "hypotonia", False, False, False),
]


@pytest.mark.parametrize("text,mention,neg,fam,unc", CASES)
def test_context_patterns(text, mention, neg, fam, unc) -> None:
    flags = detect(text, _span(text, mention))
    assert flags.negated is neg, f"negated mismatch for {mention!r} in {text!r}"
    assert flags.family_history is fam, f"family mismatch for {mention!r} in {text!r}"
    assert flags.uncertain is unc, f"uncertain mismatch for {mention!r} in {text!r}"


def test_negation_does_not_leak_across_sentence() -> None:
    # "no" in the prior sentence must not negate a finding in the next.
    text = "There is no rash. The patient has macrocephaly."
    flags = detect(text, _span(text, "macrocephaly"))
    assert flags.negated is False
