"""Unit tests for match.py."""

from __future__ import annotations

import pytest

from pleio_hpo.hpo import HPOTerm
from pleio_hpo.match import build_automaton, match


@pytest.fixture()
def terms() -> list[HPOTerm]:
    return [
        HPOTerm(id="HP:0000256", label="Macrocephaly", synonyms=["big head"]),
        HPOTerm(
            id="HP:0001263",
            label="Global developmental delay",
            synonyms=["GDD"],
        ),
        HPOTerm(id="HP:0001250", label="Seizures", synonyms=[]),
        HPOTerm(id="HP:0000545", label="Myopia", synonyms=[]),
    ]


@pytest.fixture()
def automaton(terms):
    return build_automaton(terms)


def test_single_term_match(automaton) -> None:
    hits = match("The patient has macrocephaly.", automaton)
    assert len(hits) == 1
    assert hits[0].hpo_id == "HP:0000256"


def test_synonym_match(automaton) -> None:
    hits = match("She has a big head.", automaton)
    ids = {h.hpo_id for h in hits}
    assert "HP:0000256" in ids


def test_longest_match_wins(automaton) -> None:
    # "global developmental delay" should match as a single term, not as
    # sub-strings of any shorter entries.
    hits = match("global developmental delay", automaton)
    assert len(hits) == 1
    assert hits[0].hpo_id == "HP:0001263"


def test_case_insensitive(automaton) -> None:
    hits = match("MACROCEPHALY is present.", automaton)
    assert any(h.hpo_id == "HP:0000256" for h in hits)


def test_short_word_filter_default(terms) -> None:
    # "id" is in the allowlist by default; "my" is not (< 3 chars, not in list)
    extra = HPOTerm(id="HP:0099999", label="my", synonyms=[])
    A = build_automaton(terms + [extra])
    hits = match("my patient", A)
    assert not any(h.hpo_id == "HP:0099999" for h in hits)


def test_word_boundary_no_substring_match() -> None:
    # "Tics" must not fire inside "genetics"; root "All" not inside "bilaterally".
    vocab = [
        HPOTerm(id="HP:0100033", label="Tics", synonyms=[]),
        HPOTerm(id="HP:0000001", label="All", synonyms=[]),
    ]
    A = build_automaton(vocab)
    assert match("Referred for genetics evaluation done bilaterally.", A) == []
    # Standalone words still match.
    hits = match("He has tics.", A)
    assert any(h.hpo_id == "HP:0100033" for h in hits)


# --- order-free (token-set) matching -------------------------------------


@pytest.fixture()
def ts_terms() -> list[HPOTerm]:
    return [
        HPOTerm(id="HP:0004375", label="Neoplasm of the nervous system", synonyms=[]),
        HPOTerm(id="HP:0009588", label="Vestibular schwannoma", synonyms=[]),
        HPOTerm(id="HP:0001249", label="Intellectual disability",
                synonyms=["Mental retardation"]),
        HPOTerm(id="HP:0000545", label="Myopia", synonyms=[]),
    ]


def test_token_set_matches_reordered_phrase(ts_terms) -> None:
    from pleio_hpo.match import build_token_set_index, token_set_match

    idx = build_token_set_index(ts_terms)
    hits = token_set_match("tumors aside, nervous system neoplasm was found", idx)
    assert any(h.hpo_id == "HP:0004375" for h in hits)


def test_token_set_singularizes(ts_terms) -> None:
    from pleio_hpo.match import build_token_set_index, token_set_match

    idx = build_token_set_index(ts_terms)
    hits = token_set_match("bilateral vestibular schwannomas", idx)
    assert any(h.hpo_id == "HP:0009588" for h in hits)


def test_token_set_uses_synonym_multiset(ts_terms) -> None:
    from pleio_hpo.match import build_token_set_index, token_set_match

    idx = build_token_set_index(ts_terms)
    hits = token_set_match("history of retardation mental in nature", idx)
    assert any(h.hpo_id == "HP:0001249" for h in hits)


def test_token_set_rejects_clause_crossing(ts_terms) -> None:
    from pleio_hpo.match import build_token_set_index, token_set_match

    idx = build_token_set_index(ts_terms)
    # "schwannoma. Vestibular nerve" — tokens straddle a sentence break, so the
    # multiset must NOT collapse into Vestibular schwannoma.
    hits = token_set_match("schwannoma. Vestibular nerve intact", idx)
    assert not any(h.hpo_id == "HP:0009588" for h in hits)


def test_token_set_skips_single_token_terms(ts_terms) -> None:
    from pleio_hpo.match import build_token_set_index, token_set_match

    idx = build_token_set_index(ts_terms)
    # Single-content-token "Myopia" is not indexed for order-free matching.
    assert ("myopia",) not in idx
    assert token_set_match("myopia noted", idx) == []
