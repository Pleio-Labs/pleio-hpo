"""Unit tests for annotator.py.

Loads the full pipeline (SapBERT, validator, index) once via a module-scoped
fixture, so these are slower than the pure-unit tests.
"""

from __future__ import annotations

import json

import pytest

from pleio_hpo import Annotator
from pleio_hpo.embed_match import default_index_path
from pleio_hpo.validator import default_validator_path

pytestmark = pytest.mark.skipif(
    not (default_index_path().exists() and default_validator_path().exists()),
    reason="requires bundled embedding index + validator checkpoint (built locally / via download)",
)


@pytest.fixture(scope="module")
def annotator() -> Annotator:
    return Annotator()


def test_macrocephaly_detected(annotator: Annotator) -> None:
    result = annotator.annotate("Patient has macrocephaly.")
    assert "HP:0000256" in {c.hpo_id for c in result.codes}


def test_negation_default_excludes_then_includes(annotator: Annotator) -> None:
    text = "The patient has no macrocephaly."
    excluded = annotator.annotate(text)  # include_negated=False (default)
    assert "HP:0000256" not in {c.hpo_id for c in excluded.codes}

    included = annotator.annotate(text, include_negated=True)
    macro = next((c for c in included.codes if c.hpo_id == "HP:0000256"), None)
    assert macro is not None
    assert "negated" in macro.context_flags


def test_to_json_roundtrips(annotator: Annotator) -> None:
    result = annotator.annotate("Patient has macrocephaly.")
    payload = json.loads(result.to_json())
    assert set(payload) >= {"annotator_version", "hpo_version", "text", "codes"}
    assert payload["codes"][0].keys() >= {
        "hpo_id",
        "label",
        "spans",
        "score",
        "score_source",
        "context_flags",
    }


def test_to_tsv_header(annotator: Annotator) -> None:
    result = annotator.annotate("Patient has macrocephaly.")
    first_row = result.to_tsv().splitlines()[0]
    assert first_row == "hpo_id\tlabel\tscore\tscore_source\tspans\tcontext_flags"


def test_confidence_alias_deprecated(annotator: Annotator) -> None:
    code = annotator.annotate("Patient has macrocephaly.").codes[0]
    with pytest.warns(DeprecationWarning):
        assert code.confidence == code.score
    with pytest.warns(DeprecationWarning):
        assert code.confidence_source == code.score_source
