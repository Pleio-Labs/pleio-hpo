"""End-to-end smoke test.

Runs the full Annotator on a tiny frozen subset of the consult eval and asserts
exact-match prediction sets. Any pipeline change that perturbs output trips
this. The fixture was snapshotted from the pipeline via the default flags.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pleio_hpo import Annotator
from pleio_hpo.embed_match import default_index_path
from pleio_hpo.validator import default_validator_path

pytestmark = pytest.mark.skipif(
    not (default_index_path().exists() and default_validator_path().exists()),
    reason="requires bundled embedding index + validator checkpoint (built locally / via download)",
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "eval_smoke" / "docs.json"


@pytest.fixture(scope="module")
def annotator() -> Annotator:
    return Annotator()


def _cases() -> list[dict]:
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["doc_id"])
def test_prediction_set_matches_fixture(case: dict, annotator: Annotator) -> None:
    result = annotator.annotate(case["text"])
    predicted = sorted({c.hpo_id for c in result.codes})
    assert predicted == case["expected_hpo_ids"]
