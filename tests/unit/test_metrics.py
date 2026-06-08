"""Hand-computed unit tests for eval/metrics.py.

Seven core cases, plus one extra (test 7b) that exercises the predict-parent /
gold-child direction an ancestor-only metric would miss.
"""

from __future__ import annotations

import pytest

from pleio_hpo.eval.metrics import (
    Annotation,
    ancestor_overlap_f1,
    document_micro_prf,
    span_micro_prf,
)


def test_document_prf_subset() -> None:
    # pred {1} vs gold {1,2}: tp=1, fp=0, fn=1
    p, r, f1 = document_micro_prf([{"HP:1"}], [{"HP:1", "HP:2"}])
    assert (p, r, f1) == pytest.approx((1.0, 0.5, 2 / 3))


def test_document_prf_multi_doc() -> None:
    # tp=2, fp=1, fn=1 -> P=R=F1=2/3
    p, r, f1 = document_micro_prf(
        [{"HP:1", "HP:2"}, {"HP:3"}],
        [{"HP:1"}, {"HP:3", "HP:4"}],
    )
    assert (p, r, f1) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_document_prf_both_empty() -> None:
    assert document_micro_prf([set()], [set()]) == (0.0, 0.0, 0.0)


def test_document_prf_pred_only() -> None:
    assert document_micro_prf([{"HP:1"}], [set()]) == (0.0, 0.0, 0.0)


def test_span_exact() -> None:
    # exact match, HPO mismatch, offset mismatch -> tp=1, fp=2, fn=2
    preds = [[
        Annotation(0, 5, "alpha", "HP:1"),    # exact match
        Annotation(10, 15, "beta", "HP:2"),   # HPO mismatch vs gold's HP:9
        Annotation(20, 25, "gamma", "HP:3"),  # offset mismatch vs gold's 21-25
    ]]
    golds = [[
        Annotation(0, 5, "alpha", "HP:1"),
        Annotation(10, 15, "beta", "HP:9"),
        Annotation(21, 25, "gamma", "HP:3"),
    ]]
    p, r, f1 = span_micro_prf(preds, golds, mode="exact")
    assert (p, r, f1) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_span_overlap_accepts_partial() -> None:
    # Same offset-mismatched pair as above, but overlap mode accepts it.
    preds = [[Annotation(20, 25, "gamma", "HP:3")]]
    golds = [[Annotation(21, 25, "gamma", "HP:3")]]
    assert span_micro_prf(preds, golds, mode="exact") == pytest.approx((0.0, 0.0, 0.0))
    assert span_micro_prf(preds, golds, mode="overlap") == pytest.approx((1.0, 1.0, 1.0))


# Tiny synthetic DAG: root -> child -> grandchild
PARENT_MAP = {"child": ["root"], "grandchild": ["child"]}


def test_dag1_child_predicted_for_parent_gold() -> None:
    # pred=child, gold=parent(root): within 1 hop -> F1=1.0; distance=0 -> 0.0
    f1_d1 = ancestor_overlap_f1([{"child"}], [{"root"}], PARENT_MAP, distance=1)
    f1_d0 = ancestor_overlap_f1([{"child"}], [{"root"}], PARENT_MAP, distance=0)
    assert f1_d1 == pytest.approx(1.0)
    assert f1_d0 == pytest.approx(0.0)


def test_dag1_parent_predicted_for_child_gold() -> None:
    # The direction the original (ancestor-only) spec would score 0.
    f1 = ancestor_overlap_f1([{"root"}], [{"child"}], PARENT_MAP, distance=1)
    assert f1 == pytest.approx(1.0)
