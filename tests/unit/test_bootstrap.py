"""Unit tests for eval/bootstrap.py."""

from __future__ import annotations

import pytest

from pleio_hpo.eval.bootstrap import bootstrap_f1, paired_bootstrap_delta


def test_identical_preds_tight_ci() -> None:
    preds = [{"HP:1"}, {"HP:2"}, {"HP:1", "HP:3"}, {"HP:4"}]
    golds = [set(s) for s in preds]
    res = bootstrap_f1(preds, golds, n_iter=2000)
    assert res["f1"] == pytest.approx(1.0)
    assert res["ci_hi"] - res["ci_lo"] < 0.001


def test_paired_delta_no_difference() -> None:
    preds = [{"HP:1"}, {"HP:2"}, {"HP:3"}]
    golds = [{"HP:1"}, {"HP:2"}, {"HP:9"}]
    res = paired_bootstrap_delta(preds, preds, golds, n_iter=2000)
    assert res["delta_f1"] == pytest.approx(0.0)
    assert res["p_value"] >= 0.05


def test_paired_delta_large_difference() -> None:
    n = 20
    golds = [{f"HP:{i}"} for i in range(n)]
    perfect = [set(g) for g in golds]
    empty: list[set[str]] = [set() for _ in range(n)]
    res = paired_bootstrap_delta(perfect, empty, golds, n_iter=2000)
    assert res["delta_f1"] == pytest.approx(1.0)
    assert res["p_value"] <= 0.001
