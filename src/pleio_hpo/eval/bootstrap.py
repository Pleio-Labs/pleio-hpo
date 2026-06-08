"""Bootstrap confidence intervals and paired-significance tests.

The resampling unit is the **document**, not the annotation, because annotations
within a document are correlated.

- ``bootstrap_f1``: percentile bootstrap CI on document-level micro F1.
- ``paired_bootstrap_delta``: CI on the F1 delta between two methods scored on
  the same documents (paired bootstrap), plus a two-sided p-value from a
  label-permutation test.
"""

from __future__ import annotations

import numpy as np

from pleio_hpo.eval.metrics import document_micro_prf


def bootstrap_f1(
    predictions: list[set[str]],
    golds: list[set[str]],
    n_iter: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Return ``{"f1", "ci_lo", "ci_hi"}`` (2.5/97.5 percentile CI)."""
    n = len(predictions)
    point = document_micro_prf(predictions, golds)[2]
    rng = np.random.default_rng(seed)
    samples = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        samples[i] = document_micro_prf(
            [predictions[j] for j in idx], [golds[j] for j in idx]
        )[2]
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"f1": float(point), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_bootstrap_delta(
    preds_a: list[set[str]],
    preds_b: list[set[str]],
    golds: list[set[str]],
    n_iter: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Return ``{"delta_f1", "ci_lo", "ci_hi", "p_value"}`` for F1(a) - F1(b).

    ``delta_f1`` is the observed difference. The CI comes from a paired
    document-level bootstrap; ``p_value`` is two-sided, from permuting the
    method label (a vs b) independently within each document.
    """
    n = len(golds)
    point = document_micro_prf(preds_a, golds)[2] - document_micro_prf(preds_b, golds)[2]
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        resampled_gold = [golds[j] for j in idx]
        f1_a = document_micro_prf([preds_a[j] for j in idx], resampled_gold)[2]
        f1_b = document_micro_prf([preds_b[j] for j in idx], resampled_gold)[2]
        deltas[i] = f1_a - f1_b
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    observed = abs(point)
    at_least_as_extreme = 0
    for _ in range(n_iter):
        swap = rng.random(n) < 0.5
        perm_a = [preds_b[j] if swap[j] else preds_a[j] for j in range(n)]
        perm_b = [preds_a[j] if swap[j] else preds_b[j] for j in range(n)]
        delta = document_micro_prf(perm_a, golds)[2] - document_micro_prf(perm_b, golds)[2]
        if abs(delta) >= observed - 1e-12:
            at_least_as_extreme += 1
    p_value = (at_least_as_extreme + 1) / (n_iter + 1)

    return {
        "delta_f1": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_value": float(p_value),
    }
