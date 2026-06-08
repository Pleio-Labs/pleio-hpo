"""Scoring metrics: document micro-PRF, span micro-PRF, DAG-1 F1.

DAG-1 note: rather than the asymmetric "credit if a gold term is the prediction's
ancestor" rule, we implement a **symmetric** neighbourhood — a prediction matches
if a gold term is within ``distance`` hops in either direction (ancestor or
descendant). The ``parent_map`` argument is inverted internally to obtain children.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Annotation:
    """A single span annotation, used by :func:`span_micro_prf` and the corpus
    loaders. ``negated`` is populated by datasets that record it (e.g. BC8);
    it does not affect document- or span-level scoring."""

    start: int
    end: int
    text: str
    hpo_id: str
    negated: bool = False


def document_micro_prf(
    predictions: list[set[str]],
    golds: list[set[str]],
) -> tuple[float, float, float]:
    """Micro-averaged P/R/F1 over HPO IDs at the document level.

    For each (pred_set, gold_set) pair, accumulate tp/fp/fn, then micro-average.
    Empty denominators yield 0.0 for that component (no raise).
    """
    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds, strict=True):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def span_micro_prf(
    predictions: list[list[Annotation]],
    golds: list[list[Annotation]],
    mode: Literal["exact", "overlap"] = "exact",
) -> tuple[float, float, float]:
    """Span-level micro P/R/F1.

    A predicted annotation matches an as-yet-unmatched gold annotation when the
    HPO IDs are equal and, depending on ``mode``:
      - ``exact``:   start and end are identical;
      - ``overlap``: the spans overlap (``p.start < g.end and p.end > g.start``).
    """
    tp = fp = fn = 0
    for pred_list, gold_list in zip(predictions, golds, strict=True):
        matched = [False] * len(gold_list)
        for pred in pred_list:
            hit = False
            for i, gold in enumerate(gold_list):
                if matched[i] or pred.hpo_id != gold.hpo_id:
                    continue
                if mode == "exact":
                    ok = pred.start == gold.start and pred.end == gold.end
                else:
                    ok = pred.start < gold.end and pred.end > gold.start
                if ok:
                    matched[i] = True
                    hit = True
                    break
            tp += 1 if hit else 0
            fp += 0 if hit else 1
        fn += matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _neighbourhood(
    node: str,
    parent_map: dict[str, list[str]],
    child_map: dict[str, list[str]],
    distance: int,
) -> set[str]:
    """BFS up (parents) and down (children) from ``node``, up to ``distance``
    hops, including ``node`` itself."""
    seen = {node}
    frontier: deque[tuple[str, int]] = deque([(node, 0)])
    while frontier:
        current, depth = frontier.popleft()
        if depth >= distance:
            continue
        for neighbour in (*parent_map.get(current, ()), *child_map.get(current, ())):
            if neighbour not in seen:
                seen.add(neighbour)
                frontier.append((neighbour, depth + 1))
    return seen


def ancestor_overlap_f1(
    predictions: list[set[str]],
    golds: list[set[str]],
    parent_map: dict[str, list[str]],
    distance: int = 1,
) -> float:
    """DAG-aware F1 with symmetric partial credit.

    A predicted term scores a true positive if any gold term lies within
    ``distance`` DAG hops of it (ancestor or descendant). Gold terms never
    reached by any prediction's neighbourhood are false negatives.
    """
    child_map: dict[str, list[str]] = {}
    for child, parents in parent_map.items():
        for parent in parents:
            child_map.setdefault(parent, []).append(child)

    tp = fp = fn = 0
    for pred, gold in zip(predictions, golds, strict=True):
        matched_gold: set[str] = set()
        for term in pred:
            overlap = _neighbourhood(term, parent_map, child_map, distance) & gold
            if overlap:
                tp += 1
                matched_gold |= overlap
            else:
                fp += 1
        fn += len(gold - matched_gold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
