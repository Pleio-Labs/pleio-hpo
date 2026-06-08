"""05 — Custom thresholds. Tune sensitivity vs precision.

    python examples/05-custom-thresholds.py

Two knobs trade recall for precision:
  embedding_threshold (default 0.8) — min cosine similarity for an embedding
      match. Lower = more candidates (higher recall, lower precision).
  validator_threshold (default 0.5) — min validator probability to keep an
      embedding hit. Lower = keep more borderline hits.
You can also disable the validator entirely with use_validator=False.

Thresholds are set per-Annotator, so build one per configuration you want to
compare.
"""

from __future__ import annotations

from pleio_hpo import Annotator

text = "The child shows facial dysmorphism and mildly delayed motor milestones."


def run(label: str, **kwargs: object) -> None:
    annotator = Annotator(**kwargs)  # type: ignore[arg-type]
    codes = annotator.annotate(text).codes
    summary = ", ".join(f"{c.label} ({c.score:.2f})" for c in codes)
    print(f"{label}:\n  {summary or '(none)'}\n")


run("default (emb>=0.8, validator>=0.5)")
run("higher precision (emb>=0.9)", embedding_threshold=0.9)
run("higher recall (emb>=0.7, validator>=0.3)", embedding_threshold=0.7, validator_threshold=0.3)
run("lexical only (validator + embedding gate off)", use_validator=False)
