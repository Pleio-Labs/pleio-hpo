# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.1] — 2026-06-08

### Fixed
- `pleio-hpo download` now also installs the required `en_core_web_sm` spaCy
  model, and a missing model raises a clear "run `pleio-hpo download`" message
  instead of a raw spaCy `OSError`. Previously a fresh install could fetch the
  Hub assets but still fail on the first `annotate()` call.
- `__version__` (and the `annotator_version` field in results) is read from the
  installed package metadata instead of a hard-coded string, so it no longer
  reports a stale `0.0.1.dev0`.

### Added
- `pleio-hpo info` reports the spaCy model status alongside the index/validator.

## [0.1.0] — 2026-06-08

Initial public release.

### Added
- Offline, CPU-only HPO concept extraction: a three-stage pipeline of lexical
  matching (exact + morphology-aware + order-free token-set), SapBERT embedding
  nearest-neighbour, and a fine-tuned PubMedBERT cross-encoder validator.
- `Annotator` Python API and `pleio-hpo` CLI (annotate text/file/stdin; `info`
  and `download` subcommands), with negation / family-history / uncertainty
  context filtering.
- `pleio-hpo download` fetches the validator + embedding index from the Hugging
  Face Hub (~570 MB, one time); the HPO ontology ships in the wheel.
- Evaluation utilities (`pleio_hpo.eval`): document/span micro-PRF, symmetric
  DAG-1 F1, percentile bootstrap CIs, and paired-permutation significance.
- Condensed results across three human-gold corpora in [`docs/RESULTS.md`](docs/RESULTS.md)
  and a per-tool comparison in [`COMPARISON.md`](COMPARISON.md).
