# pleio-hpo

Offline extraction of [Human Phenotype Ontology (HPO)](https://hpo.jax.org/)
codes from free-text clinical notes. Runs fully offline on CPU — no data leaves
the machine, which makes it usable with PHI.

A three-stage pipeline: lexical matching (Aho-Corasick over HPO labels +
synonyms, with morphology-aware and order-free token-set matching for plural and
reordered phrasings) → biomedical-embedding nearest-neighbour (SapBERT) for
paraphrases → a PubMedBERT cross-encoder validator that filters the fuzzy
candidates.

## Install

```bash
pip install pleio-hpo            # Python >=3.10
python -m spacy download en_core_web_sm
pleio-hpo download              # fetch the embedding index + validator (~570 MB, one time)
```

The large model assets are fetched on first use, not bundled in the wheel.
Runtime needs roughly **1.2 GB RAM** (SapBERT + the PubMedBERT validator on CPU).

## Quickstart — CLI

```bash
$ pleio-hpo "The patient has macrocephaly and hypotonia."
HP:0000256  Macrocephaly                             1.00
HP:0001252  Hypotonia                                1.00

# TSV / JSON, from a file or stdin, with context control:
pleio-hpo "Bilateral hearing loss; no microcephaly." --format tsv --include-negated
pleio-hpo --file note.txt --format json --output codes.json
echo "Short stature and seizures." | pleio-hpo --format tsv
```

`pleio-hpo info` shows the version, pinned HPO release, and asset paths.

## Quickstart — Python

```python
from pleio_hpo import Annotator

annotator = Annotator()  # construct once, reuse (models load lazily on first call)
result = annotator.annotate("Global developmental delay and seizures.")

for code in result.codes:
    print(code.hpo_id, code.label, code.score, code.score_source)

print(result.to_json())
```

`code.score` is an opaque per-source strength signal (1.0 for an exact lexical
match, cosine similarity for an embedding match) — **not** a calibrated
probability. See [`examples/`](examples/) for batch processing, context
filtering, and threshold tuning.

## Evaluation

On **GSC+** (228 PubMed abstracts, community **human** gold), under a uniform
document-level micro-F1 protocol (HPO `v2026-02-16`):

| Tool | GSC+ F1 | Offline? | Footprint |
|---|---|---|---|
| **pleio-hpo** | **0.660** | yes | pip, CPU, ~1.5 GB |
| PhenoTagger | 0.606 | yes | TensorFlow; GPU optional |
| txt2hpo | 0.556 | yes | pip, CPU |
| Doc2HPO (acdat) | 0.516 | no¹ | web API, or local + UMLS |
| PhenoGPT | 0.341 | yes | GPU, ~22 GB |

¹ The Doc2HPO numbers use its public web API (text leaves the machine); a local,
PHI-safe install needs a UMLS license.

pleio-hpo tops the full benchmark, ahead of PhenoTagger by 0.055 F1 (p<0.001). Part
of that margin is inheritance/onset coverage the benchmark scores; on phenotype-only
recognition the two are level (0.651 vs 0.647, p=0.67). Across three human-gold
corpora — GSC+, BC8, and 112 out-of-distribution genetics case reports — pleio-hpo
and PhenoTagger are statistically tied (the nominal leader varies by corpus), and the
strongest frontier-LLM extractor performs comparably. The corpora are abstracts, exam
observations, and case reports — not free-text clinical notes.

Condensed results, figures, ablation, and caveats: [`docs/RESULTS.md`](docs/RESULTS.md);
per-tool comparison: [`COMPARISON.md`](COMPARISON.md).

## HPO version

The **library runtime tracks the latest HPO release** (the index + validator are
built against the current `hp.obo`). The **reported evaluation is pinned to
`v2026-02-16`** for comparability, so a user's runtime ontology may be newer than
the evaluation snapshot ([`docs/RESULTS.md`](docs/RESULTS.md)).

## Scope and disclaimer

This tool is **decision-support, not clinical decision-making**. It is not
FDA-cleared and not for autonomous use in patient care; inferred codes may be
incorrect and should be verified by a qualified user. English-only. HPO is a
Western/English-language ontology — apply cautiously to other populations.

## License & citation

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). If you use this
work, please cite it — see [`CITATION.cff`](CITATION.cff).
