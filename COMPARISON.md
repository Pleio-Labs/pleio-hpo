# HPO extraction tools — pros, cons, and requirements

A practical comparison of the tools we benchmarked, beyond the F1 table. Accuracy
numbers are document-level micro-F1 on GSC+ (228 PubMed abstracts, human gold,
HPO `v2026-02-16`); see [`docs/RESULTS.md`](docs/RESULTS.md) for the full results and CIs.
Footprint/requirement facts come from each tool's own install and published sources
— not marketing claims.

## At a glance

| Tool | Approach | GSC+ F1 | Runs offline? | Hardware | License |
|---|---|---|---|---|---|
| **pleio-hpo (ours)** | Lexical (exact + order-free) + SapBERT + cross-encoder validator | **0.660** | **Yes, CPU** | ~1.5 GB RAM, CPU | Apache-2.0 |
| PhenoTagger | Dictionary + BioBERT/Bioformer (distant supervision) | 0.606 | Yes | TensorFlow; GPU recommended | Public domain (NCBI) |
| txt2hpo | Dictionary + scispaCy NLP | 0.556 | Yes, CPU | Light, CPU | MIT |
| Doc2HPO | Rule-based + lexical (+ optional UMLS/MetaMap) | 0.516 (acdat) | Web API, or local w/ UMLS | Light (API) / heavy (local) | Free for academic use |
| PhenoGPT v1 | LoRA-tuned Llama-2-7B + BioSent2Vec | 0.341\* | Yes (needs GPU) | GPU + ~22 GB BioSent2Vec | MIT |

All five tools can run locally (offline); they differ in **footprint**, not in
whether they phone home — the exception is Doc2HPO's hosted web API, which sends
text off-machine (its local install is offline but needs a UMLS license).

\* PhenoGPT trained on ~200/228 GSC+ docs (in-distribution); its published 0.83 does
not reproduce under a uniform document-level protocol (see [`docs/RESULTS.md`](docs/RESULTS.md)).

## Disk + memory footprint

| Tool | Model/data on disk | Peak RAM (inference) | Notes |
|---|---|---|---|
| **pleio-hpo** | ~580 MB (418 MB validator + 149 MB index + 10 MB HPO) | **~1.5 GB** | measured: cold start 7.9 s, 154 ms/paragraph (p50) on Apple-silicon CPU |
| PhenoTagger | BioBERT/Bioformer/PubMedBERT checkpoints (hundreds of MB each; pick one) | multi-GB w/ TensorFlow | TensorFlow 2.12 runtime is itself heavy |
| txt2hpo | scispaCy model + bundled HPO graph (~hundreds of MB) | modest (~hundreds of MB) | lightest real install |
| Doc2HPO | ~0 locally if using the hosted API; full UMLS (~30 GB+) if self-hosted | API: ~0; local: large | local deploy needs UMLS license + MetaMap |
| PhenoGPT | **~22 GB** BioSent2Vec + ~13 GB Llama-2-7B weights | GPU VRAM (7B model) | impractical without a GPU; we ran it on Modal |

## Per-tool pros and cons

### pleio-hpo (ours)
- **Pros:** highest GSC+ F1 of the tools measured; fully offline on CPU (PHI-safe);
  Apache-2.0; single `pip install`; honest, machine-verified numbers; context flags
  (negation/family/uncertain). Higher *precision* than PhenoTagger as well as recall.
- **Cons:** ~1.5 GB RAM and a ~580 MB one-time asset download; English-only; not
  clinically validated; part of the full-benchmark lead is covering inheritance/onset
  terms (on phenotype-only recognition pleio-hpo is *tied* with PhenoTagger, not ahead
  — see [`docs/RESULTS.md`](docs/RESULTS.md)).

### PhenoTagger (NCBI)
- **Pros:** strong, well-established "pre-LLM SOTA"; public domain; offline; multiple
  model backbones; BC8 baseline; mature.
- **Cons:** TensorFlow stack is heavy and version-fragile (we ran it amd64-only in
  Docker); GPU strongly recommended for throughput; negation/abbreviation features
  need Java (NegBio/Ab3P). Loses to pleio-hpo on GSC+ by 0.055 F1 (p<0.001); tied on
  BC8 and on GSC+ phenotype-only recognition.

### txt2hpo (GeneDx)
- **Pros:** the lightest, most Pythonic option; `pip install txt2hpo`; CPU; MIT;
  genuinely easy.
- **Cons:** lowest-but-respectable accuracy (0.556); dictionary/NLP approach misses
  paraphrases that embeddings catch; no published paper.

### Doc2HPO (WGLab)
- **Pros:** the tool most clinicians actually reach for; nice web UI; the hosted API
  is zero-install; multiple engines (acdat, ensemble).
- **Cons:** the accurate local mode needs a UMLS license + MetaMap (~24 h registration,
  multi-GB, ~1–2 day setup); the public API is a dependency you don't control (and
  sends text off-machine — not PHI-safe); lower F1 than the offline tools.

### PhenoGPT v1
- **Pros:** the "LLM fine-tuned for HPO" reference point; MIT.
- **Cons:** needs a GPU and ~22 GB of BioSent2Vec plus 7B model weights — by far the
  heaviest; not offline-practical on commodity hardware; the headline 0.83 is an
  in-distribution, non-reproducible number (0.341 under uniform protocol). Worst F1
  of the set here.

## Choosing

- **Need offline + PHI-safe + top-tier accuracy with the lightest install (pip, CPU, no GPU/TF):** pleio-hpo.
- **Want the established academic baseline and can run a TensorFlow stack (GPU optional — it runs on CPU too):** PhenoTagger.
- **Want the absolute lightest pip install and can accept lower recall:** txt2hpo.
- **Just need a quick web lookup, data can leave the machine:** Doc2HPO's hosted API.
- **Researching LLM-fine-tuning for HPO and have GPU infra:** PhenoGPT (but verify numbers yourself).

## Sources

- PhenoTagger: [Luo et al. 2021, *Bioinformatics*](https://academic.oup.com/bioinformatics/article/37/13/1884/6104813);
  [github.com/ncbi-nlp/PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger) (requirements, model downloads).
- Doc2HPO: [Liu et al. 2019](https://pubmed.ncbi.nlm.nih.gov/31114902/);
  [github.com/stormliucong/doc2hpo](https://github.com/stormliucong/doc2hpo).
- txt2hpo: [github.com/GeneDx/txt2hpo](https://github.com/GeneDx/txt2hpo).
- PhenoGPT: [Yang et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10801236/) (BioSent2Vec + Llama-2-7B).
- pleio-hpo footprint/latency figures are measured on Apple-silicon CPU (see [`docs/RESULTS.md`](docs/RESULTS.md)).
