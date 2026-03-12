# Embed Everything: Research Logbook

## Scope
- Goal: Evaluate Luxical-One embeddings for quality classification and topic clustering in the Marin pipeline
- Primary metric(s):
  - Quality: Spearman rho and Kendall tau between linear probe predictions and Claude oracle scores
  - Topics: ARI and NMI between K-Means clusters and Claude oracle labels
- Constraints: CPU-only execution, ~125 quality docs (25 per bucket), ~375 topic docs (25 per source)
- Stop criteria:
  - Quality Spearman > 0.8 -> go; < 0.6 -> no-go; 0.6-0.8 -> investigate embeddings vs oracle
  - Topics ARI > 0.4 or NMI > 0.5 -> go; both below -> no-go

## Links
- Parent issue: https://github.com/marin-community/marin/issues/3049
- Experiment issue: https://github.com/marin-community/marin/issues/3535
- Branch: `rav-emb-everything-research-1`

## Key References
- Su 2024 (Nemotron-CC): linear classifiers on Arctic embeddings for quality filtering
- Penedo 2024 (FineWeb-Edu): quality classifier approach
- Tirumala 2023 (D4): embedding-based curation, K-Means topic recovery
- Li 2024 (DCLM): linear on BGE performed poorly (red flag)
- Wettig 2025 (WebOrganizer): K-Means k=24 got NMI=0.46 on topics
- Luxical blog: https://www.datologyai.com/blog/introducing-luxical-embeddings

## Baseline
- Date: 2026-03-11
- Code refs: `experiments/embed_everything/exp3049_embed_eval.py`
- Baseline numbers: No existing embedding infrastructure in Marin. First evaluation.

## Experiment Log

### 2026-03-11 — Initial setup (EE-001)
- Hypothesis: Luxical-One embeddings encode sufficient quality and topic signal for linear methods
- Setup:
  - Created experiment code in `experiments/embed_everything/`
  - Modules: `sample.py`, `oracle.py`, `embed.py`, `evaluate.py`, `exp3049_embed_eval.py`
  - StepSpec DAG: sample -> {oracle, embed} -> evaluate
  - All steps use `@remote` for CPU execution via Iris
  - Added `luxical` and `sentence-transformers` as optional deps under `[embed]` group
  - Added `anthropic` as optional dep under `[oracle]` group
  - Temp storage: `gs://marin-tmp-*/ttl=7d/embed-everything/`
- Config:
  - Quality: 25 docs x 5 Nemotron buckets = 125 docs, 80/20 train/test
  - Topics: 25 docs x 15 Dolma sources = 375 docs
  - Oracle: Claude (claude-sonnet-4-20250514), FineWeb-Edu style 0-5 rubric
  - Embedding: DatologyAI/luxical-one via sentence-transformers (192-dim)
  - Eval: RidgeCV for quality, K-Means (k=15) for topics
- Bugs fixed during execution:
  - `read_dataset_streaming` doesn't support `.json.gz` (Dolma format); replaced with direct `gzip.GzipFile` streaming
  - Nemotron files under `kind2=actual/` subdirectory, not directly in `kind=actual/`
  - Recursive `**/*.jsonl.gz` glob on GCS too slow; switched to flat `*.jsonl.gz` with MAX_FILES_PER_STRATUM=3
  - `anthropic` not installed on oracle workers; added `oracle` pip_dependency_group
  - Early stop after 10x target samples to avoid OOM on multi-GiB gzipped files
- Iris jobs: v3-v5 failed; v6 (`/rav/iris-run-exp3049_embed_eval-20260312-001713`) succeeded
- **Results (v6)**:

  **Quality Probe (RidgeCV on 192-dim embeddings)**:
  | Metric | Value |
  |---|---|
  | Spearman rho | 0.485 (p=0.014) |
  | Kendall tau | 0.365 (p=0.018) |
  | Bucket ordinal Spearman | 0.388 |
  | R^2 | 0.221 |
  | MSE | 1.079 |
  | Ridge alpha | 100.0 |
  | Train/Test | 100 / 25 |

  **Topic Clustering (K-Means, k=15)**:
  | Metric | Value |
  |---|---|
  | ARI | 0.227 |
  | NMI | 0.463 |
  | Homogeneity | 0.477 |
  | Completeness | 0.449 |
  | V-measure | 0.463 |
  | Documents | 375 (15 true clusters) |

- **Verdict against stop criteria**:
  - Quality: Spearman 0.485 < 0.6 -> **no-go** for quality filtering with linear probe
  - Topics: NMI 0.463 close to 0.5 threshold, ARI 0.227 < 0.4 -> **borderline no-go** for topic clustering
  - Context: NMI 0.463 matches WebOrganizer's 0.46 baseline, suggesting embeddings capture topic structure, but ARI is weak
- Next actions:
  - Consider larger sample sizes (current N is small, test set only 25 for quality)
  - Try non-linear probes (MLP) for quality
  - Compare with higher-dimensional embeddings (e.g., Arctic, BGE-large)
  - Report findings in issue #3535
