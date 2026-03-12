# Embed Everything: Research Logbook

## Scope
- Goal: Evaluate Luxical-One embeddings for quality classification and topic clustering in the Marin pipeline
- Primary metric(s):
  - Quality: Spearman rho and Kendall tau between linear probe predictions and Claude oracle scores
  - Topics: ARI and NMI between K-Means clusters and Claude oracle labels
- Constraints: CPU-only execution
- Sample sizes: v6: 125 quality / 375 topic; v7: 1000 quality / 1005 topic
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
  - Scale up to ~1000 docs per problem to test whether v6 was underpowered → see EE-002

### 2026-03-11 — Scale-up to ~1000 docs (EE-002)
- Hypothesis: v6 quality results (Spearman 0.485) were underpowered due to small test set (n=25)
- Setup: Same pipeline, larger sample sizes
- Config:
  - Quality: 200 docs x 5 buckets = 1000 docs, 800 train / 200 test
  - Topics: 67 docs x 15 sources = 1005 docs
  - All other settings unchanged from EE-001
- Iris job: v7 (`/rav/iris-run-exp3049_embed_eval-20260312-005457`) — all 8 steps succeeded
- **Results (v7)**:

  **Quality Probe (RidgeCV on 192-dim embeddings)**:
  | Metric | v6 (N=125) | v7 (N=1000) | Change |
  |---|---|---|---|
  | Spearman rho | 0.485 (p=0.014) | **0.698** (p=1.5e-30) | +0.213 |
  | Kendall tau | 0.365 (p=0.018) | **0.560** (p=7.2e-26) | +0.195 |
  | Bucket ordinal Spearman | 0.388 | **0.455** | +0.067 |
  | R² | 0.221 | **0.472** | +0.251 |
  | MSE | 1.079 | **0.601** | -0.478 |
  | Ridge alpha | 100.0 | 100.0 | — |
  | Train/Test | 100/25 | 800/200 | — |

  **Topic Clustering (K-Means, k=15)**:
  | Metric | v6 (N=375) | v7 (N=1005) | Change |
  |---|---|---|---|
  | ARI | 0.227 | **0.242** | +0.015 |
  | NMI | 0.463 | **0.439** | -0.024 |
  | Homogeneity | 0.477 | **0.449** | -0.028 |
  | Completeness | 0.449 | **0.429** | -0.020 |
  | V-measure | 0.463 | **0.439** | -0.024 |
  | Documents | 375 | 1005 | — |

  Oracle topic label distribution (v7): computer_science=148, cc/news=134, web_forum=100, code=88, natural_science=88, reference=85, business=61, mathematics=60, humanities=59, medicine=55, creative_writing=52, engineering=28, social_science=20, other=15, law=12

- **Verdict against stop criteria**:
  - Quality: Spearman **0.698** is in 0.6-0.8 "investigate" zone (major improvement from 0.485). v6 was indeed underpowered. A non-linear probe (MLP) or more training data might push this above 0.8.
  - Topics: NMI **0.439** < 0.5 and ARI **0.242** < 0.4 → **no-go**. Results barely changed with 2.7x more data, confirming this is a real signal ceiling. Highly imbalanced oracle labels (148 vs 12) may contribute.
  - Key insight: Quality signal improves substantially with more data; topic signal does not. The quality probe is promising for further investigation.
- Next actions:
  - Try non-linear probes (MLP) for quality — could push Spearman > 0.8
  - Compare with higher-dimensional embeddings (e.g., Arctic, BGE-large)
  - Investigate whether topic clustering improves with PCA or UMAP dimensionality reduction
  - Report findings in issue #3535
