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
  - Try non-linear probes (MLP) for quality → see EE-003
  - Investigate PCA/UMAP for topic clustering → see EE-003

### 2026-03-12 — MLP probe and dimensionality reduction (EE-003)
- Hypothesis: (a) MLP could push quality Spearman past 0.8; (b) PCA/UMAP could improve topic clustering
- Setup: Ran locally on cached v7 embeddings/oracle data (Fray Iris detection broken on current cluster)
- Config:
  - MLP architectures: (128,), (128, 64), (256, 128); early_stopping, max_iter=500
  - PCA: n_components=[8, 16, 32, 64, 128]
  - UMAP: n_components=[8, 16, 32]
- **Results**:

  **Quality MLP vs RidgeCV**:
  | Model | Spearman ρ | Kendall τ | R² |
  |---|---|---|---|
  | RidgeCV (v7) | **0.698** | **0.560** | **0.472** |
  | MLP (256, 128) | 0.603 | 0.473 | 0.366 |
  | MLP (128, 64) | 0.566 | 0.440 | 0.299 |
  | MLP (128,) | 0.512 | 0.390 | 0.217 |

  MLP performs **worse** than linear RidgeCV. With 800 train / 192 features, MLPs overfit despite early stopping. The linear probe captures the quality signal better.

  **Topic Clustering with Dimensionality Reduction**:
  | Method | d | ARI | NMI |
  |---|---|---|---|
  | Baseline | 192 | 0.242 | 0.439 |
  | PCA | 16 | 0.247 | **0.442** |
  | PCA | 8 | 0.214 | 0.408 |
  | PCA | 32 | 0.228 | 0.423 |
  | PCA | 64 | 0.212 | 0.426 |
  | PCA | 128 | 0.218 | 0.422 |
  | UMAP | 16 | 0.237 | 0.429 |
  | UMAP | 8 | 0.213 | 0.415 |
  | UMAP | 32 | 0.222 | 0.417 |

  Neither PCA nor UMAP improves clustering. Best (PCA d=16) gives NMI 0.442 vs baseline 0.439 — essentially flat. First 16 PCA components capture 39.3% of variance.

- **Conclusions**:
  - Quality: Linear probe (RidgeCV) is the right model for this embedding space. MLP hurts. Spearman 0.698 appears to be the ceiling for Luxical 192-dim on this task.
  - Topics: The NMI ~0.44 ceiling is robust across all dimensionality reduction methods. This is a fundamental limitation of Luxical embeddings for topic separation, not a noise/dimensionality issue.
  - The quality probe verdict remains **"investigate"** (0.6-0.8). Improving further likely requires higher-dimensional or task-specialized embeddings, not better probes.
- Next actions:
  - Compare with higher-dimensional embeddings → see EE-004

### 2026-03-12 — Model comparison: Luxical vs Arctic vs BGE-large (EE-004)
- Hypothesis: Higher-dimensional embeddings (1024-dim) will outperform Luxical (192-dim) on both quality and topics
- Setup: Ran locally on cached v7 sample/oracle data. Embedded with three models, same RidgeCV and K-Means eval.
- Models:
  - Luxical-One: `DatologyAI/luxical-one` (192-dim)
  - Arctic-L: `Snowflake/snowflake-arctic-embed-l` (1024-dim, 335M params)
  - BGE-large: `BAAI/bge-large-en-v1.5` (1024-dim, 326M params)
- **Results**:

  **Quality Probe (RidgeCV)**:
  | Model | Dim | Spearman ρ | Kendall τ | R² | MSE |
  |---|---|---|---|---|---|
  | **Luxical** | 192 | **0.698** | **0.560** | **0.472** | **0.601** |
  | Arctic-L | 1024 | 0.621 | 0.493 | 0.336 | 0.755 |
  | BGE-large | 1024 | 0.510 | 0.391 | 0.150 | 0.967 |

  **Topic Clustering (K-Means, k=15)**:
  | Model | Dim | ARI | NMI | V-measure |
  |---|---|---|---|---|
  | Luxical | 192 | 0.242 | 0.439 | 0.439 |
  | Arctic-L | 1024 | 0.254 | 0.443 | 0.443 |
  | **BGE-large** | 1024 | **0.277** | **0.478** | **0.478** |

- **Key findings**:
  - Quality: **Luxical wins decisively** (Spearman 0.698 vs Arctic 0.621 vs BGE 0.510). Higher dimensionality does NOT help — in fact it hurts, likely because RidgeCV overfits on 1024 features with only 800 training samples. Luxical's compact 192-dim space is better regularized for linear probes.
  - Topics: **BGE-large edges ahead** (NMI 0.478 vs Luxical 0.439 vs Arctic 0.443). BGE-large's NMI 0.478 approaches the 0.5 threshold but still doesn't clear it. Arctic barely improves over Luxical.
  - The quality ranking (Luxical > Arctic > BGE) is the inverse of the topic ranking (BGE > Arctic > Luxical), suggesting these models encode different kinds of information. Luxical appears optimized for document quality signals; BGE for semantic/topical similarity.

- **Final verdict on stop criteria**:
  - Quality: Luxical RidgeCV Spearman **0.698** is the best result across all models/probes tested. Verdict: **investigate further** — promising but not yet go.
  - Topics: Best model (BGE-large) achieves NMI **0.478**, ARI **0.277** — closer to thresholds but still **no-go**.
  - Overall: Embedding-based quality filtering with Luxical + RidgeCV is the most promising direction. Topic clustering remains a hard problem for general-purpose embeddings.

- Next actions:
  - Try larger training sets → see EE-005

### 2026-03-12 — Quality probe scaling (EE-005)
- Hypothesis: The Spearman 0.698 ceiling is due to insufficient training data (N=800)
- Setup: Ran locally. (A) Subsampled existing oracle data for a learning curve; (B) sampled fresh Nemotron docs at scale, used bucket ordinals (0-4) as targets.
- **Results**:

  **Part A: Learning curve (oracle scores, existing 1000 docs)**:
  | N_train | Spearman ρ | Kendall τ | R² |
  |---|---|---|---|
  | 50 | 0.486 | 0.377 | 0.216 |
  | 100 | 0.496 | 0.385 | 0.262 |
  | 200 | 0.630 | 0.492 | 0.381 |
  | 400 | 0.639 | 0.506 | 0.397 |
  | 600 | 0.664 | 0.522 | 0.422 |
  | 800 | 0.698 | 0.560 | 0.472 |

  Curve still climbing at N=800 — not saturated. Biggest jump: 100→200 (+0.134).

  **Part B: Large-scale (bucket ordinals, freshly sampled)**:
  | N_total | N_train | Spearman ρ | Kendall τ | R² |
  |---|---|---|---|---|
  | 2,500 | 2,000 | 0.749 | 0.594 | 0.547 |
  | 5,000 | 4,000 | 0.740 | 0.584 | 0.531 |
  | 10,000 | 8,000 | **0.750** | **0.595** | **0.548** |

  Plateau at Spearman ~0.75 with bucket ordinals. Doubling from 2K→8K train makes no difference.

- **Conclusions**:
  - The Spearman ~0.75 ceiling is a property of the Luxical embedding space, not a data limitation.
  - More data does help up to ~2000 training samples, then plateaus.
  - Oracle scores (continuous 0-5) have slightly lower Spearman than bucket ordinals because they add noise from the LLM oracle.
  - **Updated verdict**: Quality Spearman 0.75 falls squarely in the "investigate" zone (0.6-0.8). The embedding captures meaningful quality signal but cannot fully recover the Nemotron quality ranking with a linear probe.

- **Final summary of all experiments**:
  | ID | What | Quality Spearman | Topic NMI |
  |---|---|---|---|
  | EE-001 | Luxical, N=125 | 0.485 | 0.463 |
  | EE-002 | Luxical, N=1000 | 0.698 | 0.439 |
  | EE-003 | MLP probe | 0.603 (worse) | — |
  | EE-003 | PCA/UMAP | — | 0.442 (flat) |
  | EE-004 | Arctic 1024-dim | 0.621 | 0.443 |
  | EE-004 | BGE-large 1024-dim | 0.510 | **0.478** |
  | EE-005 | Luxical, N=10000, bucket ordinals | **0.750** | — |

  Quality: Luxical + RidgeCV peaks at Spearman **0.75** with sufficient data.
  Topics: BGE-large peaks at NMI **0.478**, still below 0.5 threshold.
