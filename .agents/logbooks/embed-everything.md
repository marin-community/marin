# Embed Everything: Research Logbook

## Scope
- Goal: Evaluate Luxical-One embeddings for quality classification and topic clustering in the Marin pipeline
- Primary metric(s):
  - Quality: Spearman ρ and Kendall τ between linear probe predictions and Claude oracle scores
  - Topics: ARI and NMI between K-Means clusters and Claude oracle labels
- Constraints: CPU-only execution, ~125 quality docs (25 per bucket), ~375 topic docs (25 per source)
- Stop criteria:
  - Quality Spearman > 0.8 → go; < 0.6 → no-go; 0.6-0.8 → investigate embeddings vs oracle
  - Topics ARI > 0.4 or NMI > 0.5 → go; both below → no-go

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
  - ExecutorStep DAG: sample → {oracle, embed} → evaluate
  - All steps use `@remote` for CPU execution via Iris
  - Added `luxical` and `sentence-transformers` as optional deps under `[embed]` group
  - Temp storage: `gs://marin-tmp-*/ttl=7d/embed-everything/`
- Config:
  - Quality: 25 docs × 5 Nemotron buckets = 125 docs, 80/20 train/test
  - Topics: 25 docs × 15 Dolma sources = 375 docs
  - Oracle: Claude (claude-sonnet-4-20250514), FineWeb-Edu style 0-5 rubric
  - Embedding: DatologyAI/luxical-one via sentence-transformers
  - Eval: RidgeCV for quality, K-Means (k=15) for topics
- Result: Code written, not yet executed
- Next action: Run lint, install luxical, execute pipeline
