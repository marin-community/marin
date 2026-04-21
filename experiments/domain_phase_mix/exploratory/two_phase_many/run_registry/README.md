# Two-Phase-Many Run Registry

This directory is the canonical provenance layer for two-phase-many experiments.

Files:
- `logical_runs.csv`: one row per conceptual run
- `run_attempts.csv`: one row per discovered checkpoint-backed attempt
- `live_watchlist.csv`: current parent jobs we are actively babysitting
- `strong_tier_child_jobs.csv`: raw Iris child-job snapshot for the tracked strong-tier parent-job lineage
- `strong_tier_perplexity_ready.csv`: strong-tier rows that reached the target checkpoint and wrote perplexity evals
- `summary.json`: aggregate counts for quick handoff checks

Design:
- `logical_runs.csv` is the table analysis code should join against first
- `run_attempts.csv` preserves retries, failures, and superseded attempts
- `canonical_attempt_root` is the operational snapshot used for current job state
- `checkpoint_root` is the best analysis attempt, which may differ if an older attempt finished training and wrote evals while a newer retry is still running
- W&B is treated as a convenient mirror, not the sole source of truth
- checkpoint-backed artifacts and executor status remain authoritative

Backfill policy:
- missing metrics should be backfilled from authoritative artifacts with an idempotent script
- if metrics are pushed to W&B later, record them as backfilled rather than pretending they were original

Operational notes:
- `live_watchlist.csv` is best-effort and slower because each Iris status query establishes a controller tunnel
- `strong_tier_child_jobs.csv` is a raw operational snapshot across the tracked
  parent-job lineage; use `logical_runs.csv` for analysis joins
- for a fast deterministic refresh, use `--no-include-live-status`
