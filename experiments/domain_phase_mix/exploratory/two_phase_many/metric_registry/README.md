# Two-Phase-Many Metric Registry

This directory is the canonical metric/provenance layer for two-phase-many
analysis. It exists because W&B summaries, local collector CSVs, GCS rerun
artifacts, and manually patched values have drifted apart.

## Files

- `build_metric_registry.py`: rebuilds all registry outputs.
- `materialize_fit_dataset.py`: writes a fit-ready CSV for one objective metric.
- `backfill_wandb_from_registry.py`: dry-run-first W&B summary backfill tool.
- `runs.csv`: one row per logical run key, including phase weights when known.
- `eval_artifacts.csv`: one row per metric source loaded by the builder.
- `metrics_all_sources_long.csv.gz`: every loaded metric fact before de-duplication.
- `metrics_long.csv.gz`: canonical metric facts after source-priority selection.
- `metrics_wide.csv`: generated wide analysis view with phase weights and metrics.
- `metric_conflicts.csv.gz`: same run/metric facts with distinct source values.
- `coverage.csv`: per-metric coverage by scale and cohort.
- `manual_backfills.csv`: input ledger for explicit manual metric backfills.
- `backfills.csv`: generated view of manual backfills used in the canonical table.

## Source Priority

Canonical metric values are selected by descending `source_priority`.

1. `manual_backfill` rows from `manual_backfills.csv`.
2. Direct GCS eval collector artifacts, including olmo-base/easy-overlap reruns.
3. Strict checkpoint-success summaries such as
   `qsplit240_300m_6b_completed_vs_60m.csv`.
4. Standard seed-noise collector outputs.
5. Local collector CSV snapshots.

W&B-derived local CSVs are useful inputs, but they are not the source of truth
when a direct GCS eval artifact is available.

## Canonical Metric Keys

Metric names are normalized into a stable schema. For lm-eval few-shot tasks,
`lm_eval/piqa_5shot/bpb` is canonicalized as:

- suite: `lm_eval`
- task: `piqa`
- num_fewshot: `5`
- metric: `bpb`
- canonical key: `lm_eval/piqa_5shot/bpb`

Metrics without explicit few-shot suffixes, such as `lm_eval/piqa/bpb`, remain
separate unless their provenance confirms they are the same task setup.

The ambiguous collector aggregate prefix `lm_eval/averages/` is intentionally
excluded because different eval reruns average over different task sets.

Unsuffixed lm-eval keys remain distinct from few-shot aliases. For example,
`lm_eval/piqa/*` is not merged into `lm_eval/piqa_5shot/*`: current 300M launch
configs use `EvalTaskConfig("piqa", 10)`, while olmo-base/easy-overlap uses the
explicit `piqa_5shot` alias.

## Expected Missingness

Some registry columns are intentionally sparse:

- `trainer_seed`, `data_seed`, and `source_run_name` only apply to seed/replay
  rows.
- `checkpoint_root` is absent for runs that were not found or never produced a
  checkpoint.
- `candidate_*` fields only apply to recovered or cross-scale rows that keep a
  pointer to a source candidate.
- Legacy collapsed-topic runs may have noncanonical weight columns such as
  `phase_0_dolma3_cc/industrial`; these are not filled for 39-domain runs
  because the topology is not equivalent.

Fit-dataset materialization drops all-null columns after filtering, so these
legacy topology fields do not leak into 39-domain fit inputs.

## Commands

Rebuild the full registry, including small GCS collector CSVs:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry
```

Rebuild from local CSVs only:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.build_metric_registry --no-gcs
```

Create a fit-ready PIQA logprob dataset:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.materialize_fit_dataset \
  lm_eval/piqa_5shot/logprob
```

Create a fit-ready uncheatable macro BPB dataset:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.materialize_fit_dataset \
  eval/uncheatable_eval/macro_bpb
```

By default, fit datasets use the `fit_swarm_60m_default` run set: original
qsplit240/core baselines plus Olmix and stratified when the metric exists. Use
`--run-set all` to include validation optima and other later runs.

Dry-run a W&B backfill manifest for PIQA metrics:

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.backfill_wandb_from_registry \
  --metric lm_eval/piqa_5shot/logprob
```

The backfill tool only mutates W&B with `--apply`. Existing conflicting summary
values are skipped unless `--overwrite-conflicts` is set.
