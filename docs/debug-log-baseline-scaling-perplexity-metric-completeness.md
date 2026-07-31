# Debugging log for baseline-scaling perplexity metric completeness

## Initial Status

The interactive baseline-scaling HTML exposed many selectable `eval/...`
perplexity metrics, but most non-default metrics had fewer cells than
`eval/uncheatable_eval/bpb`.

## Hypothesis 1: Some runs only logged the default metric

I audited the generated `baseline_scaling_trajectories_points.csv`. The missing
values were not broad. Every non-primary metric was missing only for:

- `GRP no-L2`, `340M/10.4B`
- `GRP no-L2`, `900M/24B`

Both rows came through the analysis dataset as `run_registry_objective_metric`
rows.

## Result

The full target-step GCS artifacts exist for both rows:

- `.../baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b-4c051e/checkpoints/eval_metrics.jsonl`
- `.../baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_1_2b_24b-f8dfb1/checkpoints/eval_metrics.jsonl`

At the target steps, each JSON payload contains the same full 60-key eval set,
including `eval/bpb`, `eval/uncheatable_eval/macro_bpb`, Paloma metrics, and
per-domain uncheatable metrics.

## Hypothesis 2: Plot ingestion failed to hydrate analysis rows from GCS

The plot script carried full eval metrics from direct GCS overrides and
registry-backed target rows, but for analysis-dataset rows it mostly trusted the
CSV contents. For the two GRP rows, the analysis dataset carried only the
objective metric, even though the checkpoint root had the full target-step eval
payload.

## Fix

Updated
`experiments/domain_phase_mix/exploratory/paper_plots/baseline_scaling_trajectories.py`
so `_point_from_analysis_row` hydrates full eval metrics from
`checkpoints/eval_metrics.jsonl` using the exact target/final checkpoint step
when a checkpoint root is available.

## Validation

Commands:

```bash
uv run python -m py_compile experiments/domain_phase_mix/exploratory/paper_plots/baseline_scaling_trajectories.py
uv run experiments/domain_phase_mix/exploratory/paper_plots/baseline_scaling_trajectories.py
```

Result:

- `baseline_scaling_trajectories_points.csv` has 24 plotted target-ready rows.
- All 58 selectable perplexity metrics now have values for all 24 plotted rows.
- The remaining missing canonical cell is `Uniform 900M/24B`, which is not
  target-ready and therefore not plotted yet.
