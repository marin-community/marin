# Joint Model Refresh Summary, 2026-04-26

This refresh reran the MCT-LRQ structural joint model on the rebuilt canonical
analysis dataset after the latest registry refresh.

## Dataset

- Canonical rows: 640
- Scale counts: 20M/2.6B = 39, 60M/1.2B = 293, 100M/6B = 268, 340M/10.4B = 36, 900M/24B = 4
- Strong-tier registry target labels matched: 116
- New strong-tier labels enabled by the refreshed registry: 12
- Validation passed: no duplicate canonical modeling keys, no missing primary metadata, and `model_size == non_embedding_params`

## Selected Structural Law

The selected law remains `mct_lrq74_drop`, a 74-constant monotone structural law:

```text
L(w,N,D) = P(w) + E_LRQ(w)
         + A(w)((N/N0)^(-0.154791)-1)
         + B(w)((D/D0)^(-0.146425)-1)
         + C(w)((N/N0)^(-0.014295)(D/D0)^(-1.063376)-1)
```

At fixed mixture, the law is monotone decreasing in corrected non-embedding
`N` and realized train tokens `D`. At fixed `N,D`, it reduces to an LRQ mixture
regression plus the compatibility penalty `P(w)`.

## Key Metrics

| Model | Protocol | Split | n | RMSE | Spearman | Bias | Slope | Std Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mct_lrq74_drop | seed7 | seed7_holdout | 61 | 0.010034 | 0.971391 | 0.001300 | 1.034361 | 1.043821 |
| mct_lrq74_drop | seed7 | fixed340_holdout | 27 | 0.006082 | 0.931013 | 0.000539 | 0.957400 | 1.016875 |
| mct_lrq74_drop | leave900out | all900_leave_scale_out | 4 | 0.010397 | 1.000000 | -0.001060 | 0.509320 | 0.580123 |
| mct_lrq74_drop | seed7 | all_rows | 640 | 0.025518 | 0.948983 | -0.001967 | 0.937706 | 0.990640 |
| mct_lrq74_drop | leave900out | all_rows | 640 | 0.022719 | 0.953634 | -0.002193 | 0.913607 | 0.954420 |

The all-row aggregate is dominated by the large and noisier 60M swarm panel, so
the fixed-340M and 900M leave-scale-out diagnostics remain the more relevant
cross-scale checks.

## Scale Breakdown for `mct_lrq74_drop`, Seed-7 Fit

| Scale | n | RMSE | Bias | Spearman |
|---|---:|---:|---:|---:|
| 20M/2.6B | 39 | 0.022269 | 0.019085 | 0.946559 |
| 60M/1.2B | 293 | 0.035625 | -0.006631 | 0.732030 |
| 100M/6B | 268 | 0.008867 | -0.000174 | 0.881483 |
| 340M/10.4B | 36 | 0.009799 | 0.000026 | 0.891892 |
| 900M/24B | 4 | 0.015998 | -0.003631 | 0.400000 |

## Change vs Previous Validation

Compared to the previous session-12 local validation, `mct_lrq74_drop` is nearly
unchanged:

- seed7 holdout RMSE improved by 0.000050.
- fixed-340M RMSE changed by +0.000002.
- 900M leave-scale-out RMSE improved by 0.000098.
- fixed-340M slope decreased from 0.965338 to 0.957400.
- long-drop ratio `0.5x -> 2.0x` decreased from 0.946486 to 0.936467.

This is a small update, not a qualitative model change.

## Barrier-Free MCT Check

I also reran the barrier ablation on the refreshed dataset. The prediction-only
barrier-free model remains close to the full MCT model on observed rows, but it
still fails raw-simplex optimization.

| Model | Protocol | Split | n | RMSE | Spearman | Bias | Slope | Std Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mct_lrq69_balanced_no_barrier | seed7 | seed7_holdout | 61 | 0.010070 | 0.971021 | 0.001137 | 1.036483 | 1.045967 |
| mct_lrq69_balanced_no_barrier | seed7 | fixed340_holdout | 27 | 0.006130 | 0.927961 | 0.000144 | 0.942271 | 1.003268 |
| mct_lrq69_balanced_no_barrier | leave900out | all900_leave_scale_out | 4 | 0.010306 | 0.800000 | -0.002426 | 0.587351 | 0.687648 |
| mct_lrq69_balanced_no_barrier | seed7 | all_rows | 640 | 0.027002 | 0.946611 | -0.001987 | 0.931883 | 0.991198 |

The raw optimum for `mct_lrq69_balanced_no_barrier` still collapses at both
340M/10.4B and 900M/24B: phase 1 is essentially all tech, and the hard-corner
and family-collapse flags are true. So the refreshed data does not rescue the
barrier-free model as a deployment law. It remains a useful predictive ablation,
not the model to use for raw optimum selection.

## Artifacts

- Main regenerated report: `mct_lrq_refresh/REPORT.md`
- Standard split metrics: `mct_lrq_refresh/csv/metric_summary.csv`
- All-row metrics: `mct_lrq_refresh/csv/all_rows_metric_summary.csv`
- Scale-level all-row metrics: `mct_lrq_refresh/csv/all_rows_by_scale_metric_summary.csv`
- All-row prediction plot: `mct_lrq_refresh/plots/mct_lrq74_drop_all_rows_pred_actual.png`
- All-row residual plot: `mct_lrq_refresh/plots/mct_lrq74_drop_all_rows_residual_by_scale.png`
- Barrier ablation refresh: `mct_barrier_ablation_refresh/REPORT.md`
- Reproducibility archive: `mct_lrq_refresh.zip`
