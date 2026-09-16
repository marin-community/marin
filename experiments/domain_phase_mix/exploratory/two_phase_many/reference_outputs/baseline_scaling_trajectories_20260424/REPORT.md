# Baseline Scaling Trajectories

Metric: `eval/uncheatable_eval/bpb` at target-budget multiplier `1.0x`.

The solid lines are target-ready rows from `analysis_dataset/nd_scale_runs.csv`. The hollow GRP no-L2 diamonds are diagnostic objective/eval points because GRP no-L2 is not a clean target-ready series in the canonical modeling dataset.

Included scale coverage:
- GRP no-L2: 60M/1.2B, 340M/10.4B, 900M/24B
- Proportional: 20M/2.6B, 60M/1.2B, 100M/6B, 340M/10.4B, 900M/24B
- Olmix: 60M/1.2B, 100M/6B, 900M/24B
- Uniform: 20M/2.6B, 60M/1.2B, 100M/6B
- UniMax: 20M/2.6B, 60M/1.2B, 100M/6B, 340M/10.4B, 900M/24B

Caveats:
- `Uniform` means `baseline_stratified`.
- `Olmix` means `baseline_olmix_loglinear_uncheatable_bpb`.
- The GRP no-L2 340M/10.4B objective point is marked `failed` because the registry row has `logical_status=failed`; treat it as a diagnostic, not a target-ready validation.
- The x-axis labels show both corrected non-embedding `N` and realized `D`; do not interpret this as an N-only scaling curve.
