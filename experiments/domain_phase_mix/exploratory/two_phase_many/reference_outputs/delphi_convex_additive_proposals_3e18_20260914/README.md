# Fully convex and additive MARINER proposals at 3e18 (materialized 2026-09-14)

Second trained-proposal batch for Table 3 (`tab:baseline-validation`): two Table 1 ablations of MARINER fitted by the
same protocol as the quadratic and spline rows, so every Table 3 row shares one protocol. The additive proposal
replaces the earlier calibration-protocol run marked with a dagger. Fieldbook experiment
`submission/fieldbook_experiment_id.txt` (`exp_01m2fny7wgdnzpbbrwpv29wey7`, protocol prespecified before the
proposals were computed). Script: `materialize_delphi_comparator_proposals_20260909.py --comparators cvx,add`;
launcher `experiments/domain_phase_mix/launch_delphi_convex_additive_3e18.py` (twelve runs, ids 7,480,000+ and
7,480,100+); collector `collect_delphi_3e18_validation_results_20260906.py --launch convex_additive`.

- `cvx`: fully convex objective (convex harm), registry id `weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_raw_epoch_hinge`:
  harm `softplus((E - (e^tau - 1)) / e^tau)`, convex in epochs everywhere, so the proposal is a convex program.
- `add`: additive response, registry id `weibull_softplus_unscaled` (no floor or exponential link).

Protocol: fits on all 280 runs with the harness's heldout-stage protocol (certify inner folds, proportional run pinned
to training); the reference implementation's optimizer (SLSQP from the proportional mixture and four seeded swarm
rows, polished, rounded to the 1/2048 grid with the exchange search); no cap or penalty; nominal caps in the ids are
inactive. Training: v6e-8, Uncheatable at data seed 666200, suite at 662009, trainer seeds 0/1/2 for both variants.

| candidate | max epochs | active buckets | TV to MARINER | own prediction | MARINER's prediction | MARINER at its optimum |
|---|---:|---:|---:|---:|---:|---:|
| `cmp_u_cvx_cap06` | 5.58 | 26 | 0.170 | 0.9885 | 0.9846 | 0.9810 |
| `cmp_u_add_cap08` | 6.52 | 35 | 0.195 | 0.9445 | 0.9842 | 0.9810 |
| `cmp_t9_cvx_cap08` | 7.93 | 38 | 0.056 | 1.0640 | 1.0641 | 1.0631 |
| `cmp_t9_add_cap08` | 7.64 | 39 | 0.097 | 1.0051 | 1.0656 | 1.0631 |

`summary.json`, `solutions.csv`, `fits.json`, `cross_predictions_*.csv` and `tv_between_proposals_*.csv` hold the
fits, weights and predictions; `launch_dry_run/` the twelve run manifests; `submission/` the tests, safety check and
launch command. Measured results land in `../delphi_convex_additive_3e18_20260914/` after collection.

## Submission (2026-09-14)

Three attempts, all under Fieldbook experiment `exp_01m2fny7wgdnzpbbrwpv29wey7`, run `run_01m2fptqkj53vkv906519anbpb`:

1. `...-20260914` (03:17 PDT): the previous batches' exclude set bundles 25.6 MB, over Iris's 25 MB cap, so the whole
   `experiments/domain_phase_mix/exploratory/` tree was excluded; the parent failed at import because the launch path
   needs `exploratory.dsre_ceq_tools` (through `static_batch_selection`).
2. `...-r2` (03:21 PDT): known-good excludes plus media, the macOS kitoken wheel, `infra/`, `lib/*/tests/`, `lib/iris/dashboard/`
   and `starcoder_tpp10_assets/` (19.2 MB); the build failed because `infra/deploy` is a uv workspace member.
3. `...-r3` (03:23 PDT, kept): known-good excludes plus media (`png/html/pkl/npz/parquet`), the macOS wheel, `infra/grafana/`
   and `starcoder_tpp10_assets/` (21.6 MB). Parent us-east5-a, interactive priority, children v6e-8 in us-east5-b.

Attempt logs: `submit_attempt{1,2}_failed.log`, `launch_command_attempt{1,2}_failed.sh`; the kept command is
`launch_command.sh` with `east5_launch_safety_r3.log`.
