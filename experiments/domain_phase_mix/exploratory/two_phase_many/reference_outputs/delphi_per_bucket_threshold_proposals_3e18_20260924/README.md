# One-threshold-per-bucket MARINER proposals at 3e18 (materialized 2026-09-24)

Third trained-proposal batch for Table 3 (`tab:baseline-validation`): the Table 1 ablation that frees the harm
threshold per bucket (registry id `weibull_softplus_unscaled@kappa_floor_link_flat15_nocap_per_bucket_threshold`,
121 parameters per task), fitted by the same protocol as the quadratic, spline, convex and additive rows. Calvin
asked for it on 2026-09-24 ("have we not validated one threshold per bucket at 3e18? We should if we haven't");
it had never been trained. Script: `materialize_delphi_comparator_proposals_20260909.py --comparators pbt`
(comparator added the same day); launcher `experiments/domain_phase_mix/launch_delphi_per_bucket_threshold_3e18.py`
(six runs, ids 7,490,000+ and 7,490,100+, tests in `tests/test_launch_delphi_per_bucket_threshold_3e18.py`);
collector `collect_delphi_3e18_validation_results_20260906.py --launch per_bucket_threshold`.

Protocol: fits on all 280 runs with the harness's heldout-stage protocol (certify inner folds, proportional run
pinned to training; per-bucket thresholds by two coordinate sweeps from the shared fit); the reference
implementation's optimizer (SLSQP from the proportional mixture and four seeded swarm rows, polished, rounded to the
1/2048 grid with the exchange search); no cap or penalty; nominal caps in the ids are inactive. Training: Uncheatable
at data seed 666200, suite at 662009, trainer seeds 0/1/2.

| candidate | max epochs | active buckets | TV to MARINER | own prediction | MARINER's prediction | MARINER at its optimum |
|---|---:|---:|---:|---:|---:|---:|
| `cmp_u_pbt_cap06` | 5.23 | 25 | 0.062 | 0.9839 | 0.9814 | 0.9810 |
| `cmp_t9_pbt_cap08` | 7.93 | 39 | 0.049 | 1.0613 | 1.0635 | 1.0631 |

Both proposals sit within 0.06 total variation of MARINER's, closer than the convex (0.17/0.056) and additive
(0.195/0.097) proposals, and MARINER predicts them within 0.0004 BPB of its own optimum, so the measured contrast
is expected inside seed noise (about 0.001 on Uncheatable, 0.005 on the suite).

`summary.json`, `solutions.csv`, `fits.json` (per-component shapes incl. the 39 thresholds), `cross_predictions_*.csv`
and `tv_between_proposals_*.csv` hold the fits, weights and predictions; `launch_dry_run/` the six run manifests;
`submission/` the dry run, safety checks and `launch.sh` (TPU=v6e-8 default or TPU=v5p-8; SUBMIT=1 submits).
Submitted 2026-09-24 23:24 PDT on v6e-8 (Calvin: "do v6e-8"): Iris `/calvinxu/dm-delphi-3e18-per-bucket-threshold-v6e8-20260924`
(direct form, executor as the top-level job, children in us-east5-b), Fieldbook experiment `exp_01m3bkrjfm8v8z3y3n5enmsb8v`,
watch `submission/watch.sh` -> `submission/watch.log`. Killed at 23:53 PDT before any child ran, at Calvin's request: the
single-host v6e requests appeared to hold back the v6e-64 slices of the Olmix 1e21 ladder ("let's actually do it on v5p8").
Resubmitted 23:56 PDT on v5p-8 in us-east5-a as `/calvinxu/dm-delphi-3e18-per-bucket-threshold-v5p8-20260924` (launcher pin
relaxed to allow that pair; the OlmoBaseEval evaluations keep their v6e-8 resources in us-east5-b). Measured results land in
`../delphi_per_bucket_threshold_3e18_20260924/` after collection.

## Result (all six runs and evaluations, 2026-09-25 23:13 PDT)

Collected with `collect_delphi_3e18_validation_results_20260906.py --launch per_bucket_threshold` into
`../delphi_per_bucket_threshold_3e18_20260924/measured_results.csv`. Paired against MARINER's seed-matched runs (trainer
seed 0 is the frozen-procedure validation run, seeds 1 and 2 the fairness repeats, same data seeds):

| proposal | objective | measured (mean ± std, 3 seeds) | MARINER | paired Δ ± SE | other objective, paired Δ |
|---|---|---|---|---|---|
| `cmp_u_pbt_cap06` | Uncheatable | 0.9911 ± 0.0011 | 0.9825 ± 0.0010 | +0.0087 ± 0.0004 | OlmoBaseEval Easy +0.0079 |
| `cmp_t9_pbt_cap08` | OlmoBaseEval Easy | 1.0658 ± 0.0052 | 1.0678 ± 0.0052 | −0.0020 ± 0.0039 | Uncheatable +0.0115 |

The ablation loses on Uncheatable and ties on OlmoBaseEval Easy, the fully convex ablation's pattern (+0.0089, −0.0019).
Hardware: these runs trained on v5p-8, the other Table 3 rows on v6e-8. The v6e-8 convex suite proposal, a similar
distance from MARINER's mixture (TV 0.056 vs 0.049), shows the same off-objective shift (suite −0.0019, Uncheatable
+0.0118 vs −0.0020, +0.0115 here), so no v5p-8 offset is visible. MARINER predicted the Uncheatable proposal at 0.9814
(error 0.0097) and the suite proposal at 1.0635 (error 0.0023); the variant's own predictions were 0.9839 and 1.0613
(optimism +0.007 and +0.005). Optimism and parameters for the appendix table: 121 parameters per task, 3 seeds.
