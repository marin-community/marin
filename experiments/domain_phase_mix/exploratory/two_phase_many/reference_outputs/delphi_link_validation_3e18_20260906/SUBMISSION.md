# Delphi bounded log-deficit link validation at 3e18

Submitted on 2026-09-06 at 13:17 UTC as `/calvinxu/dm-delphi-3e18-lwspu-link-v6e8-20260906`
([Iris parent](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fdm-delphi-3e18-lwspu-link-v6e8-20260906)).
Requested by the user ("feel free to submit more 3e18 validation runs for uncheatable, Table-9 like
exp_01m1vbajfrdebbsatntj4sm00z on east5b v6e").

| Optimization target | Epoch cap | Surrogate | Data seed | Candidate |
| --- | ---: | --- | ---: | --- |
| Uncheatable | 6 | WSPU, bounded log-deficit link | 666200 | `lwspu_u_bl_cap06` |
| Table 9 | 6 | WSPU, bounded log-deficit link | 662009 | `lwspu_t9_bl_cap06` |
| Table 9 | 8 | WSPU, bounded log-deficit link | 662009 | `lwspu_t9_bl_cap08` |

KL coefficient is zero and trainer seed is zero. Each training uses the existing 3e18 recipe
(358,304,128 parameters, batch 128, sequence 4096, 3007 steps, 1,576,534,016 tokens; final HF checkpoint at
step 3006); both phases use identical mixture weights; every candidate receives inline Uncheatable
evaluation and native Table-9 evaluation. The CPU parent is in us-east5-a; training and evaluation children
request v6e-8 in us-east5-b; all paths use gs://marin-us-east5. The launcher releases all three rows with
max_concurrent=3.

The surrogate is `weibull_softplus_unscaled@log_deficit_bounded_link` fitted on the canonical 280 rows
(fold -1 of `delphi_link_selection_20260906`, which reproduces the reference WSPU fits to 2e-16). On the
frozen selection benchmark it is the best-calibrated model seen so far (archive optimism -0.005 / -0.015
BPB against WSPU's +0.036 / +0.070 on Uncheatable / Table 9; RMSE 0.015 / 0.021 against 0.029 / 0.038) with
pooled optima-stratum regret 0.0030 / 0.0143 (WSPU 0.0023 / 0.0157); within source blocks its Table-9
regret is worse than WSPU's by +0.0036 [+0.0009, +0.0071]. The heads were reconstructed from the saved
shapes and ridges, checked against the saved predictions (parity < 2e-15), minimized under each cap from the
same five starts as the coupling validation, rounded to counts/2048 and refined by the existing one-count
exchange on the link predictor. All three policies converged; maximum allocation transfer from the
continuous optimum is 0.0002 and the runtime penalty is below 1e-5 BPB.

Predicted values (link surrogate) at the runtime policies: Uncheatable cap 6 0.9870; Table 9 cap 6 1.0841;
Table 9 cap 8 1.0839. The link predicts the kappa-0 WSPU comparator policies at 0.9902 / 1.0866 / 1.0888, so
it expects its own optima to be better by only 0.003 / 0.0025 / 0.005 BPB. WSPU predicts the link optima at
0.9546 / 1.0155 / 1.0150. Total-variation distance from the link optimum to the WSPU kappa-0 policy is
0.137 / 0.107 / 0.155; hull distance from the panel 0.39 / 0.32 / 0.32. Effective bucket counts 11.7 /
13.5 / 13.6 (WSPU 11.8 / 14.5 / 12.1).

The prospective comparison is each policy's realized macro BPB against its target/cap matched-seed kappa-0
WSPU control (`wspu_uncheatable_cap06` 0.9834 measured; `wspu_table9_cap06` 1.0722; the cap-8 control from
the same sweep), with predicted-versus-realized values for both surrogates. Single-seed comparisons; they do
not establish seed robustness.

Evidence:

- `offline_materialization/`: restart diagnostics, continuous weights, `policies.json` with cross-predictions.
- `runtime_materialization/candidate_weights.csv` (sha256
  `82839dd68528c1365cf6aa9fa819df0cd2c924968509b5168174398f31ad1c70`), `candidate_mapping.csv`, `summary.json`.
- `launch_dry_run/`: resolved run specs and manifests per target.
- `submission/launch_command.sh`, `submit.log`, `submit_exit.txt`, `east5_launch_safety.log`
  (`--expected-child-zone us-east5-b`, passed), `launch_dry_run.log`, `launcher_tests.log`
  (`tests/test_launch_delphi_link_validation_3e18.py` and the shared sweep tests).

Fieldbook experiment: `exp_01m1vdtb243bg75c233y34chr8`.
