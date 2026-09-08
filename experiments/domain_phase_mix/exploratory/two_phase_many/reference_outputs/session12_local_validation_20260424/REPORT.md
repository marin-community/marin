# Session 12 Local Validation

## Scope

Validated the most promising ChatGPT Pro Session 12 candidates against the corrected v31 packet:

- Session 3: ordinal-isotonic wrapper around compact power-beta.
- Session 4: regularized structural power-anchor (RSP).
- Session 5: monotone curvature-tuned LRQ (MCT-LRQ).
- Session 6: A3S-SRG99 semantic residual + sparse guard.
- Session 2 and Session 10: low-parameter S2 guard donors.

Outputs are under this directory. The consolidated table is `validated_candidate_summary.csv`.

## Main Results

| model | params | holdout RMSE | fixed-340M RMSE | all-900M RMSE | raw optimum | interpretation |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `pcf_compact_p1fam10_powbeta_a4p64_gap025` | 102 | 0.009044 | 0.003980 | 0.010138 | mixed | best near-budget predictive reference |
| `oit99_token_barycenter` | 99 | 0.009079 | 0.004398 | 0.010909 | barrier-dependent | best strict-under-100 predictive wrapper, but raw argmin safety comes from support barrier |
| `pcf_compact98_p1fam6_powbeta_a3_gap025` | 98 | 0.009098 | 0.004847 | 0.011258 | mixed | compact power-beta baseline |
| `a3s_srg99_selected` | 99 | 0.010050 | 0.005543 | 0.008044 | sane | strongest all-900M structural-ish candidate |
| `a3s_srg99_alt_C_finemath_stackfim` | 99 | 0.010285 | 0.005791 | 0.006733 | sane | best all-900M RMSE, slightly weaker seed/fixed |
| `rsp95_rho070_d030_l2p2_transfer` | 100 | 0.011922 | 0.005910 | 0.007107 | sane | good high-scale transfer, weaker local prediction |
| `mct_lrq74_drop` | 74 | 0.010084 | 0.006080 | 0.010495 | sane | cleanest compact structural candidate |
| `s2_geom_reason_guard_67` | 67 | 0.011083 | 0.006821 | 0.011529 | sane | low-param S2 raw-optimum repair |

## Combination Probes

Two lightweight combination checks were run in `combination_probe/`.

1. `MCT-LRQ + ordinal token tail` is negative.
   - `mct_lrq74_drop`: holdout `0.010084`, fixed-340M `0.006080`, all-900M protocol `0.010495`.
   - `mct_lrq74_drop_plus_refit_ordinal_tail`: holdout `0.010970`, fixed-340M `0.007299`, all-900M `0.014257`.
   - The Session 3 token-tail correction appears tuned to the power-beta residual structure and does not transfer to MCT-LRQ.

2. Oracle pair blends are strongly positive but leaky.
   - Best fixed-340M oracle pair: `a3s_srg99_alt_C_finemath_stackfim | oit99_midpivot`, RMSE `0.003264`.
   - Best seed holdout oracle pair: `a3s_srg99_selected | oit99_zeroD6`, RMSE `0.007997`.
   - Best all-900M oracle pair: `mct_lrq74_drop_all900fit | rsp95_rho060_d040_l2p25_holdout_all900fit`, RMSE `0.003946`.
   - This only proves residual complementarity; it is not a valid model-selection result because weights are chosen on evaluation rows.

## Interpretation

Session 12 produced useful components, not a single promotion.

- Best strict compact predictive result: `oit99_token_barycenter`, but it is mostly `PB98 + scale-only token correction + support barrier`.
- Best compact structural result: `mct_lrq74_drop` / `mct_lrq74_balanced`. These are worse than power-beta on fixed-340M but are much cleaner and have sane raw optima.
- Best high-scale transfer donor: `a3s_srg99_alt_C_finemath_stackfim`, but the semantic residual + sparse guard is less theoretically clean than MCT-LRQ.
- Best regularized structural donor: RSP. It has sane raw optima and strong all-900M transfer, but weaker seed/fixed prediction and compressed long drops.

The most promising next modeling direction is not a direct graft of Session 3's token tail onto MCT. It is a unified compact law that combines:

- MCT-LRQ's clean monotone structural body,
- SRG/RSP-style regularized residual capacity for high-scale transfer,
- an explicit support/geometry prior that is zero or near-zero on observed mixtures,
- and a non-leaky way to learn blend/residual weights, ideally using train-only grouped CV or leave-scale validation.

## Reproduction Notes

Main commands used:

```bash
uv run --with matplotlib --with scipy python /tmp/chatgpt_pro_session_12_review/5_joint_mixture_scale_structural_v31_mct/joint_mixture_scale_structural_v31_mct/code/run_mct_lrq_law.py \
  --packet-root experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v31 \
  --outdir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/session12_local_validation_20260424/session5_mct_lrq

uv run --with matplotlib --with scipy python /tmp/chatgpt_pro_session_12_review/4_joint_model_archive_v31_pareto/joint_model_archive_v31_pareto/code/run_rsp_pareto_law.py \
  --packet-root experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v31 \
  --out-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/session12_local_validation_20260424/session4_rsp

uv run --with matplotlib --with scipy python /tmp/chatgpt_pro_session_12_review/3_joint_mixture_scale_structural_law_v33_ordinal_isotonic_archive/joint_mixture_scale_v33_ordinal_isotonic/code/run_v33_ordinal_isotonic.py \
  --packet-root experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v31 \
  --powerbeta-root experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v31/DO_NOT_READ_FIRST_REFERENCE_FORMS/powerbeta_compact \
  --out experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/session12_local_validation_20260424/session3_ordinal
```

Session 6 generated usable metrics and model artifacts but crashed during report plotting due to a stale plot-column reference. The generated CSVs were retained and included in this report.
