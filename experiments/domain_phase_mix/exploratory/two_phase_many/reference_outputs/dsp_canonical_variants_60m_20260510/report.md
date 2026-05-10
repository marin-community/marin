# DSP Canonical Form Sweep on 300M

## Setup

DSP denotes the reduced-bias domain saturation-penalty family. The sweep keeps M-dependent
parameters to at most four per domain and uses only fixed global scalars for two-phase effects.

Compared variants:

- `dsp_no_phase_penalty_nnls`: Control: per-domain saturation and penalty with no phase term.
- `dsp_phase_benefit_penalty_nnls`: Canonical reduced-bias DSP: phase-1 premium only multiplies the benefit signal.
- `dsp_effective_exposure_penalty_nnls`: Empirical comparator: phase-1 multiplier enters exposure for benefit and penalty.
- `dsp_phase_benefit_penalty_signed`: Tests nonnegative-head assumption by using a signed ridge linear head.
- `dsp_phase_benefit_no_penalty_nnls`: Tests whether explicit overexposure penalties are needed.
- `dsp_effective_exposure_no_penalty_nnls`: Tests whether the empirical phase multiplier can replace explicit penalties.

## Results

| variant                        |   m_dependent_params_per_domain |   total_param_count |   cv_rmse |   oof_spearman |   cv_foldmean_regret_at_1 |   lower_tail_optimism |   raw_nearest_observed_tv |   raw_nearest_observed_value |   phase0_max_weight |   phase1_max_weight |
|:-------------------------------|--------------------------------:|--------------------:|----------:|---------------:|--------------------------:|----------------------:|--------------------------:|-----------------------------:|--------------------:|--------------------:|
| dsp_phase_benefit_penalty_nnls |                               4 |                 158 |  0.009969 |       0.850313 |                  0.004586 |              0.004607 |                  0.826167 |                     1.068761 |            0.999988 |            0.390543 |

## Best Observed Rows By Prediction

| variant                        | best_pred_observed_run   |   best_pred_observed_pred_bpb |   best_pred_observed_actual_bpb |   best_pred_observed_actual_rank |   pred_top8_mean_actual_bpb |   pred_top8_best_actual_bpb |   actual_best_bpb | actual_best_run   |
|:-------------------------------|:-------------------------|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|----------------------------:|------------------:|:------------------|
| dsp_phase_benefit_penalty_nnls | run_00125                |                      1.057980 |                        1.057199 |                                1 |                    1.075713 |                    1.057199 |          1.057199 | run_00125         |

## Interpretation

- Best CV RMSE row: `dsp_phase_benefit_penalty_nnls` with cv_rmse=0.009969.
- Best OOF Spearman row: `dsp_phase_benefit_penalty_nnls` with oof_spearman=0.850313.
- Recommended canonical DSP form: `dsp_phase_benefit_penalty_nnls`. It keeps phase-1 as a benefit premium rather than an effective-exposure multiplier, keeps the explicit overexposure penalty, and preserves nonnegative benefit/penalty semantics.
- Empirical upper-bound comparator: `dsp_effective_exposure_penalty_nnls`. It fits best, but the phase-1 multiplier enters the exposure used by both saturation and penalty, which reintroduces the saturation/penalty bias we wanted to reduce.
- Removing explicit penalties is not viable here: both no-penalty variants lose roughly 0.005-0.006 CV RMSE versus the canonical form and have worse top-row regret.
- Allowing signed heads does not help fit or rank; it weakens semantics and increases top-row regret versus NNLS.
- Use raw optima only as diagnostics until the off-manifold/collapse issue is fixed.
- A canonical DSP form should prefer strong observed-row ranking, defensible phase semantics, and stable optima over pure in-sample fit.
