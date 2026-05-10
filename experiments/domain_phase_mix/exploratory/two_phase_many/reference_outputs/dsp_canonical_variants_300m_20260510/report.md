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

| variant                                |   m_dependent_params_per_domain |   total_param_count |   cv_rmse |   oof_spearman |   cv_foldmean_regret_at_1 |   lower_tail_optimism |   raw_nearest_observed_tv |   raw_nearest_observed_value |   phase0_max_weight |   phase1_max_weight |
|:---------------------------------------|--------------------------------:|--------------------:|----------:|---------------:|--------------------------:|----------------------:|--------------------------:|-----------------------------:|--------------------:|--------------------:|
| old_grp_no_l2                          |                      nan        |                  42 |  0.011141 |       0.804042 |                  0.007789 |              0.005019 |                  0.500000 |                     0.968801 |            0.025641 |            0.393906 |
| dsp_no_phase_penalty_nnls              |                        4.000000 |                 157 |  0.019480 |       0.306467 |                  0.005647 |              0.010248 |                  0.500000 |                     0.968801 |            0.939376 |            0.138239 |
| dsp_phase_benefit_penalty_nnls         |                        4.000000 |                 158 |  0.008835 |       0.898476 |                  0.000201 |              0.004506 |                  0.500000 |                     0.968801 |            1.000000 |            0.440866 |
| dsp_effective_exposure_penalty_nnls    |                        4.000000 |                 158 |  0.007106 |       0.919645 |                  0.001022 |              0.004280 |                  0.500000 |                     0.968801 |            0.271780 |            0.932116 |
| dsp_phase_benefit_penalty_signed       |                        4.000000 |                 158 |  0.008946 |       0.895539 |                  0.005874 |              0.003542 |                  0.500000 |                     0.968801 |            1.000000 |            0.355369 |
| dsp_phase_benefit_no_penalty_nnls      |                        2.000000 |                  80 |  0.014162 |       0.832425 |                  0.012057 |              0.004041 |                  0.500000 |                     0.968801 |            0.506229 |            0.504756 |
| dsp_effective_exposure_no_penalty_nnls |                        2.000000 |                  80 |  0.013497 |       0.805319 |                  0.011951 |              0.005002 |                  0.500000 |                     0.968801 |            0.552901 |            0.396512 |

## Best Observed Rows By Prediction

| variant                                | best_pred_observed_run                   |   best_pred_observed_pred_bpb |   best_pred_observed_actual_bpb |   best_pred_observed_actual_rank |   pred_top8_mean_actual_bpb |   pred_top8_best_actual_bpb |   actual_best_bpb | actual_best_run   |
|:---------------------------------------|:-----------------------------------------|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|----------------------------:|------------------:|:------------------|
| dsp_no_phase_penalty_nnls              | baseline_olmix_loglinear_uncheatable_bpb |                      0.957310 |                        0.956062 |                                3 |                    0.969375 |                    0.955440 |          0.955440 | run_00125         |
| dsp_phase_benefit_penalty_nnls         | run_00125                                |                      0.954995 |                        0.955440 |                                1 |                    0.962583 |                    0.955440 |          0.955440 | run_00125         |
| dsp_effective_exposure_penalty_nnls    | run_00125                                |                      0.953907 |                        0.955440 |                                1 |                    0.964047 |                    0.955440 |          0.955440 | run_00125         |
| dsp_phase_benefit_penalty_signed       | run_00125                                |                      0.956123 |                        0.955440 |                                1 |                    0.963998 |                    0.955440 |          0.955440 | run_00125         |
| dsp_phase_benefit_no_penalty_nnls      | baseline_olmix_loglinear_uncheatable_bpb |                      0.952388 |                        0.956062 |                                3 |                    0.969227 |                    0.955662 |          0.955440 | run_00125         |
| dsp_effective_exposure_no_penalty_nnls | run_00200                                |                      0.956221 |                        0.955662 |                                2 |                    0.970215 |                    0.955662 |          0.955440 | run_00125         |

## Interpretation

- Best CV RMSE row: `dsp_effective_exposure_penalty_nnls` with cv_rmse=0.007106.
- Best OOF Spearman row: `dsp_effective_exposure_penalty_nnls` with oof_spearman=0.919645.
- Recommended canonical DSP form: `dsp_phase_benefit_penalty_nnls`. It keeps phase-1 as a benefit premium rather than an effective-exposure multiplier, keeps the explicit overexposure penalty, and preserves nonnegative benefit/penalty semantics.
- Empirical upper-bound comparator: `dsp_effective_exposure_penalty_nnls`. It fits best, but the phase-1 multiplier enters the exposure used by both saturation and penalty, which reintroduces the saturation/penalty bias we wanted to reduce.
- Removing explicit penalties is not viable here: both no-penalty variants lose roughly 0.005-0.006 CV RMSE versus the canonical form and have worse top-row regret.
- Allowing signed heads does not help fit or rank; it weakens semantics and increases top-row regret versus NNLS.
- Use raw optima only as diagnostics until the off-manifold/collapse issue is fixed.
- A canonical DSP form should prefer strong observed-row ranking, defensible phase semantics, and stable optima over pure in-sample fit.
