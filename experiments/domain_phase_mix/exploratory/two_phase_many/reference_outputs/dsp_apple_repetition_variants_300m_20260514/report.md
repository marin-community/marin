# Apple-Style Repetition-Aware DSP Check on 300M

## Model Change

The Apple-style variants keep the DSP benefit/penalty head but replace the saturation exposure with
a repetition-discounted physical exposure. For domain `i`, raw physical exposure is measured in epochs:

$$r_i(w)=c_{0i}w_{0i}+c_{1i}w_{1i}.$$

The saturation exposure is transformed by:

$$\phi(r_i;r_1)=\begin{cases}r_i,& r_i\le1\\1+r_1(1-e^{-(r_i-1)/r_1}),& r_i>1.\end{cases}$$

The benefit signal is:

$$S_i(w)=\left(1+\gamma\frac{c_{1i}w_{1i}}{r_i(w)+\epsilon}\right)\left(1-e^{-\rho_i\phi(r_i(w);r_1)}\right).$$

The penalty intentionally remains on raw physical exposure:

$$P_i(w)=\operatorname{softplus}(\log(1+r_i(w))-\tau_i)^2.$$

This tests Apple-style repetition discounting without assuming phase 1 is physically repeated faster.

## Results

| variant                                                       |   total_param_count | repetition_mode     |   fitted_r1 |   fitted_r1_min |   fitted_r1_median |   fitted_r1_max |   fitted_gamma |   fitted_gamma_benefit |   fitted_gamma_saturation |   fitted_gamma_penalty |   train_rmse |   cv_rmse |   oof_spearman |   cv_foldmean_regret_at_1 |   lower_tail_optimism |   raw_nearest_observed_tv |   phase0_max_weight |   phase1_max_weight |
|:--------------------------------------------------------------|--------------------:|:--------------------|------------:|----------------:|-------------------:|----------------:|---------------:|-----------------------:|--------------------------:|-----------------------:|-------------:|----------:|---------------:|--------------------------:|----------------------:|--------------------------:|--------------------:|--------------------:|
| old_grp_no_l2                                                 |                  42 | none                |  nan        |      nan        |         nan        |      nan        |     nan        |             nan        |                nan        |             nan        |     0.010102 |  0.011141 |       0.804042 |                  0.007789 |              0.005019 |                  0.500000 |            0.025641 |            0.393906 |
| dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |                 197 | apple_per_domain_r1 |  nan        |        0.005455 |           0.247333 |       16.300788 |       4.242625 |               4.242625 |                  1.000000 |               1.000000 |     0.000946 |  0.012125 |       0.925173 |                  0.012530 |              0.009258 |                  0.798507 |            1.000000 |            0.205813 |
| dsp_phase_benefit_saturation_penalty_nnls                     |                 160 | none                |  nan        |      nan        |         nan        |      nan        |     nan        |             100.000000 |                  0.523375 |               0.175093 |     0.002204 |  0.038985 |       0.635740 |                  0.014820 |              0.052537 |                  0.744753 |            0.835157 |            0.308442 |
| dsp_phase_benefit_penalty_nnls                                |                 158 | none                |  nan        |      nan        |         nan        |      nan        |      99.895107 |              99.895107 |                  1.000000 |               1.000000 |     0.000692 |  0.039204 |       0.715008 |                  0.017985 |              0.052040 |                  0.768829 |            1.000000 |            0.171315 |
| dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls     |                 159 | apple_shared_r1     |    0.087338 |      nan        |         nan        |      nan        |      30.876608 |              30.876608 |                  1.000000 |               1.000000 |     0.005980 |  0.040506 |       0.507135 |                  0.020831 |              0.033760 |                  0.807109 |            1.000000 |            0.199091 |
| dsp_effective_exposure_penalty_nnls                           |                 158 | none                |  nan        |      nan        |         nan        |      nan        |       0.032758 |               0.000000 |                  0.032758 |               0.032758 |     0.021057 |  0.114172 |       0.162496 |                  0.040991 |              0.111022 |                  0.737334 |            0.655949 |            0.527020 |

## Best Observed Rows by Prediction

| variant                                                       | best_pred_observed_run                   |   best_pred_observed_pred_bpb |   best_pred_observed_actual_bpb |   best_pred_observed_actual_rank |   pred_top8_mean_actual_bpb |   pred_top8_best_actual_bpb |   actual_best_bpb | actual_best_run                          |
|:--------------------------------------------------------------|:-----------------------------------------|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|----------------------------:|------------------:|:-----------------------------------------|
| dsp_phase_benefit_penalty_nnls                                | baseline_olmix_loglinear_uncheatable_bpb |                      0.956295 |                        0.956062 |                                1 |                    0.970355 |                    0.956062 |          0.956062 | baseline_olmix_loglinear_uncheatable_bpb |
| dsp_effective_exposure_penalty_nnls                           | run_00040                                |                      0.954478 |                        0.992389 |                               30 |                    0.984131 |                    0.956062 |          0.956062 | baseline_olmix_loglinear_uncheatable_bpb |
| dsp_phase_benefit_saturation_penalty_nnls                     | baseline_olmix_loglinear_uncheatable_bpb |                      0.955196 |                        0.956062 |                                1 |                    0.970355 |                    0.956062 |          0.956062 | baseline_olmix_loglinear_uncheatable_bpb |
| dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls     | baseline_olmix_loglinear_uncheatable_bpb |                      0.960767 |                        0.956062 |                                1 |                    0.970803 |                    0.956062 |          0.956062 | baseline_olmix_loglinear_uncheatable_bpb |
| dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls | baseline_olmix_loglinear_uncheatable_bpb |                      0.956677 |                        0.956062 |                                1 |                    0.970355 |                    0.956062 |          0.956062 | baseline_olmix_loglinear_uncheatable_bpb |

## Interpretation

- Canonical benefit-only DSP: CV RMSE 0.039204, OOF Spearman 0.715008.
- Tied effective-exposure DSP: CV RMSE 0.114172, OOF Spearman 0.162496.
- Split phase benefit/saturation/penalty DSP: CV RMSE 0.038985, OOF Spearman 0.635740.
- Apple shared-r1 DSP: CV RMSE 0.040506, OOF Spearman 0.507135, fitted r1 0.087338.
- Apple per-domain-r1 DSP: CV RMSE 0.012125, OOF Spearman 0.925173, median fitted r1 0.247333.
- The per-domain-r1 form should only be preferred if it improves OOF fit and optimum geometry enough to justify 39 extra nonlinear parameters.
