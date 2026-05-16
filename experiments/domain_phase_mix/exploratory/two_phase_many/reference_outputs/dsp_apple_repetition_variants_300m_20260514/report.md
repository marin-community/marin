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
| dsp_phase_benefit_saturation_penalty_nnls                     |                 160 | none                |  nan        |      nan        |         nan        |      nan        |     nan        |               0.002044 |                 98.166580 |               5.527307 |     0.004391 |  0.006635 |       0.926152 |                  0.000000 |              0.002759 |                  0.499956 |            1.000000 |            0.723478 |
| dsp_effective_exposure_penalty_nnls                           |                 158 | none                |  nan        |      nan        |         nan        |      nan        |      14.362237 |               0.000000 |                 14.362237 |              14.362237 |     0.005013 |  0.007106 |       0.919645 |                  0.001022 |              0.004280 |                  0.500000 |            0.271780 |            0.932116 |
| dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |                 197 | apple_per_domain_r1 |  nan        |        0.212097 |           3.657024 |      100.000000 |      61.125711 |              61.125711 |                  1.000000 |               1.000000 |     0.005660 |  0.008739 |       0.900558 |                  0.000281 |              0.004217 |                  0.500000 |            0.999988 |            0.586610 |
| dsp_phase_benefit_penalty_nnls                                |                 158 | none                |  nan        |      nan        |         nan        |      nan        |      25.355533 |              25.355533 |                  1.000000 |               1.000000 |     0.005629 |  0.008835 |       0.898476 |                  0.000201 |              0.004506 |                  0.500000 |            1.000000 |            0.440866 |
| dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls     |                 159 | apple_shared_r1     |  100.000000 |      nan        |         nan        |      nan        |      51.349081 |              51.349081 |                  1.000000 |               1.000000 |     0.005730 |  0.009807 |       0.893272 |                  0.000281 |              0.005641 |                  0.500000 |            0.999998 |            0.573589 |
| old_grp_no_l2                                                 |                  42 | none                |  nan        |      nan        |         nan        |      nan        |     nan        |             nan        |                nan        |             nan        |     0.010102 |  0.011141 |       0.804042 |                  0.007789 |              0.005019 |                  0.500000 |            0.025641 |            0.393906 |

## Best Observed Rows by Prediction

| variant                                                       | best_pred_observed_run                   |   best_pred_observed_pred_bpb |   best_pred_observed_actual_bpb |   best_pred_observed_actual_rank |   pred_top8_mean_actual_bpb |   pred_top8_best_actual_bpb |   actual_best_bpb | actual_best_run   |
|:--------------------------------------------------------------|:-----------------------------------------|------------------------------:|--------------------------------:|---------------------------------:|----------------------------:|----------------------------:|------------------:|:------------------|
| dsp_phase_benefit_penalty_nnls                                | run_00125                                |                      0.954995 |                        0.955440 |                                1 |                    0.962583 |                    0.955440 |          0.955440 | run_00125         |
| dsp_effective_exposure_penalty_nnls                           | run_00125                                |                      0.953907 |                        0.955440 |                                1 |                    0.964047 |                    0.955440 |          0.955440 | run_00125         |
| dsp_phase_benefit_saturation_penalty_nnls                     | run_00200                                |                      0.953066 |                        0.955662 |                                2 |                    0.961966 |                    0.955440 |          0.955440 | run_00125         |
| dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls     | baseline_olmix_loglinear_uncheatable_bpb |                      0.951688 |                        0.956062 |                                3 |                    0.962860 |                    0.955440 |          0.955440 | run_00125         |
| dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls | baseline_olmix_loglinear_uncheatable_bpb |                      0.955101 |                        0.956062 |                                3 |                    0.962583 |                    0.955440 |          0.955440 | run_00125         |

## Interpretation

- Canonical benefit-only DSP: CV RMSE 0.008835, OOF Spearman 0.898476.
- Tied effective-exposure DSP: CV RMSE 0.007106, OOF Spearman 0.919645.
- Split phase benefit/saturation/penalty DSP: CV RMSE 0.006635, OOF Spearman 0.926152.
- Apple shared-r1 DSP: CV RMSE 0.009807, OOF Spearman 0.893272, fitted r1 100.000000.
- Apple per-domain-r1 DSP: CV RMSE 0.008739, OOF Spearman 0.900558, median fitted r1 3.657024.
- The per-domain-r1 form should only be preferred if it improves OOF fit and optimum geometry enough to justify 39 extra nonlinear parameters.
