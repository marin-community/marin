# DSP Fits to Issue #5416 Aggregate at 300M/6B

The issue #5416 aggregate is higher-is-better. Fits here use `objective_metric = -issue5416_aggregate` so the DSP loss-minimization code can be reused without changing the nonnegative benefit/penalty semantics.

- Signal rows: 242
- Variable-subset noise rows for projection: 10
- Selected aggregate items: 26
- Horn-selected factors: 5

## Summary

| variant                                                       |   total_param_count |   score_cv_rmse |   score_cv_r2 |   score_oof_spearman |   score_oof_pearson |   cv_foldmean_regret_at_1 | best_pred_observed_run   |   best_pred_observed_actual_score |   best_pred_observed_actual_rank |   raw_predicted_optimum_score |   raw_nearest_observed_tv | raw_nearest_observed_run_name   |   raw_nearest_observed_score |   phase0_max_weight |   phase1_max_weight |   fitted_gamma_benefit |   fitted_gamma_saturation |   fitted_gamma_penalty |
|:--------------------------------------------------------------|--------------------:|----------------:|--------------:|---------------------:|--------------------:|--------------------------:|:-------------------------|----------------------------------:|---------------------------------:|------------------------------:|--------------------------:|:--------------------------------|-----------------------------:|--------------------:|--------------------:|-----------------------:|--------------------------:|-----------------------:|
| dsp_phase_benefit_saturation_penalty_nnls                     |                 160 |        0.148561 |      0.859189 |             0.918927 |            0.926933 |                  0.003623 | run_00125                |                          1.015433 |                                1 |                      5.353890 |                  0.715133 | run_00024                       |                    -0.129205 |            0.470124 |            0.950481 |              17.140671 |                  0.642663 |               9.956507 |
| dsp_effective_exposure_penalty_nnls                           |                 158 |        0.174260 |      0.806258 |             0.880500 |            0.900836 |                  0.019935 | run_00125                |                          1.015433 |                                1 |                      3.273253 |                  0.734133 | run_00189                       |                     0.419887 |            0.372800 |            0.726671 |               0.000000 |                  9.231095 |               9.231095 |
| dsp_phase_benefit_penalty_nnls                                |                 158 |        0.179998 |      0.793290 |             0.884568 |            0.893023 |                  0.084653 | run_00125                |                          1.015433 |                                1 |                      4.935735 |                  0.740535 | run_00026                       |                    -0.057273 |            0.955087 |            0.291693 |               4.687797 |                  1.000000 |               1.000000 |
| dsp_apple_repetition_shared_r1_phase_benefit_penalty_nnls     |                 159 |        0.184027 |      0.783932 |             0.877191 |            0.887590 |                  0.084653 | run_00125                |                          1.015433 |                                1 |                      4.547053 |                  0.782137 | run_00207                       |                     1.015062 |            1.000000 |            0.255732 |              16.684155 |                  1.000000 |               1.000000 |
| dsp_apple_repetition_per_domain_r1_phase_benefit_penalty_nnls |                 197 |        0.194039 |      0.759781 |             0.862864 |            0.874201 |                  0.084088 | run_00125                |                          1.015433 |                                1 |                      4.194061 |                  0.774628 | run_00191                       |                     0.747176 |            1.000000 |            0.224521 |              39.796571 |                  1.000000 |               1.000000 |

## Interpretation

- `score_oof_spearman` is the main rank-fit number: higher means the form ranks mixtures better for the aggregate.
- `score_cv_rmse` is in aggregate-score units. The observed aggregate standard deviation is included in `summary.csv` as `target_score_std`.
- Raw optima remain diagnostic only when `raw_nearest_observed_tv` is large or phase weights collapse.
