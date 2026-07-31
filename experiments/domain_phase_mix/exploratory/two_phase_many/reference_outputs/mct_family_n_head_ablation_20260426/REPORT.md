# MCT-LRQ Family-N Head Ablation

Date: 2026-04-26

## Question

Does the compact structural MCT-LRQ law improve if both the N and D scale heads are family-dependent? The previous canonical barrier-free law used a global N head and a family-dependent D head.

## Variants

- `mct_lrq69_drop_Dfam`: drop-tuned exponents, global `A`, family `B`, global `C`.
- `mct_lrq75_drop_Nfam_Dfam`: drop-tuned exponents, family `A`, family `B`, global `C`.
- `mct_lrq69_balanced_Dfam`: balanced exponents, global `A`, family `B`, global `C`.
- `mct_lrq75_balanced_Nfam_Dfam`: balanced exponents, family `A`, family `B`, global `C`.

The `Nfam` variants add six parameters: `A(w)` changes from one nonnegative scalar to a nonnegative linear head over the intercept plus the six phase-family shares.

## Compact Summary

| model                        |   seed7_holdout_rmse |   fixed340_rmse |   fixed340_slope |   fixed340_std_ratio |   all900_leaveout_rmse |   total_constant_count |   scale_param_count |
|:-----------------------------|---------------------:|----------------:|-----------------:|---------------------:|-----------------------:|-----------------------:|--------------------:|
| mct_lrq69_balanced_Dfam      |             0.010125 |        0.006134 |         0.949477 |             1.010539 |               0.010590 |                     69 |                   9 |
| mct_lrq69_drop_Dfam          |             0.010136 |        0.006138 |         0.960968 |             1.021814 |               0.010522 |                     69 |                   9 |
| mct_lrq75_drop_Nfam_Dfam     |             0.011690 |        0.011978 |         0.808228 |             1.039706 |               0.024795 |                     75 |                  15 |
| mct_lrq75_balanced_Nfam_Dfam |             0.011661 |        0.011988 |         0.797228 |             1.029866 |               0.026504 |                     75 |                  15 |

## Predictive Metrics

| model                        | fit_protocol   | split                  |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |   low_tail_rmse |
|:-----------------------------|:---------------|:-----------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|----------------:|
| mct_lrq69_balanced_Dfam      | leave900out    | all900_leave_scale_out |   4 | 0.010590 |   0.800000 |                -0.003152 |               0.590796 |    0.696690 |        0.013371 |
| mct_lrq69_balanced_Dfam      | seed7          | fixed340_holdout       |  27 | 0.006134 |   0.927961 |                -0.000085 |               0.949477 |    1.010539 |        0.005385 |
| mct_lrq69_balanced_Dfam      | seed7          | random_supplement      |  34 | 0.012411 |   0.868602 |                 0.001952 |               1.097435 |    1.120597 |        0.011654 |
| mct_lrq69_balanced_Dfam      | seed7          | seed7_holdout          |  61 | 0.010125 |   0.971021 |                 0.001050 |               1.038894 |    1.048399 |        0.005317 |
| mct_lrq69_drop_Dfam          | leave900out    | all900_leave_scale_out |   4 | 0.010522 |   0.800000 |                -0.002879 |               0.581613 |    0.683804 |        0.013715 |
| mct_lrq69_drop_Dfam          | seed7          | fixed340_holdout       |  27 | 0.006138 |   0.923687 |                 0.000265 |               0.960968 |    1.021814 |        0.005483 |
| mct_lrq69_drop_Dfam          | seed7          | random_supplement      |  34 | 0.012426 |   0.868602 |                 0.001954 |               1.097401 |    1.120629 |        0.011700 |
| mct_lrq69_drop_Dfam          | seed7          | seed7_holdout          |  61 | 0.010136 |   0.970650 |                 0.001206 |               1.037137 |    1.046708 |        0.005344 |
| mct_lrq75_balanced_Nfam_Dfam | leave900out    | all900_leave_scale_out |   4 | 0.026504 |   0.800000 |                -0.009642 |               1.024293 |    1.691117 |        0.010085 |
| mct_lrq75_balanced_Nfam_Dfam | seed7          | fixed340_holdout       |  27 | 0.011988 |   0.763736 |                 0.000465 |               0.797228 |    1.029866 |        0.010750 |
| mct_lrq75_balanced_Nfam_Dfam | seed7          | random_supplement      |  34 | 0.011395 |   0.871963 |                 0.001338 |               1.055448 |    1.078320 |        0.010481 |
| mct_lrq75_balanced_Nfam_Dfam | seed7          | seed7_holdout          |  61 | 0.011661 |   0.957377 |                 0.000952 |               1.015442 |    1.029233 |        0.011312 |
| mct_lrq75_drop_Nfam_Dfam     | leave900out    | all900_leave_scale_out |   4 | 0.024795 |   0.800000 |                -0.008335 |               0.947482 |    1.586056 |        0.012430 |
| mct_lrq75_drop_Nfam_Dfam     | seed7          | fixed340_holdout       |  27 | 0.011978 |   0.765568 |                 0.000705 |               0.808228 |    1.039706 |        0.010737 |
| mct_lrq75_drop_Nfam_Dfam     | seed7          | random_supplement      |  34 | 0.011456 |   0.871963 |                 0.001470 |               1.045888 |    1.069612 |        0.010808 |
| mct_lrq75_drop_Nfam_Dfam     | seed7          | seed7_holdout          |  61 | 0.011690 |   0.957536 |                 0.001132 |               1.012342 |    1.026248 |        0.011482 |

## Fixed-340M Same-Mixture Drops

| model                        | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-----------------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| mct_lrq69_balanced_Dfam      | 0.5x_to_1.0x |  12 |           0.023636 |         0.023631 |         -0.000005 |          1.027316 |            0.935608 |    0.004223 |
| mct_lrq69_balanced_Dfam      | 0.5x_to_2.0x |   3 |           0.048517 |         0.044535 |         -0.003982 |          0.917113 |            0.928140 |    0.004152 |
| mct_lrq69_balanced_Dfam      | 1.0x_to_2.0x |   3 |           0.021206 |         0.020496 |         -0.000710 |          0.965928 |            0.974888 |    0.001046 |
| mct_lrq69_drop_Dfam          | 0.5x_to_1.0x |  12 |           0.023636 |         0.023830 |          0.000194 |          1.036086 |            0.946100 |    0.004229 |
| mct_lrq69_drop_Dfam          | 0.5x_to_2.0x |   3 |           0.048517 |         0.045335 |         -0.003182 |          0.933632 |            0.944064 |    0.003383 |
| mct_lrq69_drop_Dfam          | 1.0x_to_2.0x |   3 |           0.021206 |         0.021187 |         -0.000019 |          0.998532 |            1.006663 |    0.000783 |
| mct_lrq75_balanced_Nfam_Dfam | 0.5x_to_1.0x |  12 |           0.023636 |         0.021043 |         -0.002592 |          0.914182 |            0.833084 |    0.004637 |
| mct_lrq75_balanced_Nfam_Dfam | 0.5x_to_2.0x |   3 |           0.048517 |         0.040038 |         -0.008479 |          0.824538 |            0.832940 |    0.008523 |
| mct_lrq75_balanced_Nfam_Dfam | 1.0x_to_2.0x |   3 |           0.021206 |         0.018075 |         -0.003132 |          0.851803 |            0.860160 |    0.003195 |
| mct_lrq75_drop_Nfam_Dfam     | 0.5x_to_1.0x |  12 |           0.023636 |         0.021219 |         -0.002417 |          0.921831 |            0.840211 |    0.004532 |
| mct_lrq75_drop_Nfam_Dfam     | 0.5x_to_2.0x |   3 |           0.048517 |         0.040887 |         -0.007630 |          0.842051 |            0.850188 |    0.007680 |
| mct_lrq75_drop_Nfam_Dfam     | 1.0x_to_2.0x |   3 |           0.021206 |         0.018761 |         -0.002446 |          0.884157 |            0.892423 |    0.002532 |

## Raw And Constrained Optima

| model                        | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   nearest_observed_phase_mean_tv |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |
|:-----------------------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|
| mct_lrq69_balanced_Dfam      | 340M/10.4B     | raw_random_search             |        0.830584 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_Dfam      | 340M/10.4B     | top8actual_hull_random_search |        0.842438 | False              | False                       | False                      |                         0.192892 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_balanced_Dfam      | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.830584 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_Dfam      | 900M/24B       | raw_random_search             |        0.748057 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_Dfam      | 900M/24B       | top8actual_hull_random_search |        0.771577 | False              | False                       | False                      |                         0.192892 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_balanced_Dfam      | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.748057 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_Dfam          | 340M/10.4B     | raw_random_search             |        0.830598 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_Dfam          | 340M/10.4B     | top8actual_hull_random_search |        0.842756 | False              | False                       | False                      |                         0.192892 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_drop_Dfam          | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.830598 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_Dfam          | 900M/24B       | raw_random_search             |        0.746963 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_Dfam          | 900M/24B       | top8actual_hull_random_search |        0.771721 | False              | False                       | False                      |                         0.192892 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_drop_Dfam          | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.746963 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_balanced_Nfam_Dfam | 340M/10.4B     | raw_random_search             |        0.786254 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_balanced_Nfam_Dfam | 340M/10.4B     | top8actual_hull_random_search |        0.850699 | False              | False                       | False                      |                         0.191890 |              0.734597 |             0.240144 |             0.025259 |              0.705954 |             0.273775 |             0.020271 |
| mct_lrq75_balanced_Nfam_Dfam | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.786254 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_balanced_Nfam_Dfam | 900M/24B       | raw_random_search             |        0.677554 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_balanced_Nfam_Dfam | 900M/24B       | top8actual_hull_random_search |        0.785405 | False              | False                       | False                      |                         0.173965 |              0.774683 |             0.204730 |             0.020586 |              0.736817 |             0.247226 |             0.015957 |
| mct_lrq75_balanced_Nfam_Dfam | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.677554 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_drop_Nfam_Dfam     | 340M/10.4B     | raw_random_search             |        0.788485 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_drop_Nfam_Dfam     | 340M/10.4B     | top8actual_hull_random_search |        0.851205 | False              | False                       | False                      |                         0.191890 |              0.734597 |             0.240144 |             0.025259 |              0.705954 |             0.273775 |             0.020271 |
| mct_lrq75_drop_Nfam_Dfam     | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.788485 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_drop_Nfam_Dfam     | 900M/24B       | raw_random_search             |        0.681041 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq75_drop_Nfam_Dfam     | 900M/24B       | top8actual_hull_random_search |        0.785764 | False              | False                       | False                      |                         0.173965 |              0.774683 |             0.204730 |             0.020586 |              0.736817 |             0.247226 |             0.015957 |
| mct_lrq75_drop_Nfam_Dfam     | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.681041 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |

## Monotonicity Grid

| model                        |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-----------------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq69_drop_Dfam          |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037587 |                          0.000914 | True                                  |       0.419899 |       0.001171 |        0.002482 |        0.002482 |           0.002482 |
| mct_lrq75_drop_Nfam_Dfam     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.002897 |                          0.001649 | True                                  |       0.032270 |       0.001004 |        0.004847 |        0.004847 |           0.004847 |
| mct_lrq69_balanced_Dfam      |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037690 |                          0.000544 | True                                  |       0.438255 |       0.000852 |        0.001316 |        0.001316 |           0.001316 |
| mct_lrq75_balanced_Nfam_Dfam |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.003435 |                          0.001422 | True                                  |       0.039887 |       0.000722 |        0.004107 |        0.004107 |           0.004107 |

## Artifact Map

- `csv/metric_summary.csv`: split metrics.
- `csv/row_predictions.csv`: row-level predictions.
- `csv/fixed340_drop_summary.csv`: same-mixture target-budget drop metrics.
- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.
- `plots/rmse_family_n_head_ablation.png`: RMSE comparison.
- `plots/pred_actual_family_n_head_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.
- `plots/drop_ratios_family_n_head_ablation.png`: fixed-340M drop-ratio comparison.
