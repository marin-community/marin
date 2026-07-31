# MCT-LRQ Barrier Ablation

Date: 2026-04-24

## Conclusion

`P(w)` is not carrying the predictive result. On the ordinary held-out modeling rows its offset is zero, and removing it barely changes seed-7, fixed-340M, random-supplement, or leave-900M RMSE.

However, `P(w)` is carrying an important raw-optimum safety role. Without any barrier, the raw optimum collapses onto an observed but pathological 60M mixture with phase-1 almost entirely tech. A generic observed-support TV prior does not fix this, because the pathological point is itself in observed support.

The original barrier is therefore suspicious: it is active on only one observed training row, and that row is exactly the pathological raw optimum selected by the no-barrier model. This looks more like an ad hoc deployment safety prior than a clean law component.

The generic replacements tested here did not solve this cleanly. A broad `0.85` family-concentration barrier is too active and damages prediction. A near-family-collapse barrier keeps prediction mostly intact but still allows unobserved hard corners. Adding max-domain-collapse also degrades transfer and still does not produce a clean raw-simplex optimum under this fit.

Recommendation: use the barrier-free MCT law only as a predictive structural ablation, not as a raw-optimum deployment law. If raw optima matter, keep the original `P(w)` labeled as an ad hoc prior, or use constrained/hull deployment. Do not claim that current MCT has solved raw optimum quality in a clean way.

## Variants

| model                               |   fitted_param_count_counting_exponents |   total_constant_count |   anchor_feature_count_including_intercept |   scale_param_count |   donor_constant_count |   barrier_constant_count |
|:------------------------------------|----------------------------------------:|-----------------------:|-------------------------------------------:|--------------------:|-----------------------:|-------------------------:|
| mct_lrq74_balanced_barrier5         |                                      60 |                     74 |                                         47 |                   9 |                      9 |                        5 |
| mct_lrq71_balanced_family_barrier   |                                      60 |                     71 |                                         47 |                   9 |                      9 |                        2 |
| mct_lrq71_balanced_family_collapse  |                                      60 |                     71 |                                         47 |                   9 |                      9 |                        2 |
| mct_lrq73_balanced_generic_collapse |                                      60 |                     73 |                                         47 |                   9 |                      9 |                        4 |
| mct_lrq69_balanced_no_barrier       |                                      60 |                     69 |                                         47 |                   9 |                      9 |                        0 |
| mct_lrq69_drop_no_barrier           |                                      60 |                     69 |                                         47 |                   9 |                      9 |                        0 |
| mct_lrq69_balanced_support_tv       |                                      60 |                     69 |                                         47 |                   9 |                      9 |                        0 |

The `support_tv` variant wraps the barrier-free model only for raw optimum search. It does not change predictions on observed rows; it adds a generic penalty for candidate mixtures farther than TV `0.15` from observed support. It is included as a negative control because it cannot reject an observed pathological mixture.

`family_barrier` uses a broad `max_family_share > 0.85` prior and is too invasive. `family_collapse` uses a sparse `max_family_share > 0.98` prior. `generic_collapse` adds a `max_domain_weight > 0.60` prior on top. Neither sparse generic variant is a clean replacement for the original hand-coded barrier.

## Predictive Metrics

| model                               | fit_protocol   | split                  |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |   low_tail_rmse |
|:------------------------------------|:---------------|:-----------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|----------------:|
| mct_lrq69_balanced_no_barrier       | leave900out    | all900_leave_scale_out |   4 | 0.010306 |   0.800000 |                -0.002426 |               0.587351 |    0.687648 |        0.013976 |
| mct_lrq69_balanced_no_barrier       | seed7          | fixed340_holdout       |  27 | 0.006130 |   0.927961 |                 0.000144 |               0.942271 |    1.003268 |        0.005433 |
| mct_lrq69_balanced_no_barrier       | seed7          | random_supplement      |  34 | 0.012333 |   0.868602 |                 0.001926 |               1.095063 |    1.118146 |        0.011528 |
| mct_lrq69_balanced_no_barrier       | seed7          | seed7_holdout          |  61 | 0.010070 |   0.971021 |                 0.001137 |               1.036483 |    1.045967 |        0.005350 |
| mct_lrq69_balanced_support_tv       | leave900out    | all900_leave_scale_out |   4 | 0.010306 |   0.800000 |                -0.002426 |               0.587351 |    0.687648 |        0.013976 |
| mct_lrq69_balanced_support_tv       | seed7          | fixed340_holdout       |  27 | 0.006130 |   0.927961 |                 0.000144 |               0.942271 |    1.003268 |        0.005433 |
| mct_lrq69_balanced_support_tv       | seed7          | random_supplement      |  34 | 0.012333 |   0.868602 |                 0.001926 |               1.095063 |    1.118146 |        0.011528 |
| mct_lrq69_balanced_support_tv       | seed7          | seed7_holdout          |  61 | 0.010070 |   0.971021 |                 0.001137 |               1.036483 |    1.045967 |        0.005350 |
| mct_lrq69_drop_no_barrier           | leave900out    | all900_leave_scale_out |   4 | 0.010177 |   1.000000 |                -0.001833 |               0.576529 |    0.671466 |        0.014610 |
| mct_lrq69_drop_no_barrier           | seed7          | fixed340_holdout       |  27 | 0.006143 |   0.923687 |                 0.000535 |               0.952454 |    1.013199 |        0.005529 |
| mct_lrq69_drop_no_barrier           | seed7          | random_supplement      |  34 | 0.012334 |   0.868908 |                 0.001924 |               1.094590 |    1.117728 |        0.011551 |
| mct_lrq69_drop_no_barrier           | seed7          | seed7_holdout          |  61 | 0.010075 |   0.970703 |                 0.001309 |               1.034296 |    1.043839 |        0.005383 |
| mct_lrq71_balanced_family_barrier   | leave900out    | all900_leave_scale_out |   4 | 0.067009 |   0.800000 |                -0.056869 |               1.489419 |    2.389911 |        0.043189 |
| mct_lrq71_balanced_family_barrier   | seed7          | fixed340_holdout       |  27 | 0.016343 |   0.624542 |                -0.005288 |               0.676662 |    1.063120 |        0.013868 |
| mct_lrq71_balanced_family_barrier   | seed7          | random_supplement      |  34 | 0.021944 |   0.855462 |                -0.003107 |               1.317495 |    1.351361 |        0.052166 |
| mct_lrq71_balanced_family_barrier   | seed7          | seed7_holdout          |  61 | 0.019663 |   0.942464 |                -0.004072 |               1.095664 |    1.126603 |        0.040914 |
| mct_lrq71_balanced_family_collapse  | leave900out    | all900_leave_scale_out |   4 | 0.044743 |   0.800000 |                -0.022160 |               0.849496 |    2.277906 |        0.013965 |
| mct_lrq71_balanced_family_collapse  | seed7          | fixed340_holdout       |  27 | 0.006247 |   0.923687 |                 0.000114 |               0.946547 |    1.009873 |        0.005597 |
| mct_lrq71_balanced_family_collapse  | seed7          | random_supplement      |  34 | 0.019238 |   0.871658 |                -0.000411 |               1.283119 |    1.310004 |        0.044442 |
| mct_lrq71_balanced_family_collapse  | seed7          | seed7_holdout          |  61 | 0.014952 |   0.971179 |                -0.000179 |               1.077352 |    1.096249 |        0.032897 |
| mct_lrq73_balanced_generic_collapse | leave900out    | all900_leave_scale_out |   4 | 0.125880 |   0.800000 |                -0.063182 |               1.568555 |    6.112551 |        0.009339 |
| mct_lrq73_balanced_generic_collapse | seed7          | fixed340_holdout       |  27 | 0.006611 |   0.923687 |                 0.000072 |               0.934758 |    1.005740 |        0.005668 |
| mct_lrq73_balanced_generic_collapse | seed7          | random_supplement      |  34 | 0.043433 |   0.868908 |                -0.004917 |               1.655282 |    1.750728 |        0.121704 |
| mct_lrq73_balanced_generic_collapse | seed7          | seed7_holdout          |  61 | 0.032723 |   0.970703 |                -0.002709 |               1.158240 |    1.241119 |        0.091744 |
| mct_lrq74_balanced_barrier5         | leave900out    | all900_leave_scale_out |   4 | 0.010358 |   1.000000 |                -0.001646 |               0.525503 |    0.601499 |        0.015142 |
| mct_lrq74_balanced_barrier5         | seed7          | fixed340_holdout       |  27 | 0.006068 |   0.931013 |                 0.000153 |               0.946642 |    1.006390 |        0.005444 |
| mct_lrq74_balanced_barrier5         | seed7          | random_supplement      |  34 | 0.012299 |   0.869213 |                 0.001907 |               1.095154 |    1.118086 |        0.011476 |
| mct_lrq74_balanced_barrier5         | seed7          | seed7_holdout          |  61 | 0.010030 |   0.971391 |                 0.001131 |               1.036464 |    1.045867 |        0.005480 |

## Fixed-340M Same-Mixture Drops

| model                               | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:------------------------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| mct_lrq69_balanced_no_barrier       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023414 |         -0.000222 |          1.017876 |            0.926758 |    0.004202 |
| mct_lrq69_balanced_no_barrier       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044094 |         -0.004423 |          0.908029 |            0.918784 |    0.004568 |
| mct_lrq69_balanced_no_barrier       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020260 |         -0.000946 |          0.954820 |            0.963563 |    0.001209 |
| mct_lrq69_balanced_support_tv       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023414 |         -0.000222 |          1.017876 |            0.926758 |    0.004202 |
| mct_lrq69_balanced_support_tv       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044094 |         -0.004423 |          0.908029 |            0.918784 |    0.004568 |
| mct_lrq69_balanced_support_tv       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020260 |         -0.000946 |          0.954820 |            0.963563 |    0.001209 |
| mct_lrq69_drop_no_barrier           | 0.5x_to_1.0x |  12 |           0.023636 |         0.023575 |         -0.000061 |          1.024984 |            0.935621 |    0.004194 |
| mct_lrq69_drop_no_barrier           | 0.5x_to_2.0x |   3 |           0.048517 |         0.044814 |         -0.003703 |          0.922907 |            0.933046 |    0.003865 |
| mct_lrq69_drop_no_barrier           | 1.0x_to_2.0x |   3 |           0.021206 |         0.020905 |         -0.000301 |          0.985248 |            0.993167 |    0.000822 |
| mct_lrq71_balanced_family_barrier   | 0.5x_to_1.0x |  12 |           0.023636 |         0.017684 |         -0.005951 |          0.775549 |            0.724875 |    0.007876 |
| mct_lrq71_balanced_family_barrier   | 0.5x_to_2.0x |   3 |           0.048517 |         0.025142 |         -0.023375 |          0.519965 |            0.488130 |    0.023626 |
| mct_lrq71_balanced_family_barrier   | 1.0x_to_2.0x |   3 |           0.021206 |         0.010541 |         -0.010665 |          0.498702 |            0.480274 |    0.010773 |
| mct_lrq71_balanced_family_collapse  | 0.5x_to_1.0x |  12 |           0.023636 |         0.023629 |         -0.000007 |          1.027290 |            0.936432 |    0.004237 |
| mct_lrq71_balanced_family_collapse  | 0.5x_to_2.0x |   3 |           0.048517 |         0.044468 |         -0.004049 |          0.915730 |            0.926734 |    0.004214 |
| mct_lrq71_balanced_family_collapse  | 1.0x_to_2.0x |   3 |           0.021206 |         0.020468 |         -0.000738 |          0.964611 |            0.973415 |    0.001063 |
| mct_lrq73_balanced_generic_collapse | 0.5x_to_1.0x |  12 |           0.023636 |         0.023963 |          0.000327 |          1.043920 |            0.942094 |    0.004752 |
| mct_lrq73_balanced_generic_collapse | 0.5x_to_2.0x |   3 |           0.048517 |         0.042959 |         -0.005558 |          0.885017 |            0.890188 |    0.005582 |
| mct_lrq73_balanced_generic_collapse | 1.0x_to_2.0x |   3 |           0.021206 |         0.019776 |         -0.001431 |          0.932327 |            0.929293 |    0.001528 |
| mct_lrq74_balanced_barrier5         | 0.5x_to_1.0x |  12 |           0.023636 |         0.023328 |         -0.000308 |          1.013638 |            0.920644 |    0.004076 |
| mct_lrq74_balanced_barrier5         | 0.5x_to_2.0x |   3 |           0.048517 |         0.044695 |         -0.003821 |          0.920379 |            0.931530 |    0.004019 |
| mct_lrq74_balanced_barrier5         | 1.0x_to_2.0x |   3 |           0.021206 |         0.020535 |         -0.000671 |          0.967724 |            0.978337 |    0.001037 |

## Barrier / Offset Activity On Observed Rows

| model                               | split             |   n |   nonzero_count |   nonzero_frac |   max_offset |   mean_offset |   p95_offset |
|:------------------------------------|:------------------|----:|----------------:|---------------:|-------------:|--------------:|-------------:|
| mct_lrq69_balanced_no_barrier       | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | all_rows          | 640 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | seed7_train       | 579 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | all_rows          | 640 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | seed7_train       | 579 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | all_rows          | 640 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | seed7_train       | 579 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq71_balanced_family_barrier   | all900_holdout    |   4 |               3 |       0.750000 |     0.078694 |      0.040875 |     0.075138 |
| mct_lrq71_balanced_family_barrier   | all_rows          | 640 |             270 |       0.421875 |     0.138190 |      0.004602 |     0.029820 |
| mct_lrq71_balanced_family_barrier   | fixed340_holdout  |  27 |               8 |       0.296296 |     0.054988 |      0.009570 |     0.054988 |
| mct_lrq71_balanced_family_barrier   | random_supplement |  34 |              15 |       0.441176 |     0.078694 |      0.005354 |     0.027132 |
| mct_lrq71_balanced_family_barrier   | seed7_holdout     |  61 |              23 |       0.377049 |     0.078694 |      0.007220 |     0.054988 |
| mct_lrq71_balanced_family_barrier   | seed7_train       | 579 |             247 |       0.426598 |     0.138190 |      0.004326 |     0.029820 |
| mct_lrq71_balanced_family_collapse  | all900_holdout    |   4 |               1 |       0.250000 |     0.079576 |      0.019894 |     0.067640 |
| mct_lrq71_balanced_family_collapse  | all_rows          | 640 |               8 |       0.012500 |     0.079993 |      0.000825 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | random_supplement |  34 |               1 |       0.029412 |     0.079576 |      0.002340 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | seed7_holdout     |  61 |               1 |       0.016393 |     0.079576 |      0.001305 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | seed7_train       | 579 |               7 |       0.012090 |     0.079993 |      0.000775 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | all900_holdout    |   4 |               1 |       0.250000 |     0.231168 |      0.057792 |     0.196493 |
| mct_lrq73_balanced_generic_collapse | all_rows          | 640 |              10 |       0.015625 |     0.231168 |      0.001795 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | random_supplement |  34 |               1 |       0.029412 |     0.231168 |      0.006799 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | seed7_holdout     |  61 |               1 |       0.016393 |     0.231168 |      0.003790 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | seed7_train       | 579 |               9 |       0.015544 |     0.231168 |      0.001585 |     0.000000 |
| mct_lrq74_balanced_barrier5         | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | all_rows          | 640 |               1 |       0.001563 |     0.077626 |      0.000121 |     0.000000 |
| mct_lrq74_balanced_barrier5         | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | seed7_train       | 579 |               1 |       0.001727 |     0.077626 |      0.000134 |     0.000000 |

## Raw And Constrained Optima

| model                               | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   nearest_observed_phase_mean_tv |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |
|:------------------------------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | raw_random_search             |        0.831085 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | top8actual_hull_random_search |        0.843170 | False              | False                       | False                      |                         0.221662 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.831085 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | raw_random_search             |        0.749205 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | top8actual_hull_random_search |        0.772646 | False              | False                       | False                      |                         0.206398 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.749205 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | raw_random_search             |        0.831085 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | top8actual_hull_random_search |        0.843995 | False              | False                       | False                      |                         0.199512 |        0.000613 |              0.716240 |             0.255503 |             0.028256 |              0.717426 |             0.252984 |             0.029590 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.831085 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | raw_random_search             |        0.749205 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | top8actual_hull_random_search |        0.773139 | False              | False                       | False                      |                         0.185846 |        0.000321 |              0.744353 |             0.232471 |             0.023176 |              0.728845 |             0.248466 |             0.022689 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.749205 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | raw_random_search             |        0.831192 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | top8actual_hull_random_search |        0.843535 | False              | False                       | False                      |                         0.221662 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.831192 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | raw_random_search             |        0.748350 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | top8actual_hull_random_search |        0.772922 | False              | False                       | False                      |                         0.206398 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.748350 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | raw_random_search             |        0.711991 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | top8actual_hull_random_search |        0.832469 | False              | False                       | False                      |                         0.108574 |        0.000000 |              0.665450 |             0.315126 |             0.019423 |              0.636793 |             0.349999 |             0.013208 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.711991 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | raw_random_search             |        0.620030 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | top8actual_hull_random_search |        0.765580 | False              | False                       | False                      |                         0.108574 |        0.000000 |              0.665450 |             0.315126 |             0.019423 |              0.636793 |             0.349999 |             0.013208 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.620030 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_collapse  | 340M/10.4B     | raw_random_search             |        0.750436 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_collapse  | 340M/10.4B     | top8actual_hull_random_search |        0.843224 | False              | False                       | False                      |                         0.221662 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |
| mct_lrq71_balanced_family_collapse  | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.750436 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_collapse  | 900M/24B       | raw_random_search             |        0.667777 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_collapse  | 900M/24B       | top8actual_hull_random_search |        0.772564 | False              | False                       | False                      |                         0.206398 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |
| mct_lrq71_balanced_family_collapse  | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.667777 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | raw_random_search             |        0.591135 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | top8actual_hull_random_search |        0.843011 | False              | False                       | False                      |                         0.220282 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.591135 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | raw_random_search             |        0.499800 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | top8actual_hull_random_search |        0.772207 | False              | False                       | False                      |                         0.219037 |        0.000000 |              0.722755 |             0.253432 |             0.023814 |              0.694884 |             0.282321 |             0.022795 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.499800 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | raw_random_search             |        0.846379 | False              | False                       | False                      |                         0.000000 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | top8actual_hull_random_search |        0.843121 | False              | False                       | False                      |                         0.221662 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.842036 | False              | False                       | False                      |                         0.149726 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | raw_random_search             |        0.775238 | False              | False                       | False                      |                         0.000000 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | top8actual_hull_random_search |        0.772450 | False              | False                       | False                      |                         0.206398 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.771908 | False              | False                       | False                      |                         0.149941 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |

## Monotonicity Grid

| model                               |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:------------------------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_balanced_barrier5         |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037600 |                          0.000639 | True                                  |       0.437199 |       0.000855 |        0.001610 |        0.001610 |           0.001610 |
| mct_lrq71_balanced_family_barrier   |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037916 |                          0.009872 | True                                  |       0.440766 |       0.054189 |        0.009306 |        0.009306 |           0.009306 |
| mct_lrq71_balanced_family_collapse  |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037582 |                          0.000536 | True                                  |       0.436999 |       0.000851 |        0.001290 |        0.001290 |           0.001290 |
| mct_lrq73_balanced_generic_collapse |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037625 |                          0.001567 | True                                  |       0.437495 |       0.009589 |        0.001227 |        0.001227 |           0.001227 |
| mct_lrq69_balanced_no_barrier       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037624 |                          0.000625 | True                                  |       0.437485 |       0.000839 |        0.001574 |        0.001574 |           0.001574 |
| mct_lrq69_drop_no_barrier           |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037509 |                          0.000995 | True                                  |       0.419021 |       0.001151 |        0.002742 |        0.002742 |           0.002742 |
| mct_lrq69_balanced_support_tv       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037624 |                          0.000625 | True                                  |       0.437485 |       0.000839 |        0.001574 |        0.001574 |           0.001574 |

## Artifact Map

- `csv/metric_summary.csv`: split metrics for every variant.
- `csv/fixed340_drop_summary.csv`: same-mixture multiplier drop ratios.
- `csv/offset_activity.csv`: observed-row offset activity for the barrier and support-TV wrapper.
- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.
- `plots/pred_actual_barrier_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.
- `plots/rmse_barrier_ablation.png`: compact RMSE comparison.
- `plots/optimum_family_shares_340m.png`: 340M optimum family shares.
- `code/`: the local ablation runner plus the Session-12 MCT helper code used for reproduction.
