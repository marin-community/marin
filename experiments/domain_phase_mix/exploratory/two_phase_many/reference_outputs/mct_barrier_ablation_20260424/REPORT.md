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
| mct_lrq69_balanced_no_barrier       | leave900out    | all900_leave_scale_out |   4 | 0.010590 |   0.800000 |                -0.003152 |               0.590796 |    0.696690 |        0.013371 |
| mct_lrq69_balanced_no_barrier       | seed7          | fixed340_holdout       |  27 | 0.006134 |   0.927961 |                -0.000085 |               0.949477 |    1.010539 |        0.005385 |
| mct_lrq69_balanced_no_barrier       | seed7          | random_supplement      |  34 | 0.012411 |   0.868602 |                 0.001952 |               1.097435 |    1.120597 |        0.011654 |
| mct_lrq69_balanced_no_barrier       | seed7          | seed7_holdout          |  61 | 0.010125 |   0.971021 |                 0.001050 |               1.038894 |    1.048399 |        0.005317 |
| mct_lrq69_balanced_support_tv       | leave900out    | all900_leave_scale_out |   4 | 0.010590 |   0.800000 |                -0.003152 |               0.590796 |    0.696690 |        0.013371 |
| mct_lrq69_balanced_support_tv       | seed7          | fixed340_holdout       |  27 | 0.006134 |   0.927961 |                -0.000085 |               0.949477 |    1.010539 |        0.005385 |
| mct_lrq69_balanced_support_tv       | seed7          | random_supplement      |  34 | 0.012411 |   0.868602 |                 0.001952 |               1.097435 |    1.120597 |        0.011654 |
| mct_lrq69_balanced_support_tv       | seed7          | seed7_holdout          |  61 | 0.010125 |   0.971021 |                 0.001050 |               1.038894 |    1.048399 |        0.005317 |
| mct_lrq69_drop_no_barrier           | leave900out    | all900_leave_scale_out |   4 | 0.010522 |   0.800000 |                -0.002879 |               0.581613 |    0.683804 |        0.013715 |
| mct_lrq69_drop_no_barrier           | seed7          | fixed340_holdout       |  27 | 0.006138 |   0.923687 |                 0.000265 |               0.960968 |    1.021814 |        0.005483 |
| mct_lrq69_drop_no_barrier           | seed7          | random_supplement      |  34 | 0.012426 |   0.868602 |                 0.001954 |               1.097401 |    1.120629 |        0.011700 |
| mct_lrq69_drop_no_barrier           | seed7          | seed7_holdout          |  61 | 0.010136 |   0.970650 |                 0.001206 |               1.037137 |    1.046708 |        0.005344 |
| mct_lrq71_balanced_family_barrier   | leave900out    | all900_leave_scale_out |   4 | 0.032103 |   1.000000 |                -0.018089 |               2.306200 |    2.388022 |        0.045380 |
| mct_lrq71_balanced_family_barrier   | seed7          | fixed340_holdout       |  27 | 0.010489 |   0.820513 |                 0.004436 |               0.724347 |    0.861524 |        0.013859 |
| mct_lrq71_balanced_family_barrier   | seed7          | random_supplement      |  34 | 0.013572 |   0.879908 |                 0.002228 |               1.141438 |    1.164259 |        0.014638 |
| mct_lrq71_balanced_family_barrier   | seed7          | seed7_holdout          |  61 | 0.012303 |   0.963670 |                 0.003205 |               1.019920 |    1.034189 |        0.018383 |
| mct_lrq71_balanced_family_collapse  | leave900out    | all900_leave_scale_out |   4 | 0.010425 |   1.000000 |                -0.003072 |               0.576702 |    0.669566 |        0.013288 |
| mct_lrq71_balanced_family_collapse  | seed7          | fixed340_holdout       |  27 | 0.006258 |   0.927961 |                -0.000164 |               0.954253 |    1.017665 |        0.005552 |
| mct_lrq71_balanced_family_collapse  | seed7          | random_supplement      |  34 | 0.012474 |   0.871658 |                 0.001963 |               1.097641 |    1.121056 |        0.011632 |
| mct_lrq71_balanced_family_collapse  | seed7          | seed7_holdout          |  61 | 0.010201 |   0.971549 |                 0.001021 |               1.039625 |    1.049258 |        0.005357 |
| mct_lrq73_balanced_generic_collapse | leave900out    | all900_leave_scale_out |   4 | 0.013484 |   0.800000 |                -0.007256 |               0.771838 |    0.963072 |        0.007731 |
| mct_lrq73_balanced_generic_collapse | seed7          | fixed340_holdout       |  27 | 0.006628 |   0.921245 |                -0.000378 |               0.944865 |    1.015876 |        0.005453 |
| mct_lrq73_balanced_generic_collapse | seed7          | random_supplement      |  34 | 0.012901 |   0.868908 |                 0.001928 |               1.113634 |    1.137185 |        0.012770 |
| mct_lrq73_balanced_generic_collapse | seed7          | seed7_holdout          |  61 | 0.010593 |   0.970492 |                 0.000907 |               1.045053 |    1.055252 |        0.006491 |
| mct_lrq74_balanced_barrier5         | leave900out    | all900_leave_scale_out |   4 | 0.010434 |   1.000000 |                -0.001962 |               0.528150 |    0.606958 |        0.014896 |
| mct_lrq74_balanced_barrier5         | seed7          | fixed340_holdout       |  27 | 0.006073 |   0.931013 |                -0.000029 |               0.953371 |    1.013185 |        0.005432 |
| mct_lrq74_balanced_barrier5         | seed7          | random_supplement      |  34 | 0.012361 |   0.869213 |                 0.001927 |               1.097039 |    1.120033 |        0.011587 |
| mct_lrq74_balanced_barrier5         | seed7          | seed7_holdout          |  61 | 0.010074 |   0.971391 |                 0.001061 |               1.038405 |    1.047825 |        0.005472 |

## Fixed-340M Same-Mixture Drops

| model                               | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:------------------------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| mct_lrq69_balanced_no_barrier       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023631 |         -0.000005 |          1.027316 |            0.935608 |    0.004223 |
| mct_lrq69_balanced_no_barrier       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044535 |         -0.003982 |          0.917113 |            0.928140 |    0.004152 |
| mct_lrq69_balanced_no_barrier       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020496 |         -0.000710 |          0.965928 |            0.974888 |    0.001046 |
| mct_lrq69_balanced_support_tv       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023631 |         -0.000005 |          1.027316 |            0.935608 |    0.004223 |
| mct_lrq69_balanced_support_tv       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044535 |         -0.003982 |          0.917113 |            0.928140 |    0.004152 |
| mct_lrq69_balanced_support_tv       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020496 |         -0.000710 |          0.965928 |            0.974888 |    0.001046 |
| mct_lrq69_drop_no_barrier           | 0.5x_to_1.0x |  12 |           0.023636 |         0.023830 |          0.000194 |          1.036086 |            0.946100 |    0.004229 |
| mct_lrq69_drop_no_barrier           | 0.5x_to_2.0x |   3 |           0.048517 |         0.045335 |         -0.003182 |          0.933632 |            0.944064 |    0.003383 |
| mct_lrq69_drop_no_barrier           | 1.0x_to_2.0x |   3 |           0.021206 |         0.021187 |         -0.000019 |          0.998532 |            1.006663 |    0.000783 |
| mct_lrq71_balanced_family_barrier   | 0.5x_to_1.0x |  12 |           0.023636 |         0.017547 |         -0.006089 |          0.769640 |            0.719468 |    0.007989 |
| mct_lrq71_balanced_family_barrier   | 0.5x_to_2.0x |   3 |           0.048517 |         0.024809 |         -0.023708 |          0.513096 |            0.481331 |    0.023957 |
| mct_lrq71_balanced_family_barrier   | 1.0x_to_2.0x |   3 |           0.021206 |         0.010367 |         -0.010840 |          0.490480 |            0.471904 |    0.010946 |
| mct_lrq71_balanced_family_collapse  | 0.5x_to_1.0x |  12 |           0.023636 |         0.023860 |          0.000224 |          1.037342 |            0.945830 |    0.004272 |
| mct_lrq71_balanced_family_collapse  | 0.5x_to_2.0x |   3 |           0.048517 |         0.044941 |         -0.003575 |          0.925472 |            0.936771 |    0.003774 |
| mct_lrq71_balanced_family_collapse  | 1.0x_to_2.0x |   3 |           0.021206 |         0.020722 |         -0.000485 |          0.976546 |            0.985590 |    0.000920 |
| mct_lrq73_balanced_generic_collapse | 0.5x_to_1.0x |  12 |           0.023636 |         0.024272 |          0.000636 |          1.057460 |            0.953855 |    0.004832 |
| mct_lrq73_balanced_generic_collapse | 0.5x_to_2.0x |   3 |           0.048517 |         0.043557 |         -0.004960 |          0.897339 |            0.902579 |    0.004989 |
| mct_lrq73_balanced_generic_collapse | 1.0x_to_2.0x |   3 |           0.021206 |         0.020100 |         -0.001107 |          0.947602 |            0.944367 |    0.001233 |
| mct_lrq74_balanced_barrier5         | 0.5x_to_1.0x |  12 |           0.023636 |         0.023530 |         -0.000106 |          1.022437 |            0.929048 |    0.004088 |
| mct_lrq74_balanced_barrier5         | 0.5x_to_2.0x |   3 |           0.048517 |         0.045108 |         -0.003409 |          0.928873 |            0.940274 |    0.003641 |
| mct_lrq74_balanced_barrier5         | 1.0x_to_2.0x |   3 |           0.021206 |         0.020755 |         -0.000451 |          0.978074 |            0.988892 |    0.000923 |

## Barrier / Offset Activity On Observed Rows

| model                               | split             |   n |   nonzero_count |   nonzero_frac |   max_offset |   mean_offset |   p95_offset |
|:------------------------------------|:------------------|----:|----------------:|---------------:|-------------:|--------------:|-------------:|
| mct_lrq69_balanced_no_barrier       | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | all_rows          | 631 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_no_barrier       | seed7_train       | 570 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | all_rows          | 631 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_balanced_support_tv       | seed7_train       | 570 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | all_rows          | 631 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq69_drop_no_barrier           | seed7_train       | 570 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq71_balanced_family_barrier   | all900_holdout    |   4 |               3 |       0.750000 |     0.078694 |      0.040875 |     0.075138 |
| mct_lrq71_balanced_family_barrier   | all_rows          | 631 |             269 |       0.426307 |     0.138190 |      0.004665 |     0.029820 |
| mct_lrq71_balanced_family_barrier   | fixed340_holdout  |  27 |               8 |       0.296296 |     0.054988 |      0.009570 |     0.054988 |
| mct_lrq71_balanced_family_barrier   | random_supplement |  34 |              15 |       0.441176 |     0.078694 |      0.005354 |     0.027132 |
| mct_lrq71_balanced_family_barrier   | seed7_holdout     |  61 |              23 |       0.377049 |     0.078694 |      0.007220 |     0.054988 |
| mct_lrq71_balanced_family_barrier   | seed7_train       | 570 |             246 |       0.431579 |     0.138190 |      0.004391 |     0.029820 |
| mct_lrq71_balanced_family_collapse  | all900_holdout    |   4 |               1 |       0.250000 |     0.079576 |      0.019894 |     0.067640 |
| mct_lrq71_balanced_family_collapse  | all_rows          | 631 |               8 |       0.012678 |     0.079993 |      0.000837 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | random_supplement |  34 |               1 |       0.029412 |     0.079576 |      0.002340 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | seed7_holdout     |  61 |               1 |       0.016393 |     0.079576 |      0.001305 |     0.000000 |
| mct_lrq71_balanced_family_collapse  | seed7_train       | 570 |               7 |       0.012281 |     0.079993 |      0.000787 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | all900_holdout    |   4 |               1 |       0.250000 |     0.231168 |      0.057792 |     0.196493 |
| mct_lrq73_balanced_generic_collapse | all_rows          | 631 |              10 |       0.015848 |     0.231168 |      0.001821 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | random_supplement |  34 |               1 |       0.029412 |     0.231168 |      0.006799 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | seed7_holdout     |  61 |               1 |       0.016393 |     0.231168 |      0.003790 |     0.000000 |
| mct_lrq73_balanced_generic_collapse | seed7_train       | 570 |               9 |       0.015789 |     0.231168 |      0.001610 |     0.000000 |
| mct_lrq74_balanced_barrier5         | all900_holdout    |   4 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | all_rows          | 631 |               1 |       0.001585 |     0.077626 |      0.000123 |     0.000000 |
| mct_lrq74_balanced_barrier5         | fixed340_holdout  |  27 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | random_supplement |  34 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | seed7_holdout     |  61 |               0 |       0.000000 |     0.000000 |      0.000000 |     0.000000 |
| mct_lrq74_balanced_barrier5         | seed7_train       | 570 |               1 |       0.001754 |     0.077626 |      0.000136 |     0.000000 |

## Raw And Constrained Optima

| model                               | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   nearest_observed_phase_mean_tv |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |
|:------------------------------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | raw_random_search             |        0.830584 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | top8actual_hull_random_search |        0.842438 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_balanced_no_barrier       | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.830584 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | raw_random_search             |        0.748057 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | top8actual_hull_random_search |        0.771577 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_balanced_no_barrier       | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.748057 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | raw_random_search             |        0.830584 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | top8actual_hull_random_search |        0.842898 | False              | False                       | False                      |                         0.192892 |        0.000460 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_balanced_support_tv       | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.830584 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | raw_random_search             |        0.748057 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | top8actual_hull_random_search |        0.771999 | False              | False                       | False                      |                         0.184643 |        0.000300 |              0.731802 |             0.243967 |             0.024231 |              0.708885 |             0.271067 |             0.020049 |
| mct_lrq69_balanced_support_tv       | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.748057 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | raw_random_search             |        0.830598 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | top8actual_hull_random_search |        0.842756 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_drop_no_barrier           | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.830598 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | raw_random_search             |        0.746963 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | top8actual_hull_random_search |        0.771721 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq69_drop_no_barrier           | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.746963 | True               | True                        | True                       |                         0.000000 |        0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | raw_random_search             |        0.790861 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | top8actual_hull_random_search |        0.835789 | False              | False                       | False                      |                         0.212015 |        0.000000 |              0.734107 |             0.249798 |             0.016096 |              0.703700 |             0.285715 |             0.010585 |
| mct_lrq71_balanced_family_barrier   | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.790861 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | raw_random_search             |        0.699050 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | top8actual_hull_random_search |        0.769949 | False              | False                       | False                      |                         0.184137 |        0.000000 |              0.689122 |             0.289436 |             0.021442 |              0.661725 |             0.322696 |             0.015579 |
| mct_lrq71_balanced_family_barrier   | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.699050 | True               | True                        | True                       |                         0.000000 |        0.078749 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq71_balanced_family_collapse  | 340M/10.4B     | raw_random_search             |        0.809185 | True               | False                       | False                      |                         0.757564 |        0.000000 |              0.673241 |             0.131968 |             0.194791 |              0.021830 |             0.148002 |             0.830168 |
| mct_lrq71_balanced_family_collapse  | 340M/10.4B     | top8actual_hull_random_search |        0.842483 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq71_balanced_family_collapse  | 900M/24B       | raw_random_search             |        0.728300 | True               | False                       | False                      |                         0.757564 |        0.000000 |              0.673241 |             0.131968 |             0.194791 |              0.021830 |             0.148002 |             0.830168 |
| mct_lrq71_balanced_family_collapse  | 900M/24B       | top8actual_hull_random_search |        0.771402 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | raw_random_search             |        0.670178 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | top8actual_hull_random_search |        0.842236 | False              | False                       | False                      |                         0.204681 |        0.000000 |              0.709141 |             0.264099 |             0.026760 |              0.690858 |             0.285854 |             0.023287 |
| mct_lrq73_balanced_generic_collapse | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.670178 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | raw_random_search             |        0.577682 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | top8actual_hull_random_search |        0.770596 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq73_balanced_generic_collapse | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.577682 | True               | True                        | True                       |                         0.000000 |        0.079993 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | raw_random_search             |        0.846164 | False              | False                       | False                      |                         0.000000 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | top8actual_hull_random_search |        0.842457 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq74_balanced_barrier5         | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.841807 | False              | False                       | False                      |                         0.149839 |        0.000000 |              0.715506 |             0.255216 |             0.029278 |              0.712436 |             0.267363 |             0.020201 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | raw_random_search             |        0.774695 | False              | False                       | False                      |                         0.000000 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | top8actual_hull_random_search |        0.771584 | False              | False                       | False                      |                         0.192892 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |
| mct_lrq74_balanced_barrier5         | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.771188 | False              | False                       | False                      |                         0.149742 |        0.000000 |              0.728699 |             0.241596 |             0.029705 |              0.712522 |             0.267270 |             0.020208 |

## Monotonicity Grid

| model                               |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:------------------------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_balanced_barrier5         |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037645 |                          0.000565 | True                                  |       0.437727 |       0.000867 |        0.001375 |        0.001375 |           0.001375 |
| mct_lrq71_balanced_family_barrier   |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037873 |                          0.009735 | True                                  |       0.440259 |       0.052283 |        0.009470 |        0.009470 |           0.009470 |
| mct_lrq71_balanced_family_collapse  |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037670 |                          0.000448 | True                                  |       0.438027 |       0.000865 |        0.001009 |        0.001009 |           0.001009 |
| mct_lrq73_balanced_generic_collapse |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037785 |                          0.001467 | True                                  |       0.439368 |       0.010182 |        0.000840 |        0.000840 |           0.000840 |
| mct_lrq69_balanced_no_barrier       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037690 |                          0.000544 | True                                  |       0.438255 |       0.000852 |        0.001316 |        0.001316 |           0.001316 |
| mct_lrq69_drop_no_barrier           |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037587 |                          0.000914 | True                                  |       0.419899 |       0.001171 |        0.002482 |        0.002482 |           0.002482 |
| mct_lrq69_balanced_support_tv       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037690 |                          0.000544 | True                                  |       0.438255 |       0.000852 |        0.001316 |        0.001316 |           0.001316 |

## Artifact Map

- `csv/metric_summary.csv`: split metrics for every variant.
- `csv/fixed340_drop_summary.csv`: same-mixture multiplier drop ratios.
- `csv/offset_activity.csv`: observed-row offset activity for the barrier and support-TV wrapper.
- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.
- `plots/pred_actual_barrier_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.
- `plots/rmse_barrier_ablation.png`: compact RMSE comparison.
- `plots/optimum_family_shares_340m.png`: 340M optimum family shares.
- `code/`: the local ablation runner plus the Session-12 MCT helper code used for reproduction.
