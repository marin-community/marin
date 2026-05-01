# MCT-LRQ74: monotone curvature-tuned compact structural law

## Selected candidate

`mct_lrq74_drop` is a 74-constant structural law. It keeps the CBS-LRQ compatibility barrier and LRQ anchor, but retunes the Chinchilla-style scale exponents so token returns are represented as a low-curvature `D` mode plus a very small high-curvature `N,D` interaction mode.

```text
L(w,N,D) = P(w) + E_LRQ(w)
         + A(w)((N/N0)^(-0.154791)-1)
         + B(w)((D/D0)^(-0.146425)-1)
         + C(w)((N/N0)^(-0.014295)(D/D0)^(-1.063376)-1)

P(w) = 5 * relu(p0_reasoning - 0.12)^2
         * [ relu(p1_tech - 0.55)^2
             + relu(0.45 - p1_broad)^2
             + relu(max_domain - 0.45)^2 ].
```

`A`, `B`, and `C` are nonnegative. `A` and `C` are scalar heads; `B` is a nonnegative linear family-share head. Since all exponents are positive, fixed `w` is monotone decreasing in both corrected non-embedding `N` and realized train tokens `D`. At fixed `N,D`, the law is an LRQ mixture regression plus a mixture-only compatibility penalty.

The selected point is the drop-preserving member of the local exponent-search frontier. A slightly more RMSE-balanced sibling, `mct_lrq74_balanced`, is also included in the archive.

## Metrics

| model              | fit_protocol   | split                  |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   actual_std |   pred_std |   std_ratio |   low_tail_rmse |
|:-------------------|:---------------|:-----------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|-------------:|-----------:|------------:|----------------:|
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.011159 |   1.000000 |                -0.004573 |               0.502524 |     0.018344 |   0.010261 |    0.559383 |        0.011617 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.006113 |   0.931013 |                 0.000572 |               0.922896 |     0.017544 |   0.017245 |    0.982925 |        0.005694 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.013814 |   0.872269 |                 0.002902 |               1.115572 |     0.049681 |   0.056755 |    1.142385 |        0.010852 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.011086 |   0.971920 |                 0.001871 |               1.045131 |     0.068918 |   0.072786 |    1.056126 |        0.005381 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.010358 |   1.000000 |                -0.001646 |               0.525503 |     0.018344 |   0.011034 |    0.601499 |        0.015142 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.006068 |   0.931013 |                 0.000153 |               0.946642 |     0.017544 |   0.017656 |    1.006390 |        0.005444 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.012299 |   0.869213 |                 0.001907 |               1.095154 |     0.049681 |   0.055548 |    1.118086 |        0.011476 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.010030 |   0.971391 |                 0.001131 |               1.036464 |     0.068918 |   0.072079 |    1.045867 |        0.005480 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.010397 |   1.000000 |                -0.001060 |               0.509320 |     0.018344 |   0.010642 |    0.580123 |        0.015816 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.006082 |   0.931013 |                 0.000539 |               0.957400 |     0.017544 |   0.017840 |    1.016875 |        0.005550 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.012299 |   0.869213 |                 0.001904 |               1.094792 |     0.049681 |   0.055532 |    1.117765 |        0.011493 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.010034 |   0.971391 |                 0.001300 |               1.034361 |     0.068918 |   0.071938 |    1.043821 |        0.005528 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.012173 |   0.800000 |                -0.002201 |               0.473935 |     0.018344 |   0.011215 |    0.611393 |        0.017286 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.007149 |   0.919414 |                 0.001749 |               0.968219 |     0.017544 |   0.018338 |    1.045251 |        0.003774 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.013732 |   0.885409 |                 0.002153 |               1.106653 |     0.049681 |   0.056379 |    1.134823 |        0.012908 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.011301 |   0.973189 |                 0.001974 |               1.032385 |     0.068918 |   0.071980 |    1.044433 |        0.005515 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.022566 |         -0.001070 |          0.980312 |            0.892482 |    0.004175 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.043392 |         -0.005125 |          0.893465 |            0.905401 |    0.005282 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.019821 |         -0.001386 |          0.934006 |            0.946034 |    0.001593 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.023328 |         -0.000308 |          1.013638 |            0.920644 |    0.004076 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.044695 |         -0.003821 |          0.920379 |            0.931530 |    0.004019 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.020535 |         -0.000671 |          0.967724 |            0.978337 |    0.001037 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.023489 |         -0.000147 |          1.020710 |            0.925344 |    0.004052 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.045475 |         -0.003042 |          0.936467 |            0.947043 |    0.003279 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.021215 |          0.000009 |          0.999806 |            1.009802 |    0.000808 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.022729 |         -0.000907 |          0.987557 |            0.899309 |    0.004177 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.043433 |         -0.005084 |          0.894331 |            0.906190 |    0.005237 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.019840 |         -0.001367 |          0.934912 |            0.946358 |    0.001573 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.234628 |        0.001961 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.193586 |        0.003568 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.028194 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.030335 | False              | False                       | False                      |                                0 |        0.000000 |              0.648624 |             0.320498 |             0.030878 |              0.653686 |             0.315379 |             0.030935 |                         0.227717 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        0.933969 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        0.930477 | False              | False                       | False                      |                                0 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |                         0.220282 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        0.929712 | False              | False                       | False                      |                                0 |        0.000000 |              0.699369 |             0.269573 |             0.031058 |              0.703606 |             0.274587 |             0.021808 |                         0.149913 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.846729 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.843477 | False              | False                       | False                      |                                0 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |                         0.221662 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.842390 | False              | False                       | False                      |                                0 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |                         0.149726 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.775440 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.772691 | False              | False                       | False                      |                                0 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |                         0.206398 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.772137 | False              | False                       | False                      |                                0 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |                         0.149941 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037485 |                          0.001006 | True                                  |       0.418747 |       0.001175 |        0.002769 |        0.002769 |           0.002769 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037600 |                          0.000639 | True                                  |       0.437199 |       0.000855 |        0.001610 |        0.001610 |           0.001610 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.039243 |                          0.000094 | True                                  |       0.335045 |       0.000734 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.038623 |                          0.000094 | True                                  |       0.329759 |       0.000734 |        0.000000 |        0.000000 |           0.000000 |

The monotonicity guarantee is analytic for the selected law because all scale-head features and coefficients are nonnegative and all exponents are positive. The grid check is included as a regression test.

## Parameter counts

| model              |   fitted_param_count_counting_exponents |   total_constant_count |   anchor_feature_count_including_intercept |   scale_param_count |   donor_constant_count |   barrier_constant_count | exponents                                                                   |   pair_weight |
|:-------------------|----------------------------------------:|-----------------------:|-------------------------------------------:|--------------------:|-----------------------:|-------------------------:|:----------------------------------------------------------------------------|--------------:|
| mct_lrq74_drop     |                                      60 |                     74 |                                         47 |                   9 |                      9 |                        5 | {"alpha": 0.154791, "beta": 0.146425, "gamma": 0.014295, "delta": 1.063376} |      4.000000 |
| mct_lrq74_balanced |                                      60 |                     74 |                                         47 |                   9 |                      9 |                        5 | {"alpha": 0.148968, "beta": 0.209383, "gamma": 0.009859, "delta": 1.043436} |      4.000000 |
| cbs_lrq74_s5       |                                      60 |                     74 |                                         47 |                   9 |                      9 |                        5 | {"alpha": 0.2, "beta": 0.25, "gamma": 0.3, "delta": 0.65}                   |      6.000000 |
| s2_rebuild61       |                                      52 |                     61 |                                         39 |                   9 |                      9 |                        0 | {}                                                                          |      6.000000 |

## Reference comparison

| model               | role                                |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse |   drop_0.5_to_2_ratio | note                                                                                            |   fixed340_slope |   fixed340_std_ratio |   drop_0.5_to_1_ratio |   drop_1_to_2_ratio |
|:--------------------|:------------------------------------|---------:|---------------------:|----------------:|--------------:|----------------------:|:------------------------------------------------------------------------------------------------|-----------------:|---------------------:|----------------------:|--------------------:|
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.010034 |        0.006082 |      0.010397 |              0.936467 | Best drop-preserving member of the exponent-search frontier.                                    |         0.957400 |             1.016875 |              1.020710 |            0.999806 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.010030 |        0.006068 |      0.010358 |              0.920379 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.946642 |             1.006390 |              1.013638 |            0.967724 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.011086 |        0.006113 |      0.011159 |              0.893465 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.922896 |             0.982925 |              0.980312 |            0.934006 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.894331 | Clean structural reference; raw optima collapse in delayed report.                              |         0.968219 |             1.045251 |              0.987557 |            0.934912 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.010030 |        0.006068 |      0.010358 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
