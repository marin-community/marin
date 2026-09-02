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
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.030066 |   1.000000 |                -0.028282 |               0.499522 |     0.018344 |   0.010186 |    0.555254 |        0.012123 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.012473 |   0.931013 |                -0.010895 |               0.935318 |     0.017544 |   0.017460 |    0.995195 |        0.010825 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.013716 |   0.872269 |                 0.001661 |               1.157361 |     0.049681 |   0.058569 |    1.178909 |        0.017622 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.013180 |   0.971920 |                -0.003896 |               1.118441 |     0.068918 |   0.077675 |    1.127058 |        0.013593 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.023685 |   1.000000 |                -0.021406 |               0.527146 |     0.018344 |   0.011000 |    0.599656 |        0.004819 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.011038 |   0.931013 |                -0.009218 |               0.926725 |     0.017544 |   0.017308 |    0.986526 |        0.008973 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.012307 |   0.869213 |                 0.000664 |               1.124941 |     0.049681 |   0.056886 |    1.145018 |        0.017782 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.011762 |   0.971391 |                -0.003710 |               1.092885 |     0.068918 |   0.075873 |    1.100908 |        0.012755 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.023184 |   1.000000 |                -0.020791 |               0.510921 |     0.018344 |   0.010612 |    0.578496 |        0.004120 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.010738 |   0.931013 |                -0.008869 |               0.933607 |     0.017544 |   0.017423 |    0.993113 |        0.008971 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.012280 |   0.868602 |                 0.000666 |               1.124367 |     0.049681 |   0.056855 |    1.144400 |        0.017635 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.011623 |   0.971285 |                -0.003554 |               1.090853 |     0.068918 |   0.075731 |    1.098857 |        0.012589 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.028542 |   0.800000 |                -0.025915 |               0.470954 |     0.018344 |   0.011113 |    0.605830 |        0.006460 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.011814 |   0.922466 |                -0.009571 |               0.980297 |     0.017544 |   0.018538 |    1.056635 |        0.011644 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.013589 |   0.885409 |                 0.000936 |               1.148118 |     0.049681 |   0.058165 |    1.170769 |        0.018104 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.012834 |   0.973453 |                -0.003714 |               1.104856 |     0.068918 |   0.076790 |    1.114220 |        0.012513 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.022957 |         -0.000679 |          0.997309 |            0.907968 |    0.004113 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044130 |         -0.004387 |          0.908658 |            0.920792 |    0.004583 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020158 |         -0.001049 |          0.949888 |            0.962089 |    0.001321 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.022831 |         -0.000804 |          0.992152 |            0.899182 |    0.004095 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.043321 |         -0.005196 |          0.892119 |            0.902037 |    0.005308 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.019743 |         -0.001464 |          0.930410 |            0.939481 |    0.001633 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.022890 |         -0.000745 |          0.994803 |            0.900415 |    0.004068 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.043849 |         -0.004668 |          0.903035 |            0.912341 |    0.004784 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.020280 |         -0.000927 |          0.955750 |            0.964145 |    0.001182 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023109 |         -0.000526 |          1.004088 |            0.914371 |    0.004131 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.044150 |         -0.004366 |          0.909098 |            0.921150 |    0.004556 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020167 |         -0.001039 |          0.950349 |            0.961962 |    0.001309 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.256284 |        0.003530 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.217072 |        0.005175 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.060104 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.061942 | False              | False                       | False                      |                                0 |        0.000000 |              0.679866 |             0.292319 |             0.027816 |              0.676205 |             0.296599 |             0.027196 |                         0.244743 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        0.951782 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        0.948290 | False              | False                       | False                      |                                0 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |                         0.220282 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        0.947526 | False              | False                       | False                      |                                0 |        0.000000 |              0.699369 |             0.269573 |             0.031058 |              0.703606 |             0.274587 |             0.021808 |                         0.149913 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.843769 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.840514 | False              | False                       | False                      |                                0 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |                         0.221662 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.839428 | False              | False                       | False                      |                                0 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |                         0.149726 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.758876 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.756060 | False              | False                       | False                      |                                0 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |                         0.206398 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.755527 | False              | False                       | False                      |                                0 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |                         0.149941 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.048678 |                          0.001391 | True                                  |       0.525974 |       0.001101 |        0.003991 |        0.003991 |           0.003991 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.048681 |                          0.001046 | True                                  |       0.548193 |       0.000805 |        0.002894 |        0.002894 |           0.002894 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.052259 |                          0.000096 | True                                  |       0.427393 |       0.000747 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.051457 |                          0.000096 | True                                  |       0.420834 |       0.000746 |        0.000000 |        0.000000 |           0.000000 |

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
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.011623 |        0.010738 |      0.023184 |              0.903035 | Best drop-preserving member of the exponent-search frontier.                                    |         0.933607 |             0.993113 |              0.994803 |            0.955750 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.011762 |        0.011038 |      0.023685 |              0.892119 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.926725 |             0.986526 |              0.992152 |            0.930410 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.013180 |        0.012473 |      0.030066 |              0.908658 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.935318 |             0.995195 |              0.997309 |            0.949888 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.909098 | Clean structural reference; raw optima collapse in delayed report.                              |         0.980297 |             1.056635 |              1.004088 |            0.950349 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.011762 |        0.011038 |      0.023685 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
