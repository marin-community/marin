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
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.044513 |   1.000000 |                -0.043322 |               0.496233 |     0.018344 |   0.010103 |    0.550759 |        0.027203 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.017542 |   0.931013 |                -0.016459 |               0.956958 |     0.017544 |   0.017836 |    1.016659 |        0.016551 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.014298 |   0.869213 |                 0.000835 |               1.180225 |     0.049681 |   0.059679 |    1.201247 |        0.024248 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.015816 |   0.971391 |                -0.006820 |               1.153874 |     0.068918 |   0.080094 |    1.162164 |        0.021438 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.034811 |   1.000000 |                -0.033321 |               0.527681 |     0.018344 |   0.010961 |    0.597547 |        0.016878 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.014598 |   0.931624 |                -0.013273 |               0.919370 |     0.017544 |   0.017178 |    0.979140 |        0.012235 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.012853 |   0.868602 |                -0.000136 |               1.139076 |     0.049681 |   0.057619 |    1.159772 |        0.023001 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.013653 |   0.971338 |                -0.005951 |               1.116142 |     0.068918 |   0.077485 |    1.124310 |        0.018390 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.034273 |   1.000000 |                -0.032721 |               0.512485 |     0.018344 |   0.010603 |    0.578007 |        0.016197 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.014319 |   0.927350 |                -0.012974 |               0.923958 |     0.017544 |   0.017254 |    0.983457 |        0.012186 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.012807 |   0.868602 |                -0.000133 |               1.138490 |     0.049681 |   0.057584 |    1.159063 |        0.022808 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.013497 |   0.970968 |                -0.005816 |               1.114340 |     0.068918 |   0.077358 |    1.122457 |        0.018192 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.042685 |   0.800000 |                -0.040979 |               0.467681 |     0.018344 |   0.010999 |    0.599602 |        0.021564 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.016621 |   0.920635 |                -0.015104 |               1.001392 |     0.017544 |   0.018888 |    1.076623 |        0.017587 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.014151 |   0.885409 |                 0.000129 |               1.171156 |     0.049681 |   0.059273 |    1.193079 |        0.024132 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.015293 |   0.973295 |                -0.006614 |               1.140254 |     0.068918 |   0.079197 |    1.149148 |        0.020272 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023629 |         -0.000007 |          1.026492 |            0.934530 |    0.004089 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.045429 |         -0.003087 |          0.935417 |            0.947911 |    0.003393 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020751 |         -0.000455 |          0.977861 |            0.990439 |    0.000954 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.022702 |         -0.000933 |          0.986596 |            0.892473 |    0.004092 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.042733 |         -0.005784 |          0.880058 |            0.889077 |    0.005867 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.019335 |         -0.001871 |          0.911234 |            0.919177 |    0.001992 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.022701 |         -0.000935 |          0.986617 |            0.894462 |    0.004077 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.043113 |         -0.005404 |          0.887932 |            0.896327 |    0.005485 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.019788 |         -0.001418 |          0.932619 |            0.939866 |    0.001577 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.023764 |          0.000129 |          1.032541 |            0.940269 |    0.004131 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.045417 |         -0.003100 |          0.935181 |            0.947583 |    0.003394 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.020746 |         -0.000460 |          0.977615 |            0.989595 |    0.000949 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.275359 |        0.004831 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.237501 |        0.006488 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.091725 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.093370 | False              | False                       | False                      |                                0 |        0.000000 |              0.679866 |             0.292319 |             0.027816 |              0.676205 |             0.296599 |             0.027196 |                         0.244743 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        0.971626 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        0.968134 | False              | False                       | False                      |                                0 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |                         0.220282 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        0.967370 | False              | False                       | False                      |                                0 |        0.000000 |              0.699369 |             0.269573 |             0.031058 |              0.703606 |             0.274587 |             0.021808 |                         0.149913 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.847546 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.844290 | False              | False                       | False                      |                                0 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |                         0.221662 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.843205 | False              | False                       | False                      |                                0 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |                         0.149726 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.752012 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.749152 | False              | False                       | False                      |                                0 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |                         0.206398 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.748632 | False              | False                       | False                      |                                0 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |                         0.149941 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.057230 |                          0.001728 | True                                  |       0.601676 |       0.001054 |        0.005051 |        0.005051 |           0.005051 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.057119 |                          0.001403 | True                                  |       0.626472 |       0.000774 |        0.004011 |        0.004011 |           0.004011 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.062456 |                          0.000099 | True                                  |       0.493033 |       0.000769 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.061538 |                          0.000099 | True                                  |       0.485790 |       0.000767 |        0.000000 |        0.000000 |           0.000000 |

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
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.013497 |        0.014319 |      0.034273 |              0.887932 | Best drop-preserving member of the exponent-search frontier.                                    |         0.923958 |             0.983457 |              0.986617 |            0.932619 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.013653 |        0.014598 |      0.034811 |              0.880058 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.919370 |             0.979140 |              0.986596 |            0.911234 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.015816 |        0.017542 |      0.044513 |              0.935417 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.956958 |             1.016659 |              1.026492 |            0.977861 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.935181 | Clean structural reference; raw optima collapse in delayed report.                              |         1.001392 |             1.076623 |              1.032541 |            0.977615 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.013653 |        0.014598 |      0.034811 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
