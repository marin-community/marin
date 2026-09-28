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
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.053955 |   1.000000 |                -0.052973 |               0.493682 |     0.018344 |   0.010040 |    0.547335 |        0.036889 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.020000 |   0.931013 |                -0.019048 |               0.978797 |     0.017544 |   0.018219 |    1.038474 |        0.019661 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.014916 |   0.869213 |                 0.000247 |               1.192834 |     0.049681 |   0.060354 |    1.214821 |        0.028875 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.017351 |   0.971391 |                -0.008294 |               1.170055 |     0.068918 |   0.081225 |    1.178565 |        0.026150 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.042634 |   1.000000 |                -0.041436 |               0.527807 |     0.018344 |   0.010928 |    0.595703 |        0.025099 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.016077 |   0.931624 |                -0.014885 |               0.917242 |     0.017544 |   0.017139 |    0.976932 |        0.013596 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.013415 |   0.864018 |                -0.000684 |               1.146192 |     0.049681 |   0.058046 |    1.168376 |        0.026528 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.014653 |   0.970545 |                -0.006969 |               1.124473 |     0.068918 |   0.078091 |    1.133103 |        0.021600 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.042189 |   1.000000 |                -0.040958 |               0.518414 |     0.018344 |   0.010711 |    0.583901 |        0.024560 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.015834 |   0.927350 |                -0.014629 |               0.920378 |     0.017544 |   0.017190 |    0.979821 |        0.013536 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.013359 |   0.864018 |                -0.000681 |               1.145648 |     0.049681 |   0.058010 |    1.167652 |        0.026324 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.014507 |   0.970175 |                -0.006855 |               1.122900 |     0.068918 |   0.077979 |    1.131466 |        0.021402 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.052305 |   0.800000 |                -0.050925 |               0.465012 |     0.018344 |   0.010906 |    0.594515 |        0.031545 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.019035 |   0.923077 |                -0.017712 |               1.022709 |     0.017544 |   0.019246 |    1.096995 |        0.020765 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.014766 |   0.883270 |                -0.000446 |               1.184123 |     0.049681 |   0.059958 |    1.206855 |        0.028515 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.016790 |   0.973136 |                -0.008088 |               1.156725 |     0.068918 |   0.080343 |    1.165774 |        0.024968 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.024305 |          0.000670 |          1.055864 |            0.961259 |    0.004176 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.046744 |         -0.001773 |          0.962476 |            0.975336 |    0.002316 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.021352 |          0.000145 |          1.006148 |            1.019121 |    0.000885 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.022714 |         -0.000922 |          0.987130 |            0.893985 |    0.004071 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.042491 |         -0.006026 |          0.875107 |            0.883466 |    0.006096 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.019114 |         -0.002092 |          0.900841 |            0.907955 |    0.002193 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.022674 |         -0.000962 |          0.985483 |            0.894473 |    0.004065 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.042778 |         -0.005739 |          0.881059 |            0.888793 |    0.005805 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.019514 |         -0.001692 |          0.919731 |            0.926144 |    0.001817 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.024425 |          0.000789 |          1.061214 |            0.966362 |    0.004235 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.046700 |         -0.001817 |          0.961591 |            0.974351 |    0.002333 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.021332 |          0.000125 |          1.005223 |            1.017588 |    0.000873 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.290683 |        0.005821 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.253844 |        0.007482 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.121626 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.123145 | False              | False                       | False                      |                                0 |        0.000000 |              0.679866 |             0.292319 |             0.027816 |              0.676205 |             0.296599 |             0.027196 |                         0.244743 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        0.991715 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        0.988223 | False              | False                       | False                      |                                0 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |                         0.220282 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        0.987459 | False              | False                       | False                      |                                0 |        0.000000 |              0.699369 |             0.269573 |             0.031058 |              0.703606 |             0.274587 |             0.021808 |                         0.149913 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.854508 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.851250 | False              | False                       | False                      |                                0 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |                         0.221662 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.850165 | False              | False                       | False                      |                                0 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |                         0.149726 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.750199 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.747310 | False              | False                       | False                      |                                0 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |                         0.206398 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.746799 | False              | False                       | False                      |                                0 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |                         0.149941 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.064170 |                          0.002001 | True                                  |       0.659124 |       0.001024 |        0.005903 |        0.005903 |           0.005903 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.063949 |                          0.001692 | True                                  |       0.685856 |       0.000753 |        0.004909 |        0.004909 |           0.004909 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.070797 |                          0.000102 | True                                  |       0.542338 |       0.000791 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.069804 |                          0.000101 | True                                  |       0.534732 |       0.000789 |        0.000000 |        0.000000 |           0.000000 |

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
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.014507 |        0.015834 |      0.042189 |              0.881059 | Best drop-preserving member of the exponent-search frontier.                                    |         0.920378 |             0.979821 |              0.985483 |            0.919731 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.014653 |        0.016077 |      0.042634 |              0.875107 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.917242 |             0.976932 |              0.987130 |            0.900841 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.017351 |        0.020000 |      0.053955 |              0.962476 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.978797 |             1.038474 |              1.055864 |            1.006148 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.961591 | Clean structural reference; raw optima collapse in delayed report.                              |         1.022709 |             1.096995 |              1.061214 |            1.005223 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.014653 |        0.016077 |      0.042634 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
