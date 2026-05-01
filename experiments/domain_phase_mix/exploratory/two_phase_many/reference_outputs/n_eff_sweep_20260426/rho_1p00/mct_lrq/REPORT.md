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
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.059944 |   1.000000 |                -0.059059 |               0.491959 |     0.018344 |   0.009998 |    0.545055 |        0.043002 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.020939 |   0.934676 |                -0.020015 |               0.998749 |     0.017544 |   0.018571 |    1.058534 |        0.021184 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.015427 |   0.868602 |                -0.000191 |               1.199465 |     0.049681 |   0.060752 |    1.222838 |        0.032056 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.018075 |   0.971602 |                -0.008966 |               1.175698 |     0.068918 |   0.081640 |    1.184592 |        0.028966 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.048421 |   1.000000 |                -0.047376 |               0.527751 |     0.018344 |   0.010899 |    0.594144 |        0.031119 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.016416 |   0.931624 |                -0.015253 |               0.917585 |     0.017544 |   0.017144 |    0.977174 |        0.013897 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.013886 |   0.864018 |                -0.001075 |               1.149677 |     0.049681 |   0.058299 |    1.173460 |        0.028928 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.015059 |   0.970545 |                -0.007350 |               1.125353 |     0.068918 |   0.078187 |    1.134492 |        0.023446 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.048057 |   1.000000 |                -0.046993 |               0.522913 |     0.018344 |   0.010793 |    0.588383 |        0.030689 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.016207 |   0.927350 |                -0.015033 |               0.919765 |     0.017544 |   0.017178 |    0.979128 |        0.013839 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.013825 |   0.864018 |                -0.001073 |               1.149187 |     0.049681 |   0.058264 |    1.172751 |        0.028726 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.014926 |   0.970175 |                -0.007252 |               1.123977 |     0.068918 |   0.078087 |    1.133043 |        0.023256 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.058281 |   0.800000 |                -0.057047 |               0.463286 |     0.018344 |   0.010844 |    0.591175 |        0.037694 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.019997 |   0.927350 |                -0.018720 |               1.042204 |     0.017544 |   0.019575 |    1.115773 |        0.022330 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.015282 |   0.883575 |                -0.000872 |               1.191176 |     0.049681 |   0.060371 |    1.215179 |        0.031584 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.017526 |   0.973559 |                -0.008772 |               1.162790 |     0.068918 |   0.080786 |    1.172197 |        0.027797 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.024923 |          0.001287 |          1.082664 |            0.985645 |    0.004347 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.047945 |         -0.000571 |          0.987221 |            1.000416 |    0.001668 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.021901 |          0.000694 |          1.032015 |            1.045354 |    0.001141 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.022785 |         -0.000851 |          0.990267 |            0.897584 |    0.004042 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.042421 |         -0.006096 |          0.873703 |            0.881570 |    0.006158 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.018995 |         -0.002212 |          0.895229 |            0.901720 |    0.002303 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.022720 |         -0.000916 |          0.987521 |            0.897081 |    0.004042 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.042647 |         -0.005870 |          0.878395 |            0.885638 |    0.005927 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.019360 |         -0.001846 |          0.912484 |            0.918275 |    0.001955 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.025028 |          0.001392 |          1.087403 |            0.990192 |    0.004415 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.047874 |         -0.000643 |          0.985767 |            0.998854 |    0.001667 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.021868 |          0.000662 |          1.030496 |            1.043218 |    0.001113 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.302839 |        0.006573 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.266770 |        0.008234 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.149574 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.151006 | False              | False                       | False                      |                                0 |        0.000000 |              0.679866 |             0.292319 |             0.027816 |              0.676205 |             0.296599 |             0.027196 |                         0.244743 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        1.011369 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        1.007877 | False              | False                       | False                      |                                0 |        0.000000 |              0.696666 |             0.275502 |             0.027832 |              0.693037 |             0.279051 |             0.027913 |                         0.220282 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        1.007113 | False              | False                       | False                      |                                0 |        0.000000 |              0.699369 |             0.269573 |             0.031058 |              0.703606 |             0.274587 |             0.021808 |                         0.149913 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.863076 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.859816 | False              | False                       | False                      |                                0 |        0.000000 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |                         0.221662 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.858732 | False              | False                       | False                      |                                0 |        0.000000 |              0.711161 |             0.260289 |             0.028550 |              0.706912 |             0.274706 |             0.018381 |                         0.149726 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.751303 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.748394 | False              | False                       | False                      |                                0 |        0.000000 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |                         0.206398 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.747889 | False              | False                       | False                      |                                0 |        0.000000 |              0.749824 |             0.223150 |             0.027026 |              0.706985 |             0.275351 |             0.017664 |                         0.149941 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.070006 |                          0.002222 | True                                  |       0.704673 |       0.001004 |        0.006586 |        0.006586 |           0.006586 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.069682 |                          0.001924 | True                                  |       0.732941 |       0.000739 |        0.005629 |        0.005629 |           0.005629 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.077811 |                          0.000104 | True                                  |       0.580698 |       0.000812 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.076768 |                          0.000104 | True                                  |       0.572919 |       0.000809 |        0.000000 |        0.000000 |           0.000000 |

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
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.014926 |        0.016207 |      0.048057 |              0.878395 | Best drop-preserving member of the exponent-search frontier.                                    |         0.919765 |             0.979128 |              0.987521 |            0.912484 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.015059 |        0.016416 |      0.048421 |              0.873703 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.917585 |             0.977174 |              0.990267 |            0.895229 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.018075 |        0.020939 |      0.059944 |              0.987221 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.998749 |             1.058534 |              1.082664 |            1.032015 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.985767 | Clean structural reference; raw optima collapse in delayed report.                              |         1.042204 |             1.115773 |              1.087403 |            1.030496 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.015059 |        0.016416 |      0.048421 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
