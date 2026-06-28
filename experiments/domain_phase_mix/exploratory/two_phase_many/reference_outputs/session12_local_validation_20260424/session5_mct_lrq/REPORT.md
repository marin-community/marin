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
| cbs_lrq74_s5       | leave900out    | all900_leave_scale_out |   4 | 0.011103 |   1.000000 |                -0.004357 |               0.509566 |     0.018344 |   0.010522 |    0.573609 |        0.012163 |
| cbs_lrq74_s5       | seed7          | fixed340_holdout       |  27 | 0.006122 |   0.931013 |                 0.000647 |               0.923268 |     0.017544 |   0.017252 |    0.983338 |        0.005725 |
| cbs_lrq74_s5       | seed7          | random_supplement      |  34 | 0.013785 |   0.872269 |                 0.002891 |               1.114766 |     0.049681 |   0.056713 |    1.141550 |        0.010851 |
| cbs_lrq74_s5       | seed7          | seed7_holdout          |  61 | 0.011068 |   0.971920 |                 0.001898 |               1.044398 |     0.068918 |   0.072735 |    1.055382 |        0.005409 |
| mct_lrq74_balanced | leave900out    | all900_leave_scale_out |   4 | 0.010434 |   1.000000 |                -0.001962 |               0.528150 |     0.018344 |   0.011134 |    0.606958 |        0.014896 |
| mct_lrq74_balanced | seed7          | fixed340_holdout       |  27 | 0.006073 |   0.931013 |                -0.000029 |               0.953371 |     0.017544 |   0.017775 |    1.013185 |        0.005432 |
| mct_lrq74_balanced | seed7          | random_supplement      |  34 | 0.012361 |   0.869213 |                 0.001927 |               1.097039 |     0.049681 |   0.055644 |    1.120033 |        0.011587 |
| mct_lrq74_balanced | seed7          | seed7_holdout          |  61 | 0.010074 |   0.971391 |                 0.001061 |               1.038405 |     0.068918 |   0.072214 |    1.047825 |        0.005472 |
| mct_lrq74_drop     | leave900out    | all900_leave_scale_out |   4 | 0.010495 |   1.000000 |                -0.001671 |               0.513265 |     0.018344 |   0.010783 |    0.587847 |        0.015307 |
| mct_lrq74_drop     | seed7          | fixed340_holdout       |  27 | 0.006080 |   0.931013 |                 0.000318 |               0.965338 |     0.017544 |   0.017981 |    1.024914 |        0.005546 |
| mct_lrq74_drop     | seed7          | random_supplement      |  34 | 0.012373 |   0.869213 |                 0.001928 |               1.097087 |     0.049681 |   0.055649 |    1.120131 |        0.011626 |
| mct_lrq74_drop     | seed7          | seed7_holdout          |  61 | 0.010084 |   0.971391 |                 0.001215 |               1.036703 |     0.068918 |   0.072101 |    1.046186 |        0.005522 |
| s2_rebuild61       | leave900out    | all900_leave_scale_out |   4 | 0.012217 |   0.800000 |                -0.001832 |               0.484227 |     0.018344 |   0.011631 |    0.634067 |        0.017855 |
| s2_rebuild61       | seed7          | fixed340_holdout       |  27 | 0.007176 |   0.919414 |                 0.001851 |               0.968659 |     0.017544 |   0.018346 |    1.045710 |        0.003792 |
| s2_rebuild61       | seed7          | random_supplement      |  34 | 0.013694 |   0.885409 |                 0.002137 |               1.105537 |     0.049681 |   0.056322 |    1.133667 |        0.012911 |
| s2_rebuild61       | seed7          | seed7_holdout          |  61 | 0.011284 |   0.973189 |                 0.002010 |               1.031373 |     0.068918 |   0.071910 |    1.043408 |        0.005571 |

## Same-mixture fixed-340M target-budget drops

| model              | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:-------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| cbs_lrq74_s5       | 0.5x_to_1.0x |  12 |           0.023636 |         0.022588 |         -0.001048 |          0.981291 |            0.893404 |    0.004175 |
| cbs_lrq74_s5       | 0.5x_to_2.0x |   3 |           0.048517 |         0.043398 |         -0.005119 |          0.893596 |            0.905522 |    0.005275 |
| cbs_lrq74_s5       | 1.0x_to_2.0x |   3 |           0.021206 |         0.019824 |         -0.001383 |          0.934142 |            0.946093 |    0.001590 |
| mct_lrq74_balanced | 0.5x_to_1.0x |  12 |           0.023636 |         0.023530 |         -0.000106 |          1.022437 |            0.929048 |    0.004088 |
| mct_lrq74_balanced | 0.5x_to_2.0x |   3 |           0.048517 |         0.045108 |         -0.003409 |          0.928873 |            0.940274 |    0.003641 |
| mct_lrq74_balanced | 1.0x_to_2.0x |   3 |           0.021206 |         0.020755 |         -0.000451 |          0.978074 |            0.988892 |    0.000923 |
| mct_lrq74_drop     | 0.5x_to_1.0x |  12 |           0.023636 |         0.023726 |          0.000090 |          1.031035 |            0.935152 |    0.004075 |
| mct_lrq74_drop     | 0.5x_to_2.0x |   3 |           0.048517 |         0.045961 |         -0.002555 |          0.946486 |            0.957332 |    0.002850 |
| mct_lrq74_drop     | 1.0x_to_2.0x |   3 |           0.021206 |         0.021478 |          0.000272 |          1.012177 |            1.022375 |    0.000868 |
| s2_rebuild61       | 0.5x_to_1.0x |  12 |           0.023636 |         0.022753 |         -0.000883 |          0.988619 |            0.900306 |    0.004178 |
| s2_rebuild61       | 0.5x_to_2.0x |   3 |           0.048517 |         0.043444 |         -0.005073 |          0.894552 |            0.906402 |    0.005226 |
| s2_rebuild61       | 1.0x_to_2.0x |   3 |           0.021206 |         0.019845 |         -0.001362 |          0.935142 |            0.946514 |    0.001568 |

The improvement over CBS-LRQ is mostly in the long continuation pair: `0.5x -> 2.0x` moves from about `0.894` to `0.946`, and `1.0x -> 2.0x` moves from about `0.934` to `1.012`, without adding a separate continuation head.

## Fixed-340M triple beta diagnostics

| model              |   n |   actual_beta_mean |   actual_beta_std |   pred_beta_mean |   pred_beta_std |
|:-------------------|----:|-------------------:|------------------:|-----------------:|----------------:|
| cbs_lrq74_s5       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |
| mct_lrq74_balanced |   3 |           0.364509 |          0.066310 |         0.230743 |        0.001665 |
| mct_lrq74_drop     |   3 |           0.364509 |          0.066310 |         0.189103 |        0.003241 |
| s2_rebuild61       |   3 |           0.364509 |          0.066310 |         0.250000 |        0.000000 |

## Raw and constrained optima

| model          | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   scale_path_increase_violations |   barrier_value |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |   nearest_observed_phase_mean_tv |
|:---------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|---------------------------------:|
| mct_lrq74_drop | 60M/1.2B       | raw_random_search             |        1.027864 | False              | False                       | False                      |                                0 |        0.000000 |              0.637440 |             0.357900 |             0.004660 |              0.585603 |             0.296488 |             0.117909 |                         0.383056 |
| mct_lrq74_drop | 60M/1.2B       | top8actual_hull_random_search |        1.029544 | False              | False                       | False                      |                                0 |        0.000000 |              0.681761 |             0.293556 |             0.024683 |              0.672018 |             0.306530 |             0.021452 |                         0.233747 |
| mct_lrq74_drop | 100M/6B        | raw_random_search             |        0.933969 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 100M/6B        | top8actual_hull_random_search |        0.930148 | False              | False                       | False                      |                                0 |        0.000000 |              0.704934 |             0.269701 |             0.025365 |              0.700507 |             0.276237 |             0.023256 |                         0.215142 |
| mct_lrq74_drop | 100M/6B        | trustblend_hull_to_raw_cap015 |        0.929209 | False              | False                       | False                      |                                0 |        0.000000 |              0.704348 |             0.266497 |             0.029155 |              0.707263 |             0.273259 |             0.019478 |                         0.149848 |
| mct_lrq74_drop | 340M/10.4B     | raw_random_search             |        0.846469 | False              | False                       | False                      |                                0 |        0.000000 |              0.703323 |             0.260900 |             0.035777 |              0.719069 |             0.268056 |             0.012876 |                         0.000000 |
| mct_lrq74_drop | 340M/10.4B     | top8actual_hull_random_search |        0.842771 | False              | False                       | False                      |                                0 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |                         0.192892 |
| mct_lrq74_drop | 340M/10.4B     | trustblend_hull_to_raw_cap015 |        0.842118 | False              | False                       | False                      |                                0 |        0.000000 |              0.715506 |             0.255216 |             0.029278 |              0.712436 |             0.267363 |             0.020201 |                         0.149839 |
| mct_lrq74_drop | 900M/24B       | raw_random_search             |        0.774782 | False              | False                       | False                      |                                0 |        0.000000 |              0.751015 |             0.210157 |             0.038828 |              0.720849 |             0.267852 |             0.011299 |                         0.000000 |
| mct_lrq74_drop | 900M/24B       | top8actual_hull_random_search |        0.771712 | False              | False                       | False                      |                                0 |        0.000000 |              0.720777 |             0.252757 |             0.026467 |              0.709566 |             0.267064 |             0.023370 |                         0.192892 |
| mct_lrq74_drop | 900M/24B       | trustblend_hull_to_raw_cap015 |        0.771306 | False              | False                       | False                      |                                0 |        0.000000 |              0.728699 |             0.241596 |             0.029705 |              0.712522 |             0.267270 |             0.020208 |                         0.149742 |

The raw random-search optima remain non-pathological: no hard corners, no phase-1 tech-family collapse, no family-collapse flags, and no scale-path increase violations in the reported probes.

## Structural sanity

| model              |   mixtures_checked |   N_grid_steps |   D_grid_steps |   N_monotonicity_violations |   D_monotonicity_violations |   min_loss_drop_when_increasing_N |   min_loss_drop_when_increasing_D | analytic_positive_head_monotonicity   |   min_A_N_head |   min_B_D_head |   min_C_ND_head |   max_C_ND_head |   median_C_ND_head |
|:-------------------|-------------------:|---------------:|---------------:|----------------------------:|----------------------------:|----------------------------------:|----------------------------------:|:--------------------------------------|---------------:|---------------:|----------------:|----------------:|-------------------:|
| mct_lrq74_drop     |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037541 |                          0.000933 | True                                  |       0.419383 |       0.001194 |        0.002533 |        0.002533 |           0.002533 |
| mct_lrq74_balanced |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.037645 |                          0.000565 | True                                  |       0.437727 |       0.000867 |        0.001375 |        0.001375 |           0.001375 |
| cbs_lrq74_s5       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.039198 |                          0.000094 | True                                  |       0.334664 |       0.000734 |        0.000000 |        0.000000 |           0.000000 |
| s2_rebuild61       |                 83 |           1660 |           1660 |                           0 |                           0 |                          0.038563 |                          0.000094 | True                                  |       0.329247 |       0.000734 |        0.000000 |        0.000000 |           0.000000 |

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
| mct_lrq74_drop      | selected compact structural law     |       74 |             0.010084 |        0.006080 |      0.010495 |              0.946486 | Best drop-preserving member of the exponent-search frontier.                                    |         0.965338 |             1.024914 |              1.031035 |            1.012177 |
| mct_lrq74_balanced  | accuracy-balanced sibling           |       74 |             0.010074 |        0.006073 |      0.010434 |              0.928873 | Slightly better RMSE; long-drop ratio less close to one.                                        |         0.953371 |             1.013185 |              1.022437 |            0.978074 |
| cbs_lrq74_s5        | previous selected structural law    |       74 |             0.011068 |        0.006122 |      0.011103 |              0.893596 | Earlier CBS-LRQ law; good safety, more compressed long drop.                                    |         0.923268 |             0.983338 |              0.981291 |            0.934142 |
| s2_base             | delayed structural reference        |       61 |             0.011284 |        0.007176 |      0.012217 |              0.894552 | Clean structural reference; raw optima collapse in delayed report.                              |         0.968659 |             1.045710 |              0.988619 |            0.935142 |
| powerbeta_compact98 | delayed empirical compact reference |       98 |             0.009098 |        0.004847 |      0.011258 |              0.949797 | Best under-100 empirical compact delayed reference; less clean D-monotonic structural behavior. |         0.960000 |             0.990000 |              0.993157 |            0.963347 |

## Search and negative results

The archive includes `fast_exponent_grid.csv` and `fast_refine.csv`. These show that the useful move was not adding more width; it was changing the exponent geometry so the interaction term acts as a small high-curvature token-return mode. Attempts to add an explicit residual `mu` head either received zero coefficient under realized-`D` modeling or worsened fixed-340M accuracy when using `D_base = D/mu`.

| candidate                        |   params |   seed7_holdout_rmse |   fixed340_rmse |   all900_rmse | reason_rejected                                                                                                                         |
|:---------------------------------|---------:|---------------------:|----------------:|--------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| realized_D_plus_residual_mu_head |       88 |             0.011068 |        0.006122 |      0.011103 | The explicit residual continuation head received effectively zero nonnegative coefficient; realized-D terms already spanned its signal. |
| D_base_plus_power_continuation   |       88 |             0.013600 |        0.017100 |      0.016400 | Structurally appealing separation of base scale and continuation, but it severely hurt fixed-340M prediction in this LRQ body.          |
| mct_lrq74_balanced               |       74 |             0.010074 |        0.006073 |      0.010434 | Slightly better RMSE balance, but selected drop variant has better 0.5x->2.0x and 1.0x->2.0x continuation ratios.                       |

## Artifact map

- `code/run_mct_lrq_law.py`: main generation script.
- `code/cbs_lrq_base.py`: CBS-LRQ base classes/functions used by the main script.
- `code/fast_refine_exponent_search.py` and `code/fast_exponent_grid_search.py`: exponent-search scripts used to find the selected point.
- `csv/`: metrics, predictions, fixed-budget drops, beta diagnostics, optima, monotonicity, parameter counts, exponent-search tables, and negative results.
- `models/`: selected and control JSON/NPZ artifacts.
- `plots/`: prediction, drop, optima, search, and scale-path diagnostics.
