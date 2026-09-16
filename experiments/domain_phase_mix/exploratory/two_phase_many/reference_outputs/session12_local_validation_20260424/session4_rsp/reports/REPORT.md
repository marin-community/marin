# Continued structural-law pass: regularized structural power anchors

## Executive summary

This pass found a useful continuation of the SP94 direction: **regularize the mixture anchor strongly, move the power feature toward linearity (`rho=0.60-0.70`), and optionally add one global data-power carrier**.  The best balanced candidate is:

`rsp95_rho070_d030_l2p2_transfer`

It has `95` learned coefficients (`100` scalars if counting fixed `rho`, `alpha`, `beta`, `gamma`, and the extra data exponent).  The strict under-100 option, `rsp94_core_rho065_l2p2_balanced`, has `94` learned coefficients and `98` total scalars including fixed exponents.

The important result is not seed-7 holdout: holdout remains worse than the empirical compact power-beta reference.  The useful result is structural: this family keeps the monotone fixed-mixture scaling form, improves fixed-340M versus S2/SP94, predicts same-mixture target-budget drops without the severe compression seen in older compact forms, and keeps raw simplex optima broad rather than corner/family collapsed.

## Exact law

Let

```text
u_N  = (N/N0)^(-alpha) - 1
u_D  = (D/D0)^(-beta) - 1
u_ND = ((N/N0)(D/D0))^(-gamma) - 1
```

with `N0=102648576`, `D0=5999951872`, `alpha=0.18`, `beta=0.26`, `gamma=0.11`.  The mixture anchor is

```text
E_rho(w) = theta0 + sum_(phase,domain) theta[p,d] w[p,d]^rho
           + sum_phase [lambda_domain sum_d w[p,d]^2
                        + lambda_family sum_f s[p,f]^2
                        + lambda_scarcity sum_d r_d w[p,d]^2]
```

where `lambda_* >= 0`.  The scale heads use

```text
g(w) = (1, mean_tech_share, mean_reasoning_share)
```

and

```text
L(w,N,D,mu) = E_rho(w) + A(g(w)) u_N + B(g(w)) u_D + C(g(w)) u_ND
              + q_D ((D/D0)^(-0.30) - 1)
```

for the primary balanced candidate, with every coefficient inside `A`, `B`, `C`, and `q_D` constrained nonnegative.  `mu` is not a separate state variable; continuation is represented through realized train tokens `D`, so same-mixture multiplier changes remain real scale changes rather than a post-hoc offset.

For fixed `w`, every scale basis term is decreasing in `N` and/or `D`, with nonnegative coefficients, so the law is monotone.  For fixed `N,D,mu`, all scale terms are constants and the law is a mixture regression in `w`.

Fitting uses bounded least squares, same-mixture pair-difference rows, and Tikhonov regularization on the scaled design.  Primary hyperparameters: `rho=0.60-0.70`, pair-difference weight `1.5`, L2 `1.25`.

## Primary metrics

| subset              |   rows |     rmse |   spearman |      bias |    slope |   std_ratio |   low_tail_rmse |
|:--------------------|-------:|---------:|-----------:|----------:|---------:|------------:|----------------:|
| holdout             |     61 | 0.011922 |   0.967795 |  0.003200 | 1.035428 |    1.048153 |        0.004622 |
| fixed_340m          |     27 | 0.005910 |   0.923077 |  0.001308 | 1.015581 |    1.067270 |        0.004697 |
| random_supplement   |     34 | 0.015076 |   0.852406 |  0.004703 | 1.051019 |    1.088649 |        0.007358 |
| validation_900m_all |      4 | 0.007107 |   1.000000 | -0.005434 | 1.170275 |    1.184444 |        0.011844 |

Reference context from the packet:

| model | holdout RMSE | fixed-340M RMSE | fixed slope | all-900M RMSE |
|---|---:|---:|---:|---:|
| S2 structural | 0.011284 | 0.007176 | 0.969 | 0.012217 |
| power-beta compact98 | 0.009098 | 0.004847 | 0.960 | 0.011258 |
| S7 empirical frontier | 0.006232 | 0.004013 | 0.966 | 0.007869 |

Read: the RSP law is **not** a holdout-RMSE promotion over power-beta.  It is a structural candidate with much better all-900M transfer than S2/power-beta and materially better raw-optimum geometry than the empirical frontier-style forms.

## Variant summary and negative results

| model                           |   learned_params |   all_scalar_count |      rho |   pair_weight |       l2 | extra_terms            |   holdout_rmse |   fixed340_rmse |   fixed340_slope |   fixed340_std_ratio |   all900_leaveout_rmse |   drop_ratio_0.5->1 |   drop_ratio_0.5->2 |   drop_ratio_1->2 |
|:--------------------------------|-----------------:|-------------------:|---------:|--------------:|---------:|:-----------------------|---------------:|----------------:|-----------------:|---------------------:|-----------------------:|--------------------:|--------------------:|------------------:|
| rsp95_rho070_d030_l2p2_transfer |               95 |                100 | 0.700000 |      1.500000 | 2.000000 | [["D", 0.3, "const"]]  |       0.011922 |        0.005910 |         1.015581 |             1.067270 |               0.007107 |            1.049011 |            0.873612 |          0.902319 |
| rsp95_rho065_d030_l2p2_balanced |               95 |                100 | 0.650000 |      1.500000 | 2.000000 | [["D", 0.3, "const"]]  |       0.011867 |        0.005882 |         1.019493 |             1.071938 |               0.007663 |            1.049285 |            0.870286 |          0.898942 |
| rsp95_rho060_d036_l2p25_shape   |               95 |                100 | 0.600000 |      1.500000 | 2.500000 | [["D", 0.36, "const"]] |       0.011788 |        0.006123 |         0.997412 |             1.055306 |               0.008137 |            1.016162 |            0.829313 |          0.845610 |
| rsp95_rho060_d040_l2p25_holdout |               95 |                100 | 0.600000 |      1.500000 | 2.500000 | [["D", 0.4, "const"]]  |       0.011775 |        0.006202 |         0.976487 |             1.036463 |               0.007719 |            0.990537 |            0.802062 |          0.809980 |
| rsp95_prev_d022_l2p125          |               95 |                100 | 0.700000 |      1.500000 | 1.250000 | [["D", 0.22, "const"]] |       0.012046 |        0.005854 |         1.037676 |             1.088557 |               0.007782 |            1.079630 |            0.911035 |          0.952792 |
| rsp94_core_rho070_l2p2_transfer |               94 |                 98 | 0.700000 |      1.500000 | 2.000000 | []                     |       0.012008 |        0.005935 |         1.027054 |             1.079242 |               0.008518 |            1.069290 |            0.881657 |          0.917639 |
| rsp94_core_rho065_l2p2_balanced |               94 |                 98 | 0.650000 |      1.500000 | 2.000000 | []                     |       0.011948 |        0.005905 |         1.030788 |             1.083319 |               0.008909 |            1.069219 |            0.878329 |          0.914154 |
| rsp94_core_rho060_l2p25_shape   |               94 |                 98 | 0.600000 |      1.500000 | 2.500000 | []                     |       0.011899 |        0.006043 |         1.035984 |             1.090683 |               0.010298 |            1.070152 |            0.871863 |          0.907384 |

Notes:

- `rsp94_core_rho070_pw15_l2p125` is the strict under-100-total-scalar version.  It is nearly tied with the primary and has slightly better all-900M RMSE, but its fixed-340M continuation ratios are a little more compressed.
- `rsp95_d030_frontier_rho070_pw15_l2p125` improves all-900M further, but the `0.5x->2x` and `1x->2x` drop ratios move down.  This is a useful negative result: a stronger global data-curvature term can buy extrapolation by re-compressing long multiplier drops.
- Higher pair weights reduce fixed-340M RMSE locally but degrade random holdout and/or all-900M, so this pass does not use optimizer-depth or pair-heavy checkpoint selection as the main win.

## Fixed-340M target-budget drops

| drop   |   n_pairs |   actual_drop |   predicted_drop |   drop_error |   drop_ratio |   drop_rmse |
|:-------|----------:|--------------:|-----------------:|-------------:|-------------:|------------:|
| 0.5->1 |        12 |      0.023636 |         0.024059 |     0.000423 |     1.049011 |    0.004077 |
| 0.5->2 |         3 |      0.048517 |         0.042304 |    -0.006213 |     0.873612 |    0.006800 |
| 1->2   |         3 |      0.021206 |         0.019102 |    -0.002104 |     0.902319 |    0.002410 |

## Fixed-340M beta diagnostics

| model                           |   n_triples |   actual_beta_mean |   actual_beta_std |   predicted_beta_mean |   predicted_beta_std |
|:--------------------------------|------------:|-------------------:|------------------:|----------------------:|---------------------:|
| rsp95_rho070_d030_l2p2_transfer |           3 |           0.364509 |          0.066310 |              0.280515 |             0.000736 |
| rsp95_rho065_d030_l2p2_balanced |           3 |           0.364509 |          0.066310 |              0.280284 |             0.000809 |
| rsp95_rho060_d036_l2p25_shape   |           3 |           0.364509 |          0.066310 |              0.314153 |             0.002472 |
| rsp95_rho060_d040_l2p25_holdout |           3 |           0.364509 |          0.066310 |              0.339212 |             0.003654 |
| rsp95_prev_d022_l2p125          |           3 |           0.364509 |          0.066310 |              0.247434 |             0.000420 |
| rsp94_core_rho070_l2p2_transfer |           3 |           0.364509 |          0.066310 |              0.260000 |             0.000000 |
| rsp94_core_rho065_l2p2_balanced |           3 |           0.364509 |          0.066310 |              0.260000 |             0.000000 |
| rsp94_core_rho060_l2p25_shape   |           3 |           0.364509 |          0.066310 |              0.260000 |             0.000000 |

## Optimum diagnostics for the primary candidate

| model                           | optimum_type                 | target_display   |   predicted_bpb |   phase0_broad_text |   phase0_tech_code |   phase0_reasoning |   phase1_broad_text |   phase1_tech_code |   phase1_reasoning |   phase0_effective_support |   phase1_effective_support |   phase0_max_weight |   phase1_max_weight |   nearest_observed_mean_phase_tv | family_collapse   | phase1_tech_collapse   | monotone_60_100_340_900   |
|:--------------------------------|:-----------------------------|:-----------------|----------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|-------------------:|---------------------------:|---------------------------:|--------------------:|--------------------:|---------------------------------:|:------------------|:-----------------------|:--------------------------|
| rsp95_rho070_d030_l2p2_transfer | raw_unrestricted             | 60M/1.2B         |          0.9837 |              0.6351 |             0.2610 |             0.1039 |              0.5840 |             0.3393 |             0.0767 |                    29.8961 |                    20.2994 |              0.0832 |              0.1527 |                           0.2288 | False             | False                  | True                      |
| rsp95_rho070_d030_l2p2_transfer | raw_unrestricted             | 100M/6B          |          0.8700 |              0.6046 |             0.2758 |             0.1196 |              0.5568 |             0.3545 |             0.0888 |                    29.1045 |                    19.5399 |              0.0872 |              0.1587 |                           0.2373 | False             | False                  | True                      |
| rsp95_rho070_d030_l2p2_transfer | raw_unrestricted             | 340M/10.4B       |          0.7816 |              0.5828 |             0.2856 |             0.1317 |              0.5374 |             0.3643 |             0.0982 |                    28.4282 |                    19.0248 |              0.0898 |              0.1626 |                           0.2442 | False             | False                  | True                      |
| rsp95_rho070_d030_l2p2_transfer | raw_unrestricted             | 900M/24B         |          0.7100 |              0.5643 |             0.2935 |             0.1422 |              0.5211 |             0.3723 |             0.1066 |                    27.7892 |                    18.5942 |              0.0919 |              0.1657 |                           0.2508 | False             | False                  | True                      |
| rsp95_rho070_d030_l2p2_transfer | top8actual_hull              | 340M/10.4B       |          0.8031 |              0.5678 |             0.3163 |             0.1159 |              0.3280 |             0.6156 |             0.0564 |                    20.1122 |                     3.6616 |              0.1525 |              0.5157 |                           0.3649 | False             | False                  | True                      |
| rsp95_rho070_d030_l2p2_transfer | trustblend_top8actual_cap034 | 340M/10.4B       |          0.7801 |              0.5798 |             0.2917 |             0.1285 |              0.4955 |             0.4146 |             0.0899 |                    27.6791 |                    13.3011 |              0.1023 |              0.2332 |                           0.2518 | False             | False                  | True                      |

The primary raw optima are not hard corners and do not collapse to a single family.  They are farther from the observed mixture cloud than the previous SP94 raw optimum, but remain broad in both phases and are accompanied by hull/trust-region variants for deployment-style use.  The constrained `top8actual_hull` and `trustblend_top8actual_cap034` rows are included separately rather than hiding the raw solution.

## Structural sanity

| check                                             |       value | pass   |
|:--------------------------------------------------|------------:|:-------|
| positive_A_coefficients                           | 0.0914524   | True   |
| positive_B_coefficients                           | 0.0496731   | True   |
| positive_C_coefficients                           | 1.01619e-23 | True   |
| positive_extra_scale_coefficients                 | 0.0610747   | True   |
| fixed_scale_is_mixture_regression                 | 1           | True   |
| fixed_w_monotone_by_construction                  | 1           | True   |
| random_mixture_60_100_340_900_monotone_violations | 0           | True   |
| observed_prediction_min                           | 0.778011    | True   |
| observed_prediction_max                           | 1.3778      | True   |
| raw_hard_corner_count                             | 0           | True   |
| raw_family_collapse_count                         | 0           | True   |

## Artifact map

- `code/run_rsp_pareto_law.py`: reproducible fitting/evaluation script.
- `models/*_seed7_model.json`: seed-7 model artifacts.
- `models/*_all900_protocol_model.json`: all-non-900M refit artifacts.
- `csv/metric_summary.csv`: metrics for all selected variants.
- `csv/predictions_seed7.csv`, `csv/all900_predictions.csv`: row-level predictions.
- `csv/fixed340_multiplier_drop_*.csv`, `csv/fixed340_beta_hat_*.csv`: continuation diagnostics.
- `csv/optimum_diagnostics.csv`, `csv/optimum_weights.csv`: raw and constrained optimum probes.
- `csv/structural_sanity.csv`: structural checks for the primary model.
- `plots/*.png`: prediction, drop, RMSE, and optimum diagnostics.
