# Canonical Compact Structural MCT-LRQ

## Selected Model

Canonical compact structural model: `mct_lrq69_drop_no_barrier`, referred to here as `MCT-LRQ69-drop`.

This model is the compact LRQ-anchor plus monotone Chinchilla-style continuation law. It has no compatibility-barrier offset; all prediction comes from the LRQ mixture anchor and the three monotone scale-continuation terms.

Text form:

```text
L(w,N,D) = E_LRQ(w)
         + A((N/N0)^(-0.154791)-1)
         + B_fam(w)((D/D0)^(-0.146425)-1)
         + C((N/N0)^(-0.014295)(D/D0)^(-1.063376)-1)
```

Symbolic LaTeX form:
$$
L(w,N,D) =
E_{\mathrm{LRQ}}(w)
+ A\left[\left(\frac{N}{N_0}\right)^{-\alpha} - 1\right]
+ B_{\mathrm{fam}}(w)\left[\left(\frac{D}{D_0}\right)^{-\beta} - 1\right]
+ C\left[
    \left(\frac{N}{N_0}\right)^{-\gamma}
    \left(\frac{D}{D_0}\right)^{-\delta}
    - 1
  \right].
$$


```latex
\[
L(w,N,D) =
E_{\mathrm{LRQ}}(w)
+ A\left[\left(\frac{N}{N_0}\right)^{-\alpha} - 1\right]
+ B_{\mathrm{fam}}(w)\left[\left(\frac{D}{D_0}\right)^{-\beta} - 1\right]
+ C\left[
    \left(\frac{N}{N_0}\right)^{-\gamma}
    \left(\frac{D}{D_0}\right)^{-\delta}
    - 1
  \right]
\]
```

Value-substituted LaTeX form:
$$
\begin{aligned}
L(w,N,D)
&= E_{\mathrm{LRQ}}(w) \\
&\quad + 0.419021
  \left[
    \left(\frac{N}{102{,}648{,}576}\right)^{-0.154791} - 1
  \right] \\
&\quad + B_{\mathrm{fam}}(w)
  \left[
    \left(\frac{D}{5{,}999{,}951{,}872}\right)^{-0.146425} - 1
  \right] \\
&\quad + 0.002742
  \left[
    \left(\frac{N}{102{,}648{,}576}\right)^{-0.014295}
    \left(\frac{D}{5{,}999{,}951{,}872}\right)^{-1.063376}
    - 1
  \right].
\end{aligned}
$$


```latex
\[
\begin{aligned}
L(w,N,D)
&= E_{\mathrm{LRQ}}(w) \\
&\quad + 0.419021
  \left[
    \left(\frac{N}{102{,}648{,}576}\right)^{-0.154791} - 1
  \right] \\
&\quad + B_{\mathrm{fam}}(w)
  \left[
    \left(\frac{D}{5{,}999{,}951{,}872}\right)^{-0.146425} - 1
  \right] \\
&\quad + 0.002742
  \left[
    \left(\frac{N}{102{,}648{,}576}\right)^{-0.014295}
    \left(\frac{D}{5{,}999{,}951{,}872}\right)^{-1.063376}
    - 1
  \right].
\end{aligned}
\]
```

Here `E_LRQ(w)` is the LRQ mixture anchor, `A=0.419021` and `C=0.002742` are constant nonnegative heads, and `B_fam(w)` is a learned nonnegative family-share head rather than a scalar. The total constant count is 69, with 60 fitted parameters when counting the exponents.

## Anchor Point Convention

The canonical version is anchored at corrected `100M/6B`, historically named `300m_6b`. This is not just a notation choice: `E_LRQ(w)` is ridge-fit on the `100M/6B`, `mu=1.0` training rows, and the scale-continuation terms are centered so that they vanish at `(N0,D0) = (102,648,576, 5,999,951,872)`. Therefore, at the anchor point:

```latex
\[
L(w,N_0,D_0) = E_{\mathrm{LRQ}}(w).
\]
```

For comparison, I also ran a controlled local refit where both the LRQ anchor rows and `(N0,D0)` were moved to corrected `60M/1.2B`, historically named `60m_1p2b`. The structural form is the same, but the scale centering changes to:

```latex
\[
\begin{aligned}
L_{60\mathrm{M}}(w,N,D)
&= E^{60\mathrm{M}}_{\mathrm{LRQ}}(w) \\
&\quad + 0.436231
  \left[
    \left(\frac{N}{58{,}998{,}528}\right)^{-0.154791} - 1
  \right] \\
&\quad + B^{60\mathrm{M}}_{\mathrm{fam}}(w)
  \left[
    \left(\frac{D}{1{,}199{,}833{,}088}\right)^{-0.146425} - 1
  \right] \\
&\quad + 0.061626
  \left[
    \left(\frac{N}{58{,}998{,}528}\right)^{-0.014295}
    \left(\frac{D}{1{,}199{,}833{,}088}\right)^{-1.063376}
    - 1
  \right].
\end{aligned}
\]
```

The 60M-anchor version changes performance a lot and should not replace the current canonical anchor. It overfits the 60M level and worsens cross-scale continuation:

| anchor | seed7 holdout RMSE | fixed-340M RMSE | fixed-340M slope | all-900M leaveout RMSE | 0.5x->1.0x drop ratio | 0.5x->2.0x drop ratio | 1.0x->2.0x drop ratio |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 100M/6B anchor | 0.010136 | 0.006138 | 0.960968 | 0.010522 | 1.036086 | 0.933632 | 0.998532 |
| 60M/1.2B anchor | 0.020631 | 0.021211 | 0.752357 | 0.028520 | 0.747887 | 0.608377 | 0.591216 |

The takeaway is that the current MCT-LRQ law is a 100M/6B-anchored mixture regression plus scale continuation. Moving the anchor down to 60M/1.2B is not an innocuous reparameterization under the current fitting procedure because the anchor regression is separately fit at that scale.

## Cross-Term Ablation

I also ran a direct ablation that removes the last coupled scale-continuation term:

```latex
\[
C\left[
  \left(\frac{N}{N_0}\right)^{-\gamma}
  \left(\frac{D}{D_0}\right)^{-\delta}
  - 1
\right].
\]
```

The ablated model keeps the same 100M/6B LRQ anchor, keeps the same `alpha=0.154791` and `beta=0.146425`, and refits the nonnegative `A` and `B_fam` heads:

```latex
\[
\begin{aligned}
L_{\mathrm{no\ cross}}(w,N,D)
&= E_{\mathrm{LRQ}}(w) \\
&\quad + 0.425139
  \left[
    \left(\frac{N}{102{,}648{,}576}\right)^{-0.154791} - 1
  \right] \\
&\quad + B^{\mathrm{no\ cross}}_{\mathrm{fam}}(w)
  \left[
    \left(\frac{D}{5{,}999{,}951{,}872}\right)^{-0.146425} - 1
  \right].
\end{aligned}
\]
```

This saves only 3 constants, from 69 total / 60 fitted to 66 total / 57 fitted. In a same-script refit, the cross term is worth keeping:

| model | total params | fitted params | seed7 holdout RMSE | fixed-340M RMSE | fixed-340M slope | all-900M leaveout RMSE |
|:--|--:|--:|--:|--:|--:|--:|
| full cross term | 69 | 60 | 0.010136 | 0.006138 | 0.960968 | 0.010522 |
| no cross term | 66 | 57 | 0.010738 | 0.006694 | 1.043898 | 0.020816 |

The no-cross model has acceptable in-distribution RMSE degradation, but it damages the larger-scale extrapolation and makes target-budget drops too aggressive:

| model | 0.5x->1.0x drop ratio | 0.5x->2.0x drop ratio | 1.0x->2.0x drop ratio |
|:--|--:|--:|--:|
| full cross term | 1.036086 | 0.933632 | 0.998532 |
| no cross term | 1.135424 | 1.044105 | 1.134163 |

Interpretation: the cross term is not just decorative. It is a small high-curvature correction that helps balance short-vs-long target-budget continuation and prevents the model from shifting too much scale behavior into the separate `A` and `B_fam` heads. The simpler no-cross law is cleaner, but not better enough to justify the 900M and drop-shape regression.

![Cross-term ablation predicted vs actual](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/joint_model_refreshed_20260426/mct_lrq_no_barrier_canonical/plots/mct_lrq_cross_term_ablation_pred_actual.png)

## Scorecard

| model                     | canonical_name     |   seed7_holdout_rmse |   seed7_holdout_spearman |   fixed340_rmse |   fixed340_slope |   fixed340_std_ratio |   random_supplement_rmse |   all900_rmse |   all900_spearman |   param_count_total |   fitted_param_count |
|:--------------------------|:-------------------|---------------------:|-------------------------:|----------------:|-----------------:|---------------------:|-------------------------:|--------------:|------------------:|--------------------:|---------------------:|
| mct_lrq69_drop_no_barrier | MCT-LRQ69-drop |             0.010075 |                 0.970703 |        0.006143 |         0.952454 |             1.013199 |                 0.012334 |      0.010177 |          1.000000 |                  69 |                   60 |

## Predictive Metrics

| model                       | fit_protocol   | split                  |   n |     rmse |   spearman |   bias_pred_minus_actual |   slope_pred_on_actual |   std_ratio |   low_tail_rmse |
|:----------------------------|:---------------|:-----------------------|----:|---------:|-----------:|-------------------------:|-----------------------:|------------:|----------------:|
| mct_lrq74_balanced_barrier5 | seed7          | seed7_holdout          |  61 | 0.010030 |   0.971391 |                 0.001131 |               1.036464 |    1.045867 |        0.005480 |
| mct_lrq74_balanced_barrier5 | seed7          | fixed340_holdout       |  27 | 0.006068 |   0.931013 |                 0.000153 |               0.946642 |    1.006390 |        0.005444 |
| mct_lrq74_balanced_barrier5 | seed7          | random_supplement      |  34 | 0.012299 |   0.869213 |                 0.001907 |               1.095154 |    1.118086 |        0.011476 |
| mct_lrq69_drop_no_barrier   | seed7          | seed7_holdout          |  61 | 0.010075 |   0.970703 |                 0.001309 |               1.034296 |    1.043839 |        0.005383 |
| mct_lrq69_drop_no_barrier   | seed7          | fixed340_holdout       |  27 | 0.006143 |   0.923687 |                 0.000535 |               0.952454 |    1.013199 |        0.005529 |
| mct_lrq69_drop_no_barrier   | seed7          | random_supplement      |  34 | 0.012334 |   0.868908 |                 0.001924 |               1.094590 |    1.117728 |        0.011551 |
| mct_lrq74_balanced_barrier5 | leave900out    | all900_leave_scale_out |   4 | 0.010358 |   1.000000 |                -0.001646 |               0.525503 |    0.601499 |        0.015142 |
| mct_lrq69_drop_no_barrier   | leave900out    | all900_leave_scale_out |   4 | 0.010177 |   1.000000 |                -0.001833 |               0.576529 |    0.671466 |        0.014610 |

## Same-Mixture Target-Budget Drop Metrics

| model                       | drop_pair    |   n |   actual_drop_mean |   pred_drop_mean |   drop_error_mean |   drop_ratio_mean |   drop_ratio_median |   drop_rmse |
|:----------------------------|:-------------|----:|-------------------:|-----------------:|------------------:|------------------:|--------------------:|------------:|
| mct_lrq69_drop_no_barrier   | 0.5x_to_1.0x |  12 |           0.023636 |         0.023575 |         -0.000061 |          1.024984 |            0.935621 |    0.004194 |
| mct_lrq69_drop_no_barrier   | 0.5x_to_2.0x |   3 |           0.048517 |         0.044814 |         -0.003703 |          0.922907 |            0.933046 |    0.003865 |
| mct_lrq69_drop_no_barrier   | 1.0x_to_2.0x |   3 |           0.021206 |         0.020905 |         -0.000301 |          0.985248 |            0.993167 |    0.000822 |
| mct_lrq74_balanced_barrier5 | 0.5x_to_1.0x |  12 |           0.023636 |         0.023328 |         -0.000308 |          1.013638 |            0.920644 |    0.004076 |
| mct_lrq74_balanced_barrier5 | 0.5x_to_2.0x |   3 |           0.048517 |         0.044695 |         -0.003821 |          0.920379 |            0.931530 |    0.004019 |
| mct_lrq74_balanced_barrier5 | 1.0x_to_2.0x |   3 |           0.021206 |         0.020535 |         -0.000671 |          0.967724 |            0.978337 |    0.001037 |

## Optimum Diagnostics

| model                     | target_scale   | opt_kind                      |   predicted_bpb | hard_corner_flag   | phase1_tech_collapse_flag   | any_family_collapse_flag   |   nearest_observed_phase_mean_tv |   p0_broad_text_share |   p0_tech_code_share |   p0_reasoning_share |   p1_broad_text_share |   p1_tech_code_share |   p1_reasoning_share |
|:--------------------------|:---------------|:------------------------------|----------------:|:-------------------|:----------------------------|:---------------------------|---------------------------------:|----------------------:|---------------------:|---------------------:|----------------------:|---------------------:|---------------------:|
| mct_lrq69_drop_no_barrier | 340M/10.4B     | raw_random_search             |        0.831192 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier | 340M/10.4B     | top8actual_hull_random_search |        0.843535 | False              | False                       | False                      |                         0.221662 |              0.716519 |             0.259871 |             0.023610 |              0.698604 |             0.279252 |             0.022145 |
| mct_lrq69_drop_no_barrier | 900M/24B       | raw_random_search             |        0.748350 | True               | True                        | True                       |                         0.000000 |              0.475148 |             0.212085 |             0.312768 |              0.000000 |             0.999999 |             0.000001 |
| mct_lrq69_drop_no_barrier | 900M/24B       | top8actual_hull_random_search |        0.772922 | False              | False                       | False                      |                         0.206398 |              0.749294 |             0.228933 |             0.021773 |              0.700814 |             0.278689 |             0.020496 |

## Read

- Predictive quality remains at the same level as the earlier compatibility-barrier comparison model on observed rows: seed-7 holdout RMSE is about `0.0101`; fixed-340M RMSE is about `0.0061`.
- Drop preservation remains good, though the compact drop variant is slightly weaker on the long `0.5x -> 2.0x` ratio than the earlier selected `mct_lrq74_drop` comparison model.
- Structural monotonicity is preserved: the scale heads are nonnegative and all exponents are positive, with zero monotonicity-grid violations in the refreshed ablation.
- Raw simplex optima are not solved. The raw 100M+ optima collapse to the observed pathological phase-1-tech mixture. Use hull/trust-region deployment for operational optima, or keep raw optima explicitly labeled as unresolved.

## Artifacts

- `csv/metric_summary.csv`
- `csv/fixed340_drop_summary.csv`
- `csv/optimum_diagnostics.csv`
- `csv/monotonicity_grid.csv`
- `csv/row_predictions.csv`
- `csv/anchor_point_ablation.csv`
- `csv/anchor_point_ablation_heads.csv`
- `csv/cross_term_ablation_metrics.csv`
- `csv/cross_term_ablation_predictions.csv`
- `csv/cross_term_ablation_drops.csv`
- `csv/cross_term_ablation_heads.csv`
- `plots/mct_lrq69_drop_no_barrier_seed7_holdout_pred_actual.png`
- `plots/mct_lrq69_drop_no_barrier_fixed340_pred_actual.png`
- `plots/mct_lrq69_drop_no_barrier_all_rows_pred_actual.png`
- `plots/mct_lrq69_drop_no_barrier_leave900out_all_rows_pred_actual.png`
- `plots/mct_lrq69_drop_no_barrier_drop_ratios.png`
- `plots/mct_lrq69_drop_no_barrier_optimum_family_shares.png`
- `plots/mct_lrq_cross_term_ablation_pred_actual.png`
