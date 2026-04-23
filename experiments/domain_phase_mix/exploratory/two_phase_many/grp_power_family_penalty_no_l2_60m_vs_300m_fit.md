| label                 | fit_scale   |   fit_row_count |   predicted_optimum_value |   train_rmse |   cv_rmse |   train_spearman |   oof_spearman |   cv_foldmean_regret_at_1 |   cv_rawopt_nearest_tv |   phase0_broad_text |   phase0_tech_code |   phase0_reasoning |   phase1_broad_text |   phase1_tech_code |   phase1_reasoning |
|:----------------------|:------------|----------------:|--------------------------:|-------------:|----------:|-----------------:|---------------:|--------------------------:|-----------------------:|--------------------:|-------------------:|-------------------:|--------------------:|-------------------:|-------------------:|
| 60M-fit no-$L_2$ GRP  | 60m_1p2b    |             242 |                  1.013866 |     0.007598 |  0.008720 |         0.910534 |       0.869798 |                  0.002082 |               0.552822 |            0.643765 |           0.341605 |           0.014630 |            0.592771 |           0.402767 |           0.004462 |
| 300M-fit no-$L_2$ GRP | 300m_6b     |             242 |                  0.873591 |     0.010102 |  0.011141 |         0.829115 |       0.804042 |                  0.007789 |               0.539505 |            0.794872 |           0.153846 |           0.051282 |            0.000006 |           0.999994 |           0.000000 |

## Read

- The `300M` no-`L_2` GRP still fits nontrivially, but it is clearly worse than
  the original `60M` fit on every fit-quality metric that matters:
  - train RMSE: `0.00760 -> 0.01010`
  - CV RMSE: `0.00872 -> 0.01114`
  - train Spearman: `0.9105 -> 0.8291`
  - OOF Spearman: `0.8698 -> 0.8040`
  - CV foldmean regret@1: `0.00208 -> 0.00779`

- Qualitatively, the `60M` fit remains sane:
  - broad-text heavy in both phases
  - moderate tech tilt
  - almost no reasoning

- The raw `300M` optimum should **not** be trusted as a policy proposal:
  - the reported phase-0 family shares come from an exactly uniform
    `1 / 39` allocation over all domains
  - the reported phase-1 solution collapses almost entirely to tech/code

## Important caveat on the 300M optimum

The `300M` optimum is pathological for two separate reasons.

1. The continuous deployment optimizer is start-sensitive.
   The reported uniform phase-0 solution is only one shallow local basin.
   With more random starts, lower-objective corner solutions appear.

2. More seriously, the fitted `300M` surrogate becomes numerically blind to
   phase-0 composition.
   The fitted nonlinear parameters are extreme:
   - `lam = 54.6` at the top clip
   - `beta = 1e-6` at the bottom clip
   - `eta = 2980.96`

   Under those values, retained phase-0 exposure is effectively zero relative
   to the phase-1 term, so large phase-0 perturbations leave the surrogate
   design and prediction unchanged to machine precision.

So the right conclusion is:

- the `300M` no-`L_2` GRP fit is still useful as a **regression-fit datapoint**
- the `300M` no-`L_2` GRP optimized mixture is **not** trustworthy

See:

- [/Users/calvinxu/Projects/Work/Marin/marin/docs/debug-log-grp-no-l2-300m-uniform-phase0.md](/Users/calvinxu/Projects/Work/Marin/marin/docs/debug-log-grp-no-l2-300m-uniform-phase0.md)
