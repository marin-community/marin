# 300M GRP Phase-Premium Ablation

## Question

Does removing the old GRP effective-exposure phase-1 multiplier improve the 300M fit?

The ablation keeps the old GRP blocks but changes the feature for each singleton, CC pair, and family total:

$$\phi_g=(1+\gamma r_g) f_g(z_g),\qquad r_g=\frac{e_{1,g}}{z_g+\epsilon}.$$

Penalties use raw exposure `z_g`; phase 1 no longer saturates earlier by construction.
Nonlinear parameters were selected with a fast CV objective that excludes raw-optimum diagnostics; the final selected row was then evaluated with the full raw-optimum diagnostics shown below.

## Results

| variant                                       |   total_param_count |   train_rmse |   oof_spearman |   cv_rmse |   cv_foldmean_regret_at_1 |   lower_tail_optimism |   cv_depopt_best8 |   cv_rawopt_nearest_tv |   raw_nearest_observed_tv |   raw_nearest_observed_value |   phase0_max_weight |   phase1_max_weight |
|:----------------------------------------------|--------------------:|-------------:|---------------:|----------:|--------------------------:|----------------------:|------------------:|-----------------------:|--------------------------:|-----------------------------:|--------------------:|--------------------:|
| power_family_penalty_effective_exposure_no_l2 |                  42 |     0.010102 |       0.804042 |  0.011141 |                  0.007789 |              0.005019 |          0.089311 |               0.539505 |                  0.500000 |                     0.968801 |            0.025641 |            0.393906 |
| power_family_penalty_phase_premium_no_l2      |                  41 |     0.012055 |       0.780390 |  0.012981 |                  0.011016 |              0.004442 |          0.796943 |               0.828047 |                  0.500000 |                     0.968801 |            1.000000 |            0.983481 |

## Interpretation

Negative result: the phase-premium ablation fits worse and produces a less trustworthy raw optimum. CV RMSE increases from 0.011141 to 0.012981, OOF Spearman drops from 0.804042 to 0.780390, and the raw-optimum calibration penalty rises from 0.089311 to 0.796943.

- The phase-premium model is useful only if it improves CV fit and reduces raw-optimum degeneracy relative to the existing effective-exposure no-L2 GRP.
- `cv_rawopt_nearest_tv` and `raw_nearest_observed_tv` should be treated as optimizer-sanity metrics; lower is better.
- `cv_depopt_best8` is a calibration penalty for raw optima that predict substantially better than nearby observed mixtures.

## Artifacts

- `summary.csv`: comparison table.
- `coarse.csv` and `refine.csv`: nonlinear tuning records.
- `predicted_vs_actual.png`: in-sample fit plot for the phase-premium ablation.
- `raw_optimum_mixture.png`: raw optimum mixture plot.
