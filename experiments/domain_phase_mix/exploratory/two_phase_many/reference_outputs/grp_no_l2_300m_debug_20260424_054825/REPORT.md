# GRP no-L2 300M debug sprint

## Data audit

- 60M rows: `242`
- 300M rows: `242`
- 300M sources: `{'pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b': 241, 'pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b': 1}`

## Best row per trial

| trial                           | param_mode     | objective_mode   | method       |   fast_cv_rmse |   full_cv_rmse |   full_cv_spearman |   train_rmse |   train_spearman |   raw_predicted_optimum_value |   raw_nearest_observed_tv |   raw_phase0_max_weight |   raw_phase1_max_weight |   phase0_sensitivity_max_pred_delta |   hit_eta_upper |   hit_lam_upper | hit_beta_lower   | phase0_degenerate   |
|:--------------------------------|:---------------|:-----------------|:-------------|---------------:|---------------:|-------------------:|-------------:|-----------------:|------------------------------:|--------------------------:|------------------------:|------------------------:|------------------------------------:|----------------:|----------------:|:-----------------|:--------------------|
| current_fast_fixed_powell30     | standard       | fast             | Powell       |     0.00762644 |     0.00762644 |           0.915138 |   0.00674165 |         0.933709 |                      0.88148  |                  0.662893 |               0.331079  |                0.198172 |                          0.00280334 |               0 |               0 | False            | False               |
| fast_basin_hopping              | standard       | fast             | basinhopping |     0.00764918 |     0.00764918 |           0.916142 |   0.00685061 |         0.931906 |                      0.839052 |                  0.827853 |               0.999526  |                0.610382 |                          0.00578307 |               0 |               0 | False            | False               |
| fast_expanded_powell120         | standard       | fast             | Powell       |     0.00766694 |     0.00766694 |           0.912913 |   0.0069221  |         0.928402 |                      0.88112  |                  0.739066 |               0.264154  |                0.448545 |                          0.00266448 |               0 |               0 | False            | False               |
| fast_lbfgsb_expanded            | standard       | fast             | L-BFGS-B     |     0.0078786  |     0.0078786  |           0.899705 |   0.00693396 |         0.919758 |                      0.884676 |                  0.640464 |               0.274485  |                0.265947 |                          0.00139722 |               0 |               0 | False            | False               |
| fast_moderate_clip_powell120    | moderate_clip  | fast             | Powell       |     0.00742958 |     0.00742958 |           0.919069 |   0.0066294  |         0.935578 |                      0.875137 |                  0.723988 |               0.537608  |                0.315268 |                          0.00596597 |               0 |               0 | False            | False               |
| fast_nelder_mead_expanded       | standard       | fast             | Nelder-Mead  |     0.00774017 |     0.00774017 |           0.907516 |   0.00678265 |         0.929175 |                      0.870419 |                  0.736865 |               0.534075  |                0.310221 |                          0.00243874 |               0 |               0 | False            | False               |
| fast_prior_powell120            | standard       | fast             | Powell       |     0.00769564 |     0.00769564 |           0.91411  |   0.0068515  |         0.931308 |                      0.882251 |                  0.700063 |               0.275434  |                0.314608 |                          0.00332176 |               0 |               0 | False            | False               |
| fast_ridge1e-5_powell120        | standard       | fast             | Powell       |     0.00744433 |     0.00744433 |           0.919871 |   0.00677366 |         0.932551 |                      0.878809 |                  0.683694 |               0.314668  |                0.25144  |                          0.00290628 |               0 |               0 | False            | False               |
| fast_separate_phase_powell120   | separate_phase | fast             | Powell       |     0.00768931 |     0.00768931 |           0.915609 |   0.00687056 |         0.932385 |                      0.844779 |                  0.82206  |               0.90158   |                0.628861 |                          0.0112917  |             nan |             nan | False            | False               |
| legacy_300m_existing_artifact   | standard       | fast             | artifact     |     0.0104813  |     0.0104813  |           0.819166 |   0.00946544 |         0.847999 |                      0.555892 |                  0.557042 |               0.0626738 |                0.976703 |                          0          |               1 |               1 | True             | True                |
| reference_60m_existing_artifact | standard       | fast             | artifact     |     0.00919328 |     0.00919328 |           0.869798 |   0.00759754 |         0.910534 |                      1.01386  |                  0.583464 |               0.339203  |                0.207261 |                          0.00523913 |               0 |               0 | False            | False               |

## Read

- A candidate is degenerate when the 10% phase-0 perturbation changes prediction by less than `1e-5`.
- Boundary hits indicate whether the nonlinear optimizer is solving by saturating the original GRP retained-exposure parameters.
- Compare `fast_*` and `full_*` columns to see whether omitting deployment-support terms changes the selected solution.
- The 300M data panel itself passes the basic audit: `242` rows, no duplicate run names, and normalized phase weights after backfilling the missing 300M stratified baseline weights from the same run name.
- The legacy 300M artifact is the only degenerate row: `eta=2980.96`, `lam=54.60`, `beta=1e-6`, zero phase-0 sensitivity, and full CV RMSE `0.01048`.
- Better optimizer settings plus moderate clips improve the fixed-scale 300M fit to full CV RMSE `0.00743` and Spearman `0.919`, so GRP can fit this swarm when the retained-exposure parameters are kept in a sane range.
- Tiny ridge is nearly tied (`0.00744`) and has a less concentrated raw optimum than the moderate-clip winner.
- Basin hopping is not the answer here: it does not improve CV RMSE and drives the raw phase-0 optimum to a near corner (`0.999526` max weight).
- Raw optima remain off-manifold even after the fit is repaired, so these variants are regression diagnostics, not direct deployment policies.
