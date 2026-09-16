# Debugging log for GRP no-L2 300M uniform phase-0 optimum

Investigate why the freshly fit no-`L2` GRP on the `300m_6b` panel produced a phase-0 optimum that is exactly uniform over the 39 domains.

## Initial status

The comparison fit in
[/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.json](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/grp_power_family_penalty_no_l2_60m_vs_300m_fit_summary.json)
showed:

- `300M-fit no-L2 GRP`
- `phase0_max_weight = 0.025641...`, i.e. exactly `1 / 39`
- `phase1_tech_code = 0.999994...`, i.e. near-total phase-1 tech collapse

That looked too symmetric to be a meaningful optimizer result.

## Hypothesis 1

The continuous mixture optimizer is too shallow and is getting stuck in a local basin attached to the zero-logit initialization.

## Changes to make

No code changes. Probe the fitted `300M` surrogate with repeated calls to
`optimize_penalty_calibration_model(...)` using larger `n_random` and multiple seeds.

## Results

Using the fitted 300M surrogate parameters:

- `eta = 2980.96`
- `lam = 54.60` (top-clipped)
- `beta = 1e-6` (bottom-clipped)

and varying optimizer seeds / random starts:

- with `n_random = 1`, some seeds returned the reported shallow basin:
  - objective around `0.8735907`
  - phase 0 nearly uniform
- but other seeds found much lower objective values:
  - objective around `0.5570` to `0.5831`
  - phase 0 nearly one-hot
  - phase 1 even more collapsed
- with `n_random = 16` and `64`, those lower-objective corner solutions appeared more often

So the reported uniform phase-0 solution is not the global optimum of the continuous objective. It is a shallow local basin that survives because the optimizer starts from zero logits plus very few random perturbations.

## Hypothesis 2

The fitted surrogate has become numerically blind to phase-0 composition, so the zero-logit start remains a fixed point by symmetry.

## Changes to make

No code changes. Inspect retained-exposure magnitudes and compare the design matrix under large phase-0 perturbations while holding phase 1 fixed.

## Results

At the reported 300M solution:

- total retained phase-0 contribution:
  - `4.14e-15`
- total phase-1 contribution:
  - `7.39e4`
- ratio:
  - `5.60e-20`

This comes directly from the nonlinear saturation:

- `lam = 54.6` makes `exp(-lam * (1 - p1))` effectively zero unless `p1` is already large
- `eta = 2980.96` makes the `eta * c1 * p1` term dominate
- `beta = 1e-6` removes almost all low-quality pair contribution

As a result, the surrogate’s retained signal becomes:

- effectively a function of phase 1 only
- with negligible dependence on phase 0

Empirical check:

- shifting `10%` of phase-0 mass from the uniform solution into domains like:
  - `dolma3_stack_edu`
  - `dolma3_cc/health_high`
  - `dolmino_synth_instruction`
  - `dolma3_wikipedia`
- while keeping phase 1 fixed

changed:

- `max_design_diff = 0.0`
- `pred_diff = 0.0`

to machine precision.

So the uniform phase-0 solution is a true numerical degeneracy of the fitted surrogate, not just a plotting mistake.

## Future Work

- [ ] Treat the current 300M no-`L2` GRP optimum as invalid for qualitative policy interpretation.
- [ ] If this surrogate family is reused, increase `n_random` in `optimize_penalty_calibration_model(...)` so shallow local basins are less likely to be reported as the optimum.
- [ ] Add an optimization diagnostic for phase insensitivity, e.g. flag when large phase-0 perturbations leave the design unchanged.
- [ ] Consider rejecting or regularizing fits where `lam` hits the top clip and `beta` hits the bottom clip simultaneously, since that effectively removes phase-0 signal from the model.

## 2026-04-24 - expanded optimizer and model degeneracy sprint

Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_no_l2_300m_debug_20260424_054825`

Data audit passed: both the `60m_1p2b` and `300m_6b` panels have 242 rows, no duplicate run names, and phase weights normalized to 1 after backfilling the 300M stratified baseline weights from the same run name.

Summary table:

| trial                           |   full_cv_rmse |   full_cv_spearman |   param_eta |     param_lam |   param_beta |   raw_phase0_max_weight |   raw_phase1_max_weight |   phase0_sensitivity_max_pred_delta | phase0_degenerate   |
|:--------------------------------|---------------:|-------------------:|------------:|--------------:|-------------:|------------------------:|------------------------:|------------------------------------:|:--------------------|
| current_fast_fixed_powell30     |     0.00762644 |           0.915138 |    10.6417  |   0.027174    |     0.269774 |               0.331079  |                0.198172 |                          0.00280334 | False               |
| fast_basin_hopping              |     0.00764918 |           0.916142 |     8.30868 |   0.113395    |     0.370842 |               0.999526  |                0.610382 |                          0.00578307 | False               |
| fast_expanded_powell120         |     0.00766694 |           0.912913 |    12.4533  |   6.14421e-06 |     0.369169 |               0.264154  |                0.448545 |                          0.00266448 | False               |
| fast_lbfgsb_expanded            |     0.0078786  |           0.899705 |    20.8898  |   6.14421e-06 |     0.196768 |               0.274485  |                0.265947 |                          0.00139722 | False               |
| fast_moderate_clip_powell120    |     0.00742958 |           0.919069 |    10.9604  |   0.00123498  |     0.305104 |               0.537608  |                0.315268 |                          0.00596597 | False               |
| fast_nelder_mead_expanded       |     0.00774017 |           0.907516 |    12.2008  |   0.00139371  |     0.202978 |               0.534075  |                0.310221 |                          0.00243874 | False               |
| fast_prior_powell120            |     0.00769564 |           0.91411  |    10.2924  |   7.04971e-06 |     0.307185 |               0.275434  |                0.314608 |                          0.00332176 | False               |
| fast_ridge1e-5_powell120        |     0.00744433 |           0.919871 |    10.847   |   0.0282566   |     0.369407 |               0.314668  |                0.25144  |                          0.00290628 | False               |
| fast_separate_phase_powell120   |     0.00768931 |           0.915609 |   nan       | nan           |     0.339778 |               0.90158   |                0.628861 |                          0.0112917  | False               |
| legacy_300m_existing_artifact   |     0.0104813  |           0.819166 |  2980.96    |  54.5982      |     1e-06    |               0.0626738 |                0.976703 |                          0          | True                |
| reference_60m_existing_artifact |     0.00919328 |           0.869798 |     5.22244 |   7.04928e-06 |     0.196768 |               0.339203  |                0.207261 |                          0.00523913 | False               |

Interpretation:

- The originally reported 300M fit is the only degenerate row: `eta` and `lam` are both at their upper clips, `beta` is at the lower clip, and a 10% phase-0 perturbation changes prediction by `0`.
- Re-running the same family with better multistart settings avoids that exact degeneracy and improves 300M CV RMSE from `0.01048` to `0.00743`.
- The best repaired diagnostic row is `fast_moderate_clip_powell120`; the tiny-ridge row is nearly tied (`0.00744`) and has less raw-optimum concentration.
- Basin hopping did not beat the simpler Powell repair. It found a lower raw predicted optimum, but the raw optimum is a phase-0 corner (`raw_phase0_max_weight=0.999526`), so it is not the best deployment story.
- The conclusion is not "GRP cannot fit 300M." The conclusion is "unregularized GRP no-L2 has an unstable retained-exposure parameterization; moderate clips or tiny ridge make the fixed-scale regression fit much better, but raw optima remain too far off-manifold to trust without deployment constraints."
