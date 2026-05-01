# Debugging log for non-60M scale calibration

Investigate why the current direct laws systematically miscalibrate at non-`60M` target scales, especially `130M` optimism and `520M` pessimism.

## Initial status

User-observed issue:
- mixed-scale predicted-vs-actual plots are visibly off-diagonal
- the by-scale view shows systematic bias outside the central scale band

Known current evidence:
- `130M` targets are predicted too optimistically
- `520M` targets are predicted too pessimistically
- `300M` targets are closer to centered but still under-dispersed

## Hypothesis 1

The main miss is not "bad mixture fit" but a scale-calibration error caused by the direct-law fitting setup itself.

## Changes to make

- No code changes yet.
- Gather the existing scale-split residuals and calibration summaries.
- Trace how the direct-law fit parameterization, feature standardization, and fold selection interact with sparse `520M` supervision.

## Future Work

- [ ] Check whether the benchmark split is skewing target-scale calibration
- [ ] Check whether feature standardization is compressing the scale response
- [ ] Check whether the floor-log fit is introducing under-dispersion
- [ ] Check whether frozen GRP scale couplings dominate the miss
- [ ] Check whether the issue persists for the newer direct backbones

## Results

- Existing artifact check confirms the visible pattern:
  - `130M` is predicted too optimistically
  - `520M` is predicted too pessimistically
  - `300M` is less biased on average but still under-dispersed
- Direct benchmark train-set composition is highly unbalanced:
  - `60M`: `293` primary rows
  - `130M`: `39` primary rows total (`34` in train)
  - `300M`: `268` primary rows total (`240` in train)
  - `520M`: `4` primary rows total, all used as fixed holdout, so `0` in train
- This means:
  - `520M` direct prediction is pure extrapolation
  - `130M` is strongly underrepresented relative to `60M` and `300M`

## Hypothesis 2

The scale miss is mostly a benchmark split artifact caused by scale imbalance and missing `520M` train supervision.

## Changes to make

- Compare training-set and holdout-set by-scale calibration for:
  - `direct_scalar_grp`
  - `direct_drop_uNuD`
  - `direct_shared_score_tilt_poly4`
- Check whether scale-balanced weighting materially changes the by-scale bias.

## Results

- The same scale-specific bias is already present on the training rows.
- Seed-7 training calibration for the current direct models:
  - `direct_scalar_grp`
    - `60M`: bias `-0.020`, slope `0.366`, std-ratio `0.533`
    - `130M`: bias `-0.043`, slope `0.561`, std-ratio `0.756`
    - `300M`: bias `-0.009`, slope `0.323`, std-ratio `0.525`
  - `direct_drop_uNuD`
    - `60M`: bias `-0.008`, slope `0.386`, std-ratio `0.563`
    - `130M`: bias `-0.035`, slope `0.642`, std-ratio `0.841`
    - `300M`: bias `-0.003`, slope `0.342`, std-ratio `0.547`
  - `direct_shared_score_tilt_poly4`
    - `60M`: bias `-0.001`, slope `0.387`, std-ratio `0.570`
    - `130M`: bias `-0.029`, slope `0.661`, std-ratio `0.885`
    - `300M`: bias `+0.000`, slope `0.366`, std-ratio `0.581`
- This is the key debugging result:
  - the non-`60M` problem is **not** only a holdout-generalization issue
  - the direct laws are already under-dispersed within scale on the rows they fit
- Scale-balanced weighting helps the underrepresented `130M` scale a lot:
  - seed-7 holdout `direct_scalar_grp_scale_balanced`
    - `130M` bias improves from `-0.041` to `-0.006`
    - `130M` RMSE improves from `0.0419` to `0.0070`
  - but `520M` stays pessimistic:
    - bias only changes from `+0.037` to `+0.035`
- Interpretation:
  - scale imbalance is a real contributor, especially for `130M`
  - but it is not the whole story, because within-scale under-dispersion already
    appears on the train set
  - `520M` remains difficult because there are no direct train rows at that scale

## Hypothesis 3

The dominant residual cause is structural underfitting from the direct-law family: floor-log fitting plus strong shrinkage and a too-rigid scale/mixture interaction.

## Changes to make

- Compare newer direct families and previously tested variants.
- Check whether unfreezing GRP basis constants or adding scale-residual variants fixes the problem.

## Results

- The issue persists across direct families:
  - `direct_scalar_grp`
  - `direct_drop_uNuD`
  - `direct_shared_score_tilt_poly4`
  all show the same overall pattern
- Partial/full unfreezing of the GRP basis did not fix high-scale robustness.
- A scale-residual variant did not repair the calibration cleanly and often worsened
  `520M` behavior.
- Therefore the main root cause is:
  1. no trustworthy `520M` direct supervision in train
  2. severe scale imbalance (`130M` especially)
  3. direct-law under-dispersion already on train, which points to model-shape
     rigidity / shrinkage rather than just split bad luck
