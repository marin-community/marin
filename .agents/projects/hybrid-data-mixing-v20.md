# Hybrid Data Mixing v20

## Goal

Improve the hybrid data-mixing model

\[
\hat y(w, N, D \mid \mathcal O_w) = f(w, N, D) + u(w, N, D, \mathcal O_w)
\]

with emphasis on:

1. keeping direct-mode quality at least as strong as the chosen base law,
2. improving conditioned prediction without same-scale leakage,
3. preserving exact identity,
4. keeping candidate geometry as plausible as possible,
5. running a real hillclimb rather than one-off experiments.

## Current Baselines

- direct base:
  - `direct_scalar_grp`
- direct challenger:
  - `direct_enriched_grp_family_a`
- conditioned transfer reference:
  - `source_old10_scale_calibrated`

## Immediate Search Matrix

Backbones:

1. `direct_scalar_grp`
2. `direct_enriched_grp_family_a`

Update families:

1. `nw_single_bw`
2. `nw_pair_bw`
3. `nw_pair_bw_gain`
4. `lowrank_transport`

Minimum hillclimb budget:

- at least 20 seed-7 evaluated configs
- then 8-seed robustness on the best few

## Success Criteria For This Round

- no same-scale leakage in conditioned tracks
- exact identity remains true
- at least one hybrid candidate that is clearly better than the previous CC kernel baseline
- ideally a model that improves:
  - direct fallback via better backbone, and/or
  - conditioned `obs60_and_300` / `multi_obs` metrics without pathological robustness failures

## Notes

- Report both:
  - full-holdout end-to-end metrics
  - applicable-subset-only conditioned metrics
- Treat fixed `520M` as a stress test, not the only objective.
- Candidate plausibility remains a gate, not an afterthought.
