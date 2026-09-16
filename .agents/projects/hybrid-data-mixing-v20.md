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

## Low-Rank Scaling-Law TODO

Direct-backbone follow-up to test whether mixture effects are mostly a low-rank
modulation of a shared scale curve:

1. `shared_curve_only`
   - `L = floor + C_N N^-alpha + C_D D^-beta`
2. `parallel_rho`
   - `L = floor + rho(w) * (C_N N^-alpha + C_D D^-beta)`
3. `offset_plus_rho`
   - `L = floor + delta(w) + rho(w) * (C_N N^-alpha + C_D D^-beta)`
4. `split_rho_nd`
   - `L = floor + rho_N(w) C_N N^-alpha + rho_D(w) C_D D^-beta`
5. `scale_floor_plus_rho`
   - shared scale-dependent floor plus `rho(w)` on the reducible term

Current status after the first sweep:

- none beat corrected `direct_scalar_grp`
- `shared_curve_only` is the strongest low-rank baseline
- `offset_plus_rho` is the least-bad GRP-lite extension
- full `rho(w)` models are collapsing toward `baseline_stratified`, which means
  the current `rho` features or the multiplicative-only assumption are too weak
  to recover the observed broad-heavy winners

Current status after the additive/factorized follow-up:

- `family_factorized_scale_grp_lite` is the best of the newly added structures
  on mean overall RMSE/Spearman, but still loses to `direct_scalar_grp`
- `offset_only_grp_lite` is better than the original `offset_rho_grp_lite`
- `offset_weak_rho_grp_lite` does not improve on `offset_only_grp_lite`
- none of the new forms fix the `520M` stress-test weakness

Current status after the shared-curve + structured-GRP sweep:

- `warp_plus_delta_grp_lite` is the best new overall challenger from the latest batch,
  but it still loses materially to `direct_scalar_grp`
- `family_factorized_scale_grp_lite` remains the strongest low-rank “clean” model
- `offset_only_full_grp` is a clear negative result; full GRP additive residuals on
  top of the shared curve are unstable and over-optimistic
- the best-scoring new models still optimize to either:
  - `baseline_stratified`, or
  - a degenerate tech-only corner
- so the current issue is not just lack of mixture features; it is the way
  mixture-specific scale behavior is being expressed

Current status after the targeted refinement pass:

- `parallel_rho_full_grp` and `offset_rho_full_grp` are both strong negative
  results:
  - putting a full GRP mixture term on top of the shared curve does **not**
    rescue the backbone
  - both collapse to the exact `baseline_stratified` interior mixture
  - both are badly over-optimistic at larger scales
- `warp_family_scores` and `warp_plus_delta_family_scores` confirm that the
  warp idea is real numerically but still unsafe geometrically:
  - `warp_plus_delta_family_scores` reaches mean overall `RMSE 0.03188` and
    mean overall `Spearman 0.7635`, which is close to `direct_scalar_grp`
  - but both family-score warp variants still collapse to a near one-hot
    tech-only corner and fail the trustworthy `520M` stress test
- working conclusion:
  - elegant shared-curve backbones still look conceptually right
  - but the current ways of attaching mixture-specific scale behavior to them
    are not yet faithful enough to recover the broad-heavy trustworthy winners

Current status after the direct-scalar ablation pass:

- `direct_scalar_grp` is not winning because the mixture body alone is strong
  enough; it is winning because the explicit 5-term scale head is carrying a
  large fraction of the predictive signal
- removing the explicit scale head is catastrophic
- removing the internal GRP scale shifts is also clearly harmful, especially on
  fixed `520M`
- replacing the explicit 5-term head with shared-curve terms is not yet enough:
  - `direct_shared_curve_basis` keeps some fixed-`520M` rank
  - but loses badly overall
- `shared_curve_only` remains the clean elegant baseline, but underfits in a
  way that the current ad hoc scale head is patching over
- next elegant-direction question is not “can we delete the scale head?”; it is
  “can we build a richer shared-curve backbone that matches the explicit scale
  capacity of the current 5-term head while keeping the theory cleaner?”

Current status after the direct-scalar design pass:

- there is a useful new simplification signal inside the incumbent itself
- best simplification candidate so far:
  - `drop_uNuD`
  - slightly better corrected 4-seed overall metrics than `direct_scalar_grp`
  - fixed `520M` stress test unchanged
  - candidate geometry remains close to baseline
- second simplification candidate:
  - `drop_uD2`
  - nearly neutral overall, slightly better fixed-`520M` RMSE
  - candidate geometry also remains close to baseline
- false friend:
  - `signals + penalties + scale`
  - very compact and numerically strong, but collapses to an implausible
    broad-text corner
- updated direct-law simplification takeaway:
  - pruning the current scale head looks more promising than replacing it with
    shared-curve-only elements

Current status after promoting the pruning variants:

- `direct_drop_uNuD` is now the strongest simplification candidate and should
  be treated as a live incumbent challenger, not just a one-off note
- current corrected ordering:
  1. `direct_drop_uNuD`
  2. `direct_scalar_grp`
  3. `direct_drop_uD2`
- `direct_drop_uNuD` appears to keep the same qualitative candidate geometry as
  the incumbent while slightly improving corrected 4-seed overall metrics
- if we hand work back out, this variant should be in the packet and benchmark
  narrative
