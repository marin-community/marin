# Report: Delphi phase-1 branch continuation modeling

**Date:** 2026-08-25
**Driver:** `experiments/domain_phase_mix/exploratory/two_phase_many/audit_delphi_phase1_branch_modeling_20260825.py`
(commit `76f5c37356`). Reproduces every frozen fact and audits the draft acquisition. No training job was
submitted.

## 1. Reproduction

All five artifact hashes match. 108 result rows, 100 fit rows, 1,944 long-form metric rows. The prefix
(`phase_0_*`) is **byte-identical across all 108 rows**, and the tied continuation equals the prefix mixture
at total variation exactly `0.00000000`, so the single-action-variable claim holds exactly.

| quantity | reproduced | handoff |
|---|---|---|
| tied | `0.98989356` | `0.98989356` |
| best fitted branch (`fit_wave1_extension_18`) | `1.00035429` | `1.00035429` |
| best minus tied | `+0.01046073` | `+0.01046073` |
| fit actions better than tied | **0 of 100** | every fit action worse |
| GitHub C++, best fit vs tied | `0.74710315` / `0.75805378` | same |
| TV from tied, min / median / max | `0.43359` / `0.58667` / `0.81006` | same |
| Wave-1A to 1B `radial7_sparse_sqrt` | rho `0.787179`, RMSE `0.010870` vs `0.018147`, regret `0.009777` / `0.005975` | same |
| full-panel best RMSE (`elasticnet_radial_sqrt`) | `0.011390` | same |

Two things do not reproduce as stated, and both weaken the case for a local optimum.

**The run-noise scale is softer than one number suggests.** `0.00084656` is the larger of two within-group
estimates, each from five repeats. The other group is `0.00033960` and the pooled within-group value is
`0.00064498`. The two differ by 2.5x, each has four degrees of freedom, and **neither group sits at tied** --
they are repeats of `control_proportional` and `fit_maximin_26`. Treat `0.00084656` as a conservative
placeholder with roughly a factor-of-two uncertainty, not a measured constant, and put repeats at tied in
Wave 2.

**The historical frontier's `0.98245525` is not an endpoint in this panel.** That number comes from the
historical two-phase run. The same continuation executed from *this* prefix is `control_incumbent_planned`,
which scores `0.99960303` -- worse than tied by `+0.00971`. Its distance from tied here is TV `0.49512`, not
`0.10200`; the `0.10200` figure is the historical run's own internal phase-0-to-phase-1 contrast, measured
against a different prefix. Every control in this panel lies between TV `0.40527` and `0.49512` from tied:

| control | TV from tied | Uncheatable |
|---|---|---|
| `control_unimax8` | `0.40527` | `1.00357556` |
| `control_proportional` | `0.47363` | `1.00645995` |
| `control_wave1a_anchor_fit_maximin_00` | `0.48633` | `1.00611699` |
| `control_incumbent_planned` | `0.49512` | `0.99960303` |

So the support gap is real and the handoff is right that nothing is measured inside TV `0.43`. But the
inference that "the useful optimum is likely local" does not follow from the `0.10200` comparison, because
that distance lives in a different geometry. What the panel actually supports is weaker and worth stating
plainly: **we have no evidence about the region inside TV 0.43, and every action we have measured -- including
the historical frontier's own continuation -- is worse than doing nothing.**

One correction to the handoff's own framing: in the Wave-1A-to-1B table, `even2_sparse_sqrt` beats the frozen
`radial7_sparse_sqrt` on both RMSE (`0.010761` vs `0.010870`) and top-1 regret (`0.000000` vs `0.009777`),
selecting the true best branch. The handoff is right to quarantine it as post hoc, but the claim that the
transferred critic "ranks grossly harmful actions but does not place the optimum" is specific to radial7, not
a property of the model class.

## 2. Is the anchored odd/even form the right minimal critic? (Q2)

The form is sound and the fixed prefix is what makes it so: with one checkpoint, optimizer state and data
seed, `dL` really is a function of `z` alone, which is exactly the confound that made the general two-phase
problem intractable. Enforcing `dL(0) = 0` is free and correct. Three attacks:

- **Antithetic leakage.** `o(r,d) = r beta'd + O(r^3)`. At the proposed radii the relative cubic leak is `r^2`,
  about `0.023` at `r = 0.15`, which is tolerable unless third derivatives are large. Cheap insurance: the two
  historical rays already carry three radii, so fit `o` against `r` and `r^3` there and check the cubic term.
- **Boundary clipping breaks the pairing.** If `-rd` clips at the simplex boundary while `+rd` does not, the
  pair is no longer antithetic and the clipping asymmetry lands in the even channel as fake curvature. The
  design sets `MINIMUM_PREFIX_WEIGHT_FOR_SPARSE_RAY = 0.005`, which suggests this was considered; it must be
  asserted per row, not assumed.
- **The null is the live hypothesis.** Zero of 100 measured actions beat tied, and the minimum over the whole
  measured set is `+0.0105`. A model whose only job is to find an improvement will be fitted on data that may
  contain none. The critic must be able to express `beta ~= 0, alpha > 0` -- tied is the local optimum -- and
  the acquisition must be powered to *confirm* that, not only to search against it.

## 3. Coordinate (Q3)

**Square-root / Hellinger**, with centered log-ratio as the fallback, chosen inside folds.

The evidence is in the existing probe and it is not close: every sqrt-coordinate model beats its raw
counterpart by a wide margin on the same panel -- `ridge_sqrt39` reaches Spearman `0.737070` against
`ridge_raw39` at `0.179554`, and the four best models on the full panel are all sqrt-coordinate. Hellinger is
also the radius the design already uses, so radii and coordinates agree. Two caveats: this is far-field
evidence and may not transfer to `r <= 0.23`, and CLR is undefined at zero weight, so it needs the same
minimum-weight floor the design already applies. Direct displacement is the one to drop -- a fixed TV step
means wildly different epoch changes across buckets spanning 359x in epochs per unit weight.

## 4. Is one isotropic curvature coefficient defensible? (Q4)

Defensible as a starting point, and the design can identify it. Do not free it further without a nested test.

The audit shows the even channel is well posed as proposed: over the 40 plus-sign fit rows the correlation
between squared radius and the label-blind repetition channel is `-0.1161` at condition number `3.92`. So
`alpha` and `gamma` are separable, which was the thing most likely to be broken and is not.

The smallest defensible extension is **curvature grouped by exposure stratum** -- three coefficients instead
of one, label-blind, identifiable because directions differ in how their mass sits across strata. That is
`+2` parameters against 40 even observations. Anything larger fails: a rank-one extension `alpha I + lambda
uu'` costs `1 + 38 + 1`, and a free Hessian's 741 coefficients are hopeless, as the handoff says. Pre-register
the three-stratum form as a single nested alternative and test it inside folds by likelihood ratio.

## 5. Does a repetition channel add value? (Q5)

Keep it, and correct the premise. This is **not** a low-repetition setting: across the proposed fit rows the
maximum total materialized epoch runs `9.09` to `15.88`. There is both substantial repetition and substantial
variation in it, and it is nearly orthogonal to radius (`-0.1161`). So the channel is neither degenerate nor
redundant with radial distance, and the design can estimate its coefficient.

Whether it is *predictive* cannot be known from Wave 1, which has no local data. What the audit establishes is
that the question is answerable by the proposed experiment, which is the relevant standard at design time.

## 6. Two heads or one? (Q6)

**Two, and the evidence is already decisive.** A broad radial critic fitted to all 100 rows predicts the
held-out tied control at `0.964255` against an observed `0.989894` -- an error of `-0.025639`, roughly 30 run
noises, in the exact place the decision gets made. No amount of broad CV quality repairs that.

Avoid an arbitrary gating threshold by giving the heads different *jobs* rather than different regions:

- the broad critic is a **feasibility filter** with a reject-only decision -- it never ranks candidates and
  never proposes an optimum;
- the anchored local critic **ranks**, and is only ever evaluated inside its measured support radius, i.e. at
  Hellinger no greater than the largest fitted radius (`0.23` under the draft).

Nothing needs to blend, because no candidate is ever scored by both for the same purpose. Any proposal outside
the local critic's support radius is out of scope by construction, not by threshold.

## 7. Does the draft panel identify the model? (Q7)

**It identifies the even channel and under-identifies the odd one.** Recommendation: reallocate to
**36 distinct directions, four of them at two radii, for 40 antithetic pairs and the same 80 fit rows.**

The draft buys 28 distinct fit directions -- 18 dense geometry, 8 sparse geometry, 2 historical -- spanning
**rank 28 of the 38-dimensional tangent space, leaving 10 dimensions unmeasured**, with radius replication on
only 10 directions (18 at one radius, 8 at two, 2 at three). Curvature is fine, as section 4 shows. Direction
is not. Simulating sparse recovery from 40 antithetic pairs at the measured odd-channel noise of `0.000599`,
mean support recall is:

| allocation | directions | s=3, SNR 2 | s=3, SNR 5 | s=6, SNR 2 | s=6, SNR 5 | s=10, SNR 2 | s=10, SNR 5 |
|---|---|---|---|---|---|---|---|
| draft | 28 | 0.09 | 0.34 | 0.09 | 0.23 | 0.13 | 0.18 |
| alternative | 32 | 0.10 | 0.50 | 0.10 | 0.26 | 0.18 | 0.28 |
| **alternative** | **36** | **0.16** | **0.70** | **0.15** | **0.35** | 0.10 | 0.27 |
| alternative | 40 | 0.23 | 0.64 | 0.11 | 0.45 | 0.06 | 0.34 |

SNR is on the observed contrast, so it already includes the radius; for scale, a linear extrapolation of the
far panel's response to `r = 0.15` gives roughly seven noise units, so SNR 5 is near the optimistic end.
**The draft is worst of the four at every cell.** Thirty-six directions with four double-radius rays keeps
enough replication to pin `alpha` and `gamma` -- which needed only that the repetition channel not be
collinear with `r^2`, and it is not -- while recovering most of the span.

The honest caveat is that **no allocation reaches useful recall**. Even the best is `0.70` at the most
favourable sparsity and an optimistic SNR, and falls under `0.35` once the true support exceeds three. Wave 2
should therefore be planned as a **screening and null-confirmation wave**, not as a wave that estimates
`beta`. If the real goal is to decide whether any continuation beats tied, more replication at fewer
well-chosen directions would buy more than a wider span.

Keep the historical ray labelled separately in every gate. It is outcome-selected from prior work, and as
section 1 shows, its continuation from this prefix is already known to be worse than tied -- so it is a
powered scientific probe, not discovery evidence.

## 8. Frozen registry and protocol (Q8)

Freeze this registry before any Wave-2 outcome exists, with the coordinate fixed to sqrt/Hellinger:

| id | form | parameters |
|---|---|---|
| `null_tied` | `dL(z) = 0` | 0 |
| `anchored_iso` | `beta'z + alpha||z||^2` | sparse beta + 1 |
| `anchored_iso_rep` | `+ gamma R_rep` | + 1 |
| `anchored_stratum_rep` | curvature per exposure stratum | + 2 |
| `even_odd_posthoc` | the post-Wave-1B even/odd critic | as specified, frozen now |

`null_tied` is the baseline every other candidate must beat; given 0 of 100 measured actions beat tied, it is
the favourite and should be treated as such. Protocol: direction-level folds with all signs and radii of one
line in the same fold; coordinate, sparsity and curvature form selected inside training folds only; the four
sealed referee directions untouched by selection; tied repeats and confirmation runs outside the fit budget.

Primary gate: a candidate is promotable only if the action it selects beats tied on fresh confirmation runs by
more than twice the run-noise scale, with the scale re-estimated from tied repeats rather than from
`0.00084656`. Ranking the broad harmful panel earns nothing.

## 9. What should be frozen next, and is anything promotable? (Q9)

**Nothing is promotable, and the honest reading is that the null is currently winning.** Zero of 100 fitted
actions beat tied; every operator control is worse than tied; the historical frontier's continuation run from
this prefix is worse than tied; and the one critic that places the optimum was formulated after seeing the
outcomes it places. The single defensible claim from Wave 1 is that gross harm far from tied is rankable, and
that claim does not support a branch optimizer.

Concrete recommendation: freeze the registry in section 8 and run the 80-row local acquisition **reallocated to
36 directions with four double-radius rays**, explicitly as a screening wave whose pre-registered primary
question is "does `dL` have any negative direction at `r <= 0.23`, or is tied locally optimal?" Add tied
repeats inside Wave 2 so the noise scale is measured where the decision is made. Do not plan a promotion claim
from this wave.

## Post-hoc choices in this report

- The four alternative allocations, the sparsity grid and the SNR grid were chosen by me after seeing Wave 1;
  they are design analysis, not confirmation.
- Alternative allocations use random tangent directions as a stand-in, not directions generated by the actual
  maximin construction; only the 28-direction row uses the real design.
- Support in the recovery simulation is drawn uniformly in the tangent space, which is friendlier to the draft
  than bucket-sparse truth would be.
- The repetition channel uses `max_total_materialized_epoch` from the design's own summary as a label-blind
  proxy; the report does not commit to that being the right `R_rep`.
