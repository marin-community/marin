> **VOID 2026-08-10.** Beyond the supersession noted below, every number in this document was
> produced by a solver that projected the response onto a basis containing an arbitrary direction
> outside the free block's column space (rank-deficient reduced QR). Predictions moved by RMS 0.090
> BPB under family relabelling, against gates of 0.008. Fixed in `general_mixture_surrogate_20260809.py`
> via `column_space`. Nothing here is quotable; see the corrected verdict.
>
> **SUPERSEDED 2026-08-10.** The numbers below were computed before three defects were found and fixed:
> a tally that merged two model configurations, an optimiser that landed in different basins under
> structurally-null changes, and a late-share coefficient read as unstable when only its contrast was
> identified. The authoritative result is now in
> `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/general_surrogate_round_verdict_20260810.md`:
> **GEN-002 scores 40/44 over 11 seeds in a single stamped configuration** — Regret@1 11/11, RMSE 10/11,
> gain error 10/11, optimum distance 9/11 — and **no model is promoted**. The text below is kept as the
> record of what was claimed at the time.

# Follow-up: bucket-general surrogate (GEN-001, GEN-002)

Supersedes the promotion case in `two_phase_surrogate_promotion_case_20260809.md`, which was **withdrawn**
after your audit. All four of your findings verified and were accepted; see `WSD80-SUR-116`.

## What changed and why

Your audit's positive half identified three load-bearing ingredients. A separate constraint from the
experimenters then made the decisive difference: **in production, buckets are not semantically labelled.**
They arrive classified by topic, with quality splits inside each topic, and nothing says which bucket the
eval is about. The audited model hand-assigned three of its five structures using knowledge that one
domain was code and the other off-target text, so it could not be deployed as written.

Rebuilding without any semantic assignment did not merely preserve performance. **It fixed the gate that
had blocked the project throughout.**

## GEN-001: remove every semantic assignment

Every structure becomes a sum over all buckets. Nonlinear parameters pool at the family (topic) level;
per-bucket freedom enters only as shrunk amplitude departures, so a quality label groups but is never a
feature. The signed late control, which had read the off-target domain's late share, becomes a
**free-signed late share per family** — spanning the original exactly on two domains, requiring no
designated eval domain on thirty-nine.

**WSD80, 11 seeds: 42/44**, against the semantic model's 35/44 on the same eleven.

| gate | general | semantic |
|---|---|---|
| interior OOF RMSE | 11/11 | 11/11 |
| Regret@1 | 10/11 | 10/11 |
| **optimum distance** | **10/11** (0.027–0.109) | **3/11** |
| gain error (full-data diagnostic, not a gate) | 11/11 | 11/11 |

**The cause is identified and it was my error.** I had hand-assigned the early-boundary kernel to the
CODE bucket. Given freedom to place it, the model puts it on BROAD text — slot 0 scale fits 0.18–1.10
against an early-epoch range of 0–0.797, fully active — and switches the code one OFF, slot 1 pinned at
its bound 316.2 where `exp(-21.089/316) = 0.935` is near-constant. Mechanically sensible: the failure mode
is starving a domain's early phase; broad text is the domain that *can* be starved early and recovered
late cheaply; code at 21.089 epochs/unit has its early share pinned by damage regardless.

Every structural fix attempted before this — saturating damage, gate placement, multi-start, the
resolution-limit argument — was compensating for a misplacement I had introduced.

**300M Uncheatable, 3 seeds:** all-row RMSE 0.006306/0.006484/0.006214, beating HPR's 0.006800 outright;
predicted mean pair gain correctly negative at −0.000397/−0.000854/−0.000473 against observed −0.001086.
Regret@1 fails at 0.005816.

## GEN-002: pool geometry by exposure, taste by topic

A prediction I made **failed**, and produced the next step. The corrected boundary-risk metric (total
epochs per unit weight) does not predict which family gets an active kernel: ranked by pool-exhaustion
speed the 300M families run 2 > 1 > 0; ranked by kernel activity they run 1 > 0 > 2.

The reason is a grouping mismatch. **Topic 0's 31 buckets span 4.80 to 1723.89 epochs per unit weight — a
359-fold range across all three exposure strata** — so one boundary scale served geometry it could not
possibly fit; topic 2 had only two buckets to fit a scale from.

So parameters are now pooled by whatever determines them. **Taste** (readout exponent, amplitudes — how
much a topic helps this eval) stays on topic. **Geometry** (boundary scale — how fast a bucket exhausts
its pool) moves to exposure stratum, cut on the observed log-epoch distribution. Neither needs semantics.

**Prediction confirmed, at no fit cost:**

| | boundary scales across seeds | worst spread |
|---|---|---|
| by topic | [28.1, 7.5, 29.9] · [32.3, 6.1, **87.7**] · [27.4, 6.7, **0.73**] | **120×** |
| by stratum | [5.9, 7.9, 24.5] · [6.4, 10.1, 26.1] · [16.1, 13.7, 22.5] | **2.7×** |

RMSE 0.006222–0.006550 against 0.006214–0.006484, both beating HPR. Pair gain stays negative, seed 1 at
−0.001047 against observed −0.001086.

This also retroactively explains parameters I had concluded were intrinsically unidentifiable: at least
on 300M, I was fitting one parameter to buckets that do not share the property it describes.

## Two corrections found after the results above were computed

**Optimizer path-dependence (`GEN-007`).** On WSD80 the objective is multi-basin and the optimiser's path
decides which optimum is found. Two configurations that are STRUCTURALLY IDENTICAL on this panel — two
buckets with two distinct exposure rates give each bucket its own boundary scale either way, the extra
stratum's parameter being simply unused — produce seed 3 at 4/4 with distance 0.033541 and optimum
(0.070, 0.485) in one case and 2/4 with distance 0.092500 at (0.013, 0.530) in the other. The only
difference is search dimension, 10 parameters against 9.

**Consequence: every single-run tally in this document is provisional**, including the 42/44. Multi-start
over each boundary scale's full range, selecting on the inner-fold objective and never on a gate, is now
applied and all eleven seeds are rerunning. If results still move under null reparameterisations
afterwards, the basins are genuinely competitive in CV and **the model's optimum is not determined by the
data** — which would be more serious than any gate failure, since a surrogate whose recommended mixture
depends on optimiser seed cannot be used to pick a mixture.

**A merged tally, withdrawn (`GEN-006`).** A 43/44 figure with distance 11/11 was computed for the
stratified model and is withdrawn: it combined six seeds run before an empty-stratum fix with five run
after. Same defect as your finding #1, committed within an hour of my writing the rule against it into the
registry. Every result line now carries a configuration stamp so merging incompatible runs is visible
rather than silent.

## Open, stated plainly

1. **300M Regret@1 is not a valid comparison** (found after drafting, `GEN-004`). It is exactly 0.005816
   for every variant because every model picks row 405, the second-best TIED policy — sensible, since the
   panel's truth is that asymmetric policies do not help. The observed best, row 480, is an isolated
   outlier sitting 0.004334 clear of its nearest rival against a local spacing of 0.000222. Decisively:
   only **one** row of 520 lies within HPR's published regret of 0.002678, namely row 480 itself, so that
   reference is **unachievable on this row set under this definition** — any model scoring ≤0.002678 must
   have picked row 480 and would score 0.000000. Every 300M Regret@1 "fail" in this project is therefore
   uninterpretable rather than a genuine shortfall, and the reference's provenance needs establishing.
2. **The late-share "sign instability" was my error** (`GEN-003`). On a simplex the family shares sum to
   one, so only their CONTRAST is identified and the common level is absorbed by the intercept. I read the
   wandering common mode as instability and reported it that way; your audit reasonably endorsed the
   nuisance-control reading on the strength of my description. The contrast is stable: −0.314, −0.348,
   −0.357, −0.337, −0.319, −0.321, spread 0.043 — tightened 3.6× by stratification. The older single
   `w1_broad` column had a *different* defect, R²=0.75 collinearity with the late-broad benefit column,
   which is why its sign moved with the column set. The contrast form avoids both. This upgrades the term
   from nuisance control to a stable directional statement; it still is not evidence of gradient conflict.
3. **Topic 0's readout exponent is pinned** at its lower bound 0.005 on all three 300M seeds.
4. One WSD80 seed (8 of 11) still collapses to early share exactly 0.000.
5. The WSD80 null check for GEN-002 is running; its two buckets fall in separate strata so results should
   be UNCHANGED, and movement would mean the change did something unintended.
6. Seed counts are 3 on 300M and 11 on WSD80.

## What I would most like checked

- Is the 42/44 real, or is the distance improvement exploiting the same argmin noise your audit says my
  47% ceiling mis-measured? The failure and pass rates flipped almost exactly (3/11 → 10/11), which is a
  large swing to attribute to one feature's placement.
- Is the taste/geometry pooling split principled, or am I fitting the grouping to the panel?
- 300M Regret@1: nothing has moved it. Is that a model limitation or a property of that panel's argmin?
