# WSD80 mechanistic surrogate

Series prefix `WSD80-SUR`. Fieldbook experiment `exp_01kymbv4xg2k6yanx59zfjjg1g`.

Canonical north star:
[Two-phase mixture surrogate charter](../projects/two_phase_surrogate_north_star.md).
The charter defines the invariant objective and evidence gates; this logbook
records mutable hypotheses, implementations, and results.

## Problem

Build a simple, mechanistically defensible surrogate whose optimized two-phase
policy validates better than the strongest existing one-phase and two-phase
baselines. The immediate identification problem is the high-TPP 300M /
6B-token, 39-bucket setting. Model discovery may use the original 280-row
two-phase panel plus its aggregate/tied counterparts; recovering the result
under a fixed 280-total-row acquisition budget is a later sample-efficiency
ablation.

The 80/20 WSD StarCoder panel matters because it is the one panel in the project where a two-phase
policy demonstrably beats the entire one-phase class. Every incumbent surrogate was developed against
panels where that does not happen, so none has ever been asked to represent a real two-phase optimum.

## Success metrics

The evidence hierarchy is fixed before subsequent model selection.

1. **WSD80 geometry gate.** Compare against the repaired incumbent baselines on
   the full 346-coordinate surface. Report blocked-region and random-fold RMSE
   in training-seed sigma (`sigma = 0.004633 BPB`), tied-row RMSE, raw-optimum
   location against `(0.10, 0.50)`, net two-phase advantage against `0.009594`
   BPB, and fixed-fiber gain and ordering across all measured aggregates.
2. **Mandatory high-TPP 39-bucket gate.** Use the 300M / 6B-token original
   two-phase panel plus qsplit240 exposure-average ablation: 520 rows, 282
   physically tied policies, 238 asymmetric policies, and 238 exact
   aggregate-matched pairs. On both Uncheatable and Table-9 report grouped OOF
   RMSE and rank, asymmetric-policy RMSE, paired-delta RMSE and rank, paired
   bias, sign accuracy, and fold regret. A phase mechanism must improve or
   preserve both targets, not merely pooled absolute BPB.
3. **Optimization gate.** Audit the raw optimum for support distance, maximum
   bucket weight and epochs, phase divergence, and bootstrap stability. Repeat
   the comparison under the fixed 280-row acquisition budget before claiming
   a sample-efficiency improvement.
4. **Secondary transfer diagnostics.** Delphi 3e18 and 60M remain useful for
   scale-transfer and pathology checks, but they do not select the phase
   mechanism. Delphi's total/non-embedding TPP is only 4.40/12.27 versus
   29.83/58.45 at 300M / 6B, so weak Delphi phase fit cannot excuse failure on
   the high-TPP panel. TPP is a plausible moderator, not an established causal
   explanation.

## Context carried in

Two results from the fiber analysis (`exp_01kx82syex2d345hpz87w34jye`, notes
`note_01kym5qr53jc9h984pzxngq6bg` and `note_01kym5ve22tpj58b3wsh8gmxf3`) set the design constraints.

The phase gain closes to zero exactly at the one-phase optimum. The linear ordering coefficient kappa
is indistinguishable from zero at aggregate 0.30 and around -0.15 at 0.18, so a surrogate that gets the
phase channel right must make its ordering term vanish where the aggregate response is stationary,
rather than carrying a free phase term everywhere.

And any model in which the schedule enters only through a per-domain reweighted cumulative exposure
predicts exactly zero two-phase gain. Writing `E = beta0*Phi0*w0 + beta1*Phi1*w1` with any positive
diagonal leverage matrices, a tied policy reaches any target `E` by setting `w_i` proportional to
`E_i/m_i`, so the one-phase class already sweeps the whole effective-exposure simplex. This rules out
the entire effective-exposure family as a source of two-phase gain, which covers most of the incumbent
zoo. Getting a two-phase gain requires genuine interaction between what was seen when.

## Hypothesis queue

- **H1 (promoted, evidence below).** The incumbent failure on this panel is primarily in the aggregate
  channel, not the phase channel: exponential saturation `1 - exp(-rho*e)` cannot reproduce a
  power-law data-scaling response, so the fitted one-phase curve misplaces the best constant mixture.
- **H2 (promoted, evidence below).** Replacing exponential saturation with an offset power law
  `A*(e + E0)^-a` plus a power-law repetition-damage term `B*e^g` fixes the aggregate channel.
- **H3 (open).** With the aggregate channel fixed, a phase term is still required, and it must be an
  interaction rather than a reweighting. Candidate mechanisms, from the fiber analysis: interference
  and forgetting, within-window repetition, consolidation during the anneal, state-dependent
  plasticity. Untested against each other.
- **H4 (open).** The phase term should be constructed so its linear part is proportional to the
  per-phase marginal value, which makes it vanish automatically at the aggregate optimum. This
  reproduces the measured `kappa(a*) = 0` without fitting it.
- **H5 (open, risk).** A form good enough for two buckets may be unidentifiable at 39. Every prior
  attempt at a bucket-resolution ordering field was judged unidentifiable and pooled to at most three
  families.

## WSD80-SUR-001: incumbent baseline on the 80/20 WSD surface

Panel loader `starcoder_wsd80_panel_20260728.py`; benchmark
`benchmark_wsd80_incumbents_20260728.py`. 166 unique coordinates at reference seed 20260711, merged
surface including both measured fibers, 63 of 63 reference-seed fiber coordinates present. Training-
seed sigma 0.004633 BPB from 63 replicated coordinates over five seeds.

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py
```

Ground truth on the panel: best one-phase `p = 0.300` at 0.945062; best two-phase `(0.100, 0.500)` at
0.935468, aggregate 0.180; two-phase advantage +0.009594 BPB, 2.1 sigma. Measured phase gain 0.000000
at aggregate 0.30 and +0.019855 at 0.18.

| model | CV RMSE | sigma | Spearman | tied RMSE | sigma | predicted tied optimum | predicted optimum | gain at 0.30 | gain at 0.18 |
|---|---|---|---|---|---|---|---|---|---|
| effective_exposure | 0.0615 | 13.28 | 0.772 | 0.0410 | 8.85 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| effective_exposure_geometry | 0.0587 | 12.68 | 0.752 | 0.0405 | 8.74 | 0.410 | (0.410, 0.410) | +0.0009 | +0.0165 |
| canonical | 0.0615 | 13.28 | 0.772 | 0.0410 | 8.85 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| separate_heads | 0.0745 | 16.07 | 0.900 | 0.1409 | 30.42 | 0.250 | (0.080, 0.360) | +0.0071 | +0.0422 |

Interpretation. Every incumbent misses the best constant mixture, by 0.10 in aggregate for the DSP
family. Tied-row RMSE is 8.7 to 8.9 sigma for DSP and 30 sigma for separate heads, so the one-phase
channel alone is already far outside noise. `effective_exposure_geometry` places its optimum at exactly
zero contrast, which is what the effective-exposure argument above predicts. `separate_heads` is the
only incumbent whose optimum is off the diagonal at a low aggregate, and it has the worst tied fit, so
among the incumbents mechanism and accuracy are anti-correlated.

`effective_exposure_geometry` reproduces the phase-gain profile reasonably (+0.0009 against 0.0000, and
+0.0165 against +0.0199) while still placing its optimum on the diagonal at aggregate 0.410. The
aggregate error dominates the argmin.

## WSD80-SUR-002: is the aggregate channel learnable at all?

Fit forms to the 19 tied rows only, unconstrained by the rest of the surface, to separate "wrong shape"
from "hard data". Chebyshev polynomials are included as an unconstrained reference: if a smooth
polynomial cannot do it either, the curve is genuinely hard.

| form | params | RMSE | sigma | argmin |
|---|---|---|---|---|
| offset power law both domains + code repetition damage | 9 | 0.00288 | 0.62 | 0.268 |
| log benefit both domains + damage | 7 | 0.00321 | 0.69 | 0.268 |
| offset power law both domains, no damage | 7 | 0.00644 | 1.39 | 0.244 |
| Chebyshev degree 8 | 9 | 0.01447 | 3.12 | 0.321 |
| DSP-shaped, exponential saturation + soft hinge | 7 | 0.01606 | 3.47 | 0.164 |
| Chebyshev degree 6 | 7 | 0.03692 | 7.97 | 0.147 |
| incumbent DSP fitted on the full surface | - | 0.0410 | 8.85 | 0.400 |

Measured tied optimum `p = 0.300`, value 0.945062. The log-benefit form's own minimum is 0.9451.

Interpretation. The one-phase response is learnable to under one sigma with seven parameters, so the
data is not the problem and the incumbent's 8.85 sigma is a shape failure. The discriminating
ingredient is the exposure response: an offset power law `A*(e + E0)^-a` fits, exponential saturation
`1 - exp(-rho*e)` does not, and a free polynomial of the same order does worse than either, so this is
about having the right heavy tail rather than about flexibility. That matches the scaling-law
literature, where loss falls as a power of data rather than saturating exponentially.

Caveat on identifiability. With two buckets and a fixed budget, a domain's token count and its epoch
count are proportional, so a volume term and a repetition term are the same variable within this panel.
The nine-parameter form's split between `A*(e+E0)^-a` and `B*e^g` is therefore not identified here and
should not be interpreted mechanistically until it is fitted somewhere the two can be separated.

**Erratum risk to watch.** Reported argmin 0.268 against a measured 0.300 is inside the sampling
resolution of the tied diagonal near its minimum, which is only sampled at 0.2405, 0.25, 0.30 and 0.40.
Do not treat 0.268 versus 0.300 as a residual model error until the diagonal is denser.

## Next

Use the heterogeneous 300M design structurally rather than fitting all 520
absolute BPB values as exchangeable rows:

1. identify the aggregate response `A(a)` from the 282 physically tied rows;
2. identify a phase residual `G(a, delta)`, with `G(a, 0) = 0`, from the 238
   exact aggregate-matched deltas;
3. combine them only after both channels clear their own grouped OOF gates;
4. test WSD80 geometry and both 300M targets before any secondary scale audit;
5. rerun the surviving procedure under a fixed 280-row tied/asymmetric budget.

Reuse retained-power-law state and response components where justified, but do
not add another divergence or free residual feature. The next material question
is whether structured estimation and partial pooling can recover the high-SNR
phase channel more efficiently than the existing joint RPL fit.

## WSD80-SUR-003: the model

`retained_power_law_model_20260728.py`. Three terms, one per mechanism:

```
L = b + sum_i A_i * (S_i + E0)^-a       benefit, power law in retained token share
      + sum_i B_i * max(D_i - T, 0)^g   damage, from re-reading a finite pool
      + J * concentration               Jensen gap of within-window intensity
```

`S_i = exp(-lambda*(1 - w1_i)) * alpha0*w0_i + alpha1*w1_i` is retained token share; `D_i` is raw
epochs. Amplitudes are nonnegative with a free intercept via bounded least squares; the five shape
parameters and the ridge are selected together on a discrete grid by out-of-fold error, the same
protocol the incumbents get.

Three design decisions each cost a round to find.

*Token share, not epochs, in the benefit term.* A first version used epochs and reached 33.6 sigma
cross-validated against the incumbent's 13.3. With two buckets whose pools differ by 26x, one shared
offset cannot serve both, and the fitted offset collapsed to its lower bound and swung on held-out
points near zero exposure. Switching the benefit argument to token share, which is on the same scale
for every domain, took it to 10.3 sigma. Scaling the offset by the proportional share instead was tried
and was worse, 33.4 sigma, for the same reason.

*Discrete shape grid, not continuous search.* Continuous L-BFGS over the shape parameters fitted the
training folds better and cross-validated three times worse. The grid is doing real regularisation.

*Log-deficit link rejected.* It was 56 sigma against 10 for the identity link on this panel, the
opposite of its effect on the 39-bucket panels.

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py
```

| model | CV RMSE | sigma | Spearman | tied sigma | tied opt | optimum | gain 0.30 | gain 0.18 |
|---|---|---|---|---|---|---|---|---|
| effective_exposure | 0.0615 | 13.28 | 0.772 | 8.85 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| effective_exposure_geometry | 0.0587 | 12.68 | 0.752 | 8.74 | 0.410 | (0.410, 0.410) | +0.0009 | +0.0165 |
| canonical | 0.0615 | 13.28 | 0.772 | 8.85 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| separate_heads | 0.0745 | 16.07 | 0.900 | 30.42 | 0.250 | (0.080, 0.360) | +0.0071 | +0.0422 |
| **retained_power_law** | **0.0390** | **8.41** | 0.874 | 12.38 | 0.280 | (0.180, 0.485) | +0.0036 | +0.0146 |

Truth: tied optimum 0.300, optimum (0.100, 0.500) at aggregate 0.180, gain 0.0000 at aggregate 0.30 and
+0.0199 at 0.18.

Cross-validated RMSE improves 34 percent on the best incumbent. It is the only model that combines a
small phase gain at the aggregate optimum with a phase gain near the measured value below it, and its
optimum is the closest to the measured one among the models that also fit the surface well.

Two honest weaknesses. Tied-row RMSE is 12.38 sigma, worse than the DSP family's 8.85, so the overall
gain is bought on the off-diagonal rows; the predicted tied optimum is nonetheless much better, 0.280
against 0.400. And the concentration coefficient fits to zero on every panel tried, so the Jensen term
currently earns nothing and the phase channel is carried entirely by the retention gate. Ablation:
setting retention to zero and dropping concentration moves the tied fit to 33.7 sigma and the optimum
to a degenerate corner, so the interaction channel is load-bearing.

## WSD80-SUR-004: does it hold up at 39 buckets?

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_retained_power_law_swarm39_20260728.py
```

Same model, same shape grid, same ridge grid, run through the existing swarm39 harness against the
incumbents registered there, under that harness's grouped out-of-fold protocol.

| panel / target | OOF rank | OOF RMSE | best incumbent | heldout rank | heldout RMSE |
|---|---|---|---|---|---|
| delphi_3e18 / uncheatable | 3 of 11 | 0.010209 | 0.008209 (hierarchical_phase_replay) | 2 of 11 | 0.045647 |
| delphi_3e18 / table9 | 4 of 11 | 0.026867 | 0.021851 (hierarchical_phase_replay) | 3 of 11 | 0.066942 |
| 60m / uncheatable | 4 of 11 | 0.010895 | 0.008732 (hierarchical_phase_replay) | 6 of 11 | 0.041086 |
| 60m / table9 | 5 of 11 | 0.028019 | 0.021894 (hierarchical_phase_replay) | 9 of 11 | 0.015192 |

**This is a partial result and should not be reported as a pass.** The model is mid-pack: never worst,
never best, within about 25 percent of the best out-of-fold error, and second of eleven on the 3e18
uncheatable heldout. But `hierarchical_phase_replay` and `bucket_family_grp` beat it out-of-fold in all
four cells, so "better than or on par with current models" is met in the weak sense of no serious
regression and not in the strong sense of matching the leaders.

The plausible reason is that the model spends 2m+1 free amplitudes with no pooling, while the leaders
pool buckets into families and shrink bucket-level excess toward the family value. At m = 39 that is 79
amplitudes against roughly 50 for the leaders. Family pooling is the obvious next change and does not
require touching the three mechanisms.

## Repository bug found and fixed

`swarm39_harness_20260725.fit_head` computed `float(design.mean(axis=0, keepdims=True) @ coefficients)`.
NumPy 2.5 turned conversion of a size-one non-scalar array into a hard error, so **every** model on that
track raised `TypeError` under the resolved environment, not only the new one. Fixed by extracting the
element explicitly. Numerically identical where it previously ran.

## Hypothesis queue update

- H1, H2 promoted to stable: the aggregate-channel shape failure is confirmed and the power-law fix
  carries the WSD80 improvement.
- H3 partially supported: an interaction is required and the retention gate supplies one, but the
  Jensen concentration term contributes nothing on any panel tried and should be re-examined or cut.
- H4 still open and now the leading candidate for the remaining gain miscalibration: the model predicts
  +0.0036 at the aggregate optimum where the measurement is 0.0000, because nothing forces its ordering
  term to vanish there.
- H5 confirmed as a real risk, in the specific form of amplitude count rather than of the phase term.
  Next change is family pooling of the benefit and damage amplitudes.

## WSD80-SUR-005: hierarchical pooling of amplitudes

WSD80-SUR-004 blamed the 39-bucket shortfall on amplitude count: 79 free amplitudes with no pooling
against roughly 50 for the leaders, which pool buckets into families and shrink bucket-level departures
toward the family value. Implemented that structure directly.

Each amplitude is written as its family's value plus a bucket departure, giving one design column per
family and one per bucket, and only the departure columns carry the ridge. That is expressed through
the swarm39 harness's `penalty_scale` hook, so the prior is part of the model rather than a separate
preprocessing step. With one domain per family the departure columns would duplicate the family
columns exactly, so they are dropped; on the two-bucket StarCoder panel the parameterisation therefore
collapses to plain per-domain amplitudes and, because nothing is left to shrink toward, the ridge is
inert there. That is a deliberate consequence, not an oversight, but it does mean the ridge grid does
no work on WSD80.

Same commands as before.

WSD80 is unchanged, as expected: CV RMSE 0.0389, 8.40 sigma, Spearman 0.8730, tied 12.15 sigma,
predicted tied optimum 0.285, optimum (0.185, 0.485), gains +0.0036 and +0.0146.

39-bucket, out-of-fold, essentially unmoved:

| panel / target | pooled OOF | unpooled OOF | rank |
|---|---|---|---|
| delphi_3e18 / uncheatable | 0.010207 | 0.010209 | 3 of 11 |
| delphi_3e18 / table9 | 0.026877 | 0.026867 | 4 of 11 |
| 60m / uncheatable | 0.010880 | 0.010895 | 4 of 11 |
| 60m / table9 | 0.027803 | 0.028019 | 4 of 11 |

39-bucket, coordinate-disjoint heldout, materially improved:

| panel / target | retained_power_law | hierarchical_phase_replay | bucket_family_grp | rank |
|---|---|---|---|---|
| delphi_3e18 / uncheatable | **0.045649** | 0.053921 | 0.054018 | 2 of 11 |
| delphi_3e18 / table9 | **0.059286** | 0.065768 | 0.069464 | 2 of 11 |
| 60m / uncheatable | **0.041097** | 0.044568 | 0.045682 | 7 of 11 |
| 60m / table9 | **0.012967** | 0.013085 | 0.013985 | 5 of 11 |

The model beats both out-of-fold leaders on the heldout panel in **all four cells**, having previously
been ninth on one of them. Pooling left within-panel error alone and bought generalisation, which is
what a better prior is supposed to do.

Mean ratio of heldout to out-of-fold RMSE across all four cells, as a crude overfitting measure:

| model | ratio |
|---|---|
| separate_heads | 2.45 |
| discounted_coverage_phase | 2.47 |
| discounted_coverage | 2.54 |
| unique_coverage_phase | 2.63 |
| **retained_power_law** | **2.73** |
| bounded_saturation | 2.75 |
| unique_coverage | 2.80 |
| compact_retained_state | 2.83 |
| effective_exposure_dsp | 2.89 |
| bucket_family_grp | 3.65 |
| hierarchical_phase_replay | 3.82 |

The two models that beat everything out-of-fold have the two worst ratios, so their within-panel
advantage does not transfer to coordinate-disjoint policies.

**Interpretation, stated carefully.** On the metric the harness provides for generalisation, the model
is now better than the incumbent leaders everywhere tested. On within-panel out-of-fold error it
remains third or fourth of eleven. Which of the two matters more is a judgement about the intended use:
for proposing unseen mixtures the heldout panel is the relevant one. The claim being made is that the
model is better on heldout in four of four cells and mid-pack out-of-fold, not that it is uniformly
best.

## Scalar phase-weighted dose: the null model, stated in epoch coordinates

Recording the collaborator's formalisation because it is the cleanest statement of what the retention
gate has to beat, and it agrees with the fiber analysis.

With `D` total tokens and `S_i` the size of bucket i, phase-specific materialised epochs are
`e0_i = beta0*D*w0_i/S_i` and `e1_i = beta1*D*w1_i/S_i`. Write the aggregate `ebar_i = e0_i + e1_i` and
the late epoch displacement `r_i = e1_i - beta1*ebar_i`, so `e0_i = beta0*ebar_i - r_i` and
`e1_i = beta1*ebar_i + r_i`. The tied policy has `r = 0`; positive `r_i` moves bucket i later at fixed
aggregate epochs. Preserving total compute requires `sum_i S_i*r_i = 0`.

The null model is single-index and additive in exposure but not linear in the response:
`Z = phi0*e0 + phi1*e1`, `L = F(Z)`, normalised by `beta0*phi0 + beta1*phi1 = 1` so a tied schedule has
`Z = ebar`. Then `Z = ebar + (phi1 - phi0)*r`. Timing changes only a constant early-late exchange rate;
once `Z` is known the separate values of `e0` and `e1` carry nothing. Forgetting, within-phase
repetition, consolidation and state-dependent interactions are all excluded by construction.

At an interior tied optimum `F'(ebar*) = 0`, so both phase derivatives vanish *separately* and
`kappa(ebar*) = 0`. That is strictly stronger than tied optimality, which only forces the early and
late directional effects to cancel.

Fitted on the 80/20 WSD StarCoder panel: `Z ~ 0.20*e0 + 4.20*e1`, equivalently
`E ~ 0.16*p0 + 0.84*p1 = a + 0.64*delta`. The curvature invariant `sqrt(rho/A'')` is 0.6399 at a = 0.18
and 0.6463 at a = 0.30, agreeing to about 1 percent, so the one-dimensional approximation describes the
local geometry of both fibers. But the first-order ordering coefficient is aggregate-dependent:
`kappa` is effectively zero (-0.001 to -0.010) at a = 0.30 and -0.14 to -0.17 at a = 0.18.

When `kappa = 0` and `rho > 0` the ordering effect starts at cubic order and the asymmetry cost at
quadratic, so the cost dominates *sufficiently close to* t = 0. That is a local statement; it is not
implied by the feasible range and does not guarantee the tied point wins along the whole fiber.
Empirically the tied point was best throughout the sampled a = 0.30 fiber, which is a measurement
rather than a corollary.

Globally the null model is rejected. Whole-surface scalar-dose RMSE is about 0.011 BPB against a seed
SD of 0.0046, and structurally it can produce no two-phase advantage at all, since every attainable `Z`
is also reachable by some tied policy. The observed optimum `(0.10, 0.50)` at a = 0.18 beats the tied
policy at its own aggregate by 0.0199 BPB and the globally best tied policy by 0.0096.

Equivalently, with `g(a) = max_t [L(a,0) - L(a,t)]`, the one-phase problem minimises `A(a) = L(a,0)`
and the two-phase problem minimises `A(a) - g(a)`. The optima differ because `g` is strongly
aggregate-dependent, near zero at a = 0.30 and large at a = 0.18. That aggregate-phase interaction is
exactly what a single scalar dose cannot carry, and it is what the retention gate exists to supply.

## WSD80-SUR-006: the ordering channel, and two additions that did not survive

Three changes tried on top of WSD80-SUR-005. One kept, two reverted.

**Kept: a marginal-value ordering block (H4).** Expanding the benefit term in the phase contrast
`d_i = w1_i - w0_i` around the tied policy at the same aggregate gives a first-order piece
proportional to `(abar_i + E0)^-(a+1) * d_i` and a second-order piece proportional to
`(abar_i + E0)^-(a+2) * d_i^2`. Taking both directly from the benefit function, rather than fitting a
free phase term, gives the property the fiber measurements demand without fitting it: the ordering
coefficient scales with a bucket's marginal value, so it is large where a domain is undersupplied and
small where the aggregate response has flattened. Verified structurally -- every phase column is
exactly zero at the tied policy. The ordering columns are pooled by family and split into positive and
negative parts because the head is nonnegative; a per-bucket ordering field is not offered.

On WSD80 this moved the predicted phase gain at the best tied aggregate from +0.0036 to +0.0018
against a measured 0.0000, and cross-validated error from 8.40 to 8.37 sigma with Spearman from 0.873
to 0.893.

**Reverted: leaving the ordering columns unpenalised.** Cost up to 1.7x on the 39-bucket heldout panels
while *improving* within-panel out-of-fold error. Penalising them on the same footing as bucket
departures recovered only part of it.

**Reverted: a family-substitution block**, the same power law applied to each family's total exposure
rather than bucket by bucket, which is the coarse nonlinearity the incumbent leaders have. It improved
out-of-fold error in all four cells and made heldout error wildly erratic: best-of-eleven on
60m/table9 at 0.008034, worst-of-eleven on 60m/uncheatable at 0.069589. It is near-collinear with the
family-pooled sum of per-bucket benefits, and adding both destabilises the head. Removed.

**The ordering channel is enabled per panel, and that decision used heldout data.** It is on for WSD80
and off for the 39-bucket panels, controlled by `ORDERING_CHANNEL`. The justification is that the
channel is identified only where the panel resolves a fixed-aggregate fiber, which the StarCoder
surface does and the 39-bucket panels do not. But within-panel cross-validation prefers it *on*
everywhere, so the switch could not be set by cross-validation and was set by looking at heldout error.
That is a form of test-set selection and the flag's value is therefore not an out-of-sample claim. What
is clean: with the flag off, the 39-bucket model is exactly the WSD80-SUR-005 configuration, whose
heldout numbers were measured before the ordering channel existed.

## Final state

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_retained_power_law_swarm39_20260728.py
```

WSD80, 166 coordinates, sigma 0.004633:

| model | CV sigma | Spearman | tied opt | optimum | gain 0.30 | gain 0.18 |
|---|---|---|---|---|---|---|
| effective_exposure_geometry | 12.68 | 0.752 | 0.410 | (0.410, 0.410) | +0.0009 | +0.0165 |
| effective_exposure | 13.28 | 0.772 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| separate_heads | 16.07 | 0.900 | 0.250 | (0.080, 0.360) | +0.0071 | +0.0422 |
| **retained_power_law** | **8.37** | 0.893 | 0.275 | (0.190, 0.430) | +0.0018 | +0.0117 |

Truth: tied optimum 0.300, optimum (0.100, 0.500), gains 0.0000 and +0.0199. Cross-validated error is
34 percent better than the best incumbent and Spearman is second by 0.007.

39-bucket, out-of-fold and coordinate-disjoint heldout:

| panel / target | OOF | rank | heldout | rank | HPR heldout | BFG heldout |
|---|---|---|---|---|---|---|
| 60m / table9 | 0.027803 | 4 | **0.012967** | 5 | 0.013085 | 0.013985 |
| 60m / uncheatable | 0.010880 | 4 | **0.041097** | 7 | 0.044568 | 0.045682 |
| delphi_3e18 / table9 | 0.026877 | 4 | **0.059286** | 2 | 0.065768 | 0.069464 |
| delphi_3e18 / uncheatable | 0.010207 | 3 | **0.045649** | 2 | 0.053921 | 0.054018 |

Beats both out-of-fold leaders on heldout in four of four cells; third or fourth of eleven out-of-fold.
Mean heldout-to-out-of-fold ratio 2.73 against 3.65 and 3.82 for those two leaders.

## Where this leaves the goal

WSD80 is met: 34 percent better cross-validated error than the best incumbent, from a form whose three
terms are each a named mechanism, with a phase channel that vanishes at the tied policy by
construction.

The 39-bucket half is met on heldout and not on out-of-fold. Two rounds of trying to close the
out-of-fold gap both produced the same signature -- better in-fold, worse out-of-sample -- which is
consistent with the leaders' own advantage being within-panel overfitting rather than something the
model is missing. I do not think the remaining out-of-fold gap should be closed by adding capacity.
The honest next step is a better test rather than a bigger model: an out-of-fold protocol whose folds
are as coordinate-disjoint as the heldout panel, so that the two metrics stop disagreeing.

Remaining open, unchanged: the concentration term still fits to zero everywhere and should be cut or
reformulated; tied-row RMSE is 12.5 sigma against the DSP family's 8.85, so the WSD80 win is bought on
off-diagonal rows; and the phase gain at aggregate 0.18 is now under-predicted at +0.0117 against a
measured +0.0199, having been over-predicted before the ordering block.

## WSD80-SUR-007: removing the test-set selection, and what that cost

WSD80-SUR-006 pinned `ORDERING_CHANNEL` per panel using heldout error. That is test-set selection and
had to go. Three things were tried, in order.

**Make the folds honest.** The 39-bucket harness splits by policy group, which keeps a policy's own
replicates together but says nothing about how close two *different* policies are; the WSD80 benchmark
used plain random K-fold on a densely sampled surface. In both cases a held-out row usually has a near
neighbour left in training, so out-of-fold error measures interpolation between adjacent mixtures
rather than prediction of new ones, and it rewards capacity that will not generalise. Added
`mixture_blocked_splits` to the harness, K-means blocking on the concatenated phase mixtures, so each
fold holds out a region the training folds do not cover, which is the structure the coordinate-disjoint
heldout panel has. `fit_model` takes a `split_fn` defaulting to the existing `grouped_splits`, so no
existing caller changes behaviour.

On WSD80 the effect is large and clarifying. Every model gets worse under blocking, which is expected,
but not equally:

| model | random folds | blocked folds | Spearman, blocked |
|---|---|---|---|
| effective_exposure | 13.28 | 18.07 | 0.320 |
| effective_exposure_geometry | 12.68 | 25.76 | 0.207 |
| separate_heads | 16.07 | 30.90 | 0.675 |
| retained_power_law | 8.37 | 9.91 | 0.929 |

The incumbents degrade by 40 to 100 percent and this model by 18. Under the honest protocol the gap
widens from 34 percent to **45 percent** on RMSE, and rank correlation separates completely, 0.929
against 0.320 for the best incumbent.

**Make the ordering channel selectable rather than pinned.** Added it as a grid dimension so
cross-validation chooses it. It did not work. Under blocked folds, cross-validation still selects it on
all four 39-bucket cells and it still destroys heldout error, now worse: Delphi 3e18 table9 heldout
went to 0.277536 against roughly 0.06 without it.

**So the ordering channel was removed.** A component that every fold protocol prefers and that every
protocol's heldout contradicts is not identified by these panels. Keeping it would have required
choosing it with test-set information, which is the thing being fixed. This is a negative result worth
stating plainly: the marginal-value ordering block is theoretically well-motivated, it vanishes at the
tied policy by construction, it improved the StarCoder fit, and it is still not admissible, because
nothing computable from the fit panel distinguishes the case where it helps from the case where it
ruins the prediction.

**The blocked folds themselves are wrong for the 39-bucket panels.** Removing the ordering channel left
Delphi 3e18 table9 heldout unchanged at 0.277536, which shows the blowup came from blocked-fold
selection, not from the ordering columns. Those panels are intervention-designed rather than densely
sampled, so blocking in mixture space produces folds that are unrepresentative extrapolations and
selection against them picks an extreme shape. Reverted to the harness's documented `grouped_splits`
there, which is also what every incumbent number on record was produced under, so comparability is
preserved.

**Where the protocol choice now comes from.** WSD80 uses blocked folds because the surface is densely
sampled on a grid, so random folds demonstrably leak; that is a property of the fit panel. The
39-bucket panels use the harness's established protocol, unchanged. Neither choice consults heldout
error, and the model itself is now identical on every panel with no flags.

## Final state after WSD80-SUR-007

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_retained_power_law_swarm39_20260728.py
```

WSD80, 166 coordinates, mixture-blocked folds, sigma 0.004633:

| model | CV sigma | Spearman | tied opt | optimum | gain 0.30 | gain 0.18 |
|---|---|---|---|---|---|---|
| effective_exposure | 18.07 | 0.320 | 0.400 | (0.405, 0.400) | +0.0119 | +0.0516 |
| effective_exposure_geometry | 25.76 | 0.207 | 0.410 | (0.410, 0.410) | +0.0009 | +0.0165 |
| separate_heads | 30.90 | 0.675 | 0.250 | (0.080, 0.360) | +0.0071 | +0.0422 |
| **retained_power_law** | **9.91** | **0.929** | **0.300** | (0.085, 0.465) | +0.0101 | +0.0463 |

Truth: tied optimum 0.300, optimum (0.100, 0.500), gains 0.0000 and +0.0199. The predicted tied optimum
is now exactly right and the predicted optimum (0.085, 0.465) is the closest any model has come to
(0.100, 0.500). The phase gains are over-predicted, which is the cost of dropping the ordering block.

39-bucket, harness grouped protocol:

| panel / target | OOF | rank | heldout | rank | HPR heldout | BFG heldout |
|---|---|---|---|---|---|---|
| 60m / table9 | 0.027803 | 4 | **0.012967** | 5 | 0.013085 | 0.013985 |
| 60m / uncheatable | 0.010880 | 4 | **0.041097** | 7 | 0.044568 | 0.045682 |
| delphi_3e18 / table9 | 0.026877 | 4 | **0.059286** | 2 | 0.065768 | 0.069464 |
| delphi_3e18 / uncheatable | 0.010207 | 3 | **0.045649** | 2 | 0.053921 | 0.054018 |

Beats both out-of-fold leaders on heldout in four of four cells. Mean heldout-to-out-of-fold ratio 2.73
against 3.65 and 3.82 for those two.

## Hypothesis queue update

- H4 falsified as an admissible model component. The marginal-value ordering block does what it was
  designed to do and cannot be selected without test-set information on these panels. Recorded as a
  negative result rather than carried as a flag.
- New: the fold protocol is itself a modelling decision with a large effect, and the right one differs
  between a densely sampled grid and an intervention-designed panel. Both choices here are justified
  by fit-panel structure, but that justification deserves an explicit test rather than an argument.

## WSD80-SUR-008: the panel triples, and three model repairs

The StarCoder panel grew from 166 to 346 unique coordinates with six new fixed-aggregate fibers at
0.35, 0.40, 0.50, 0.60, 0.70 and 0.80. The optima did not move -- best tied still 0.300 at 0.945062,
best overall still (0.100, 0.500) at 0.935468 -- but the aggregate axis is now resolved, and it changes
what the panel can test.

**The phase gain g(a) is V-shaped with its minimum exactly at the one-phase optimum, and the best
contrast changes sign there.**

| a | 0.18 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| g(a) | +0.0199 | 0.0000 | +0.0017 | +0.0040 | +0.0395 | +0.0696 | +0.1374 | +0.2433 |
| best contrast | +0.400 | 0.000 | -0.104 | -0.167 | -0.375 | -0.542 | -0.708 | -0.833 |

Code late below a = 0.30, code early above it. That is `kappa(a) = K * F'(E(a))` changing sign with
`F'`, measured at eight points rather than argued from two.

Three repairs followed, each traceable to a measurement rather than to a hunch.

**The retention gate was corrupting the one-phase channel.** It was `exp(-lambda*(1 - w1))`, which
varies along the tied diagonal, so the phase term was silently rewriting the aggregate response --
moving retained share by up to 0.43. A phase term must be identically neutral at every tied policy.
The first repair, gating on `max(w0 - w1, 0)`, is neutral but one-sided: it can penalise putting a
domain early and can never reward putting it late. That collapsed the fitted optimum onto the diagonal
and took contrast-sign agreement to 0 of 7. Gating on the signed contrast, `exp(lambda*(w1 - w0))`, is
exactly one at every tied policy and moves both ways.

**The even term is not always a cost.** Codex's fiber decomposition
(`reference_outputs/wsd80_fixed_aggregate_fiber_decomposition_20260728/report.md`) reported that the
orientation-averaged term goes negative at high aggregate, and that a phase model must not force it
nonnegative. Verified: 12 of 79 antithetic pairs have `c(d) < 0`, all at aggregates 0.50 and above,
which is exactly where the six new fibers live. A nonnegative head cannot represent that at all, so
every even column is now entered with both signs.

**The odd term needs both derivatives.** The aggregate response has a benefit term and a damage term,
so its derivative in the contrast direction has two, and a column built from the benefit derivative
alone is monotone in the aggregate and can never change sign. Fitting both separately lets their
difference cross zero where the net marginal value does. Worth 7 percent on blocked-fold error.

**One change was confounded and one diagnostic resolved it.** Tightening the gate clip was bundled with
the damage-derivative columns and the pair regressed. Because the phase columns are identically zero at
tied policies, they cannot move tied-row error; tied error moved from 12.6 to 17.5 sigma, which located
the damage in the gate alone. Reverting the clip recovered everything and kept the improvement.

## Ablations

| configuration | random | blocked | Spearman | tied | contrast sign |
|---|---|---|---|---|---|
| gate on, ordering block on | **9.74** | **13.45** | 0.877 | **12.45** | 3/7 |
| gate on, ordering block off | 10.83 | 16.39 | 0.861 | 16.75 | **6/7** |
| gate off, ordering block on | 15.95 | 23.06 | 0.582 | 12.16 | 1/7 |

With the gate off the phase channel dies: `g(a)` is zero at every aggregate. **The retention gate is
the mechanism; the marginal-value ordering columns are not.** That is the opposite of what the H4
reasoning predicted and it is worth stating plainly -- the columns were designed to carry the sign
flip, and what they actually buy is accuracy, while *costing* sign fidelity.

The ordering block is a cross-validated grid dimension, so selection picks it on, on RMSE. Contrast
sign is reported as a diagnostic and is not optimised against.

## Final WSD80, 346 coordinates

| model | random | rho | blocked | rho | tied | optimum |
|---|---|---|---|---|---|---|
| effective_exposure | 11.81 | 0.784 | 22.95 | 0.164 | 7.13 | (0.460, 0.235) |
| effective_exposure_geometry | 10.80 | 0.841 | 24.62 | 0.182 | **5.23** | (0.345, 0.345) |
| separate_heads | 13.87 | 0.911 | 28.80 | 0.612 | 33.58 | (0.045, 0.300) |
| **retained_power_law** | **9.74** | **0.934** | **13.45** | **0.877** | 12.45 | (0.025, 0.740) |

Truth: optimum (0.100, 0.500). Best on both fold protocols, 10 percent better than the best incumbent
under random folds and 41 percent under blocked, with much higher rank correlation under both.

Two weaknesses stand. Tied-row error is 12.45 sigma against `effective_exposure_geometry`'s 5.23, so
the one-phase channel is still the worst part of the model. And the optimum overshoots late, (0.025,
0.740) against a measured (0.100, 0.500), with contrast-sign agreement 3 of 7 against 7 of 7 for the
plain effective-exposure DSP. The model is more accurate and less structurally faithful than the
incumbent it beats, which is an uncomfortable combination and the obvious target for the next round.

## WSD80-SUR-009: the 39-bucket side regresses, and the cause is not what I assumed

Running the current model through nested cross-validation on the 39-bucket panels, with an adapter bug
fixed first (see below), gives:

| panel / target | nested | rank | heldout | rank | previous nested | previous heldout |
|---|---|---|---|---|---|---|
| 60m / table9 | 0.035212 | 9 | **0.010920** | **3** | 0.029225 | 0.012967 |
| 60m / uncheatable | 0.014714 | 8 | **0.034837** | **1** | 0.011468 | 0.041097 |
| delphi_3e18 / table9 | 0.028501 | 6 | 0.059829 | 2 | 0.026881 | 0.059286 |
| delphi_3e18 / uncheatable | 0.011369 | 6 | 0.048704 | 3 | 0.010456 | 0.045649 |

The WSD80 repairs traded 39-bucket nested accuracy for heldout accuracy. Nested is worse in all four
cells and now ranks sixth to ninth of eleven; heldout improved to first through third. The stated bar
is good cross-validated results on the fixed 280-row panel, and sixth of eleven does not meet it.

Two explanations were tested and both are wrong.

**Not regularisation strength.** Extending the ridge grid to 10 and 100 moved nested error by under one
percent and cross-validation never selected the new values, choosing 1.0 or 0.01 as before.

**Not the phase machinery.** Forcing the ordering channel off makes nested error slightly *worse* in
all four cells: 0.011480 against 0.011369, 0.028526 against 0.028501, 0.015046 against 0.014714,
0.035459 against 0.035212. The extra columns are not what the panel cannot afford.

So the gap to `hierarchical_phase_replay`, which is 34 percent on Delphi uncheatable, lives in the
aggregate part of the model rather than the phase part. That contradicts the working assumption behind
the last several rounds and redirects the remaining work: what needs improving at 39 buckets is the
benefit-and-damage response, most likely the family-coverage nonlinearity the leaders have and this
model does not, applied *instead of* the pooled sum of per-bucket terms rather than alongside it. The
earlier attempt added it alongside and the two were near-collinear.

## A process failure worth recording

The first attempt at this table reported numbers that were an hour stale. String patches to the
swarm39 adapter had silently failed to apply because the formatter had joined the target lines, so the
design grew to 100 columns while the names stayed at 85; the assertion fired in every cell, the model
was skipped everywhere, and the run's own guard caught it. The stale CSV was then read and reported as
current. The same class of mistake had already happened once with a progress count.

Two changes: the adapter's assertion now prints both counts rather than a bare message, and a smoke
test that fits one cell is run before any results file is read.

## WSD80-SUR-010: a visual check finds what eight rounds of metrics missed

Added `plot_wsd80_model_vs_surface_20260728.py`: the fitted response as a continuous sheet, the
measurements as bare points on top, orthographic cameras from five fixed angles. The measured data is
deliberately never interpolated, because two smooth sheets read as agreeing even when they do not.

The first render showed a sharp fold across the fitted sheet with no counterpart in the data. The first
diagnosis, that the gate's hard clip was putting a kink in the response, was wrong: replacing the clip
with a scaled `tanh` left the fold exactly where it was. The real cause was the sign-freedom trick in
the ordering block. Splitting a column as `max(o, 0)` and `max(-o, 0)` with independent nonnegative
coefficients gives `a*max(o,0) + b*max(-o,0)`, which is kinked wherever `o` crosses zero, and for a
family-pooled ordering column that locus is a curve across the mixture square. Entering `o` and `-o`
instead leaves the fitted combination linear in `o`, so it reaches either sign and stays smooth. The
even terms already used the correct form; the ordering columns did not.

Full-panel fit before and after, in training-seed sigma:

| | median residual | rmse | worst over-prediction |
|---|---|---|---|
| hinge split | 2.66 | 8.51 | -59.5 |
| signed pair | **1.06** | 7.25 | **-7.3** |

A median residual of 1.06 sigma means a typical policy is predicted to within one training seed of
noise. **No scalar metric being tracked showed this bug**: not cross-validated RMSE, not Spearman, not
tied-row error, not contrast-sign agreement, not the g(a) profile. It was only visible as geometry.

The smooth-saturation gate was kept even though it did not fix the fold, because a hard clip does put a
derivative jump in the response and there is no reason to prefer one.

## Robust fitting

The measured surface is smooth and the scatter around it is noise with occasional far outliers, so a
squared-error head spends amplitudes on points that carry no signal. The head is now fitted by
iteratively reweighted least squares with a Huber weight whose cut comes from the residual median
absolute deviation, so the same setting transfers to panels with different noise scales. Median
absolute residual is reported alongside RMSE for every model, because the two move in opposite
directions when outliers are what RMSE is measuring.

## WSD80 after both changes, 346 coordinates

| model | random rmse | random median | rho | blocked rmse | blocked median | rho | optimum |
|---|---|---|---|---|---|---|---|
| effective_exposure | 11.81 | 4.72 | 0.784 | 22.95 | 11.01 | 0.164 | (0.460, 0.235) |
| effective_exposure_geometry | 10.80 | 3.42 | 0.841 | 24.62 | 11.52 | 0.182 | (0.345, 0.345) |
| separate_heads | 13.87 | 5.88 | 0.911 | 28.80 | 16.43 | 0.612 | (0.045, 0.300) |
| **retained_power_law** | **9.43** | **1.21** | **0.977** | **12.03** | **3.24** | **0.915** | (0.035, 0.500) |

Truth: optimum (0.100, 0.500). Median residual is 2.8x better than the best incumbent under random
folds and 3.4x better under blocked; blocked RMSE is 48 percent better; the predicted phase-1 share is
exactly right. Contrast-sign agreement improved from 3 of 7 to 6 of 7, against 7 of 7 for the plain
effective-exposure DSP.

Residual error is now confined to the boundary: the `p0 = 0` edge and the all-StarCoder corner, where
the response rises faster than the model does. The interior, which is where any deployable policy sits,
is fitted to about one sigma.

## WSD80-SUR-011: WSD80 met decisively; the 39-bucket search narrows by elimination

Extending the retention range to 10 was the change that closed the StarCoder half. It was found by a
sweep motivated by the residual map, which showed the only failing region was policies removing a
domain from the late phase entirely.

Final WSD80, 346 coordinates, both fold protocols, verified fresh:

| model | random rmse / median / rho | blocked rmse / median / rho | contrast sign | optimum |
|---|---|---|---|---|
| effective_exposure | 11.81 / 4.72 / 0.784 | 22.95 / 11.01 / 0.164 | 7/7 | (0.460, 0.235) |
| effective_exposure_geometry | 10.80 / 3.42 / 0.841 | 24.62 / 11.52 / 0.182 | 5/7 | (0.345, 0.345) |
| separate_heads | 13.87 / 5.88 / 0.911 | 28.80 / 16.43 / 0.612 | 7/7 | (0.045, 0.300) |
| **retained_power_law** | **6.01 / 1.11 / 0.980** | **10.53 / 2.31 / 0.957** | **7/7** | (0.015, 0.485) |

Truth (0.100, 0.500). Against the best incumbent on each axis: 44 percent better random RMSE, 54
percent better blocked RMSE, 3.1x better random median residual, 4.8x better blocked median. Median
residual of 1.11 sigma is the training-seed noise floor. Contrast-sign agreement reached 7 of 7 from 3
of 7 two entries ago, so the model no longer trades structural fidelity for accuracy: it now leads on
both at once, which it did not before the kink fix.

## What has been eliminated on the 39-bucket side

Each tested by nested cross-validation on the fixed 280-row panel, against
`hierarchical_phase_replay` at 0.008531 on Delphi uncheatable.

| hypothesis | result |
|---|---|
| regularisation strength | ridge extended to 100 moves error under one percent and is never selected |
| the phase machinery is unaffordable | forcing the ordering channel off makes nested error *worse* in all four cells |
| family aggregation form | power-of-sum against sum-of-powers, 0.009970 against 0.010279, about 3 percent |
| damage threshold form | multiplicative log hinge 0.010377 against additive 0.010279, slightly worse |

Four negative results, and the second is the informative one: the intuitive story that thirty-nine
buckets cannot afford the extra phase columns is simply false, because removing them costs accuracy.
The panels do carry phase signal this model is not fully using.

## Two tests in flight

A capacity diagnostic that no error metric provides: each model's *own* predicted two-phase gain, its
best tied policy minus its best policy overall. A model whose response depends on the two phases only
through an additive phase-weighted dose must return exactly zero, because the tied class already
attains every reachable dose. The measurement is +0.009594 BPB. This separates models by what they can
represent rather than by residual size, and is the right frame for comparing against
`effective_exposure`, which places its optimum at (0.460, 0.235) with phase 0 above phase 1, the wrong
side of the diagonal entirely.

And family-specific retention. The gate currently carries one global rate, so every domain is assumed
to forget identically; that is the weakest remaining assumption in the phase channel. Making it a
per-family shape parameter would multiply the grid by the family count, so instead the family-pooled
benefit block is entered at every rung of the retention ladder with the extra rungs shrunk, letting the
head give each family an effective retention profile through amplitudes. This is the first change that
adds a property the model lacks rather than permuting one it has.

## WSD80-SUR-012: the model was never running on the 39-bucket path

Codex's third review found that `harness.fit_model` calls its own module-level nonnegative
least-squares head at lines 532 and 542, so `retained.solve_head` was never invoked on the swarm path.
Verified by reading. Every 39-bucket number reported before this entry measured a different estimator
than the StarCoder numbers: no Huber weighting, a different column scaling, and therefore a different
effective ridge.

**All seven eliminations below are provisional.** They establish that these changes do not help an
NNLS-headed variant, which is a much weaker claim than the one they were used to support.

| hypothesis | result under the NNLS head |
|---|---|
| regularisation strength | under one percent, never selected |
| phase machinery unaffordable | false; removing it makes nested worse |
| family aggregation form | about three percent |
| damage threshold form | slightly worse |
| benefit functional form | three variants within six percent |
| family-specific retention | no change at all |
| selection variance from grid size | refuted in the opposite direction: 2880 combinations beat 480, 128 and 32 monotonically, so the large search earns its size |

The grid-size result is worth keeping. It says the search is not too big for a 280-row panel, which was
a reasonable hypothesis and is now closed.

## Fixes

A `head` hook on `harness.Model`, defaulting to `None` so every existing caller keeps the module head.
A model whose definition includes how its coefficients are estimated has to be able to supply that, or
the harness silently benchmarks a different estimator.

And the IRLS loop, which Codex separately flagged as running a fixed six passes with no convergence
test. The first repair tested convergence on coefficients, which was measurably wrong: cost scaled
linearly with the iteration cap, 17 ms at one iteration against 352 ms at fifty, meaning the criterion
essentially never fired. Collinear columns let the coefficient vector drift long after the fitted
response has settled. Testing the change in predictions against the residual scale instead cuts the
median solve from 352 ms to 51 ms, a factor of seven, and makes nested cross-validation on the swarm
panels feasible at about one hour per cell.

Even so the robust head is 23 times the cost of the harness's single NNLS solve, which is a real and
permanent price for routing the model's own estimator through this path.

## Process note

This is the second silent substitution in this session, after the adapter name drift that skipped the
model in every cell. Both produced complete runs and plausible numbers while measuring something other
than what was claimed. The guards now in place -- an assertion that prints both column counts, the head
hook, and a freshness check before any results file is read -- were each added in response to a
specific incident rather than as precaution.

## WSD80-SUR-013: gradient-tied aggregate control is too rigid

Before seeing its results, a low-capacity aggregate-conditioned phase model was frozen:

```
L(w0, w1) = A(abar)
          + xi * grad_T A(abar)^T (w1 - w0)
          + gamma * I_beta(abar, w1 - w0)

I_beta(abar, delta) = 0.5 * beta0 * beta1 * sum_i delta_i^2 / abar_i.
```

`A` is the hierarchical power-law benefit plus repetition-damage response. Its amplitudes are shared
exactly with the directional derivative, so this is not another free phase field. `I_beta` is the
Fisher-quadratic term of weighted Jensen-Shannon divergence. The full weighted divergence was rejected
during the algebraic audit because at unequal 80/20 phase lengths it contains odd higher-order terms
and is not exactly invariant under fixed-aggregate contrast reversal.

The joint estimator failed immediately. Random-fold RMSE was 14.981 training sigma, tied RMSE was
0.0925 BPB, the raw optimum was `(0.000, 0.420)`, and the model predicted a 0.0505 BPB two-phase gain.
Asymmetric rows rewrote the shared aggregate response.

A preregistered identification repair then fit `A` only on tied rows, froze it, and estimated only the
two nonnegative phase scalars from asymmetric residuals. This restored the aggregate channel:
tied RMSE was 0.0075 BPB and its optimum was 0.295 against the measured 0.300. It also recovered all
seven nonzero contrast signs. The overall model still failed:

| diagnostic | measured | two-stage gradient-tied |
|---|---:|---:|
| random-fold RMSE | -- | 14.804 training sigma |
| global optimum | `(0.100, 0.500)` | `(0.000, 0.210)` |
| two-phase gain | 0.009594 | 0.003270 |
| gain at aggregate 0.18 | 0.019855 | 0.0085 |
| sign agreement | 7 fibers | 7 / 7 |

The failure is specific: the aggregate gradient predicts the direction of phase control but one global
`xi` cannot predict its magnitude. It underprices the low-aggregate fiber and overprices several
high-aggregate fibers, then selects a boundary aggregate. This instantiation is blocked and will not
consume a 39-bucket nested run. Reopening it requires a materially new state variable for
aggregate-dependent plasticity or control susceptibility, identified without tuning to these WSD
residuals; adding a free aggregate calibration of `xi` would merely name the missing response.

## WSD80-SUR-014: authoritative RPL reruns and a baseline bug

The corrected retained-power-law runs completed on both fold protocols:

| protocol | RMSE sigma | median absolute sigma | Spearman | optimum | predicted gain |
|---|---:|---:|---:|---|---:|
| random | 3.3114 | 0.7767 | 0.9950 | `(0.075, 0.475)` | 0.008044 |
| blocked | 11.6490 | 1.4084 | 0.9373 | `(0.075, 0.475)` | 0.007720 |

The blocked RMSE is worse than the provisional number in the handoff, but the optimum, gain and all
seven contrast signs remain stable. The benchmark's `canonical` DSP row was invalid:
`FitConfig("canonical", False)` silently inherited `variant_name="effective_exposure"`, making it an
exact duplicate of effective-exposure DSP. The configuration now explicitly passes `"canonical"`;
canonical-only reruns give 14.7795 training sigma under random folds and 25.0081 under blocked folds.
Its raw optimum is the corner `(0.000, 1.000)`, with a predicted 0.064766 BPB two-phase gain. Canonical
DSP is therefore worse than RPL on fit and optimum behavior; the repaired baseline does not change the
leading candidate.

## WSD80-SUR-015: continuous-time retained-state transition

**2026-07-29 19:04 PDT; preregistered before benchmark results.**

The simplest aggregate-conditioned phase-control model was blocked in WSD80-SUR-013. Its aggregate
gradient recovered all seven phase-order signs but not their magnitude, and adding a free
aggregate-dependent multiplier would only rename the missing response. The next test introduces a
different mechanism rather than calibrating that residual: a latent bucket state with exact
acquisition-and-forgetting dynamics,

```
dS_i / dt = q_p w_i - lambda (1 - w_i) S_i.
```

The first term acquires bucket state; the second loses existing state while other buckets are trained.
For a constant phase, the transition is solved analytically. Two limits make the mechanism auditable:

- `lambda = 0` reduces exactly to additive phase-weighted dose.
- With equal phase utility, a tied policy is invariant to where training is partitioned because the
  state transition is a semigroup.

Only the retained-state transition changes. Power-law benefit, epoch-based repetition damage,
concentration cost, family hierarchy, robust head, shape grid cardinality, and nested selection stay
the same as retained power law. This isolates whether autonomous interference-driven forgetting is the
missing 39-bucket mechanism.

Algebraic tests passed for both limits, finite corner behavior, and nonnegative state. Lint passed.
The first detached launch was reaped after dependency setup; the persistent reruns are:

```
env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv-cache uv run \
  experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py \
  --folds random --models retained_state_dynamics \
  --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/retained_state_dynamics_wsd80_20260729

env PYTHONUNBUFFERED=1 UV_CACHE_DIR=/tmp/uv-cache uv run \
  experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_wsd80_incumbents_20260728.py \
  --folds blocked --models retained_state_dynamics \
  --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/retained_state_dynamics_wsd80_20260729
```

Promotion gate, frozen before results:

1. The raw optimum remains near the observed `(0.10, 0.50)` rather than moving to a boundary.
2. Predicted two-phase gain remains near the observed `0.009594` BPB.
3. Blocked-region error improves or is preserved relative to retained power law; an interpolation-only
   gain is insufficient.
4. Only a passing WSD80 transition may consume a 39-bucket nested evaluation.

**Launch correction, 2026-07-29 19:14 PDT.** The two foreground terminal sessions above were
terminated when their Codex response closed, despite being live during that turn. They produced no
result files and are not evidence. Both commands were relaunched under detached `tmux` sessions
`codex_rsd_random_20260729` and `codex_rsd_blocked_20260729`, with stdout in `random.log` and
`blocked.log` and explicit exit codes in matching `.exit` files. The Python workers were CPU-active
and both logs had loaded the 346-coordinate panel immediately after launch.

## WSD80-SUR-016: reject brute-force execution, not just bad model forms

**2026-07-29 19:48 PDT.** The four 39-bucket reruns were stopped after 1 hour 50 minutes without a
completed row. The commands omitted `--models retained_power_law`, so every cell was scheduled to
rerun eleven models: retained power law, five Observatory baselines, and five older candidates. This
was unnecessary for confirming the corrected retained-power-law numbers.

The retained model alone searches 540 shapes by three ridges. The full fit plus five outer nested-CV
fits each use five inner folds, for about 48,600 robust regressions per target cell. Each robust
regression invokes an initial bounded solve plus iterative Huber solves. A 30-fit sample on the Delphi
panel averaged 5.87 bounded solves and 113 ms per robust fit. The original jobs ran that workload
single-threaded and then intended to repeat nested evaluation for ten unrelated models.

The benchmark now:

- accepts `--outer-workers` and evaluates independent outer folds concurrently;
- prints each completed outer fold;
- writes a per-cell partial CSV immediately after a model completes;
- preserves the same five-outer-by-five-inner nested-CV estimand.

A three-fold smoke test produced bit-identical serial and three-worker metrics. The four cells were
relaunched with `--models retained_power_law --outer-workers 3`.

The WSD gate had a separate ordering inefficiency: it ran fifteen outer fits before computing the raw
optimum that can reject a model after one selected fit. Both 33-minute incomplete runs were stopped.
The benchmark now supports `--diagnostics-only`; the retained-state dynamics candidate is first being
tested on its raw optimum, tied response, two-phase gain, and fiber profile. Repeated CV is conditional
on passing those structural checks.

## WSD80-SUR-017: retained-state dynamics fails the diagnostics-first gate

**2026-07-29 19:52 PDT.** Both diagnostics-only fits completed successfully. The continuous-time
retained-state transition captures much of the measured phase-gain profile and predicts a nonzero
two-phase advantage, but it fails the simpler tied-response and optimum-location checks:

| fold protocol used for shape selection | tied RMSE | predicted tied optimum | raw two-phase optimum | distance from observed optimum | predicted gain |
|---|---:|---:|---|---:|---:|
| random | 0.0669 | 0.270 | `(0.120, 0.385)` | 0.1167 | 0.01148 |
| blocked | 0.0759 | 0.280 | `(0.120, 0.385)` | 0.1167 | 0.01148 |

The measured optimum is `(0.100, 0.500)` with a 0.009594 BPB advantage. Random-fold selection
recovered six of seven phase-gain signs; blocked selection recovered seven of seven. This is useful
mechanistic evidence that autonomous forgetting can reproduce the broad phase-gain profile, but the
transition distorts the one-phase response by 14.4--16.4 training-seed standard deviations and moves
the optimum substantially farther from truth than retained power law. It is blocked before repeated
CV and will not consume a 39-bucket run.

## WSD80-SUR-018: two-tier fitting protocol for fast model iteration

**2026-07-29 20:10 PDT.** Added
`experiments/domain_phase_mix/exploratory/two_phase_many/fast_surrogate_iteration_20260729.py` as an
explicit development-only first rung. It:

- uses one deterministic grouped three-fold partition of the fit swarm;
- selects the nonlinear shape and ridge without touching append-only heldouts;
- evaluates independent nonlinear shapes concurrently;
- uses the same robust constrained head as the authoritative retained-power-law benchmark.

The authoritative five-outer-by-five-inner nested audit remains unchanged and is reserved for frozen
survivors. A smoke test over eight shapes and two ridge values selected exactly the same shape, ridge,
and OOF RMSE in serial and three-worker modes:

```
OOF RMSE = 0.010060702000921476
ridge = 0.01
shape = {
  benefit_exponent: 0.25,
  benefit_offset: 0.01,
  damage_exponent: 1.5,
  damage_threshold: 0.0,
  retention: 0.0,
  late_multiplier: 4.0,
  ordering_channel: true,
}
```

This changes iteration latency, not the evidentiary standard. Development screens may reject a model
cheaply; only a model frozen after passing algebraic, StarCoder, and fit-swarm screens advances to the
full nested and heldout audits.

**Measured full-grid latency.** The complete Delphi-Uncheatable development screen evaluated all 540
shapes, three ridges, and three folds in 622.6 seconds and selected OOF RMSE 0.009226, ridge 1.0,
benefit exponent 1.0, benefit offset 0.1, damage exponent 1.5, retention 10, late multiplier 4, and the
ordering channel enabled. Eight process workers sustained real CPU parallelism. A four-shape
serial-versus-process check was bit-identical. Ten minutes is substantially below the full nested audit,
but still too expensive for every small mechanism edit.

The next iteration protocol is therefore hierarchical:

1. algebraic and raw-optimum diagnostics;
2. mechanism-local ablation with the parent's nuisance shape and ridge fixed;
3. full three-fold joint grid only after the new mechanism contributes;
4. five-by-five nested CV and heldouts only for a frozen survivor.

The current trust-region bounded solver is also expensive. SciPy's dense BVLS method was 2.34 times
faster over 18 representative robust fits, but one fit changed predictions by 0.416 BPB despite a
median maximum difference of only 1.3e-6. The hierarchical design has enough non-identifiability that
solver substitution can change the estimator materially. BVLS is not adopted.

## WSD80-SUR-019: paper-inspired repetition terms do not improve RPL

**2026-07-29 22:12 PDT.** Tested a frozen mechanism-local ladder motivated by
*Prescriptive Scaling Laws for Data Constrained Training*: one-epoch repetition onset, phase-local
overfit cost, normalized retained state, and the phase-local Jensen excess over aggregate repetition.
The benchmark and complete results are in
`reference_outputs/factorized_retained_overfit_20260729/`.

The clean factorization failed. Normalized retention worsened Delphi-Uncheatable OOF by 6.6%.
Replacing aggregate damage with phase-local repetition improved that OOF score by only 0.7% while
raising StarCoder blocked error from 3.64 to 15.60 seed SD and moving the raw optimum from
`(0.075, 0.475)` to `(0.115, 0.220)`. Removing the explicit ordering block was worse again:
Delphi OOF increased 26.6% and StarCoder blocked error reached 28.09 seed SD. RPL's phase-order
channel remains necessary.

One additive global phase-repetition-excess coefficient reduced Delphi OOF from 0.009226 to 0.009054
and preserved the StarCoder optimum and gain. Full joint selection retained the parent nonlinear
shape and ridge. This apparent survivor failed the frozen Delphi development archive: heldout RMSE
rose from 0.05452 to 0.05470, Regret@1 rose from 0.00514 to 0.00898, low-tail RMSE rose from 0.01915
to 0.02034, and low-tail optimism rose from 0.01790 to 0.01915. Calibration slope improved slightly
and one fewer error exceeded 0.05 BPB, but those gains do not offset the selection regression.

The new feature is 0.9727 correlated with RPL's existing concentration basis and its fitted
coefficient is only 0.000121. It is blocked as a weakly identified curvature reparameterization, not
promoted as a new mechanism. The paper's scale-dependent `N/U` mechanism remains a candidate for a
future cross-scale aggregate model, but its direct phase-order transplant is rejected.

## WSD80-SUR-020: aggregate-conditioned replay control preregistration

**2026-07-30; frozen before candidate scoring.** The next route uses the original 280-row 300M
two-phase fit panel plus exactly the 240 single-phase qsplit exposure-average collapses. The latter
form 240 exact aggregate-matched pairs: each tied weight vector equals `0.8 w0 + 0.2 w1` for its
source two-phase row with maximum absolute error `3.61e-16`. The remaining 40 two-phase rows are the
39 proportional domain deletions and one stratified baseline. Correspondence keys are indivisible CV
groups, leaving 520 observations in 280 groups.

The frozen candidate is

```
L(a, delta) = A(a)
            + xi R(a)^(-q) grad A(a)^T delta
            + gamma I_beta(a, delta)
            + zeta J(a, delta).
```

`A` is the RPL hierarchical power-law benefit plus aggregate repetition-damage backbone, identified
only from tied qsplit rows. `R` is token-weighted expected materialized epochs normalized to one at
the proportional policy; it is a new dimensionless plasticity state rather than an output
calibration. `I_beta` is Fisher-quadratic phase information and `J` is the Jensen gap of convex
within-window replay beyond one epoch. All phase amplitudes are nonnegative.

This form encodes the fiber hypothesis exactly: at an interior optimum of `A`, its tangent gradient
is zero, so the odd control term vanishes for every feasible contrast, while both even terms remain
nonnegative. It can still prefer a two-phase policy at an off-optimum aggregate. The frozen replay
exponents are `q in {0, 0.5, 1, 2}` plus the aggregate-only ablation. The aggregate grid and ridge are
unchanged from the non-phase RPL backbone. No 3e18 heldout or adversarial outcome may select this
batch. Full details are in
`reference_outputs/aggregate_conditioned_replay_control_20260730/preregistration.md`.

## WSD80-SUR-021: protocol repair and aggregate-Hessian control preregistration

**2026-07-30; frozen before candidate scoring.** Independent review of
WSD80-SUR-020 found that the requested 520-row panel has 282 physically tied
policies and 238 genuinely asymmetric policies. Forty-two rows labeled
`two_phase` are physically tied and cannot identify phase response. The prior
policy-class metrics and paired count are therefore invalidated and will be
regenerated from phase-weight equality.

The review also found nominal-versus-realized phase-fraction drift in the
StarCoder fiber diagnostics, random inner selection inside blocked outer
folds, duplicated full-fit diagnostics across protocol labels, and unequal
tied-versus-square optimum-grid resolution. These are repaired before another
mechanism is scored. The complete correction ledger is
`reference_outputs/aggregate_conditioned_replay_control_20260730/postrun_audit.md`.

The replay taper `R(a)^(-q)` is rejected: every positive exponent worsened both
300M targets and both StarCoder protocols. The next frozen candidate replaces
the nearly collinear generic phase costs with the tied response's own second
directional curvature:

```
L(a, delta) = A(a)
            + xi grad A(a)^T delta
            + eta delta^T H_A(a) delta,
```

where `A` is fitted only on tied policies and `xi, eta >= 0`. The Hessian term
is a nonnegative second-order Taylor cost under the frozen convex aggregate
basis; it is not a sign-changing response and is not a deployment regularizer.
It adds one phase amplitude and no per-bucket phase parameters. The candidate,
data roles, and rejection gate are frozen in
`reference_outputs/aggregate_hessian_control_20260730/preregistration.md`.

## WSD80-SUR-022: Hessian rejection and late-reactivation preregistration

**2026-07-30.** Aggregate-Hessian control is rejected. On the repaired 300M
panel it worsens Uncheatable asymmetric RMSE by 9.2% while improving Table-9 by
3.2%. Its StarCoder raw optimum moves to the opposite boundary, `(0.62, 0)`,
and the observed policy near that coordinate is about 0.267 BPB worse than
predicted. Curvature remains nonzero across folds, so this is a response-law
failure rather than coefficient collapse. Full results and the independent
Claude Opus 5 review synthesis are in
`reference_outputs/aggregate_hessian_control_20260730/report.md`.

The review proposed a late-presence floor. A zero-cost 300M falsification
blocks the literal universal form: 63/238 asymmetric policies omit a bucket
late, but omitted mass is nearly uncorrelated with first-order residual and the
effect changes sign by bucket and target. Free bucket-specific floor
coefficients are not admissible under this evidence.

The materially new follow-up is a one-epoch late-reactivation state. It uses
`g(e) = e / (1 + e)` on actual late materialized epochs, normalizes the state
to equal aggregate exposure when tied, and prices its nonlinear departure with
the tied benefit response's Bregman divergence. It adds one amplitude, no phase
shape grid, and no per-bucket phase coefficients. The equation and gate are
frozen in
`reference_outputs/late_reactivation_control_20260730/preregistration.md`.

Evidence priority is now explicit. WSD80 tests two-phase representability; the
300M / 6B-token panel is the mandatory 39-bucket gate (TPP 29.83 total, 58.45
non-embedding); Delphi 3e18 is secondary low-TPP transfer evidence (TPP 4.40
total, 12.27 non-embedding) and cannot excuse failure on 300M.

## WSD80-SUR-023: late-reactivation control rejected

**2026-07-30.** The late-reactivation state is rejected. It improved
Uncheatable asymmetric-policy RMSE from 0.009402 to 0.008968 and reduced the
WSD random-fold RMSE from 13.04 to 11.05 seed standard deviations, but the
mechanism failed the diagnostics it was intended to repair:

- 300M Table-9 asymmetric RMSE worsened from 0.018696 to 0.019366.
- Paired phase-delta rank, sign accuracy, and absolute bias worsened on both
  300M targets.
- WSD blocked-region RMSE rose from 51.93 to 70.05 seed standard deviations.
- The WSD raw optimum moved to `(0, 0.625)` rather than toward the observed
  `(0.10, 0.50)`.
- The `p1=0` edge improved substantially, but the error moved to the opposite
  `p0=0` edge and median WSD error worsened.

The coefficient remained active, so this is not a collapsed ablation. The
state is one-sided: it prices late reactivation but cannot price early
omission. Independent Claude Opus 5 review found no implementation bug and
agreed that the high-TPP paired diagnostics and blocked-region failure are
decisive. Exact tables and the review synthesis are in
`reference_outputs/late_reactivation_control_20260730/report.md`.

## WSD80-SUR-024: repaired retained-power-law baseline screen

**2026-07-30; frozen before scoring.** Before introducing another mechanism,
the existing retained-power-law model is being evaluated under the repaired
protocol. Its previous WSD result was promising, but it has not been scored on
the required 520-row 300M panel. The screen uses the existing equation, shape
grid, robust constrained head, and hierarchical ridge without retuning.

The staged gate requires correct WSD blocked geometry and a stable interior
raw optimum before paying for the 300M nested audit. It then requires no
regression from the first-order control on either 300M target's asymmetric
RMSE, paired bias, or phase-gain sign accuracy. The frozen numerical gate is in
`reference_outputs/rpl_repaired_baseline_screen_20260730/preregistration.md`.

The shape search is process-parallelized only for runtime. A spawned-process
test confirms bit-identical shape scores to serial execution; folds,
hyperparameters, and the estimator are unchanged.

## WSD80-SUR-025: high-TPP evidence hierarchy and Compact aggregate rejection

**2026-07-30 05:03 PDT.** The evidence hierarchy is now fixed for subsequent
model development:

1. WSD80 tests whether a surrogate can represent genuine two-phase geometry.
2. The 300M / 6B-token original panel plus qsplit240 exposure-average
   ablation is the mandatory high-TPP 39-bucket gate.
3. Delphi 3e18 is a secondary low-TPP transfer/null check, not a selector for
   the phase mechanism.

The compute regimes support this ordering. Delphi 3e18 has total/non-embedding
TPP 4.40/12.27. The 300M / 6B-token setting has 29.83/58.45. In the fixed-model
WSD80 ladder, the net two-phase advantage grows from 0.010072 BPB at total TPP
6.35 to 0.020620 BPB at TPP 50.79. TPP is not established as the sole causal
moderator, but low Delphi TPP is a plausible reason to expect weak phase signal
there and cannot excuse failure at 300M.

The 300M panel itself contains strong, balanced phase evidence: 238 exact
aggregate-matched asymmetric/tied pairs. Their observed pair-difference
RMS is 0.017990 BPB on Uncheatable and 0.036104 on Table-9, about 10.7 and 7.3
times the approximate paired repeat-noise scales. The asymmetric arm is better
in 49.2% and 51.7% of pairs respectively. This is not a class-imbalance or
absent-signal panel.

Command:

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_compact_tied_backbone_20260730.py
```

The independently restricted Compact Retained State aggregate backbone is
rejected before any new phase mechanism:

| target | tied OOF RMSE | gate | median fold/full raw-optimum L1 | gate |
|---|---:|---:|---:|---:|
| Uncheatable | 0.006691 | 0.0056 | 0.160932 | 0.05 |
| Table-9 | 0.013832 | 0.0125 | 0.189032 | 0.05 |

The full raw optimum is also severely overoptimistic: 0.891253 BPB predicted
against 0.951105 best observed tied on Uncheatable, and 0.777547 against
0.982774 on Table-9. An earlier WSD OOF-RMSE gate was invalid because it
compared nested OOF predictions with incumbent in-sample residuals; it is
explicitly invalidated in the artifact. WSD shape and raw-optimum diagnostics
remain descriptive.

Artifacts:

- `reference_outputs/compact_tied_backbone_audit_20260730/preregistration.md`
- `reference_outputs/compact_tied_backbone_audit_20260730/report.md`

Decision: CRS fixed sparse-face collapse but replaced it with a flat,
fold-unstable aggregate optimum. It is not an eligible backbone for the next
phase-mechanism iteration.

## WSD80-SUR-026: corpus-scaled inverse-power curvature rejected

**2026-07-30 05:41 PDT.** Tested one materially new aggregate invariant:

```
k_i = c0_i + c1_i
k_g = geometric_mean(k)
alpha_i = alpha_0 * (k_i / k_g)^nu
L(a) = b + sum_i A_i (a_i + E0)^(-alpha_i)
         + sum_i B_i (k_i a_i)^g
```

Positive `nu` gives smaller finite pools faster diminishing-return curvature
in mixture-share space; `nu=0` is the exact shared-exponent ablation. The
screen used only the 20 WSD80 tied policies. It did not read any asymmetric
300M outcome or Delphi row.

The first execution exposed a protocol defect under Claude Opus 5 review:
singleton-domain WSD geometries had no bucket-departure columns, so the
hierarchical penalty vector was all zero and the advertised ridge was inert.
The initial report is preserved. Before rerunning, the repair was frozen:
directly ridge singleton amplitudes, use realized phase fraction 3040/3814,
persist all outer-fold shape choices, and require absolute improvement over
the existing CRS nested OOF comparator.

Command:

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/screen_corpus_scaled_deficit_tied_backbone_20260730.py
```

Repaired result:

| protocol | nu=0 RMSE | candidate RMSE | CRS comparator | selected nu | raw tied optimum | decision |
|---|---:|---:|---:|---:|---:|---|
| random | 0.131878 | 0.061591 | 0.087592 | 0.25 | 0.2622 | pass |
| blocked | 0.521965 | 0.130550 | 0.099032 | 0.50 | 0.2394 | reject |

The random arm supports a real interior corpus-scale effect, with fold-selected
`nu` values 0.25, 0.25, and 0.50. The route nevertheless fails the mandatory
coordinate-blocked absolute gate and misses the measured tied optimum at 0.30
by 0.0606, beyond the frozen 0.05 tolerance. The conditional 300M tied gate
was therefore not run.

Claude Opus 5 agreed that the blocked absolute comparison alone is decisive.
It also retracted a proposed cross-rung backbone-shape fit after inspection:
the WSD token ladder deliberately keeps simulated epochs fixed and contains
only six tied coordinates per rung, so it adds replication rather than tied
shape support. The already observed ladder instead establishes a model-free
phase trend: phase gain strengthens with token budget.

Artifacts:

- `screen_corpus_scaled_deficit_tied_backbone_20260730.py`
- `reference_outputs/corpus_scaled_deficit_tied_screen_20260730/preregistration.md`
- `reference_outputs/corpus_scaled_deficit_tied_screen_20260730/repair_preregistration.md`
- `reference_outputs/corpus_scaled_deficit_tied_screen_20260730/report.md`
- `reference_outputs/corpus_scaled_deficit_tied_screen_20260730/initial_report_pre_review.md`

Decision: block the corpus-scaled exponent route. Do not reopen it by widening
the `nu` grid or freeing per-bucket exponents. Resume only with a materially
new latent state or response invariant that addresses coordinate-held-out
geometry.

## WSD80-SUR-027: pointwise-authority saturation blocked by design support

**2026-07-30 06:11 PDT; zero-outcome gate.** Tested whether the first-order
aggregate-control coordinate could support a bounded odd response:

```
a = beta0 * w0 + beta1 * w1
delta = w1 - w0
u = grad A(a)^T delta
c = max_i grad_i A(a) - min_i grad_i A(a)
S_tau = tau * c * tanh(u / (tau * c))
L = A(a) + xi * S_tau + gamma * I_beta + zeta * J.
```

`tau = infinity` is the exact linear-control ablation. The finite grid
`{0.25, 0.5, 1, 2}` was frozen before the design audit. No candidate outcomes
were scored because the preregistered Tier-0 support gate failed:

| panel | p95 `abs(u)/c` | positive `u` | negative `u` |
|---|---:|---:|---:|
| WSD80 | 0.7979 | 38.3% | 61.7% |
| 300M Uncheatable | 0.0924 | 15.5% | 84.5% |
| 300M Table-9 | 0.1039 | 19.3% | 80.7% |

The frozen requirements were p95 at least 0.3 and both signs in at least 25%
of asymmetric rows. The nominal tied-zero tolerance was also too strict for
the fitted numerical gradient (`7.3e-12` against `1e-12`), but that does not
affect the decisive magnitude and sign failures. Future exact-zero checks use
a relative `1e-9` numerical tolerance.

At the observed 300M control magnitudes, even the smallest finite `tau` changes
the odd term by only a few percent at the extreme row and much less typically.
The nonlinear response is therefore not identified. Do not reopen this route
by widening the `tau` grid or changing `f(u)`.

Command:

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_saturating_phase_control_20260730.py
```

Artifacts:

- `reference_outputs/saturating_phase_control_20260730/preregistration.md`
- `reference_outputs/saturating_phase_control_20260730/design_screen.csv`
- `reference_outputs/saturating_phase_control_20260730/design_gate.json`

## WSD80-SUR-028: large phase movement, poor scalar-coordinate alignment

**2026-07-30 06:11 PDT; zero-outcome gate.** Separated transported phase mass
from alignment with the tied aggregate gradient:

```
m = 0.5 * ||delta||_1
q = |grad A(a)^T delta| / (m * (max(grad A) - min(grad A))).
```

The 300M policies are not timid. Median transported mass is 0.510 and the 95th
percentile is 0.612. Instead, their high-dimensional phase contrasts are mostly
orthogonal to the scalar gradient coordinate:

| panel | median `q` | p95 `q` |
|---|---:|---:|
| WSD80 | 1.0000 | 1.0000 |
| 300M Uncheatable | 0.0708 | 0.1785 |
| 300M Table-9 | 0.0743 | 0.1994 |

Also screened the proposed gradient-orthogonal scalar
`s = ||g|| * a^T delta_perp / ||a||`. It is identically zero in the two-domain
WSD80 geometry. In 300M its p95 normalized magnitude is only 0.135/0.159 and
its sign is negative on 99.2%/98.7% of asymmetric rows, failing both support
gates without reading outcomes.

This is an identification result, not a negative statement about 300M phase
signal. The panel contains 238 exact aggregate-matched pairs whose phase-delta
RMS is 10.7 and 7.3 times approximate repeat noise. The proposed scalar states
discard most of that high-dimensional signal.

Independent review added two important qualifications. First, `q = 1` in
WSD80 is forced by its one-dimensional tangent space, so WSD80 supplies no
alignment evidence. Second, `q` near 0.07 may be the generic
`O(1 / sqrt(d))` consequence of projecting a 38-dimensional contrast onto one
direction rather than a defect in the 300M sampling design. The next
zero-outcome audit therefore compares observed alignment with a matched
permutation null. The inherited 0.3 magnitude threshold is also not
transferable to the unbounded orthogonal scalar; that channel remains rejected
because it is exactly zero in WSD80 and has only 0.8%/1.3% positive values in
300M.

Command:

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_phase_control_alignment_20260730.py
```

Artifacts:

- `reference_outputs/phase_control_alignment_20260730/report.md`
- `reference_outputs/phase_control_alignment_20260730/summary.csv`
- `reference_outputs/phase_control_alignment_20260730/rows.csv`

Decision: reject `a^T delta_perp` and arbitrary fixed scalar projections. The
next candidate must retain retained-power-law's WSD80 interaction geometry and
use the 39-bucket contrast structure through a low-dimensional mechanistic
state, not through free per-bucket phase coefficients.

## WSD80-SUR-029: non-projective phase-functional survey blocks concentration

**2026-07-30; zero-outcome gate.** Independent review of WSD80-SUR-028
recommended testing whether the low 300M gradient alignment was generic in 38
tangent dimensions and surveying bucket-summed functionals before fitting
another response. The protocol was frozen in
`reference_outputs/phase_functional_design_survey_20260730/preregistration.md`.
No asymmetric outcome was read.

The geometric null confirms that the observed 300M alignment is modestly above,
not below, a random Gaussian tangent direction:

| panel | observed median `q` | null median `q` | observed p95 | null p95 |
|---|---:|---:|---:|---:|
| 300M Uncheatable | 0.0708 | 0.0470 | 0.1785 | 0.1363 |
| 300M Table-9 | 0.0743 | 0.0556 | 0.1994 | 0.1604 |

This supports the dimensionality interpretation. A single projection captures
little of a 38-dimensional contrast even when the panel is not unusually
misaligned. WSD80's `q = 1` remains an algebraic identity of its
one-dimensional tangent space.

The primary candidate was signed late-minus-early quadratic replay
concentration, gated by the norm of the tied aggregate gradient:

```
C(w) = sum_i (c0_i + c1_i) * w_i^2
Phi = C(w1) - C(w0)
kappa(a) = ||grad_T A(a)|| / ||grad_T A(a_proportional)||
h = kappa(a) * Phi.
```

It was required to be tied-zero, sign-balanced, below 0.5 absolute correlation
with first-order control, retain at least 20% residualized norm against
`{u, I_beta, J}`, and remain well-conditioned. It failed only the frozen
correlation clause:

| panel | `corr(h,u)` | residualized norm | condition number |
|---|---:|---:|---:|
| WSD80 | 0.8337 | 0.3773 | 11.72 |
| 300M Uncheatable | 0.6167 | 0.7828 | 2.12 |
| 300M Table-9 | 0.5224 | 0.8460 | 1.86 |

Entropy concentration and one-epoch overload were diagnostic alternatives, not
eligible substitutes. Each failed at least one mandatory panel. Per the frozen
protocol, no functional was outcome-fit and no threshold was moved after
inspection.

Command:

```
uv run experiments/domain_phase_mix/exploratory/two_phase_many/survey_phase_functional_design_20260730.py
```

Artifacts:

- `reference_outputs/phase_functional_design_survey_20260730/preregistration.md`
- `reference_outputs/phase_functional_design_survey_20260730/report.md`
- `reference_outputs/phase_functional_design_survey_20260730/functional_support.csv`
- `reference_outputs/phase_functional_design_survey_20260730/alignment_null.csv`

Decision: block phase-ordered concentration before outcome fitting. The
candidate is not numerically unidentified, but it does not supply the
independent mechanistic coordinate the preregistration required.

## WSD80-SUR-030: corrected functional audit permits only a generic coordinate

**2026-07-30; zero-outcome correction.** The WSD80-SUR-029 preregistration
contained two incompatible independence clauses: an absolute correlation
threshold and a residual-span threshold. The correlation clause is
structurally near-unpassable in the one-dimensional WSD80 tangent space and
also confounded the candidate with the aggregate-gradient magnitude. The
correction was frozen before recomputation in
`reference_outputs/phase_functional_independence_audit_20260730/preregistration.md`.
It uses one binding criterion: residual norm after projecting the raw
functional against normalized first-order control, phase information, and the
replay Jensen gap.

The quadratic functional cleared the corrected numerical gate:

| panel | control correlation | residual norm | deployed condition |
|---|---:|---:|---:|
| WSD80 | 0.6328 | 0.6658 | 11.72 |
| 300M Uncheatable | 0.5907 | 0.8032 | 2.12 |
| 300M Table-9 | 0.4978 | 0.8616 | 1.86 |

However, the fixed functional basis is effectively one-dimensional:
quadratic-versus-overload correlation is 0.999999 on WSD80 and 0.979925 on
300M; stable rank is 1.319 and 1.089. The corrected audit therefore permitted
only a generic phase-concentration coordinate, not identification of replay,
overload, or entropy as the mechanism.

Command:

```text
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_phase_functional_independence_20260730.py
```

Artifacts:

- `reference_outputs/phase_functional_independence_audit_20260730/report.md`
- `reference_outputs/phase_functional_independence_audit_20260730/corrected_functional_support.csv`
- `reference_outputs/phase_functional_independence_audit_20260730/basis_diagnostics.csv`
- `reference_outputs/phase_functional_independence_audit_20260730/cc_review.md`

Independent Opus 5 review verified the implementation and corrected its own
earlier claim that deconfounding explained the 300M pass. It did not:
correlation moved only from 0.6167 to 0.5907 and from 0.5224 to 0.4978. The
pass comes from selecting the preregistered residual-span criterion as the
single coherent criterion. Future preregistrations must state exactly one
binding independence test.

## WSD80-SUR-031: unequal-phase parity blocks phase concentration

**2026-07-30; algebraic rejection before outcome fitting.** Before fitting the
permitted coordinate, substitute the fixed-aggregate parameterization

```text
w0 = a - beta1 * delta
w1 = a + beta0 * delta
```

into the proposed quadratic concentration difference:

```text
Phi_raw = C(w1) - C(w0)
        = 2 * sum_i k_i a_i delta_i
          + (beta0 - beta1) * sum_i k_i delta_i^2
        = Phi_odd + Phi_even.
```

For 80/20 phases, the even coefficient is 0.6. `Phi_raw` is therefore not a
pure phase-order coordinate. One free-signed coefficient would tie an odd
ordering effect to a nonnegative dispersion cost at a ratio set by the phase
schedule, and a negative fitted sign would reward increasing asymmetry.

An outcome-free decomposition shows why the raw term appeared eligible:

| panel | term | positive / negative | residual norm |
|---|---|---:|---:|
| WSD80 | raw | 59.5% / 40.5% | 0.666 |
| WSD80 | odd | 58.9% / 41.1% | 0.644 |
| WSD80 | even | 100% / 0% | 0.006 |
| 300M | raw | 31.5% / 68.5% | 0.803 / 0.862 |
| 300M | odd | 16.8% / 83.2% | 0.718 / 0.772 |
| 300M | even | 100% / 0% | 0.206 / 0.205 |

The strictly positive even component flips 35 of 238 300M policies across
zero, creating the raw term's apparent sign balance. The clean odd term fails
the frozen 25% two-sided-support clause on both mandatory 300M targets. The
even term is 99.4% explained by existing WSD80 cost columns and adds no
independent geometry there.

Artifacts:

- `reference_outputs/phase_functional_independence_audit_20260730/cc_odd_even_review_prompt.md`
- `reference_outputs/phase_functional_independence_audit_20260730/cc_odd_even_review.md`

Decision: retract the provisional WSD80-SUR-030 recommendation and block the
bucket-summed concentration route without reading candidate outcomes. This is
an algebraic rejection, not test-set selection. Every future phase functional
must have parity derived explicitly under fixed-aggregate contrast reversal
before interpreting sign balance or fitting a response.

The evidence hierarchy remains:

1. WSD80 for genuine two-phase representability and optimum geometry.
2. The 300M / 6B-token 520-row matched panel as the mandatory high-TPP
   39-bucket gate.
3. Delphi 3e18 only as a secondary low-TPP transfer check.

The exact 300M staged-control baseline uses 238 physically asymmetric pairs:

| target | pair RMSE | pair bias | sign accuracy | constant-sign floor | asymmetric RMSE |
|---|---:|---:|---:|---:|---:|
| Uncheatable | 0.0092435 | -0.0003236 | 201/238 | 50.8% | 0.0094017 |
| Table-9 | 0.0204596 | -0.0002832 | 200/238 | 51.7% | 0.0186957 |

This supersedes stale artifact rows that counted the two physically tied
proportional and UniMax controls as paired asymmetric observations.

## WSD80-SUR-032: Fisher phase-information cost is a blocked reopening

**2026-07-30; provenance and mechanism review before outcome fitting.** The
next proposed minimal extension was retained power law plus one nonnegative
Fisher phase-information cost:

```text
L_new = L_RPL + gamma * I_beta(a, delta)
I_beta = 0.5 * beta0 * beta1 * sum_i delta_i^2 / a_i
gamma >= 0.
```

The term is algebraically clean: dimensionless, tied-zero, nonnegative, and
even under fixed-aggregate reversal. It is nevertheless not a new mechanism.
It is exactly the second-order Taylor approximation to the phase-label
Jensen-Shannon information already rejected as approach-registry route AB.
That transfer audit froze each panel's strongest available base and fit only
the nested amplitude; the coefficient collapsed on WSD80 and 300M
Uncheatable and was unstable on 300M Table-9.

An outcome-free redundancy screen also works against reopening:

| panel | correlation with RPL concentration | residual norm | condition |
|---|---:|---:|---:|
| WSD80 | 0.9409 | 0.3387 | 5.73 |
| 300M | 0.0548 | 0.9985 | 1.06 |

The 300M result establishes only independence from one concentration column,
not from the full RPL phase design. More importantly, this even term provides
no directional information. It can mechanically reduce RPL's negative paired
bias by shifting every asymmetric prediction upward, but that is a parity-based
bias correction rather than evidence for a training mechanism and can worsen
pair-sign accuracy.

Independent Opus 5 review reached the same decision and noted that KL, JS,
chi-square, Hellinger, and Fisher quadratic distances are the same local
information geometry. Changing divergence or base surrogate is not a
materially new route.

Artifacts:

- `reference_outputs/phase_functional_independence_audit_20260730/cc_rpl_fisher_cost_review_prompt.md`
- `reference_outputs/phase_functional_independence_audit_20260730/cc_rpl_fisher_cost_review.md`

Decision: **block without outcome fitting**. Reopen only with a non-even
invariant, a genuine dynamic switch-debt state, or a new outcome-free
identification argument against the full phase design plus an explanation for
the prior coefficient collapse.

## WSD80-SUR-033: family-pooled retained response is blocked before outcomes

**2026-07-30; outcome-free response-law audit.** RPL applies diminishing
returns bucketwise and then pools:

```text
B_post_f = sum_i (s_i + e0)^(-p).
```

The candidate instead applied the same power law to total retained semantic
family evidence:

```text
B_pre_f = (sum_i s_i + |f| e0)^(-p).
```

It added no columns, amplitudes, nonlinear axes, or family assignments.
Bucket residuals and repetition damage remained bucket-specific. WSD80 has
singleton families, so this candidate was required to be exactly RPL there;
all 540 shipped shapes passed that invariant on all 346 rows.

The candidate failed both binding 300M outcome-free gates:

| gate | frozen threshold | measured |
|---|---:|---:|
| minimum residual after projection on the full RPL benefit block | >= 0.20 | 0.1025 |
| maximum nonzero standardized condition | < 1e4 | 127,839 |

Median projection residual was 0.2341, but the admissible shape grid contains
low-exponent, large-offset shapes where pre-pooling is mostly representable by
the existing family-plus-bucket block. Per-family minima were 0.1309,
0.1225, and 0.0893. The new block therefore does not define a uniformly
independent response coordinate and worsens an already rank-deficient design.

The implementation also clarified an ordering-derivative issue before any
outcome fit. Under the retained-state transition,

```text
ds_i / d delta_i at tied
  = lambda beta0 a_i + beta0 beta1 (eta - 1).
```

Consequently, a within-family contrast whose raw entries sum to zero can
still change total retained family state. The legacy RPL ordering feature is
only an approximate residual, not the exact derivative of either family
response. It was held fixed in both models to keep this a single-axis test.

Commands:

```text
uv run pytest -q \
  experiments/domain_phase_mix/exploratory/two_phase_many/test_family_pooled_retained_power_law_20260730.py
uv run \
  experiments/domain_phase_mix/exploratory/two_phase_many/audit_family_pooled_retained_power_law_20260730.py
```

Artifacts:

- `reference_outputs/family_pooled_retained_power_law_20260730/preregistration.md`
- `reference_outputs/family_pooled_retained_power_law_20260730/outcome_free_audit.md`
- `reference_outputs/family_pooled_retained_power_law_20260730/condition_by_shape.csv`
- `reference_outputs/family_pooled_retained_power_law_20260730/projection_by_benefit_shape.csv`
- `reference_outputs/family_pooled_retained_power_law_20260730/cc_ordering_derivative_review.md`

Decision: **block without fitting BPB outcomes**. The earlier invalid
wrong-estimator pre/post-pooling comparison remains invalid, but a corrected
retest is not warranted because the candidate fails the preregistered
mechanism-identification gate first.

## WSD80-SUR-034: centered hierarchical RPL does not survive shape pinning

**2026-07-30; mandatory high-TPP 300M gate.** The centered hierarchy shrinks
bucket response coefficients toward their semantic-family means instead of
toward zero. WSD80 has singleton families, so this is exactly RPL there and
cannot be selected by the two-bucket panel.

The initial nested screen allowed the centered and ordinary heads to choose
different nonlinear configurations. A frozen 2 x 2 follow-up therefore crossed
both heads with both selected configurations. This isolates the linear-head
prior from nonlinear-shape selection:

| Target | Configuration | RPL pair RMSE | Centered pair RMSE | Pair change | Overall change |
|---|---|---:|---:|---:|---:|
| Uncheatable | RPL-selected | 0.008640 | 0.008641 | +0.01% | +0.34% |
| Uncheatable | centered-selected | 0.008884 | 0.008567 | -3.56% | -2.22% |
| Table-9 | RPL-selected | 0.019595 | 0.018854 | -3.78% | +0.63% |
| Table-9 | centered-selected | 0.018696 | 0.018423 | -1.46% | -0.92% |

No target improves pair RMSE by at least 2% under both pinned
configurations. The apparent nested benefit therefore depends on
selection-mediated shape changes rather than a stable centered prior.

Artifacts:

- `reference_outputs/centered_hierarchical_rpl_physical_20260730/pinned_2x2_report.md`
- `reference_outputs/centered_hierarchical_rpl_physical_20260730/pinned_2x2.csv`

Decision: **reject the centered-prior route**. Do not run more seeds.

## WSD80-SUR-035: high-TPP hierarchy is binding, not explanatory rhetoric

**2026-07-30.** The original two-phase panel plus qsplit240 ablation contains
282 physically tied rows and 238 asymmetric rows. The latter have 238 exact
aggregate-matched tied counterparts. Pair-difference RMS is 0.017990 BPB on
Uncheatable and 0.036104 on Table-9, about 10.7 and 7.3 times paired
repeat-noise scale. The asymmetric arm is better in 49.2% and 51.7% of pairs.

Consequences:

- there is no useful class-imbalance or absent-phase-signal explanation for a
  failed 300M model;
- phase mechanisms are selected on exact paired deltas, while the aggregate
  response is selected on tied rows;
- WSD80 remains the low-dimensional representability and optimum-geometry
  gate;
- Delphi 3e18 remains a secondary low-TPP transfer diagnostic and cannot veto
  an otherwise successful high-TPP phase mechanism merely through weak phase
  fit.

This hierarchy is now reflected in the living success metrics at the top of
this logbook and in Fieldbook note `note_01kysgrb63bqx0j6z48qx372sh`.

## WSD80-SUR-036: signed family mediation deletes rather than shares contrast

**2026-07-30; outcome-free rejection.** The candidate decomposed each
within-family contrast into a bucket-specific residual plus its
aggregate-weight-proportional family mean,

```text
D_f = sum_{i in f} delta_i
u_i(q) = (1 - q) delta_i + q (a_i / sum_{j in f} a_j) D_f.
```

Only RPL's survival gate used `u`. It was exact at `q=0`, exact when tied,
family-total conserving, and exactly RPL on singleton-family WSD80. At `q=1`,
however, the mediated/original state-norm ratio was only 0.227 in median. The
31-bucket family retained 0.080 of the original contrast norm, the 6-bucket
family 0.395, and the 2-bucket family 0.855. The route therefore removes most
of the high-dimensional phase state through cancellation rather than pooling
it.

Artifacts:

- `reference_outputs/semantic_family_mediated_retention_20260730/preregistration.md`
- `reference_outputs/semantic_family_mediated_retention_20260730/outcome_free_audit.md`

Decision: **block before reading outcomes**. A global RMS rescaling was not
adopted because it would break family-total conservation and introduce an
arbitrary contrast gain.

## WSD80-SUR-037: coarse family churn restates global divergence

**2026-07-30; mechanism-provenance rejection.** The candidate charged a
within-family Hellinger churn hazard,

```text
H_f = 2 (sqrt(W0_f W1_f) - sum_{i in f} sqrt(w0_i w1_i)),
gate_i = retention * delta_i - gamma H_f.
```

It was active, tied-zero, and numerically regular. The coarse 31/6/2 family
partition nevertheless assigns a median 0.3453 of total 0.4158 churn to the
31-bucket family, and 95.2% of global Hellinger lies within the partition.
Total coarse churn is 92.5% explained by concentration, TV, and global
Hellinger. In the one-family limit this mechanism is exactly the global
Hellinger divergence already rejected after coefficient collapse and
selection regression.

Independent Opus 5 review also noted that a nonnegative even hazard can appear
to help by correcting mean asymmetric-policy bias. Any future even cost must
improve de-biased pair rank and sign, not merely pooled RMSE.

Artifacts:

- `reference_outputs/family_churn_hazard_rpl_20260730/preregistration.md`
- `reference_outputs/family_churn_hazard_rpl_20260730/outcome_free_audit.md`

Decision: **block without outcome fitting** as a coarse reparameterization of
an exposed divergence route.

## WSD80-SUR-038: quality-pair churn fails the frozen Stage-0 sign test

**2026-07-30; Uncheatable only, before any fitted hazard amplitude.** A finer
candidate used the 13 exact Dolma3 Common Crawl high/low quality pairs as
dynamic groups and left the other 13 buckets as singleton controls. Conditional
within-pair Hellinger churn was only 28.9% explained by concentration, TV, and
global Hellinger, was active on all 238 exact pairs, preserved tied and WSD80
invariants, and was numerically regular. This materially differed from the
coarse global-divergence proxy.

The frozen Stage-0 gate asked whether residualized quality-pair churn predicts
the incumbent RPL's observed-minus-predicted paired delta on Uncheatable. It
did not:

| Diagnostic | Result |
|---|---:|
| exact pairs | 238 |
| raw Spearman | -0.070 |
| residualized Spearman | -0.041 |
| paired-bootstrap 95% interval | [-0.163, +0.077] |
| probability correlation is positive | 0.238 |
| standardized slope | -0.000377 BPB per churn SD |

The sign is opposite the forgetting-hazard premise and the interval spans no
effect. Table-9 was not read, and `gamma` was never fitted.

Independent Opus 5 review, launched before Stage 0 ran, independently reached
the same block. It derived

```text
C_f = H_f / (2 sqrt(W0_f W1_f)),
```

so the fine conditional churn is the preceding family Hellinger churn divided
by pair mass. Its leading fixed-aggregate term is the within-pair conditional
Fisher quadratic; its odd action only attenuates RPL's existing ordering
signal and cannot introduce a new ordering direction. The review also caught
that ordinary phase swapping is not fixed-aggregate contrast reversal under
80/20 phases. These are mechanism-provenance problems, not reasons to amend
the already failed Stage-0 test.

Artifacts:

- `reference_outputs/quality_pair_churn_hazard_rpl_20260730/preregistration.md`
- `reference_outputs/quality_pair_churn_hazard_rpl_20260730/outcome_free_audit.md`
- `reference_outputs/quality_pair_churn_hazard_rpl_20260730/stage0_preregistration.md`
- `reference_outputs/quality_pair_churn_hazard_rpl_20260730/stage0_report.md`

Decision: **reject the quality-pair churn route at Stage 0**. The next
iteration changes estimation structure rather than inventing another
divergence feature.

## WSD80-SUR-039: exact-pair channel weighting is directionally useful but fails uncertainty

**2026-07-30; frozen high-TPP Stage 1.** This iteration left the RPL equation,
latent state, nonlinear shape, ridge, constraints, and parameter count fixed.
For each exact aggregate-matched pair it replaced the asymmetric absolute
equation with the phase-difference equation:

```text
aggregate: X(x_tied) theta = y_tied
phase:    [X(x_asym) - X(x_tied)] theta = y_asym - y_tied.
```

The 282 aggregate equations and 238 phase equations received equal total
weight, normalized to preserve the ordinary fit's total data weight and ridge
scale. Column scales came from the original absolute design, and Huber scales
were estimated separately by channel. Shapes and ridges were pinned to the
ordinary nested-RPL selections so the candidate could not win through a
selection-mediated configuration change.

| Target | Ordinary RMSE | Paired RMSE | Ordinary delta RMSE | Paired delta RMSE | Delta change | Bootstrap 95% interval | Regret change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uncheatable | 0.007236 | 0.007489 | 0.008640 | 0.008434 | -2.39% | [-1.39%, +6.32% improvement] | +0.000804 |
| Table-9 | 0.015065 | 0.014976 | 0.019595 | 0.019215 | -1.94% | [-0.83%, +4.66% improvement] | +0.001229 |

Delta rank, sign accuracy, and bias improve on both targets, so exact-pair
supervision is aligned with the phase task. The primary Uncheatable bootstrap
interval nevertheless includes zero, and fold Regret@1 worsens on both
targets. The frozen Stage-1 gate therefore fails.

The equal-channel generalized least-squares interpretation is also a poor
description of ordinary OOF residuals. It implies an aggregate/phase-innovation
variance ratio of 1.185 and within-pair correlation 0.736. Measured ratios are
0.585 and 0.478, and correlations are 0.348 and 0.227, on Uncheatable and
Table-9 respectively. A covariance-whitened estimator would require a new
preregistration and would move weight back toward the aggregate channel; it
cannot be treated as a free follow-up sweep.

WSD80 has zero asymmetric rows with an exactly observed tied aggregate under
the realized phase fractions. The estimator therefore has no WSD80 arm:
unchanged WSD80 behavior follows from leaving the model form untouched and is
not supporting evidence.

Independent Opus 5 review confirmed the gate failure and recommended against a
feasible-GLS successor. The two channel weights imply
`Cov(r_tied, r_asym) = Var(r_tied)` for every positive weighting, so changing
their ratio cannot reproduce the observed covariance. A full 2 x 2 covariance
could, but it would sit closer to ordinary RPL, should make the already
uncertain delta gain smaller, and would need roughly 620 rather than 238 pairs
to resolve an effect of this size at the frozen threshold. The review also
noted that the naive pair bootstrap conditions on three shared fitted models
and is therefore optimistic, and that changing the transformed Gram matrix
changes effective shrinkage even at fixed nominal ridge. This result rejects
the tested loss on this panel; it does not prove that all paired-channel losses
are ineffective.

Artifacts:

- `paired_channel_retained_power_law_20260730.py`
- `audit_paired_channel_retained_power_law_20260730.py`
- `reference_outputs/paired_channel_rpl_estimation_20260730/stage1_report.md`
- `reference_outputs/paired_channel_rpl_estimation_20260730/paired_bootstrap.csv`
- `reference_outputs/paired_channel_rpl_estimation_20260730/cc_result_review.md`

Decision: **reject equal-channel paired estimation as a headline improvement**.
It provides a useful estimation diagnostic, but it does not establish a
selection improvement and does not repair the retained-state response form.

## WSD80-SUR-040: hierarchical retention random slopes blocked before outcomes

**2026-07-30; mechanism and identification review only.** The proposed
successor replaced RPL's one global retention rate with centered
within-family rates,

```text
lambda_i = lambda exp(u_i),  sum_{i in family f} u_i = 0.
```

The transition remained exactly tied-neutral and reduced exactly to RPL for
singleton-family WSD80. It would have added 36 nominal random slopes for the
31/6/2 family partition, with effective degrees of freedom capped at 12 and
identified only from the 238 exact 300M phase pairs.

Independent Opus 5 review found the mechanism itself new but blocked the
estimator before outcomes:

- the local derivative is still a 36-dimensional per-bucket ordering field
  with a pinned smooth weight;
- the same-dimensional Fisher field previously required 28.1 effective
  degrees of freedom on Table-9;
- 238 pairs cannot resolve the frozen 2% improvement at the required
  uncertainty level;
- log-centering raises the arithmetic mean retention rate whenever deviations
  are nonzero, confounding heterogeneity with a shared-rate shift;
- identity to RPL on WSD80 is an exemption from, not evidence on, the new
  mechanism;
- the preregistered projection and parity gates were incomplete.

Artifacts:

- `reference_outputs/hierarchical_retention_random_slopes_20260730/preregistration.md`
- `reference_outputs/hierarchical_retention_random_slopes_20260730/cc_mechanism_review.md`

Decision: **block without fitting BPB outcomes**. Do not repair this into
another high-dimensional phase field. A successor must change the physical
contrast state with one or very few transferable parameters and remain active
on WSD80.

## WSD80-SUR-041: coverage-normalized epoch contrast blocked before outcomes

**2026-07-30; mechanism review only.** The candidate replaced RPL's raw
phase-weight contrast in the retention transition by

```text
r_i = k_i (w1_i - w0_i) / (1 + k_i a_i),
```

where `k_i` is simulated epochs per unit mixture weight and the fixed
denominator floor is one corpus pass. It added no fitted parameter and remained
exactly tied-neutral.

Independent Opus 5 review blocked the test before BPB outcomes. At the pinned
parent retention rate, the substitution multiplies the gate argument by about
27x at the median 300M bucket and by 2.5x--905x across buckets. The frozen grid
contains no scale-matched arm, so a measured difference would not identify
coverage normalization separately from gate steepness. The premise is also in
direct tension with the WSD80 token ladder: simulated epochs are held fixed
across rungs while the measured two-phase advantage grows from 0.010072 to
0.020620 BPB.

The review additionally found that the preregistered 300M gate was weaker than
the uncertainty and Regret@1 standard that rejected SUR-039. No outcome-free
design audit or BPB fit was run.

Artifacts:

- `reference_outputs/coverage_normalized_epoch_contrast_20260730/preregistration.md`
- `reference_outputs/coverage_normalized_epoch_contrast_20260730/cc_mechanism_review.md`

Decision: **reject this formulation without fitting outcomes**. Do not repair
it by sweeping the epoch floor or a contrast exponent. The next route should
use aggregate training state to moderate phase response and must keep WSD80
active while exploiting the 238 exact high-TPP 300M phase pairs.

## WSD80-SUR-042: aggregate-gradient plasticity does not transfer as one state

**2026-07-30; mechanism review only.** This route attempted to make staged
retained control vanish on a stationary tied fiber. It fit the aggregate
response `A(a)` on tied rows, formed the tangent-gradient RMS,

```text
s(a) = rms(grad_T A(a)) / median_tied rms(grad_T A),
h(a) = s(a) / (1 + s(a)),
```

and multiplied both retention and late-phase leverage by `h`. Retained benefit
was evaluated at the resulting latent state while physical repetition damage
remained at aggregate `a`. The construction was exactly tied-neutral and
encoded the fiber-optimality hypothesis.

Independent Opus 5 review verified those algebraic properties but blocked the
state before asymmetric outcomes. With two WSD80 buckets, the tangent-gradient
RMS is exactly a distance from tied stationarity. With 39 buckets, the
power-law derivative makes it a heavy-tailed statistic dominated by the
smallest-weight buckets; the 39 deletion policies force it toward one. The
model therefore does not transfer the same plasticity state between the two
mandatory panels and opens phase control most strongly near boundaries, where
three earlier controls selected pathological raw optima.

The review also found that:

- the benefit-only retained residual omits the damage derivative needed for
  the measured WSD80 phase-sign reversal;
- one nonnegative amplitude ties odd and even response at a fixed ratio;
- the one-column candidate-minus-parent change has stable rank one, below the
  preregistered Stage-0 minima by construction;
- the 2% high-TPP improvement gate remains underpowered at 238 pairs under the
  uncertainty standard that blocked SUR-039 and SUR-040.

Artifacts:

- `reference_outputs/aggregate_conditioned_retained_control_20260730/preregistration.md`
- `reference_outputs/aggregate_conditioned_retained_control_20260730/cc_mechanism_review.md`

Decision: **block without fitting asymmetric BPB outcomes**. Do not reopen by
changing the gradient norm, saturation function, or exponent. Those are
target-derived recalibrations rather than a new training-state mechanism.

## WSD80-SUR-043: cross-metric transfer is strong for code and fails for broad text

**2026-07-30; 346-coordinate local audit over 29 complete BPB targets.** The
Programming-Languages-selected RPL nonlinear shape was frozen while only the
linear target head was refit for every metric on the same checkpoints. A
second, nested pass independently selected the existing RPL shape grid inside
each outer training fold for nine representative targets. Boundary, interior,
lower-tail, optimum-neighborhood, and policy-selection diagnostics were kept
separate.

The frozen geometry transfers to the two independent GitHub code targets:

- Programming Languages: interior median absolute error 0.00340 BPB
  (0.73 pooled seed SD), optimum-neighborhood 0.00036 SD under the frozen
  reference, and Regret@1 0.00016 BPB. The frozen result is selected on this
  full target and is therefore structural reference evidence, not an honest
  primary-target OOF estimate; nested Regret@1 is 0.00284 BPB.
- GitHub Python: interior median absolute error 0.00358 BPB (0.81 SD),
  optimum-neighborhood 0.31 SD, and Regret@1 0.00485 BPB.
- GitHub C++: 0.00368 BPB (0.83 SD), 0.38 SD, and 0.00587 BPB.
- The full-fit code optima remain interior and predict two-phase gains of
  0.0064--0.0080 BPB versus observed sampled gains of 0.0096--0.0127 BPB.

The same frozen shape does not transfer to broad-text optimum selection:
C4 English and RefinedWeb have interior median errors 0.00515 and 0.00519 BPB,
but Regret@1 0.01292 and 0.01140 BPB. Their observed sampled two-phase gains
are zero while RPL predicts 0.01680 and 0.01605 BPB. Target-specific nested
shape selection improves some ordinary and lower-tail errors but does not
repair the decision surface; it predicts even larger full-fit gains for C4
and RefinedWeb, 0.02662 and 0.02641 BPB.

The failure is not licensed by accepting noisy boundaries. Sixteen of 29
frozen full-fit optima land at phase-0 StarCoder share zero with a small
positive phase-1 share; their median predicted gain is 0.01644 BPB while the
median observed sampled gain is zero. RPL's derivative-based phase block
contains low-aggregate factors proportional to
`(aggregate + offset)^-2` and `(aggregate + offset)^-3`; the latter asymmetry
columns are unpenalized. Prior repeated-anchor evidence still establishes
heteroskedasticity, especially at the StarCoder-only vertex, but these
unsupported raw optima and their interior counterparts are a structural
selection problem rather than merely large boundary residuals.

An explicit no-ordering ablation retained the retention gate, late multiplier,
aggregate response, and concentration gap while removing the derivative-based
ordering/asymmetry block. It severely worsened the code targets
(Programming-Languages Regret@1 0.00016 to 0.01728) but improved C4 English
(0.01292 to 0.00661), Dolma v1.5 (0.00850 to 0.00231), and RefinedWeb
(0.01140 to 0.00464). Retention alone still predicts unsupported gains, so the
failure is a coupled, objective-dependent phase-control problem rather than
one bad column.

Method caveats are explicit:

- the frozen Programming-Languages shape came from a full-panel selection;
- lower-tail and optimum-neighborhood memberships use observed outcomes and
  are descriptive, although predictions remain out of fold;
- pooled seed SD is measured from replicated coordinates and is not a local
  boundary-noise estimate;
- the 29 targets include correlated aggregates and are not 29 independent
  transfer experiments;
- optimum-neighborhood sample counts are much smaller for low-StarCoder broad
  optima than for the code optimum.

Decision: **retain RPL as a strong code-objective WSD80 model, but reject a
claim of objective-universal transfer.** Do not use pooled cross-metric RMSE or
boundary tolerance to justify its raw optima. The next clean falsification is
to regularize every phase-control column, select phase complexity against a
nested lower-tail/selection criterion rather than pooled RMSE, and test whether
the phase block shrinks toward zero on broad targets while surviving on all
three code targets.

Artifacts:

- `reference_outputs/wsd80_cross_metric_rpl_20260730/report.md`
- `reference_outputs/wsd80_cross_metric_rpl_20260730/cross_metric_surface_gallery.html`
- `reference_outputs/wsd80_cross_metric_rpl_20260730/retuned_cross_metric_parameters.csv`
- `reference_outputs/wsd80_cross_metric_rpl_no_ordering_ablation_20260730/report.md`
- `audit_wsd80_cross_metric_rpl_20260730.py`

## WSD80-SUR-044: freeze the two-phase surrogate north star

**2026-07-30; research-scope decision.** The invariant objective and evidence
hierarchy are now recorded in
`.agents/projects/two_phase_surrogate_north_star.md`.

The primary outcome is the observed performance of the target-specific policy
selected by the surrogate, not ordinary fit quality. Model discovery may use
the original 280-row high-TPP 300M two-phase panel plus all aggregate/tied
counterparts in the 520-row structured design. Recovering the result under a
fixed 280-total-row acquisition budget is a later sample-efficiency ablation,
not the current model-discovery constraint.

Existing baselines must be refit on the same expanded rows and folds before a
gain is attributed to model form. WSD80 remains the representability gate, the
high-TPP 39-bucket panel is the primary identification gate, and a newly sealed
3e18 panel is the final policy-selection gate. The scientific stretch target
remains an approximately 0.01 BPB two-phase advantage over the comparable
optimized one-phase policy, but the model must not manufacture unsupported
headroom.

Decision: **treat the north-star charter as binding across compactions and
handoffs**. Failed candidates may update the hypothesis queue, but they do not
silently weaken the acceptance criteria.

## WSD80-SUR-045: enforce mechanism provenance across handoffs

**2026-07-30; process audit.** The append-only WSD80 logbook prevented local
repetition but was not linked to the 99-route registry from the July 19
mechanistic-surrogate drive. The July 29 handoff named only this logbook and
`AGENTS.md`. This allowed prior mechanisms to re-enter without an explicit
reopening argument.

The audit found one direct reopening and two undocumented partial reopenings:

- SUR-020's Fisher phase-information cost is the local quadratic form of
  historical route `prior_AB`. SUR-032 eventually caught this.
- SUR-015 grafted an acquisition-and-forgetting ODE onto RPL without citing
  the related `prior_A`, `prior_J`, and `PLAFK` failures.
- SUR-019 refactored exposed replay and finite-corpus mechanisms without
  citing `prior_B`, `prior_T`, `prior_W`, `prior_Y`, `prior_AH`, `RMR`,
  `FSCR`, or `JARA`.

The current candidate series is now backfilled in
`.agents/projects/two_phase_surrogate_active_registry.csv`. The full audit is
`.agents/projects/two_phase_surrogate_anti_circularity_audit.md`. The
north-star charter and July 29 handoff now require both registries before
candidate derivation.

Decision: **no new outcome fit may run without an active-registry row**. The
row must name the nearest prior routes, material novelty, why the old rejection
does not apply, and the cheapest falsification. Reparameterization or a new
base surrogate alone is not a new route.

## WSD80-SUR-046: identifiable penalized RPL head preregistered

**2026-07-30 20:29 PDT; estimator repair frozen before candidate outcomes.**
The expanded high-TPP 300M baseline protocol is
`e30c84f654eb55e9d428eb9ee1afeac69a111d629abe45de6f96eb81db026185`.
It contains exactly 520 observations, 280 independent
`phase_correspondence_key` groups, 282 physically tied observations, 238
asymmetric observations, and 238 exact aggregate-matched contrasts. Both outer
and nested inner folds treat the correspondence key as indivisible. Eighteen of
twenty baseline cells were complete when this entry was written; the two parent
retained-power-law cells were still running and had not been inspected.

The preregistered repair leaves RPL's retained-share state, transition, benefit
and repetition response, nonlinear shape grid, and admissible response span
unchanged. It changes only the estimator:

```text
parent signed pair:  b+ x + b- (-x),  b+, b- >= 0
repaired head:       gamma x,          gamma unrestricted
```

Every signed phase pair is collapsed to one identified coefficient. Aggregate
columns retain max-absolute fold scaling; phase controls use training-fold RMS
scaling. Family amplitudes remain free, bucket departures remain ridge-shrunk,
and every phase-control coefficient is ridge-shrunk. Aggregate amplitudes stay
nonnegative while phase coefficients are signed.

Nested selection is frozen as:

1. retain candidates within `1.05 * min(all-RMSE)`;
2. among those, retain candidates within `0.002 BPB` of minimum asymmetric
   Regret@1;
3. minimize asymmetric lower-tail RMSE, then exact-pair delta RMSE,
   asymmetric RMSE, all-RMSE, and canonical grid order.

The final candidate protocol is
`a829181d36a9b3707b307bf802f81966905225304f94e6d6c4dc92ccb5838734`.
The correspondence-cluster bootstrap companion was separately frozen at
`9ee340ca38459994af38daecf477dc6468ae5a5a18b6ecb47ec74937836900c8`;
it resamples correspondence groups within outer folds and keeps exact contrasts
paired.

Outcome-free checks passed:

- direct column reconstruction proves zero numerical difference between the
  parent and collapsed response spans on representative shapes;
- all repaired phase-control columns are below `1e-9` on the 282 tied rows;
- the real-data nested smoke test produced six finite candidates over two
  shapes and three ridge values;
- the 238 pair indices and 280 correspondence groups are preserved;
- `pyrefly` and the repository pre-commit checks pass.

Exact commands:

```text
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_repaired_rpl_300m_20260731.py --prepare-only --optimizer-starts 8
uv run experiments/domain_phase_mix/exploratory/two_phase_many/bootstrap_expanded_300m_pareto_baseline_20260731.py --prepare-only
```

Decision: **active and preregistered**. Run both 300M targets only after the two
parent-RPL baseline cells release their workers. Reject the repair without
retuning if it fails the frozen high-TPP gate, fails WSD80 code representability,
or retains unsupported broad-text phase gain.

### WSD80-SUR-046 positive/negative control protocol

**2026-07-30 20:43 PDT; frozen before repaired-estimator outcomes.** The
target-specific WSD80 control harness is
`benchmark_repaired_rpl_wsd80_controls_20260731.py`, with protocol
`120228b2786b8b27e3577f98a30cf49a9603235d2f399c3536122c1d1d1aa422`.
It uses the same 346-coordinate surface as WSD80-SUR-043. The panel contains 20
tied coordinates, 326 asymmetric coordinates, and 255 asymmetric-to-tied
contrasts that exactly preserve the experiment's nominal 80/20 aggregate.
Physical epoch dynamics continue to use the realized 3040/3814 phase split;
the nominal split is used only to identify the designed fibers.

The three positive controls are Programming Languages, GitHub Python, and
GitHub C++. C4 English and Falcon RefinedWeb are negative controls against an
invented phase advantage. Each target is fitted independently with nested
target-specific shape selection under both random and mixture-blocked folds.
Random folds carry the frozen numerical gate because the exposed
target-specific parent-RPL comparison used that protocol; blocked folds are
mandatory transfer diagnostics and cannot rescue a failed random gate.

The frozen per-target random-fold checks are:

- interior RMSE no greater than 1.05 times target-specific parent RPL;
- code Regret@1 no greater than 0.006 BPB and predicted phase gain at least
  0.004 BPB;
- broad-text Regret@1 no greater than 0.005 BPB and predicted phase gain no
  greater than 0.005 BPB;
- Programming-Languages continuous interior optimum within Euclidean distance
  0.10 of the observed optimum.

Static audits passed: the PEP 723 entry point prepares from a clean dependency
environment, `pyrefly` and repository pre-commit checks pass, and a two-shape
smoke fit produced finite held-out predictions while preserving all 255
aggregate-matched contrasts. No full-grid repaired-head target result had been
run when these thresholds were recorded.

## Expanded 300M Pareto baseline completed

**2026-07-30 21:12 PDT; frozen pre-candidate evidence.** All 20 cells of the
expanded baseline completed under protocol
`e30c84f654eb55e9d428eb9ee1afeac69a111d629abe45de6f96eb81db026185`.
The correspondence-cluster bootstrap completed 4,000 draws under protocol
`9ee340ca38459994af38daecf477dc6468ae5a5a18b6ecb47ec74937836900c8`.
Both outer and nested inner folds keep each of the 280
`phase_correspondence_key` groups intact; the 238 exact aggregate-matched
contrasts remain paired under bootstrap resampling.

Hierarchical phase replay is the strongest eligible OOF baseline on both
targets:

| Target | Model | All RMSE | Asymmetric RMSE | Pair-delta RMSE |
|---|---|---:|---:|---:|
| Uncheatable | hierarchical phase replay | 0.006800 | 0.007896 | 0.007850 |
| Uncheatable | parent retained power law | 0.007503 | 0.008803 | 0.008984 |
| Table-9 | hierarchical phase replay | 0.013001 | 0.015345 | 0.016902 |
| Table-9 | parent retained power law | 0.014782 | 0.017166 | 0.019220 |

The cluster bootstrap excludes zero for every HPR-minus-parent-RPL difference:
Uncheatable differences are -0.000703, -0.000907, and -0.001134 BPB for
all-RMSE, asymmetric RMSE, and pair-delta RMSE; Table-9 differences are
-0.001781, -0.001822, and -0.002317 BPB. HPR-band remains reference-only and
does not materially improve the single HPR fit.

No baseline produces a defensible raw optimum. Parent RPL is the clearest
failure: its Uncheatable optimum has phase TV 0.985, max bucket weight 0.988,
nearest-policy TV 0.807, and predicted fiber gain 0.200 BPB; its Table-9
optimum has phase TV 0.955, max bucket weight 0.955, nearest-policy TV 0.795,
and predicted fiber gain 0.579 BPB. The full Uncheatable fit selects retention
10 and late multiplier 4, both grid maxima. HPR keeps phase TV near zero but
still optimizes the aggregate far outside empirical support, with
nearest-policy TV about 0.61 and implausibly low predicted BPB.

Decision: **the baseline is frozen and complete, but there is no incumbent
raw-policy solution.** OOF leadership and raw-optimum credibility remain
separate gates. Proceed with preregistered estimator repair `WSD80-SUR-046`
without changing its equation, grid, selector, or acceptance thresholds.

Artifacts:

- `reference_outputs/expanded_300m_pareto_baseline_20260731/report.md`
- `reference_outputs/expanded_300m_pareto_baseline_20260731/baseline_metrics.csv`
- `reference_outputs/expanded_300m_pareto_baseline_20260731/baseline_pair_metrics.csv`
- `reference_outputs/expanded_300m_pareto_baseline_20260731/baseline_raw_optima.csv`
- `reference_outputs/expanded_300m_pareto_bootstrap_20260731/report.md`
- `reference_outputs/expanded_300m_pareto_bootstrap_20260731/pairwise_differences.csv`

## WSD80-SUR-046: identifiable penalized RPL head rejected

**2026-07-30 22:07 PDT; frozen-candidate decision.** All two 300M targets and
all ten WSD80 control cells completed under the preregistered protocols. The
correspondence-cluster bootstrap resampled the 280
`phase_correspondence_key` groups within outer fold for 4,000 draws, preserving
the 238 exact aggregate-matched contrasts.

The estimator repair does not clear the primary 300M gate:

| Target | Model | All RMSE | Asymmetric RMSE | Pair-delta RMSE |
|---|---|---:|---:|---:|
| Uncheatable | hierarchical phase replay | 0.006800 | 0.007896 | 0.007850 |
| Uncheatable | parent retained power law | 0.007503 | 0.008803 | 0.008984 |
| Uncheatable | repaired retained power law | 0.007850 | 0.009258 | 0.009328 |
| Table-9 | hierarchical phase replay | 0.013001 | 0.015345 | 0.016902 |
| Table-9 | parent retained power law | 0.014782 | 0.017166 | 0.019220 |
| Table-9 | repaired retained power law | 0.015110 | 0.017322 | 0.019467 |

Relative to hierarchical phase replay, repaired RPL worsens all-RMSE by 15.4%
and 16.2%, asymmetric RMSE by 17.2% and 12.9%, and pair-delta RMSE by 18.8%
and 15.2% on Uncheatable and Table-9 respectively. Every one of these six
candidate-minus-HPR loss intervals excludes zero. On Uncheatable it also
worsens parent RPL significantly: all-RMSE `+0.000347`
(`95% CI [0.000183, 0.000504]`), asymmetric RMSE `+0.000455`
(`[0.000215, 0.000696]`), and pair-delta RMSE `+0.000344`
(`[0.000078, 0.000619]`). Table-9 Regret@1 has a better point estimate than
parent RPL, but its paired interval crosses zero and only matches HPR's point
estimate. There is no bootstrap-supported phase-sensitive improvement.

The repaired model has 99 nominal parameters. Its outer-fold active-set proxy
ranges from 69 to 81 on Uncheatable and 67 to 71 on Table-9. All eight phase
coefficients remain nonzero with stable signs across folds, so the failure is
not coefficient collapse. Instead, the selected nonlinear shape is unstable
and boundary-seeking. Both full fits choose `retention=10`,
`late_multiplier=4`, and `ridge=1`, all at the maxima of their frozen grids.
The three Uncheatable outer folds choose three different benefit
exponent/offset combinations and ridge values from `0.01` to `1.0`.

The raw optima are unusable:

| Target | Predicted BPB | Predicted fiber gain | Phase TV | Max bucket weight | Nearest-policy TV |
|---|---:|---:|---:|---:|---:|
| Uncheatable | 0.764372 | 0.377306 | 0.994875 | 0.994875 | 0.817737 |
| Table-9 | 0.620209 | 0.575609 | 0.988525 | 0.988455 | 0.775679 |

The WSD80 random-fold positive controls remain representable: Programming
Languages predicts a `0.008048` BPB phase gain and locates the optimum within
`0.0354`, while GitHub Python and C++ predict `0.006014` and `0.007883` BPB
gains. The same form invents `0.029430` and `0.029460` BPB gains for C4 English
and Falcon RefinedWeb, whose observed sampled gains are zero. Blocked-region
RMSE ratios range from 1.72 to 4.81 relative to parent RPL, so random-fold
interpolation does not transfer across mixture regions.

Decision: **reject the estimator repair without retuning.** Identifying and
shrinking every phase coefficient does not fix RPL's response law. The model
can express code-objective phase benefit, but it cannot condition finite
contrast on aggregate state and target response strongly enough to suppress
unsupported broad-target gain or prevent near-corner 39-bucket optima. Do not
reopen this route through another ridge grid, feature scaling, signed-column
encoding, or nested selector. The next candidate must add a materially new
aggregate-conditioned phase mechanism with an independently bounded
finite-contrast response.

Artifacts:

- `reference_outputs/repaired_rpl_300m_20260731/report.md`
- `reference_outputs/repaired_rpl_300m_bootstrap_20260731/report.md`
- `reference_outputs/repaired_rpl_wsd80_controls_20260731/report.md`
- `reference_outputs/repaired_rpl_wsd80_controls_20260731/control_summary.html`

## WSD80-SUR-046 independent review and bootstrap erratum

**2026-07-31; independent reviews and corrected diagnostics.** Claude Opus 5
performed separate statistical and mechanistic audits of the frozen candidate.
Both returned **FAIL**, and neither found a reason to reverse the rejection.

The statistical audit verified the 520 rows, 280 correspondence groups, 238
exact asymmetric pairs, nested correspondence-grouped folds, and paired
comparisons. It found two bootstrap defects: reverse-orientation comparisons
credited exact ties as wins, and Regret resampling changed the candidate
population by discarding multiplicity. Protocol
`35ec63528f377f98227d41e326b2ad4d688dd5ec0880edf7dd8faa1d96519158`
corrects both. Strict better/tied/worse probabilities now sum to one and exact
Regret ties are reported as ties. All primary SUR-046 conclusions are
unchanged.

The mechanistic audit found an error in the wording of the original rejection.
SUR-046 penalizes every *explicit phase-control column*, but it does not
penalize the dominant phase-sensitive retained-share benefit channel. The same
nonnegative amplitude multiplies aggregate benefit and the retained state, so
the model cannot express “aggregate response present, phase response absent.”
The estimator repair therefore did not test whether shrinking that shared
channel fixes RPL. This is an erratum to the rationale, not a rescue of the
candidate.

The phase-blind refit under protocol
`60bc552672b9ba1efcbf69d6794f514e824213f22d70478b63c56f948c8bc4a0`
shows that temporal RPL channels carry real signal. Removing them worsens
pair-delta RMSE from `0.009328` to `0.017990` on Uncheatable and from
`0.019467` to `0.036104` on Table-9. The decisive diagnosis is therefore:
phase information is identifiable, but RPL's retained-state transition and
shared response law are inferior to HPR and extrapolate pathologically.

The next diagnostic was frozen before inspecting its correlations. For every
exact aggregate-matched pair, compute the retained-mass ratio
`M=sum(S_asymmetric)/sum(S_tied)`. If observed pair deltas are unrelated to
`M` (`|Spearman| < 0.2`) while RPL predictions track it (`|Spearman| > 0.6`),
the gain is a retained-mass artifact and a conserved retained-share state is
admissible for a new candidate. If observed deltas track `M`, conservation is
contradicted; intermediate results are inconclusive.

Artifacts:

- `reference_outputs/repaired_rpl_300m_20260731/cc_review_reconciliation.md`
- `reference_outputs/repaired_rpl_300m_bootstrap_20260731/report.md`
- `reference_outputs/repaired_rpl_phase_blind_diagnostic_20260731/report.md`

## WSD80-SUR-047: conserved retained-share state blocked

**2026-07-31; no model fit.** The independent mechanistic review proposed
normalizing RPL's retained state onto the simplex,
`S_tilde_i=S_i/sum_j(S_j)`, to separate composition from an unconserved
retained-mass multiplier. This would preserve the tied aggregate response
exactly and prevent a uniform late shift from receiving free benefit.

Nearest prior routes are `HRC`, `FOMF`, and `prior_AG` for finite conserved
capacity, `PMVT`/`PWD` for aggregate-fiber conservation, and
`WSD80-SUR-041` for normalized contrast. The materially new claim was
conservation of the *post-transition retained state*, rather than conservation
of policy aggregate, capacity, or feature scale.

The no-fit falsification was frozen before outcomes under protocol
`f2bc8f8a54db910fb91c94eebc7d767da016ae48e05a268034c890d6331b2b35`.
For every exact aggregate-matched pair it computed
`M=sum(S_asymmetric)/sum(S_tied)` under the frozen target-specific full-fit
shape. Conservation was admissible only if observed pair deltas were unrelated
to `M` (`|Spearman|<0.2`) while model-predicted deltas tracked it
(`|Spearman|>0.6`).

The result contradicts conservation on all three WSD80 code controls:

| Target | Spearman(M, observed delta) | Spearman(M, predicted delta) | Decision |
|---|---:|---:|---|
| Programming Languages | 0.439 | 0.460 | contradicts |
| GitHub C++ | 0.490 | 0.517 | contradicts |
| GitHub Python | 0.512 | 0.537 | contradicts |
| C4 English | 0.140 | 0.128 | inconclusive |
| Falcon RefinedWeb | 0.133 | 0.125 | inconclusive |
| 300M Uncheatable | 0.117 | 0.137 | inconclusive |
| 300M Table-9 | 0.142 | 0.177 | inconclusive |

Decision: **block before fitting.** Exact retained-mass conservation would
remove a coordinate associated with real code-target phase response, while no
cell satisfies the preregistered retained-mass-artifact signature. This does
not establish retained mass as causal; it establishes that normalization is
not justified by this diagnostic. The proposed per-family transient-fraction
extension is also blocked because it required the conserved state.

Artifacts:

- `reference_outputs/retained_mass_artifact_diagnostic_20260731/report.md`
- `reference_outputs/retained_mass_artifact_diagnostic_20260731/summary.csv`
- `reference_outputs/retained_mass_artifact_diagnostic_20260731/pair_diagnostics.csv`

## WSD80-SUR-048: counterfactual labile-fraction response blocked

**2026-07-31; no outcome fit.** The proposed candidate froze an aggregate
response \(F(a)\), reused RPL retained state \(S_i\), formed the exact tied
counterfactual \(S_i^0(a)\), and proposed

\[
\Delta L
=-\sum_f \phi_f\sum_{i\in f}A_i
\left[
\frac{S_i}{S_i+\tau}
-\frac{S_i^0}{S_i^0+\tau}
\right],
\qquad 0\leq\phi_f\leq1.
\]

The intended advantages were exact zero phase response when tied, bounded
finite-contrast response, and separate control over aggregate benefit and
phase sensitivity. These properties are real, but the route is not new. Its
nearest historical forms are `TEA`, `OGGTR`, `OTFSC`, `PMVT`, `MCR`,
`prior_P`, `prior_AL`, `IFSC`, and `MCCF`; the existing
`staged_retained_phase_control_20260730.py` already implements the same staged
tied-counterfactual retained-control construction with a different response
map.

The second-order expansion makes the duplication explicit:

\[
\Delta L
\simeq
-\sum_f\phi_f\sum_{i\in f}
A_i\frac{\tau}{(S_i^0+\tau)^2}\delta_i
+
\sum_f\phi_f\sum_{i\in f}
A_i\frac{\tau}{(S_i^0+\tau)^3}\delta_i^2.
\]

This is PMVT's odd transport plus even quadratic cost with their ratio fixed by
\(\tau\). Replacing inverse-power response by \(S/(S+\tau)\) is a saturator
change, not a new latent state, invariant, transition, or identification
equation.

Independent Claude Opus 5 review found additional blockers:

- \(S_i^0=(\beta_0+m\beta_1)a_i\), so tied evidence identifies only
  \(\tau/(\beta_0+m\beta_1)\); the saturation scale is confounded with the
  target-selected late multiplier.
- Reusing aggregate amplitudes \(A_i\) on a different response basis makes
  \(\phi_f\) absorb an arbitrary scale. Its \([0,1]\) bound therefore does not
  establish a physical labile fraction.
- WSD80 has two singleton families with no mapping to the 39-bucket
  `broad_text`, `tech_code`, and `reasoning` families.
- The selected retention gate is a saturated switch over two-bucket WSD80 but
  a soft ramp over the 39-bucket panel, so the state is not scale-covariant.
- WSD80 antithetic pairs preserve nominal 80/20 aggregate, while the model uses
  the realized `3040/3814` split. The aggregate backbone therefore does not
  cancel exactly on the positive-control panel.

Decision: **block before outcomes.** Do not reopen the family through another
bounded saturator, family multiplier, or tied-counterfactual response map.

The review proposed one genuinely different identification test: treat
retained mass as a measured effective-budget coordinate and import the
target's response slope from an orthogonal token-budget ladder rather than
fitting a phase amplitude. A second diagnostic factors retained state into
total mass and normalized composition, then asks whether composition retains
phase-delta signal after controlling for mass. Both require frozen,
outcome-independent protocols before evaluation.

Artifact:

- `reference_outputs/counterfactual_labile_fraction_review_20260731/report.md`

## WSD80-SUR-049: effective-budget equivalence diagnostic preregistered

**2026-07-31; protocol frozen before phase-pair evaluation.** SUR-047 showed
that the retained-mass coordinate cannot simply be normalized away on WSD80
code targets, while SUR-048 showed that fitting another phase-response scale
would reopen already rejected counterfactual-control routes. SUR-049 instead
tests an independently identified quantitative hypothesis:

\[
\Delta L_{\mathrm{budget}}
=s_{\mathrm{token}}
\log\left(M_{\mathrm{asym}}/M_{\mathrm{tied}}\right).
\]

Here \(M_{\mathrm{asym}}/M_{\mathrm{tied}}\) is frozen from the exposed
target-specific SUR-046 retained-state fit. The response scale
\(s_{\mathrm{token}}\) is not estimated from asymmetric-policy outcomes. It is
the equally weighted mean of six BPB-versus-log-materialized-token slopes,
each fitted at one tied aggregate using the independent 1B, 2B, 4B, and 8B
fixed-model, fixed-simulated-epoch token ladder.

The nominal WSD80 fibers use an exact `0.8/0.2` design split, while training
used the realized `3040/3814` boundary. The diagnostic therefore subtracts
the tied-only piecewise-linear estimate of
\(F(a_{\mathrm{asym,real}})-F(a_{\mathrm{tied,real}})\) before evaluating the
phase delta. The primary subset contains nominal aggregates in the token
ladder's `0.10--0.35` support; all phase pairs are secondary.

Frozen protocol:
`8e8835118fdec7468595adb43e9fde29c43c851141dd9f1ad86468f5c093d528`.
Its gate requires:

1. all six independently estimated token slopes are negative;
2. maximum leave-one-aggregate relative slope change is at most `0.25`;
3. the `20,000`-draw paired-bootstrap 95% lower bound for RMSE improvement
   over the zero-phase null is positive;
4. the calibration-slope interval contains one and excludes zero; and
5. absolute bias is no worse than the zero-phase null.

A pass supports effective-budget equivalence only for the WSD80
Programming-Languages development panel. It does not establish retained mass
as causal or provide a 39-bucket surrogate. A failure rejects this
parameter-free equivalence but does not establish retained mass as irrelevant.

Artifacts:

- `diagnose_effective_budget_equivalence_20260731.py`
- `reference_outputs/effective_budget_equivalence_20260731/protocol.json`

### SUR-049 outcome: quantitative equivalence rejected

The frozen diagnostic ran exactly once without adaptation. Its externally
estimated token response was coherent: all six aggregate-specific slopes were
negative, their equally weighted mean was `-0.062287` BPB per log materialized
token, and the maximum leave-one-aggregate relative change was only `0.040`.
The phase prediction nevertheless pointed in the wrong direction.

| Subset | Pairs | Zero-phase RMSE | Effective-budget RMSE | Improvement | Calibration slope | Spearman | Sign accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary aggregate support | 104 | 0.069218 | 0.118593 | -0.049376 | -0.909 | -0.705 | 0.269 |
| all pairs | 255 | 0.091633 | 0.130695 | -0.039062 | -0.913 | -0.439 | 0.400 |

The primary RMSE-improvement bootstrap interval was
`[-0.057968,-0.041206]` BPB and the calibration-slope interval was
`[-1.217,-0.628]`. Absolute bias also worsened. The nominal-versus-realized
aggregate correction was not responsible: its mean absolute size was only
`0.000192` BPB, and the Spearman correlation between log retained mass and
observed pair delta changed only from `0.706` before correction to `0.705`
after correction.

Decision: **reject effective-budget equivalence.** Larger RPL retained mass is
associated with worse observed asymmetric-policy performance, whereas more
real training tokens lower BPB. Retained mass therefore cannot be interpreted
as extra useful budget. This does not make the coordinate irrelevant: it may
be a proxy for even asymmetry damage. Do not repair the inversion by fitting a
free sign or scale to these exposed pairs.

The next admissible no-fit question is whether normalized retained-state
composition carries phase-order signal after conditioning on total retained
mass. It must be tested on both high-TPP 300M targets under a frozen protocol
before introducing a composition transition into a fitted surrogate.

Additional artifacts:

- `reference_outputs/effective_budget_equivalence_20260731/report.md`
- `reference_outputs/effective_budget_equivalence_20260731/summary.csv`
- `reference_outputs/effective_budget_equivalence_20260731/pair_predictions.csv`
- `reference_outputs/effective_budget_equivalence_20260731/effective_budget_predicted_vs_observed.html`

## WSD80-SUR-049 audit and WSD80-SUR-050 block

**2026-07-31; post-outcome audit plus outcome-free admissibility review.**
The frozen SUR-049 rejection remains valid, but its mechanistic interpretation
and uncertainty statement required correction. A bounded, read-only Claude
Opus 5 review (`700f34fd-a6a1-4318-82f0-04a405cad50a`) inspected the source
protocol, implementation, token slopes, and all 255 pair rows without accessing
any `targeted_pairwise` path. The row-level claims and cluster uncertainty were
then independently reproduced with:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_sur049_retained_composition_20260731.py
```

The audit found:

1. `M_tied` is constant at `1.6088096487` for every tied policy, so
   `log(M_asym/M_tied)` is only a shifted asymmetric-mass coordinate. The tied
   leg removes no nuisance.
2. Six nearly fixed-mass predictions span only `0.000851` BPB while their
   observed phase deltas span `0.146319` BPB, ordered primarily by aggregate.
3. On the aggregate-`0.18` fiber containing the observed WSD80 optimum,
   retained mass and phase delta have Spearman `-0.040`. Rows with almost equal
   mass have opposite-sign deltas. On the aggregate-`0.30` fiber, the
   correlation is `0.934`, so mass behaves like even damage only there. It is
   not a stable state coordinate across aggregates.
4. The original IID pair bootstrap ignores repeated tied controls and smooth
   within-fiber dependence. Cluster bootstraps over nine rounded aggregates and
   nine tied controls give RMSE-improvement intervals
   `[-0.063646,-0.019540]` and `[-0.063646,-0.019685]`. The rejection survives,
   but the original interval overstated precision.
5. The frozen protocol text says that mismatch correction uses a `1B`
   tied-only curve, while the implementation correctly uses tied rows from the
   complete WSD80 panel. The protocol remains byte-for-byte frozen; the
   discrepancy is recorded in `protocol_erratum.json`.

Decision: retain only the narrow SUR-049 conclusion that retained mass is not
quantitatively equivalent to extra useful training budget on the WSD80
Programming-Languages panel. Withdraw the post-outcome suggestion that it may
be a generally valid asymmetry-damage proxy.

The proposed follow-up,

\[
M=\sum_i S_i,\qquad \pi_i=S_i/M,
\]

is recorded as **WSD80-SUR-050, blocked before fit**. It has no material
novelty: `pi` is SUR-047's normalization of the same fitted RPL state. At fixed
bucketwise aggregate, `S` is a deterministic function of the 38-dimensional
phase contrast, so `(M,pi)` supplies no separately intervened coordinates.
Conditioning on aggregate reduces the proposal to the exposed PMVT/PWD
aggregate-fiber equation; omitting aggregate leaves the measured `0.146319`
BPB confound. The features also depend on retention and late multiplier
selected on these outcomes at grid maxima, while SUR-046 already establishes
that the residual temporal block carries pair signal. A positive result would
therefore be nearly guaranteed without identifying mass, composition, or the
transition law.

Reopen only with temporal mass and composition states that are separately
physically measured or independently identified, have units covariant under
bucket refinement, and retain the same meaning in WSD80 and the 39-bucket
panel.

Artifacts:

- `audit_sur049_retained_composition_20260731.py`
- `reference_outputs/sur049_retained_composition_audit_20260731/report.md`
- `reference_outputs/sur049_retained_composition_audit_20260731/summary.json`
- `reference_outputs/sur049_retained_composition_audit_20260731/protocol_erratum.json`

## WSD80-SUR-051 preregistration: HPR/RPL phase-block attribution

**2026-07-31; outcome-free diagnostic.** The expanded 300M baseline leaves a
specific unresolved discrepancy. HPR predicts exact aggregate-matched phase
contrasts better than RPL on both targets, while phase-blind RPL ablation
already proves that the panel contains identifiable phase signal. Further RPL
retuning and aggregate-conditioned phase corrections are closed routes.

`WSD80-SUR-051` therefore introduces no candidate model. It freezes the
baseline outer folds and each fold's selected nonlinear configuration, then
decomposes OOF exact-pair deltas into physically named response blocks. Each
block is also removed with the original linear head refit on the outer training
rows. This separates exact algebraic contribution from unique predictive
necessity under collinearity.

The diagnostic may motivate one new mechanism only if a non-TV physical block
worsens refit-omission pair RMSE on both targets, is fold-stable, clears a
paired-bootstrap gate on at least one target, has a common directional
interpretation, and has a meaningful WSD80 analogue. Diffuse attribution,
cross-target instability, or an advantage carried only by global policy TV is
a negative result and does not license an HPR/RPL hybrid.

Frozen protocol:

- `reference_outputs/hpr_rpl_phase_block_attribution_20260731/preregistration.md`

**Pre-outcome correction.** The reconstruction invariant rejected physical
block attribution for the original RPL head before any block result was
produced. Its unpenalized duplicate `x,-x` columns are non-identifiable enough
that a different matrix-summation order moves the Table-9 block total by
`2.0e-4` BPB. The diagnostic RPL arm now uses the previously frozen
`WSD80-SUR-046` identifiable head and selections. That estimator preserves the
RPL state and response span while assigning one penalized signed coefficient
per phase feature. HPR and its expanded-baseline folds remain unchanged.

## WSD80-SUR-051 decision and WSD80-SUR-052 preregistration

**2026-07-31; completed diagnostic and frozen follow-up.** SUR-051 reproduced
every frozen OOF prediction to `2.22e-16` and found two numerically eligible
HPR blocks rather than one isolated mechanism. Removing and refitting without
the retained-bucket benefit worsens exact-pair RMSE by `0.005914` on
Uncheatable and `0.009591` on Table-9. That block is the full retained-state
spine, however: it bundles transition, response, and hierarchical amplitudes.
Family-member replay also passes the numerical rule at `0.000080` and
`0.000657`, but the Uncheatable interval reaches zero and its WSD80 singleton
analogue is inseparable from family overexposure.

Global phase TV is necessary for pair RMSE (`0.002709/0.003554`) but explains
only `0.5%/0.6%` of pair covariance. It is a generic even-cost correction, not
a phase-order mechanism. SUR-051 is therefore a completed negative attribution
for model promotion and does not license an HPR/RPL hybrid.

Before any crossover outcome is computed, `WSD80-SUR-052` freezes a
diagnostic-only `2 x 2` comparison of HPR versus RPL retained state and HPR
power versus RPL inverse-power response. It replaces only HPR's retained-bucket
block, preserves all other HPR columns and the frozen common folds, and matches
alternative column scales on training rows without using targets. A component
can motivate a new route only if it explains at least half the parent
HPR-versus-RPL pair-RMSE gap on both targets with the frozen cross-target,
fold-sign, bootstrap, and WSD80-analogue conditions.

Frozen protocol:

- `reference_outputs/retained_state_response_crossover_20260731/preregistration.md`

## WSD80-SUR-052 decision: no component graft is licensed

**2026-07-31; frozen crossover completed.** The diagnostic ran with 10,000
paired bootstrap resamples under protocol
`54815b82c68f2abadb70b024efbc4d50c84b56bd51941394988778685ba556c1`.
The exact HPR control reproduced its persisted OOF predictions to
`2.22e-16`.

| Target | Factor | Main effect | 95% interval | Positive folds | Frozen half-gap |
|:--|:--|--:|:--|--:|--:|
| Uncheatable | HPR state | +0.000899 | [+0.000395, +0.001398] | 3/3 | 0.000567 |
| Table-9 | HPR state | +0.001082 | [+0.000341, +0.001868] | 3/3 | 0.001159 |
| Uncheatable | HPR link | +0.001051 | [+0.000540, +0.001559] | 3/3 | 0.000567 |
| Table-9 | HPR link | +0.000325 | [-0.000319, +0.001048] | 1/3 | 0.001159 |

The HPR retained state is genuinely useful, but it misses the preregistered
Table-9 magnitude gate by `0.000077` BPB. The HPR response link is not stable
on Table-9 and reverses slightly under the RPL state (`-0.000075` BPB).
State-by-link interactions are significantly negative on both targets
(`-0.000668/-0.000800`), so the individual advantages diminish rather than
add when crossed.

Decision: **complete SUR-052 as a negative attribution.** Neither component
licenses a new graft, and the Table-9 threshold must not be relaxed after the
near miss. The literal factorial result does not by itself establish that
HPR's state and response are intrinsically inseparable. The next route must
introduce a materially new temporal state or response invariant identified
independently of these crossover outcomes.

Artifacts:

- `diagnose_retained_state_response_crossover_20260731.py`
- `reference_outputs/retained_state_response_crossover_20260731/report.md`
- `reference_outputs/retained_state_response_crossover_20260731/decision_gate.csv`
- `reference_outputs/retained_state_response_crossover_20260731/state_response_crossover.html`

### Post-outcome identification audit

A bounded read-only Claude Opus 5 review, run through the verified subscription
with `ANTHROPIC_API_KEY` removed and `Agent` explicitly disabled, found two
limitations that were independently reproduced:

1. the HPR and RPL state arms use different late multipliers
   (`4.982800` versus `4.000000`), so the state contrast does not isolate the
   gate argument; and
2. alternative columns were matched by uncentered RMS, while HPR's ridge head
   centers the design before penalization.

The native-shape diagonal also has an estimator advantage over both crossed
cells, so the significantly negative interaction cannot establish intrinsic
state-link inseparability. The original implementation is faithful to its
literal preregistration and its no-pass decision remains valid. The stronger
mechanistic interpretation is withdrawn.

`WSD80-SUR-053` is frozen as a closure diagnostic, not a candidate. It keeps
HPR's link and late multiplier fixed, compares the HPR absence gate with the
RPL signed-contrast gate using training-only centered RMS scaling, and includes
a local-slope-matched contrast-gate control. It retains the original
`0.000567/0.001159` half-gap thresholds. A pass can preserve HPR's existing
gate as an empirical baseline; it cannot promote a new route.

Frozen protocol:

- `reference_outputs/hpr_absence_gate_isolation_20260731/preregistration.md`

### WSD80-SUR-053 decision

**2026-07-31; frozen closure diagnostic completed.** Protocol
`a10790f30a82f018bd5d8ee5703c0f02483d734bab755a7764ec47daa819e93d`
reproduced HPR OOF predictions within `2.22e-16`.

With HPR's late multiplier held common and alternative columns matched by
training-fold centered RMS, replacing HPR's absence gate with the native-rate
RPL contrast gate worsened exact-pair RMSE by:

- `+0.000816` BPB on Uncheatable, 95% interval
  `[+0.000228,+0.001468]`, positive in `3/3` folds; and
- `+0.000886` BPB on Table-9, 95% interval
  `[+0.000212,+0.001579]`, positive in `3/3` folds.

The effect is real but misses the unchanged Table-9 threshold of `0.001159`
BPB. The slope-matched contrast gate is worse on both targets
(`+0.001341/+0.002127`) but was frozen as interpretive only and cannot change
the decision.

Decision: **complete SUR-053 as negative and close retained-state component
attribution.** Preserve HPR as an empirical baseline; do not reopen the
absence gate through another rate, late multiplier, scaling rule, or response
graft. The corrected diagnostic licenses no candidate.

Artifacts:

- `diagnose_hpr_absence_gate_isolation_20260731.py`
- `reference_outputs/hpr_absence_gate_isolation_20260731/report.md`
- `reference_outputs/hpr_absence_gate_isolation_20260731/decision_gate.csv`
- `reference_outputs/hpr_absence_gate_isolation_20260731/absence_gate_isolation.html`

## Final shared-surrogate decision

**2026-07-31; WSD80-SUR-054 registry-exhaustion audit.** The 99-route
historical registry and 32-entry current registry were re-audited after the
SUR-051--053 attribution sequence. No untested endpoint-state proposal
survives the novelty boundary. Common scalar temporal kernels collapse
algebraically to one effective phase multiplier (`NG-LK`), affine scalar
endpoint dynamics have a tied reachable-set equivalent (`NG-AES`), and the
registered nonlinear fast/slow, path, coverage, replay, and consolidation
families have already failed their shape, identification, transfer, or
raw-optimum gates.

The empirical result is a split decision:

- hierarchical phase replay remains the strongest 300M exact-contrast
  baseline (`0.007850/0.016902` pair RMSE on Uncheatable/Table-9), but its raw
  optima are essentially tied and it has not solved the desired two-phase
  policy-selection problem;
- repaired retained power law preserves the WSD80 code-target advantage
  (`0.008048` predicted versus `0.009594` observed, optimum distance
  `0.035355`) but invents about `0.0294` BPB of phase gain on each broad-text
  negative control and selects near-corner 300M raw optima.

No new model clears the frozen acceptance gate. Do not submit a training
validation from this drive. Reopening requires a separately measured temporal
state or an intervention that distinguishes path-dependent dynamics from
endpoint exposure summaries, not another gate, multiplier, coefficient grid,
or response-link swap.

Final synthesis:

- `reference_outputs/two_phase_surrogate_final_synthesis_20260731/report.md`
- `reference_outputs/two_phase_surrogate_final_synthesis_20260731/acceptance_gate.csv`
- `reference_outputs/two_phase_surrogate_final_synthesis_20260731/decision.json`

## Adversarial audit of the negative decision

**2026-07-31; WSD80-SUR-055 adversarial audit.** The negative decision was
re-derived from the persisted source predictions rather than from the synthesis
prose. It **survives**, and the strongest supporting evidence is observational
rather than model-comparative. Two of the disqualifiers attached to repaired RPL
do not survive, and one gate evidence string overstates its finding.

### The sharpest counter-hypothesis, refuted

Exact-pair `delta_rmse` is the gate's primary phase-sensitive diagnostic, and HPR
wins it while predicting near-tied optima. That combination is the signature of a
predictor shrunk toward zero, which under a squared-error metric beats an
honest-magnitude predictor with any noise. If true, the gate would have rewarded
the absence of a phase response.

It is false. Against the null predictor (predict zero delta), whose RMSE is the
observed delta RMS:

| target | null RMSE | HPR | ratio | amplitude ratio | slope obs~pred |
|:--|--:|--:|--:|--:|--:|
| uncheatable | 0.017990 | 0.007850 | 0.436 | 0.881 | 1.021 |
| table9 | 0.036104 | 0.016902 | 0.468 | 0.800 | 1.109 |

HPR predicts deltas at 80-88% of observed amplitude with calibration slope near
one, explaining roughly 80% of delta variance. Repaired RPL is the *more* shrunk
of the two on Table-9 (0.732). The diagnostic is sound.

### The decisive fact: no two-phase advantage exists in this design

| target | best tied | best asymmetric | asymmetric policies beating best tied |
|:--|--:|--:|--:|
| uncheatable | 0.951105 | 0.955440 | **0/238** |
| table9 | 0.982774 | 0.988999 | **0/238** |

Provenance checked: the winning tied rows are physically trained
`single_phase_exposure_average` runs (`singleavg_run_*`) from the same source
experiment as the two-phase runs, not constructed counterparts. At matched
aggregate the phase effect is a coin flip (117/238 and 123/238 asymmetric arms
beat their own counterpart).

HPR's near-tied raw optimum is therefore the **empirically correct answer**, not
a deficiency. The synthesis frames it as HPR "not solving" two-phase selection;
on this design there is nothing to solve, and a model that reported a large gain
would be wrong.

### The mechanism, measured on both panels

Phase benefit is monotone in aggregate suboptimality and crosses zero mid-range.

| aggregate quintile | tied BPB | mean delta (unch.) | two-phase wins |
|:--|--:|--:|--:|
| 0 best | 0.973 | +0.0138 | 17% |
| 2 | 0.997 | -0.0000 | 50% |
| 4 worst | 1.023 | -0.0108 | 69% |

Spearman(tied BPB, delta) = `-0.467` / `-0.497`, p = 2.6e-14 / 2.8e-16. The
best-decile splitting penalty is `+0.0119`/`+0.0310` BPB, Wilcoxon p = 1e-4,
paired and therefore robust to the absence of replicates (the design has 520
distinct coordinates and zero repeats, so it carries no internal noise estimate).

The same law holds on StarCoder, where `g(a)` is exactly zero at the best tied
aggregate `a=0.30` and rises monotonically away from it. Spearman(excess, gain)
is `+0.943`/`+0.829`/`+0.976` on uncheatable/table9/WSD80.

What separates the panels is the ratio of achievable gain to the aggregate
penalty it costs:

- 300M: ratio 0.50-0.71 throughout; maximum net `-0.004334`/`-0.006225`. Tied
  always wins.
- WSD80: ratio 0.21-0.75 for every `a >= 0.35`, but `1.935` at `a=0.18`, giving
  net `+0.009593`. The entire observed two-phase advantage comes from one
  aggregate.

Excess magnitude alone does not explain it: `a=0.18` (excess 0.0103) yields gain
0.0199 while `a=0.35` (excess 0.0081) yields 0.0017, a twelve-fold difference at
comparable excess. The distinguishing feature is direction - `a=0.18` is *below*
the tied optimum with best contrast `+0.400` (late-heavy), `a=0.35` is above it
with `-0.104`. That directional marginal-value structure is already what RPL's
`marginal_phase_block` encodes, so it is not a new mechanism.

### Two disqualifiers that do not survive

**1. The broad-text false gains are protocol-dependent by an order of magnitude.**

| target | random folds | blocked folds | frozen threshold |
|:--|--:|--:|--:|
| C4 English | 0.029430 | 0.003185 | 0.005 |
| Falcon RefinedWeb | 0.029460 | 0.002973 | 0.005 |

Under blocked folds repaired RPL **passes** the negative control, and improves on
the original RPL's 0.0266. `primary_protocol: random` was frozen in advance with a
documented rationale (the earlier RPL audit reported only random-fold refits), so
this is not a post-hoc inconsistency and is not reported as one. It does mean the
finding is a random-fold overfitting symptom - neighbours leak across folds, the
surface sharpens, the optimizer runs to a corner - rather than a property of the
functional form. The primary 300M gate uses grouped folds.

**2. The late multiplier is pinned at its grid maximum in 6 of 6 folds.**

| target | fold 0 | fold 1 | fold 2 |
|:--|:--|:--|:--|
| uncheatable | ret 10 MAX, late 4 MAX, ridge 0.01 | ret 5, late 4 MAX, ridge 0.01 | ret 10 MAX, late 4 MAX, ridge 1 MAX |
| table9 | ret 5, late 4 MAX, ridge 1 MAX | ret 10 MAX, late 4 MAX, ridge 1 MAX | ret 10 MAX, late 4 MAX, ridge 1 MAX |

`LATE_MULTIPLIERS = (1.0, 2.0, 4.0)` was set on 2026-07-29 from sweeps on the
**delphi_3e18** panel, where 8/16/32 measured worse. The repaired estimator calls
`parent.shape_grid()` and inherits it; it was never validated at 300M. A parameter
at its bound in every fold is not identified within the searched range, so both
the "15-16% worse than HPR" comparison and the near-corner raw optimum are
measured at a constrained optimum. An extended-range re-fit is running as an audit
of the existing decision, writing to `audit_late_boundary_300m_20260731`; the
frozen artifacts are untouched.

**3. Minor: the gate evidence string overstates the boundary finding.**
`acceptance_gate.csv` states repaired RPL "selects retention=10, late_multiplier=4,
and ridge=1 at screened maxima on both targets". Only the late multiplier is
unanimous; retention is 5 in two folds and ridge is 0.01 in two folds.

### Audit verdict

The decision stands: no candidate should be promoted, and no training validation
should be submitted. The negative result is stronger than stated, because the
governing fact is observational - zero of 238 exact contrasts beat the best tied
policy - rather than a matter of which model fit better. Correcting the two
unsound disqualifiers does not change it, since a model predicting 0.20-0.58 BPB
of two-phase gain is wrong regardless of how its multiplier grid was chosen.

The reopening question is also sharper than "which temporal state". It is: what
sets the ratio of achievable phase gain to aggregate penalty, and why is it near
2 at StarCoder `a=0.18` and at most about 0.75 everywhere else on both panels?
A trajectory panel that varies phase-boundary placement at fixed endpoint
exposure should be designed to measure that ratio directly, at aggregates both
below and above the tied optimum, because the direction of departure and not its
magnitude is what distinguishes the one aggregate where two-phase wins.

### Late-multiplier boundary audit, completed

`audit_late_boundary_300m_20260731`, extended `LATE_MULTIPLIERS = (1,2,4,8,16)`,
2700 candidates, protocol otherwise identical to the frozen repaired-RPL run.

| target | metric | frozen (late<=4) | audit (late<=16) | HPR |
|:--|:--|--:|--:|--:|
| uncheatable | all_rmse | 0.007850 (+15.4%) | 0.007353 (+8.1%) | 0.006800 |
| uncheatable | pair_rmse | 0.009328 (+18.8%) | 0.008628 (+9.9%) | 0.007850 |
| table9 | all_rmse | 0.015110 (+16.2%) | 0.014008 (+7.7%) | 0.013001 |
| table9 | pair_rmse | 0.019467 (+15.2%) | 0.018066 (+6.9%) | 0.016902 |

Three conclusions.

**The frozen gate evidence overstated the deficit.** Fairly tuned, repaired RPL
is 7-10% worse than HPR rather than 15-16%. It still fails the 5% grouped-OOF
tolerance, so the verdict is unchanged, but the margin is roughly half what was
reported and the "15.4%/16.2%" figures should not be quoted without this caveat.

**The late multiplier is unidentified.** It selects 16.0 in 5/6 folds, having
selected 4.0 in 6/6 under the previous ceiling: it saturates whatever bound is
offered. This is the empirical counterpart of the theoretical `NG-LK` result that
a common scalar temporal kernel collapses to one effective phase multiplier. The
endpoint design cannot pin it, which is precisely the quantity the proposed
trajectory panel exists to measure. Extending the grid further is pointless;
identification has to come from intermediate evaluations, not a wider search.

**The corner optimum is intrinsic, not a search artifact.** It persists and
slightly worsens under the extended grid, phase TV `0.994875` to `0.995566` and
maximum bucket weight `0.994875` to `1.000000` on uncheatable, with predicted
fiber gain `0.296724` against a maximum *observed* achievable gain of about
`0.049` anywhere in the 238 exact contrasts. Correcting the multiplier range does
not rehabilitate the raw surface, which is the outcome the charter's optimization
audit is meant to detect.

The negative decision is therefore confirmed on all three counts, with one piece
of its supporting evidence corrected and one of its theoretical claims upgraded
from algebraic argument to measured fact.

### Correction: the late multiplier is identified, and is not an offset proxy

The boundary-audit entry above concluded that the late multiplier "saturates
whatever bound is offered" and is therefore unidentified from endpoint data.
**That conclusion was wrong**, and is corrected here. Two boundary hits were read
as a monotone likelihood; they are not.

Holding every other parameter at the shape the frozen run selected and sweeping
the multiplier over `(1,2,4,8,16,32,64)` on the correspondence folds:

| target | sweep | late 1 | best | argmin | late 64 |
|:--|:--|--:|--:|--:|--:|
| uncheatable | A: offset fixed 0.01 | 0.008728 | 0.007277 | 8 | 0.008432 |
| uncheatable | B: offset = 0.01 x late | 0.008728 | 0.007194 | 8 | 0.007426 |
| table9 | A: offset fixed 0.01 | 0.016173 | 0.014829 | 8 | 0.016693 |
| table9 | B: offset = 0.01 x late | 0.016173 | 0.013990 | 16 | 0.014058 |

The likelihood turns over. The multiplier has an interior optimum near 8-16 on
both targets under both sweeps, and 64 is clearly worse than 8. The full-grid
selections landed on 4 and then 16 because the joint criterion trades the
multiplier against the other shape axes, not because the surface is monotone;
4 was genuinely below the optimum and 16 sits at it.

Sweep B is the point of the diagnostic. The multiplier enters only through
`S = survival*beta0*w0 + late*beta1*w1` with `benefit = (S + E0)**-a`, so raising
it both suppresses phase-0 exposure and drives the effective offset `E0/late`
toward zero. Tying `E0` to the multiplier closes the offset channel; exact
scale-equivalence would additionally require scaling `beta0`, which the design
fixes, so the two channels do separate. Closing the offset channel does not
remove the gain, it slightly enlarges it: `-17.6%/-13.5%` against `-16.6%/-8.3%`.

**The multiplier is therefore a real effect, not a reparameterization.** Early
exposure carries roughly eight to sixteen times less predictive weight than late
exposure at 300M, and that ratio is estimable from endpoint data alone.

This also corrects how `NG-LK` should be cited. The algebraic collapse of a
common scalar temporal kernel to one effective phase multiplier is real, and the
resulting scalar *is* identified by this design, so "trajectories are needed to
identify the multiplier" is not the argument. The argument is that one scalar
cannot separate retention, forgetting, within-window repetition, and
optimizer-time evolution, which all collapse into it. Identifiability of the
scalar is not identifiability of the mechanism, and only the second is what a
trajectory panel buys.

A confirmation run under the full nested selection with the multiplier grid
extended to 64 is in `audit_late_interior_300m_20260731`; if the corrected
reading is right it should select an interior value rather than the ceiling.

**Confirmation, grid extended to 64.** The corrected reading holds. Selections
are interior and identical to the ceiling-16 run: `late=16` in 5/6 folds and
`late=8` in the remaining fold on both targets, with the ceiling at 64 unused.
Nested metrics are unchanged, `all_rmse` `+8.1%/+7.7%` and `pair_rmse`
`+9.9%/+6.9%` against HPR, so those figures are now stable under any further
widening and are the final fair comparison.

The raw optimum moved the wrong way. On uncheatable it is now fully degenerate,
phase TV `1.000000` and maximum bucket weight `1.000000`, a policy placing an
entire phase on one bucket; Table-9 is unchanged at `0.966940/0.999799`. With the
multiplier identified at an interior value and the grid demonstrably wide enough,
the corner optimum can no longer be attributed to a constrained search. It is
intrinsic to the form, which is what the charter's optimization audit exists to
detect and is sufficient on its own to refuse promotion.

Net effect of the whole audit on the frozen decision: unchanged in outcome, with
one evidence figure corrected downward (15-16% to 8%), one disqualifier shown to
be protocol-dependent (broad-text false gains pass under blocked folds), one of
my own intermediate claims retracted (the multiplier is identified, not
saturating), and the strongest supporting evidence relocated from model
comparison to the observational fact that 0/238 exact contrasts beat the best
tied policy.

## 2026-07-31: physically tied phase-blind RPL aggregate-spine audit

The aggregate-first charter requires rejecting a phase-blind spine before any
temporal state is added if it cannot fit tied policies and optimize plausibly.
`WSD80-SUR-063` therefore isolated repaired RPL's aggregate restriction:

\[
A(w)=b+\sum_i a_i(w_i+e_0)^{-p}
     +\sum_i d_i\max(E_i(w)-\tau,0)^q.
\]

Only 282 physically tied 300M policies entered the fit. Five outer
correspondence-grouped folds and three inner grouped folds nested nonlinear-shape
and ridge selection. Raw tied optimization used no KL penalty, trust region,
support term, or output calibration. The frozen protocol hash is
`e4fce3b2bc4659306a6920f73beb77d6b37b01dc3742adab1d78733ef6b4d371`.

Command:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_phase_blind_rpl_tied_spine_20260731.py --workers 8
```

| target | OOF RMSE | frozen reference | relative | raw predicted | observed tied frontier | nearest support TV | near-zero buckets |
|:--|--:|--:|--:|--:|--:|--:|--:|
| Uncheatable | 0.005588 | 0.004713 | +18.6% | 0.888506 | 0.951105 | 0.537 | 18 |
| Table-9 | 0.013017 | 0.010357 | +25.7% | 0.854732 | 0.982774 | 0.570 | 20 |

The raw optimum is not stable enough to rescue the form. Median fold-to-full L1
is `0.208/0.252` and conditional group-bootstrap median L1 is `0.293/0.186`.
Outer folds also disagree on the benefit exponent, offset, and ridge.

**Decision: rejected before temporal modeling.** Literal materialized-epoch
damage prevents the most extreme replay dose, but it cannot constrain mixture
composition. Independent bucket and family benefits still accumulate into
unsupported aggregate gains, yielding sparse optima more than 0.53 TV outside
the observed panel. Do not reopen with another offset, exponent, ridge, damage
threshold, or bounded scalar output link. The next aggregate candidate must
couple bucket contributions through a materially new production mechanism.

Artifacts:

- `reference_outputs/phase_blind_rpl_tied_spine_20260731/report.md`
- `reference_outputs/phase_blind_rpl_tied_spine_20260731/analysis.md`
- `reference_outputs/phase_blind_rpl_tied_spine_20260731/tied_spine_diagnostics.html`

## 2026-07-31: intervention-identified benchmark-production preregistration

`WSD80-SUR-064` tests the next admissible aggregate mechanism without first
fitting another nonlinear response. The 78 central proportional log tilts form
39 antithetic pairs and span the 38-dimensional tangent space of the 39-bucket
simplex. For each named benchmark component, the diagnostic estimates

\[
d_i=\frac{L_i(+\alpha)-L_i(-\alpha)}{2\alpha},\qquad
d=Aq,
\]

using only those tilts. The transfer map is then frozen and tested on the 39
full domain deletions, which are a distinct finite-move intervention class.
The nearly saturated in-sample directional fit is explicitly non-evidence.

This is not prior Q's collection of endpoint-fitted component heads and not
prior D's anonymous latent factorization: the factors are observed benchmark
components and the bucket transfer map is identified from named interventions.
The deletion panel is not used for fitting, calibration, or hyperparameter
selection.

Coverage preflight passed before freezing:

- 78 complete central log-tilt rows and 39 complete deletion rows;
- all policies are physically tied and all weights sum to one;
- 11 proportional controls for anchor and noise estimation;
- seven complete Uncheatable components, reconstructing the aggregate within
  `1.14e-7` BPB on untouched deletions;
- 51 complete Table-9 components, whose unweighted mean is exact.

Protocol hash:
`6fe95816365d2ed036cf3112e7c731c919388abd0d0eed35e885eedf98dc61d8`.

Command after preregistration:

```bash
env PYTHONPATH=. uv run experiments/domain_phase_mix/exploratory/two_phase_many/diagnose_intervention_identified_component_transfer_20260731.py --mode evaluate
```

Both aggregate targets and their component-support summaries must pass. A pass
licenses a bounded benchmark-production aggregate model; it does not license a
temporal state or promote a full surrogate. A failure blocks this route before
new nonlinear flexibility is introduced.

## 2026-07-31: intervention-identified benchmark production does not transfer on both targets

The frozen `WSD80-SUR-064` evaluation completed without changing the transfer
map, deletion panel, component definitions, or decision thresholds. The local
map was estimated from the 78 central log tilts and evaluated on the 39 full
domain deletions.

| target | deletion RMSE | anchor-null RMSE | RMSE improvement | Spearman | sign accuracy | observed-on-predicted slope | frozen gate |
|:--|--:|--:|--:|--:|--:|--:|:--|
| Uncheatable | 0.001962 | 0.004851 | 59.6% | 0.684 | 0.744 | 1.192 | pass |
| Table-9 | 0.007180 | 0.009803 | 26.8% | 0.404 | 0.718 | 0.923 | fail |

The Uncheatable map is a real identification result: all seven component
gradients survive BH correction, every component has positive deletion
Spearman, and the domain-bootstrap RMSE-improvement interval is `[0.176,0.682]`.
Table-9 is not comparably stable. Only `31/51` component gradients survive the
same correction, median component deletion Spearman is `0.202`, and the
domain-bootstrap aggregate RMSE-improvement interval is `[-0.203,0.525]`.
The point estimate improves over the null, but it does not establish transfer
to finite deletions under the preregistered rule.

**Decision: reject the shared route before nonlinear fitting.** Both targets
were mandatory, so the successful Uncheatable arm does not license a shared
benchmark-production aggregate model. In particular, do not fit a nonlinear
response, deletion calibration, latent factor, or component selection rule to
repair Table-9 after observing its deletion outcomes. The result instead says
that local named-component transfer is target-dependent: it is sufficient for
the seven-component Uncheatable macro and too heterogeneous for the 51-component
Table-9 macro under the available interventions.

Artifacts:

- `reference_outputs/intervention_identified_component_transfer_20260731/report.md`
- `reference_outputs/intervention_identified_component_transfer_20260731/aggregate_metrics.csv`
- `reference_outputs/intervention_identified_component_transfer_20260731/component_metrics.csv`
- `reference_outputs/intervention_identified_component_transfer_20260731/decision.json`

## 2026-07-31: unique-evidence demand allocation preregistration

`WSD80-SUR-065` tests a composition-coupled aggregate production law before
another endpoint fit is allowed. For tied aggregate weights, physical epochs
`E_i` and proportional pool mass `p_i` define exact materialized unique evidence

\[
m_i=p_i\min(E_i,1),\qquad U=\sum_i m_i,
\]

and a prior-smoothed evidence composition

\[
s_i=\frac{m_i+\epsilon p_i}{U+\epsilon}.
\]

The proposed aggregate response is

\[
A(w)=b+M D_{\mathrm{KL}}(q\Vert s(w))
       +h\{E_{\mathrm{total}}-U(w)\},
\]

where `q` is a target-specific benchmark-demand distribution, `M` and `h` have
BPB units, and all other quantities are dimensionless. In the linear head this
is represented by nonnegative `a_i=M q_i` on `-log s_i`; `M=sum_i a_i` and
`q_i=a_i/M` remove the amplitude/distribution symmetry. The lower bound follows
from nonnegativity of KL and repeated mass. Unlike an additive bucket-benefit
spine, surplus evidence in one bucket changes the normalized allocation and
cannot contribute an independent unbounded credit.

Nearest prior routes are `prior_B`, `prior_AH`, `prior_K`, `prior_L`, `prior_AN`,
and `HRC`. The material change is that unique coverage and target support are
not auxiliary features added to an additive response: conserved evidence
quantity and composition are the entire production state. This is still only a
candidate mechanism. A normalized amplitude is not by itself novel, and the
route is blocked unless its implied curvature transfers independently.

The frozen cheapest falsification uses the 39 physically tied antithetic
proportional log-tilt pairs. Only pair-odd effects select the prior pseudocount,
ridge, and demand amplitudes. The pair-even effects relative to the mean of 11
proportional controls are not used for selection and test the production law's
implied curvature. Both Uncheatable and Table-9 must pass; in-sample odd fit is
declared non-evidence. A failure blocks further tied-panel fitting rather than
inviting another pseudocount, occupancy exponent, or output link.

### Frozen result: rejected before endpoint fitting

Protocol `d39c727b7166e53d8605940778cd5036b80ab7c2451fae5ce64f025af4f63f28`
selected the shared interior pseudocount `epsilon=0.03` using pair-odd outcomes
only. The result fails in the same direction on both targets:

| target | odd CV RMSE | odd zero RMSE | odd improvement | even RMSE | even zero RMSE | even improvement | even Spearman | sign accuracy | calibration slope |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Uncheatable | 0.002903 | 0.002912 | +0.3% | 0.001407 | 0.000963 | -46.0% | -0.271 | 0.359 | -1.11 |
| Table-9 | 0.006091 | 0.006012 | -1.3% | 0.004474 | 0.003693 | -21.1% | -0.308 | 0.462 | -1.72 |

The global repeated-mass coefficient is numerically zero on both targets. The
independently implied even curvature is not merely weak: it is anti-correlated
with the measured curvature and has the wrong calibration sign. Uncheatable's
bootstrapped demand direction is also unstable (median cosine `0.759`, below the
frozen `0.8` gate); Table-9 demand stability cannot rescue the failed parity
prediction.

**Decision: reject before the 282-row tied endpoint fit.** Conserved unique
evidence and target-demand cross-entropy do not supply the missing aggregate
production law. Do not reopen through another pseudocount, occupancy exponent,
ridge, replay feature, bounded output link, or direct endpoint fit. The failure
also sharpens the search: the needed compositional coupling cannot be a static
normalization of independently accumulated bucket evidence.

Artifacts:

- `reference_outputs/unique_evidence_demand_allocation_20260731/report.md`
- `reference_outputs/unique_evidence_demand_allocation_20260731/aggregate_metrics.csv`
- `reference_outputs/unique_evidence_demand_allocation_20260731/pair_predictions.csv`
- `reference_outputs/unique_evidence_demand_allocation_20260731/odd_to_even_transfer.html`

## 2026-07-31: odd-only Fisher-Rao demand overlap is structurally underidentified

`WSD80-SUR-066` proposed the bounded aggregate law

\[
B(w,q)=\sum_i\sqrt{q_iw_i},\qquad
A(w)=b+M\{1-B(w,q)\}+hR(w),
\]

where `q` is a target-demand distribution and `R(w)` is exact repeated
materialized mass. Writing `a_i=M sqrt(q_i)` gives a convex linear head on
`-sqrt(w_i)` with `a_i>=0`; `M=||a||_2` and
`q_i=a_i^2/||a||_2^2`. Unlike the old Power-Ridge baseline, this law has no
free signed linear terms. Without replay its unique raw optimum is `w=q`, and
the deficit `1-B` is bounded on the simplex.

The first proposed estimator used only the 39 pair-odd effects from the central
proportional log tilts to identify 39 demand amplitudes and one replay
amplitude. A frozen no-outcome rank audit blocked that estimator:

- `39` antithetic directions;
- `40` mechanistic coefficients;
- scaled design rank `39` and nullity `1`;
- nonzero singular-value condition number `1.740`.

The exact null direction mixes the replay coefficient with all 39 demand
amplitudes. Ridge would choose a unique numerical solution, but it would not
identify `q` or `h`; bootstrap stability under the same penalty would not fix
that structural defect. No target outcome was evaluated. Protocol
`0897a62db7187fdaa1407ccea9f7a9b7ca69d4c437d5a978d83feab55845ca90`
therefore hard-blocks `--mode evaluate`.

Artifacts:

- `reference_outputs/fisher_rao_demand_overlap_20260731/protocol.json`
- `reference_outputs/fisher_rao_demand_overlap_20260731/preflight.json`
- `reference_outputs/fisher_rao_demand_overlap_20260731/design_decision.json`

## 2026-07-31: tied-identified Fisher-Rao aggregate preregistration

`WSD80-SUR-067` preserves the same production law but uses the data source
specified by the aggregate-spine objective. All coefficients and
hyperparameters are fit only on the `282` physically tied 300M policies:

\[
A(w)=c-\sum_i a_i\sqrt{w_i}+hR(w),\qquad a_i\ge0,\ h\ge0.
\]

Here `c=b+M`, `M=||a||_2`, and `q_i=a_i^2/M^2`. Mixture weights, materialized
epochs, overlap, and replay mass are dimensionless; `c`, `a_i`, `M`, and `h`
have BPB units. There is no continuous parameter symmetry after recovering
`M` and `q`, and the raw tied optimization is convex. The 39 proportional
antithetic pairs do not fit or select any parameter; they are an external test
of local odd and even geometry.

Nearest routes are old Power-Ridge, `prior_L`, `SUR-059`, `SUR-063`, `SUR-065`,
and `SUR-066`. The material novelty relative to Power-Ridge is not the square
root itself: it is the removal of free signed linear terms and the restriction
to one normalized nonnegative demand distribution plus literal replay. Relative
to `SUR-065`, the production invariant changes from KL mismatch in normalized
unique evidence to bounded Bhattacharyya overlap in the mixture allocation.

The frozen evaluation will use five correspondence-grouped outer folds and
three grouped inner folds. Ridge is selected only from
`{0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}` by inner tied-policy RMSE. Both
Uncheatable and Table-9 are mandatory. Promotion to temporal-state work
requires all of the following:

1. tied grouped-OOF RMSE within `5%` of the frozen independently fitted
   one-phase references (`0.0047134` Uncheatable and `0.0103573` Table-9);
2. external antithetic odd RMSE improvement at least `20%` over zero;
3. external even RMSE improvement at least `5%`, paired-bootstrap lower bound
   nonnegative, Spearman at least `0.25`, and sign accuracy at least `0.60`;
4. a raw optimum no more than `0.02` BPB below the best observed tied policy,
   nearest aggregate TV at most `0.20`, maximum bucket weight no more than
   `0.05` above observed support, and maximum epochs no more than `1.25x`
   observed support;
5. outer-fold and group-bootstrap optimum median L1 at most `0.25` and maximum
   L1 at most `0.75`, with median cross-fold demand cosine at least `0.80`.

The exact metric definitions, source hashes, folds, and bootstrap seeds will be
sealed in `protocol.json` before any candidate result is generated. Failure
cannot be repaired with signed coefficients, free linear terms, another power,
output calibration, a trust region, or deployment regularization.

### Frozen result: rejected

Protocol `97fd00958363566547ce98b82e137224a75639482deb460367e4e749f954f7aa`
was evaluated unchanged. The demand direction is highly stable across folds,
but the production law fails both mandatory targets:

| target | tied OOF RMSE | relative to reference | pair-odd improvement | pair-even improvement | even Spearman | even sign accuracy | raw support TV | frontier optimism |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Uncheatable | 0.009553 | +102.7% | 47.8% | 18.9% | 0.310 | 0.590 | 0.419 | 0.068 BPB |
| Table-9 | 0.015370 | +48.4% | 34.4% | 6.6% | 0.216 | 0.487 | 0.435 | 0.133 BPB |

The pair-even point estimates improve on zero, but both bootstrap lower bounds
are negative and the preregistered rank/sign gates fail. Exact replay harm
collapses to numerical zero on both targets. The raw optima are stable across
folds and conditional group bootstraps, yet remain far outside empirical
support and substantially below the observed tied frontiers. This is therefore
stable structural misspecification, not an optimizer or variance problem.

**Decision: reject the aggregate route and do not proceed to a temporal-state
fit.** A sum of independent square-root bucket credits does not adequately
price substitution among 39 buckets. Do not reopen through signed or linear
credits, another power, ridge, output calibration, a trust region, or
deployment regularization.

Artifacts:

- `reference_outputs/fisher_rao_tied_spine_20260731/protocol.json`
- `reference_outputs/fisher_rao_tied_spine_20260731/report.md`
- `reference_outputs/fisher_rao_tied_spine_20260731/metrics.csv`
- `reference_outputs/fisher_rao_tied_spine_20260731/decision.json`

## 2026-07-31: component-observed phase-local relaxation preregistration

The prior trajectory audit cached only aggregate Uncheatable BPB, but the raw
W&B histories contain the seven constituent component BPBs at every one of the
same checkpoints. `WSD80-SUR-068` uses these as a new independent observable to
test a phase-local relaxation law before another endpoint phase model is
allowed.

For an exact aggregate-matched asymmetric/tied pair, let `d0_c` be component
`c`'s observed BPB difference immediately before the phase switch. If the
component response relaxes toward the current phase's local equilibrium, then

\[
d_c(s)=d_{1,c}+(d_{0,c}-d_{1,c})\exp(-\lambda s),
\qquad
d_{1,c}=-\frac{\beta_0}{\beta_1}\gamma d_{0,c}.
\]

Here `s` is phase-1 progress, `lambda` is a dimensionless relaxation rate, and
`gamma` is the late-versus-early equilibrium-response ratio. The factor
`beta0/beta1` is fixed by aggregate preservation, not fitted. Both parameters
are shared across all seven components. The tied restriction is exact because
`d0_c=0` implies `d_c(s)=0` for all `s`.

This is distinct from SUR-060/061: their competence ODEs generated a latent
state from bucket exposure and mapped it to one aggregate target. SUR-068 first
tests the transition directly against observed component states. It is also
distinct from RPL's endpoint-selected late multiplier: `gamma` and `lambda`
are selected only from component trajectories ending by step 21000.

The frozen first stage reserves step 22000 and the final step 22887, uses no
aggregate Uncheatable value for parameter selection, and leaves Table-9 and
WSD80 untouched. Correspondence-grouped folds and leave-one-component-out fits
must show stable interior parameters. At the untouched step 22000, component
and aggregate predictions must improve both zero-effect and persistence nulls,
with stable rank, sign, and calibration. The final aggregate endpoint must
approach HPR's exact-pair RMSE without refitting. A pass identifies only the
relaxation law; it does not identify the policy-to-equilibrium map or license a
full surrogate.

Failure cannot be repaired with component-specific rates, free time
calibration, endpoint tuning, bucket-resolved response fields, or an additional
timescale.

### Frozen result: rejected

Protocol `5f21792b08844e9167a22b51de728f99576a57b59eef95c2b23cef5f155d0768`
was evaluated unchanged after three pre-fit protocol errata. The errata corrected
JSON canonicalization, replaced sampled W&B history with exhaustive
`scan_history`, and froze exact complete-case sets after a missingness-only
audit. No parameter was fit and no BPB outcome was inspected before the final
protocol was sealed.

The selected shared parameters are interior and stable:

- late-equilibrium multiplier `gamma = 0.172162`;
- relaxation rate `lambda = 11.517264` per normalized phase-1 duration;
- fold CVs `0.0326` and `0.0219` for `gamma` and `lambda`;
- leave-component-out CVs `0.0971` and `0.0397`.

Despite that stability, the transition misses the frozen magnitude and endpoint
gates:

| scope | RMSE | zero improvement | Spearman | slope | amplitude ratio | bias |
|:--|--:|--:|--:|--:|--:|--:|
| fit-step component OOF | 0.02692 | 12.6% | 0.691 | 1.269 | 0.503 | -0.01056 |
| step-22000 components | 0.02626 | 18.1% | 0.734 | 1.222 | 0.572 | -0.00864 |
| step-22000 aggregate | 0.01531 | 13.4% | 0.761 | 1.512 | 0.626 | -0.00848 |
| final components | 0.02623 | 16.5% | 0.724 | 1.138 | 0.584 | -0.00822 |
| final aggregate | 0.01540 | 12.8% | 0.731 | 1.484 | 0.618 | -0.00816 |

The transition substantially improves on persistence and recovers much of the
phase-order rank, but it systematically predicts too-negative endpoint deltas
and only about 60% of the observed aggregate amplitude. Component behavior is
not a scalar rescaling of the pre-switch difference: Wikipedia is the clearest
counterexample, worsening the zero predictor by `20.9%` at step 22000 and
`52.3%` at final, while code and arXiv components retain useful rank signal.

**Decision: reject the shared scalar relaxation law.** This is structural
misspecification, not parameter instability. Reopen only if an independently
observed state explains component-specific late equilibria and the common
aggregate bias. Component-specific rates, free time calibration, endpoint
tuning, bucket-response fields, and extra exponential timescales are not
admissible repairs.

Artifacts:

- `reference_outputs/component_phase_relaxation_20260731/protocol.json`
- `reference_outputs/component_phase_relaxation_20260731/report.md`
- `reference_outputs/component_phase_relaxation_20260731/metrics.csv`
- `reference_outputs/component_phase_relaxation_20260731/component_metrics.csv`
- `reference_outputs/component_phase_relaxation_20260731/holdout_predictions.csv`
- `reference_outputs/component_phase_relaxation_20260731/phase_local_relaxation.html`

## 2026-07-31: SUR-069/070 identify a policy-computable switch transient

SUR-069 tested a state that was observed independently of endpoint BPB: the
asymmetric-minus-tied jump in total gradient norm and training loss immediately
across the 300M phase boundary. On 229 complete exact pairs, the gradient shock
predicts the common residual left by SUR-068 at steps 19000 and 20000
(`Spearman=0.340/0.250`; zero-predictor RMSE improvement `25.3%/21.6%`). The
association decays toward final. Static TV, KL, JS, Hellinger, and Fisher
coordinates have near-zero shock rank, so this does not license another global
divergence penalty.

SUR-070 then froze a policy-input map under protocol
`b1d76f86d7ebe0dabfc6e5ae7f7b2c76049884873b05d1291d32db65128a2c3e`. For
each predeclared family `f`, its new state is counterfactual late
unfamiliarity:

```text
U_f = sum_{i in f} (wbar_i * ebar0_i - w1_i * e0_i).
```

`U_f` is the phase-0 materialized exposure expected under the tied late
distribution minus the exposure expected under the actual phase-1
distribution. It has epoch units, is exactly zero for tied policies, and has no
fitted bucket parameter. Family mass shift controls static phase-1 composition;
a separate late-static arm adds phase-1 repetition intensity.

Under five outer mixture-space blocks with nested ridge selection, the simpler
family-shift plus unfamiliarity block reaches OOF Spearman `0.748` for gradient
shock and `0.912` for training-loss shock, improving train-fold-mean RMSE by
`34.6%` and `62.9%`. All family unfamiliarity coefficients have the expected
positive sign in all five folds. The nominal full block beats the late-static
control in every fold with paired-bootstrap RMSE-difference intervals
`[-0.0691,-0.0455]` and `[-0.1950,-0.1407]`, proving that prior phase-0 state
adds information beyond static late-batch composition and repetition. Late
repetition itself worsens both targets relative to the simpler cross-phase
block and is rejected.

Without fitting an endpoint amplitude, OOF predicted gradient shock retains
the observed shock's smooth-target residual rank at steps 19000 and 20000:
`Spearman=0.376/0.202`, or `111%/81%` rank retention. By step 22000 and final,
the rank falls to `0.056/0.053`, consistent with a transient rather than a
persistent endpoint state.

**Decision:** SUR-069 and SUR-070 are positive identification diagnostics, not
surrogates. The surviving state is family shift plus counterfactual late
unfamiliarity; the late-repetition channel is removed. The next admissible test
is a frozen bounded transition whose rate and response are selected only from
pre-22000 smooth-target dynamics. No endpoint correction or aggregate-spine
change is licensed.

The shock outcomes and SUR-068 residual trajectories were exposed during this
development round. Any result here remains provisional and requires a future
untouched confirmation design.

Artifacts:

- `reference_outputs/switch_gradient_shock_20260731/report.md`
- `reference_outputs/policy_predictable_switch_shock_20260731/protocol.json`
- `reference_outputs/policy_predictable_switch_shock_20260731/report.md`
- `reference_outputs/policy_predictable_switch_shock_20260731/metrics.csv`
- `reference_outputs/policy_predictable_switch_shock_20260731/transfer_metrics.csv`
- `reference_outputs/policy_predictable_switch_shock_20260731/policy_predictable_switch_shock.html`

## 2026-07-31: SUR-071 rejects exponential persistence of switch shock

SUR-071 froze the minimal transition

```text
h_p(s) = q_p exp(-lambda s)
r_p(s) = a h_p(s)
```

where `q_p` is SUR-070's blocked-OOF cross-phase prediction of the observed
gradient shock. The nonnegative BPB amplitude `a` and dimensionless decay rate
`lambda` were fit only on common SUR-068 residuals at steps 19000--21000. Step
22000 and final were applied without refitting.

The shock state is useful relative to no correction: pooled fit-step OOF RMSE
improves the zero residual by `15.5%`. The exponential transition itself is not
identified. The static-shock ablation is significantly better on the fit
window, with dynamic-minus-static RMSE bootstrap interval
`[+0.000015,+0.000054]` and only one of five outer-fold wins. Two folds select
exactly zero decay. The full decay rate is `0.160`, corresponding to a half-life
of `4.33` entire phase-1 durations, effectively static over this schedule.

The tiny fitted decay happens to avoid some static-shock harm at step 22000 and
final, but those outcomes were already exposed and cannot rescue the failed fit
identification. The result distinguishes a policy-predictable boundary
diagnostic from a durable temporal state: the former is real, while the latter
is not established.

**Decision:** reject SUR-071. Do not add another exponential, persistent
offset, component-specific rate, or endpoint-selected calibration. A durable
state now requires a new independently observed quantity or a switch-time
intervention.

Artifacts:

- `reference_outputs/shock_initiated_transient_20260731/protocol.json`
- `reference_outputs/shock_initiated_transient_20260731/report.md`
- `reference_outputs/shock_initiated_transient_20260731/metrics.csv`
- `reference_outputs/shock_initiated_transient_20260731/shock_initiated_transient.html`

## 2026-07-31: family-anisotropic Hellinger aggregate preregistration

`WSD80-SUR-072` reopens the rejected Fisher-Rao aggregate route only through a
family-specific metric tensor. It does **not** restore the old Power-Ridge
model's 39 free signed linear terms. In an identified zero-sum family gauge,

\[
A(w)=c+\sum_f k_f W_f-\sum_i a_i\sqrt{w_i}+hR(w),
\qquad a_i\ge0,\ h\ge0,\ \sum_f k_f=0.
\]

For a scalar gauge shift `lambda > -min_f k_f` satisfying

\[
\sum_i\left\{\frac{a_i}{2(k_{f(i)}+\lambda)}\right\}^2=1,
\]

define `K_f=k_f+lambda` and
`q_i={a_i/[2K_{f(i)}]}^2`. The response is then, up to its intercept,

\[
A(w)=b+\sum_f K_f\sum_{i\in f}
       \left(\sqrt{w_i}-\sqrt{q_i}\right)^2+hR(w).
\]

Thus the two new coefficients are the identifiable anisotropy of a bounded
Hellinger bowl across the predeclared `broad_text`, `reasoning`, and
`tech_code` families. The isotropic restriction `k_f=0` is exactly
`WSD80-SUR-067`. The raw optimization remains convex. A fit that cannot recover
positive `K_f`, that zeros any bucket amplitude, or that loses this mapping in
any outer fold is an algebraic failure rather than an invitation to reinterpret
the family slopes.

Protocol
`d2f72f62d064b7d661662f53c3bd88af3b44ff8d55c25b10afe8bdd2fea204b7`
was frozen before target evaluation. The tied design has rank `43` for `43`
nominal parameters on both targets. No asymmetric row or proportional
intervention outcome enters fitting or selection. In addition to the inherited
SUR-067 OOF, external odd/even, raw-optimum, and stability gates, both targets
must beat the isotropic ablation with a correspondence-cluster bootstrap upper
bound no greater than zero and at least four of five outer-fold wins.

Artifacts:

- `reference_outputs/family_anisotropic_hellinger_spine_20260731/protocol.json`
- `reference_outputs/family_anisotropic_hellinger_spine_20260731/preflight.json`

## 2026-07-31: family-anisotropic Hellinger aggregate result

`WSD80-SUR-072` is rejected under frozen protocol
`d2f72f62d064b7d661662f53c3bd88af3b44ff8d55c25b10afe8bdd2fea204b7`.
The result is structural rather than an optimization or identifiability failure.
The positive-curvature Hellinger-bowl mapping exists in the full fit and every
outer fold, and the learned family-demand directions are stable, but the model
does not improve its isotropic ablation and remains severely overoptimistic.

For Uncheatable, OOF RMSE is `0.009848`, `+108.9%` relative to the
`0.004713` reference and `+0.000154` relative to SUR-067. The paired-bootstrap
candidate-minus-ablation interval is `[-0.000091,+0.000420]`, with only one of
five fold wins. For Table-9, OOF RMSE is `0.015776`, `+52.3%` relative to the
`0.010357` reference and `-0.000033` relative to SUR-067; its interval is
`[-0.000415,+0.000378]`, with four of five wins. Neither target clears the
ablation gate.

The raw optima are still unsupported. Their support TV distances are
`0.394`/`0.463` and their optimism is `0.062`/`0.146` BPB for
Uncheatable/Table-9. Only `36/39` and `37/39` bucket amplitudes remain active,
and the replay coefficient collapses to numerical zero. The external
aggregate-preserving intervention audit also fails its even-effect sign and
uncertainty gates.

**Decision:** family anisotropy is identified but is not the missing bounded
aggregate mechanism. Do not reopen this route through another family basis,
metric tensor, curvature grid, free bucket linear terms, output link, trust
region, or deployment regularization. A successor needs an independently
measured composition-production state or a materially different bounded
coupling with an externally falsifiable finite-move prediction.

Artifacts:

- `reference_outputs/family_anisotropic_hellinger_spine_20260731/report.md`
- `reference_outputs/family_anisotropic_hellinger_spine_20260731/metrics.csv`
- `reference_outputs/family_anisotropic_hellinger_spine_20260731/raw_optima.csv`
- `reference_outputs/family_anisotropic_hellinger_spine_20260731/fold_parameters.csv`

## 2026-07-31: intervention-identified signed dose potential preregistration

`WSD80-SUR-073` moves aggregate-spine identification to the independent tied
conditional epoch-dose intervention. For proportional weights `p_i` and
relative materialized dose `r_i=w_i/p_i`, the target-specific response is

\[
A(w)=b+\sum_i p_i g_i(r_i-1)
     +\sum_f K_f\sum_{i\in f}p_i\Phi_q(r_i),
\qquad p^\top g=0,\quad K_f\ge0.
\]

`Phi_q` is the finite-at-zero convex Cressie-Read generator. The signed local
bucket utility `g`, generator order `q`, global-versus-family curvature, and
ridge are selected only on the complete 60M intervention panel through x16.
The x32 rows are extrapolation-only. The complete Delphi intervention panel
remains sealed unless both 60M target gates pass; 300M receives only the frozen
form, with strict source-potential transfer and coefficient-refit results
reported separately.

The final parent protocol is
`5cb1b104ae8b9c954418bdfa88e4ffe3c531c4c42f21d4ffda53e4620e089dc5`.
The initial staged evaluator protocol was
`4301321360d73aa8763db16741f37551ac35b56965ac6a2c3df6a2fa91bdaf74`.
Before complete materialization it was superseded by evaluator-v2 protocol
`4ae9eafb2cf0a9e076ffe4fc69b19c1c3645eecc1eadaf65b2b03cfbeae1d0d3`.
V2 changes only training-run identity and exact-final-step persisted-metric
recovery; equations, candidate grid, folds, gates, and selection are unchanged.
Rerunning preparation reproduced its serialized file hash
`36ddba9caa76fd9dad3f0a79a6dba41985760634034c51161a1bc1360ca7fce6`.
The data-use ledger preserves both superseded pre-outcome protocols and records
that no full 60M, Delphi, or 300M outcome was materialized during protocol
development.

Algebraic preflight finds all eight candidate designs full rank over 277 tied
policies, with 38 identified utility directions and maximum gauge error
`1.69e-17`. A synthetic family-entropy target selected the true `q=0` family
form, recovered minimum curvature `0.0100`, and converged from all eight raw
optimizer starts with objective spread `2.13e-13`. The positive affine transfer
map and failed-gate sealing behavior also reproduced exactly. Black, Ruff,
Pyrefly, and targeted repository pre-commit checks pass.

The frozen 300M optimization gate now rejects a raw optimum if any of the
following holds: nearest-support TV exceeds `0.35`, maximum bucket weight
exceeds `0.30`, median bootstrap TV exceeds `0.10`, any selected curvature is
inactive, or optimism relative to the best observed tied policy exceeds two
candidate OOF RMSEs. These are model-form falsification gates, not deployment
regularization.

Artifacts:

- `reference_outputs/intervention_identified_signed_dose_potential_20260731/protocol.json`
- `reference_outputs/intervention_identified_signed_dose_potential_20260731/evaluation_protocol.json`
- `reference_outputs/intervention_identified_signed_dose_potential_20260731/data_use_ledger.csv`

## 2026-07-31: SUR-073 materialization remains sealed on incomplete coverage

The first permitted 60M materialization attempt stopped before fitting or
selection because `19/277` policies lacked a complete paired target record.
Sixteen named gaps were native Table-9 evaluations that the live parent was
still computing. Three policies had no finite training outcome available from
the materializer: `p000_proportional_anchor`, `p241_d34_m0p25`, and
`p248_d35_m0`. No `observations_60m.csv` or `selected_60m.json` was written,
and Delphi outcomes remain sealed.

Iris showed both full parents running with zero logical failures and two
recovered worker preemptions. Live logs showed the missing 60M Table-9 tasks
advancing. The incomplete attempt is therefore an availability check, not a
candidate result; rerun materialization only after all 277 policies have
finite Uncheatable and Table-9 outcomes.

Artifact:

- `reference_outputs/intervention_identified_signed_dose_potential_20260731/materialization_missing_60m.json`

The recovery audit found two distinct provenance defects. The proportional
anchor carries the exact `p000_proportional_anchor` tag but uses an underscore
display-name convention. Two preempted runs, `p241_d34_m0p25` and
`p248_d35_m0`, have complete step-4576 `eval_metrics.jsonl` records in their
declared checkpoint roots despite missing W&B final summaries. Evaluator v2
matches the exact tag first and recovers only the manifest's exact expected
final step. Every retry must still agree within `1e-10` BPB.

## 2026-07-31: SUR-074 persistent radial parameter displacement preregistration

SUR-071 rejected the hypothesis that the policy-predicted boundary gradient
shock itself persists as an exponential endpoint state. The W&B histories also
contain an independently logged quantity that had not been evaluated across
exact pairs: `params/norm/total`. SUR-074 tests whether phase 1 instead creates
a durable radial parameter displacement relative to the exact tied run.

For pair `p`, define

```text
n_p(s) = log(||theta_2p(s)||_2) - log(||theta_tied(s)||_2)
d_p(s) = n_p(s) - median_pre_switch[n_p].
```

The transition is normalized by terminal phase progress:

```text
d_p(s) = q_p g_k(s)
g_k(s) = (1 - exp(-k s)) / (1 - exp(-k)),  g_0(s) = s.
```

The shared saturation rate is selected only from parameter telemetry through
step 21000; step 22000 and the final logged parameter norm are temporal
holdouts. The terminal state `q_p` is mapped from the already predeclared
SUR-070 family-shift and counterfactual-late-unfamiliarity features under its
frozen mixture blocks. One signed response scale is fit on common smooth-target
residuals at steps 19000--21000. Step 22000 and the final BPB endpoint are
falsification rows, with zero and static-terminal-state ablations.

Protocol
`d0ccee45eec68c292b8748f25742ee905d373a0b08016ad0b8edfb732214b741`
was frozen before any asymmetric-minus-tied parameter-norm trajectory was
materialized. Before freeze, only the W&B history-key availability and unpaired
parameter-norm range of one one-phase run were inspected. This is exposed
development evidence and cannot confirm a surrogate.

Artifact:

- `reference_outputs/persistent_parameter_displacement_20260731/protocol.json`

## 2026-07-31: SUR-074 identifies durable radial displacement but rejects it as a performance state

SUR-074 fails its frozen development gate under protocol
`d0ccee45eec68c292b8748f25742ee905d373a0b08016ad0b8edfb732214b741`.
The parameter trajectory itself is strongly identified. The shared transition
selects the interior saturation rate `k=2` in all five outer omissions,
improves fit telemetry by `23.1%` over the linear-displacement ablation, and
beats linear on both step-22000 and final-telemetry holdouts. State rank is
extremely persistent from step 21000 to 22000 (`Spearman=0.978`).

The state is not sufficiently policy-predictable or performance-relevant.
Adding counterfactual late unfamiliarity to family shift improves the OOF
policy map in `5/5` folds with bootstrap RMSE-difference interval
`[-0.003280,-0.000370]`, but absolute prediction reaches only
`Spearman=0.206` and `5.4%` improvement over zero. More importantly, the
dynamic BPB response is significantly worse than its static-state ablation
(`dynamic-static` RMSE interval `[+0.000102,+0.000441]`). It worsens the zero
predictor by `8.4%` at step 22000 and `15.6%` at final, with final
`Spearman=-0.159`.

**Decision:** reject scalar radial parameter displacement as the temporal
performance state. It is a genuine durable optimizer coordinate, but its
direction in parameter space matters and total norm discards that direction.
Do not repair this route with per-layer response coefficients, another
timescale, endpoint calibration, or a larger grid. A reopen requires an
independently observed directional state or a switch-time intervention.

Artifacts:

- `reference_outputs/persistent_parameter_displacement_20260731/report.md`
- `reference_outputs/persistent_parameter_displacement_20260731/acceptance_gate.csv`
- `reference_outputs/persistent_parameter_displacement_20260731/policy_metrics.csv`
- `reference_outputs/persistent_parameter_displacement_20260731/response_metrics.csv`
- `reference_outputs/persistent_parameter_displacement_20260731/persistent_parameter_displacement.html`

## 2026-07-31: SUR-075 architecture-relative parameter redistribution preregistration

SUR-074 established that the scalar parameter-norm radius is durable but not a
performance state. W&B also logs 110 per-tensor parameter norms. SUR-075 removes
the global radius and asks whether phase schedules induce one stable, signed
redistribution direction among eleven predeclared architecture modules:

```text
u_pg(s) = log(||theta_2p,g|| / ||theta_2p,total||)
          - log(||theta_tied,g|| / ||theta_tied,total||)
d_pg(s) = u_pg(s) - median_pre_switch[u_pg].
```

The groups are embeddings, both layer-normalization positions, attention
Q/K/V/O, MLP gate/up/down, and final normalization. One uncentered principal
direction is learned from the step-21000 telemetry matrix only. Projected
scores follow the same normalized bounded transition family as SUR-074, with
the shared rate selected from telemetry through step 21000. Step 22000 and the
final parameter telemetry are temporal holdouts. A nested policy map uses the
frozen SUR-070 mixture blocks and telemetry scores only. One signed response
scale is fit on prefinal common smooth-target residuals; step 22000 and final
BPB are falsification rows.

The construction deliberately forbids per-module BPB response coefficients,
additional telemetry directions, endpoint-selected centering or component
count, another timescale, persistent offsets, calibration, and post-outcome
grid changes. Multiplying every parameter norm by a common factor leaves the
state invariant, and a tied policy compared with itself has zero state.

Protocol
`6d924276c9597ef506d723810e703cdde96a955e41cef4c45ac6ee658a70ee8b`
was frozen before inspecting any asymmetric-minus-tied module-relative
trajectory, telemetry principal direction, explained energy, or relationship
to BPB. Before freeze, only W&B key names, key counts, and the existence of 110
per-tensor norms were inspected. A pass licenses one directional state for a
nested surrogate ablation after aggregate-spine selection; a failure closes
current parameter-norm telemetry and leaves switch-time intervention as the
next admissible identification route.

Artifact:

- `reference_outputs/architecture_relative_parameter_state_20260731/protocol.json`

## 2026-07-31: SUR-075 finds a stable directional state that is anti-predictive of BPB

SUR-075 fails the frozen development gate under protocol
`6d924276c9597ef506d723810e703cdde96a955e41cef4c45ac6ee658a70ee8b`.
This is not an identification or policy-map failure. The first telemetry-only
direction explains `89.7%` of module-relative displacement energy. Its minimum
outer-fold cosine is `0.998`, its bootstrap cosine lower bound is `0.999`, and
the selected interior saturation rate is `k=1` in all five folds. The bounded
transition improves telemetry fit over linear by `26.6%`, remains better at
both telemetry holdouts, and preserves rank from step 21000 to 22000 at
`Spearman=0.989`. The policy-input map is also strong: OOF `Spearman=0.893`
and `49.4%` RMSE improvement over the zero-state predictor.

The smooth-target response nevertheless fails every magnitude and transfer
gate. The dynamic correction worsens the zero predictor by `1.4%` on its
prefinal fit rows, `7.9%` at step 22000, and `8.4%` at final. It is significantly
worse than the static-state ablation, with paired-bootstrap dynamic-minus-static
RMSE interval `[+0.000004,+0.000293]`, and final rank is reversed
(`Spearman=-0.362`). All five response folds choose the same negative sign, so
the failure is stable rather than a sign convention or optimizer accident.

The telemetry direction is almost entirely final-normalization redistribution:
the final-norm loading is `0.996`; every main attention and MLP matrix has a
loading near `-0.025`. This is a real, durable, policy-computable optimizer
coordinate, but it tracks architecture rescaling rather than the missing
terminal performance state.

**Decision:** reject SUR-075 and close current parameter-norm telemetry. Do not
post-hoc exclude final norm, add a second principal direction, fit per-module
BPB coefficients, change centering or grids, or recalibrate the endpoint. The
next admissible temporal identification route is a switch-time intervention.

Artifacts:

- `reference_outputs/architecture_relative_parameter_state_20260731/report.md`
- `reference_outputs/architecture_relative_parameter_state_20260731/acceptance_gate.csv`
- `reference_outputs/architecture_relative_parameter_state_20260731/direction_loadings.csv`
- `reference_outputs/architecture_relative_parameter_state_20260731/policy_predictions.csv`
- `reference_outputs/architecture_relative_parameter_state_20260731/response_predictions.csv`
- `reference_outputs/architecture_relative_parameter_state_20260731/architecture_relative_parameter_state.html`

## 2026-07-31: fixed-model scale-specific tied fibers locally support fiber optimality

The completed 132-run StarCoder 80/20 WSD panel tests aggregate-held phase
fibers at 1B, 2B, 4B, and 8B materialized tokens. The fixed model has about
157.5M total parameters, so the ladder spans total-parameter TPP 6.35, 12.70,
25.40, and 50.79. Each fiber uses

```text
p0 = a - 0.2 d
p1 = a + 0.8 d
```

and therefore holds the 80/20 aggregate `a` exactly fixed. The reference seed
samples `d` from -0.25 to +0.25; `d=-0.20, 0, +0.20` receive four additional
joint-randomness seeds. Forty-three W&B runs had stale intermediate summaries
despite completed checkpoints, so their final values were recovered from
persisted `checkpoints/eval_metrics.jsonl`; all 138 observations are complete.

For each anchor, the sign at `|d|=0.20` was selected using only the reference
seed and then evaluated on the four fresh seeds. At the tested anchors nearest
the local-quadratic tied-optimum estimates, no replicated improvement is
detected:

| tokens | tested `a` | estimated tied optimum | fresh mean two-phase - tied BPB | 95% CI |
|---:|---:|---:|---:|---:|
| 1B | 0.30 | 0.2865 | +0.000978 | [-0.003734, +0.005691] |
| 2B | 0.40 | 0.3847 | +0.000329 | [-0.005624, +0.006283] |
| 4B | 0.55 | 0.5336 | +0.001264 | [-0.000972, +0.003501] |
| 8B | 0.75 | 0.7290 | -0.000032 | [-0.002222, +0.002158] |

All four one-sided Holm-adjusted p-values are 1.0. Thus the panel does not
contradict, and locally supports, the hypothesis that no policy on the globally
optimal tied policy's fiber improves on that tied policy. It does not prove the
global statement: only `|d| <= 0.25` was sampled, only `|d|=0.20` was repeated,
and the population tied optimum is estimated rather than known.

The important exception is the 2B `a=0.35` measured-grid anchor. Its
reference-selected `d=+0.20` improves by -0.003860 BPB on the four fresh seeds
(95% CI [-0.006175, -0.001545], one-sided p=0.00653, Holm p=0.0261 across the
four measured-grid anchors). All four fresh seeds improve. Because `a=0.35` is
off the estimated tied optimum and tied `a=0.35` versus `a=0.40` is itself
indistinguishable at five seeds, this is evidence that phase gain depends on
the aggregate. It supports the profile view: a globally best two-phase policy
may lie on another aggregate's fiber even if the tied optimum's own fiber is
locally null.

Artifacts:

- `reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/report.md`
- `reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/scale_tied_fibers.html`
- `reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/fresh_seed_confirmation.csv`
- `reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/hypothesis_verdict.json`

## 2026-07-31: erratum — the 2B fiber result rejects a hard fiber-optimality constraint

The preceding interpretation put too much weight on the local-quadratic point
estimate `a*=0.3847` and incorrectly described the overall evidence as locally
supporting fiber optimality. The corrected conclusion separates three claims:

1. The literal statement about the exact population tied optimum remains
   unresolved because that optimum and its complete feasible fiber are not
   observed.
2. The finite-grid version is contradicted at 2B. The reference-seed tied-grid
   minimum is `a=0.35`, and `d=+0.20` improves it by `0.003860` BPB over four
   fresh seeds (95% CI for two-phase minus tied
   `[-0.006175,-0.001545]`, Holm `p=0.0261`). All four fresh seeds improve; all
   five seeds improve when the sign-selection seed is included, with mean gain
   `0.003215` BPB.
3. A robust tied-optimal-set version is also contradicted. The tied difference
   `L(a=0.40)-L(a=0.35)` is `-0.000152` BPB over five matched seeds, with 95% CI
   `[-0.002140,+0.001836]` and `p=0.842`. The data cannot distinguish the two
   anchors, so dismissing `a=0.35` as off-optimum is unsupported. In a
   post-hoc matched comparison, the asymmetric `a=0.35,d=+0.20` policy
   also beats tied `a=0.40` in all five seeds by `0.003063` BPB (95% CI
   `[0.002188,0.003938]` as a positive gain), so it beats both measured tied
   candidates bracketing the quadratic estimate. This cross-anchor comparison
   strengthens interpretation but is not the multiplicity-adjusted primary
   test.

The repeated `|d|=0.20` tests at 1B, 4B, 8B, and 2B `a=0.40` do not detect
improvements. These are radius-specific null results, not evidence for a
universal statement over an unknown optimum's whole fiber. At 2B `a=0.40`, the
reference seed also favors `+d` at `|d|=0.05,0.10,0.15,0.20`; only the last
radius was repeated, where the mean effect is null.

**Revised decision:** do not encode “the optimal tied policy is fiber-optimal”
as a hard surrogate constraint. It remains a valid implication of a globally
sufficient phase-weighted-dose model with tied reachability, and can be used as
a falsifiable local null or soft prior. Empirically, phase response must remain
aggregate-conditioned and able to improve policies inside the tied-optimal
uncertainty set.

## 2026-07-31: SUR-076 switch-time intervention is blocked before training

The switch-time design was revised after the 2B fiber counterexample so its
direct response could represent a gain inside the tied-optimal uncertainty
region:

```text
Delta L = {theta_0 + theta_1(a - 0.35)} d_x + theta_2 d_x^2.
```

The phase-weighted-potential response remains only a constrained null. The
five-anchor design contains 115 unique policies and 290 observations, including
32 antithetic pairs, fixed-phase-0 and fixed-phase-1 arms, three asymmetric
seeds, and six tied-spine seeds. Protocol
`128af6a1a9c61e259da64b2fd44bf736e64d79df040b434b8251bae6c923813b`
passes source-hash validation, launcher dry-run, compile/lint, and a noiseless
end-to-end synthetic seal. The synthetic workflow recovers each response
restriction and blocks refitting after endpoint unseal.

This is not enough to launch. Fresh independent mechanistic and statistical
Opus 5 reviews both returned `REVISE`. The decisive failures are scientific:

- transition licensing gives half its weight to pre-switch rows, where a
  bounded dynamic state is also a static saturating exposure response and the
  static switch null is identically zero;
- three late switch folds have no post-switch transition observations, while a
  fourth has one;
- one tied-control run is shared across every coordinate at one anchor/seed,
  and leave-switch folds leak that persistent offset into the
  aggregate-conditioned response;
- response heads are nested but treated as competing forms, aggregate is
  nearly confounded with contrast magnitude, and the feature gate can be
  carried by the even column while signed columns remain aliased;
- arbitrarily slow or infinite relaxation and boundary rates can pass;
- the endpoint summary averages antithetic signs and therefore estimates the
  even response rather than the signed gain;
- the three-seed MDE is `0.008600` BPB versus the replicated `0.003860` BPB
  effect, and the twelve-cluster residual bootstrap omits selection variance.

The matched existing-data interaction between the `a=0.35` and `a=0.40`
`d=+0.20` effects is `-0.003272` BPB with 95% CI
`[-0.007720,+0.001176]` and `p=0.111`. Thus the gain at `a=0.35` is
established, but aggregate conditioning remains a hypothesis; it must compete
with an aggregate-invariant signed ablation.

**Decision:** block SUR-076 before training. A successor requires
post-switch-only licensing, seed-clean outer predictions, shape-matched static
equilibrium and decaying switch-shock nulls, finite observed relaxation with an
interior rate, column-wise signed-feature separation, matched contrast
magnitudes at separated anchors, direct odd/even estimands, and an enforced
full-pipeline synthetic null/power audit. No job was submitted.

Artifact:

- `reference_outputs/switch_time_intervention_design_20260731/independent_review_decision.md`

## 2026-08-04: cross-cell total-TPP diagnostic, with superseded interpretation

The initial preregistered interpretation below is retained for traceability.
The post-outcome adversarial correction later in this entry supersedes the
claim that total TPP is licensed as a moderator.

`WSD80-SUR-077` tested whether the missing cross-cell phase coordinate is tied
to model size, token horizon, or their ratio. The diagnostic fits each held
cell's aggregate response from its tied diagonal, excludes that cell's untied
outcomes from phase-law fitting and selection, and selects the clock family and
ridge inside nested leave-cell-out folds. The phase residual uses the same
low-dimensional odd/even basis in every cell. This is an exposed development
diagnostic, not a deployable model.

Protocol
`a38cd81d4616dd9888f6e824569dbc93d66d85c0a94e52079bcf2c66b478a29d`
passes every frozen licensing gate:

- the nested clock selector beats the scale-blind LR-dose-plus-Taylor baseline
  in all ten held cells, with exact one-sided sign-flip `p=1/1024`;
- mean cell RMSE falls from `0.046737` to `0.026042` BPB;
- total TPP is selected in nine of ten outer folds, while joint `D,N` is
  selected once;
- the descriptive direct total-TPP fit reaches `0.021877` mean cell RMSE.

The result is not sufficient for promotion. The nested selector still has an
observed-on-predicted slope of `0.414` (`0.503` for the direct total-TPP fit),
mean qualified optimum-coordinate error `0.106`, and RMSE `0.181521` on the
highest-token fixed-`N` cell. TPP is also constant within one 300M panel, so it
cannot identify the 39-bucket phase law there. The allowed conclusion is only
that one later mechanistic candidate may let total TPP moderate a physical
transition or response. Track indicators, per-cell heads, learned clock
exponents, and direct TPP output calibration remain forbidden.

Artifacts:

- `reference_outputs/wsd80_crosscell_phase_control_v3_20260804/protocol.json`
- `reference_outputs/wsd80_crosscell_phase_control_v3_20260804/report.md`
- `reference_outputs/wsd80_crosscell_phase_control_v3_20260804/phase_model_metrics.csv`
- `reference_outputs/wsd80_crosscell_phase_control_v3_20260804/structured_holdout_cell_metrics.csv`

### Post-outcome adversarial correction

Independent Opus 5 review identified that the frozen gate compares against
`lr_dose_plus_taylor`, which is itself worse than predicting zero phase effect.
A separate audit preserves the original preregistration and reaches a narrower
conclusion:

- the selector beats zero phase in 9/10 cells, but the exact one-sided
  sign-flip test is `p=0.204102` because the single loss is large;
- pooled RMSE is `0.058022` versus `0.045117` for zero phase, a `+28.6%`
  regression;
- on `r3_increase_d_h0640_s28260`, selector RMSE is `0.181521` versus
  `0.053074` for zero phase, and its support-boundary optimum predicts
  `0.317702` BPB gain versus the `0.005721` reference, a `55.5x`
  over-prediction;
- total and non-embedding TPP are not distinguished by paired cell evidence
  (two-sided sign-flip `p=0.117188`); and
- total TPP is strongly aliased with materialized tokens (`rho=0.932`), model
  size (`rho=-0.721`), optimizer steps (`rho=0.932`), and non-embedding TPP
  (`rho=0.891`).

**Revised decision:** `WSD80-SUR-077` is mixed evidence that a scale coordinate
matters, not a license to encode total TPP. A matched-overlap experiment must
separate `N`, `D`, data reuse, optimizer steps, and TPP before a clock enters a
surrogate. The proposed HPR/late-unfamiliarity composite is blocked: it reopens
SUR-048/SUR-053, `U_f` has already failed endpoint persistence in
SUR-071/074/075, SUR-073's sealed aggregate gate is unresolved, and TPP is
constant on the mandatory 300M panel.

Artifact:

- `reference_outputs/wsd80_crosscell_phase_control_adversarial_audit_20260804/report.md`

## 2026-08-04: intervention power replaces another endpoint-form iteration

Independent Opus 5 review blocked the proposed aggregate-potential plus
HPR/late-unfamiliarity composite. It reopens SUR-048 and SUR-053, uses SUR-070's
boundary transient despite failed endpoint persistence in SUR-071/074/075,
depends on unresolved SUR-073, and cannot identify a TPP coefficient within
the primary 300M panel. Endpoint-only phase-state iteration is therefore paused.

An outcome-free power audit under protocol
`29202828e23fda8cd5662f53700226855b5f7433a90fb9b26cdffb9bebf5fd69`
uses only exposed WSD80 repeats and architecture metadata. It corrects a prior
description: the measured fiber had 63 reference-seed coordinates, but only 11
coordinates with five seeds. Direct odd/even noise comes from 10 complete
antithetic triples with five same-seed observations each.

| estimand | pooled-RMS SD | maximum SD | repeats for 0.0039 | repeats for 0.0020 |
| --- | ---: | ---: | ---: | ---: |
| odd | 0.001363 | 0.001914 | 5 | 10 |
| even | 0.001996 | 0.002904 | 7 | 19 |
| better-orientation net | 0.002734 | 0.004439 | 13 | 41 |

Two designs fit the 200-run envelope without pretending that existing raw
minima identify a clock:

- `I2`: three matched-clock cells, each tied/+d/-d with 20 seeds, for 180
  runs. Worst-observed-noise MDEs are 0.001264 odd, 0.001918 even, and 0.002932
  BPB net. The h896 cells separately match the h640 base on total and
  non-embedding TPP, but token horizon remains aliased with optimizer steps
  unless batch size is also intervened on.
- `I3`: two aggregates, three switch times, antithetic arms, and 13 seeds, for
  182 runs. Its worst-observed-noise net MDE is 0.003759 BPB, sufficient for
  the replicated 0.0039-BPB effect but not a 0.002-BPB net effect.

**Decision:** no new experiment or surrogate is submitted. First obtain an
independent review of the estimands, remaining clock aliases, and allocation.
`Phi(TPP)` may enter a future model only if a matched intervention predicts a
held cell's phase-gain magnitude and optimum location; otherwise `Phi=1`.

Artifacts:

- `reference_outputs/two_phase_intervention_power_20260804/protocol.json`
- `reference_outputs/two_phase_intervention_power_20260804/report.md`
- `reference_outputs/two_phase_intervention_power_20260804/noise_estimates.csv`
- `reference_outputs/two_phase_intervention_power_20260804/design_envelopes.csv`

## 2026-08-04 02:37 PDT: intervention power v2 blocks both proposed protocols

The preceding power result remains as an audit trail but is superseded for
design decisions. Independent Opus 5 review found that it used a biased net
estimand and powered condition levels rather than the between-condition changes
needed to identify a transition or clock.

Protocol
`33b02c0b289fc44ca6c8596c191ca95028d8df2314688e6c472f4e85461eaaf6`
repairs those defects without reading a sealed endpoint panel or fitting a
surrogate:

- primary estimands are same-seed odd ordering effect and even asymmetry cost;
  `min(L(+d),L(-d))-L(0)` is descriptive only because the per-seed minimum is
  downward biased near a null;
- ten complete antithetic triples contain eight independent tied-control
  clusters; the three closest design-neighborhood triples provide 12 variance
  degrees of freedom;
- design-neighborhood odd/even SDs are `0.001615` and `0.001810` BPB, with 95%
  upper limits `0.002447` and `0.002742`;
- between-condition power assumes crossed seeds but zero covariance and thus
  multiplies single-condition SD by `sqrt(2)`;
- effect targets are `0.0039`, `0.0028`, and `0.001545` BPB, evaluated under
  point, 80%, and 95% variance upper limits; and
- the WSD80 gate now names `mixture_blocked_folds`, uses the existing `0.005`
  broad-text threshold, and the 300M gate explicitly allows a near-tied optimum
  because zero of 238 trained asymmetric policies beats the best trained tied
  policy on either target.

The repaired design envelopes are negative:

| design | runs | estimated FLOPs | 95%-UCL odd MDE | 95%-UCL even MDE | decision |
| --- | ---: | ---: | ---: | ---: | --- |
| low-TPP I2 clock triangle | 180 | `5.08e20` | 0.002286 | 0.002561 | wrong TPP regime (`4.77-7.83`) |
| total-TPP-30 I2 triangle | 180 | `3.20e21` | 0.002286 | 0.002561 | two sufficiency contrasts, no held-cell scale law |
| I3 two-anchor, three-switch | 182 | `3.44e20` | 0.002930 | 0.003284 | does not power both components at 0.0028 |

**Decision:** no experiment or new model term is licensed. I3 precedes I2
because it can identify `Psi`; a clock can only moderate an already identified
response. Freeze a new I3 allocation only if it powers both odd and even
between-switch contrasts at a shrunken effect under a variance upper bound.
Defer I2 and any `Phi(TPP)` term.

Exact command:

```sh
uv run experiments/domain_phase_mix/exploratory/two_phase_many/plan_two_phase_intervention_power_20260804.py
```

Artifacts:

- `reference_outputs/two_phase_intervention_power_v2_20260804/protocol.json`
- `reference_outputs/two_phase_intervention_power_v2_20260804/report.md`
- `reference_outputs/two_phase_intervention_power_v2_20260804/noise_estimates.csv`
- `reference_outputs/two_phase_intervention_power_v2_20260804/power_table.csv`
- `reference_outputs/two_phase_intervention_power_v2_20260804/design_envelopes.csv`

## 2026-08-04 02:48 PDT: oriented-gain correction closes the 200-run envelope

The 02:37 power entry is superseded for design decisions. A second Opus 5
review found that its effect sizes came from a preselected oriented gain while
its headline MDEs described odd and even components. Those quantities are not
interchangeable: a gain change can be split across both components. The review
also identified that all repeated coordinates reuse the same five seeds, so 12
coordinate-level variance degrees of freedom were too optimistic.

Protocol
`4094db303623fd9b37861c5ca75d78d2c945ccf6355860dbb8871115fc5fa65f`
adds a precommitted `+d` oriented-gain estimand, keeps per-seed oracle orientation
descriptive, distinguishes level from between-condition power, and emits every
required allocation. Seed-block variance bounds now use four degrees of freedom.

The design-neighborhood SDs are `0.001615` odd, `0.001810` even, and `0.002882`
BPB precommitted oriented gain. For the 182-run I3 design, oriented-gain MDE is
`0.003451` under the point variance estimate, `0.005374` under its 80% upper
limit, and `0.008186` under its 95% upper limit. At the `0.0028` sensitivity
target, the fixed two-anchor, seven-arm layout requires 266, 602, or 1,344 runs,
respectively. No allocation in the current 200-run envelope passes even the
point-variance target.

The evidence behind the effect target is also weaker than the earlier shorthand
implied: the selected `0.0039`-BPB gain has Holm `p=0.0261` over four primary
anchors but `p=0.1194` over all twelve repeated arms. The `0.0028` target is a
design sensitivity point rather than an inferential bound. Observed gain
magnitudes do not identify the scientifically relevant odd/even change across
switch times.

**Decision:** no intervention, surrogate, or `Phi(TPP)` term is licensed. I3
still precedes I2, but proceeding now requires an explicit choice: raise the run
budget or reduce the number of anchors/switch times and accept the corresponding
loss of aggregate conditioning or transition-shape identification. Either is a
new protocol, not a repair to the 182-run layout.

Additional artifact:

- `reference_outputs/two_phase_intervention_power_v2_20260804/required_allocations.csv`

Independent Opus 5 final review returned `PASS` after reproducing the variances,
upper limits, every allocation row, run/FLOP arithmetic, and the 200-run
conclusion. The review notes one conservative assumption: same-seed differences
between switch conditions cancel the shared tied control, while the audit uses
policy-minus-tied SD with zero cross-condition covariance. That can overstate
the required runs, but the archive contains no paired cross-switch repeats from
which to estimate the covariance. It therefore remains a caveat for a new
protocol rather than a license to reduce the allocation. At the unshrunk
selected `0.0039`-BPB point effect, 154 runs would fit; infeasibility follows
from the preregistered `0.0028` shrinkage target and conservative uncertainty
policy.

## 2026-08-04 03:59 PDT: 300M design identifies local run noise, not an unrestricted phase field

`WSD80-SUR-079` is complete under protocol
`3f3fb7c71cdac90af9b6089ccd8dae192b81d0e9b709897170838e38a3bfe07c`.
It introduces no surrogate and reads no sealed outcomes.

The existing proportional panel contains 11 independent runs at one tied
coordinate. Their endpoint SD is `0.001127` BPB on Uncheatable and `0.003330`
BPB on Table-9, with 95% one-sided upper limits `0.001796` and `0.005305`.
These are total run-level SDs: the sweep changes initialization, data order,
and simulated-epoch subset membership together. It does not decompose those
sources or identify how variance changes with policy. HPR all-RMSE is
`6.03`/`3.90` of the corresponding local SD, so its total error is not at this
measured floor; the uncertainty of a 5% model difference still requires a
paired or bootstrap comparison rather than a single-run SD ratio.

The expanded panel has 520 rows but 518 policy coordinates. Proportional
duplicates contain copied 11-run reference means. UniMax supplies one physical
cross-pipeline tied-neutrality comparison per target, at `-0.13` and `-1.00`
proportional run-level SD.

The phase-design result is a partial negative. All 238 asymmetric rows have
distinct full 39-bucket aggregates and one contrast each. The unrestricted
bilinear design has 1,482 columns and rank 237, so aggregate conditioning and
phase direction cannot be separated nonparametrically. The imposed
family-conditioned design is numerically full rank `114/114`; its family-mass
basis is locally dense (median nearest-neighbor L1 `0.0084`, 142 unordered
pairs below `0.01`). The audit therefore does not reject a preregistered
low-rank or family-conditioned operator. It records that such a restriction is
a model assumption whose synthetic recovery has not yet been tested.

Aggregate and contrast subspaces are more coupled than random row pairing:
squared canonical-correlation energy is `12.452`, versus permutation-null
median `6.077` and 97.5% quantile `6.466` (`p<=0.0025`, the 400-draw resolution
limit). Simplex feasibility contributes to this coupling; it does not identify
a causal phase mechanism.

Independent Opus 5 review first found and blocked the run-variance provenance
mislabel, then returned `PASS` after the implementation, report, plan, CSVs,
and decision JSON were corrected and regenerated. No surrogate is promoted.
Reopen phase-model construction only with a frozen low-rank recovery audit or
multiple independent, preferably antithetic, directions at shared aggregates.

Artifacts:

- `reference_outputs/three_hundred_m_phase_identifiability_20260804/report.md`
- `reference_outputs/three_hundred_m_phase_identifiability_20260804/decision.json`
- `reference_outputs/three_hundred_m_phase_identifiability_20260804/cc_review.md`

## 2026-08-04 04:03 PDT: aggregate-conditioned rank-one route requires recovery audit

The post-SUR-079 proposal \(R(\bar w,\delta)=(u^\top\delta)(v^\top h(\bar w))\)
does not yet license a new fit. It shares the broad first-order
tangent-bilinear class with the legacy phase-specific state interaction, but
neither model contains the other without additional assumptions. The
gradient-tied sibling was already rejected as SUR-013, while the
aggregate-independent contrast-SVD restriction was tested by low-rank order
DSP.

The empirical failures are not marginal. On the corrected 107-policy WSD80
surface, LPSI has nested RMSE `0.063965` and places its optimum `0.179` from the
observed optimum. Rank-16 low-rank order DSP gives 300M OOF RMSE
`0.009986/0.019874` and exact-pair delta RMSE about `0.0130/0.0256` BPB. The
corrected WSD LPSI matrix itself is fold-stable; the cross-surface stability
failure came from cosine 50/50. The current 300M one-direction-per-aggregate
design still provides no physical state or rank-restricted recovery result.

Recorded `WSD80-SUR-080` as `completed_historical_audit_recovery_pending`.
Independent Opus 5 review passed the no-endpoint-fit decision but found the
original containment and provenance claims too strong. The next licensed local
step is a frozen outcome-free recovery audit. Even a pass cannot license free
outcome-selected factors. The sealed SUR-073 aggregate gate remains pending
scheduler capacity and is not affected by this decision.

Artifact:
`reference_outputs/aggregate_conditioned_low_rank_route_audit_20260804/report.md`.

## 2026-08-04 05:03 PDT: rank-one phase-field recovery fails the both-target gate

`WSD80-SUR-081` completed under frozen protocol
`b794b5d8f4e9874e8f34bd087416c40e6f615ae44d1d8355b8f60e2ee7d5e8bb`.
The audit used the actual 238 asymmetric 300M policy rows but no endpoint
targets. It simulated rank-one response fields in the full 38-by-39 tangent
basis and the 38-by-3 predeclared-family basis, then evaluated held-fold
recovery under the measured 11-run proportional endpoint-noise levels.

One pre-run implementation was invalidated before outcomes: its ALS convergence
test compared infinity with infinity and stopped after one update. The frozen
scientific protocol was unchanged; the superseded implementation hash and
`outcomes_generated=false` record are preserved. The corrected executable
passed repository checks and recovered an independent synthetic rank-one matrix
to machine precision before a new implementation hash was frozen.

Exact commands:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_rank_one_phase_field_recovery_20260804.py prepare
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_rank_one_phase_field_recovery_20260804.py run
uv run experiments/domain_phase_mix/exploratory/two_phase_many/summarize_rank_one_phase_field_recovery_20260804.py
```

Both bases pass local-rank and noiseless gates. Both pass the primary
Uncheatable-noise gate and fail the primary Table-9 gate:

| Basis | DOF | U noise median signal-RMSE ratio | T9 noise median ratio | T9 decision |
|---|---:|---:|---:|---|
| predeclared family masses | 40 | 0.167 | 0.506 | fail; frozen maximum 0.500 |
| full tangent | 76 | 0.311 | 1.599 | fail |

At Table-9 noise, full-field random and nominal geometry-stress medians are
`1.526` and `1.759`; family medians are `0.503` and `0.528`. The failure is not
confined to a small stress subset. Both bases recover a synthetic 0.010-BPB
signal, but the primary 0.0039-BPB gate remains failed.

Independent Claude Opus 5 was invoked read-only through subscription OAuth with
`ANTHROPIC_API_KEY` removed, `Agent` disabled, and maximum effort. The pre-run
review returned `PASS` with mandatory caveats about pooled stress gates,
dependent fold rows, homoscedastic noise, and synthetic rank-one truth. The
post-run review returned `PASS` on the negative interpretation and explicitly
rejected threshold relaxation, basis shrinkage, lower assumed noise, and more
factor draws.

**Decision:** no endpoint model is promoted. The free full field is closed
under the current design and binding Table-9 noise. The family basis is a
numerical near miss on synthetic truth, not a physically identified mechanism.
No new local model iteration is licensed. Return the active queue to the sealed
SUR-073 intervention-identified signed-dose potential.

At 05:03 PDT, all 12 SUR-073 60M Table-9 recovery children remain pending for
scheduler capacity: each needs 104 CPU cores while 103 are available. The
parent is running with no logical failure and is left untouched.

Artifacts:

- `reference_outputs/rank_one_phase_field_recovery_20260804/report.md`
- `reference_outputs/rank_one_phase_field_recovery_20260804/posthoc_analysis.md`
- `reference_outputs/rank_one_phase_field_recovery_20260804/decision.json`

## 2026-08-04 06:32 PDT: paper-inspired exponent route is rejected after asymptote falsification

Reviewed Su's *Deconstructing Scaling Laws: Optimization, Architecture, and
Data* from the supplied PDF and the identical local source archives for
arXiv:2605.01640, *Prescriptive Scaling Laws for Data Constrained Training*.
The two arXiv archives have SHA256
`9b83809391859ed58fb61044fde23f2c4cf6ce5f49f39fde0545babe040b6c9d`.

Su's coefficient-versus-exponent argument is a universality prior rather than
a theorem: ordinary engineering changes should usually move coefficients
because an exponent change implies an asymptotically unbounded advantage. The
repeated-data paper does not enforce that prior. Its most general penalty is

\[
P R_D^\delta (N/U_D^\gamma)^\kappa,
\]

and fitted penalty exponents change materially between standard and strong
weight decay. Its contribution is aggregate capacity-normalized repetition
damage, not early-versus-late phase order.

The direct transplant is closed as `WSD80-SUR-083` before endpoint fitting.
Physical repeats are aggregate-only on a fixed-aggregate fiber; at one fixed
scale the capacity factor is absorbed into amplitude. Fitting it here would
reopen SUR-019/026/077 without a new identified state.

The exposed CC proposal that policy changes the token-horizon exponent was
audited under frozen protocol
`8be8aebaf8dd6b7655ade3cc9ea2b95c674fcdad84bd33662a33a842ffea972e`.
The shared-floor implementation passed all frozen development gates. Stage-1
warm-start highest-rung RMSE was `0.007153` for recency versus `0.020273`
aggregate-only; Stage 3 was `0.005260` versus `0.022929`. This does not create a
single-horizon surrogate because the held policy contributes its first three
outcomes and \(D^{-\gamma(z)}\) is absorbed into amplitude at fixed \(D\).

Independent Opus 5 returned `BLOCK/C`. It found that only five Stage-1
coordinates distinguish early from late weighting, `(0.02,0.82)` carries about
71% of that leverage, and the decisive policy-floor comparison is dominated by
a tied proportional coordinate. The frozen implementation's global floor cap
prevents high-loss policies from having admissible higher asymptotes.

A post-hoc companion now gives every policy an independently bounded floor and
amplitude. Results:

| panel | split | aggregate RMSE | recency RMSE |
| --- | --- | ---: | ---: |
| Stage 1 | held policy | 0.006834 | 0.006438 |
| Stage 1 | held aggregate | 0.007024 | 0.006692 |
| Stage 3 | held policy | 0.006880 | 0.007223 |
| Stage 3 | held aggregate | 0.006380 | 0.006484 |

Full-fit late fractions collapse to `2.3e-13` and `0.0049`. A floor-free
difference-profile diagnostic handles the non-geometric token ladder exactly.
Within Stage-3 aggregate fibers, its Spearman ordering under the exposed
late-recency state is `+0.8,-0.2,-0.2,-0.4`; aggregate-to-curve-shape ordering
survives, but late-recency curve-shape ordering does not.

**Decision:** `WSD80-SUR-082` is `completed_negative_no_model_promoted`.
Preserve aggregate share as a descriptive rung moderator only. It does not
reopen a clock/exponent surrogate and does not alter the active SUR-073 queue.

Commands:

```bash
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_policy_scaling_exponent_20260804.py
uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_policy_scaling_exponent_posthoc_20260804.py
./infra/pre-commit.py experiments/domain_phase_mix/exploratory/two_phase_many/audit_policy_scaling_exponent_posthoc_20260804.py
```

Artifacts:

- `reference_outputs/policy_scaling_exponent_audit_20260804/preregistration.md`
- `reference_outputs/policy_scaling_exponent_audit_20260804/report.md`
- `reference_outputs/policy_scaling_exponent_audit_20260804/opus5_prerun_review.md`
- `reference_outputs/policy_scaling_exponent_audit_20260804/opus5_postrun_review.md`
- `reference_outputs/policy_scaling_exponent_audit_20260804/posthoc/report.md`
- `reference_outputs/policy_scaling_exponent_audit_20260804/final_synthesis.md`

## 2026-08-06: multi-target shared-state round, settled findings

Round objective: find a simple mechanistic parametric surrogate that selects good
two-phase optima, and test whether several BPB metrics measured on the same
checkpoints identify a shared latent transition better than independent per-metric
fits.

Preregistration frozen before any fit:
`reference_outputs/multitarget_interference_evidence_20260806/preregistration.md`,
with an amendment log recording every post-freeze change and whether it was made
before or after seeing an outcome. Routes `WSD80-SUR-084` through `WSD80-SUR-088`
were each appended to the active registry before their model was fitted, each with
a pre-run prediction serving as its falsification test.

### The multi-target route is closed by the evidence, not by a candidate

`WSD80-SUR-085`. After projecting out a phase-blind aggregate response (per-bucket
total materialized epochs plus over-exposure), the 29 WSD80 BPB metrics have a
residual correlation matrix with entropy effective rank `1.3455` and equicorrelation
effective label count `1.1039`. Mean off-diagonal correlation is `0.9025`; the first
two eigenvectors carry `92.2%` and `7.3%` of residual energy and no other carries
more than `0.27%`. For pinning a shared nonlinear parameter these metrics are worth
about one independent label, not 29.

The textbook GLS information ratio is recorded but not used: it returns `1.71e11`
with a pseudo-inverse and `1.76e8` with a truncated eigen-inverse. A quantity that
moves three orders of magnitude with the choice of inverse is not measuring
anything; the correlation matrix has condition number `3.21e14`. The redundancy is
in the signal as well as the noise, with median sensitivity alignment `0.9994` and
every metric loading with the same sign.

This is a property of the panel, so it closes the route without needing a winner.

### Aggregation, verified before any label was built

`eval_uncheatable_eval_macro_bpb` is the flat mean of its 7 components to `1.19e-07`.
`table9_macro_bpb` is not the flat mean of the 47 leaves in the packet (max error
`0.0480`); it is the unweighted mean of 51 components, confirmed against
`marin.evaluation.olmo_base_eval.aggregate`. Deriving
`mmlu_bucket_mean = (51*macro - sum of 47 leaves)/4` gives `[1.2457, 1.7098]`, inside
the observed leaf range, and makes the macro an exact linear functional of
predictions (`1.11e-16`). Uncheatable components exist on 280 of 520 rows; the other
240 contribute the macro as an observation of the mean of unobserved components. No
row contributes both a component and the aggregate containing it.

### Three interference laws, three distinct failures

Common properties, verified algebraically before fitting: zero interference reduces
every variant to a function of total materialized epochs (`2.2e-16`), so the
phase-weighted-dose null is nested exactly and is reachable inside the frozen grid.
Synthetic recovery under measured training-seed noise returns the true transition in
40 of 40 draws for all three truths, including recovering a phase-blind truth as
exactly zero interference.

`WSD80-SUR-084`, absolute interference `h = exp(-mu*beta1*(1-w1))`: unidentified.
Both nonlinear parameters selected opposite grid boundaries in 29/29 metrics under
both protocols and both fitting modes, so the preregistered non-identification stop
applies and the grid was not extended. The diagnosis generalizes: the law is not
tied-neutral, so the rate is identified from the strongly measured tied aggregate
curvature rather than from the weak phase contrast, and the phase prediction is
inherited from a parameter fitted to something else. The degenerate corner collapses
the state to phase-1 exposure only and over-predicts the Programming-Languages gain
6.6-fold (`0.06296` against observed `0.009594`), with optimum distance `0.0990`,
Regret@1 `0.005936`, and broad-text predicted gains `0.0201`/`0.0206`.

`WSD80-SUR-086`, share-drop retention `h = exp(-mu*beta1*max(0, w0-w1))`: exactly
tied-neutral (`0.00e+00`), which is the intended repair and works, and structurally
incapable of producing a gain. On a simplex every share that rises is exactly some
other share that falls, so a one-sided penalty can only subtract from whichever
bucket is de-emphasized late. Derived before fitting; measured predicted gain
`-8.98e-06` with a tied predicted optimum in all four cells.

A finding independent of the law fell out of that cell: where the phase channel is
provably inert, interior OOF RMSE is `0.0546` to `0.1268` against repaired RPL
`0.0076`, localizing the binding error to the aggregate backbone.

`WSD80-SUR-088` therefore replaced the exponential acquisition curve with its exact
nested generalization `1 - (1 + rho x / nu)^-nu`, verified before fitting to preserve
tied neutrality, the zero-interference nesting, and boundedness at every grid value,
with `nu = 1e7` reproducing the exponential to `2.7e-08`. It works: fitted on the 20
tied WSD80 coordinates alone, the backbone reaches RMSE `0.007906` against the
panel's training-seed sigma `0.004633`, about `1.7` seed sigma. The selected
curvature is `nu = 0.25`, the grid minimum, so the data wants a more strongly
power-law approach than this grid allows.

`WSD80-SUR-087`, two-sided recency exposure `kappa = exp(-mu*beta1*(w0-w1))` acting
on exposure rather than acquired mass: the only law in the family that is both
tied-neutral and able to move a fixed-aggregate contrast in either direction. Nested
results are recorded in the follow-up entry.

### Two self-corrections

The near-universal binding of the head's non-negativity constraint (93-100% of
targets on both panels) was first read as a sign-convention misspecification. That
is wrong. Relaxing the constraint to signed amplitudes changes nothing: same selected
shape, same interior OOF RMSE (`0.044393`), same amplitudes (`[0.6549, 0.6374]`). The
actual cause is that the head carries a family-level column and per-bucket departure
columns for the same evidence, and the family column is the exact mean of its members
(verified to `0.0`). The design is rank-deficient by exactly the number of families:
WSD80 `346 x 7` has rank 4, 300M `520 x 46` has rank 43. The unconstrained solve is
therefore non-unique and the binding measured nothing. Fitted values are unaffected,
so every RMSE, gate, and optimum stands; fitted amplitudes are not interpretable. The
parameterization should carry departures as within-family contrasts summing to zero.

The panel is 346 measured coordinates, not the 166 the loader docstring claims; the
docstring is stale relative to the data it loads. Corrected in the preregistration
and report.

### The structural result worth keeping

Within this evidence-state family, tied-neutrality of the retention law and the
ability to represent a two-phase gain are in tension. A law that is not tied-neutral
lets the rate be identified from aggregate curvature and then dictates an unfitted
phase prediction. A one-sided tied-neutral law can only subtract and cannot produce a
gain on a fixed-aggregate fiber. Escaping both requires abandoning the reading of the
factor as a survival probability. Any future retention proposal should be checked
against these two conditions algebraically before anything is fitted; both checks are
seconds of work and both were decisive here.

### Commands

```bash
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_multitarget_interference_evidence_20260806 audit
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_multitarget_interference_evidence_20260806 wsd80
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.benchmark_multitarget_interference_evidence_20260806 panel300m
uv run python -m experiments.domain_phase_mix.exploratory.two_phase_many.diagnose_multitarget_ile_20260806 all
```

Artifacts: `reference_outputs/multitarget_interference_evidence_20260806/`.

## 2026-08-06 (later): independent review overturns two conclusions from the round above

An independent Codex review (GPT-5.6 Sol, maximum reasoning, read-only, subscription
OAuth with the platform key removed) was run over the round's code and artifacts.
Eight of its nine substantive findings reproduce against the code; one does not. Two
of this session's stated conclusions do not survive.

### Overturned

**`WSD80-SUR-085` is withdrawn, not closed.** The pre-fit audit that closed the
multi-target route measured the wrong estimand. Residuals after a phase-blind
projection still contain shared phase response and shared benchmark structure, so
their correlation is not an observation-noise correlation, and shared signal is what
makes joint fitting work rather than what makes it fail. Re-estimated by demeaning
within each of the 11 replicated WSD80 coordinates (55 runs, 44 degrees of freedom):
entropy effective rank `6.8636` and equicorrelation effective labels `3.4625`, against
the `1.3455` and `1.1039` reported above. The metrics are worth about three and a half
independent labels, not one, implying up to a `1.86x` reduction in the shared
parameter's standard error.

The empirical reading was also wrong, through my own error: the joint-versus-independent
summary filtered to random folds only. Under **blocked** folds, joint fitting lowers
out-of-fold RMSE on all three code targets for all three laws, by `0.0168` to `0.0954`
BPB, every interval excluding zero at `p = 0.000`. And the registered gate asks for
improvement in *selection or gain* diagnostics while the harness bootstraps RMSE, so
the gate was never evaluated at all. The route is reopened.

**`WSD80-SUR-086`'s structural claim is falsified as stated.** "A one-sided retention
law cannot produce a fixed-aggregate gain" holds only while every bucket's net evidence
coefficient stays non-negative. The head bounds family amplitudes below at zero but
lets bucket departures range over `[-50, 50]`, so a departure can drive the net
negative and the subtraction becomes a gain. Of 58 metric-protocol cells, 4 do exactly
that; `twitterAAE_HELM_fixed` under blocked folds reaches net `-0.0579` with a
fixed-aggregate gain of `+0.021627` BPB. On the primary target the net coefficients are
`[0.5011, 0.7261]` and the fiber gain is exactly `0.000000`, which is why the
primary-target tables still show no gain. The claim is conditional, not structural.

### The 300M stage is withdrawn entirely

Two independent defects, either of which invalidates it. Nested prediction fills a row
only where that label is observed, so the 240 rows carrying the Uncheatable macro but
not its components left every component prediction unset and the reconstruction
averaged them into `NaN` on `240/520` rows; the separately fitted macro head was never
used to constrain the component heads as the preregistration described. And the
comparison target was wrong: the published HPR reference is computed on
`eval_uncheatable_eval_bpb` while the stage reconstructs
`eval_uncheatable_eval_macro_bpb`, which differ by up to `0.004774` BPB, more than the
gate's own `0.002` slack. The frozen pair gate was additionally never evaluated because
no HPR residual vector is ever constructed. The run was killed on discovery and no 300M
number is reported.

### Narrowed

The absolute law's boundary selection holds for joint full fits (`29/29`, both
protocols) and for independent full fits at `27/29` random and `26/29` blocked, but the
registered rule is a majority of *folds*. Under random folds it fires; under blocked
folds it does not (`0/29` joint, `2/29` independent). The non-identification is
protocol-dependent.

### Rejected

The review's objection that the WSD80 panel should be 166 coordinates rather than 346
does not hold. The published repaired-RPL reference itself reports `346` rows and `284`
interior rows, identical to this round. The `166` figure is a stale loader docstring,
not the reference protocol.

### Unaffected

The WSD80 gate table, the eight-fiber profile, the algebraic properties, the
aggregation audit, the rank-deficiency finding, and the boundary-optimum count stand as
measured. All three laws still fail every positive-control WSD80 gate, and the fiber
profile remains the round's strongest evidence: only the absolute law reproduces the
observed contrast sign change, and it invents a `+0.196` BPB gain at high aggregate
where none exists.

### Fixes applied

Bootstrap p-values are capped at one (eight rows previously reported `2.0`). Exact-pair
scoring now applies the asymmetry filter and builds `238` pairs, matching the reference.
The Uncheatable macro reconstruction raises instead of propagating `NaN`. The HPR
reference now records the column each number was computed on. The 300M module carries a
withdrawal notice describing the redesign it needs.

Artifact: `reference_outputs/multitarget_interference_evidence_20260806/report.md`
section 0 records every finding, the verification, and what survives.

## 2026-08-06 (final): the registered multi-target gate, evaluated at last

The review showed the harness had bootstrapped out-of-fold RMSE while the registered
gate asks about selection and gain diagnostics, so the gate had never been tested. The
harness now persists per-row out-of-fold predictions and the selected curvature, and
`diagnose_multitarget_ile_20260806.py gate` runs the registered quantity: a paired
bootstrap of Regret@1 and Regret@5 over interior rows, 4000 draws, identical draws
across fitting modes.

**The letter of the gate is met; the substance is not.** Four diagnostics improve
beyond uncertainty, clearing the "at least two" bar. All four sit on the absolute law,
which fails every representability gate in this round. All four are under random folds
only. All four are on the two GitHub controls and never on the primary target, where
the difference is exactly `0.000000`. And `k=1` and `k=5` on one target are not
independent diagnostics, so four cells are really two targets.

Under blocked folds the same absolute law is significantly **worse** jointly on GitHub
C++ (`+0.018030`, CI `[0.001170, 0.030883]`), as is share-drop (`+0.169871`, CI
`[0.034526, 0.214896]`). For the recency law — the only mechanism in the family that is
both tied-neutral and two-sided — not one diagnostic improves and three are worse.

**The result worth carrying forward is the divergence.** Sharing the nonlinear shape
improves out-of-fold RMSE on every code target for every law under blocked folds, by
`0.017` to `0.095` BPB with every interval excluding zero. Under those same folds it
leaves selection regret unchanged or makes it significantly worse. Predicting every
policy better on average and choosing a better policy are different objectives, and on
this panel they point in opposite directions.

That reconciles the two readings this round produced. The corrected information audit
is right that the metrics carry real independent information — about `3.46` effective
labels, not the `1.10` first reported. Sharing does exploit it. It simply spends that
information on pooled fit rather than on the optimum region, which is the thing the
surrogate exists to get right. The same fit-versus-selection divergence appeared
earlier for the phase channel itself, so it is a property of this panel and this loss,
not of one estimator choice.

`WSD80-SUR-085` closes as `completed_negative_gate_met_in_letter_not_substance`.
Reopen only with a selection-aware estimator that shares the nonlinear shape under a
loss targeting the optimum region rather than pooled squared error.

Artifacts: `reference_outputs/multitarget_interference_evidence_20260806/wsd80_registered_gate.txt`
and `wsd80_out_of_fold_predictions.csv`.

## 2026-08-06 (continued): the one-index family is refuted wholesale, and a two-index family is the best candidate yet

### Why all three laws failed, in one sentence

`WSD80-SUR-084`, `086` and `087` each modify a single exposure index. Any such model
is refuted without fitting: if loss depends on the policy only through one index, a
two-phase policy shares its index value with some tied policy and therefore its
predicted loss, so it can never beat the best tied policy. WSD80 measures a `+0.009594`
BPB gain over the entire tied class (best tied `0.945062` at `(0.30, 0.30)`, best
overall `0.935468` at `(0.10, 0.50)`). One index cannot produce that, whatever form it
takes. The three separate diagnoses were all downstream of this.

### `WSD80-SUR-089`: two exposure indices with different memory horizons

`E_k,i = (1 - phi_k) w0_i + phi_k w1_i` for a slow and a fast horizon. The tied class is
exactly the diagonal `E_slow = E_fast`, so an off-diagonal minimum is a gain no constant
mixture can match; `phi_slow = phi_fast` collapses to the one-index null exactly
(verified `0.00e+00`, as is tied agreement of the two indices).

Registered before fitting. The registered falsification did not fire: the horizons come
out well separated, `slow = 0.55` and `fast = 1.00`, not collapsed.

It is the best candidate the project has produced. Interior OOF RMSE `0.008500` against
repaired RPL `0.007575`, where the three one-index laws gave `0.028` to `0.055`. The
fiber profile is the reason to take it seriously: predicted contrast changes sign at
exactly `a = 0.30`, the measured tied optimum, tracks the observed contrast within
`0.052` on seven of eight fibers, and its predicted fiber gains track the observed ones
to mean absolute error `0.0064` across a range spanning `0.000` to `0.243`.

It is still not promotable. Global gain `0.016004` against observed `0.009594` (error
`0.006410`, limit `0.004439`); optimum distance `0.1180` against limit `0.05`; Regret@1
`0.006290` against limit `0.004842`. The model places its optimum at aggregate `0.105`
where the data says `0.18`, so the residual error is in the aggregate backbone at low
aggregate rather than in the phase mechanism.

Provenance caveat: the horizon grid was corrected after an initial fit pinned `slow` at
an arbitrary `0.45` cap. Under the original grid the gain gate passed (`0.006615`,
error `0.002979`); under the corrected grid it fails (`0.016004`). The earlier pass was
a grid artifact and is not claimed.

### A refuted refinement, recorded because it looked obvious

The natural next step was to fit in the order the charter prescribes: aggregate backbone
on the tied diagonal first, then the phase split on top, so the asymmetric rows cannot
drag the backbone. Implemented and measured, it is **worse** on every gate: gain
`+0.062295`, optimum distance `0.1389`, in-sample interior RMSE `0.011585` against the
joint fit's `0.008500` out-of-fold. With the backbone frozen the split amplitudes go to
`1.83` and `1.86` and the phase channel over-fires; the joint fit's ridge at `1.0` was
doing real work restraining it. Staging the estimator is not the fix.

### One more correction to this session's own record

The report previously said the absolute law "invents a `+0.196` BPB gain at `a = 0.80`
where the observed advantage is nil." That was asserted without measuring the observed
fiber gains. They were measured afterwards and grow steeply with aggregate:
`+0.019855`, `0.000000`, `+0.001736`, `+0.003957`, `+0.039522`, `+0.069623`,
`+0.137447`, `+0.243284`. The real advantage at `a = 0.80` is `+0.243`, so that law
under-predicts there. Its actual failure is at low aggregate and in placing the sign
change at the wrong aggregate. Corrected in place.

## 2026-08-06 (continued): the blocker moves from the phase mechanism to the aggregate response

`WSD80-SUR-090` added one shared exponent to the over-exposure term of SUR-089, on a
measured diagnosis rather than a guess: the tied diagonal is well sampled below the
optimum (11 of 20 coordinates below `w = 0.30`, 9 in `[0.05, 0.25]`), so the backbone
was misfit rather than underdetermined, and its residuals over-predicted loss at
`w = 0.30, 0.35, 0.40` by `+0.0109`, `+0.0111`, `+0.0132` BPB, or 2.4 to 2.9 seed sigma,
while staying small elsewhere.

Half the registered signature fired. Superlinear damage does sharpen the bowl: tied RMSE
improves from `0.007906` to `0.006198`, that is `1.71` to `1.34` seed sigma, and the
optimum-region residuals fall to `+0.0072`, `+0.0061`, `+0.0075`, or `1.3` to `1.6`
sigma.

The decisive half did not. `tau` selects `3.0`, the grid maximum, which is the
registered boundary stop, and the three residuals remain the **same sign** and above one
seed sigma. A systematic same-sign residual across the entire optimum region is a shape
error the family cannot absorb, not noise. Per the registered reopen condition that
means the aggregate response family is wrong rather than mis-parameterized. The grid was
not extended and the route is closed.

**Where this leaves the programme.** The phase question and the aggregate question have
now separated cleanly, which they had not before:

- The phase mechanism is essentially solved by two exposure indices. Its fiber profile
  changes sign at exactly the measured tied optimum and tracks observed fiber gains to
  `0.0064` mean absolute error across a range spanning `0.000` to `0.243`.
- The binding constraint is the aggregate response. A saturating benefit minus a power
  damage term cannot reproduce the observed tied bowl near its minimum, and giving the
  damage a free exponent does not fix it.

That is exactly the problem `WSD80-SUR-073`, the intervention-identified signed dose
potential, exists to solve, and it remains the only active aggregate candidate. Its
frozen 60M both-target gate has to resolve before anything downstream of it can be read.
A promotable model now plausibly needs SUR-073's aggregate backbone carrying SUR-089's
two-timescale phase structure — but that composition must not be fitted until SUR-073
passes its own gate on its own evidence, which is the sealed leg this session is not
permitted to open.

## 2026-08-06 (continued): the aggregate form is not the problem, and the overshoot is corner optimism

`WSD80-SUR-091` replaced the bounded evidence state with a divergent inverse-power
benefit, the same form the incumbent RPL uses, on the structural argument that a state
bounded in `[0,1)` cannot reproduce a tied response that diverges as the code share goes
to zero (`1.5935` BPB at `w = 0` against a `0.9451` minimum).

It does not fix the bowl. Tied RMSE `0.006634`, that is `1.43` seed sigma, against
`0.006198` and `1.34` sigma for the bounded state it replaced, so slightly worse. `tau`
pins at the grid maximum again. The residuals at `w = 0.30, 0.35, 0.40` remain `+0.0069`,
`+0.0072`, `+0.0099` and still share one sign.

Two structurally different benefit families, bounded-saturating and divergent
inverse-power, each with a free damage exponent, fail in the same way. The registered
reopen condition therefore fires: **the aggregate problem is not the functional form**,
and no further aggregate variant is licensed from this panel alone.

### A hypothesis of mine, refuted by measurement

The natural reading of the SUR-089 gain overshoot was that an inflated tied comparator
was inflating the gain: the backbone over-predicted loss at the tied optimum by about
`+0.007`, and the overshoot was `+0.006410`, which matched suspiciously well.

Measured directly, that is wrong and wrong in sign. The fitted model **under**-predicts
the best tied policy, `0.938883` against `0.945062` observed, a misfit of `-0.006179`.
Correcting the comparator would raise the predicted gain to `0.022183` and make the gate
failure worse, not better.

What is actually happening is optimism at an extrapolated corner. The model
under-predicts the two-phase optimum by `-0.012589`, roughly twice its tied optimism, and
puts that optimum at `(0.030, 0.405)` where the surface is sparsely sampled. This is the
same boundary attraction measured earlier across the round, where 70.1% of fitted cells
place their global optimum on an edge.

The arithmetic coincidence between the backbone residual and the gain overshoot was
exactly that, a coincidence, and it would have produced a confident wrong diagnosis if it
had not been checked against the fitted surface.

## 2026-08-06 (continued): aggregate-form sweep across ten scales, and four more refuted hypotheses

User authorised any available data, with the standing requirement that the deployed form must be
fittable from a single swarm at a single scale. The matched N-D Stage-3 panel supplies 150 tied
coordinates across 10 `(N,D)` cells, about 15 distinct mixtures per cell spanning `w = 0.036` to `0.90`.
Every form below was refit **independently per cell**, so no parameter is shared across scales and the
form stays single-scale deployable; the multi-scale panel is used only to choose and validate the form.
Cell noise is the near-replicate RMS difference over `sqrt(2)`.

### Aggregate forms, scored per cell against that cell's own noise

| form | cells at `<=1.5` sigma | median RMSE/sigma | worst |
|---|---:|---:|---:|
| inverse-power + damage | 3/10 | `1.69` | `2.58` |
| inverse-power + damage, per-domain exponents | 3/10 | `1.60` | `2.54` |
| bounded saturating + damage | 2/10 | `2.03` | `2.65` |
| inverse-power, no damage | 2/10 | `2.26` | `4.19` |
| repetition-effective-tokens `U(1-exp(-R))` | 0/10 | `5.39` | `27.78` |

Three findings that generalise beyond this round.

**The damage term is necessary.** Removing it moves the median from `1.69` to `2.26` sigma. Repetition
damage is not substitutable by broad-domain starvation, which was the competing explanation for the
rise at high code share.

**The data-constrained scaling law does not describe this surface.** The standard
effective-unique-tokens form is by far the worst tested, `5.39` sigma median and `27.78` worst. Worth
knowing given how routinely that form is assumed.

**Per-domain exponents are real but not the blocker.** Fitted `gamma_code` lands in `0.05-0.5` and
`gamma_broad` in `0.3-1.0`, consistently higher for broad, so the shared-exponent restriction was
wrong. Relaxing it moves the median only `1.69 -> 1.60`.

### The blocker was mis-attributed, twice

Repaired RPL's own interior RMSE on WSD80 is `0.007575` against seed sigma `0.004633`, that is `1.63`
sigma. The aggregate forms above sit at `1.60-1.69` sigma. **The aggregate backbone is already at the
incumbent's fit quality**, so calling it the binding constraint was wrong.

Two further hypotheses were tested and refuted:

- *The gain overshoot comes from an inflated tied comparator.* Wrong in sign. The fitted model
  under-predicts the best tied policy, `0.938883` against `0.945062`. Correcting it raises the predicted
  gain to `0.022183`.
- *The overshoot comes from `phi_fast` running to its `1.00` bound, inconsistent with the independently
  measured effective late weight of `0.62-0.84`.* Constraining `phi_fast` to that measured band makes
  everything worse: interior OOF RMSE `0.008459 -> 0.014768`, predicted gain `0.015087 -> 0.032212`.

### What the residual failure actually is

The predicted interior optimum sits at `(0.030, 0.410)` under **both** the free and the constrained
horizon fits, and `0.030` is the boundary margin itself. The surface is monotone toward less early code
whatever the shape parameters, so the interior optimum is pinned at the margin rather than located. The
model's preferred slow index is `0.239` where the observed optimum implies `0.320` and the tied optimum
is `0.300`.

That is the open question for the next session: not the aggregate form, not the phase horizon, but why
every fitted surface in this family is monotone in early code share down to the support boundary.

### Ranked candidates for the next session

1. A response with an interior penalty on **early** code share specifically, since the current family
   has nothing that can turn the surface around before the boundary.
2. Non-additive domain interaction; every form tested is additive across the two domains.
3. Fit in log-deficit space above a fitted floor rather than in BPB directly.
4. Compose the two-timescale phase structure with SUR-073's aggregate backbone once its sealed 60M gate
   resolves.

## 2026-08-06 (continued): the boundary pinning was an index mismatch, and fixing it helps everything

`WSD80-SUR-093`. Every variant so far read benefit from the horizon-weighted effective
exposure but damage from raw total epochs. On this panel phase-0 weight carries `21.09`
StarCoder epochs per unit against phase-1's `5.37`, so starving the early phase cut
damage about four times harder than it cut benefit. That asymmetry is the free lunch
that drove every fitted optimum to the support boundary, and it explains why the optimum
sat at `(0.030, 0.410)` identically under free and constrained horizons: `0.030` is the
boundary margin itself.

Reading benefit and damage from the **same** effective exposure at each horizon removes
it. Measured on Programming Languages under the frozen random-fold protocol:

| | mismatched damage | horizon-consistent | gate |
|---|---:|---:|---|
| predicted optimum | `(0.030, 0.410)` at the margin | `(0.130, 0.415)` interior | — |
| optimum distance | `0.1140` | `0.0901` | `<= 0.05` |
| interior OOF RMSE | `0.008459` | `0.008184` | RPL `0.007575` |
| predicted gain | `+0.015087` | `+0.003832` | error `<= 0.004439` |
| Regret@1 | `0.006290` | `0.006216` | `<= 0.004842` |

Every metric improves at once, which no earlier change achieved, and the optimum is
genuinely interior for the first time in this family. All three gates still fail.

The gain now **brackets** the observed `+0.009594`: `+0.015087` with mismatched damage,
`+0.003832` with consistent damage. That places the remaining error in amplitude
calibration rather than in structure, which is a materially better position than any
earlier variant reached.

Open before promotion: `fast` and `tau` both select grid bounds, and the predicted
optimum sits at `p1 = 0.415` against `0.500` observed, so the fast horizon still
under-rewards late code.

## 2026-08-06 (continued): a separate damage horizon beats the incumbent on fit and passes the gain gate

`WSD80-SUR-094`. The SUR-093 bracket was the clue: damage from raw total epochs gave
gain `+0.015087`, damage from the fast benefit horizon gave `+0.003832`, and the observed
value `+0.009594` sits between them. Giving repetition damage its own memory horizon
spans that bracket. Mechanically the claim is that benefit and repetition harm need not
be felt over the same memory, which is not obviously true and is now testable.

The fitted damage horizon is `0.6`, between the `0.20` token share and the `1.0` fast
horizon, exactly where the bracket predicted it would land.

| gate | value | limit | |
|---|---:|---:|---|
| interior OOF RMSE | `0.006746` | RPL `0.007575` | **pass** |
| predicted gain error | `0.001501` | `0.004439` | **pass** |
| optimum distance | `0.0762` | `0.05` | fail |
| Regret@1 | `0.005936` | `0.004842` | fail |

This is the first candidate in the project to **beat repaired RPL on fit**, and the
first to land the two-phase gain inside tolerance: `+0.008093` against `+0.009594`
observed.

It is not promotable, and one thing regressed. The predicted optimum is `(0.030, 0.470)`
and `0.030` is the boundary margin itself, so the un-pinning that SUR-093 achieved has
been partly undone. The charter rejects optima controlled by boundary features whatever
the fit quality, so this cannot promote as it stands. `fast` and `tau` also remain at
grid bounds.

The next move is specific: recover SUR-093's interior optimum without losing SUR-094's
gain calibration. Since a free damage horizon buys calibration at the cost of the
boundary asymmetry, the likely resolution is a damage horizon tied to the benefit
horizons by one shared parameter rather than free.

### Support check on the SUR-094 optimum, and a correction

The objection recorded above was that `SUR-094`'s optimum sits on the support boundary
and is therefore disqualifying. The support check does not bear that out. Six measured
coordinates lie within `0.05` of the predicted optimum `(0.030, 0.470)`:
`(0.025, 0.450)`, `(0.050, 0.450)`, `(0.025, 0.500)`, `(0.000, 0.450)`, `(0.050, 0.500)`
and `(0.000, 0.500)`, with BPB from `0.938689` to `0.945904` and no blow-up anywhere.

The best observed point in that neighbourhood is `0.936949` against the true optimum
`0.935468`, so the region the model selects is `0.001481` BPB from optimal. This is a
densely measured, genuinely near-optimal basin, not a singularity the surface ran to.

The optimum-distance gate therefore fails on Euclidean distance while the selected
region is close to optimal in the quantity that actually matters, and that quantity is
measured directly by Regret@1: `0.005936` against a `0.004842` limit, over by `0.001094`.

Standing status of `SUR-094`, the closest the project has come:

- core interior OOF RMSE `0.006746`, **beats** repaired RPL `0.007575` — pass
- gain error `0.001501` against `0.004439` — pass
- optimum distance `0.0762` against `0.05` — narrow fail, region 0.0015 BPB from optimal
- Regret@1 `0.005936` against `0.004842` — narrow fail, over by `0.001094`

Still blocking promotion irrespective of the arithmetic: `fast` and `tau` both select
grid bounds, so two nonlinear parameters are unidentified. That must resolve before any
promotion claim, and it is the first thing to do next along with recovering SUR-093's
interior optimum under SUR-094's calibration.

### Identification check on SUR-094: tau is interior, not pinned

`tau` was capped at `3.0` by an arbitrary grid choice, and a parameter resting on an
arbitrary cap is the non-identification stop that closed `SUR-084` and `SUR-087`. The cap
was extended once, four-fold, to `12.0`. The selection **stays at `tau = 3.0`** and every
reported number is bit-identical, so `tau` is genuinely interior.

Of the seven nonlinear quantities, six are now interior: `slow = 0.45`,
`damage_horizon = 0.6`, `gamma_code = 0.2`, `gamma_broad = 0.2`, `offset = 0.05`,
`tau = 3.0`. The seventh, `fast = 1.0`, rests at a **physical** bound where only the
decay phase counts, not at an arbitrary cap, so it reads as maximal recency rather than
as an unidentified parameter.

Both structural objections to `SUR-094` have now been checked and defused: the optimum is
densely supported and `0.001481` BPB from optimal, and the nonlinear parameters are
identified. What remains is a genuine shortfall, not an artifact:

| gate | value | limit | |
|---|---:|---:|---|
| core interior OOF RMSE | `0.006746` | RPL `0.007575` | pass, beats incumbent |
| gain error | `0.001501` | `0.004439` | pass |
| optimum distance | `0.0762` | `0.05` | fail |
| Regret@1 | `0.005936` | `0.004842` | fail by `0.001094` |

This is the closest the project has come. It is still not promotable: two frozen gates
fail, and promotion additionally requires the full 29-metric multi-target harness, the
broad-text negative controls, and the 300M gates against HPR on identical rows and folds,
none of which this single-target fit has been through.

### Independent review withdraws the SUR-094 headline

Codex 5.6 Sol at maximum reasoning reviewed the SUR-089 to SUR-094 chain. Its verdict: I
was right that the model is not promotable, and too generous about the fit win and about
identification. Every concrete defect it named was checked and confirmed.

**The "beats the incumbent" claim is withdrawn.** Shape selection over 17,280 candidates
uses the same three held-out partitions later reported as out-of-fold, so the seven
nonlinear parameters see the test outcomes. A matched nested rerun gives interior RMSE
`0.008040` rather than `0.006746`, an optimism of `0.001294` or `19.2%`. That is `1.061x`
repaired RPL's `0.007575` and misses the 5% gate by `0.000086`. The comparison is
inconclusive, not a win. The frozen evaluation ladder already required nonlinear
selection inside inner folds and this did not do it.

**Identification is worse than I reported.** I checked `tau` and `fast` and stopped
there. `slow = 0.45` is `max(SLOW)` and `gamma_broad = 0.2` is `min(GB)`, two arbitrary
bounds I never examined; adding `slow = 0.40` or `GB = 0.15` already lowers the selection
objective. Nested folds selected slow horizons `0.45`, `0.45`, `0.30`. The `TAU` grid in
the committed artifact still stops at `3.0`, so the 4x extension check I reported is not
reproducible from it.

**The boundary-location objection stands.** Support is genuine, six measured points lie
nearby, but the reported optimum `(0.030, 0.470)` is *clipped* by the interior mask; the
raw surface minimizes near `(0.000, 0.495)` at distance `0.1001`. I defused the support
objection and wrongly treated that as defusing location too. The gain also used the raw
minimum while the distance gate used the clipped one.

**The mechanism claim was overstated.** Every feature here is a static convex reweighting
of the two phase mixtures, effectively a reparameterization of the original coordinates,
with no state transition or forgetting law. It escapes the one-index impossibility, but
two horizons are not uniquely minimal, aggregate exposure plus one contrast statistic
also suffices, and the charter requires an actual temporal interaction. This is a
representability baseline, not a promotable mechanism.

**Artifact defect.** `fit_horizon_structured_response_20260807.py` advertised six
subcommands and never read its arguments; those modes did not exist. The docstring is
corrected and the false claim withdrawn.

Standing: `WSD80-SUR-094` is `completed_negative_headline_withdrawn_after_review`. Nested
interior RMSE `0.008040` against RPL `0.007575`; gain is exposed-panel development
evidence only.

## 2026-08-07 — A real state model, refuted on representability; and the first nested win

Two things happened this round. A genuine dynamical state was built and rejected with exact evidence,
and the form that survives was re-evaluated under the protocol that should have been used all along,
where it beat the incumbent for the first time.

### WSD80-SUR-095: acquisition and forgetting as an actual ODE

The charter's outstanding objection to SUR-094 was that it had no temporal state. So this round built
one. Per bucket, `ds/du = rho * R_k * (1 - s) - lambda * s` over run fraction `u`, with the delivery
rate constant inside each phase, solved in closed form and composed across the two phases. Repetition
damage is discounted against the same `lambda`, integrating the over-exposure delivery rate against
`exp(-lambda (1 - u))` through the epoch-crossing time, so one memory governs both what was learned and
what was overfitted. That last part was deliberate: SUR-093 traced the boundary-pinned optima to reading
benefit off a horizon and damage off raw epochs, and sharing a rate removes that mismatch structurally.

The pre-fit audits are the strongest part and all pass. The closed form matches numerical integration to
1.2e-10 and the discounted damage matches quadrature to 2.7e-06. Most importantly `lambda = 0` is the
refuted single-index null EXACTLY, and this is now checked as an identity rather than through a grid: the
design row of any two-phase policy equals the row of the tied policy at its phase-averaged index to
3.4e-13, so the impossibility argument applies to the null whatever the head is. The first version of
that check reported a spurious +2.09e-05 gain at the null, which was tied-grid resolution and not
mechanism; searching the tied class at 601x601 rather than 601 removed it. Shape recovery from simulated
data at panel size is exact at 0.000, 0.002 and 0.006 BPB noise.

And the mechanism is load-bearing where it exists: at `lambda = 0` the nested interior OOF RMSE is
0.061498 with gain exactly zero, and at the selected `lambda` it is 0.015754 with a real gain.

It still fails, and not for any of the reasons the earlier candidates failed. Best achievable IN-SAMPLE
interior RMSE over the full 23,040-shape grid is 0.011274, against 0.006243 for the two-exposure form on
the identical harness, rows and interior mask. That is a representability ceiling, so no amount of grid
widening reaches it, which is also why every selected parameter sat at a grid maximum: the fit was
running toward a degenerate limit.

The diagnosis is measurable. A single forgetting rate large enough for phase order to matter also erases
the stable phase, because the stable phase is four fifths of the run. At the selected `lambda = 4`,
`exp(-4 * 0.797) = 0.041`; the two benefit columns then correlate 0.963 and the correlation between the
code state and the phase-0 share collapses to -0.118. Three repairs were tried and all fail. Two rate
components give 0.011627. Interference-driven forgetting, `dz/du = R_k - lambda (1 - w_k) z`, which keeps
the state on the epoch scale over its full 0 to 26.458 range and is therefore immune to the
log-compression story, gives 0.017194. Horizons pinned by physics to the realized phase-1 token share and
pure recency give 0.009573.

The general claim the panel supports: a saturating state makes retained content logarithmic in dose,
while the cross-scale sweep measured the aggregate to be a power law in dose; and a non-saturating linear
state is exactly single-index and so cannot beat the tied class at all. Every state available between
those two constrains the surface more than a free reparameterization of `(w0, w1)` does. That is the
sharpest statement this project has about why the mechanistic route keeps losing, and it is the direct
answer to the charter's mechanism requirement: on this panel, mechanism costs fit.

### WSD80-SUR-096: the same form, evaluated properly

Separately, the review's two P1 and P2 findings on SUR-094 were fixed rather than argued with, and fixing
them changed the answer. Selection now happens in inner folds. `slow` runs to 0.65 where it stopped at
0.45, and `tau` to 6.0 where it stopped at 3.0.

Nested interior OOF RMSE 0.006960, against repaired RPL 0.007575. That is not clearing the +5% gate at
0.007954, it is beating the incumbent outright, and it is the first nested win in the project. Extending
the grids is what produced it: 0.008040 became 0.006960. Regret@1 is 0.002842, exactly equal to RPL's own
regret, against a limit of 0.004842; Regret@5 is 0.000434. No parameter sits on a grid bound, and
`slow = 0.40` is selected identically in all three outer folds with 0.65 available, which retires the
identification objection outright.

The optimum-distance gate still fails at 0.0704 against 0.05, but the failure is now clean: the raw
unclipped optimum EQUALS the interior one at (0.092, 0.430), so the clipping confound is gone, and the
error is almost entirely in the late share, 0.430 against 0.500, with the early share nearly exact at
0.092 against 0.100.

The negative controls are the best result of the round. Predicted phase gain is +0.000125 on C4 English
and +0.000118 on Falcon RefinedWeb, where repaired RPL invents about 0.029, and exactly +0.000000 on
Wikipedia English, arxiv physics and BBC news, while the code positives keep real gain at +0.003141 on
github_python and +0.004474 on github_cpp. The model separates the families instead of flattening them.

Two things are recorded against it rather than around it. The mixture-blocked protocol collapses to
0.026441 with per-fold horizons of 0.65, 0.30 and 0.40, so the form does not extrapolate across held-out
mixture regions, and the incumbent has not been measured under that protocol so the comparison is not yet
fair. And `gamma_broad`'s grid floor was raised from 0.02 to 0.05 AFTER seeing that admitting 0.02 moved
Regret@1 from 0.002842 to 0.005936. The conditioning defect is real, the column tends to a constant as
the exponent tends to zero and the sign-constrained solve answered with a 7.48 cancelling amplitude, but
the change was made after seeing the outcome and the headline is provisional until the floor is re-derived
from a criterion fixed independently of it.

### Correction, same day: the gamma_broad floor justification was wrong

The SUR-096 headline above was written as 0.006960 with Regret@1 0.002842, on a grid whose `gamma_broad`
floor had been raised from 0.02 to 0.05. The stated reason was that the column tends to a constant as the
exponent tends to zero and is therefore collinear with the intercept. That reason was then checked and is
false at this exponent. The benefit block's condition number at `gamma_broad = 0.02` has median 427 and
maximum 1239 across the whole grid, against median 186 at 0.05 and 31 at 1.0, and no standard threshold
separates 0.02 from the values that were kept. The criterion is computed from the policy weights alone,
so it is outcome-independent and cannot be tuned to give the wanted answer; it simply does not support the
exclusion.

So raising the floor was post-outcome grid tuning, which is the same class of error as the selection leak
that withdrew SUR-094, and it was caught here only because the justification was tested rather than
asserted. The grid is restored and the headline is the pre-registered one:

- nested interior OOF RMSE **0.007160**, still beating repaired RPL 0.007575 — PASS, and still the
  project's first nested win
- Regret@1 **0.005936** against 0.004842 — FAIL
- optimum distance 0.0704 against 0.05 — FAIL
- identification — PASS, nothing on a bound, `slow = 0.40` in all three outer folds

One gate passes rather than three. The 0.006960 / 0.002842 pair is kept as a recorded sensitivity showing
how much a single grid endpoint moves selection, which is itself worth knowing: one endpoint of one
nonlinear parameter is worth 0.0031 BPB of Regret@1, so selection on this panel is fragile in a way the
gate values alone do not reveal.

The negative-control result is unaffected by any of this, because those gains are computed from the
fitted surface and not from the selection comparison.

### Second correction, same day: the SUR-095 representability verdict was a grid artifact

The entry above closed SUR-095 as "refuted on representability" and generalised it into a claim that on
this panel every genuine dynamical state constrains the surface more than a free reparameterization does,
losing 1.8x to 2.8x. That claim was wrong and is withdrawn in full.

It rested on a best in-sample interior RMSE of 0.011274 over a 23,040-shape grid. Running the identical
objective under continuous optimisation instead gives **0.005362**, against **0.004503** for the
two-exposure form under the same treatment. A 19 percent gap, not a factor of two. The grid had not
covered the region the family wanted: the continuous optimum sits at rho 0.01, fast rate 49.8,
gamma_broad 0.01 and offset 2.3e-04, and every one of those is outside the grid's range. The tell was
there and was misread — every selected parameter sitting at a grid bound was recorded as evidence the fit
was "running toward a degenerate limit" when it was the ordinary signature of a grid that stops too soon.

Re-evaluated properly, with continuous selection inside inner folds, nested 3x3 random seed 0:

- interior OOF RMSE **0.007289**, beating repaired RPL 0.007575, and within 0.000129 of the two-exposure
  form's 0.007160
- **Regret@5 0.000000** — the top five contains the observed optimum outright
- the two forgetting rates separate stably in every outer fold: (3.17, 46.61), (3.50, 43.50),
  (3.21, 55.10). That is a real two-timescale identification, and it is the signature the family was
  registered in advance to predict.

Three gates still fail. Regret@1 is 0.005936 against 0.004842. The raw optimum is (0.000, 0.545), ON the
boundary in early code share, at distance 0.1097 — so the pre-declared failure mode did occur, and
sharing one rate between benefit and damage did NOT prevent the free lunch as the registration argued it
would. Predicted gain +0.017665 against observed +0.009594 overshoots by 0.008071. Identification is
partial: lambda is well determined, but rho at 0.0050 and both readout exponents at 0.016 and 0.0413 sit
at or near their lower bounds, a degenerate ridge where slow acquisition and a flat readout trade off.

The methodological lesson is the durable part of this round: a grid search reported a representability
ceiling that a continuous search beat by a factor of two. A ceiling claim from a grid is not a ceiling.

### Independent review, and the structural result that ends the search on this panel

Codex 5.6 Sol at max reasoning reviewed SUR-095 and SUR-096 adversarially. It independently reproduced
both self-corrections made above, and found five further things, three of which were errors of mine.

Confirmed against me. The blocked-protocol collapse is NOT specific to this form: repaired RPL's blocked
RMSE is already checked in at 0.026530 in `reference_outputs/repaired_rpl_wsd80_controls_20260731/`,
against 0.026371 here, so both extrapolate badly and they are essentially tied. The registry's statement
that the incumbent had not been measured under that protocol was false and is corrected. The
identification claim was overstated: bounds were checked only on the full-panel shape and only the slow
horizon was printed per fold, whereas across outer folds the selections are `slow [0.45, 0.40, 0.40]`,
not identical, and `gamma_broad [0.02, 0.10, 0.02]`, at its lower bound in two of three folds. And the
RMSE comparison is not a win: a paired row bootstrap on the difference against RPL gives a 95 percent
interval of about `[-0.00195, +0.00114]`, crossing zero, with this form winning one fold, tying one and
losing one. "First nested win" is withdrawn; **level with the incumbent** is the honest statement.

Also confirmed: no programmatic outer-fold leakage, identical run ids, outcomes, interior mask, outer
partitions and metric definitions, and the negative controls reproduce independently. A real bug was
found and fixed — the checked-in SUR-095 fitter constructed scalar forgetting values against the
tuple-valued API and raised `TypeError` on execution.

The review also supplied a state construction I had not tried, which directly satisfied the reopen
condition I had written: solve the same ODE, then invert the *tied-policy* terminal map and read the
power law off `E_eff = F^-1(s_f)`, so that `E_eff` equals total epochs exactly on the diagonal for every
`lambda`. Registered and tested as SUR-097. The identity holds to 1.1e-13, so the construction works; it
costs fit, giving 0.007567 in-sample with two rate components against 0.005362 for the unconstrained
state and 0.004503 for the two-exposure form. Across all four forms the in-sample ordering is monotone in
how much each constrains the surface.

**The result that matters most is an identification argument, not a fit.** With one fixed two-phase
duration, any terminal state is observationally only a function of `(w0, w1)`. So on this panel a genuine
dynamical state and a clever static reparameterization are not distinguishable *even in principle*, no
matter which functional form is tried. That explains the whole shape of this project's history: every
mechanism proposed has been fitted, ranked and argued about on a panel that cannot identify mechanism.
The charter's requirement for "an actual temporal interaction" cannot be adjudicated here. It needs a
second schedule, or intermediate-trajectory measurements, and that is a measurement decision rather than
a modelling one.

## 2026-08-07, later — testing the experimenters' two mechanisms directly

New hypotheses arrived from Will and Kaiyue: two-phase advantage comes from scheduling away gradient
conflicts late in training, and from avoiding excess epoching on select domains. Both are time-dependent.
This round tested them as model structure rather than treating them as motivation.

### What the conflict hypothesis buys

Every form in this project constrains benefit amplitudes non-negative on `(E + offset)^-gamma`, so more
broad text always LOWERS predicted code BPB. If broad-text gradients are anti-aligned with the code eval
gradient late, then heavy late broad text must RAISE it, and no prior form could express that. Adding a
signed channel in the decay-phase off-domain share fixes it, and the channel is supported:

- coefficient fits strictly POSITIVE, the predicted sign
- stable across outer folds at +0.10582, +0.09229, +0.10281, full panel +0.12460
- nested interior OOF RMSE 0.005252 -> 0.004948

The paired row bootstrap on that RMSE difference is `-0.000304` with 95 percent interval
`[-0.000770, +0.000278]`, crossing zero, so the fit improvement alone is not decisive. The fold-to-fold
stability of the coefficient is the stronger evidence, and its sign is a prediction the batch-size sweep
can test independently, since gradient conflict should scale with gradient noise while epoching harm
should not.

### What the two mechanisms are each made of

With the damage horizon free, the predicted two-phase gain decomposes:

- full model gain **+0.014036** against observed +0.009594
- conflict coefficient to zero: gain collapses to **+0.000506**, so conflict carries **96.4 percent**
- damage amplitude to zero: gain **+0.008393**, so epoching carries **40.2 percent**

They exceed 100 percent because they interact, and the interaction is the useful part. Without damage the
optimum runs to (0.410, 0.855). So repetition harm is not what creates the two-phase advantage; it is
what bounds how far the schedule should be pushed. **Conflict creates the opportunity, epoching limits
its exploitation.** An earlier version of this decomposition reported the epoching share as 0.0 percent;
that was an artifact of pinning damage to uniform-weighted total epochs, and is withdrawn.

### What did not work

Per-domain value schedules, the natural reading of "starcoder is as valuable as nemotron at the start but
much more so at the end", are NOT supported. At matched capacity they buy 0.004503 -> 0.004454 for two
extra parameters, and the pre-declared signature fails outright: `phi_broad` was predicted near the
uniform-in-time 0.203 and comes out between 0.63 and 0.99 in every variant. Whatever those parameters are
reading, it is not a per-domain value schedule.

Spacing-aware damage, from the note that forgetting is set by the gap between repeat views, is refuted.
It fits worse (0.007962 against 0.006117 on identical folds) and fails at the one thing it was built for.
The reason is diagnosable: the threshold above which repetition hurts fits at 11.91 epochs per unit run
time, so at low early share the phase-0 rate never reaches it and the convexity that would push toward
even spreading never engages.

### The measured cliff at zero early share

The persistent failure across every candidate is an optimum at early code share exactly 0.000, against an
observed 0.100. That is not an unconstrained extrapolation. The panel SAMPLES early share 0.0000, and
those rows are clearly worse than early share 0.0250 at matched late share:

| late share | early 0.000 | early 0.025 | penalty |
|---|---|---|---|
| 0.45 | 0.945904 | 0.938689 | 0.007215 |
| 0.50 | 0.941774 | 0.936949 | 0.004825 |
| 0.55 | 0.940216 | 0.938095 | 0.002121 |
| 0.60 | 0.941685 | 0.939337 | 0.002348 |

A penalty of 0.002 to 0.007 BPB concentrated at exactly zero. Every additive model in the registry places
its optimum at that point, and no additive model can represent the cliff, because a sum of independent
exposure channels has no way to say "this only matters when the domain is entirely absent early".

Gating late exposure by early exposure, `late * exp(-kappa * early)`, is the first non-additive candidate
in the project and the first to leave the boundary: the optimum moves from (0.000, 0.547) to
(0.033, 0.520) and optimum distance improves 0.1107 -> 0.0704. It costs fit, 0.005901 against 0.004941.
The gating form was the first one tried and was not searched, and the measured cliff is sharper than an
exponential gate can represent, so this is the most promising open route rather than a finished one.

### Where the numbers stand

Best nested interior OOF RMSE this round is **0.004941**, against repaired RPL's 0.007575 — a 35 percent
improvement, and most of that came from replacing grid selection with continuous selection inside the
inner folds. Regret@5 reaches **0.000000**, meaning the top five contains the observed optimum outright.
Regret@1 is stuck at 0.005936 against a 0.004842 limit, and the optimum-distance gate still fails.

### Composite dynamics surrogate (SUR-101): three gates robust, the fourth is not

Assembling the three mechanisms as independent channels — signed conflict, multiplicatively gated
absorption, policy-responsive horizon — produced the best result the project has had, and a 4-of-4
headline that did not survive its own robustness check.

With the Hill exponent capped at 12: interior OOF RMSE 0.004843, Regret@1 0.002842, optimum distance
exactly 0.0500 against a 0.05 limit, gain error 0.000673. Four gates. Widening that cap to 40 flips the
SAME configuration at the SAME seed to three: the exponent runs to 35.06, still on a bound, and the
optimum jumps from (0.050, 0.500) to (0.101, 0.394), distance 0.1063. The distance gate had passed only
by tying the limit exactly. This is the SUR-094 failure mode a third time, caught this time before
publication rather than by review.

Across six configurations (three fold seeds x adaptive horizon on/off, surface grid 801, exponent to 40):

| gate | passes | range |
|---|---|---|
| interior OOF RMSE | **6/6** | 0.004684 – 0.005350 (RPL 0.007575) |
| gain error | **6/6** | 0.000126 – 0.003846 |
| Regret@1 | **5/6** | 0.002842 – 0.005936 |
| optimum distance | **1/6** | 0.0416 – 0.1141 |

The failure is now precisely located, and it is not a fit failure. The surface RANKS policies well and
stably — Regret@1 reaches 0.002842 in three configurations, equalling repaired RPL and selecting the same
policy (0.050, 0.500), with Regret@5 reaching 0.000000 — while beating RPL's RMSE by 30 to 38 percent
under every setting. What wanders is the continuous argmin: early share 0.033 to 0.109, late share 0.386
to 0.521.

Three component caveats worth carrying, all of them against the interesting story:

The policy-responsive horizon fits at `delta` about -0.010 in every fold, which is effectively zero. It
improves results as a nuisance parameter that perturbs the selection landscape, not as a mechanism, and
must not be reported as one.

The conflict channel's SIGN is not robustly identified. It fits +0.19477 with the groundwork column
present and -0.10524 with its own exponent free and groundwork absent. The "sign confirmed" statement in
the SUR-098 entry is over-strong and is qualified: what is stable is the coefficient across FOLDS at a
fixed column set, not across column sets.

The gate scale `kappa` swings 2.76 to 17.06 across folds, and pinning it to the physically natural value
of one early code epoch (early share 0.0474) makes things worse, 2 of 4 gates. So the Hill exponent is
standing in for a cliff sharper than this family can express, and it will run to whatever bound it is
given.

### SUR-102: gated absorption clears the distance gate, Regret@1 becomes the blocker

The structure that finally moved the optimum was giving the BROAD domain a groundwork gate too, not just
code. Late exposure of each domain counts only through a Hill gate on that domain's early exposure, so
late tokens are worth something only to the extent something exists to absorb them — and the general
ability that code capability rides on has to be built before the decay phase as well. Adding the broad
gate took optimum distance from 1 of 6 seeds to 5 of 6.

Six seeds, nested 3 outer by 5 inner, continuous selection, surface grid 801:

| gate | passes | range |
|---|---|---|
| interior OOF RMSE | **6/6** | 0.004696 – 0.005865 (RPL 0.007575) |
| gain error | **6/6** | 0.002396 – 0.003726 |
| optimum distance | **5/6** | 0.038770 – 0.083104 |
| Regret@1 | **3/6** | 0.000155 – 0.005936 |

20 of 24. The optimum is now stable at early share 0.051–0.064 in five of six seeds against an observed
0.100, late share 0.485–0.530 against 0.500. Regret@5 reaches 0.000000 in four of six.

It is not promotable. Regret@1 fails under the registered seed-0 protocol at 0.005936 against 0.004842,
short by 0.001094. And the two remaining gates trade against each other rather than both improving: at
three inner folds Regret@1 passes 4 of 6 and distance 4 of 6; at five inner folds distance rises to 5 of
6 and Regret@1 falls to 3 of 6. No setting passes both robustly, and tuning the fold count to pick the
better pair would be exactly the post-outcome selection that has already cost this project two headlines.

The Regret@1 failure is now a very specific thing: the failing seeds rank (0.050, 0.450) first where the
incumbent ranks (0.050, 0.500). That is one grid step in late share, worth 0.0031 BPB against an OOF RMSE
of about 0.0050, so it sits inside the model's own noise. It needs better LOCAL accuracy near the
optimum, not more structure.

One component finding is worth keeping regardless of the model's fate. Splitting repetition damage by
phase puts the stable-phase amplitude at EXACTLY 0.00e+00 in all six runs. The head assigns zero cost to
repeating data during the stable phase, which is a direct confirmation of the experimenters' claim that
early repeats cost little because the data is not yet differentiated. It is not adopted into the model
because it also costs about 0.002 BPB of fit, but as a measurement it stands on its own.

### SUR-102 controls, and why the last two gates cannot both be had

Full documentation is in `reference_outputs/gated_absorption_wsd80_20260807/report.md`, with a rerunnable
artifact and source hashes.

The controls are the strongest result the project has produced. Across all 29 panel metrics at seed 0 the
separation is total with no overlap: **all 26 broad-text metrics predict |phase gain| below 0.001, all 3
code metrics predict between +0.009 and +0.013.** C4 English is +0.000032 and Falcon RefinedWeb
+0.000008, against roughly 0.029 invented by repaired RPL — about a 900-fold reduction on the sharpest
control in this project. Wikipedia English is +0.000002, arxiv physics -0.000026, BBC news -0.000060.
On the positive side, github_python predicts +0.008993, within 0.0006 of the observed two-phase gain of
+0.009594, on a metric the model was never tuned against.

Then the thing that stops promotion, which turns out to be structural rather than a tuning shortfall.
Three protocol settings were run and the total never exceeds 20 of 24:

| setting | Regret@1 | distance | total |
|---|---|---|---|
| 3 inner folds, no ridge | 4/6 | 4/6 | 20/24 |
| 5 inner folds, no ridge | 3/6 | **5/6** | 20/24 |
| 5 inner folds, fold-selected ridge | **5/6** | 2/6 | 19/24 |

The ridge case explains the mechanism. A ridge on the scaled amplitudes, strength selected inside the
inner folds, does exactly what it was added to do: it cuts out-of-fold prediction variance and lifts
Regret@1 from 3 of 6 to 5 of 6. It also drags the optimum back toward low early share, 0.036 to 0.064,
and distance collapses from 5 of 6 to 2 of 6.

So **the groundwork gate channels are simultaneously the only thing that moves the optimum off the
boundary and the first thing regularization removes.** They are the least fit-supported part of the model
while being the only part that fixes the optimum's location. That is why improving the ranking costs the
optimum and improving the optimum costs the ranking.

The implication is not that a better setting exists to be found. It is that this panel's fit criterion
does not reward the structure strongly enough to hold it in place, so the gate channels need support from
outside the mixture panel. The simulated-epoching sweep is the natural source, since this decomposition
predicts disabling epoching removes about 40 percent of the gain rather than all of it.

## 2026-08-08: SUR-073 clears the corrected 60M source-scale gate

The five remaining native Table-9 evaluations completed, sealing all 277 rows of the
60M conditional epoch-dose panel. A source audit found four finite W&B Uncheatable
summaries that stopped before the exact final checkpoint: `p240_d34_m0` at step 2000,
`p247_d34_m32` at step 3000, `p251_d35_m2` at step 2000, and `p255_d35_m32` at step
4000. The versioned erratum uniformly reads Uncheatable from persisted step-4576
checkpoint metrics and verifies every frozen Table-9 value against its native 51-component
artifact. The candidate equations, folds, candidate grid, x32 holdout, optimizer, and gates
were not changed. Opus 5 approved the final provenance chain after two review passes.

The corrected frozen gate passes on both targets:

| target | selected form | candidate OOF RMSE | signed-linear RMSE | ratio | Spearman | calibration slope | x32 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| Uncheatable | q=0, global curvature, ridge=10 | 0.006855 | 0.015927 | 0.430 | 0.770 | 0.758 | 0.014029 |
| Table-9 | q=0, global curvature, ridge=10 | 0.011259 | 0.023512 | 0.479 | 0.775 | 0.801 | 0.032932 |

This is direct evidence that nonlinear finite-dose curvature is identified substantially
better than a signed-linear dose response at 60M. The intervention curves are not monotone:
Stack-Edu and Stack-Edu FIM peak near x8 on both targets, while x32 is worse than the best
measured dose for 17/39 buckets and is harmful relative to proportional for 15/39
Uncheatable and 12/39 Table-9 interventions. Across all interventions through x16, target
gains have Spearman 0.809, but only 21/39 buckets choose the same best multiplier, so target
specificity remains material.

This is not a model promotion. The Delphi full grid remains intentionally incomplete at
96/277 rows, so cross-scale transfer is not established; neither the frozen 300M tied OOF
gate nor the raw-optimum audit has run. Registry status is therefore
`passed_60m_source_gate_pending_cross_scale`.

Canonical artifact:
`reference_outputs/intervention_identified_signed_dose_potential_20260808_erratum/report.md`.
The evaluator and report were rerun after materialization and were byte-stable.

## 2026-08-08: full-pool intervention rejects exact cache replay, not the temporal residual

The completed WSD80 physical-full-pool panel contains six paired seeds for three
fixed high-D policies: asymmetric A, its aggregate-matched tied control B, and the
best tied control C. The physical cache support is 773.46 times the historical
subset and no exact source index wraps. All 18 endpoints and HF exports are
durable; five Iris children failed only during process teardown after final
evaluation and export.

The frozen A-vs-B contrast remains large without exact index replay:
`B-A=+0.111431` BPB, 95% CI `[+0.110228,+0.112634]`, with all six seeds favoring
A. Relative to the repeated-subset base, the gap increases by `+0.032346` BPB,
95% CI `[+0.028858,+0.035835]`. This falsifies the preregistered claim that exact
cache-index repetition mostly creates the fixed-policy phase effect.

It does not establish a global two-phase advantage. Policy B is a poor aggregate:
`B-C=+0.112086` BPB, while `C-A=-0.000655` BPB with 95% CI
`[-0.001439,+0.000129]`. A is therefore statistically indistinguishable from,
and slightly worse in mean than, the best tied control C. The intervention also
changes physical content and leaves semantic duplication uncontrolled. The
supported conclusion is that exact replay is not the fixed A/B mechanism; the
remaining temporal residual and any reoptimized policy-class gain require dense
surfaces with unique-token support varied independently of source distribution.

Canonical artifact:
`reference_outputs/starcoder_wsd80_full_pool_results_20260807/report.md`.

## 2026-08-08 — everything previously deferred, run

Four things this project had never run for any candidate: the mixture-blocked protocol, the 300M
39-bucket panel on both targets, the cross-panel consistency test, and the multi-target joint fit.
Full write-up in `reference_outputs/gated_absorption_wsd80_20260807/report.md` sections 11.1 to 11.5.

**Blocked protocol overturns an earlier reading.** Interior OOF RMSE 0.012067, 0.018740, 0.009696 and
0.015421 across four seeds against repaired RPL's checked-in 0.026530, so 29 to 63 percent better under
every seed. SUR-096 gave 0.026371 here, essentially tied, and blocked extrapolation went into the record
as a weakness of the approach. It was a weakness of that form only.

**300M Uncheatable beats HPR outright.** All-row RMSE 0.006447, 0.006606, 0.006284 against HPR's 0.006800,
so past the reference itself rather than merely inside the 5 percent gate at 0.007140. The hierarchical
per-bucket departures are what did it: family means alone gave 0.011070 to 0.011365. Regret@1 is 0.005816
on every seed against a 0.004678 limit, and fails. Table-9 misses its own RMSE gate by 1 to 9 percent
(0.013766 to 0.014237 against a 0.013651 limit) and fails Regret@1 clearly.

**The cross-panel result is the strongest evidence the project has produced.** The 300M panel is the
mirror image of WSD80: none of its 238 asymmetric policies beats the best tied one, and it supplies an
exact aggregate-matched tied counterpart for each. The same form, fitted independently per panel, predicts

| panel | truth | predicted |
|---|---|---|
| WSD80 | +0.009594 | +0.012632 |
| 300M Uncheatable | −0.001086 | −0.000458 to −0.000680 |
| 300M Table-9 | −0.002290 | −0.001046 to −0.002180 |
| 26 broad-text controls | 0 | all below 0.001 |
| 3 code controls | real | +0.009 to +0.013 |

It says "schedule matters a lot" where it does and "schedule does not matter" where it does not, with the
right sign and rough magnitude every time. That is the discriminating test this project exists to run, and
no earlier candidate was even tested this way.

**The multi-target joint fit is refuted, and the refutation contains the most useful number of the round.**
One shared shape across all 29 metrics gives mean interior OOF RMSE 0.008472 against 0.005540 for
independent fits, losing on 29 of 29 targets with a paired bootstrap interval of [+0.002092, +0.003775],
entirely positive. It also fails its own registered gate, improving only one of six code selection
diagnostics where two were required. Metrics on this panel do not identify a shared state better than
they identify their own.

But the primary target's Regret@1 under the shared shape is **0.000155**, against 0.002842 for the
independent fit and 0.002842 for the incumbent. Regret@1 is the standing blocker on every panel, and
sharing the shape cuts it eighteenfold while making the fit worse. That is a variance signature: the other
28 metrics regularise the nonlinear parameters, and the argmin of the surface becomes far more stable even
as the surface fits the primary less well. It is the ridge experiment's tension seen from the other side.

**Regret@1 is now one coherent blocker rather than several.** It fails on WSD80 (0.005936 against 0.004842),
300M Uncheatable (0.005816 against 0.004678) and Table-9 (0.009912 to 0.018271 against 0.005304), while
the surfaces beat the incumbent on fit and extrapolation and answer the phase question correctly
everywhere. The failure is not a mis-specified surface. It is that argmin-of-prediction is a
higher-variance functional than RMSE, and nothing in the model or its selection objective targets it.

## 2026-08-10 — Removing the hand-crafted features fixed the gate they were meant to serve

The round after the independent audit went in an unexpected direction. The audit withdrew the promotion
case on four verified defects, then a separate constraint from the experimenters decided the round:
**production swarms do not label buckets semantically.** They arrive classified by topic, with quality
splits inside each topic, and nothing says which bucket the eval is about. The audited model hand-assigned
three of its five structures using knowledge that one domain was code and the other off-target text, so it
could not be deployed as written.

Rebuilding without any semantic assignment did not merely preserve performance. It **fixed the gate that
had blocked the project throughout**, and the reason was an error of mine.

### The boundary kernel was on the wrong bucket

I had assigned the early-boundary kernel to CODE. Given freedom to place it, the model puts it on BROAD
text — the fitted scale runs 0.18–1.10 against an early-epoch range of 0–0.797, fully active — and
switches the code one OFF, its scale pinned at 316.2 where `exp(-21.089/316) = 0.935` is near-constant and
absorbed by the intercept.

Mechanically this is sensible. The failure mode is starving a domain's early phase; broad text is the
domain that CAN be starved early and recovered late cheaply; code at 21.089 epochs per unit weight has its
early share pinned by damage regardless. **Every structural fix attempted in the previous round —
saturating damage, gate placement, multi-start, the resolution-limit argument — was compensating for a
misplacement I had introduced.**

### Pool parameters by what determines them

A prediction I made then failed, and produced the next step. The corrected boundary-risk metric does not
predict which family gets an active kernel. The reason is a grouping mismatch: on the 39-bucket panel one
topic spans 4.80 to 1723.89 epochs per unit weight, a **359-fold range across all three exposure strata**,
so a single boundary scale served geometry it could not fit, while another topic had only two buckets to
fit a scale from.

So parameters now pool by whatever determines them. **Taste** — readout exponents and amplitudes, how much
a topic helps this eval — stays on topic. **Geometry** — the boundary scale, how fast a bucket exhausts its
pool — moves to exposure stratum. Neither needs semantics: pool size comes from the token counts the
exposure columns are already built from.

That stabilised the geometry parameter **44-fold at no fit cost**: worst-case spread across seeds fell from
120x (by topic: 0.73 to 87.67) to 2.7x (by stratum), with 300M RMSE unchanged at 0.006222–0.006550 and the
pair gain still correctly negative.

### Final tally, one configuration, honestly obtained

GEN-002 under multi-start, 11 seeds, configuration stamped `NF=2 NS=2` on every line:

| gate | result |
|---|---|
| Regret@1 | **11/11** |
| interior OOF RMSE | 10/11 |
| gain error | 10/11 |
| optimum distance | 9/11 (0.027–0.096) |
| **total** | **40/44** |

Regret@1 at 11/11 is the headline: the model picks a good policy on every fold draw, and that was the
project's original blocker. Primary-target gain calibration is near-exact, +0.009615 against an observed
+0.009594. Controls pass 25 of 26, the failure being twitterAAE at +0.005783 against a 0.005 limit.
300M beats HPR outright on RMSE with the correct pair-gain sign.

Note the trajectory as methodology tightened: **42/44 → 43/44 → 40/44**, each drop from removing a defect
rather than the model worsening. The first two merged selection seeds or configurations; only the last is
a single configuration with the optimiser fix applied.

### Four corrections, all mine

- A **43/44 tally merged two configurations** — six seeds pre-fix, five post-fix. Same defect the audit
  found, committed within an hour of my writing the rule against it. Result lines now carry a config stamp
  so merging is visible rather than silent.
- The **late-share "sign instability"** I had reported since SUR-098, and which the audit endorsed on the
  strength of my description, was an unidentified common mode: on a simplex the shares sum to one, so only
  the CONTRAST is estimable. It is stable at −0.314 to −0.357, spread 0.043.
- **300M Regret@1 has no valid reference.** Only one row of 520 lies within HPR's published 0.002678 —
  the observed best itself — so a model scoring at or below it must have picked that row and would score
  0.000000. Every 300M Regret@1 "fail" in this registry is uninterpretable.
- I claimed the optimum was **underdetermined**, withdrew that and claimed it **well determined**, and
  neither was supported. Both came from two-point spreads under mismatched optimiser budgets.

## 2026-08-10 — Codex review voided the round; corrected reruns

An independent Codex 5.6 Sol Max review of GEN-002 found a fatal solver defect, confirmed here from
scratch. The free design block held the intercept plus one late-share column per family. Family shares
sum to one, so that block is rank deficient by construction — on WSD80 a 346×3 matrix of rank 2 with a
third singular value of exactly zero. `fit_head` partialled it out with a plain reduced QR, which returns
one basis column per *input* column; the surplus direction lay entirely outside the column space
(measured distance 1.0), so the projector deleted real signal from the response and every constrained
column. Relabelling the two families — a mathematically identical model — moved predictions by RMS
0.090 BPB, max 1.664, against gates of 0.008.

The identifiability was documented in this round's own term specification and then never checked in the
solver. The defect was introduced *by* the round's central change: generalising one designated late-share
column into one per family. The semantic predecessor used only the identified contrast and is full rank,
so SUR-102/103 and the 300M port are unaffected.

Fixed with `column_space`, an SVD truncation; relabelling invariance restored to RMS 2.2e-12. Everything
was rerun. Results in `reference_outputs/general_surrogate_round_verdict_20260810.md` and the
`gen0*_20260810.txt` outputs beside it.

What the reruns changed:

- WSD80 total is coincidentally still 40/44, but distance gained a seat and gain error lost one. Now
  reported as two groups — 21/22 nested-OOF, 19/22 full-data — because summing them repeats an accepted
  defect from SUR-116.
- Seed 3, the "specification failure" that GEN-007 spent a long investigation on, now **passes** distance
  at 0.045277. That saga was chasing a solver artifact.
- Negative controls are 23/26 on five seeds, not 25/26 on one, and the failures are correlated: two of
  five selections inflate gain on nearly every metric at once.
- 300M components show four regressions, not two.

Two claims withdrawn. The mechanism headline — that dropping semantic features let the model place the
boundary kernel on the right bucket, evidenced by a large fitted code scale — is wrong: `exp(-E/k)`
linearises for large `k` and the unit-norm column scaling cancels the `1/k`, so a large scale turns the
term into a linear early-exposure feature at full amplitude rather than switching it off. And a four-seed
ablation headline of 16/16 did not survive to eleven seeds.

The ablation's real result is better than either claim. Over 11 seeds full 42/44 and ablated 41/44 are
indistinguishable, but the aggregate hides two unanimous opposing effects: without the kernel, gain error
is better on 11/11 seeds (median 0.000440 vs 0.003348) and RMSE on 10/11, while distance is worse.
Decomposed against the true (0.100, 0.500), the full model averages early share 0.0693 and late 0.4854;
ablated averages early 0.1049 and late 0.4539. **The kernel breaks the early share and fixes the late
share** — 4× better early without it, 3× worse late. That localises the specification gap to a single
coordinate instead of leaving it as a distance failure.

New and clean: cross-scale transfer. θ selected at 60M and applied at 300M (and the reverse), with only
the per-target head refit, recovers 80–98% of in-scale variance on 11 of 12 cells. The nonlinear
parameters look like a property of training rather than of one panel.

### Second Codex review, and the fragility it uncovered

The corrected round was sent back to Codex 5.6 Sol Max. It confirmed the SVD fix is correct (and proved
why: `Fb̂ + Câ = Py + MCâ`, so the non-unique `b̂` never reaches the predictions), then found five further
problems. All were verified before acceptance.

Two mattered. **The ablation mechanism claim was wrong-sign** — the boundary column has a non-positive
derivative in early share, so with a non-negative amplitude it *encourages* a higher early share, the
opposite of what I claimed it did. The measured shift is a total-refitting effect. **Cross-scale transfer
is far more qualified than claimed**: the own arm leaked its selection, the recovery statistic differenced
RMSE rather than MSE, and — decisively — the panels share their policy coordinates almost entirely (60M/300M
share 241 of 242; 300M/delphi_3e18 share all 280; epoch geometry identical). It measures shape stability
across model scale at fixed design, not generalisation.

Fixing the leak led somewhere neither review predicted. With a fair comparison, transferred θ scored as
well as in-scale θ, which raised the worry that θ simply does not matter. A random-θ control refutes that
— random θ recovers only ~0.25 of explainable MSE on 60M Uncheatable — but shows something worse: on 300M,
random θ drawn from the model's own search box is up to **23× worse than predicting the mean**.

That is not a hypothetical. The 300M component sweep at seed 2 produced an `arc_challenge` RMSE of
**562.68 BPB** (ratio 6586) from ordinary nested selection, and a 60M Table-9 cell where in-scale selection
lost outright to transferred θ. The component regression count is consequently seed-dependent — 4, 1 and 5
on seeds 0, 1, 2, with only `csqa` and `basic_skills_pattern` recurring.

**The GEN family has a robustness defect: parameter values inside its declared search box produce
predictions orders of magnitude outside the data range, and ordinary selection lands there often enough to
corrupt a sweep.** Bound the design or reject out-of-range fits before quoting any further gate number.

A regression test (`test_general_mixture_surrogate.py`) now locks in relabelling invariance; reintroducing
the old QR makes 4 of 5 checks fail.

### Follow-up: the robustness defect is real, partially fixable, and every remedy trades

Diagnosed the blow-up completely. Two unbounded sources: repetition damage `max(E-1,0)^tau` reaching
6.3e22, and the readout `(E+offset)^-gamma` capped only by `offset^-gamma` on buckets with exactly zero
weight — 41 such entries on 300M against 2 on WSD80, whose smallest real exposure is 0.1 epochs. That
asymmetry is why the catastrophe was 300M-specific and WSD80 looked clean throughout.

Both are now bounded. Damage uses its Hill form with the knee fixed at the independently measured 105
excess epochs; the offset floor is 1e-2. Random-parameter recovery of explainable MSE on 300M goes from
−1559 to +0.517, so random parameters now beat an intercept where they were 23× worse.

Then the trades, measured on both panels rather than inferred from one. The offset floor that would tame
arc_challenge (1e-1 or tighter) collapses WSD80 from 10/12 to 3/12 and then 1/12, with optima drifting to
(0.000,0.535) and (0.188,0.240). So 1e-2 is the maximum safe floor and it leaves 300M extrapolating.

A four-arm ablation separated the two structural changes. The saturating readout is free to slightly
better on 300M (0.005811 vs 0.005885) and costs one WSD80 cell, while delivering most of the benefit —
arc_challenge 18.5 → 2.84 on its own. Dropping the per-bucket departures block finished the job
(arc_challenge → 0.96, predictions entirely inside the observed range) but **costs 66% on 300M Uncheatable
RMSE against a 5% gate**, so it is rejected.

Two things worth remembering. WSD80 cannot inform the departures question at all: with one bucket per
family, `family_sums(near) ≡ near` exactly, so the block is a duplicate and dropping it is a no-op there —
its 11/12 is the baseline's number, not evidence. And the reference-normalised readout, which I nearly
recommended as the structural fix, is a **provable no-op** (8.9e-16) — the same defect SUR-109 was killed
for, nearly repeated within the same round.

Net: no free fix among the remedies tested. The open direction is keeping departures but shrinking them
much harder, with the penalty selected on held-out prediction *range* rather than error.

## 2026-08-15 - Convex-surrogate viability audit

Audited whether the dense StarCoder WSD80 surfaces support replacing the current nonconvex policy search
with a convex surrogate. This was local analysis only; no training jobs were submitted.

### Reproduction and provenance

- Base commit: `cb3d1b72b61b53756192afda2aa7a612d84b2ac9`.
- Entry point:
  `experiments/domain_phase_mix/exploratory/two_phase_many/audit_starcoder_wsd80_convex_surrogate_viability_20260815.py`.
- Command:
  `uv run experiments/domain_phase_mix/exploratory/two_phase_many/audit_starcoder_wsd80_convex_surrogate_viability_20260815.py`.
- Script SHA-256: `2f70b25119eb7b1bf0233bf9f3fb5e9514a309063f974deaab6013ccc226650a`.
- Report and machine-readable outputs:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/starcoder_wsd80_convex_surrogate_viability_20260815/`.
- The protocol records source hashes, four outer and three inner spatial folds, fixed fold seeds, ridge grids,
  multiplicity correction, and the fresh-selected-policy gate. All models fit all 125 coordinates per cell;
  the single-observation shape screen uses the 84 interior coordinates.
- `./infra/pre-commit.py` passed on the entry point. Nested SCS solves sometimes returned an
  `inaccurate` status, but every accepted fit was checked to have maximum shape-constraint violation at
  most `1e-4`; all full nonparametric convex fits used CLARABEL.

### Model-free shape evidence

- There are 180 exact collinear chord tests with four aligned seeds. Two positive Jensen gaps survive
  correction within their r0/m400 cell, but none survives one Holm family across all 180 tests. The two
  within-cell findings are tied-diagonal chords with gaps `+0.175204` and `+0.116432` BPB.
- The direct fixed-aggregate evidence is narrow: 24 aligned-seed chord tests, all on one coordinate
  geometry, with no positive mean Jensen gap. It is compatible with conditional phase convexity but cannot
  certify it.
- The larger pooled-variance screen finds 23 globally corrected violations in three m400 cells, but this
  result relies on extrapolating heteroskedastic variance beyond the calibrated aggregate range. It falls
  to four violations at 4x variance and zero at 9x. It is exploratory, not load-bearing.
- Decision: the current data do **not** familywise reject global convexity of raw BPB. They also do not
  establish it, especially along the fixed-aggregate directions relevant to phase control.

### Predictive and optimization audit

- An exact PSD quadratic costs essentially nothing relative to the identical unconstrained quadratic:
  median/p90 interior blocked-CV RMSE ratios are `0.9926/1.0030`. This is not adequacy: its absolute median
  interior RMSE is `0.169249` BPB, or `21.89` modeled coordinate SDs, and it fails all three fresh optimum
  gates.
- The aggregate-conditioned cubic
  `L(a, delta) = A(a) + B(a) delta + 1/2 C(a) delta^2`, with `C(a) >= 0`, is worse than its matched
  unconstrained form: median/p90 ratios `1.2292/1.3759`. Its median interior RMSE is `0.276367` BPB and both
  variants fail all fresh optimum gates. Ridge selection changes across outer folds in all 28 cells, with
  median penalty span `1000x`.
- A deliberately non-mechanistic max-affine convex regression tested whether the failures were merely due
  to low polynomial expressivity. It reaches median in-sample interior RMSE `0.047394` BPB but degrades to
  `0.229661` under blocked CV, has unstable regularization in 27/28 cells, and fails all fresh optimum gates.
- Fresh-confirmed selected-pair gains at r3 are `0.007576`, `0.010487`, and `0.013843` BPB for m100,
  m200, and m400. No tested convex model simultaneously locates these policies and predicts their paired
  gains inside the fresh 95% intervals.
- The matched unconstrained versions also fail, so convexity is not the only bottleneck. The tested state
  representations and their spatial generalization are inadequate.

### Decision and surviving route

No globally convex raw-BPB surrogate is promoted. This is not a proof that a useful convex representation
cannot exist: a fixed, mechanistically declared, strictly increasing response link could make a latent
score convex while preserving the optimum. No such link was frozen or tested here, and an arbitrary
learned monotone calibration layer remains inadmissible.

The defensible convexification is conditional rather than global. For fixed aggregate `a` and frozen
incoming state, test a phase-control objective

`V_a(delta) = c(a) - b(a)^T delta + 1/2 delta^T H(a) delta + sum[p,i] lambda[p,i] [e[p,i] - tau[p,i]]_+^q[p,i]`,

where `H(a)` is PSD, `lambda >= 0`, `q >= 1`, and each exposure is affine in `delta`. The feasible phase
contrast set is linear, so the inner phase solve is convex. Joint optimization over aggregate and phase is
generally nonconvex because aggregate-conditioned alignment, curvature, and retained-state transitions
introduce products or nonlinear state dependence. The next admissible route is therefore a flexible outer
aggregate model/search plus a convex inner phase-control solve, or sequential convex trust regions. Do not
reinstate the falsified hard fiber-optimality constraint.

This audit is WSD80 development evidence only. Any candidate derived from it must be frozen before testing
on 300M 39-bucket data and cannot use a post-hoc output calibrator. Two prior Opus reviews informed the
statistical and scope corrections; an additional final invocation returned no review and was not treated
as evidence.

## 2026-08-15 - Noise-aware repaired-RPL head ablation

Tested whether observation noise, heavy tails, or aggregate-matched phase differences explain the remaining
surrogate error without changing the repaired RPL equation. This was a local fixed-shape screen; no training
jobs were submitted.

### Frozen protocol and reproduction

- Base commit: `cb3d1b72b61b53756192afda2aa7a612d84b2ac9`.
- Primary entry point:
  `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_noise_aware_rpl_heads_20260815.py`.
- Primary command:
  `uv run experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_noise_aware_rpl_heads_20260815.py`.
- Primary script SHA-256: `0681a9b8000cf506492484937f92effa6d9583e94011bb8ef19d92ef40aa669d`.
- Optimizer diagnostic:
  `experiments/domain_phase_mix/exploratory/two_phase_many/diagnose_noise_aware_rpl_student_t_20260815.py`.
- Diagnostic command:
  `uv run experiments/domain_phase_mix/exploratory/two_phase_many/diagnose_noise_aware_rpl_student_t_20260815.py`.
- Diagnostic script SHA-256: `36641ee01c2435c2023a0c7bc2a8c935f4686e06211be1705264a918f2b08e53`.
- Artifacts:
  `reference_outputs/noise_aware_rpl_head_ablation_20260815/` and
  `reference_outputs/noise_aware_rpl_student_t_diagnostic_20260815/` under the two-phase experiment directory.
- Published fold-specific and full-fit RPL shapes and ridges were reused. Only the linear-head residual
  objective changed. WSD80 headline metrics used the previously published interior mask; 300M metrics used
  all 520 rows with correspondence-grouped folds.
- `./infra/pre-commit.py` passed on both entry points.

### Results

No noise-aware head is promoted.

- On WSD80 random folds, Student-t changes interior RMSE from `0.007575` to `0.007417` BPB. The
  aggregate-cluster bootstrap difference is `-0.000158`, 95% interval `[-0.000683, +0.000314]`. MAE does
  improve by `-0.000403`, interval `[-0.000667, -0.000119]`.
- On WSD80 blocked folds, Student-t is decisively worse: RMSE `0.033174` versus Huber `0.026530`, difference
  `+0.006644`, interval `[+0.004339, +0.009114]`; Regret@1 is `0.025108` versus `0.001900`; optimism errors
  above 0.05 BPB are 21 versus 9.
- A fixed-scale, two-start bounded Student-t diagnostic confirmed this is not an optimizer basin. All six
  solves succeeded, the least-squares and Huber starts agreed within `3.1e-14` objective units, blocked RMSE
  remained `0.032428`, and Regret@1 remained `0.025108`.
- Student-t and the paired auxiliary objective do not improve 300M Uncheatable or Table-9 RMSE beyond
  correspondence-cluster bootstrap uncertainty. The paired objective slightly improves some 300M phase-delta
  diagnostics but preserves or worsens end-to-end RMSE and Regret@1.
- The transferred WSD80 variance shape does not help. It estimates seed noise, while blocked residuals are
  dominated by model misspecification; it is nearly indistinguishable from homoskedastic Student-t.

### Review corrections and decision

An independent read-only Opus 5 review was run through the verified OAuth subscription. It found that the
WSD80 paired arm is not decision-valid: 255 differences reuse a small set of tied fiber anchors, but the
diagonal pair-variance approximation treats them as independent. The arm therefore concentrates anchor
leverage and double-counts shared noise. It remains a failed diagnostic, not evidence that paired information
is useless. The same defect does not affect the 300M arm, whose 238 pairs have distinct matched anchors.

The review also identified the strongest surviving lead: plain MSE beats Huber on the frozen blocked shape.
Its RMSE is `0.021544` versus `0.026530`, a difference of `-0.004986` BPB with interval
`[-0.008637, -0.002832]`; Spearman, Regret@1, and optimism counts also improve. This does not promote MSE
because the nonlinear shape and ridge were selected under Huber and only one spatial partition was tested.

**Decision:** robust losses improve ordinary interpolation by ignoring large residuals, but those residuals
are precisely the spatial anchors needed for extrapolation. Quantile or rank loss should not replace cardinal
mean-BPB regression. The next high-value local test is fully nested MSE shape/ridge selection across multiple
blocked-region partitions. Do not spend another round tuning Student-t, the current heteroskedastic instrument,
or the invalid WSD pair pseudo-likelihood.

## 2026-08-15 - Successor selected-policy acceptance protocol

Removed coordinate distance from the current deployment acceptance decision without rewriting the exposed
44-cell history. The successor protocol uses one fresh same-seed paired non-inferiority gate: for
`d_s = BPB_s(candidate) - BPB_s(reference)`, pass only when the one-sided 95% upper confidence bound on
`mean(d)` is at most the predeclared `0.002` BPB practical margin. A nonsignificant or underpowered comparison
does not pass. Coordinate distance remains descriptive.

The checked GEN-039 driver had two reproducibility defects: nested fold selection built the design on the full
panel while indexing a fold-local response, and its repository-root path prevented direct PEP 723 execution.
Both are repaired in `fit_frontier_wsd80_20260812.py`. The corrected distance-free diagnostics pass `33/33`
cells over 11 seeds: RMSE `11/11`, Regret@1 `11/11`, and phase-gain error `11/11`. Mean selected coordinate is
`(0.0616, 0.4807)`. Distance would fail seed 4 at `0.050062` and seed 8 at `0.100778`; seed 8 nevertheless has
observed-row Regret@1 `0.001495` BPB, showing why geometry is not a performance gate.

The current seed-0 recommendation `(0.06, 0.4875)` has no exact repeated observations. The new deployment gate
therefore returns `not_evaluable_missing_candidate_coordinate`; it is not promoted by the `33/33` diagnostic
tally. Existing five-seed repeats at `(0.30, 0.30)` versus `(0.10, 0.50)` provide a sanity check: mean candidate
minus reference is `+0.010072` BPB and the one-sided upper bound is `+0.012151`, correctly failing the `0.002`
margin.

Protocol and report:
`reference_outputs/wsd80_selected_policy_noninferiority_gate_20260815/`. Evaluator:
`evaluate_wsd80_selected_policy_noninferiority_20260815.py`. Four focused tests and targeted pre-commit pass.

### Review correction and protocol v2

The preceding entry records the first local draft and must not be treated as the final protocol. Independent
read-only Opus 5 review found that `n >= 5` left optional stopping open and had only about 52% power to certify
a truly equal policy under the observed `0.002181` BPB paired-SD proxy. It also found a zero-variance auto-pass
and correctly identified that observed-row Regret@1 is not the regret of the continuous full-fit optimum. The
seed-8 Regret@1/distance comparison in the preceding entry is therefore withdrawn as a justification for
removing distance.

Protocol v2 freezes candidate `(0.0575, 0.4900)`, reference `(0.1000, 0.5000)`, and exactly twelve fresh paired
seeds `20260816` through `20260827`. At the SD proxy, twelve pairs provide approximately 90.8% power at true
equality for the one-sided `0.002` BPB non-inferiority margin. The evaluator has no CLI overrides for alpha,
margin, or seeds; records input SHA-256; rejects identical coordinates and zero paired variance; and reports
manifest mismatches and nearest coordinates. Six focused tests and targeted pre-commit pass.

The frontier driver now also separates inner-fold seeds using the frozen `31000` base, uses equal tied and 2D
grid resolution for gain error, and asserts complete OOF coverage. The resulting model-form tally remains
`33/33`, with mean selected coordinate `(0.0664, 0.4811)`. Under the historical four-gate denominator the same
run is `42/44`, not `43/44`: distance fails seeds 4 and 5. The `33/33` count reflects removal of one diagnostic,
not an improvement in the model, and none of these diagnostics certifies the continuous selected policy.

The defensible reason to demote coordinate distance is that it measures proximity to a noisy, selection-biased
one-seed argmin and has no fixed mapping to expected BPB regret. The selected coordinate currently has no direct
performance evidence and remains not evaluable until the frozen fresh paired panel is complete.

Final targeted verification adds an explicit out-of-simplex coordinate guard: seven focused tests now pass.

The protocol CLI now compiles in the exact frozen candidate and reference coordinates rather than accepting
coordinate overrides under the same protocol identifier. This closes the final provenance loophole before
any fresh outcomes exist; eight focused tests pass.

### The with-repetition two-bucket cell, scored on runs the models never saw

The criterion this cell had been using was measuring noise, and replacing it changed every conclusion.
Predicted gain was scored against `y[best tied] - y[global min]` on the single-seed discovery panel.
Both terms are extrema of a noisy surface: across the sixteen discovery-positive blocks the discovery
gain exceeds the fresh-seed gain by `0.001412` BPB, against true gains of a few thousandths, and per-run
seed noise is `0.00310` BPB so a difference of two runs carries `0.00439` -- more than twice the `0.002`
margin. Separately, `np.argmin` was choosing recommendations for models that cannot make them: a
single-index model is exactly flat along a family of policies, and the raveling order returns the lowest
phase-0 share on that plateau, which at 4x replay is the `p0 = 0` edge where the true code optimum
happens to sit. Scoring the plateau's mean realised loss moved `two-bucket`'s regret there from `0.01134`
to `0.05340`.

The replacement was already on disk and had never been used to score a model: 28 preregistered blocks,
each rerunning a frozen tied and a frozen untied policy on five fresh seeds, 280 separately trained runs.
On that target, fitted per block, no candidate predicts the gain -- RMSE `0.016` to `0.029` against a
measurement floor of `0.00191` and a total truth range of `0.024` -- and reading the discovery argmin with
no model at all beats every model.

What passes is one mechanism fitted across the repetition ladder rather than per condition. The seven
supports are the same experiment with the StarCoder pool size changed, an exact doubling of the epoch rate
from `2.646` to `84.674`, sharing a zero-StarCoder run whose BPB agrees across supports to `0.00000`.
Fitted leave-one-support-out on the finite ladder, `two-horizon-split-damage` gives mean fresh-seed
decision regret `0.00048` to `0.00102` BPB with support-clustered upper bounds of `0.00122`, `0.00111` and
`0.00183` -- inside the margin on five of five seeds -- and Spearman `0.452` to `0.638`, `p <= 0.027`
throughout. It never sees the held-out pool size, and it matches the no-model baseline that reads the
held-out condition's own panel.

The registered ablation carries it. The split replaces the incumbent's single damage column with two
increments that sum back to it exactly, adding one head column and no shared parameter, so the ablation is
the incumbent itself: unsplit damage fails on both backbones, `0/5` and `1/5` seeds, while both split
forms pass `5/5`. Two boundaries came with it. Fitted per block on 125 rows the same split *hurts*,
dropping rank correlation from `0.610` to `0.310`, because the extra column needs pool-size variation to
be identified. And pooling the no-repetition support in sends `two-bucket` and `two-horizon`
anti-correlated with the truth, `-0.527` and `-0.547`, since one readout cannot span a 3000x epoch-rate
range.

Gain magnitude remains unpredicted: RMSE `0.031` to `0.147` against a `0.00178` floor. This is a decision
rule, not a value estimate, on one metric, between two frozen policies per block.
