# Two-Phase Surrogate Iteration Plan

Date: 2026-08-04

Status: active, model fitting paused at an identification boundary

North star: [two-phase surrogate charter](two_phase_surrogate_north_star.md)

Registry: [active approach registry](two_phase_surrogate_active_registry.csv)

Canonical logbook: [WSD80 mechanistic surrogate](../logbooks/wsd80-mechanistic-surrogate.md)

## Current Decision

Do not fit the proposed aggregate-potential plus HPR/late-unfamiliarity
composite. Independent review and the active registry show that it reopens
SUR-048, SUR-053, and SUR-071/074/075. Its only new ingredient, a TPP
moderator, is constant on the primary 300M panel and failed the post-outcome
gain-magnitude and raw-optimum audit on WSD80.

The immediate work has two independent legs:

1. resolve the already sealed aggregate-potential gate; and
2. obtain evidence that identifies a temporal mechanism rather than another
   endpoint phase coordinate.

Do not combine the legs until both pass their own gates.

## Frozen Facts

### Aggregate response

`WSD80-SUR-073` is the only active aggregate candidate. Its signed bucket
utility and convex power-divergence curvature are identified from the
independent conditional epoch-dose intervention. Twelve 60M Table-9 evaluations
remain missing. The frozen 60M both-target gate must resolve before Delphi or
300M outcomes are read.

If SUR-073 fails, report the aggregate route as rejected. Do not extend its
generator, exponent, ridge, or family grids after outcomes.

### Phase response

HPR remains the strongest 300M empirical baseline, and RPL remains the WSD80
representability control. Neither is an identified shared mechanism.

`WSD80-SUR-077` establishes only that a scale coordinate improves phase
residual prediction in most WSD80 cells. It does not identify total TPP:

- selector versus zero-phase exact sign-flip `p=0.204102`;
- pooled RMSE is 28.6% worse than zero phase;
- the maximum-TPP cell has a 55.5x gain over-prediction at a support boundary;
- total versus non-embedding TPP is unresolved; and
- TPP is strongly aliased with `N`, `D`, optimizer steps, and data reuse.

No existing endpoint-only temporal state is admissible for another coefficient
or response-link iteration. A new phase model requires a new identification
argument or intervention.

## Execution Order

### I0. Resolve SUR-073 without changing its protocol

1. Recover the twelve missing 60M Table-9 evaluations.
2. Run `materialize-60m` and `evaluate-60m` with SciPy 1.16.3.
3. Stop if either target fails the frozen nonlinear-dose gate.
4. If both pass, materialize and evaluate Delphi exactly once.
5. Adjudicate whether the selected generator is materially distinct from
   SUR-072 and Power-Ridge before reading 300M outcomes.
6. If admissible, run the frozen 300M tied-OOF and raw-optimum gate.

SUR-073 may become an aggregate backbone. It cannot promote a temporal model by
itself.

### I0.5. Audit 300M endpoint-noise identifiability

Completed under protocol
`3f3fb7c71cdac90af9b6089ccd8dae192b81d0e9b709897170838e38a3bfe07c`.
The audit identifies total run-level endpoint variance at one tied proportional
policy, not its component sources or its dependence on policy.

- A purpose-built 11-run proportional panel gives endpoint SDs of `0.001127`
  BPB on Uncheatable and `0.003330` BPB on Table-9. Ten runs sweep
  `trainer_seed`; because data and simulated-epoch subset seeds are unset, the
  measured SD jointly includes initialization, data-order, and subset
  variation. Their 95% upper confidence limits are `0.001796` and `0.005305`
  BPB.
- HPR's grouped-OOF RMSE is `6.03` and `3.90` of those local SDs. Five percent
  of HPR RMSE is only `0.30` and `0.20` local run-level SD. These ratios do not
  give the uncertainty of a panel-level RMSE difference; model comparisons
  still require paired or bootstrap uncertainty. The full HPR RMSE is not at
  the measured run-level floor.
- The expanded panel has 520 rows but 518 distinct policy coordinates.
  Proportional and UniMax each occur twice at tied coordinates. The
  proportional target values contain the 11-seed reference mean and are not
  independent endpoint replicates. UniMax supplies one physical cross-pipeline
  tied-neutrality check per target, not seed variance.
- Late-trajectory finite differences and policy-space semivariogram intercepts
  remain sensitivity diagnostics only. The former mixes smooth WSD-decay
  dynamics with temporal evaluation noise; the latter is unstable under the
  local-distance model and cannot be interpreted as a noise floor.

Do not repeat proportional merely to estimate its local total run-level
variance. A future calibration panel must first name its estimand: total
endpoint variance at another policy, decomposition of its component sources,
or same-seed policy-contrast variance.

Artifact:
[300M endpoint-noise and phase-design audit](../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/three_hundred_m_phase_identifiability_20260804/report.md).

### I0.6. Audit what phase-response dimension the design identifies

Completed under the same protocol as I0.5. The current 300M design has 238
asymmetric rows at 238 distinct aggregate policies, with one contrast direction
per aggregate. It therefore cannot nonparametrically separate aggregate
conditioning from phase direction for an unrestricted field.

- A full tangent-space bilinear operator has 1,482 raw coefficients but design
  rank 237. This is primarily parameter counting, not a newly discovered
  geometric obstruction.
- Aggregate and contrast sample subspaces are more aligned than random row
  pairing: squared canonical-correlation energy is `12.452`, versus null median
  `6.077` and 97.5% quantile `6.466` (`p<=0.0025`, the 400-draw resolution
  limit). This demonstrates design coupling, partly induced by simplex
  feasibility, not a causal phase mechanism.
- Constant and predeclared family-conditioned operators can be fit numerically,
  but their restrictions are assumptions imposed by the model rather than
  identified by repeated directions.
- A rank-one full-tangent bilinear operator has nominally 76 degrees of
  freedom, but rank-restricted injectivity and recovery were not assessed. The
  audit rejects only an unrestricted phase field; it does not reject every
  preregistered low-rank operator.

Do not call a regularized unique fit an identified mechanism. Reopen this route
only with a frozen low-rank recovery audit or with multiple independent,
preferably antithetic, contrast directions at shared preregistered aggregates.

### I0.7. Audit the unlicensed aggregate-conditioned low-rank route

The historical comparison and frozen outcome-free recovery test are complete.
The proposed rank-one response

\[
R(\bar w,\delta)=(u^\top\delta)(v^\top h(\bar w))
\]

shares the broad first-order tangent-bilinear class of the rejected LPSI
interaction, but neither family contains the other without extra assumptions.
SUR-013 is a sibling restriction that ties the phase field to the aggregate
gradient; low-rank order DSP is the aggregate-independent contrast-SVD
restriction. LPSI missed the corrected WSD80 optimum by `0.179`; rank-16
low-rank order DSP gave 300M exact-pair delta RMSE near `0.0130/0.0256` BPB.

The synthetic audit used the actual 238-row asymmetric design under frozen
protocol `b794b5d8f4e9874e8f34bd087416c40e6f615ae44d1d8355b8f60e2ee7d5e8bb`.
Both bases were structurally recoverable and passed at the Uncheatable noise
level. Both failed the primary Table-9 gate at 0.0039-BPB signal RMS. The full
76-DOF field had median held-fold signal-RMSE ratio `1.599`; the predeclared
40-DOF family basis missed the frozen `0.500` median limit at `0.506`. Random
and nominal geometry-stress factors failed similarly, so the result is not a
small stress-subset artifact. The threshold is not relaxed.

No endpoint model is promoted. The free full field is closed under the current
design and binding Table-9 noise level. The family basis remains a numerical
near miss on synthetic rank-one truth, not a physically identified phase
mechanism. Reopen only with multiple independent phase directions at shared
preregistered aggregates or an independently fixed physical basis. The active
local queue returns to SUR-073.

Artifact:
[aggregate-conditioned low-rank route audit](../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/aggregate_conditioned_low_rank_route_audit_20260804/report.md).

Recovery artifact:
[rank-one phase-field recovery](../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/rank_one_phase_field_recovery_20260804/report.md).

### I1. Outcome-free intervention power audit

Completed under protocol
`4094db303623fd9b37861c5ca75d78d2c945ccf6355860dbb8871115fc5fa65f`.
The audit uses only already-exposed repeat outcomes and architecture metadata;
it does not fit a surrogate or inspect sealed endpoint outcomes.

- Odd and even are the mechanistic component estimands, and a sign fixed before
  intervention is the oriented-gain estimand. Per-seed best-orientation net gain
  is descriptive only because taking `min(+d,-d)` is downward biased near a
  null.
- Ten complete antithetic triples with five same-seed observations estimate
  pooled-RMS odd/even SDs of `0.001363` and `0.001996` BPB. The three closest
  design-neighborhood triples estimate `0.001615` and `0.001810` BPB. All
  coordinates reuse the same five seeds, so variance upper limits conservatively
  use four seed-level degrees of freedom. The precommitted-oriented-gain SD is
  `0.002882` BPB.
- Between-condition power uses crossed seeds but assumes zero covariance, so
  the single-condition SD is multiplied by `sqrt(2)`. Effects are checked at
  `0.0039`, `0.0028`, and the replicated confidence-limit effect `0.001545`
  BPB, under point, 80%, and 95% variance upper limits.
- The 180-run low-TPP clock triangle is in the wrong regime (`4.77-7.83` total
  TPP). A high-TPP version reaches about 30 total TPP but costs `3.20e21` FLOPs,
  over six times the low-TPP triangle, and still supplies only two sufficiency
  contrasts rather than a held-cell scale-law test.
- The 182-run switch-time design has point/80%-UCL/95%-UCL MDEs of
  `0.003451`/`0.005374`/`0.008186` BPB for precommitted oriented gain. It does
  not power the `0.0028` shrinkage target even at the point variance estimate.
  No two-anchor, three-switch allocation within 200 runs does so.
- The unshrunk selected `0.0039` effect is feasible at the point variance (154
  runs). The stricter conclusion is intentional: it does not trust the selected
  point estimate. The `sqrt(2)` contrast variance is additionally conservative
  because same-seed switch differences cancel their tied control; the archive
  does not identify the remaining cross-switch covariance, so a smaller run
  count cannot be preregistered from it.
- The observed `0.0039` gain clears Holm correction over four primary anchors
  (`p=0.0261`) but not over all twelve repeated arms (`p=0.1194`). The `0.0028`
  value is a sensitivity target, not an inferential bound, and gain magnitudes
  do not automatically map to odd/even component changes.

Artifact:
[two-phase intervention power audit v2](../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/two_phase_intervention_power_v2_20260804/report.md).

The exposed overlap between the matched N-D grid and the fixed-157.5M ladder
is not a decision test: coordinate coverage, architecture, and winner's-curse
bias differ. Do not spend an iteration fitting or adjudicating another
descriptive TPP surface from those raw minima.

### I2. Matched-overlap scale intervention

Defer until a temporal response is identified by I3 or I4. A clock cannot
identify the response it is supposed to moderate. Cross `N` and `D` only after
`Psi` is frozen, so at least two cells share each selected TPP value.

The current triangle uses a h640 base, a h896 total-TPP match, and a h896
non-embedding-TPP match. It changes embedding fraction and compares two token
horizons at fixed h896 architecture, but still aliases token horizon with
optimizer steps unless batch size is intervened on. At low TPP it tests the
wrong regime; at total TPP about 30 it is too expensive to justify before
`Psi` exists.

Hold these quantities fixed whenever the estimand requires it:

- LR schedule shape and phase-boundary fraction;
- optimizer-step count where the estimand requires it;
- per-bucket unique-data pool and simulated epochs;
- coordinate design and seed allocation; and
- target and evaluation procedure.

Primary estimands are fixed-coordinate odd effect and even cost, not raw
best-observed minima or per-seed best orientation. Use crossed seeds and power
the between-cell contrasts, not only each cell's level. Predeclare an
outcome-free physical basis for the primary clock. If total and non-embedding
TPP remain equally defensible, preregister both and an outcome-free tie-break
rather than choosing from held-cell BPB.

The three-cell triangle can falsify clock sufficiency through two contrasts. It
cannot support leave-one-cell scale-law prediction or identify the causal clock.
Any promotion design must contain enough cells to hold one `(N,D)` cell out.

This intervention passes only if one clock predicts both held-cell phase-gain
magnitude and raw optimum location. Directional agreement or lower residual
RMSE alone is insufficient. Power must be recomputed if the estimand, seed
sharing, contrast, or coordinate changes.

If no scale coordinate predicts held-cell phase magnitude and optimum location,
remove scale moderation from the surrogate rather than replacing it with a
panel scalar.

### I3. Gain-to-penalty transition intervention

Repair the blocked SUR-076 design rather than submitting its current form. The
new design must provide:

- post-switch-only transition identification;
- seed-clean outer predictions with no shared tied-control leakage;
- matched contrast magnitudes at separated aggregate anchors;
- direct antithetic odd and even estimands;
- shape-matched static-equilibrium and decaying-shock nulls;
- enough repeated seeds to resolve approximately 0.0039 BPB rather than the
  prior 0.0086-BPB minimum detectable effect, and to retain power at a
  preregistered shrunken effect under a variance upper bound; the current two
  aggregates by three switch times by 13 seeds allocation fails this stricter
  gate;
- finite, interior transition rates; and
- a full-pipeline synthetic null and power audit including model selection.

The objective is to identify a state that predicts the terminal
gain-to-asymmetry-cost ratio. Boundary optimizer shock alone is insufficient.
I3, not I2, is the first phase-identification intervention. Freeze an exact
allocation only after it powers the precommitted oriented-gain contrast and
states a scientifically meaningful component-change threshold. No such
two-anchor, three-switch allocation fits the current 200-run envelope at the
`0.0028` gain target. Raising the budget or reducing anchors/switch times is a
scope change requiring a new protocol; do not claim the 182-run layout is
adequate.

### I4. Unique-volume versus repetition intervention

Hold materialized tokens fixed while independently varying unique pool size,
epoch count, and phase placement. This resolves a two-bucket confound that has
blocked replay and shortage interpretations: changing StarCoder weight changes
both composition and repeated exposure in the existing surfaces.

The intervention should distinguish:

- useful unique evidence;
- within-window repetition harm;
- cross-phase retention or forgetting; and
- a pure phase-weighted cumulative-dose null.

Do not add another replay feature until this intervention chooses among those
mechanisms.

## Conditional Model After Identification

Only if I0 and at least one phase-identification intervention pass, fit the
smallest composite

\[
L_y(\bar w,\delta,\tau)
=A_y(\bar w)+\Phi(\tau)\Psi_y(\bar w,\delta),
\]

where:

- `A` is frozen from the independently passed aggregate leg;
- `Psi(wbar, 0) = 0` at realized phase fractions;
- `Psi` contains exactly one preregistered odd state and one even state;
- aggregate conditioning enters through the quantity measured by the
  intervention, not a free function or fitted aggregate gradient;
- `Phi` is present only if I2 passes held-cell gain-magnitude and optimum-location
  gates. Otherwise set `Phi = 1`; do not count a failed clock as an ablation;
- if present, `Phi` has at most one dimensionless scale parameter and
  `Phi = 1` is an exact ablation; and
- target-specific linear amplitudes are allowed, but the transition and scale
  law are shared.

Required algebraic gates before outcomes:

1. tied neutrality at realized boundaries;
2. invariance to an unused boundary for tied schedules;
3. bucket-refinement covariance;
4. exact nesting of the phase-weighted-dose null;
5. full design rank and identified parameter directions;
6. canonical-angle separation of aggregate, odd, and even blocks;
7. synthetic recovery and synthetic temporal-null shrinkage; and
8. an interior transition-rate profile.

## Frozen Evaluation Ladder

1. Fit and select every nonlinear quantity inside grouped inner folds.
2. On 300M, compare against HPR refit on identical rows and correspondence
   folds; report all-row, asymmetric, exact-pair, calibration, tail, and
   `Regret@1/3/5` metrics.
3. Use WSD80 only after the 300M form is frozen. Programming Languages and
   GitHub code are positive controls; C4 and RefinedWeb are negative controls.
   Freeze `mixture_blocked_folds` before fitting. The positive-control gate
   requires Programming-Languages gain error no larger than `0.004439` BPB,
   locating the optimum within Euclidean distance `0.05` of `(0.10, 0.50)`, and
   not predicting a broad-text phase gain larger than `0.005` BPB. These three
   summaries are necessary but not sufficient. Also evaluate the eight frozen
   aggregate fibers at `a = 0.18, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80`:
   report the full predicted-versus-observed gain profile, require the best
   contrast to change sign around the tied optimum at `a = 0.30`, and reject a
   candidate whose profile error is driven by one fitted optimum while missing
   the remaining fibers. Freeze a numerical profile tolerance only after its
   anchor-level uncertainty and the repaired-RPL reference profile are
   reproduced from source predictions.
4. The 300M panel is exposed development evidence after many prior model and
   diagnostic looks. Before each new round, freeze equations, folds, effect
   size, and estimator, then append the look to the data-use ledger. Confidence
   intervals cluster by correspondence key and designed direction. The panel
   cannot become independent evidence again. The candidate passes only if it
   improves a phase-sensitive selection diagnostic
   beyond correspondence-key clustered paired-bootstrap uncertainty on one
   target, preserves the other target, and keeps core grouped-OOF RMSE within
   5% of HPR.
5. Reject any raw optimum with support TV above 0.35, maximum bucket weight
   above 0.30, median bootstrap policy TV above 0.10, or optimism above two
   candidate OOF RMSEs.
6. Do not require a nonzero 300M phase gain. None of 238 trained asymmetric
   policies beats the best trained tied policy on either target, so a stable
   near-tied optimum may be the correct result.
7. Only then evaluate a sealed deployment panel.

## Stop Conditions

Stop model-form iteration and report a negative result if:

- SUR-073 fails its sealed aggregate gate;
- no intervention identifies a terminal temporal state beyond cumulative dose;
- the phase term improves rank but not gain magnitude or raw optimum location;
- a new coefficient is constant within the panel that is supposed to identify
  it;
- the temporal state changes under a semantics-preserving bucket split; or
- improvement appears only after tuning on WSD80 or 300M outcome folds.

At the program level, stop after I0 and two independently powered temporal
interventions fail to identify a terminal state. Do not replace a failed
intervention with another endpoint coordinate or reopen a rejected family by
renaming its state.

Under those conditions, HPR remains an empirical 300M baseline and RPL remains
a WSD80 shape control; neither should be presented as the shared mechanistic
methodology.

## 2026-08-04 Paper-Inspired Round

The Su coefficient/exponent heuristic and the capacity-normalized repeated-data
law produced no promoted surrogate.

`WSD80-SUR-082` tested whether a late-weighted policy state changes the WSD80
token-horizon exponent. Its frozen shared-floor gate passed, but independent
review identified a policy-asymptote confound. Under per-policy floors and a
floor-free difference-profile audit, aggregate-conditioned curve shape remains
while late-recency shape ordering disappears. This is a descriptive rung
moderator, not a fixed-horizon surrogate or a reopened scale law.

`WSD80-SUR-083` blocks direct transplantation of
\(P R_D^\delta(N/U_D^\gamma)^\kappa\). Repeat count is aggregate-only along the
phase fibers of interest, and fixed-scale capacity factors are absorbed into
amplitude. Do not fit this term without a design that independently varies
model size, token horizon, unique-data supply, repeat count, and phase
placement.

The active queue is unchanged: wait for the independently frozen SUR-073
aggregate gate. A future exponent experiment must use untouched antithetic
aggregate fibers, at least six token rungs, seed replication, policy-specific
floors, and unique-data supply separated from token horizon. No additional
endpoint coefficient sweep is licensed.
