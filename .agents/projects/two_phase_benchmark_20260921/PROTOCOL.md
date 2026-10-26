# Two-phase surrogate benchmark and promotion protocol

Status: implementation and baseline qualification, 21 September 2026. No new
surrogate candidate or training run is authorized by a benchmark pass.

The benchmark develops an endpoint predictor from asymmetric two-phase runs.
StarCoder surfaces supply the first empirical rejection gate. Surviving model
forms must predict and select policies on Delphi TPP40 and three older swarms.
Prediction error, frontier calibration, and selection are separate requirements.
All currently inspected outcomes are development evidence.

This protocol supersedes the expanded tied-plus-asymmetric fitting contract in
the [north-star charter](../two_phase_surrogate_north_star.md) for this round.
The change follows the user's requirement that an ordinary swarm need not buy
aggregate-matched tied counterparts. Historical results keep their original
data contracts and numerical attribution.

## Information available to fitting

The primary procedure receives two mixture vectors, realized phase fractions,
bucket identities and pool sizes, exposure-reference metadata, and final losses
from physically asymmetric policies. Calibration, shape selection, floors,
regularization, learned features, and early stopping use only eligible training
rows inside each actual inner or outer partition.

Tied outcomes, previously fitted single-phase MARINER coefficients, matched
gain labels, and measured phase-boundary scores are excluded from fitting.
Optional state-conditioned or tied-assisted procedures require separate named
tracks and cannot satisfy the primary input contract. Fitting does not require
a noise estimate from a tied reference. If a method needs a calibration point,
it must declare an eligible asymmetric point and its availability before folds
are generated; it must not score that same observation as held out.

For early and late mixtures p and q, define a = alpha p + (1-alpha) q and
d = q-p. The model must recover exactly the MARINER function class at d=0,
including its response link, exposure convention and coefficient constraints.
Its aggregate parameters are learned from the permitted asymmetric outcomes.
A proof of the parameter mapping and numerical checks are required; equality
at a few diagonal coordinates is insufficient. This structural property does
not establish identification or prediction accuracy at the diagonal.

The pinned class is the standalone `mixture_selection.py` response law:

$$f_t(a)=\phi_t+\exp\left(c_t-\sum_i A_{ti}b(E_i(a))+\sum_i B_{ti}h(E_i(a))\right),$$

where \(A,B\geq0\), \(b(E)=1-\exp(-(rE)^p)\), and
\(h(E)=\operatorname{softplus}(\log(1+E)-\tau)^2\). Its source hash is in the
prepared manifest. A candidate's diagonal is compared with this law using the
same aggregate parameters, with maximum absolute error at most 1e-8 BPB on
frozen simplex probes. Equality to an independently fitted model is not required:
different fitted coefficients can represent the same function class.

The phase-blind reference `aggregate_mariner_train_median` reuses the standalone's
complete shape/ridge and two-stage gamma search. It uses its existing training-median
floor branch, with no external repeat SD. On each actual fitting partition,
\(\phi=m-\gamma(m-\min y)\), where m is that partition's median; the standalone's
positive-gap fallback handles constant responses. Every inner fit recomputes this
floor from its own labels. This reference has the MARINER response class; it is
not the validated proportional-anchor MARINER estimator. Its performance must be
measured anew.

The primary cross-setting claim concerns the same equation and estimator
refitted within each setting. Transfer of fitted coefficients is a separate
claim, with its own held-setting or held-policy protocol.

## Frozen populations

| Population | Asymmetric fit rows | Tied evaluation rows | Use |
| --- | ---: | ---: | --- |
| Delphi TPP40, 20 September snapshot | 174 | 23 | Primary swarm |
| Delphi 3e18 canonical | 238 | 42 | Required older-setting benchmark |
| Llama 160M / 1.2B audited panel | 239 | 3 | Required architecture/scale benchmark |
| Llama 200M / 6B canonical | 238 | 42 | Required higher-TPP benchmark |
| StarCoder WSD80 dense surface | 326 | 20 | First empirical screen |
| Each of 28 StarCoder support/horizon panels | 99 | 26 | Full surface and support-transfer gate |

The Llama 200M expanded 520-row design contains 238 asymmetric rows and 282
tied rows. Its matched controls provide evaluation evidence and do not enlarge
the asymmetric fitting pool. The labels 60m and 300m in old paths refer to
approximately 157M and 201M total parameters, respectively.

Canonical-swarm and augmented-data tracks remain separate. The large Delphi
frontier archive and fixed-prefix branch archive are development stress tests,
after physical-checkpoint and policy-coordinate deduplication. They must not
silently increase the fit budget. Existing single-phase and two-phase tables
overlap at tied observations. A source filename containing heldout is not proof
that its outcomes are uninspected.

Every frozen population records source hashes, physical or archival observation
identity, policy identity, source cohort, training/data/subset seeds when known,
phase weights, realized and design phase fractions, pool/exposure units,
hardware, evaluator, exact target and availability mask, and exposure status.
Unavailable provenance is explicit. No remote retrieval or sealed-series reads
occur during preparation. Unverified sources are ineligible for promotion.

Only complete metric-specific observations enter a target's score. Missing
components are never filled with another task or silently reweighted. Uncheatable
uses its native byte weighting; Table-9 uses its frozen component inventory and
weights. An anchor selected for Table-9 but scored on Uncheatable remains an
Uncheatable observation. Paloma and Table-9 have separate coverage masks.

The baseline CLI fits each requested scalar target separately. Aggregate-target
smoke runs verify execution only: fitting one head to aggregate BPB is not the
same estimator as fitting every component and averaging its predicted BPB.
Objective-level MARINER comparisons require the latter construction with the
stored `objective_components` and weights, or an explicitly named direct-target
track shared by both candidate and reference. No such component-assembled
objective certification is claimed by the current smoke runs.

TPP40 has 197 of the 280 designed policies in this snapshot. The 83 absent
policies are not automatically sealed evidence: their current completion and
inspection status has not been refreshed. Before claiming temporal confirmation,
freeze their exact policy IDs, an exposure-status audit, candidate fits and
predictions before reading outcomes. Previously inspected or alias observations
remain development. No collection or new training is part of this benchmark task.

## Splits and dependence

Primary folds are deterministic, outcome-independent aggregate-neighborhood
blocks. Exact aggregate counterparts, aliases, repeated policies, and opposite
contrasts belonging to a designed aggregate group stay together. StarCoder's
nominal fiber membership is retained as a design label; physical exposures use
the realized split, and the mismatch is reported. Inner partitions are rebuilt
from the outer asymmetric training rows. Tied rows appear only in the test
neighborhood assigned to their group and never in an inner fit.

Groups use aggregate coordinates rounded to 10 decimals under the recorded
design fraction. Outer blocks use seeded k-means on square roots of asymmetric
group means. Tied groups follow the same group or nearest asymmetric centroid.
Inner blocks are rebuilt from outer-training asymmetric groups. The hashed
`folds.csv` is authoritative. Preparation reports fold and inner-fold sizes,
physical aliases, group counts and sparse challenges. Quick-screen prediction
cells need at least 10 asymmetric rows and two aggregate groups per outer test
fold. Smaller cells are diagnostic; they cannot silently satisfy a gate.
Llama160's three tied observations are diagnostic only for calibration; no
per-fold tied R-squared or calibration-slope gate is required there.

Source/design-family exclusions, held repetition support, and held horizon are
separate named challenges. Each must have enough independent groups; unavailable
challenges are reported as missing. A single acquisition cohort cannot supply
source-disjoint validation. Crossed-prefix tests exclude the requested prefix
and action identities at both tuning levels. An unseen-policy cross-horizon
claim excludes the shared policy identity at both horizons; same-policy scaling
is a different task.

The support-transfer plan holds one of the six finite supports out at a fixed
horizon and fits the other five. Every physical checkpoint appearing in the
held support is excluded from training, including asymmetric aliases; remaining
training checkpoints are deduplicated. Inner support exclusions repeat that
rule. Zero-StarCoder tied observations cannot be pinned to training. The full-pool
surface is a separate required prediction cell, with failure classified as
prediction failure. Independent within-panel fits do not establish support transfer.
Held-horizon unseen-policy tests must cross the horizon exclusion with held
aggregate groups; excluding every shared policy across entire dense grids would
leave no training data. That challenge requires a separate split receipt.

TPP40 region is not a hardware-equivalence certificate: original and recovered
runs can use different devices. Regional sensitivities and calibration bias are
reported separately. The TPP40 source pools expand with the token budget, so
this comparison does not isolate additional repetition over fixed pools.

## StarCoder first

The quick screen uses the dense WSD80 surface, Programming Languages, GitHub
C++ and Python, and C4/Falcon RefinedWeb controls where their atomic endpoints
are available. The full gate extends to all available preregistered atomic
targets and four horizons. The six finite-repetition supports form a joint
pool-size-transfer challenge; the full-pool condition is also scored separately.
It cannot be dropped because a model fails its wider exposure range.

Candidates must represent aggregate-dependent ordering, the good-loss region,
the tied restriction, and the observed response along fixed-aggregate fibers.
Directional and common departure effects are scored separately on feasible
opposite contrasts. Common departure effects are not constrained to be positive
everywhere: the historical high-aggregate fibers contain negative even effects.
At an unequal phase fraction, reversing contrast at fixed aggregate is not
equivalent to swapping the two mixture vectors.

Interior, boundary, and frontier strata remain visible. Whole-surface RMSE
cannot hide poor frontier calibration. Selected-policy value is primary to
exact coordinate distance: repeated observations did not reliably distinguish
some nearby interior and boundary optima. A sampled argmin is not a true
continuous optimum.

Reference baselines include phase-blind aggregate DSP and MARINER-class fits, effective-exposure DSP,
compact two-phase DSP/RPL, and an optional flexible surface diagnostic. Refit
all compared baselines on the same asymmetric rows and tuning folds. Historical
RPL scores obtained with tied rows are contextual references. ATOM-015 is a
finite-replay decision comparator; its poor gain magnitudes prevent using it as
a calibration standard.

The quick empirical fit reference is `effective_exposure_dsp`, named before
inspecting the new baseline scores. Joint-menu selection also compares with
`aggregate_mariner_train_median`. These are separate policies: both must pass.
RPL is eligible only where its native constant-exposure geometry is exact; it
cannot be a required reference on discrete-exposure support panels. Its native
robust head and full search are retained wherever it is run.

## Evaluation ladder

| Stage | Evaluation scope | Advancement |
| --- | --- | --- |
| Structural | Equation, units, tied mapping, limiting cases, rank and synthetic recovery on actual design | Required checks pass |
| Smoke | One StarCoder outer fold and fixed task subset, native estimator | Finite predictions and successful replay; no scientific promotion |
| Screen | All fixed StarCoder folds on the quick tasks; finite-support/horizon checks on survivors | Prespecified reference-relative fit, calibration and selection gates |
| Certify | Full StarCoder targets and required swarms, complete supported objectives, external development stress tests | All required evidence and numerical gates pass |
| Finalist | Repeated partitions, whole-procedure refits, ablations, raw and rounded optima, independent replay | Offline qualification only |
| Prospective | Frozen selected policies, independent measured outcomes and paired-seed comparison | Deferred; requires separately authorized training |

Cheap stages reduce evaluation scope, not the estimator. Search grids, floors
and regularization cannot be quietly simplified to make a screen faster.
Cache features that depend only on permitted metadata; learned transformations
remain training-local. Cache keys include inputs, code, splits and configuration.

## Scores and promotion

Prediction scores include RMSE, MAE, predictive R-squared (1-SSE/SST), signed
bias, and observed-on-predicted calibration slope/intercept. A constant observed
target has undefined R-squared. Undefined required scores make a gate incomplete.
The frontier is the lowest predicted 20 percent within a held menu; it is chosen
without observed outcomes. Optimism is observed minus predicted BPB, positive
when the forecast is too favorable.

Regret@1/3/5 is measured within an identical observed menu. Predictions tied at
the selection boundary are ordered by stable policy identity, never observed
loss. Selection among predictions produced by different outer fits is not a
single deployable policy-selection test; fold-menu regrets are summarized as
separate choices. Candidate/reference row identities and observed values must
agree exactly. Raw BPB across tasks or swarms is not pooled into one verdict.

Every outer fold also supplies a `joint_test` menu containing both its asymmetric
and tied test rows. This is the required selection menu: systematically optimistic
asymmetric predictions can lose to a tied policy even when each separate role
has good ranking. Regret@1 is reported; shortlist regret@3 is the quick-screen gate.

Matched tied/asymmetric and antithetic diagnostics require explicit compatible
pair identities. Shared controls create dependence. Repeated folds are not
independent training replications, and row bootstraps do not estimate new-seed
uncertainty. Report refit, conditional menu, and training-seed uncertainty with
their actual resampling units. Existing archives remain adaptive development.

Freeze numerical margins after baseline-only calibration and before new
candidate evaluation. A 5 percent relative RMSE allowance is a proposed screen
guardrail; it is not a universal certification tolerance. Selection, frontier
optimism and gain calibration need fixed panel/target budgets based on decision
resolution and available repeat evidence. Tolerances never scale with the
candidate's own RMSE. Missing margins, required populations or evidence produce
an incomplete verdict. Statistical resolution requires separate evidence.
An empirical screen advance is not certification. A candidate may advance by
matching the references within the fixed allowances; this rejection gate does
not require a demonstrated improvement.

The current scorer's `inconclusive` means that a policy requiring some improvement
found none among otherwise passing point estimates; it does not compute statistical
confidence. Quick-screen margins use five baseline-only partition repeats. For
each task, the reference-relative RMSE allowance is the larger of 5% of repeat-0
reference RMSE and the between-partition standard deviation of that metric.
Frontier RMSE follows the same rule. Regret@3 allowance is the larger of 5% of
reference RMSE and the between-partition regret@3 standard deviation. The upper
budget for positive frontier/selected optimism is the nonnegative repeat-0
reference value plus the larger of 5% reference RMSE and that optimism metric's
between-partition standard deviation. These are empirical triage tolerances,
not estimates of independent training-seed noise. Gates use fold-macro summaries,
with per-fold selections and frontier sets. Both reference policies and baseline
hashes must be frozen before any new candidate is scored.

Data/split contracts and promotion policies are separate artifacts. Freezing
margins must not invalidate prepared rows or folds. Evidence supplied to the
promotion CLI consists of local JSON records and SHA-256 receipts. Each record
names the evidence, candidate, benchmark protocol and data-contract hash, and
records a pass status. The CLI verifies these fields as well as file integrity.
Reference predictions must come from the exact frozen baseline receipt; candidate
and reference inputs are separate and restricted to required menus. Candidate
advances identify the candidate, prediction-file hash and policy hash. Missing
benchmark evidence makes a verdict incomplete; a candidate that omits a required
available prediction cell fails estimation.

Each candidate declares its equation, nearest prior route, new mechanism or
identification argument, parameter count, allowed training information, native
estimator, and cheapest falsifier before scoring. A rejection labels structural,
estimation, prediction, calibration, or selection failure. Reopening a route
requires a material change or a verified implementation defect. Mechanism
claims require matched ablations around the actual retained final model.

Known failure controls must fail the appropriate diagnostics. MTP-006's useful
local continuation choices and exact tied nesting did not prevent exaggerated
frontier gains. Its two historical anchor panels are both scored on Uncheatable.
A phase-null synthetic surface must not acquire a confident profitable phase
optimum. Synthetic recovery refits aggregate and phase terms, floors, tuning,
and optimization; knowing the true aggregate coefficients is a separate test.

## Optimizer qualification

Include the independently optimized tied restriction as a feasible two-phase
candidate. Compare multiple starts, boundary solutions, stationarity, exact
sampler rounding, predicted objective values, exposures, support distance, and
refit sensitivity. A raw optimum outside measured support is unvalidated even
if its coordinates are stable. Post-hoc KL, caps, clipping or gain rescaling do
not repair model qualification. Flat valleys can produce unstable weights with
stable loss, so both quantities are reported.

## Current executable scope

Local preparation, baseline fitting, scoring, empirical margin freezing and
receipt-checked promotion are implemented. The first gate has two frozen policies
under `reference_outputs/two_phase_benchmark_20260921/v3/screen_policies` and a
five-repeat baseline calibration. A candidate must pass both policies, with its
evidence receipts, to advance to full surface testing. No new candidate was scored.

Support-transfer row/split plans are materialized; their model fits and held-horizon
execution remain required later checks. Full certification, external frontier
qualification, candidate synthetic recovery, raw optimization and prospective
confirmation have not run. The [implementation report](../../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/two_phase_benchmark_20260921/REPORT.md)
records executed checks, commands, hashes and remaining coverage.
