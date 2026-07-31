# Two-Phase Mixture Surrogate: North-Star Charter

Status: active

Canonical logbook: [WSD80 mechanistic surrogate](../logbooks/wsd80-mechanistic-surrogate.md)

Active candidate registry:
[two-phase surrogate active registry](two_phase_surrogate_active_registry.csv)

Historical 99-route registry:
[mechanistic surrogate discovery registry](../../experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/mechanistic_surrogate_discovery_20260719/approach_registry.csv)

Provenance audit:
[anti-circularity audit](two_phase_surrogate_anti_circularity_audit.md)

Fieldbook experiment: `exp_01kymbv4xg2k6yanx59zfjjg1g`

This document is the invariant objective for the surrogate-modeling effort. The
logbook records mutable hypotheses, implementations, and results. Future
sessions must not silently weaken this charter in response to a failed model or
an attractive intermediate metric.

## North Star

Develop and validate a **simple, mechanistically defensible parametric
surrogate** that uses data-mixture policies and their realized exposures to
select target-specific one-phase or two-phase training mixtures.

The primary outcome is the **observed performance of the policy selected by the
surrogate**, not ordinary fit quality. For both Uncheatable BPB and Table-9
macro BPB, the selected two-phase policy should:

1. outperform policies selected by the strongest existing Observatory models;
2. outperform the same model form fitted and optimized in the one-phase policy
   class; and
3. establish a new validated 3e18 frontier when the improvement is larger than
   experimental uncertainty.

The scientific stretch target is a two-phase advantage of approximately
`0.01 BPB` over the comparable optimized one-phase policy. The evidence may
show that this headroom does not exist in a particular setting. The model must
not manufacture this gap.

## Immediate Research Objective

The immediate objective is to solve the two-phase optimization problem in the
**high-TPP 300M / 6B-token, 39-bucket setting**, while retaining the ability to
represent the known two-phase optimum on StarCoder 80/20 WSD.

For model discovery, it is acceptable to use the complete high-TPP development
design:

- the original 280-row 300M two-phase fit panel; and
- its available aggregate/tied evidence, including the qsplit240
  exposure-average counterparts.

Together these provide the 520-row structured design currently described in
the logbook: 282 physically tied policies, 238 asymmetric policies, and 238
exact aggregate-matched contrasts. This is not constrained to a 280-total-row
fit budget.

**Sample efficiency is a later ablation.** Once a model form and fitting
procedure reliably select a good optimum, subsample the development design to
determine whether the result can be recovered from 280 total checkpoints. Do
not reject a scientifically correct model during discovery merely because it
needs the aggregate evidence that is already available.

## Scientific Questions

The work must answer three questions in order:

1. **Representability:** Can the surrogate represent a real two-phase advantage
   and locate the observed StarCoder WSD80 optimum without relying on boundary
   singularities?
2. **Identification:** Can the same functional form separate aggregate-mixture
   quality from phase-order effects in the 39-bucket high-TPP design?
3. **Selection:** Does optimizing the fitted raw surface produce a stable,
   plausible policy whose observed 3e18 performance beats the strongest
   one-phase and two-phase baselines?

A model that answers only the first question is a useful StarCoder model, not a
solution to the Marin mixture-selection problem. A model that fits the
39-bucket observations but fails policy selection is also not a solution.

## Required Model Qualities

A promoted model must:

- have a concise equation and a mechanistic interpretation for every term;
- define its latent state, state transition, response link, units or
  dimensionless quantities, parameter symmetries, and limiting cases;
- accept mixture weights, phase fractions, bucket sizes, and realized or
  simulated exposure, with predeclared bucket families allowed;
- have a clean phase-tied restriction and support fitting that restricted form
  independently on one-phase data;
- use the same functional form across swarms and targets, although
  target-specific fitted parameters are allowed;
- represent phase benefit through an actual temporal interaction, such as
  retention, forgetting, within-window repetition, consolidation, or
  state-dependent plasticity, rather than only a reweighted cumulative dose;
- make unsupported phase effects shrink toward zero;
- support direct optimization of a raw, unregularized response surface;
- report parameter count, effective degrees of freedom, identifiability, sign
  stability, and bootstrap stability; and
- produce a plausible raw optimum before deployment regularization is applied.

Avoid arbitrary output calibration, nearest-neighbor corrections, lookup
tables, unconstrained residual learners, candidate-series indicators, and
ensembles whose components have no common mechanistic interpretation.
Deployment regularization may constrain a risky optimum, but it does not count
as evidence that the response model is correct.

## Working Structural Decomposition

The current favored decomposition is a hypothesis, not a predetermined winner.
For phase fractions \(\beta_0+\beta_1=1\), write

\[
\bar w=\beta_0 w^{(0)}+\beta_1 w^{(1)}, \qquad
\delta=w^{(0)}-w^{(1)}.
\]

A useful model class separates aggregate response from phase response:

\[
L_y(\bar w,\delta)
=A_y(\bar w)+O_y(\bar w,\delta)+C_y(\bar w,\delta),
\]

where:

- \(A_y\) is the target-specific tied/aggregate response;
- \(O_y(\bar w,-\delta)=-O_y(\bar w,\delta)\) is a bounded ordering effect;
- \(C_y(\bar w,-\delta)=C_y(\bar w,\delta)\) is an even asymmetry cost, normally
  constrained to be nonnegative near a well-identified tied optimum; and
- \(O_y(\bar w,0)=C_y(\bar w,0)=0\).

Both phase terms must depend on the aggregate. StarCoder demonstrates that the
best phase contrast can be nearly null on the best tied aggregate while being
large and beneficial on a worse aggregate. The global two-phase optimum is
therefore a joint optimum over \((\bar w,\delta)\), not “the best phase order on
the best tied mixture.”

Retained power law is the current starting point because it captures the
StarCoder aggregate response and two-phase geometry. It is not privileged:
retain only components that survive nested ablation and cross-panel
falsification.

## Evidence Hierarchy

### 1. StarCoder 80/20 WSD: shape and representability gate

Use the complete measured surface, including both fixed-aggregate fibers and
replicates. The Programming Languages target and GitHub code targets are
positive controls for a meaningful phase mechanism. Broad-text targets such as
C4 and RefinedWeb are negative controls against inventing unsupported phase
gains.

A viable model should:

- preserve the observed Programming Languages two-phase advantage of
  approximately `0.009594 BPB`;
- locate the observed optimum near
  \((p^{(0)},p^{(1)})=(0.10,0.50)\), with small selected-policy regret;
- fit the interior and optimum region at approximately the local experimental
  noise scale;
- tolerate heteroskedastic boundary measurements without allowing boundary
  singularities to control the raw optimum; and
- shrink phase gain toward zero on targets whose observed sampled optimum is
  tied.

The same checkpoints evaluated on different metrics are not independent tests,
but they provide strong positive and negative controls for structural transfer.

### 2. High-TPP 300M / 6B, 39 buckets: primary identification gate

Use the original two-phase panel and aggregate/tied counterparts as a
structured design, not as 520 exchangeable absolute-BPB rows:

1. learn the aggregate response from tied and aggregate evidence;
2. learn phase residuals from asymmetric policies and exact
   aggregate-matched contrasts;
3. group cross-validation folds by aggregate neighborhood, counterpart pair,
   and designed direction so that related rows cannot leak across folds; and
4. combine the channels only after each clears its own diagnostics.

Evaluate Uncheatable and Table-9 separately. Compare every candidate with the
strongest Observatory baselines refitted on the **same expanded data and folds**.
Extra data may improve the final policy, but it must not be misreported as a
model-form improvement.

Primary diagnostics are:

- aggregate-response grouped OOF RMSE and rank;
- asymmetric-policy grouped OOF RMSE;
- exact-pair phase-delta RMSE, rank, bias, and sign accuracy;
- lower-tail RMSE and optimism;
- Regret@1, Regret@3, and Regret@5;
- observed-on-predicted calibration;
- performance stratified by phase divergence and support distance; and
- bootstrap stability of both the fitted parameters and selected policy.

The candidate must improve a phase-sensitive selection diagnostic beyond
paired-bootstrap uncertainty on at least one target, preserve the other target,
and not worsen a core grouped-OOF RMSE by more than 5%.

### 3. Secondary transfer panels

Use 60M, production Grug-MoE, StarCoder 50/50 cosine, and Delphi 3e18 to detect
misspecification and study scale or schedule dependence. They do not excuse a
failure on the primary high-TPP 300M panel.

Delphi 3e18 is a particularly important deployment check but a weaker source
for identifying phase terms because its TPP is much lower. Treat TPP as a
plausible moderator until a scaling experiment establishes its effect; do not
hard-code a TPP dependence merely to reconcile scales.

### 4. Frozen deployment validation

All previously inspected 3e18 outcomes are development evidence. Before final
validation:

1. freeze the model equation, fitting procedure, hyperparameters, data ledger,
   optimizer, and deployment-regularization sweep;
2. generate separate Uncheatable-optimized and Table-9-optimized candidates;
3. include the independently fitted one-phase restriction and current
   one-phase and two-phase frontiers as controls;
4. preregister the comparison and uncertainty rule; and
5. inspect the new training outcomes only after the panel completes.

Final success requires observed improvement over both the model's one-phase
ablation and the best pre-existing target-matched frontier. If the gap is near
the noise scale, use repeats before claiming a frontier.

## Optimization Audit

Every serious candidate must be optimized before deployment regularization.
For each raw optimum report:

- predicted BPB and predicted gain over the model's fitted one-phase optimum;
- maximum phase-specific bucket weight;
- maximum materialized or simulated epochs;
- aggregate exposure and repetition;
- phase total variation and other phase-divergence measures;
- distance to empirical and convex support;
- agreement across fitting folds and bootstrap refits; and
- sensitivity to optimizer initialization and solver choice.

Reject optima controlled by singular low-exposure features, unstable parameter
boundaries, one bucket receiving implausible mass, or a narrow optimizer basin
that disappears across bootstrap fits. A trust region can define a deployable
candidate, but it cannot rehabilitate a structurally wrong raw surface.

## Baselines and Fair Comparisons

At minimum, reproduce or import source predictions for:

- canonical DSP;
- effective-exposure DSP and geometry variants;
- separate heads;
- original and bucket-resolved GRP variants;
- compact retained state;
- hierarchical phase replay and any defensible band variant; and
- retained power law.

Use common folds, common rows, common target definitions, and common
optimization constraints. Report both:

1. **best-achievable comparison:** every model refitted using the expanded
   high-TPP development evidence; and
2. **sample-efficiency comparison:** surviving procedures evaluated under the
   same fixed acquisition budget after model discovery.

Do not mix these claims.

## What Does Not Count as Success

The following are insufficient:

- better pooled RMSE or Spearman without better optimum-region selection;
- an impressive predicted optimum with no observed validation;
- success only on Programming Languages StarCoder;
- fitting broad metrics by suppressing a phase mechanism that is otherwise
  uncontrolled;
- a phase coefficient that collapses to zero or changes sign across folds;
- a lower error obtained from an unconstrained output calibrator;
- an optimum made plausible only by strong KL, TV, or support regularization;
- a result selected after inspecting the same adversarial or deployment
  outcomes; or
- an improvement caused solely by giving the candidate more rows than the
  baselines.

## Near-Term Execution Order

1. Reproduce the expanded-data Pareto baseline with common folds and metric
   definitions.
2. Repair retained-power-law estimation before adding mechanisms:
   standardize phase features, penalize every phase-control column, and select
   hyperparameters using lower-tail and Regret@\(k\) diagnostics rather than
   pooled RMSE alone.
3. Re-run the WSD80 positive/negative controls and the structured 300M gate.
4. If estimation repair fails, test the finite-contrast aggregate-conditioned
   even/odd decomposition above. Do not add another unconstrained residual
   feature.
5. Promote at most three mechanistically distinct families. Require nested
   ablations and an independent Claude Code review of each completed candidate
   round.
6. Freeze one candidate and its one-phase restriction before proposing new
   3e18 validation.
7. Only after a reliable optimum is established, run the 280-row
   sample-efficiency ablation and design a lower-cost acquisition procedure.

## Stopping and Negative Results

Do not assume a successful shared surrogate exists. Reject or block a route
when:

- its new coefficient collapses to zero or a parameter boundary;
- parameters or selected policies are unstable across folds or bootstrap
  refits;
- it improves pooled fit while worsening optimum-region optimism or regret;
- it requires target-specific structural changes rather than target-specific
  parameter values;
- it reproduces a previously rejected dose-reweighting model under new
  notation; or
- it adds complexity without useful improvement on both WSD80 and the primary
  300M panel.

A rigorous conclusion that no tested simple shared form can identify the
39-bucket two-phase optimum is acceptable. Such a conclusion must state which
mechanisms were falsified, why they failed, and what new experiment would be
needed to distinguish the remaining possibilities.

## Current State at Charter Creation

- Retained power law models the WSD80 Programming Languages surface and optimum
  substantially better than the incumbent Observatory forms.
- Cross-metric WSD80 evaluation shows that this success transfers to code
  targets but not automatically to broad-text targets.
- The current phase block can produce unsupported low-aggregate or boundary
  optima; several singular asymmetry features were insufficiently regularized.
- The unresolved problem is no longer whether a compact model can represent one
  two-domain surface. It is whether a simple, regularized temporal interaction
  can be identified in the high-TPP 39-bucket design and optimized without
  extrapolation failure.

## Compaction and Handoff Protocol

At the start of every resumed session:

1. read this charter;
2. read the historical 99-route registry and active candidate registry;
3. read the latest entries and `Next` section of the canonical logbook;
4. inspect the current worktree without reverting unrelated changes;
5. identify which datasets are development, sealed, or newly confirmatory;
6. reproduce the current baseline before changing an acceptance gate; and
7. append equations, commands, results, rejections, and the next decision to the
   logbook.

Before fitting any new candidate, append its exact equation, nearest historical
route IDs, materially new mechanism or identification argument, prior-rejection
exception, and cheapest falsification test to the active registry. If those
fields do not identify a material difference, block the route without fitting.

If a new user decision changes this charter, record the change explicitly with
the date and rationale. Do not infer a change from an unsuccessful experiment.
