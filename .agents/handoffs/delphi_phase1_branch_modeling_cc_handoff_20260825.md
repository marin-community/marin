# Handoff: model Delphi phase-1 branches after completed Wave 1

## Objective

Develop a simple, defensible surrogate for choosing the phase-1 mixture from one fixed trained prefix in the
39-bucket Delphi 3e18 setting. The scientific target is endpoint Uncheatable BPB. The operational goal is not
generic endpoint fit: it is to select a continuation that improves on the tied continuation and ultimately the
historical two-phase frontier.

All of Wave 1 is complete. It materially simplifies the old end-to-end problem because every fitted branch
starts from the same exact checkpoint, optimizer state, tokenizer, training configuration, data seed, and
prefix mixture. The only fitted action variable is the phase-1 mixture. However, the completed acquisition has
a severe support defect: all 100 fitted actions are far from tied, while the useful optimum is likely local.
Treat Wave 1 as a broad-response panel, not as evidence that the local optimum is identified.

Do not launch Wave 2. The user wants the model and acquisition logic reviewed first.

## Start here

The Fieldbook experiment is `exp_01m0sdg8kpna3kjnqe3xc8r9xf`, **Sequential prefix-branch optimization for
two-phase mixtures**. Read the latest research note first:

```bash
cd /Users/calvinxu/Projects/Work/Marin/marin
fieldbook note show note_01m0waeyamj8bwf4c2p01nj1zr --json
```

The canonical complete materialization is in this clean worktree:

```text
/Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-wave2-acquisition-20260825
```

Relevant immutable inputs:

- Wave-1 materialization manifest:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_kl0p05_wave1_results_20260825/materialization_manifest.json`
  - SHA-256: `0d3f239002637768f696decb095877591bd7406193ba7573ffa6fc0e87ed5ebc`
- Fitted matrix:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_kl0p05_wave1_results_20260825/branch_fit_matrix.csv`
  - 100 rows, SHA-256: `399ec79150a4f88de6d31917ac7fc1807410804f69c84400611a9eeaa6636e3c`
- All results, controls included:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_kl0p05_wave1_results_20260825/branch_results.csv`
  - 108 rows, SHA-256: `8841baadddc90efa8da8fa95bd76bb31748a2973388380ea38ea3dfcbe94e54d`
- Long-form Uncheatable components:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_kl0p05_wave1_results_20260825/uncheatable_metrics_long.csv`
  - 1,944 rows, SHA-256: `c9849f1283f3e943337fb69de5b249265bfe3b1b717c8a9fbc57e9e18b2a99c4`
- Noise-control manifest:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_phase1_kl0p05_noise_results_20260825/materialization_manifest.json`
  - SHA-256: `aefdf412fe73fb5cade0193c1061b2f942950d8e59da30428aa19dc647051e67`
- Frozen acquisition implementation and canonical panel geometry:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/design_delphi_phase1_common_branches_20260824.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/select_delphi_phase1_kl0p05_wave2_20260825.py`
  - reviewed commit `0dd17851fd31abb3238201a64ee38d60404bece2`

The exploratory critic driver is currently uncommitted scratch code:

```text
/Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-execution-20260825/branch_model_probe.py
```

Run it against the canonical 100-row materialization, not the stale partial copy in its own worktree:

```bash
cd /Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-wave2-acquisition-20260825
uv run /Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-execution-20260825/branch_model_probe.py
```

Use this only as an audit scaffold. Replace it with a clean reproducible driver if you continue the modeling.

## Frozen facts

Wave 1 contains 50 Wave-1A fit actions and 50 untouched Wave-1B fit actions, plus eight controls. Materialization
has zero missing rows and an exact cross-wave anchor. The prefix is
`shared_bounded_ensemble_kl0p05`; all fitted continuations start from its same post-update-2399 checkpoint and
run the final 20% of training through update 3007. Prefix training used v5p-8; continuations used v6e-8. The
paired hardware migration gate measured only `+0.00005281` Uncheatable BPB between v5p and v6e, well below the
run-noise scale.

Endpoint outcomes:

- tied continuation: `0.98989356` Uncheatable BPB;
- best fitted branch: `fit_wave1_extension_18` at `1.00035429`;
- best fitted branch minus tied: `+0.01046073` BPB, so every fit action is worse than tied;
- conservative run-noise SD: `0.00084656` BPB;
- historical 3e18 two-phase frontier: `0.98245525` BPB.

The best fitted branch does improve GitHub C++ (`0.74710315` versus tied `0.75805378`), but this is a component
tradeoff, not an Uncheatable win. GitHub C++ and the component table are diagnostics; do not silently change the
selection target away from Uncheatable.

## What the completed panel identifies

Broad response is learnable. A Wave-1A-developed, frozen `radial7_sparse_sqrt` critic transfers to untouched
Wave 1B with:

- Spearman `0.787179`;
- RMSE `0.010870`, versus `0.018147` for the Wave-1A mean baseline;
- top-1 regret `0.009777` within Wave 1B;
- best-of-top-3 regret `0.005975`.

Thus it ranks grossly harmful actions but does not place the optimum. On all 100 rows, compact sparse/radial
models obtain roughly `0.73-0.77` spatial-CV Spearman and `0.64-0.68` of mean-baseline RMSE, but their selected
actions are unstable. The full-panel best RMSE in the scratch ladder is `0.011390` from
`elasticnet_radial_sqrt`; that number is post-outcome development, not a confirmation result.

The four outcome-blind 14-column candidates frozen before Wave-1 outcomes (`hellinger_linear_14`,
`valley_quadratic_14`, `incremental_dsp_14`, and `hybrid_14`) all failed their Wave-1A eligibility gate: each had
worse RMSE than the mean predictor. Do not rewrite that result after seeing Wave 1.

An explicit even/odd critic was formulated after partial Wave-1B outcomes were visible. It is scientifically
plausible and selects the observed best branch in the A-to-B diagnostic, but it is post hoc. It must be frozen
before new local outcomes and tested on untouched directions before receiving any promotion claim.

## What the panel does not identify

The 100 fit mixtures have total-variation distance from tied in
`[0.43359, 0.81006]`, with median `0.58667`. By comparison, the historical frontier's phase-0/phase-1 contrast
has TV `0.10200` and Hellinger `0.08925`. There are no fitted actions between tied and TV 0.43.

This is load-bearing. A broad-trained radial critic fitted to all 100 rows predicts the held-out tied control as
`0.964255`, an unsupported `-0.025639` error relative to the observed `0.989894`, despite good broad CV. Wave 1
therefore identifies gross switch harm far from tied, not the local derivative or curvature that determines
whether any beneficial continuation exists.

Do not report a branch optimizer as solved from Wave 1. Good global RMSE or Spearman is insufficient.

## Modeling hypothesis to attack

Use a tied-anchored state-action critic. Let `w*` be the tied continuation and let `z` be a local simplex-tangent
coordinate for a candidate phase-1 mixture. Compare at least direct mixture displacement, centered log-ratio,
and square-root/Hellinger coordinates; choose the coordinate inside an honest split rather than by full-panel
fit. Enforce `Delta L(0) = 0`.

A minimal local form is

$$
\Delta L(z)
=
\beta^\top z
+
\alpha \lVert z\rVert_2^2
+
\gamma R_{\mathrm{rep}}(z),
$$

where:

- `beta^T z` is the odd, direction-dependent continuation value;
- `alpha ||z||^2`, with `alpha >= 0`, is a shared even switch-cost or local valley-curvature term;
- `R_rep` is at most one explicit, label-blind repetition-damage channel unless data supports more;
- `beta` must be sparse, low-rank, or otherwise strongly shrunk; and
- a free 38-by-38 Hessian is prohibited because its 741 quadratic coefficients are not identifiable.

For an antithetic pair on direction `d` and radius `r`, the design directly separates the two channels:

$$
o(r,d)=\frac{L(rd)-L(-rd)}{2}\approx r\,\beta^\top d,
$$

$$
e(r,d)=\frac{L(rd)+L(-rd)}{2}-L(0)
\approx \alpha r^2 + \gamma R_{\mathrm{rep}}(r,d).
$$

This is the main reason to prefer paired local rays over another broad Dirichlet panel. Challenge the isotropic
curvature assumption rather than accepting it automatically. If you propose diagonal, grouped, or low-rank
curvature, state its exact parameter count and show that the acquisition can identify it.

Semantic bucket families are banned. In particular, do not revive the old three-family 39-bucket partition.
Domain classification plus quality splits may be used only if explicitly justified and ablated; label-blind
exposure geometry, pool size, repetition, and runtime support are preferred. Do not use an unrestricted bucket
identity model merely because it improves in-sample fit.

It may be cleaner to retain two roles rather than force one head to do both:

1. a broad safety critic that rejects large harmful departures; and
2. a local anchored critic that ranks plausible continuations near tied.

Test whether this decomposition is actually needed.

## Draft local acquisition requiring review

The main session has a post-Wave-1 draft, not a frozen or reviewed launch artifact:

```text
/Users/calvinxu/Projects/Work/Marin/marin-delphi-phase1-local-wave2b-20260825/
experiments/domain_phase_mix/exploratory/two_phase_many/
design_delphi_phase1_kl0p05_local_wave2b_20260825.py
```

It proposes:

- 80 fit rows as antithetic centered-log-ratio rays around tied;
- 12 rows on the historical-frontier and proportional directions, at Hellinger radii 0.08, 0.15, and 0.23;
- 32 rows on eight sparse geometry directions, at radii 0.08 and 0.15;
- 36 rows on 18 dense maximin directions, at radius 0.15;
- eight sealed non-fit referee rows on four additional dense directions; and
- eight fresh tied controls outside the fit budget.

The fit budget is 40 antithetic pairs but only 28 distinct fit directions because several directions receive
multiple radii. Its directional design rank is therefore at most 28, below the 38-dimensional simplex tangent
space. This may be adequate for a sparse `beta` and coarse curvature, but it cannot identify an unrestricted
38-dimensional linear term. Audit this explicitly. Compare it with 38-40 distinct antithetic directions, or a
hybrid that preserves enough multi-radius rays to estimate curvature while spanning the tangent space.

The historical-frontier ray is outcome-selected from prior work. It is useful as a powered scientific direction
but is not outcome-blind discovery evidence. Keep it labeled separately in any gate or claim. The proportional
ray and sparse/dense maximin rays are outcome-blind.

The earlier reviewed broad Wave-2 contract
`0ba8d66e1b58e351f747cdfa8fd037ecd60d20ea315965ed732c4466d6d61b91` remains reproducible but is paused and
scientifically superseded by the observed local-support failure. Do not launch either design from this handoff.

## Evaluation contract

Preserve these distinctions:

- Wave-1A to Wave-1B is the only existing untouched transfer check for models developed on Wave 1A.
- Any model or feature proposed after seeing Wave 1B is developmental and must be frozen before local Wave-2
  outcomes.
- Spatial folds must keep nearby actions together. For paired local data, hold out whole directions: all signs
  and radii from one line must stay in the same fold.
- Hyperparameters, coordinate choice, sparsity, and curvature form must be selected inside training folds.
- The sealed referee directions must not be used for model selection.
- Confirmation and noise repeats stay outside the fit budget.

Report at least:

- absolute endpoint RMSE and Spearman for broad safety;
- odd-pair contrast error and even-pair curvature error for local mechanism fit;
- local top-1 and best-of-top-k regret;
- whether selected regret is distinguishable from the `0.00084656` BPB run-noise scale;
- tied non-inferiority and gain versus tied;
- support distance of every proposed optimum from measured actions; and
- coefficient and selected-optimum stability across folds and reasonable curvature constraints.

The primary gate is decision quality near tied, not coordinate distance by itself and not a count of unrelated
diagnostics. A candidate is not promotable merely because it ranks the broad harmful panel.

## Questions for CC

1. Reproduce the materialization hashes, endpoint ordering, and frozen Wave-1A-to-Wave-1B numbers. Flag any
   discrepancy before modeling.
2. Is the anchored odd/even form the right minimal critic for one fixed prefix? Try to break its mechanistic and
   statistical assumptions.
3. Which local coordinate gives the best balance of fit, convex or otherwise reliable optimization, and runtime
   simplex feasibility: direct displacement, centered log-ratio, or square-root/Hellinger?
4. Is one isotropic nonnegative curvature coefficient defensible? If not, what is the smallest identifiable
   extension?
5. Does a single repetition-damage feature add incremental out-of-fold value after radial distance, or is it
   redundant in this low-repetition branch setting?
6. Should broad safety and local ranking be separate heads? If combined, give a principled transition rather
   than an arbitrary gating threshold.
7. Does the draft 28-direction/40-pair panel identify the proposed model? Recommend the exact direction/radius
   allocation under the same 80-row fit budget.
8. Define the frozen pre-outcome model registry and direction-level evaluation protocol for local Wave 2,
   including the untouched referee gate.
9. Given the existing evidence, what exact model and acquisition should be frozen next? If the honest answer is
   that no model is yet promotable, say so plainly.

## Deliverable

Produce a concise adversarial report plus a reproducible modeling driver. State every post-hoc choice. End with
one concrete recommendation for the model class and the 80-row local acquisition. Update Fieldbook with the
reproduction, frozen candidate registry, and remaining blockers. Do not submit training jobs.
