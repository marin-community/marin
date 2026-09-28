# Review of the proposed single-phase procedure freeze

2026-09-07. Read-only review of the [handoff](/Users/calvinxu/Projects/Work/Marin/marin/.agents/handoffs/single_phase_freeze_review_handoff_20260907.md), the implementation it names, and existing local outputs. No fitting, training, launches, or changes to the procedure were performed. Fieldbook was inspected for context; the Echo search returned access denied, so the review relies on local evidence. Run values below are the handoff's collected results, not a new live collection.

## Recommendation

Freeze `weibull_softplus_unscaled@kappa_floor_link_flat15` with cap 6 for both headline objectives after closing the specification and reproducibility gaps below. Keep cap 8 as sensitivity. The staged fitting procedure is defensible; another model-search win is not a prerequisite for freezing it.

The evidence supports substantially reduced optimism and useful mixture selection. It does **not** establish that the final model selects better optima than every predecessor. Across the pooled Table-9 optima bank, regret changes from additive WSPU's 0.0157 to the final model's 0.0151, while optimism falls from about 0.070 to 0.008. A different statistic, the equally weighted mean of within-source regret differences across 14 sources, is +0.0008 for final minus additive WSPU: positive means worse. Its descriptive 95% source-bootstrap interval [−0.0004, +0.0026] includes zero, with no multiplicity correction. These selection results do not establish superiority. Freezing for better calibration and a declared preference for conservative predictions is reasonable. That rationale should be explicit.

Separate three decisions: locking a reproducible procedure, obtaining independent confirmation of its selected policies, and supporting the paper's comparative or scaling claims. The first needs bounded offline corrections. The latter two determine which new training is worthwhile.

## 1. Close the procedure's specification and reproducibility gaps

**The anchor CSV is outside the cache fingerprint.** The registry reads an external [anchors.csv through `floor_anchors()`](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_registry_20260902.py:1514), but the frozen benchmark's [fingerprint and cache check](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_selection_20260906.py:196) cover frozen `inputs/` and Python sources, not this CSV. Changing anchors can therefore leave cached fits accepted under an unchanged fingerprint. Include anchors and their run identities in the complete input manifest and cache identity.

**The saved fit is insufficient for prediction without refitting.** [Fit shards](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_selection_20260906.py:237) contain shape, ridge, diagnostics, and predictions, but not complete fitted heads. [Reconstruction](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/materialize_delphi_link_validation_20260906.py:130) refits amplitudes and intercepts using the current registry, source, and anchors, then checks prediction parity. This is a useful check, but it should accompany a serialized inference artifact containing all coefficients, intercepts, floors, caps, shapes, component ordering, exposure mapping, and aggregation weights.

The final candidate CSV still matches its [saved hash](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_kappa_floor_flat15_validation_3e18_20260907/runtime_materialization/summary.json). Current model and registry sources differ from the hashes in that validation bundle; the current registry includes additional, separately named sensitivity variants, and the proposed final entry still has `joint_rounds=0`. Source drift alone does not invalidate the results. Resolve it by pinning the source that produced the chosen artifact and preserving an exact inference payload, rather than making future replay depend on a moving working tree.

**Correct these descriptions before declaring the recipe frozen:**

| Item | Actual procedure and required clarification |
|---|---|
| Anchor samples | Table-9 anchors import a mean and SD computed from **10 repeats**; Uncheatable computes them from **11 rows**. The handoff says eleven for both. Freeze the actual inputs or deliberately harmonize them and refit; do not treat the latter as a prose-only change. |
| Cross-validation loss | Both shape/ridge selection and gamma refinement score **RMSE in original BPB**, after reversing the link. Only the head-fitting objective is in log-deficit space. |
| Staging | Shape and ridge are selected at provisional gamma 2.5; gamma is then refined with **both** fixed. This is a legitimate staged algorithm. |
| Search | Gamma lies in \([1,6]\), so the search bounds for log gamma are \([0,\log 6]\). There are at most 24 scalar-search evaluations, with an additional evaluation when the default is scored. |
| Default | The default 1.5 is triggered when the selected gamma is at least \(6^{0.9}\approx5.02\). This detects proximity to the upper boundary; it does not test profile flatness. |
| Numerical link | Training uses \(\log\max(y-\phi,10^{-9})\). Prediction clips the linear predictor to \([-30,\min(30,\max z_{\rm train}+0.5)]\), where \(z_{\rm train}\) is the transformed **observed** response. |
| Floor edge case | If proportional anchor minus training minimum is nonpositive, the code substitutes a scale based on training spread. Include this in the executable specification. |

Sources: [anchor builder](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/build_delphi_floor_anchors_20260907.py:29), [Table-9 input construction](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/table9_reliability_fit_metrics_20260905.py:189), [CV loss](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_models_20260902.py:1088), [gamma search and staged fit](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_models_20260902.py:1468), [floor/link](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_models_20260902.py:402), and [cap construction](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_models_20260902.py:620).

The [manuscript's cap sentence](</Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/sections/methods.tex:48>) says “largest fitted value” and claims the cap prevents extrapolating harm beyond observed repetition. Neither follows from this implementation. It caps predicted log-deficit; it is not a bound on exposure coordinates. Likewise, an epoch cap alone does not guarantee that a proposed mixture is inside the swarm's joint support. Put the three-SD maximum directly in the floor equation, and state the actual search/default rule in the appendix.

The anchors are fixed across CV folds, while the minimum/gap is correctly computed from each training fold. I did not establish direct leakage of a held-out panel measurement: the extra Uncheatable proportional row differs numerically from the canonical panel baseline. Describe OOF performance as conditional on the supplied proportional calibration measurements, and identify those measurements explicitly.

Read-only checks of the 58 saved final fits found the stated four Uncheatable and six Table-9 defaults. Their floors match current anchors; all inspected training deficits are positive, with minima 0.00111 and 0.00183 BPB. There is no observed log-domain failure in these final fits.

**The selected optimizer outputs look numerically settled.** All three selected continuous restarts report successful convergence; feasible-start objective spreads are below \(4\times10^{-10}\). Rounding penalties are about 0.000004, 0.000040, and 0.000010 BPB for Uncheatable cap 6, Table-9 cap 6, and Table-9 cap 8. The realized maximum epochs are **5.2062, 5.9978, and 7.5015**, respectively; replace the handoff's stale 5.86 and 7.47 values. The exchange search improves the initially rounded point; it does not guarantee improvement over the continuous optimum. Preserve starts, solver tolerances, feasibility/failure handling, integer sampler counts, and continuous/runtime predictions in the freeze bundle.

These checks use the saved [restart diagnostics](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_kappa_floor_flat15_validation_3e18_20260907/offline_materialization/restart_diagnostics.csv) and [runtime candidate mapping](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_kappa_floor_flat15_validation_3e18_20260907/runtime_materialization/candidate_mapping.csv), rather than a new optimization.

## 2. Draw the validation boundary at procedure selection

The 23 fresh runs cannot all be described as independent validation of the final procedure. The [default sweep](/Users/calvinxu/Projects/Work/Marin/marin/.agents/handoffs/single_phase_freeze_review_handoff_20260907.md:91) uses fresh-run calibration to choose a default, and the proposed joint-fit adoption rule also consults those outcomes. A model can avoid fitting its coefficients to a run while its overall procedure is still selected using that run.

Create a small chronology table: procedure version and freeze time; outcomes available then; decisions informed by those outcomes; later confirmation runs. Some runs may remain prospective for their particular version. Do not retroactively count every run trained after an earlier freeze as independent of the final method.

Recompute the calibration comparison on identical run IDs: final-on-23 versus additive-on-14 is not a paired comparison. Also rename `bias_own_optima`: the [code](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/predict_delphi_fresh_runs_20260907.py:89) filters proposals targeted to that objective, regardless of which model proposed them. “Objective-targeted proposals” is accurate; “the final model's own optima” would imply a different subset.

The final model has not yet been evaluated on the Llama swarms. Define their proportional-anchor construction before examining final-model results, then evaluate the frozen recipe. The [current fallback](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_registry_20260902.py:1578) silently uses a training-median anchor and zero noise margin for a panel without anchors. Running the same registry ID under that fallback would test a different recipe. If Llama results inform another method change, label them development evidence too.

## 3. Revise the two main interpretations

**Report 1.0680 as the final cap-6 Table-9 result.** The four observations in handoff §4.5 come from two models and two caps. Their mean is 1.06625 and sample SD is 0.00334. Substituting that mean for the final method's result improves its apparent performance by 0.00175 through inclusion of other policies. Show all four points as descriptive local evidence, with their identities; do not use their spread as a replicate uncertainty estimate for the final policy.

The conclusion that nearby mixtures are not exact replicates is correct. However, their component differences do not establish excess noise without component-specific uncertainty. The cap-6 difference of 0.0067 is 1.76 proportional-run SDs; for two independent equal-variance runs it would be 1.25 SDs of the difference. Shared seeds introduce unknown covariance. Neither calculation establishes an expected policy difference.

**Uncheatable shows several competitive proposals, not a demonstrated floor or surrogate-independent optimum.** Distinct mixtures reach similar observed aggregate values and trade off component performance. That is interesting evidence of a favorable region. It does not identify the minimum achievable loss or demonstrate practical equivalence. The unbounded proposal measured 0.9890 versus the final 0.9832, so surrogate choice can matter even within this comparison.

Use wording such as: “Several distinct proposals achieved similar aggregate Uncheatable losses at 3e18, with compensating component differences; their predictions differed substantially.” Reconcile §4.4's additive measured range with §4.1: 0.9834 appears there as a prediction, while the listed additive measurements are 0.9841, 0.9835, and 0.9827. The bank best, 0.9811, is an observed value rather than a lower bound; check its cap-6 feasibility before using it as a constrained reference.

The bounded link's 0.987 prediction versus 0.982 measurement demonstrates pessimistic prediction. It does not by itself prove a floor violation; that needs comparison with the actual per-task floors. For the final model, the separate floor check does establish several held-out floor violations. Describe those floors as a conservative modeling restriction, not an estimated attainable loss limit.

Two related evidence claims also need narrowing:

- **Factorial:** the [analysis](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analyze_delphi_frontier_factorial_20260906.py:64) fits 16 coefficients to 16 corners, imports noise from proportional repeats, and uses a resolution-V design. Undetected two-factor interactions, aliased with three-factor interactions, do not establish additivity. Say “no resolved two-factor interactions in these directions under the assumed noise model.” Effects are high-minus-low, not center-to-high. The corner mean, 1.06755, versus replicated center 1.0639 is an existing curvature/batch-consistency check worth examining. The best corner uses **11.78 epochs**, outside caps 6 and 8; its 1.0557 is not a feasible cap-6 regret reference.
- **Top-band ordering:** the [0.92 reference](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/analyze_delphi_top_band_ordering_20260906.py:95) plugs observed differences into a noise model after selection on observed outcomes. It is not an established oracle ceiling. The local kernel uses additional measured bank outcomes, so its 0.79 shows the potential value of local measurements, not superiority under the panel-only training budget.

These are interpretation or reanalysis issues; none requires training just to freeze the procedure.

## 4. Make the comparisons answer the intended claim

**Caps matter empirically.** At cap 4, additive WSPU's Table-9 value is 1.0840 versus Olmix's 1.0769. An unmatched cap-6 comparison reverses this ordering. For a claim about the response family, compare under common constraints. Comparing each complete method under its own chosen cap is also possible, but label it as a full-procedure comparison.

The smallest useful training addition is one common-cap comparison for each objective: either Olmix at headline cap 6 or the final model at cap 4. Prioritize Table 9; there is no need to train both directions and cap 8 immediately. Predeclare Olmix's KL-selection rule. Ridge-tuned Olmix under the same inner CV is a useful offline check; ridge in fitting and KL in mixture optimization are different interventions.

**Keep the scaling ladder attributed to additive WSPU.** Its mixtures are TV 0.17/0.08 from the final ones. Those runs cannot demonstrate scale transfer of the final fitted-floor procedure. The same applies to existing additive learning curves and fit-quality tables: label their method, or recompute the particular panels used to make final-method claims. A full ladder relaunch is unnecessary if the paper keeps this attribution; selected final-policy higher-rung checks are warranted if final-method transfer is central.

**Do not treat TV 0.006 as identical for KL.** It reallocates 0.6% of mixture mass. Reusing the unpenalized measurement is justified only if realized sampler counts and the training recipe coincide exactly. Otherwise, proximity can make the KL run lower priority, but it does not determine its outcome. The three unbounded-model KL results support that tested configuration, not every final-model regularization claim.

## 5. What to finish, and which experiments to prioritize

| Proposed loose end | Recommendation |
|---|---|
| Joint shape/gamma selection | Optional development check. One re-score is alternating refinement, not exhaustive joint selection. Do not require another bank/fresh-run win before freezing. Any adopted change creates a new version and resets its confirmation boundary. |
| Three-SD margin / half-nat cap sensitivity | Useful small offline sensitivity analysis. Predeclare the variants and report them; avoid an open-ended selection cycle. Not a prerequisite if their heuristic status is explicit. |
| One headline cap | Resolve now: cap 6 for both objectives; cap 8 sensitivity. |
| One bank | Resolve before reporting: shared membership, deduplication and source rules. For constrained selection metrics, candidate membership must match across methods within a given cap; membership can differ between caps. Otherwise, explicitly label an unrestricted retrospective-bank analysis. |
| Llama anchors and fits | Define calibration inputs before evaluation. Needed for claims about the final model across all three swarms, not to identify the Qwen candidate. |

Recommended training order after the lock:

1. **Two exact final-policy repeats, one per objective.** Preserve realized mixture counts and recipe; specify what randomness changes. A second trainer seed tests a different source of variation from the existing data-order repeats, which share materialized subsets. These are initial confirmation, not precise confidence intervals. If relative superiority is the endpoint, pair baseline runs under the same seed plan.
2. **A matched-cap comparison, Table 9 first.** Prefer Olmix at cap 6 for both objectives if cap 6 is the headline. The alternative is final-model cap-4 runs against existing Olmix controls. Cap-8 fairness runs can wait.
3. **One selected moderate higher rung per objective, conditional on the paper's scaling claim.** Compare final policies with the existing additive/Olmix measurements. This answers more directly than attributing the old ladder to the new method and costs less than restarting the whole ladder.
4. **Final-model Uncheatable KL 0.05**, if the paper needs that specific regularization conclusion. The ten floor repeats and two older link-optimum repeats are lower priority for this freeze; they address broader uncertainty and mechanism questions.

Priorities 1 and 2 can share controls: a four-run batch containing the two exact final-policy repeats and two Olmix cap-6 runs under the same new seed plan would provide initial confirmation and a paired common-cap comparison. Do not schedule a second, duplicate set of Olmix controls. One paired realization per objective still leaves substantial uncertainty about relative expected performance.

No new training is necessary to correct the description, freeze complete artifacts, compare calibration on common rows, or narrow the unsupported claims. Exact-policy confirmation and a fair constrained comparison are the most useful new experiments for the submission.

## Review snapshot

SHA-256 values identify the working-copy files inspected; these are review provenance, not a new procedure freeze. Source line numbers above refer to this snapshot.

| File | SHA-256 |
|---|---|
| `single_phase_freeze_review_handoff_20260907.md` | `7f54212743c455bc16761c182baaf0fff15686d974a625c0dc9c70a285bbab04` |
| `single_phase_observatory_models_20260902.py` | `7a85c7e28bb1d6d91f5785c2f79eb79c193b15d5c40ec6e0629a8f545973df9a` |
| `single_phase_observatory_registry_20260902.py` | `afc0eb3b5090222651d5b9646d7fff9fa73b894ea8ed85132dc327a159613311` |
| `delphi_floor_anchors_20260907/anchors.csv` | `484b74bcec46290302fdcfbfcde445e40a5dc2bae27e625a2397360548e89247` |
