# Finalizing the single-phase fitting and prediction method

2026-09-07. The objective is to deliver the simplest defensible end-to-end method that preserves the demonstrated performance. The validated `weibull_softplus_unscaled@kappa_floor_link_flat15` procedure is the incumbent for comparison. Freezing is the final step after simplification, validation, and paper alignment. This supersedes the immediate-freeze recommendation in the [previous review](/Users/calvinxu/Projects/Work/Marin/marin/.agents/handoffs/single_phase_freeze_review_feedback_20260907.md); its factual findings remain relevant.

This brief incorporates the completed loose-end sweep and inspection of the new Llama anchor construction. No model source was changed, no new fitting search or training was run, and the running three-swarm harness was not interrupted. One additional read-only diagnostic inspected cap activity in saved final fits.

## 1. Simplify the actual method

Preserve the core supported by existing evidence: materialized-epoch inputs, shared benefit/harm shapes with task/bucket amplitudes, nonnegative fitting, a floored log-deficit link, fixed evaluation aggregation, and constrained mixture optimization. The remaining question is which calibration and search mechanisms earn their complexity.

The completed sweep supports leaving out another alternating shape/gamma refinement. It does not demonstrate that the noise margin or statistical prediction cap is necessary: robustness to nearby values is different from a removal test. The default sweep likewise varied gamma only for fallback tasks; it did not test whether per-task gamma fitting is necessary at all.

Run one bounded round of four simplification candidates, changing each mechanism separately relative to the incumbent:

| Candidate | Exact change | Complexity it could remove |
|---|---|---|
| No statistical response cap | Remove the training-derived half-nat ceiling in both CV prediction and final inference; retain numerical overflow protection. | An extra modeling constant and clipping-induced flattening of high-loss predictions. |
| Fixed gamma 1.5 | Use 1.5 for every task, selecting shape and ridge directly at that value. | Provisional gamma, scalar search, upper-bound trigger, and fallback rule. |
| Fixed gamma 2.5 | The same fixed-gamma control at the current provisional value. | Checks whether the simpler model depends on choosing the existing fallback value. |
| No noise margin | Remove the three-SD term from the floor rule; retain explicit numerical handling of zero/nonpositive deficits. | Dependence on per-component repeat SDs for fitting the response surface. |

These are proposed tests, not established equivalent replacements. In particular, fixed-gamma controls must refit shape/ridge at their fixed value, rather than overwrite gamma after an incumbent fit. Removing the response cap also needs a refit because it affects CV scores. Combine successful deletions only after inspecting the individual results, then evaluate the combined candidate; separate passes do not guarantee that a combination passes.

The response cap is a promising first deletion. In the saved full-panel heads, it binds on 5/408 Uncheatable bank coordinates and 6/247 Table-9 coordinates, and on neither pooled-bank winner. No saved training prediction is capped. For fixed heads, uncapping only increases predictions elsewhere, so those finite-bank winners remain optimal. This does not prove preservation of the refitted model or continuously optimized policy. The diagnostic compared each saved prediction with its ceiling, computed as \(\phi+\exp(\min(30,\max\log\max(y-\phi,10^{-9})+0.5))\), using the saved training indices and floors.

Do not add a new interaction family, ensemble, or bank-label-trained correction to the final path without a specific failure that requires it. Keep reliability analysis as a diagnostic when it does not change the retained objective. A no-op screen should not appear to be a necessary selection stage.

## 2. Repair the calibration protocol before judging cross-swarm performance

The new [Llama anchor builder](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/build_llama_floor_anchors_20260907.py:60) uses the panel's proportional observation as a fixed anchor for all 58 components of `60m_39bucket` (Llama 160M/1.2B) and the seven Uncheatable components of `300m_39bucket` (Llama 200M/6B). The same coordinate is held out in outer and inner CV.

This is confirmed in the current [split manifest](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_observatory_final_model_20260907/split_manifest.csv:16): proportional row 14 is scored in the 60M panel's outer fold 0, and proportional row 0 in the 300M panel's outer fold 1. Completed shards contain 65 component predictions for which the scored held-out outcome exactly equals the globally supplied anchor. The coordinate also appears in inner validation in the other outer folds. Those predictions are not out of sample with respect to the supplied calibration data.

Prefer an explicit calibration set: proportional calibration coordinates remain available to fitting and are excluded from both levels of validation; partition the other mixture coordinates, and give comparison methods the same calibration information. An alternative is a genuinely fold-local anchor estimator that uses only training measurements. Define the intended deployment input contract first; do not choose between these protocols by which scores better. Merely dropping contaminated outer predictions afterward does not repair inner selection.

The builder also copies aggregate repeat SD to every component in these three panel/target combinations. Aggregate variability is not a measured component SD. Recover per-component evaluations from existing repeat checkpoints or outputs where possible. If that cannot be done, explicitly specify and evaluate the approximation; do not claim identical per-component noise calibration across swarms. The no-noise-margin candidate may remove this fitting dependency if it passes the performance gate.

The 300M Table-9 anchor uses eleven reference rows. Its reference baseline differs numerically from the scored panel baseline; reconcile run identity before claiming either direct overlap or independence. The running harness can finish and remain useful diagnostically, but its current results do not close these protocol issues.

## 3. Preserve performance at the level that matters

Use the current cap-6 selected policies as the reference: measured Uncheatable 0.9832 and Table-9 1.0680 BPB. These are observed runs, not estimates precise enough to guarantee an expected-loss bound.

For a behavior-preserving cleanup, require prediction parity at the existing \(10^{-8}\)-BPB reconstruction tolerance and exact realized sampler-count equality. Preserve evaluation weights, training recipe and relevant seeds. Such changes can retain the existing measured policy results without new LM training.

For a methodological change, evaluate three distinct outputs on common rows and splits:

1. Selection: selected bank coordinate, regret and shortlist regret, with source-specific results and paired source contrasts.
2. Prediction: calibration at the same measured proposals, predictive fit metrics, and component failures. Improved whole-bank RMSE cannot compensate silently for degraded optimum selection.
3. Actual recommendation: run the same optimizer and rounding procedure; compare exact realized counts and predictions for both headline objectives. An unchanged bank pick does not establish an unchanged optimizer result.

For the first simplification screen, use a strict acceptance gate: no increase in constrained-bank regret at one, objective-targeted proposal RMSE, or absolute objective-targeted mean bias for either objective, allowing only the existing numerical tolerance. Source contrasts, shortlist regret, whole-bank fit and component errors are diagnostics; inspect them for regressions hidden by aggregation. Better whole-bank fit cannot rescue a failed primary gate. Among candidates that pass, prefer fewer fitted quantities and conditional rules, then lower fitting cost. These are development gates, not statistical guarantees of future performance.

The [loose-end table](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_loose_ends_sweep_20260907/comparison.csv:7) itself changes the joint-fit Uncheatable bank pick from rank 4 to 3. Correct the statement that no variant changes the pick. The sweep does not provide materialized-policy equality evidence.

If a simpler method produces the exact same realized policies, their measured performance can be retained, while the new fitting/calibration claims still need their own evaluation. If policies change, validate them against the incumbent before replacing reported results. Predeclare paired runs, sample size and a confidence-bound rule. Treat the allowed regression margin as zero unless a practical positive margin is explicitly agreed before confirmation; proportional-run SD is not that margin, and a nonsignificant difference is not equivalence. Until such evidence exists, preserve the incumbent's measured result and label the changed policy a candidate.

## 4. Finish the method as one coherent deliverable

Adopt cap 6 for both headline objectives, cap 8 as sensitivity, and one canonical bank. Candidate membership must match across methods within a cap; it can differ between caps. Preserve source identities and distinguish unrestricted bank selection from constrained optimization.

The final package should contain one active fitting path, serialized prediction parameters, one optimization/materialization path, and a complete input/source/environment manifest including anchors. Keep historical variants available for reproducing research; remove their branching and defaults from the final method's interface. Verify serialization parity against the chosen final implementation. Require parity with the incumbent only when claiming a behavior-preserving cleanup, and preserve its reproduction path before deleting any shared implementation.

Then rerun the retained mechanism ablations around the actual final method, particularly no-harm and pool-size controls. A new final-method row alongside additive-model ablations does not establish the final method's mechanism. Align the remaining fit-quality, selection, calibration and learning-curve panels with their method identities. Existing additive scaling runs retain their attribution; changed final policies need selected higher-rung validation if the paper claims their transfer.

Completion means: one explicit calibration protocol, one concise fitting/selection rule, one tested final implementation, retained or revalidated headline policies, and paper equations/tables/captions that describe that same method. Sensitivity checks and provenance packaging support this deliverable; they do not substitute for the simplification round.
