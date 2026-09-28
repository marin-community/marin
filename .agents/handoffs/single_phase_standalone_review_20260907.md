# Review of the standalone mixture-selection implementation

Reviewed 2026-09-07: `/Users/calvinxu/Projects/Work/Marin/mixture-selection`, commit `b1707f7c1f62305b733a17870e437e7aad0e5165`. The checkout was clean. This review covers the bundled default fit, its three untrained 3e18 policies, paper fidelity, calibration folds, and Olmix reproduction. No source files or training state were changed; verification included the supplied full self-test and bounded numerical diagnostics.

## Assessment

**The bundled default implementation is suitable for validating its proposed mixtures.** I found no default-path implementation failure. Offline selection and prediction checks support testing the policies, but do not estimate the probability or severity of a real performance regression. Extraction fidelity is verified: the full self-test passes, with maximum task-prediction discrepancies of approximately \(7.1\times10^{-15}\) on Uncheatable and \(2.1\times10^{-13}\) on Table 9, and exact realized-count parity for all three policies.

**Measured nonregression is still unknown.** The protocol correction changes the policies from those already trained. The model family and fitting procedure are defensible as an empirically regularized surrogate, but several current explanations overclaim what the algorithm establishes. The artifact also needs two reusable-API fixes and clearer baseline/data provenance before becoming the authoritative paper release.

## 1. Evidence concerning regression of the new policies

The cap deletion and protocol correction should be evaluated separately. Under the corrected folds, capped and uncapped models produce identical realized count vectors and identical predictions on the 23 existing validation runs. The policy movement comes from the changed fitting protocol. The [corrected screen](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_corrected_screen_20260908/comparison.csv:7) retains the previous optima-bank picks: Uncheatable rank 4/170, regret 0.0016214; Table 9 rank 12/157, regret 0.0150784. Removing the cap improves the corresponding bank RMSE from 0.012535 to 0.009857 and from 0.020232 to 0.019730.

The populations used in this review are distinct subsets of the same retrospective development bank:

| Population | Uncheatable / Table-9 coordinates | Filter |
|---|---:|---|
| Entire shipped bank | 408 / 247 | All sources, no epoch-cap filter; the standalone CLI default |
| Source-defined “optima” stratum | 170 / 157 | Excludes the dose-response and archived-baseline intervention sources; no epoch-cap filter |
| Cap-6-feasible bank | 231 / 134 | All sources, retaining coordinates whose maximum materialized epoch is at most six |

The source and feasibility filters do not define the same population. The historical optima stratum is defined by [the scorer](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/score_delphi_selection_20260906.py:239); it is not restricted to proposals optimized for the objective being scored. None of these subsets contains a measurement of the new policies.

On the same 23 existing measured runs, old-flat15 versus corrected-uncapped RMSE changes from 0.005608 to 0.005117 for Uncheatable and from 0.006503 to 0.006037 for Table 9. Objective-targeted mean bias improves from +0.001306 to +0.000852 and +0.004417 to +0.003875. Uncheatable Spearman declines from 0.9644 to 0.9545, so the result is not improvement on every diagnostic. These are development-used observations; they do not measure the new proposals. See the [corrected calibration results](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_fresh_run_calibration_corrected_20260908/summary.csv).

For a more direct check, I reconstructed the old and corrected heads at their saved shape/ridge/gamma values, verified panel/bank parity within \(1.3\times10^{-15}\), and cross-scored the old and new policies:

| Policy | Old fit: old → new mixture | Corrected fit: old → new mixture | Old mixture measured |
|---|---:|---:|---:|
| Uncheatable, cap 6 | 0.9807317 → 0.9808329 | 0.9811196 → 0.9810315 | 0.9832098 |
| Table 9, cap 6 | 1.0634798 → 1.0636276 | 1.0639627 → 1.0638310 | 1.0679847 |
| Table 9, cap 8 | 1.0626296 → 1.0627834 | 1.0632272 → 1.0630767 | 1.0685273 |

The old fit predicts a small worsening, about 0.00010–0.00015 BPB; the corrected fit predicts a small improvement, about 0.00009–0.00015. This is reassuring evidence of locally flat predictions, with disagreement in sign. The two closely related models are not independent confirmations. In particular, comparing new predicted 1.063831 with old measured 1.067985 would manufacture an apparent 0.00415 improvement: the corrected model's predicted improvement is only 0.000132.

There are concrete reasons to retain uncertainty:

- Uncheatable reallocates 78 of 2,048 sampler blocks, TV 0.038086, across 23 buckets. Table-9 caps 6/8 reallocate 52/55 blocks, TV 0.025391/0.026855, across 26/28 buckets. These are distinct training policies. Uncheatable adds 3.223 percentage points of Dolmino CC-HQ and removes 1.367 points of high-quality literature; Table 9 adds 1.5625 points of synthetic QA.
- Nearby measured Table-9 policies do not resolve the predicted changes: neighbours at TV 0.02539 and 0.02686 from the new cap-6 policy measured 1.067985 and 1.061277. Their 0.006707 difference is far larger than the modeled policy difference. They are not replicates.
- No frozen-bank coordinate is within TV 0.05 of a new policy. Nearest distances are 0.15381 for Uncheatable and 0.08447/0.08838 for Table 9. Distance from the training-mixture convex hull increases modestly to 0.44949, 0.32869 and 0.33915. See the [policy summary](/Users/calvinxu/Projects/Work/Marin/mixture-selection/data/reference_policy_summary.csv:2).

**Recommendation:** proceed to exact-policy validation, retaining the old flat15 runs as the incumbent comparison. The primary endpoints are each cap-6 policy's own target aggregate BPB, reported separately. Match trainer seed, data seed, materialized subsets and recipe; count an existing incumbent run as paired only when those match. Cap 8 is secondary: report its measured change relative to corrected cap 6 as an observed sensitivity result, not proof of equivalence from one run.

Initial validation should report observed candidate-minus-incumbent losses and label uncertain comparisons inconclusive. For the stated goal of demonstrating no expected regression, predeclare a paired confidence-bound rule and sample size: the upper bound on that difference must be at most the allowed loss margin for each objective. Treat that margin as zero unless a practical positive margin is explicitly chosen before confirmation. One new realization per policy generally cannot establish that claim. Do not attribute the old measured values to the new implementation or repeatedly add seeds until a favorable result appears.

## 2. The core method is defensible; its current explanations need correction

The extraction retains one response family, nonnegative amplitudes, fixed evaluation aggregation, a finite shape/ridge search, a bounded floor search, and a deterministic optimization/materialization recipe. Centering, QR reduction, and numerical exponential guards serve implementation purposes; they are not unnecessary model variants. The complete fitted heads now serialize, so prediction can be independent of refitting. Keeping the three-SD margin after its ablation showed stable aggregate behavior is defensible. It should not be called computationally inert: it binds in six of the 51 final Table-9 heads.

Before submission, correct the following claims together in Methods, Appendix A.6, README, and the source overview:

1. **Head fitting and hyperparameter selection use different losses.** NNLS fits log-deficit targets; both CV stages score predictions in original BPB. This is a coherent staged procedure. The [source overview](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:21) incorrectly says log-deficit-space scoring; the actual [CV calculation](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:313) is unambiguous. State that shape and ridge are selected at provisional gamma 2.5, then both are held fixed during gamma selection.
2. **The default detects a boundary, not flatness.** [The code](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:410) substitutes 1.5 whenever the search returns gamma at least \(6^{0.9}\approx5.02\). It does not test curvature, uncertainty, or whether 1.5 has indistinguishable CV error. Describe it as a conservative boundary-triggered rule supported by development evidence. Neither the bounded scalar search nor the fallback certifies the global CV optimum.
3. **Describe the actual bounds and their empirical origin.** Gamma is in \([1,6]\), so log gamma is in \([0,\log6]\). The paper's assertion that six is the largest observed held-out improvement ratio conflicts with the [handoff's ratios reaching 8.2](/Users/calvinxu/Projects/Work/Marin/marin/.agents/handoffs/single_phase_freeze_review_handoff_20260907.md:103). Do not manufacture a theoretical or maximum-observed rationale for a development-selected bound.
4. **Floors restrict predictions; they are not established achievable-loss limits.** Some measured component outcomes fall below them. The method can still be useful, but its justification is regularization and tested selection/calibration behavior. The best observed swarm run is also a noisy extreme, not automatically one of the most reliable measurements.
5. **Remove the old half-nat cap from the main procedure description.** [Methods line 48](</Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/sections/methods.tex:48>) and [Appendix line 200](</Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/sections/appendix.tex:200>) still describe it. Document the calibration-pinning protocol, including that OOF scoring excludes the calibration coordinate. The current main text does not state that restriction.
6. **The response family can express turnover, but positive amplitudes do not guarantee an initial decrease.** For allowed power one, \(\eta'(0)=-\alpha\rho+2\beta\operatorname{softplus}(-\tau)\operatorname{sigmoid}(-\tau)\). With alpha=beta=1, rate=0.05 and threshold=1, this is about +0.1185. Correct the universal claim in [Methods line 39](</Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/sections/methods.tex:39>). Likewise, epoch caps do not establish joint swarm support, and SLSQP returns the best endpoint found rather than a certified global optimum.

These corrections do not require changing the default model or rerunning the simplification search. They make the paper's claims match the empirical method. Intense review will also require keeping the bank's development role explicit and distinguishing the final method's ablations/validation from older additive-model evidence.

## 3. Calibration is correct for the shipped data; two API paths need repair

**Ship the exact fold table.** For reproducibility it is preferable to requiring readers to reconstruct k-means assignments. I verified that the final validation folds contain 117, 40 and 122 observations, cover all 279 noncalibration rows, and exclude the proportional row. All 25 supplied outer-fold records keep that row in training with inner label −1. Thus the earlier calibration-label leak is fixed for the bundled data.

**[P1 for reuse] Supplied fold tables bypass the invariant.** [final_inner_folds](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:222) checks row coverage but never checks or pins calibration when a table is supplied. An in-memory stale table placing the proportional row in validation was accepted. Validate labels, nonempty folds, disjoint train/validation membership, and calibration exclusion. Bind the table to run identities and input hashes, not only integer row positions. Missing explicit split files should fail rather than silently trigger a different fold construction.

**[P1 for reuse] Subset fitting and fold indices disagree.** [fit_task](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:427) slices data to `train`, while [folds_from_labels](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:215) produces indices in original-row space. Passing `train=[1,2,3,5]` with helper-generated folds raises an out-of-bounds error before fitting; other combinations can address the wrong observations. Establish and enforce one index convention. This matters for reproducing outer-CV and subset/learning-curve fits.

Neither issue affects the shipped all-row fit tested here. Both should be repaired before presenting the module as a reusable authoritative fitting API. If general subset fitting is outside scope, remove that misleading interface and say that outer-CV figures are reproduced by a separate, specified driver.

## 4. Olmix: correct proposal mathematics, approximate fitted baseline

For fixed fitted laws, the convex objective in [olmix_exact_proposal](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:737) agrees with upstream. A check using saved laws produced solver-to-solver differences around \(10^{-5}\) TV, with feasible optimal solutions. That is reassuring proposer-level agreement.

**[P1 for comparative claims] The complete baseline is not an exact upstream replay.** The [standalone law fitter](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:701) uses SciPy, 48 starts and a different initialization/precision regime. The inspected upstream wrapper uses 300 starts and Torch fitting. It also sets delta=0.02 and max_step=100, overriding lower-level defaults of 0.01/20. No corresponding override was found in the saved reproduction YAML/wrapper, so the night report's stated upstream recipe needs reconciliation. These are implementation facts at the pinned upstream commit; an undocumented runtime override remains possible. See [upstream fitting wrapper](https://github.com/allenai/olmix/blob/9586977981e01b60c8b20330f623047f0b693fe3/olmix/fit/utils.py) and [lower-level fitting](https://github.com/allenai/olmix/blob/9586977981e01b60c8b20330f623047f0b693fe3/olmix/fit/law.py).

The verified SciPy/upstream differences of TV 0.05643 for Table 9 and 0.12155 for Uncheatable are not policy parity or measured equivalence, and cannot be attributed solely to the optimizer without matching the whole fitting recipe. Call this an independent Olmix-law implementation with an exact convex proposer. For a strong claim against upstream Olmix, use pinned upstream fitted laws or establish a matched-input/recipe comparison and validate the materially different baseline proposals. This evidence does not establish which implementation performs better.

The [saved reproduction comparison](/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_olmix_reproduction_20260907/summary.json) also shows that the current-input SciPy proposals are TV 0.28867/0.30626 from the historical trained comparators. The bundled panel therefore cannot regenerate those historical policies. The original June Table-9 panel is reported unavailable; state that limit rather than describing an approximate rerun as exact reproduction. Finally, retain common-cap comparisons when isolating the response-family advantage: an Olmix cap-4 default is not matched to the headline cap-6 policy.

## 5. Release requirements and the two switches

**Keep `fit --cap-margin` and `optimize --kl`.** Both have legitimate paper-reproduction uses and their defaults identify the proposed main method. Record their values and the recipe identity in output metadata. Their existence does not make the main procedure indefensible.

However, `--cap-margin 0.5` alone does not reproduce the historical validated fits: the default fold table now uses the corrected protocol. Historical reproduction also requires the old inputs, anchors, folds and fitting recipe. The [README opening](/Users/calvinxu/Projects/Work/Marin/mixture-selection/README.md:5) and [source opening](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:7) currently imply validated-policy parity, while the bundled policy summary correctly says `unknown_not_run`. Resolve this identity conflict before sharing the artifact.

The remaining release work is concrete:

- Add a manifest linking the exported data, folds, anchor run identities/counts, source revision, complete fitted predictors, policy counts and measured runs. Reconcile the README's eleven-run anchor claim with the original Table-9 ten-repeat construction. Save an environment lock, including the optional Olmix solver dependencies.
- Include ready-to-load fitted JSONs and serialized Olmix laws. The WSPU serializer is complete, but the bundled task-fit CSV omits coefficients/intercepts and the Olmix command discards its fitted laws. Save optimizer diagnostics and cap/KL/materialization settings beside every policy.
- Make the bank evaluation population explicit. [evaluate](/Users/calvinxu/Projects/Work/Marin/mixture-selection/mixture_selection.py:846) scores the unrestricted bank and does not reproduce source-stratified, optima-only, or cap-constrained tables. With cap-6 filtering, the shipped bank contains 231 Uncheatable coordinates (rank 2, regret 0.0004033) and 134 Table-9 coordinates (rank 1, regret zero). Old and corrected fits select the same feasible coordinates, so this supports stability rather than improvement. Its unrestricted Table-9 pick reaches 6.99 epochs, above the headline cap. These are different retrospective questions; expose their labels/filters and do not treat the constrained bank winner as the newly optimized policy.
- Provide executable commands for the retained OOF/ablation metrics, or clearly delimit the artifact to fitting/proposal reproduction and name the versioned evaluation driver. The current `evaluate` command alone does not regenerate the paper's evidence tables.

Before training, identify the exact intended standalone outputs and retain their current hashes. Fixes limited to documentation, split validation and unsupported subset handling need not change those policies; rerun parity if shared numerical code changes. Further model searching is not required by this review. Before publication, close the method-description, baseline-faithfulness and artifact-identity issues above, and attach measured results to the corrected procedure's own policies.

## Verification record

The supplied self-test was run in an isolated uv environment with numpy 2.3.5, pandas 2.2.2 and scipy 1.17.0. It refitted both objectives and checked all three policy count vectors successfully. Additional diagnostics checked supplied folds, API index behavior, saved-head cross-predictions, policy/count differences, bank constraints and fixed-law Olmix proposal agreement. No LM training or cloud jobs were launched.

Source SHA-256: `mixture_selection.py` = `4094d8f3334dcc1e544809fc5132fcb6ea0400938884912e118c09ecfb9e9c9d`; `data/splits.csv` = `90efb83c80e26f1b954c70fdace40eb4c25713bf03b5abf3d45024ab70263ee3`; `data/reference_policies.csv` = `6379fa0aecfd26aa6e6334ca2604d6dd518024d64776d692e996fce06cdf534f`.
