# DSP Synthetic Recovery: Research Logbook

## Scope
- Goal: Improve or validate the current DSP fitting/optimizer procedure by testing whether it can recover known DSP-generated responses at production-swarm scale.
- Primary metrics: heldout prediction quality, candidate-rank recovery, predicted-optimum regret under the synthetic ground truth, solver stability across seeds/starts, and boundary-hit diagnostics.
- Constraints: Keep the procedure close enough to the current real-data DSP workflow that any improvement remains usable on actual swarm data. Avoid solver complexity that only works in synthetic settings.
- Fieldbook: `exp_01ktkb2yackps56nyagp4bcmh3` (`DSP synthetic recovery solver diagnostics`).

## Baseline
- Date: 2026-06-08.
- Code refs:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_300m.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/design_production_swarm_167p.py`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/production_swarm_mixture_design_167p_20260523_collaborator_packet_proplogit_exp_tau20_lam0p25/production_swarm_167p_candidate_mixtures.csv`
- Baseline numbers:
  - Production partition count: 167.
  - Production candidate packet rows: 1501.
  - D-optimal rows: 1200.
  - Full penalty DSP parameter count at 167 partitions: 670 total fitted parameters, with roughly 335 nonlinear optimizer dimensions and the remaining linear head solved internally.

## Experiment Log

### 2026-06-08 10:05 - Kickoff
- Hypothesis: A synthetic recovery benchmark can distinguish optimizer/identifiability failures from real-data model misspecification. If the current DSP solver cannot recover rankings and low-regret optima when DSP is the data-generating process, solver improvements are justified before interpreting real-data failures.
- Command: `fieldbook experiment create --json --name "DSP synthetic recovery solver diagnostics" ...`
- Config: Use actual 167-partition production candidate matrices, sweep sample sizes around the full penalty DSP parameter count, and test multiple noise regimes.
- Result: Fieldbook experiment created and initial logbook started.
- Interpretation: The first iteration should produce a baseline recovery report before changing the solver.
- Next action: Implement a self-contained synthetic recovery script that reuses the existing DSP functional form and current production candidates, then request CC review on the benchmark design and first baseline results.

### 2026-06-08 10:22 - First synthetic recovery benchmark
- Hypothesis: At production scale, the current profiled nonlinear DSP solver may be overkill or unstable because the full penalty form has 670 reported parameters and roughly 335 nonlinear optimizer dimensions for 167 partitions.
- Command:
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile current --sample-sizes 335 --noise-sigmas 0 --seeds 0 --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_current_failure_n335`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --sample-sizes 335,670,1200,1501 --noise-sigmas 0,0.05,0.15 --seeds 0,1,2 --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_coarse_grid`
- Config: Synthetic truth was `dsp_effective_exposure_penalty_nnls` on the 1501-row production candidate packet. Targets were standardized lower-is-better values. The `current` profile matched the existing high-dimensional nonlinear refinement path. The `coarse_only_nnls20` profile used the same DSP start bank and NNLS linear head but skipped nonlinear refinement and raised NNLS max iterations to `20 * n_features`.
- Result:
  - Exact current profile at `n=335` failed immediately with `RuntimeError: Maximum number of iterations reached` inside SciPy NNLS.
  - `current_nnls20` and signed-ridge nonlinear-refinement smoke trials avoided immediate failure but were too slow for local iterative sweeps, taking more than 1-2 minutes without completing.
  - `coarse_only_nnls20` completed the full 36-trial grid quickly. Overall Spearman by sample size: `n=335 mean 0.847 min 0.654`; `n=670 mean 0.967 min 0.936`; `n=1200 mean 0.983 min 0.971`; `n=1501 mean 0.986 min 0.976`.
  - Mean predicted-best regret by sample size: `n=335 0.523`; `n=670 0.024`; `n=1200 0.010`; `n=1501 0.015`. Regret is on standardized synthetic target units.
- Interpretation: The immediate issue is not that DSP cannot be recovered under its own truth. A much simpler solver path can recover rankings and low-regret candidates once `n` is near or above the reported parameter count. The high-dimensional nonlinear refinement path is currently too slow/fragile at 167 partitions to be the default production-scale fitting procedure without stronger evidence. The weak point is `n=335` with moderate noise, where coarse-only can select a bad optimum despite decent average rank recovery.
- Next action: Get CC review on whether the principled next step should be (a) coarse-only as a robust production-scale baseline, (b) a low-dimensional nonlinear polish around coarse starts, or (c) a regularized nonlinear fit with shared/family shape parameters.

### 2026-06-08 10:35 - CC review and corrected off-grid benchmark
- Hypothesis: The first coarse-only result may be biased if the synthetic truth distribution is too close to the solver start-bank prior.
- Command:
  - `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-8 --effort max -p '<scoped review prompt>'`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --sample-sizes 335,670,1200,1501 --noise-sigmas 0.05,0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime positive --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_offgrid_coarse_grid`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime mixed --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_offgrid_mixed_sign`
- Config: Added `truth_regime=off_grid`, which increases independent per-domain perturbations for `rho` and `tau` and changes the phase multiplier distribution. Added `coefficient_regime=mixed` as a diagnostic for the NNLS sign constraint. Added incremental CSV writes so interrupted long solver probes preserve completed trials.
- Result:
  - CC flagged a blocker in the first benchmark: `start_bank_prior` truth was generated from the same median-exposure and percentile heuristics as `_start_bank`, so coarse-only had nonlinear shape information for free.
  - Off-grid positive-coefficient coarse-only remains strong but not perfect. At `n=670, sigma=0.15`: mean Spearman `0.949`, min `0.847`, mean regret `0.062`, max regret `0.144`. At `n=1200, sigma=0.15`: mean Spearman `0.967`, min `0.874`, mean regret `0.079`, max regret `0.284`.
  - At `n=335`, off-grid positive coefficients remain unreliable: for `sigma=0.15`, mean regret `0.497`, max regret `2.877`.
  - Mixed-sign truth changes the linear-head conclusion. At `n=670, sigma=0.15`, coarse NNLS mean regret `0.306` while signed ridge mean regret `0.088`. At `n=1200, sigma=0.15`, coarse NNLS mean regret `0.438` while signed ridge mean regret `0.081`.
  - Light high-dimensional nonlinear polish (`start_top_k=2`, `maxiter=12`) was still too slow for interactive iteration at 167 partitions and was stopped after more than two minutes on one trial.
- Interpretation: The main robust finding is now more cautious and more useful: full high-dimensional nonlinear refinement is too slow locally; coarse-only with NNLS is a strong, simple baseline when the NNLS sign assumption is correct and \(n \gtrsim 2p_\text{linear}\); signed ridge is the right diagnostic/control when the target may require mixed signs. We should not adopt coarse-only purely from start-bank-prior truth. The next principled variant is not more global nonlinear optimization, but a bounded family of cheap linear-head choices under fixed/coarse nonlinear shapes: NNLS, signed ridge, and ridge/shrinkage with validation.
- Next action: Ask CC to review the corrected evidence and choose the next minimal variant: likely cross-validated ridge/elastic signed head or NNLS with stronger regularization, while leaving high-dimensional nonlinear polishing as a non-default optional diagnostic.

### 2026-06-08 10:45 - Ridge-head iteration
- Hypothesis: If the linear head is the main lever, a signed ridge head with simple shrinkage may be a better production-scale solver candidate than NNLS or high-dimensional nonlinear refinement.
- Command:
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_signed_ridge --solver-profile coarse_only_signed_ridge_cv --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime mixed --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_mixed_sign_ridge_cv`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge_cv --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime positive --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_positive_nnls_vs_ridge_cv`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge_alpha1 --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime positive --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_positive_alpha1`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_signed_ridge --solver-profile coarse_only_signed_ridge_alpha1 --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime mixed --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_mixed_alpha1`
- Config: Added `coarse_only_signed_ridge_cv`, `coarse_only_signed_ridge_cv_cap1`, and `coarse_only_signed_ridge_alpha1`. CV chooses ridge strength by five-fold MSE over fixed coarse DSP features. Alpha1 is a simpler fixed shrinkage profile.
- Result:
  - Mixed-sign truth: CV ridge improved `n=670` mean regret from fixed signed ridge `0.088` to `0.027`, but worsened `n=1200` mean regret from `0.081` to `0.123`. Capping CV lambdas at 1 reversed part of that tradeoff: `n=1200` mean regret `0.081`, but `n=670` mean regret `0.098`.
  - Positive-coefficient truth: CV ridge improved over NNLS in this synthetic regime. At `n=670`, NNLS mean regret `0.062`; CV ridge mean regret `0.017`; fixed alpha1 mean regret `0.037`. At `n=1200`, NNLS mean regret `0.079`; CV ridge mean regret `0.019`; fixed alpha1 mean regret `0.044`.
  - Fixed alpha1 is simple and fast. On mixed-sign truth it matched fixed signed ridge in this benchmark; on positive truth it improved over NNLS but not as much as full CV ridge.
- Interpretation: The synthetic winner is not high-dimensional nonlinear refinement. The most promising family is fixed/coarse nonlinear features plus a signed ridge linear head. However, CV ridge has a real tradeoff and is more complex; fixed alpha1 is simpler and more stable but may leave performance on the table. This remains a synthetic result only.
- Next action: Get CC review on whether fixed alpha1 is enough to carry forward, and run real-data cross-validation before changing the actual DSP fitter.

### 2026-06-08 11:55 - Corrected alpha1 bug and real-data gate
- Hypothesis: A fixed signed ridge head may be a safer production-scale DSP fitter if it preserves top-candidate selection under synthetic truth and improves held-out real-data ranking/regret.
- Command:
  - `uv run python -m py_compile experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py experiments/domain_phase_mix/exploratory/two_phase_many/compare_dsp_solver_profiles_real_300m.py`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge_alpha1 --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime positive --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_positive_alpha1_fixed`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_signed_ridge --solver-profile coarse_only_signed_ridge_alpha1 --sample-sizes 670,1200 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime mixed --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_mixed_alpha1_fixed`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/compare_dsp_solver_profiles_real_300m.py --profiles current,coarse_only_nnls20,coarse_only_signed_ridge,coarse_only_signed_ridge_alpha1 --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_solver_real_cv_300m_20260608_signed_compare`
- Config: Fixed a benchmark-script bug where `coarse_only_signed_ridge_alpha1` returned through the default `_fit_linear_head` path before applying `fixed_ridge_alpha=1.0`. Corrected artifacts record `linear_reg=1.0`. The earlier alpha1 artifacts from `dsp_synthetic_recovery_167p_20260608_positive_alpha1` and `dsp_synthetic_recovery_167p_20260608_mixed_alpha1` are superseded.
- Result:
  - Corrected off-grid positive synthetic, `sigma=0.15`: alpha1 improved over coarse NNLS. At `n=670`, alpha1 mean regret `0.029`, max `0.139`; NNLS mean regret `0.062`, max `0.144`. At `n=1200`, alpha1 mean regret `0.025`, max `0.111`; NNLS mean regret `0.079`, max `0.284`.
  - Corrected off-grid mixed synthetic, `sigma=0.15`: alpha1 did not improve over the tiny-ridge signed head. At `n=670`, alpha1 mean regret `0.098`, max `0.417`; tiny-ridge mean regret `0.088`, max `0.417`. At `n=1200`, both had mean regret `0.081` and max `0.300`.
  - Real 300M CV on `eval/uncheatable_eval/bpb`, `rows=100`, `5` folds: current mean heldout regret `0.056`, max `0.119`; coarse NNLS mean `0.053`, max `0.126`; tiny-ridge signed head mean `0.046`, max `0.201`; alpha1 mean `0.082`, max `0.239`.
  - Real 300M heldout Spearman remains weak across all profiles: current mean `0.021`, coarse NNLS `0.002`, tiny-ridge signed head `0.128`, alpha1 `-0.008`. Alpha1 has the lowest heldout RMSE (`0.074`) but worse top-candidate regret, which makes RMSE the wrong adoption criterion for mixture optimization.
- Interpretation: Synthetic recovery is useful as a solver diagnostic, but fixed alpha1 should not be adopted as the production default from current evidence. It helps positive synthetic truth, does not help mixed synthetic truth, and loses the real-data regret gate. The tiny-ridge signed head is interesting as a comparator because it improves mean real-CV regret, but its worst fold is worse than current and coarse NNLS; that is not stable enough to replace the current fitter without a stronger rank/regret-tuned validation procedure.
- Next action: Ask CC to review the corrected evidence. Likely decision is to keep the current production fitter for now, keep coarse NNLS and signed-ridge heads as fast diagnostics, and only revisit a solver change if real-data CV across more metrics shows lower worst-fold regret.

### 2026-06-08 12:18 - Low-dimensional shared-head diagnostic
- Hypothesis: The optimizer-facing failure may be caused by fitting too many linear-head coefficients relative to available observations. A shared benefit/penalty head with two coefficients could trade bias for much lower variance.
- Command:
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge --solver-profile coarse_only_shared_nnls20 --solver-profile coarse_only_shared_signed_ridge --solver-profile coarse_only_shared_signed_ridge_alpha1 --sample-sizes 80,100,670 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime positive --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_shared_positive`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_dsp_synthetic_recovery_167p.py --solver-profile coarse_only_nnls20 --solver-profile coarse_only_signed_ridge --solver-profile coarse_only_shared_nnls20 --solver-profile coarse_only_shared_signed_ridge --solver-profile coarse_only_shared_signed_ridge_alpha1 --sample-sizes 80,100,670 --noise-sigmas 0.15 --seeds 0,1,2,3,4,5,6,7,8,9 --truth-regime off_grid --coefficient-regime mixed --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_synthetic_recovery_167p_20260608_shared_mixed`
  - `uv run python experiments/domain_phase_mix/exploratory/two_phase_many/compare_dsp_solver_profiles_real_300m.py --profiles current,coarse_only_nnls20,coarse_only_signed_ridge,coarse_only_shared_nnls20,coarse_only_shared_signed_ridge,coarse_only_shared_signed_ridge_alpha1 --output-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_solver_real_cv_300m_20260608_shared_head`
- Config: Added `LinearHeadMode.SHARED`, which aggregates the DSP benefit features into one shared benefit column and penalty features into one shared penalty column, fits a two-coefficient head, then expands the two coefficients back over domains so the existing `_predict` path remains unchanged. The real 300M packet has 39 domains, so the normal domain head has 78 linear coefficients; the 334-coefficient count applies to the 167-partition production packet, not this 300M real-data CV.
- Result:
  - Synthetic positive truth, `sigma=0.15`: shared heads are more stable at small `n` but biased at large `n`. At `n=80`, shared signed mean regret `0.341`, max `0.763`; domain signed mean regret `0.821`, max `2.633`. At `n=670`, shared signed mean regret `0.284`; domain signed mean regret `0.037`.
  - Synthetic mixed truth, `sigma=0.15`: shared heads are too biased. At `n=100`, shared signed mean regret `1.967`; domain signed mean regret `0.928`. At `n=670`, shared signed mean regret `1.812`; domain signed mean regret `0.088`.
  - Real 300M CV on `eval/uncheatable_eval/bpb`: shared signed alpha1 has mean regret `0.044`, max `0.097`; current has mean `0.056`, max `0.119`; coarse NNLS has mean `0.053`, max `0.126`; domain signed has mean `0.046`, max `0.201`. Shared NNLS is not useful; it produced constant predictions in several folds and has mean regret `0.073`.
  - Real 300M heldout Spearman remains weak: shared signed alpha1 mean `0.050`, current `0.021`, domain signed `0.128`, coarse NNLS `0.002`. The shared-head win is therefore mostly a regret-stability result, not strong evidence of reliable rank modeling.
- Interpretation: A two-coefficient shared head directly addresses small-sample variance and gives a better one-metric real-CV regret profile, but synthetic mixed-sign truth shows the bias is severe when domain-specific directions matter. This is a useful diagnostic and possible regularized fallback, not enough evidence for a production DSP fitter change.
- Next action: Ask CC for a follow-up review. The likely next rigorous step, if we keep iterating, is a grouped shared head with a small number of family buckets rather than a fully shared pair or fully domain-specific head.

### 2026-06-08 12:25 - CC follow-up and final solver decision
- Hypothesis: A grouped-head variant might be justified if the two-coefficient shared head shows that reducing effective dimensionality rescues held-out rank or optimizer-facing regret.
- Command: `env -u ANTHROPIC_API_KEY claude --resume d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490 --model claude-opus-4-8 --effort max -p '<scoped shared-head follow-up prompt>'`
- Config: CC reviewed the shared-head implementation and the corrected 39-domain versus 167-partition distinction.
- Result:
  - CC verified the shared-head implementation is mathematically exact: summing per-domain design columns, fitting two coefficients, and expanding them uniformly is equivalent to predicting with `intercept + beta * sum(signal_d) + gamma * sum(penalty_d)`.
  - Shared NNLS degeneracy is expected from nonnegative clipping, not a code bug. Treat shared NNLS as unusable.
  - Shared signed ridge / alpha1 are valid diagnostics but not a production fitter change: the one-metric real-CV regret improvement is marginal and not accompanied by a heldout Spearman lift.
  - Corrected framing: real 300M CV is `n_train≈80` versus `p=78`, so it is near the domain-head parameter count rather than the `p=334` production regime. The fact that the well-determined two-parameter head still has heldout Spearman near zero suggests weak model-to-metric signal for this metric, not merely an optimizer/head-dimensionality problem.
  - Production 167-partition fitting remains more underdetermined (`p=334`; need roughly `n≳2p≈670` real observations based on synthetic behavior).
- Interpretation: Stop solver/head-variant iteration for now. None of current, coarse NNLS, domain signed ridge, fixed alpha1, shared NNLS, or shared signed ridge shows a clear multi-metric, statistically supported held-out rank/regret improvement. The useful output is the diagnostic harness and the conclusion that improving DSP fitting likely requires more real observations and/or evidence that the DSP model has held-out signal for the target metrics, not just another solver knob.
- Decision: Do not patch the production DSP fitter. Keep current production fitter. Retain coarse NNLS, domain signed ridge, and shared signed ridge as diagnostic comparators for future modeling studies.
