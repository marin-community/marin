# Reduced-Bias Domain GRP: Research Logbook

## Scope
- Goal: test whether less structured, per-domain saturation models improve 300M data-mixture fitting.
- Primary metric: 300M/6B `eval/uncheatable_eval/bpb` fit and raw-optimum sanity.
- Constraint: local modeling only; no training launch.

### 2026-05-10 - 300M reduced-bias domain model sweep
- Hypothesis: removing GRP family/quality inductive bias and giving each domain its own saturation curve may improve 300M fit and optima.
- Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_reduced_bias_domain_models_300m.py`
- Config: 300M/6B swarm-like 242-row fit frame, 39-domain reduced-bias variants, NNLS variable projection, bounded nonlinear fitting.
- Result: best CV RMSE `domain_effective_exposure_penalty_158` cv_rmse=0.008024; best rank `domain_effective_exposure_penalty_158` oof_spearman=0.905364.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/reduced_bias_domain_models_300m_20260510`.
- Interpretation: per-domain saturation has real predictive signal. The defensible reduced-bias 158-param form (`domain_phase_benefit_penalty_158`, phase-1 premium only in the benefit term) improves CV RMSE from old GRP 0.011141 to 0.008848 and OOF Spearman from 0.804042 to 0.899626. The less defensible effective-exposure analogue is even stronger on fit (CV RMSE 0.008024, OOF Spearman 0.905364), which says the old phase multiplier bias remains useful empirically.
- Optimum quality: unconstrained raw optima are still not deployable. All reduced-bias raw optima have nearest-observed TV 0.5, and several collapse a phase to a small number of domains. The reduced-bias models are currently best understood as stronger observed-row rankers, not as safe direct argmin laws.
- Best-observed check: the defensible 158-param form predicts observed best `run_00125` as best; the effective-exposure and 159-param variants choose `baseline_olmix_loglinear_uncheatable_bpb`, actual rank 3. See `predicted_observed_leaderboard.csv` and `report.md`.

### 2026-05-10 - DSP canonical form sweep
- Hypothesis: the canonical DSP form can be chosen by testing phase semantics, penalties, and head constraints under a 4-parameters-per-domain budget.
- Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_300m.py`
- Config: 300M/6B 242-row fit frame, DSP variants with at most four M-dependent parameters per domain.
- Result: best CV RMSE `dsp_effective_exposure_penalty_nnls` cv_rmse=0.007106; best rank `dsp_effective_exposure_penalty_nnls` oof_spearman=0.919645.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510`.
- Interpretation: canonical DSP should be `dsp_phase_benefit_penalty_nnls`, not the numerically best effective-exposure comparator. The canonical form has CV RMSE 0.008835 and OOF Spearman 0.898476, predicts observed best `run_00125` as best, preserves nonnegative benefit/penalty semantics, and avoids putting the phase-1 multiplier inside the saturation/penalty exposure. The effective-exposure version is an empirical upper bound (CV RMSE 0.007106, Spearman 0.919645) but reintroduces the phase-saturation/penalty bias.
- Negative ablations: no-phase remains unusable (CV RMSE 0.019480, Spearman 0.306467); no-penalty variants degrade strongly (CV RMSE 0.014162 / 0.013497); signed heads do not improve fit and weaken semantics.
- Remaining blocker: raw optima for all DSP variants remain off-manifold (`raw_nearest_observed_tv=0.5`), so DSP is currently a stronger ranker/surrogate, not a safe unconstrained optimizer.

### 2026-05-10 - DSP retention ablation
- Hypothesis: reintroducing a global retention term could recover useful two-phase signal without restoring GRP family/quality structure.
- Command: `uv run --no-project --with matplotlib --with scipy --with scikit-learn --with pandas --with tabulate --with numpy python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_300m.py`
- Config: same 300M/6B 242-row DSP sweep. Added three 159-parameter variants:
  - `dsp_phase_benefit_exp_retention_penalty_nnls`: canonical reduced-bias DSP plus original GRP-style `exp(-lambda * (1 - p1))` phase-0 retention.
  - `dsp_phase_benefit_overlap_retention_penalty_nnls`: canonical reduced-bias DSP plus a simple scalar repeated-exposure overlap term.
  - `dsp_effective_exposure_exp_retention_penalty_nnls`: original-style retained phase-0 exposure plus phase-1 multiplier inside effective exposure.
- Result:
  - Canonical DSP remains `dsp_phase_benefit_penalty_nnls`: CV RMSE `0.008835`, OOF Spearman `0.898476`.
  - Original exponential retention on canonical DSP overfits: train RMSE improves `0.005629 -> 0.004992`, but CV RMSE worsens `0.008835 -> 0.011462`. OOF Spearman rises slightly to `0.904389`, but Pearson collapses to `0.811259`.
  - Simple overlap retention is nearly neutral: CV RMSE `0.008914`, OOF Spearman `0.900624`, fitted overlap scalar `lambda=0.020934`.
  - Effective-exposure exponential retention does not beat the existing empirical comparator: CV RMSE `0.007320` vs `0.007106`, OOF Spearman `0.917789` vs `0.919645`.
- Interpretation:
  - Do not add retention to canonical DSP for now. The simple scalar term is too close to zero to justify an extra parameter, and original exponential retention mainly buys in-sample fit.
  - If we revisit retention, prefer a constrained or regularized overlap term rather than GRP-style exponential retention.

### 2026-05-10 - perturbation-row inclusion ablation
- Question: do the 55 proportional perturbation rows materially improve DSP fit quality at 60M/1.2B or 100M/6B, enough to justify including local intervention rows in future swarms?
- Command: `uv run --no-project --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_dsp_perturbation_inclusion_20260510.py`
- Config: retuned nonlinear coefficients for canonical DSP and the effective-exposure empirical comparator. Compared original-only fits against original-plus-perturbation fits. Original-row evaluation used OOF folds over original rows with all perturbations available to augmented training folds. Perturbation-row evaluation used OOF folds over perturbation rows with all original rows available to augmented training folds.
- Result for canonical DSP:
  - 60M original rows: RMSE `0.009969 -> 0.009732`, Spearman `0.850313 -> 0.859136`; paired bootstrap RMSE-delta CI `[-0.000716, 0.000277]`.
  - 60M perturbation rows: external/OOF RMSE `0.007144 -> 0.006263`, Spearman `0.785065 -> 0.836147`; CI `[-0.001616, 0.000145]`.
  - 100M original rows: RMSE `0.008835 -> 0.008841`, Spearman `0.898476 -> 0.902808`; CI `[-0.000290, 0.000306]`.
  - 100M perturbation rows: external/OOF RMSE `0.007833 -> 0.007006`, Spearman `0.763997 -> 0.770491`; CI `[-0.001322, -0.000281]`.
- Result for effective-exposure comparator:
  - 60M original rows: RMSE `0.007280 -> 0.007316`, Spearman `0.903576 -> 0.911618`; CI `[-0.000398, 0.000531]`.
  - 60M perturbation rows: RMSE `0.004947 -> 0.004769`, Spearman `0.805556 -> 0.830592`; CI `[-0.000446, 0.000299]`.
  - 100M original rows: RMSE `0.007106 -> 0.007911`, Spearman `0.919645 -> 0.902334`; CI `[-0.000338, 0.002703]`.
  - 100M perturbation rows: RMSE `0.005113 -> 0.003454`, Spearman `0.789971 -> 0.762987`; CI `[-0.002049, -0.001175]`.
- Interpretation: perturbation rows improve interpolation/extrapolation on the local perturbation manifold, especially at 100M, but they do not significantly improve original-swarm OOF fit. Future swarms should include a small, deliberate perturbation block for gradient and local-causal diagnostics, not replace broad randomized/D-optimal coverage with many proportional perturbations.
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_perturbation_inclusion_ablation_20260510/`.

### 2026-05-10 - DSP split phase-mechanism ablation
- Hypothesis: the strong `dsp_effective_exposure_penalty_nnls` result may be hiding multiple phase effects under one tied scalar. Split the phase effect into global benefit, saturation, and penalty multipliers while keeping the M-dependent budget at four parameters per domain.
- Command: `uv run --no-project --with matplotlib --with scipy --with scikit-learn --with tabulate --with pandas --with numpy python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_300m.py`
- Config: 300M/6B 242-row fit frame. Added:
  - `dsp_phase_benefit_saturation_penalty_nnls`: separate `gamma_benefit`, `gamma_saturation`, `gamma_penalty`.
  - `dsp_phase_benefit_saturation_nnls`: canonical plus saturation multiplier only.
  - `dsp_phase_benefit_penalty_phase_nnls`: canonical plus penalty multiplier only.
  - `dsp_saturation_only_penalty_nnls`: saturation multiplier only.
  - `dsp_penalty_phase_only_nnls`: penalty multiplier only.
  - `dsp_saturation_penalty_split_nnls`: separate saturation and penalty multipliers, no benefit premium.
  - `dsp_benefit_effective_exposure_nnls`: canonical benefit premium plus tied saturation/penalty multiplier.
- Result:
  - Original collaborator-facing canonical `dsp_phase_benefit_penalty_nnls` reproduced exactly: CV RMSE `0.008835`, OOF Spearman `0.898476`, fitted `gamma_benefit=25.355533`.
  - Tied effective-exposure comparator remains strong: CV RMSE `0.007106`, OOF Spearman `0.919645`, fitted tied gamma `14.362237`.
  - Best fit is now `dsp_saturation_penalty_split_nnls`: CV RMSE `0.006407`, OOF Spearman `0.929312`, fitted `gamma_saturation=69.936629`, `gamma_penalty=5.654080`, no benefit premium.
  - Fully split `dsp_phase_benefit_saturation_penalty_nnls` is second-best: CV RMSE `0.006635`, OOF Spearman `0.926152`, fitted `gamma_benefit=0.002044`, `gamma_saturation=98.166580`, `gamma_penalty=5.527307`.
  - Canonical plus saturation only improves modestly: CV RMSE `0.008287`.
  - Canonical plus penalty-phase only improves more: CV RMSE `0.007511`.
  - Saturation-only without benefit/penalty-phase is weaker than canonical: CV RMSE `0.009525`; penalty-only is weaker still: `0.010855`.
- Interpretation:
  - Effective exposure was not just a nuisance overfit. Phase-1 curvature/penalty exposure has real predictive signal on the 300M swarm.
  - The unfounded part was tying all phase effects to one scalar. Once split, the model mostly discards benefit premium and uses separate saturation and penalty exposure multipliers.
  - The fitted saturation multiplier is very large and near the upper bound in the fully split model, so do not promote this as final canonical without cross-scale and perturbation-gradient checks.
  - Raw optima are still off-manifold (`raw_nearest_observed_tv` about `0.5`), so these remain ranking/surrogate models rather than safe direct optimizers.
- Artifacts: `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_300m_20260510/`.

### 2026-05-10 - split DSP cross-scale and perturbation check
- Hypothesis: a candidate replacement canonical DSP form should hold up at both 60M and 100M and should not only improve broad-swarm CV; it should also behave sensibly on the proportional perturbation panel.
- Commands:
  - `uv run --no-project --with matplotlib --with scipy --with scikit-learn --with tabulate --with pandas --with numpy python experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_60m.py`
  - `uv run --no-project --with numpy --with pandas --with scipy --with scikit-learn --with matplotlib --with tabulate python experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_dsp_perturbation_inclusion_20260510.py`
- 60M result:
  - Best CV RMSE: `dsp_phase_benefit_saturation_penalty_nnls`, CV RMSE `0.006766`, OOF Spearman `0.916493`, fitted `gamma_benefit=0.002105`, `gamma_saturation=9.467781`, `gamma_penalty=4.521505`.
  - `dsp_saturation_penalty_split_nnls`: CV RMSE `0.007321`, OOF Spearman `0.897725`.
  - Tied effective exposure: CV RMSE `0.007280`, OOF Spearman `0.903576`.
- 100M result:
  - Best CV RMSE: `dsp_saturation_penalty_split_nnls`, CV RMSE `0.006407`, OOF Spearman `0.929312`, fitted `gamma_saturation=69.936629`, `gamma_penalty=5.654080`.
  - Fully split `dsp_phase_benefit_saturation_penalty_nnls`: CV RMSE `0.006635`, OOF Spearman `0.926152`, fitted `gamma_benefit=0.002044`, `gamma_saturation=98.166580`, `gamma_penalty=5.527307`.
- Perturbation inclusion check:
  - 60M fully split original rows degrade when adding perturbations: RMSE `0.006766 -> 0.007328`; perturb rows improve `0.005290 -> 0.004560`.
  - 60M saturation/penalty split original rows improve slightly with perturbations: RMSE `0.007321 -> 0.007023`; perturb rows improve `0.005652 -> 0.003970`.
  - 100M fully split original rows degrade when adding perturbations: RMSE `0.006635 -> 0.007810`; perturb rows are roughly neutral `0.006059 -> 0.006233`.
  - 100M saturation/penalty split original rows improve slightly with perturbations: RMSE `0.006407 -> 0.006227`; perturb rows improve strongly `0.006440 -> 0.003181`.
- Interpretation:
  - The safest provisional new best form is `dsp_saturation_penalty_split_nnls`: it is best at 100M, close at 60M, improves under perturbation inclusion at both scales, and is simpler than the fully split form.
  - The fully split form is the best 60M CV model, but its learned benefit premium is essentially zero at both scales and it degrades under perturbation inclusion, especially at 100M.
  - The raw optima remain off-manifold and collapsed, so use predicted optima as diagnostics only.
- Artifacts:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_60m_20260510/`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_perturbation_inclusion_ablation_20260510/`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_phase_cross_scale_summary_20260510.csv`

### 2026-05-10 - 60M DSP split phase-mechanism check
- Hypothesis: split phase-mechanism DSP should be checked at 60M before promoting the 300M winner.
- Command: `uv run --no-project --with matplotlib --with scipy --with scikit-learn --with tabulate --with pandas --with numpy python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_canonical_variants_60m.py`
- Config: 60M/1.2B 242-row fit frame, same DSP variants as the 300M split sweep.
- Result: best CV RMSE `dsp_phase_benefit_saturation_penalty_nnls` cv_rmse=0.006766; best rank `dsp_phase_benefit_saturation_penalty_nnls` oof_spearman=0.916493.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_canonical_variants_60m_20260510`.

### 2026-05-14 - Apple-style repetition-aware DSP check
- Hypothesis: explicit Apple-style repetition discounting may improve DSP by separating physical repeated exposure from learned saturation.
- Command: `uv run --with matplotlib --with scipy --with scikit-learn --with tabulate python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_dsp_apple_repetition_variants_300m.py`
- Config: 300M/6B 242-row fit frame; compared canonical, effective-exposure, split-phase, shared-r1 Apple DSP, and per-domain-r1 Apple DSP.
- Result: shared-r1 cv_rmse=0.009807, oof_spearman=0.893272; per-domain-r1 cv_rmse=0.008739, oof_spearman=0.900558.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_apple_repetition_variants_300m_20260514`.
- Interpretation: see `report.md`.

### 2026-05-14 - split DSP vs effective-exposure perturbation-gradient checks
- Hypothesis: if `dsp_phase_benefit_saturation_penalty_nnls` is a stronger canonical candidate than `dsp_effective_exposure_penalty_nnls`, it should not only improve OOF fit but also agree with measured proportional domain-bump effects and cross-scale perturbation transfer.
- Command: `uv run --with matplotlib --with pandas --with scipy --with scikit-learn --with tabulate --with plotly --with kaleido python -m experiments.domain_phase_mix.exploratory.two_phase_many.compare_dsp_split_vs_effective_checks`
- Config: cached 60M and 100M DSP fits; proportional perturbation domain-bump panel; compared finite `DSP(w_bump)-DSP(w_prop)`, local directional derivative at proportional, cross-scale transfer, and 60M-to-100M interaction prediction.
- Result:
  - Split improves OOF fit at both scales: 60M CV RMSE `0.006766` vs effective `0.007280`; 100M CV RMSE `0.006635` vs effective `0.007106`.
  - Within-scale finite perturbation predictions are close: at 60M split has Pearson `0.9135` vs effective `0.8997`; at 100M effective has Pearson `0.9069` vs split `0.8957`.
  - Local-gradient behavior is worse for split at 100M: local Pearson `0.0127` vs effective `0.6779`, although split sign agreement remains `31/39`.
  - Scale-interaction finite prediction is mixed: split has slightly higher Pearson `0.6502` vs effective `0.6242`, but worse RMSE `0.003347` vs `0.002734` and lower Spearman `0.2874` vs `0.3506`.
- Interpretation: split benefit/saturation/penalty DSP is the better broad-swarm fit and a competitive finite-bump surrogate, but effective-exposure DSP remains the cleaner local-gradient comparator near proportional. Do not use split raw optima directly; they remain off-manifold and collapsed.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/dsp_split_vs_effective_checks_20260514/`.
