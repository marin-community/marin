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
