# GRP Phase-Bias Ablation: Research Logbook

## Scope
- Goal: test whether old GRP's phase-1 effective-exposure multiplier is a harmful inductive bias.
- Primary metric: 300M/6B `eval/uncheatable_eval/bpb` fit and raw-optimum sanity.
- Constraint: local modeling only; no training launch.

### 2026-05-10 - 300M phase-premium no-L2 GRP ablation
- Hypothesis: the old GRP phase-1 multiplier inside effective exposure may incorrectly force phase-1 exposure to saturate earlier.
- Command: `uv run python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_grp_phase_premium_ablation_300m.py`
- Config: 300M/6B qsplit-core + Olmix/Uniform baseline fit frame, old `power_family_penalty` GRP blocks, no L2, phase premium outside raw exposure saturation.
- Result: cv_rmse=0.012981, oof_spearman=0.780390, cv_foldmean_regret_at_1=0.011016, cv_rawopt_nearest_tv=0.828047, raw_nearest_observed_tv=0.500000.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_phase_premium_ablation_300m_20260510`.
- Interpretation: see `report.md` in the artifact directory.

### 2026-05-10 - 300M phase benefit+penalty premium no-L2 GRP ablation
- Hypothesis: the old GRP phase-1 multiplier inside effective exposure may incorrectly force phase-1 exposure to saturate earlier; a benefit premium plus matched penalty premium could preserve phase-1 value without the effective-exposure coupling.
- Command: `uv run python /Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/fit_grp_phase_premium_ablation_300m.py`
- Config: 300M/6B qsplit-core + Olmix/Uniform baseline fit frame, old `power_family_penalty` GRP blocks, no L2, phase benefit premium and penalty premium outside raw exposure saturation.
- Result: cv_rmse=0.011993, oof_spearman=0.835427, cv_foldmean_regret_at_1=0.003261, cv_rawopt_nearest_tv=0.769456, raw_nearest_observed_tv=0.500000.
- Artifacts: `/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/grp_phase_benefit_penalty_premium_ablation_300m_20260510`.
- Interpretation: see `report.md` in the artifact directory.
