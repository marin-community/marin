# GPT-6 Astra brief: Delphi surrogate and optimum selection

## Objective

Improve Delphi single-phase surrogate modeling and mixture-optimum selection. Treat
`weibull_softplus_unscaled` (WSPU) as the incumbent, not as a required functional form. The goal is a
better selection procedure under the frozen evaluation contract, not merely lower fit-panel RMSE.

Work offline. Do not submit training, evaluation, or infrastructure jobs, and do not inspect or tune
against the running WSPU cross-scale ladder.

## Environment

Use the current Marin checkout, or explicitly make its artifact root available. The required
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/` tree is excluded from Git and
will not appear automatically in a clean worktree. Preserve unrelated dirty and untracked files.

## Read first

1. `.agents/handoffs/delphi_apriori_swarm_280_handoff_20260904.md`, especially Sections 1 and 1b and the
   six linked modeling-round reports.
2. `.agents/handoffs/delphi_apriori_swarm_pilot_results_cc_handoff_20260906.md`.
3. `.agents/handoffs/single_phase_observatory_ablation_and_modeling_cc_handoff_20260902.md`.
4. `.agents/handoffs/single_phase_observatory_benchmark_cc_report_20260902.md`, including its 2026-09-06
   300M repeat-noise erratum.
5. `.agents/projects/delphi_offline_selection_20260906.md`.

Inspect these implementations after reading the evidence:

- `experiments/domain_phase_mix/exploratory/two_phase_many/single_phase_observatory_models_20260902.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_single_phase_observatory_20260902.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/evaluate_delphi_apriori_pilot_predictive_value_20260906.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/benchmark_delphi_selection_20260906.py`
- `experiments/domain_phase_mix/exploratory/two_phase_many/delphi_selection_models_20260906.py`

## Corrected repeat evidence

Do not repeat the prior claim that the 300M swarm lacks same-mixture repeats. A dedicated proportional
noise sweep trained ten repeats at both 60M and the historically named `300m_6b` scale, with trainer seeds
`10000..10009`. The 300M rows use the same 39-bucket proportional mixture as the canonical baseline and
have complete Uncheatable and 51-component Table-9 measurements.

The frozen 300M repeat source is:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/one_phase_swarm_scores_export_300m_20260630/proportional_reference_uncheatable_table9_scores_300m.csv`

Across the ten dedicated repeats, aggregate SD is `0.0011879189` BPB for Uncheatable and `0.0035056454`
BPB for Table 9. The Observatory harness now exposes these values on `300m_39bucket`. Historical 300M
noise-normalized metrics and basin-hit tolerances predate this correction; raw predictions, RMSE, rank
correlations, and continuous regrets remain usable.

The corrected protocol snapshot is
`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/single_phase_observatory_benchmark_20260902/repeat_noise_correction_20260906/protocol.json`.

## Evaluation contract

- Reproduce WSPU, canonical per-bucket DSP, and repository taskwise OLMix before evaluating alternatives.
- Fit any final candidate only on the canonical 280-run Delphi panel for the apples-to-apples comparison.
- Treat the external registry, modeling rounds 1-6, epoch-cap results, and the 37-run a-priori pilot as
  development evidence. They may guide model selection but cannot confirm the selected model.
- Keep coordinate aliases, repeat seeds, pool variants, and source memberships from crossing fitted splits.
- Report regret at 1, best-of-5 and best-of-10 regret, selected measured rank, selection optimism, RMSE,
  Spearman correlation, and behavior by source family and intervention stratum.
- Diagnose representation, pooling, heteroskedasticity, optimization, extrapolation, and the selection
  objective separately. Use matched ablations and report effective degrees of freedom.
- New parametric, hierarchical, low-rank, uncertainty-aware, kernel, or direct-selection methods are in
  scope when their complexity is defensible for 280 rows.

## Deliverable

Produce reproducible local artifacts and a concise report that gives either:

1. one fully specified frozen candidate plus a prospective paired-seed confirmation plan; or
2. an explicit null result explaining why the available development evidence does not support replacing
   the incumbent.

Do not claim improvement from retrospective bank performance alone.
