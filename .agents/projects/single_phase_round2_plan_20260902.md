# Single-phase modeling, round 2 (2026-09-02)

Goal: improve the single-phase successor `weibull_softplus_unscaled` under the round-1 protocol
(`.agents/handoffs/single_phase_observatory_benchmark_cc_report_20260902.md`), in the payoff order below, with
matched ablations and controls at Screen, promotion, Certify, heldout selection, the StarCoder gate, a five-repeat
finalist, and read-only Codex and DeepSeek reviews before any successor is named. Hard requirement from the user:
beat canonical DSP on all 45 two-bucket StarCoder curves (in-sample fit and argmin, and out of fold).

## Mechanisms, in order

1. Shared shapes. Two-stage fit: pass 1 stores each component fit's inner-CV RMSE per shape; pass 2 selects one
   shape per sharing unit (target group, panel, or all 39-bucket panels), normalizing each component's CV error by
   its repeat SD, and refits every head with that shape. Entries: `@shared_shape_target`, `@shared_shape_panel`,
   `@shared_shape_scale`.
2. Bucket interactions. Additive columns: total-benefit squared (`interaction=total_square`) and within-family
   high x low benefit products (`interaction=family_products`); OOF residual-structure diagnostic on the additive
   successor.
3. Heteroskedastic targets and significance prior. Repeat-SD weighting enters through (1). A per-(bucket, metric)
   ridge multiplier or mask from `reference_outputs/domain_ablation_pvalue_matrix_with_training_eval_20260623`
   (`@ablation_prior`), plus a scrambled-prior control.
4. Quality-axis pooling on Michael panels: benefit and harm columns pooled by quality bin across clusters
   (`quality_axis=benefit|harm|both`), with a shuffled-quality control.
5. Cross-scale sharing: shape shared across 60M/300M/Delphi with per-panel amplitudes (part of 1).
6. StarCoder: `@wide_grid` (rates to 400, power to 0.15), `@huber_head`, bounded log-deficit link
   (`link=log_deficit_bounded`, linear predictor capped at the largest training deficit plus a margin).

## Protocol additions

- Every models-module edit: dump a cache generation first, keep descriptions of existing entries unchanged,
  refresh helper pins only after a shard-reproduction check (refit a sample of existing tasks, compare predictions).
- Pooled sign tests gain Holm-corrected p-values; the unit correlation caveat stays in the report.
- Per-curve StarCoder gate table: in-sample RMSE, argmin, regret and OOF RMSE, regret for the successor candidates
  against DSP on all 45 curves.

## Outputs

Same output directory (`reference_outputs/single_phase_observatory_benchmark_20260902/`), report section 14 in
the round-1 report or a new report `single_phase_observatory_round2_cc_report_20260902.md`, Fieldbook experiment
`exp_01m1ge7ye6hz2epd0mjkbkrvt8`.
