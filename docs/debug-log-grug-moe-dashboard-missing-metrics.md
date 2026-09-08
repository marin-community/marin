# Debugging log for Grug-MoE dashboard missing task plots

The goal is to explain why GSM8K/HumanEval task panels are missing from the
Grug-MoE dashboard and update the dashboard so available task metrics are shown
instead of silently filtered out.

## Initial status

The dashboard at `reference_outputs/grug_moe_mix_dashboard_20260517` shows
per-task scaling plots but the user noticed GSM8K 5-shot and HumanEval 10-shot
are not visible. Local CSVs contain `logprob_gsm8k_5shot` and
`logprob_humaneval_10shot` rows, so the first hypothesis is that these tasks are
filtered out by dashboard completeness logic rather than missing from the raw
eval results.

## Hypothesis 1: complete-grid filtering hides partial tasks

`build_grug_moe_mix_dashboard.py` only plots task rows with
`common_task=True`. The GSM8K and HumanEval logprob rows have one missing
track/scale cell in the current local output, so both are classified as partial
coverage and excluded from the common-task plots.

## Changes to make

- Update task selection outputs to include `available_cells`, `expected_cells`,
  and a compact `coverage` label.
- Plot available task metrics instead of only common tasks.
- Keep common-task filtering for the aggregate, where complete grids are still
  the right default.
- Add coverage labels to per-task small-multiple titles so partial task panels
  are not mistaken for complete scaling ladders.

## Results

The regenerated dashboard includes all available loss-like task panels. The
GSM8K/HumanEval smooth tasks are present as:

- `logprob_gsm8k_5shot`: `19/20` cells
- `logprob_humaneval_10shot`: `19/20` cells

Both are missing the v4 d1536 cell, so they remain excluded from the strict
common-task aggregate but are now visible in the diagnostic per-task scaling
plot and task coverage table.
