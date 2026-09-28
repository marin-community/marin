# StarCoder refinement completion check, 12 September 2026

The 40 completed refinement artifacts were rechecked against the frozen plan, including exact config fingerprints, permanent final checkpoint metadata, child runtime receipts, finite final-step PALOMA scores, and finished W&B runs with matching metrics.

Five recipes remain queued under `/calvinxu/starcoder-tpp10-refinement-batch-recovery`: target p=55, 60, 65, 80 and matched p=80 (trainer 20260911, subset 20260914). The current coordinator is healthy and the five TPU jobs already use BATCH priority. Central1 v5p preemptible capacity is quota-blocked (2048 limit); this is not a CPU coordinator bottleneck. No jobs were canceled, duplicated, or resubmitted.

Saved temporary target checkpoints survive at steps 10759, 6958, 2283, 728 of 11491 respectively; the pending matched run has no checkpoint. Exact generation and metadata hashes are in `pending_checkpoint_receipt.json`. Successful final endpoints are in `verified_endpoints.json`. The parent has a 48-hour execution timeout ending 13 September 12:38:22 UTC (parent start verified from Iris); if it expires, first confirm terminal descendants and inactive leases before using the existing recovery command. Keep all frozen fingerprints and success reuse.

Final plots should be rebuilt only after the missing endpoints pass the same checks; no point is imputed and no incomplete target regret is reported.

## Completion commands

Run from `/Users/calvinxu/Projects/Work/Marin/marin` after Iris reports every remaining child succeeded:

```bash
uv run python -m experiments.domain_phase_mix.launch_starcoder_tpp10_refinement \
  --plan-path .agents/projects/starcoder_tpp10/live/refinement_plan.json \
  --collect-results .agents/projects/starcoder_tpp10/live/completion_20260912/refinement_metrics.csv
uv run python -m experiments.domain_phase_mix.plot_starcoder_tpp10_refinement \
  --pilot-plan .agents/projects/starcoder_tpp10/live/pilot_plan.json \
  --pilot-metrics '/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/revision_notes/20260911_tpp10_measured_curves/data/pilot_metrics.csv' \
  --refinement-plan .agents/projects/starcoder_tpp10/live/refinement_plan.json \
  --output .agents/projects/starcoder_tpp10/live/completion_20260912 --refresh
```

The first command requires all 45 final artifacts and exact identities. The second adds permanent checkpoint and W&B verification and emits the verified snapshot, per-curve measured points, analysis, and diagnostic curves. Require `refinement_complete=45`, an empty missing list, and an empty `verified_but_unplotted` list before finalizing the plot. Preserve the previous partial snapshot if overwriting it.

## Common-grid analysis and paper figure

The existing plot collector deliberately leaves `pilot_common_grid_analysis` on the seven-point pilot grid even after collecting refinements. Finalizing the paper requires an explicit analysis update: evaluate the union `[0, 10, 30, 40, 50, 55, 60, 65, 70, 80, 90, 100]` from the separately verified 57 pilot and 45 refinement recipes (102 distinct trained artifacts). Keep both original plan hashes. Do not alter either frozen plan, monkey-patch the pilot/dense constants, or call this the unreleased 21-coordinate dense stage.

Use the same estimator as the pilot: average the two trainer seeds for the unmatched curve and separately for each of the three matched subsets; retain one target seed. Select the minimum measured value on the complete common grid, breaking exact ties toward smaller fractions. For each proxy selection, calculate target regret as target BPB at that fraction minus the minimum target BPB across the same 12 points. Retain separate subset choices and curve-agreement diagnostics; they are not independent target replications.

Update the paper builder to read this new complete-common-grid analysis instead of its seven-point `pilot_common_grid_analysis`. Its other input, `data/allocation_audit.json`, must cover any newly selected percentage; retain the actual allocated epoch counts. If the three subset means select different percentages, replace the builder's current same-selection assumption with distinct labels or a concise range; do not pool silently. Recompute percentage brackets from the new target minimum and selections. The builder's existing numerical checks should compare against the new common-grid regrets.

Paper root: `/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase`.

Builder: `revision_notes/20260912_outline_figures/build_epoch_matching.py`; data input: its sibling `data/epoch_matching_analysis.json`; outputs: `figures/epoch_matching_tpp10.pdf`, `figures/epoch_matching_tpp10.png`, and sibling `figure5_receipt.json`. Preserve the approved two-panel styling, raw measured points, absolute BPB axes, and percentage/epoch annotations. Run the builder with `uv run` after updating its verified data and complete-grid reference, then render and inspect it. Rebuild the paper and update the caption, outline, and CC change list if selected percentages, epochs, or excess-loss percentages move.

Monitoring ownership was handed to the root task's 15-minute heartbeat on 12 September. Its recovery boundary and exact command are recorded in `scratch/20260912_starcoder_refinement_completion_monitoring_state.json`.
