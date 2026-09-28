# Final plot implementation notes (read-only review)

After all 45 refinements pass, extract the existing numerical estimator in `experiments/domain_phase_mix/analyze_starcoder_tpp10.py:52` into `analyze_common_grid(runs, values, grid)`. Preserve the current pilot/dense validation wrapper and metadata. These analysis files are not in the frozen training code hashes.

In `plot_starcoder_tpp10_refinement.py:70`, add `complete_common_grid_analysis` only after all 45 verified endpoints, 102 distinct combined pilot/refinement run names, all five curves covering `[0,10,30,40,50,55,60,65,70,80,90,100]`, and empty missing/unplotted lists. Retain both original plan hashes and the historical seven-point `pilot_common_grid_analysis`. Keep two-seed means within each of the three matched subsets and the shared unmatched p=0 observations.

The paper builder `revision_notes/20260912_outline_figures/build_epoch_matching.py:39` must consume the complete result and use `(loss, percent)` ties. Its current first-subset collapse and averaged marker are valid only if all three subset choices coincide. Otherwise show each subset's selection, group labels only at equal percentages, show separate target-selected points, report regret/reduction ranges and save per-subset choices/epochs/percentages. Do not silently replace them with a pooled selection.

The paper and canonical allocation audits are byte-identical, SHA256 `561124284d1d90b4e25fcb873f232ff5b9cbd6afb9f81dbaf20faed770497167`, and already cover all 21 five-percent coordinates with the matching design hash. No expansion is needed.

After full verification, update `sections/simulated_epoching.tex:24`, `sections/appendix.tex:254` and current outline facts to specify the twelve-point grid and remove incomplete-run placeholders. Retain dated pilot history. Preserve approved two-panel crop, colors, absolute BPB axes, epoch labels and measured-point segments; no fitted-curve overlay or header edits. Compile and visually verify before pausing the heartbeat.

This review made no code, data or manuscript changes.
