# CC handoff: consistent BPB in the StarCoder figure

The target-curve jump was caused by combining 97 legacy BPB endpoints with five corrected endpoints after interactive recovery. All 102 plotted endpoints now use total loss bits divided by total scored bytes, reconstructed from the saved final token-average losses and the audited PALOMA population. Original run artifacts, raw CSVs, the verified endpoint snapshot and frozen training plans are preserved.

The complete twelve-point grid now has its target minimum at 70% StarCoder (0.7655645236 BPB). Each of the three matched subset means still selects 50%; unmatched selects 100%. Target losses at those selections are 0.7756673393 and 0.8006188044 BPB. Excess losses are 0.0101028157 and 0.0350542808 BPB, so simulated epoching avoids 71.1795% of unmatched selection regret. Figure brackets show +1.32%, +4.58%, and 71.2% less excess loss. One individual matched replicate changes its selected fraction from 80% to 55%; the corresponding two-seed subset mean still selects 50%.

The figure retains its approved two-panel layout, colors, measured points and segments, epoch annotations, and absolute BPB axes. Both proxy and target curves are corrected; the proxy limits expand downward to retain the corrected endpoint. No fitted curve was added. The main paragraph reports 71%, the caption 71.2%, and Appendix B.1 defines BPB as total prediction loss bits divided by total scored bytes. Current outline facts and caption agree; earlier incorrect values are explicitly marked superseded. The header figure and unrelated plots are untouched.

The canonical refinement plotter now requires the audited metric manifest and pinned population counts. It verifies all raw JSONL hashes, run identities, final steps, original reported BPB, finite token loss, known schema and agreement of schema 2 values with reconstruction. Its output records the common metric definition and preserves each raw value and schema separately in metric_provenance.json. Both plotters refuse unnormalized input. The paper builder was tested against the original mixed analysis and rejects it before rendering. Frozen training launchers and their original code pins were not changed.

All 60 plotted curve means and five selections agree with the independent audit to within 1e-12. The 12 focused regression tests and required targeted lint pass. The standalone main figure, diagnostic curves and compiled pages 5,16,17 were visually checked; an independent image-only reader understood the selection/regret comparison and found no overlap or clipping. The paper remains 43 pages, references begin on page 10, and there are no warnings, undefined references or overfull boxes.

Canonical corrected output is .agents/projects/starcoder_tpp10/live/consistent_bpb_20260912/ in the Marin repository. Its README carries the offline regeneration command. The paper input is revision_notes/20260912_outline_figures/data/epoch_matching_analysis.json. This revision's data/ retains the normalized analysis, per-run metric provenance, plotted points, exact population counts, independent audit and final figure receipt. before/ retains the previous paper inputs and outputs. Fieldbook experiment exp_01m23ddmn78breygyvkkpzyrq8 records the correction. No new training, checkpoint evaluation, commit or push was performed; the completed monitor stays paused.

## Rebuild

From the Marin repository:

```bash
uv run python -m experiments.domain_phase_mix.plot_starcoder_tpp10_refinement \
  --pilot-plan .agents/projects/starcoder_tpp10/live/pilot_plan.json \
  --pilot-metrics '/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/revision_notes/20260911_tpp10_measured_curves/data/pilot_metrics.csv' \
  --refinement-plan .agents/projects/starcoder_tpp10/live/refinement_plan.json \
  --snapshot .agents/projects/starcoder_tpp10/live/completion_20260912/verified_endpoints.json \
  --metric-records .agents/projects/starcoder_tpp10/live/target_jump_audit_20260912/all_102_final_metric_records.json \
  --population-counts .agents/projects/starcoder_tpp10/domain_sweeps/repairs_20260911/population_counts.json \
  --output .agents/projects/starcoder_tpp10/live/consistent_bpb_20260912
```

Copy analysis.json to the paper builder's input and run its PEP723 script with uv run, then run the paper's ./build.sh.
