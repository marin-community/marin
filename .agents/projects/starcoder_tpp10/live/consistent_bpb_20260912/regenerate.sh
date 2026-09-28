#!/bin/sh
set -eu
uv run python -m experiments.domain_phase_mix.plot_starcoder_tpp10_refinement \
  --pilot-plan .agents/projects/starcoder_tpp10/live/pilot_plan.json \
  --pilot-metrics '/Users/calvinxu/Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper_one_phase/revision_notes/20260911_tpp10_measured_curves/data/pilot_metrics.csv' \
  --refinement-plan .agents/projects/starcoder_tpp10/live/refinement_plan.json \
  --snapshot .agents/projects/starcoder_tpp10/live/completion_20260912/verified_endpoints.json \
  --metric-records .agents/projects/starcoder_tpp10/live/target_jump_audit_20260912/all_102_final_metric_records.json \
  --population-counts .agents/projects/starcoder_tpp10/domain_sweeps/repairs_20260911/population_counts.json \
  --output .agents/projects/starcoder_tpp10/live/consistent_bpb_20260912
