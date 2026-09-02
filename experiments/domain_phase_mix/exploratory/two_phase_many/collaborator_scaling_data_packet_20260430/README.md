# Collaborator Scaling Data Packet

Generated: 2026-05-01T03:17:05.810158+00:00

This packet contains the refreshed data portion for joint data-mixture and scale modeling, plus small
self-contained reference implementations for fixed-scale GRP-style fitting and the MCT-LRQ joint scale
law. The included scripts do not depend on this branch or the broader Marin repository.

## Data Contract

- `data/analysis_dataset/nd_scale_runs.csv` is the canonical row table for joint scale/mix modeling.
- `data/analysis_dataset/nd_scale_packet.npz` is the packet-compatible array form.
- `model_size` / `model_sizes` mean corrected non-embedding parameter count, not historical nominal labels.
- Historical scale strings such as `300m_6b` are stable IDs only.
- Display labels use corrected non-embedding `N` plus realized target-budget `D`:
  `20M/2.6B`, `60M/1.2B`, `100M/6B`, `340M/10.4B`, `900M/24B`.
- Do not make N-only scaling claims from this data without also showing `D`.

## Included Tables

- `data/analysis_dataset/nd_scale_runs.csv`: canonical scaling-law modeling rows.
- `data/analysis_dataset/nd_scale_packet.npz`: arrays for fitting scripts.
- `data/metric_registry/metrics_wide.csv`: refreshed wide metric registry view.
- `data/raw_metric_matrix_300m/raw_metric_matrix_300m.csv`: 242-row 300M/6B qsplit-core metric matrix with
  `phase_0_*`, `phase_1_*`, and `exposure_80_20_*` mixture columns.
- `data/raw_metric_matrix_300m/raw_metric_matrix_300m_with_noise.csv`: the same signal rows plus any
  completed `run_00097` noise rows.
- `data/raw_metric_matrix_300m/noise_baseline_run00097_fixed_subset_300m.csv`: trainer-seed noise with
  the simulated-epoch subset fixed to `run_00097`.
- `data/raw_metric_matrix_300m/noise_baseline_run00097_variable_subset_300m.csv`: trainer-seed noise with
  the simulated-epoch subset left variable, matching the original swarm sampling more closely.
- `data/grp_no_l2/two_phase_many.csv`: original 60M/1.2B GRP fit panel.
- `data/grp_no_l2/two_phase_many_epoch_metadata.csv`: epoch multipliers used by the exact GRP fit.
- `data/grp_no_l2/grp_penalty_calibration_variants_best.csv`: regularized GRP best row used to seed the
  no-L2 retune.
- `data/grp_no_l2/grp_power_family_penalty_no_l2_retune_best.csv`: repo-generated no-L2 best row for
  reproducibility checks.
- `data/run_registry/logical_runs.csv`: refreshed operational provenance table.

## Standalone Code

- `standalone_code/load_packet.py`: validates the shipped CSV/NPZ files and reconstructs phase weights.
- `standalone_code/grp_no_l2_exact.py`: exact standalone port of the repo's GRP power-family-penalty
  no-L2 path, including nonlinear retuning.
- `standalone_code/fit_mct_lrq.py`: compact MCT-LRQ-style joint mixture/scale law using only packet CSVs.
- `standalone_code/requirements.txt`: minimal Python package requirements.

Run from the packet root:

```bash
python standalone_code/load_packet.py
python standalone_code/grp_no_l2_exact.py --mode fit-best --output-dir outputs/grp_no_l2_exact
python standalone_code/grp_no_l2_exact.py --mode retune --method Powell --coarse-top-k 3
python standalone_code/fit_mct_lrq.py --output-dir outputs/mct_lrq_demo
```

The packet still includes `reference_outputs/joint_model_refreshed_20260426/` reports, CSVs, and model
JSONs for comparison, but it intentionally does not copy branch-dependent Marin experiment scripts.

## Refreshed Counts

```json
{
  "analysis_dataset": {
    "dropped_stale_or_unready_strong_tier_rows": 4,
    "dropped_total_rows": 4,
    "dropped_unlabeled_rows": 3,
    "output_rows": 641,
    "source_rows": 645
  },
  "metric_registry": {
    "canonical_metric_count": 965,
    "conflict_count": 99423,
    "metric_fact_count_canonical": 744365,
    "run_count": 1217,
    "source_count": 22
  },
  "raw_metric_matrix_300m": {
    "columns": 1209,
    "domains": [
      "dolma3_arxiv",
      "dolma3_cc/art_and_design_high",
      "dolma3_cc/art_and_design_low",
      "dolma3_cc/crime_and_law_high",
      "dolma3_cc/crime_and_law_low",
      "dolma3_cc/education_and_jobs_high",
      "dolma3_cc/education_and_jobs_low",
      "dolma3_cc/electronics_and_hardware_high",
      "dolma3_cc/electronics_and_hardware_low",
      "dolma3_cc/entertainment_high",
      "dolma3_cc/entertainment_low",
      "dolma3_cc/finance_and_business_high",
      "dolma3_cc/finance_and_business_low",
      "dolma3_cc/food_and_dining_high",
      "dolma3_cc/food_and_dining_low",
      "dolma3_cc/games_high",
      "dolma3_cc/games_low",
      "dolma3_cc/health_high",
      "dolma3_cc/health_low",
      "dolma3_cc/history_and_geography_high",
      "dolma3_cc/history_and_geography_low",
      "dolma3_cc/industrial_high",
      "dolma3_cc/industrial_low",
      "dolma3_cc/literature_high",
      "dolma3_cc/literature_low",
      "dolma3_cc/science_math_and_technology_high",
      "dolma3_cc/science_math_and_technology_low",
      "dolma3_finemath_3plus",
      "dolma3_stack_edu",
      "dolma3_wikipedia",
      "dolmino_common_crawl_hq",
      "dolmino_olmocr_pdfs_hq",
      "dolmino_stack_edu_fim",
      "dolmino_stem_heavy_crawl",
      "dolmino_synth_code",
      "dolmino_synth_instruction",
      "dolmino_synth_math",
      "dolmino_synth_qa",
      "dolmino_synth_thinking"
    ],
    "exposure_average_weight_columns": 39,
    "id_columns": 19,
    "metric_columns": 1073,
    "path": "/Users/calvinxu/Projects/Work/Marin/marin/experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv",
    "phase0_weight_columns": 39,
    "phase1_weight_columns": 39,
    "rows": 242
  },
  "run_registry": {
    "logical_run_count": 950,
    "logical_runs_by_status": {
      "completed": 937,
      "failed": 6,
      "planned": 2,
      "running": 5
    },
    "strong_tier_perplexity_ready_rows": 117
  }
}
```

## Notes

The metric registry may report many source conflicts because local historical CSVs, checkpoint metrics,
and collected eval outputs overlap. The `metrics_wide.csv` table uses the registry's canonical source
priority resolution.
