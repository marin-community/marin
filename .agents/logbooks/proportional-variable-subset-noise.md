# Proportional Variable-Subset Noise Baseline

## Goal

Estimate noise around the proportional mixture at both `60m_1p2b` and
`300m_6b` (`60M/1.2B` and corrected `100M/6B`) using variable simulated-epoch
subsets. This complements the existing `run_00097` noise baselines: the anchor
mixture is now the operational proportional baseline rather than a sampled
swarm point.

## Training Design

- Source mixture: `baseline_proportional`.
- Scales: `60m_1p2b`, `300m_6b`.
- Repeats: `trainer_seed=10000..10009` at each scale.
- Variable subset controls: `data_seed=None` and
  `simulated_epoch_subset_seed=None`.
- Phase schedule: both `phase_0` and `phase_1` use the same proportional
  mixture.
- Resources: `v5p-8`, `us-east5-a`, `max_concurrent=20`.
- Training evals: perplexity-only, with final checkpoints and HF exports
  preserved for downstream eval completion.

## Implementation

- Training launcher:
  `experiments/domain_phase_mix/launch_proportional_variable_subset_noise_baseline.py`
- Registry integration:
  `experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py`
- Extra eval candidate builder:
  `experiments/domain_phase_mix/build_proportional_variable_subset_noise_eval_candidates.py`
- Proportional-noise matrix exporter:
  `experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_proportional_variable_subset_noise_matrices.py`

The shared downstream eval candidate discovery now accepts additional
registry-backed panels through `MARIN_EXTRA_EVAL_CANDIDATES_CSVS`, and the
affected eval launchers use each candidate's expected final checkpoint step
instead of assuming `300m_6b` step `22887`.

## Validation

Dry-run command:

```bash
.venv/bin/python experiments/domain_phase_mix/launch_proportional_variable_subset_noise_baseline.py \
  --dry-run \
  --scales 60m_1p2b,300m_6b \
  --tpu-region us-east5 \
  --tpu-zone us-east5-a \
  --max-concurrent 20
```

Dry-run checks:

- `20` total training specs.
- `10` specs for `60m_1p2b`, `10` for `300m_6b`.
- `data_seed=None` for every spec.
- `simulated_epoch_subset_seed=None` for every spec.
- Run names are stable:
  `propvar_<scale>_trainer_seed_<seed>`.

Targeted compile and lint passed:

```bash
.venv/bin/python -m py_compile \
  experiments/domain_phase_mix/launch_proportional_variable_subset_noise_baseline.py \
  experiments/domain_phase_mix/build_proportional_variable_subset_noise_eval_candidates.py \
  experiments/domain_phase_mix/launch_300m_gsm8k_humaneval_evals.py \
  experiments/domain_phase_mix/launch_300m_generative_smooth_proxy_evals.py \
  experiments/domain_phase_mix/launch_300m_mcq_smooth_proxy_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/build_eval_signal_to_noise_all_metrics_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_raw_metric_matrix_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_proportional_variable_subset_noise_matrices.py

.venv/bin/ruff check \
  experiments/domain_phase_mix/launch_proportional_variable_subset_noise_baseline.py \
  experiments/domain_phase_mix/build_proportional_variable_subset_noise_eval_candidates.py \
  experiments/domain_phase_mix/launch_300m_gsm8k_humaneval_evals.py \
  experiments/domain_phase_mix/launch_300m_generative_smooth_proxy_evals.py \
  experiments/domain_phase_mix/launch_300m_mcq_smooth_proxy_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/build_eval_signal_to_noise_all_metrics_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_raw_metric_matrix_300m.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_proportional_variable_subset_noise_matrices.py
```

## Launch

Initial submission hit a stale local Iris tunnel on port `10000`; the remote
controller was healthy after a fresh tunnel check. The local submit process was
stopped and resubmitted.

Submitted parent:

```text
/calvinxu/dm-propvar-noise-20260508-053335
```

Submitted command:

```bash
.venv/bin/iris --cluster marin job run \
  --no-wait \
  --enable-extra-resources \
  --cpu 1 \
  --memory 16GB \
  --disk 20GB \
  --region us-east5 \
  --zone us-east5-a \
  --job-name "dm-propvar-noise-20260508-053335" \
  -- python experiments/domain_phase_mix/launch_proportional_variable_subset_noise_baseline.py \
    --scales 60m_1p2b,300m_6b \
    --tpu-type v5p-8 \
    --tpu-region us-east5 \
    --tpu-zone us-east5-a \
    --max-concurrent 20
```

Post-submit controller query:

```text
state=1 pending: 20 child jobs
state=3 running: 1 parent job
```

Children materialized:

- `train_lm_propvar_60m_1p2b_trainer_seed_10000..10009`
- `train_lm_propvar_300m_6b_trainer_seed_10000..10009`

## Post-Training Eval Flow

After the registry sees all 20 target-ready rows:

```bash
.venv/bin/python experiments/domain_phase_mix/build_proportional_variable_subset_noise_eval_candidates.py
```

Then use the generated CSV as the extra candidate source for the downstream
eval launchers:

```bash
export MARIN_EXTRA_EVAL_CANDIDATES_CSVS=experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/proportional_variable_subset_noise/proportional_variable_subset_eval_candidates.csv
export MARIN_300M_CANDIDATE_PANELS=proportional_variable_subset_noise_60m_1p2b,proportional_variable_subset_noise_300m_6b
```

Run the same matrix-scope eval families: GSM8K/HumanEval hard evals,
English-lite, teacher-forced GSM8K/HumanEval smooth proxies, MCQ smooth
proxies, MMLU/parity aliases, and agentic-coding BPB. Use `--allow-partial` on
agentic-coding if the launcher expects the full 262-row matrix population.

Finally rebuild proportional-noise exports:

```bash
.venv/bin/python experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_proportional_variable_subset_noise_matrices.py
```

## 2026-05-19 00:36 PT - 300M Recovery And Full Eval Completion

Goal: dogfood Fieldbook while recovering the proportional-anchor 300M
variable-subset noise baseline and ensuring every current metric-matrix eval
family can be populated for the 10 noise rows.

Fieldbook experiment:

```text
exp_01krzjmttdggstv0s0qhk29pkj
```

Audit:

- Local `run_registry/logical_runs.csv` was stale and showed one 300M seed as
  failed plus nine as running.
- Direct GCS checks found final east5 checkpoints for all sampled 300M rows:
  `hf/step-22887`, `checkpoints/step-22887`, `.executor_status=SUCCESS`, and
  final `checkpoints/eval_metrics.jsonl`.
- The existing proportional-noise matrix exports were empty because
  `metrics_wide.csv` had no proportional variable-subset rows.

Local fixes:

- `build_metric_registry.py` now includes
  `proportional_variable_subset_noise_60m_1p2b` and
  `proportional_variable_subset_noise_300m_6b` as checkpoint-backed metric
  sources and normalizes their metric-registry cohort to `seed_sweep`.
- `build_proportional_variable_subset_noise_eval_candidates.py` now treats rows
  as eval-ready when the exact final HF checkpoint and final eval metrics exist
  on GCS, even if the local registry status is stale.
- `launch_300m_raw_ppl_evals.py` now supports
  `MARIN_EXTRA_EVAL_CANDIDATES_CSVS` and `MARIN_300M_CANDIDATE_PANELS`, and
  avoids reading local raw-matrix CSVs for panels not requested.

Candidate manifest:

```text
gs://marin-us-east5/pinlin_calvin_xu/data_mixture/propvar300m_noise_baseline/20260519/proportional_variable_subset_eval_candidates_300m.csv
```

Dry-run results:

| Eval family | Launch rows |
| :--- | ---: |
| GSM8K/HumanEval hard | 10 |
| English-lite | 10 |
| Teacher-forced GSM8K/HumanEval smooth | 10 |
| MCQ smooth proxies | 10 |
| Noise parity aliases | 10 |
| Agentic coding BPB | 10 |
| Raw-PPL priority + FineWeb2 representative | 10 |

The first live submission used the local candidate CSV path and all seven
parents failed before child dispatch. The corrected submission uses the GCS
candidate URI.

Running parent jobs:

```text
/calvinxu/dm-propvar300m-gsm8k-humaneval-20260519-075739
/calvinxu/dm-propvar300m-english-lite-20260519-075739
/calvinxu/dm-propvar300m-gen-smooth-20260519-075739
/calvinxu/dm-propvar300m-mcq-smooth-20260519-075739
/calvinxu/dm-propvar300m-noise-parity-20260519-075739
/calvinxu/dm-propvar300m-agentic-bpb-20260519-075739
/calvinxu/dm-propvar300m-raw-ppl-20260519-075739
```

First status check after resubmission:

```text
State: running, failures=0, preemptions=0 for all seven parents.
```

Next action after parents finish:

1. Collect each eval family from the corresponding `name_prefix` GCS output.
2. Rebuild `metric_registry/build_metric_registry.py`.
3. Rebuild `metric_registry/build_raw_metric_matrix_300m.py`.
4. Rebuild `metric_registry/build_proportional_variable_subset_noise_matrices.py`.
5. Verify `noise_baseline_proportional_variable_subset_300m.csv` has 10 rows
   and no missing current matrix metric columns.

## 2026-05-19 15:14 PT - English-Lite Failed-Only Retry

Status refresh:

- Succeeded parents: GSM8K/HumanEval hard, teacher-forced generative smooth,
  and MCQ smooth proxies.
- Running parents: noise parity, agentic BPB, and raw-PPL. All are parent-level
  running with preemptions but no terminal failure.
- Failed parent: English-lite. Iris child listing shows exactly one failed
  child, `propvar_300m_6b_trainer_seed_10008`, with SIGSEGV / TPU port setup:
  `Failed to add port to server: No address added out of total 1 resolved for
  '[::]:8482'`.

Failed-only retry state:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_english_lite_completion/300m_english_lite_eval_retry_state_propvar300m_seed10008.csv
gs://marin-us-east5/pinlin_calvin_xu/data_mixture/propvar300m_noise_baseline/20260519/300m_english_lite_eval_retry_state_propvar300m_seed10008.csv
```

Dry-run with the GCS retry state prepared exactly one English-lite eval step
for `propvar_300m_6b_trainer_seed_10008` and `17` task aliases.

Submitted retry:

```text
/calvinxu/dm-propvar300m-english-lite-retry-20260519-221120
```

Startup check:

- Parent is running.
- Child listing shows one TPU eval child for seed `10008` plus one succeeded CPU
  dataset-cache child.
- The eval child is pending on TPU capacity, not rerunning the whole 10-row
  panel.

## 2026-05-19 17:09 PT - Retry Progress And Raw-PPL Failed-Only Retry

Status refresh:

- English-lite failed-only retry
  `/calvinxu/dm-propvar300m-english-lite-retry-20260519-221120` succeeded.
- Agentic BPB parent
  `/calvinxu/dm-propvar300m-agentic-bpb-20260519-075739` succeeded.
- Noise parity parent
  `/calvinxu/dm-propvar300m-noise-parity-20260519-075739` is still running
  with no terminal child failures.
- Raw-PPL parent
  `/calvinxu/dm-propvar300m-raw-ppl-20260519-075739` failed with exactly two
  failed children:
  - `propvar_300m_6b_trainer_seed_10000`: TPU SIGSEGV / port setup failure.
  - `propvar_300m_6b_trainer_seed_10002`: transient GCS/HF config load failure
    while reading the HF checkpoint. `config.json` exists at the expected
    east5 HF path.

Raw-PPL failed-only retry state:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_raw_ppl_completion/300m_raw_ppl_eval_retry_state_propvar300m_seeds10000_10002.csv
gs://marin-us-east5/pinlin_calvin_xu/data_mixture/propvar300m_noise_baseline/20260519/300m_raw_ppl_eval_retry_state_propvar300m_seeds10000_10002.csv
```

Dry-run with the GCS retry state prepared exactly `2` raw-PPL eval steps over
the two failed checkpoints and `55` raw-PPL datasets.

Submitted retry:

```text
/calvinxu/dm-propvar300m-raw-ppl-retry-20260520-000630
```

Startup check:

- Parent is running.
- Child listing shows exactly two TPU eval children:
  `propvar_300m_6b_trainer_seed_10000` and
  `propvar_300m_6b_trainer_seed_10002`.
- Both children are pending on TPU capacity; this is not rerunning the other
  eight successful raw-PPL rows.

## 2026-05-20 15:46 PT - 300M Eval Collection Complete

Final status refresh:

- Noise parity parent
  `/calvinxu/dm-propvar300m-noise-parity-20260519-075739` succeeded with
  `failures=0` and `preemptions=7`.
- Raw-PPL failed-only retry
  `/calvinxu/dm-propvar300m-raw-ppl-retry-20260520-000630` succeeded with
  `failures=0`.

Collected proportional-overlay CSVs:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_gsm8k_humaneval_completion/300m_gsm8k_humaneval_eval_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_english_lite_completion/300m_english_lite_eval_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_generative_smooth_proxy_completion/300m_generative_smooth_proxy_eval_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_mcq_smooth_proxy_completion/300m_mcq_smooth_proxy_eval_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_noise_parity_completion/300m_noise_parity_eval_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_agentic_coding_bpb/300m_agentic_coding_bpb_results_proportional_variable_subset_noise.csv
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/300m_raw_ppl_completion/300m_raw_ppl_eval_results_proportional_variable_subset_noise.csv
```

Collection validation:

| Eval family | Rows | Collection status |
| :--- | ---: | :--- |
| GSM8K/HumanEval hard | 10 | `collected=10` |
| English-lite | 10 | `collected=10` |
| Teacher-forced GSM8K/HumanEval smooth | 10 | `collected=10` |
| MCQ smooth proxies | 10 | `collected=10` |
| Noise parity aliases | 10 | `collected=10` |
| Agentic-coding BPB | 10 | `collected=10` |
| Raw-PPL priority + FineWeb2 representative | 10 | `collected=10` |

Registry/matrix notes:

- The all-source `build_metric_registry.py` rebuild stalled on unrelated GCS
  reads. I stopped it and performed a targeted hydration of only
  `proportional_variable_subset_noise_60m_1p2b` and
  `proportional_variable_subset_noise_300m_6b` into `metrics_wide.csv`.
- `build_eval_signal_to_noise_all_metrics_300m.py` now includes the
  proportional-overlay CSVs in its default overlays.
- `build_proportional_variable_subset_noise_matrices.py` normalizes the
  proportional seed-sweep rows to `status=completed`; the local checkpoint/eval
  audit is fresher than the stale logical registry statuses.

Final 300M proportional-noise export:

```text
experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/raw_metric_matrix_300m/noise_baseline_proportional_variable_subset_300m_6b.csv
```

Validation:

- Rows: `10`.
- Metric columns: `1,386`.
- Missing metric values: `0`.
- Signal metric columns absent or all-missing in proportional noise: `0`.
- Row kind: `noise_variable_subset_proportional`.
- Status: `completed` for all rows.

Follow-up fix on 2026-05-21:

- The full canonical raw-matrix rebuild blocker was fixed. The cause was a
  local-only metric-registry rebuild that saw `baseline_stratified` only through
  local eval overlays, which do not carry `run_id`.
- `build_metric_registry.py` now fills stable static baseline metadata for
  `baseline_stratified -> run_id=3` and preserves proportional noise rows as
  provenance-only rows during local-only registry rebuilds.
- `build_proportional_variable_subset_noise_matrices.py` now directly hydrates
  final-step training `eval/*` metrics from checkpoint `eval_metrics.jsonl`
  files, so the 60M proportional-noise export remains complete even without a
  full GCS registry hydration.
- `build_raw_metric_matrix_300m.py` now completes: canonical signal rows `242`,
  `with_noise` rows `272`, and `with_proportional_noise` rows `252`.
