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

