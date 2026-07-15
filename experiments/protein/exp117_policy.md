# Exp 117 Adaptive Sweep Policy

Operator-approved policy for the contacts-v1 **1.5B** LR / weight-decay tuning sweep across an
epoch resource ladder ([MarinFold #117](https://github.com/Open-Athena/MarinFold/issues/117),
the multi-epoch extension of #75). Consumed by `run-adaptive-sweep`; trials are driven by
`experiments/protein/exp117_sweep.py`, one `(epochs, lr, wd, tpu, region)` point per launch.

Scope is the 1.5B / global-batch-128 recipe only: the trainer fixes `MODEL_CONFIG` (Qwen3 1.47B)
and `BATCH_SIZE=128`, and varies only `(epochs, lr, wd)`. The #117 follow-ups (3B model, global
batch {64,128,256}) are **out of scope** here — they need separate trainer entry points / policies.

## Required Inputs

```yaml
experiment:
  training_script: experiments/protein/exp117_sweep.py
  # TPU selects adaptive batch sizing (global batch stays 128 on every slice); REGION sets the
  # regional W&B + checkpoint identity. A same-region slice change resumes the same run; a region
  # change starts a fresh run under a fresh regional identity.
  single_job_command: >
    EPOCHS={epochs} LR={learning_rate} WD={weight_decay} TPU={tpu_slice} REGION={region}
    uv run python -m experiments.protein.exp117_sweep
  objective:
    # Full W&B metric key, comparable across every trial within a rung. Carries the `tokenized/`
    # prefix because the component key is the tokenize handle name (see Operator Directives).
    wandb_metric: eval/tokenized/contacts-v1-val/loss
    # Value recorded at the final training step of the run.
    observation: final_step
    direction: minimize

search:
  grid:
    learning_rate:
      # Three log-spaced values, exactly 1e-3 -> 1e-2, constant ratio sqrt(10).
      values: [1.0e-3, 3.1623e-3, 1.0e-2]
      scale: log10
      # Preferred largest transformed grid gap. 0.5 == the sqrt(10) (half-decade) spacing above.
      preferred_max_gap: 0.5
      # Hard search bounds (~1.5 decades below / one above the initial edges) for the #117
      # LR-boundary-extension analysis; current edges may expand toward these at half-decade steps.
      domain: {min: 3.1623e-5, max: 1.0e-1}
    weight_decay:
      # Five log-spaced values, x2 increments.
      values: [0.1, 0.2, 0.4, 0.8, 1.6]
      scale: log10
      # Preferred largest transformed grid gap. Just above the x2 (0.30103) spacing so the initial
      # points do not read as under-resolved.
      preferred_max_gap: 0.31
      # Hard search bounds (~1.5 x2-steps beyond each edge); edges may expand toward these at x2.
      domain: {min: 0.025, max: 6.4}
  resource_ladder:
    name: epochs
    # Ordered rungs; each must converge. 8ep = 35,680 steps (37.4B tok); 16ep = 71,360 (74.8B);
    # 32ep = 142,720 (149.7B). steps/epoch = 4460 (from the exact train-token count).
    levels: [8, 16, 32]
    # Expected work relative to the 8-epoch rung: steps scale linearly with epochs at fixed
    # per-step cost, so cost does too.
    resource_ratios: [1, 2, 4]

execution:
  # Local orchestration record for this sweep.
  state_db: scratch/exp117-adaptive-sweep.sqlite
  # Hard elapsed sweep limit, including queueing and retries.
  wall_time: 8 weeks
  # Maximum requested TPU chips across submitted, running, or retrying dispatches.
  max_inflight_chips: 256
  # Dispatcher cadence for polling Iris, logs, and W&B. Throughput is recomputed on each observation.
  observation_interval: 15m
  # Resource level at which placement uses only the best currently observed target. Earlier rungs
  # admit progressively more lower-ranked or untried targets. Set to the highest rung.
  full_exploitation_level: 32
  stagnation:
    # All TPU capacity here is preemptible (TRC batch); 12h+ no-progress stretches are normal, so
    # these are deliberately loose — a preemption gap must not read as a failure worth relocating.
    # Initial execution may move within its region when no W&B run appears by this time.
    initial_wandb_timeout: 3h
    # A W&B-registered run may move within its region after this long without progress.
    progress_stall_timeout: 24h
    # A stalled run may restart elsewhere only after a same-region move also fails to progress.
    cross_region_restart_timeout: 96h
```

## Execution Preferences

Targets are the issue's region/slice plan, confirmed against ground truth: the contacts-v1 raw docs
must be present **region-local** (the trainer tokenizes in-pipeline region-local) and a
`gs://marin-<region>` bucket must exist. Verified 2026-07-14 — docs present in all five regions:
us-east5, us-east1, us-central1, us-west4, and europe-west4 (bucket `gs://marin-eu-west4`, note the
`eu-` abbreviation).

```yaml
targets:
  allow:
    - region: us-east5
      tpu_slices: ["v5p-{16,32,64,128,256}", "v6e-{8,16,32,64,128}"]
    - region: us-east1
      tpu_slices: ["v6e-{8,16,32,64,128}"]
    - region: us-central1
      tpu_slices: ["v5p-{16,32,64,128,256}"]
    - region: us-west4
      tpu_slices: ["v5litepod-{32,64,128}"]
    - region: europe-west4
      tpu_slices: ["v6e-{8,16,32,64,128}", "v5litepod-{32,64,128}"]
  block: {regions: [], tpu_slices: []}
```

## Operator Directives

- Append `--user "$USERNAME"` to every Iris job submission and resubmission.
- Show me an assembled Iris job-run command and ask for review before the first job submission.
