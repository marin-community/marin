# Exp 117 Adaptive Sweep Policy

Operator-approved policy for the contacts-v1 **1.5B** LR / weight-decay / global-batch sweep across an
epoch resource ladder ([MarinFold #117](https://github.com/Open-Athena/MarinFold/issues/117)).
Consumed by `run-adaptive-sweep`; trials are driven by
`experiments/protein/exp117_sweep.py`, one `(epochs, lr, wd, batch_size, tpu, region)` point per launch.

Scope is the Qwen3 1.47B recipe. The trainer fixes `MODEL_CONFIG` and varies
`(epochs, learning_rate, weight_decay, batch_size)`.

## Required Inputs

```yaml
experiment:
  training_script: experiments/protein/exp117_sweep.py
  # BATCH_SIZE is trial identity; TPU is placement only. REGION sets regional W&B and checkpoint
  # identity. Same-region relocation resumes; region changes restart the trial.
  single_job_command: >
    EPOCHS={epochs} LR={learning_rate} WD={weight_decay} BATCH_SIZE={batch_size}
    TPU={tpu_slice} REGION={region}
    uv run python -m experiments.protein.exp117_sweep
  objective:
    # Full W&B metric key, comparable across every trial within a rung. Carries the `tokenized/`
    # prefix because the component key is the tokenize handle name.
    wandb_metric: eval/tokenized/contacts-v1-val/loss
    # Value recorded at the final training step of the run.
    observation: final_step
    direction: minimize

search:
  grid:
    learning_rate:
      # Four half-decade (sqrt(10)) values spanning 1.5 decades up to the hard upper bound. The top
      # value is the prior sweep's best point (final loss ~2.73, a top-edge optimum); the extra low
      # anchor characterizes the low-LR stability gradient. domain.max == the top grid value, so no
      # upward extension is intended: the top edge converges against its hard bound.
      values: [3.1623e-4, 1.0e-3, 3.1623e-3, 1.0e-2]
      scale: log10
      # Half-decade spacing.
      preferred_max_gap: 0.5
      # Hard bounds; the low edge may extend one half-decade toward the floor if evidence supports it.
      domain: {min: 3.1623e-5, max: 1.0e-2}
    weight_decay:
      # Five log-spaced values, x2 increments.
      values: [0.1, 0.2, 0.4, 0.8, 1.6]
      scale: log10
      # Preferred largest transformed grid gap. Just above the x2 (0.30103) spacing so the initial
      # points do not read as under-resolved.
      preferred_max_gap: 0.31
      # Hard search bounds (~1.5 x2-steps beyond each edge); edges may expand toward these at x2.
      domain: {min: 0.025, max: 6.4}
    batch_size:
      # Joint search axis: smaller batches take more optimizer steps at fixed epochs; do not
      # rescale LR or weight decay outside the grid. The low edge (64) may extend one step
      # toward 32 if smaller batch keeps winning — best single point so far is bs64 and the
      # best-of-batch envelope favors smaller.
      values: [64, 128, 256]
      scale: log2
      preferred_max_gap: 1.0
      domain: {min: 32, max: 1024}
  resource_ladder:
    name: epochs
    # Each rung holds corpus epochs fixed across batch sizes. At 8 epochs, batch 64/128/256 uses
    # 71,360/35,680/17,840 steps, respectively, and approximately 37.4B tokens in every case.
    levels: [8, 16, 32]
    # Expected work relative to the 8-epoch rung; every batch-size point doubles with each rung.
    resource_ratios: [1, 2, 4]

execution:
  # Local orchestration record for this sweep.
  state_db: scratch/exp117-adaptive-sweep-s02.sqlite
  # Hard elapsed sweep limit, including queueing and retries.
  wall_time: 8 weeks
  # Maximum requested TPU chips across submitted, running, or retrying dispatches.
  max_inflight_chips: 2048
  # Dispatcher cadence for polling Iris, logs, and W&B. Throughput is recomputed on each observation.
  observation_interval: 15m
  # Resource level at which placement uses only the best currently observed target. Earlier rungs
  # admit progressively more lower-ranked or untried targets. Set to the highest rung.
  full_exploitation_level: 32
  recovery:
    # TRC batch capacity is preemptible; long no-progress gaps are normal, so relocation gates are loose.
    # Relocate startup within its region when no W&B run appears by this time.
    startup_relocation_timeout: 3h
    # A W&B-registered run still running in Iris may restart on the same target after this stall;
    # it resumes its checkpoint but releases TPU capacity and may requeue. Terminal failures retry immediately.
    same_target_restart_timeout: 6h
    # Relocate a W&B-registered run within its region after this long without progress.
    same_region_relocation_timeout: 24h
    # Restart elsewhere only after a same-region relocation also fails to progress.
    cross_region_restart_timeout: 96h
```

## Execution Preferences

The trainer uses data + tensor parallelism, so **any allowed slice is feasible for any batch** — a
batch too small to fill a slice runs with a model (tensor-parallel) axis. `v5p-N` names count cores
and have `N/2` chips.

The contacts-v1 raw docs and bucket must be region-local. Verified 2026-07-14 in us-east5, us-east1,
us-central1, us-west4, and europe-west4 (`gs://marin-eu-west4`). Large-slice capacity is not guaranteed.

```yaml
placement:
  # Any slice fits any batch; the trainer derives the data/tensor-parallel split (see runbook).
targets:
  allow:
    - region: us-east5
      tpu_slices: ["v5p-{16,32,64,128,256,512,1024,2048}", "v6e-{8,16,32,64,128,256}"]
    - region: us-east1
      tpu_slices: ["v6e-{8,16,32,64,128,256}"]
    - region: us-central1
      tpu_slices: ["v5p-{16,32,64,128,256,512,1024,2048}"]
    - region: us-west4
      tpu_slices: ["v5litepod-{32,64,128,256}"]
    - region: europe-west4
      tpu_slices: ["v6e-{8,16,32,64,128,256}", "v5litepod-{32,64,128,256}"]
  block: {regions: [], tpu_slices: []}
```

## Operator Directives

- Append `--user "$USERNAME"` to every Iris job submission.
- Show me an assembled Iris job run command and ask for review before the first job submission.

## Reviewed Assumptions

- Data uses hierarchical Feistel block shuffle with `data_seed=0`.
- Batch size changes optimizer-step count, warmup/decay cadence, and cumulative AdamW decay at fixed
  corpus epochs. Learning rate, weight decay, and batch size are therefore evaluated jointly.
- The initial grid has 60 configurations per rung (4 LR x 5 WD x 3 batch) and 180 logical trials
  across three rungs. An exhaustive initial-grid run costs 420 eight-epoch equivalents before any
  grid expansion.
- A SIGSEGV on a multi-host slice (nearly all slices here) is treated as a preempted gang cosibling
  — retry in place, not a code fault to investigate — absent a specific reason.
