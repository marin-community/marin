# Exp 146 Adaptive Sweep Policy

Operator-approved policy for the contacts-v1 **3B** LR / weight-decay /
global-batch sweep across an epoch resource ladder
([MarinFold #146](https://github.com/Open-Athena/MarinFold/issues/146)).
Consumed by `run-adaptive-sweep`; trials are driven by
`experiments/protein/exp146_sweep.py`, one
`(epochs, lr, wd, batch_size, tpu, region)` point per launch.

## Required Inputs

```yaml
experiment:
  training_script: experiments/protein/exp146_sweep.py
  # BATCH_SIZE is trial identity; TPU is placement only. REGION sets regional
  # W&B and checkpoint identity. Same-region relocation resumes; a region
  # change restarts the trial.
  single_job_command: >
    EPOCHS={epochs} LR={learning_rate} WD={weight_decay}
    BATCH_SIZE={batch_size} TPU={tpu_slice} REGION={region}
    uv run python -m experiments.protein.exp146_sweep
  objective:
    # Full W&B metric key, comparable across every trial within a rung.
    wandb_metric: eval/tokenized/contacts-v1-val/loss
    # Value recorded at the final training step of the run.
    observation: final_step
    direction: minimize

search:
  grid:
    learning_rate:
      # Three half-decade (sqrt(10)) values. Edges may expand toward the hard
      # bounds at the same half-decade resolution.
      values: [3.1623e-4, 1.0e-3, 3.1623e-3]
      scale: log10
      preferred_max_gap: 0.5
      domain: {min: 3.1623e-5, max: 1.0e-2}
    weight_decay:
      # Five log-spaced values in x2 increments.
      values: [0.1, 0.2, 0.4, 0.8, 1.6]
      scale: log10
      preferred_max_gap: 0.31
      # Hard bounds approximately 1.5 x2-steps beyond each initial edge.
      domain: {min: 0.025, max: 6.4}
    batch_size:
      # Joint search axis: do not rescale LR or weight decay outside the grid.
      # The low edge may extend to 32 if smaller batches keep winning.
      values: [64, 128, 256]
      scale: log2
      preferred_max_gap: 1.0
      domain: {min: 32, max: 1024}
  resource_ladder:
    name: epochs
    # Each rung holds corpus epochs fixed across batch sizes. At 8 epochs,
    # batch 64/128/256 uses 71,360/35,680/17,840 steps, respectively, and
    # approximately 37.4B tokens in every case.
    levels: [2, 4, 8]
    resource_ratios: [1, 2, 4]

execution:
  state_db: scratch/exp146-adaptive-sweep-s01.sqlite
  # Hard elapsed sweep limit, including queueing and retries.
  wall_time: 8 weeks
  # Maximum requested TPU chips across submitted, running, or retrying dispatches.
  max_inflight_chips: 2048
  observation_interval: 15m
  full_exploitation_level: 8
  recovery:
    # TRC batch capacity is preemptible; long no-progress gaps are normal.
    startup_relocation_timeout: 3h
    same_target_restart_timeout: 6h
    same_region_relocation_timeout: 24h
    cross_region_restart_timeout: 96h
```

## Execution Preferences

The trainer uses data and tensor parallelism, so **any allowed slice is feasible
for any batch size**. A batch smaller than the slice runs with a model
(tensor-parallel) axis; do not filter targets by global batch size. `v5p-N`
names count cores and have `N/2` chips, so the smallest allowed v5p slice is
`v5p-32` (16 chips). `v6e-N` and `v5litepod-N` count chips and start at
`v6e-16` and `v5litepod-16`.

All allowed TPU families are peers: rank them by current observed throughput
and availability, with no family-level preference. TPU placement is an
execution detail, not trial identity.

The contacts-v1 raw documents and bucket must be region-local. The data is
available in us-east5, us-east1, us-central1, us-west4, and europe-west4
(`gs://marin-eu-west4`). Large-slice capacity is not guaranteed.

Region preference exploits recent-best throughput but is conditional on
availability. If work queues or stalls across an entire preferred region,
re-explore by spreading lower-value but still-useful trials across every
allowed region not already saturated with running trials. Re-map current
capacity instead of piling onto the preferred pool. A stale availability read
is missing evidence, not proof of unavailability. Resume exploitation of
proven regions after capacity is re-mapped; every trial must still earn its
place in the grid.

```yaml
targets:
  allow:
    - region: us-east5
      tpu_slices: ["v5p-{32,64,128,256,512,1024,2048}", "v6e-{16,32,64,128,256}"]
    - region: us-east1
      tpu_slices: ["v6e-{16,32,64,128,256}"]
    - region: us-central1
      tpu_slices: ["v5p-{32,64,128,256,512,1024,2048}"]
    - region: us-west4
      tpu_slices: ["v5litepod-{16,32,64,128,256}"]
    - region: europe-west4
      tpu_slices: ["v6e-{16,32,64,128,256}", "v5litepod-{16,32,64,128,256}"]
  block: {regions: [], tpu_slices: []}
```

## Operator Directives

- Append `--user "$USERNAME"` to every Iris job submission and resubmission.
- Show me an assembled Iris job-run command and ask for review before the first
  submission.
- **NEVER parse W&B run IDs to recover run metadata. ALWAYS read epochs, LR,
  weight decay, batch size, region, placement, and other metadata from W&B
  tags/config or another structured dispatch record. Treat run IDs as opaque
  identity keys only.**

## Reviewed Assumptions

- Data uses hierarchical Feistel block shuffle with `data_seed=0`.
- Batch size changes optimizer-step count, warmup/decay cadence, and cumulative
  AdamW decay at fixed corpus epochs. Learning rate, weight decay, and batch
  size are therefore evaluated jointly.
- Grid edges may expand within the declared hard domains. Preserve the preferred
  transformed spacing unless evidence justifies and records a different step.
- Training is deterministic; duplicate logical trials accomplish nothing.
- A SIGSEGV on a multi-host slice is treated as a preempted gang cosibling:
  retry in place, not as a code fault to investigate, absent a specific reason.
- **Liveness = the W&B run `state` (favored default).** `state=running` iff the
  trial is training; count/report "active" only from `state=running`.
  `crashed`/`failed`/`finished` mean NOT active — investigate and recover; do not
  casually reclassify a `crashed` run as a transient "flap." In Iris, parent and
  child `running` states are only scheduling gates; neither shows that training
  has started. Only the child's individual tasks running as a complete
  coscheduled gang provide meaningful Iris-side evidence. Use `iris job summary`
  to drill from parent to child to tasks when a specific run genuinely needs
  deeper debugging; never infer liveness from parent or child job state, and
  favor W&B over reaching for the task-level gang view.
- **Heartbeats report two placement spans:** chips, regions, and slices (a)
  **submitted** to Iris in any state and (b) **running per W&B**
  (`state=running`).
- **Iris job name = `<wandb_run_id>-<slice>-<unique>`; every submission is
  unique and there is no in-place "resubmit".** Any same-target restart,
  relocation, or slice/region change stops the old Iris job and submits a new
  uniquely named job. Resume comes from the region checkpoint, not the job
  name.
- **Invariant: at most one active dispatch per
  `(epochs, lr, wd, batch_size, region)`.** Otherwise two jobs can co-write the
  regional checkpoint and corrupt it.
- A same-region relocation retains run identity and checkpoint progress. A
  cross-region restart creates a new regional run and begins from zero; no
  checkpoint data moves between regions.
- Terminal failures retry immediately. Stall-based same-target restarts and
  relocations follow the recovery thresholds above.
