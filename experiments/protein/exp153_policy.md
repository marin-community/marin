# Exp 153 Adaptive Sweep Policy

Operator-approved policy for the contacts-v1 **6B** LR / weight-decay / global-batch
sweep across an epoch resource ladder
([MarinFold #153](https://github.com/Open-Athena/MarinFold/issues/153), under the
parameter-scaling parent [#154](https://github.com/Open-Athena/MarinFold/issues/154)).
Consumed by `run-adaptive-sweep`; trials are driven by
`experiments/protein/exp153_sweep.py`, one
`(epochs, lr, wd, batch_size, cluster, nodes)` point per launch.

This sweep runs on **CoreWeave GPUs**, not TRC TPUs. The two differences that matter to
the harness: every CoreWeave cluster reads the same object-storage bucket, so a trial can
move between clusters and resume its checkpoint; and capacity is far more stable than
TRC's preemptible pools, so the recovery clocks are much longer.

## Required Inputs

```yaml
experiment:
  training_script: experiments/protein/exp153_sweep.py
  # BATCH_SIZE is trial identity; CLUSTER and NODES are placement only. There is no
  # regional identity: any re-dispatch resumes the same run from the same checkpoint.
  single_job_command: >
    EPOCHS={epochs} LR={learning_rate} WD={weight_decay}
    BATCH_SIZE={batch_size} CLUSTER={cluster} NODES={nodes}
    uv run python -m experiments.protein.exp153_sweep
  objective:
    # Full W&B metric key, comparable across every trial within a rung, and identical to
    # the #117 / #146 rungs of the same scaling ladder.
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
      values: [64, 128, 256]
      scale: log2
      preferred_max_gap: 1.0
      domain: {min: 32, max: 1024}
  resource_ladder:
    name: epochs
    # Each rung holds corpus epochs fixed across batch sizes. At 8 epochs, batch
    # 64/128/256 uses 71,360/35,680/17,840 steps respectively, and approximately
    # 37.4B tokens in every case.
    levels: [2, 4, 8]
    resource_ratios: [1, 2, 4]

execution:
  state_db: scratch/exp153-adaptive-sweep-s01.sqlite
  # Hard elapsed sweep limit, including queueing and retries.
  wall_time: 8 weeks
  # Maximum requested GPU NODES across submitted, running, or retrying dispatches.
  # Nodes, not GPUs: a node is the atomic schedulable unit and its GPU count differs by
  # cluster (8 on H100, 4 on GB200). Set high deliberately -- there is no real quota to
  # enforce here, and a low cap was proving more confusing than useful. Actual usage stays
  # far below this while the sweep eases in; batch priority is what protects other users.
  max_inflight_nodes: 256
  observation_interval: 15m
  full_exploitation_level: 8
  recovery:
    # CoreWeave batch capacity is preemptible but far steadier than TRC, so a
    # no-progress gap is evidence of a real fault rather than routine scarcity.
    # These are correspondingly long: prefer waiting over churning a gang.
    startup_relocation_timeout: 6h
    same_target_restart_timeout: 12h
    relocation_timeout: 48h
```

There is no cross-region restart. On TRC a region change meant abandoning a regional run
and starting from zero, because checkpoints never moved between regions. Here every
cluster reads `s3://marin-us-east-02a`, so relocation always retains run identity and
checkpoint progress, and the concept does not apply.

## Execution Preferences

The trainer uses data and tensor parallelism, so **any allowed gang is feasible for any
batch size**, subject to one hardware rule the script enforces: the tensor-parallel axis
must stay inside a single node's NVLink domain. `gpu_batch_config` rejects a gang where
the implied model axis would span the inter-node fabric; treat that rejection as "this
target is infeasible for this batch size", not as a code fault.

A gang is always **whole nodes**. Sub-node GPU requests do schedule on CoreWeave but
fragment the 8-GPU InfiniBand pool, so they are never used.

Node count is the wall-clock knob and nothing else: the global batch is fixed by the
trial, so steps, LR schedule, and objective are identical at any node count. More nodes
buy speed and reduce gradient accumulation; they do not change trial semantics.

Rank targets by observed throughput and availability. Cluster placement is an execution
detail, not trial identity, and a trial may be relocated across clusters freely.

```yaml
targets:
  allow:
    # H100: 8 GPUs/node, 80 GB each. The proven path (MarinFold #108) and bucket-local.
    - cluster: cw-us-east-02a
      nodes: [1, 2, 4, 8]
    # GB200: 4 GPUs/node, 186 GB each. ENABLE ONLY after a GB200 smoke run passes —
    # levanter dense-LM training has never run on Blackwell in this repo.
    - cluster: cw-us-east-08a
      nodes: []
  block:
    # Data-processing cluster, near-fully subscribed and remote from the bucket.
    clusters: [cw-rno2a]
```

Node counts are powers of two. A gang of 3 is legal but yields a degenerate mesh — at batch
128 the device count 24 forces `data=8, tensor=3` instead of pure data parallelism — so the
ladder skips non-powers of two.

## Storage Layout

`MARIN_PREFIX` is set by the sweep script to
`s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1` on both driver and gang; the
gang does not inherit the driver's environment, so it is forwarded explicitly. Everything
the experiment writes therefore lands under one MarinFold-prefixed directory, per #108's
standing rule. The directory is named for the parent scaling sweep (#154); the runs inside
it are exp153.

| What | Path |
| --- | --- |
| Raw documents | `s3://marin-us-east-02a/MarinFold/data/document_structures/contacts_v1/{train,val}` |
| Token caches | `<prefix>/tokenized/contacts-v1{,-val}/2026.07.25/` |
| Run outputs | `<prefix>/checkpoints/<run_id>/2026.07.25.01/{checkpoints,hf}` |
| W&B | `eric-czech/marin`, group `exp153-contacts-v1-6b-tune` |

Token caches are validated byte-for-byte against #117: train 4,676,753,425 tokens /
4,129,682 docs; val 47,821,958 / 41,954. `TRAIN_TOKENS` in the sweep script is the train
figure and steps/epoch derive from it. (MarinFold #108's 4,672,623,743 came from raw
generation stats and is not the tokenized count.)

## Environment Notes

Facts about this cluster family that are not obvious and have already cost time:

- **`attn_backend=JAX_FLASH` is mandatory.** Levanter's GPU default is Transformer Engine,
  absent from the `gpu` extra, and it silently falls back to an O(seq^2) reference kernel
  that OOMs at seq 8192 (#108; upstream marin#7013 still open).
- **Kubernetes enforces memory requests as hard limits** — unlike the GCE VMs the TPU
  sweeps ran on. Anything sized for GCP gets OOM-killed (exit 137) here. Concretely: driver
  jobs need >=3 GB (>=4 GB requires `--enable-extra-resources`), and Zephyr's default
  coordinator is too small for the task image's startup `uv sync`, so the tokenize step
  passes `coordinator_resources` explicitly.
- **CPU-only work lands on the CPU pool, not GPU-node spare cores.** cw-us-east-02a has 4
  `cd-gp-a192-genoa` nodes (192 vCPU each) that also host the controller, hence the capped
  tokenize fan-out.
- A Zephyr worker pool reported `killed` / "Terminated by user" once its stage completes is
  normal teardown, not a fault.

## Measured Throughput (2026-07-25)

Batch 128 x seq 8192, steady state. Early steps are contaminated by a ~8 minute first-step
XLA compile and by tqdm's cumulative average, so every figure here is a delta between two
late step timestamps.

| nodes | GPUs | s/step | speedup | scaling efficiency |
| --- | --- | --- | --- | --- |
| 1 | 8 | 33.7 | 1.00x | — |
| 2 | 16 | 17.0 | 1.98x | 99% |
| 8 | 64 | 4.5 | 7.49x | 94% |

Scaling holds up well to 8 nodes. MarinFold #108's 8-node JAX coordination-bootstrap
failure does **not** reproduce; the `jax.distributed.initialize` timeout was raised to
1800s nine days after they hit it.

Projected full-trial wall clock at batch 128 on 8 nodes: **~45 h (1.9 days)** for the
8-epoch rung (35,680 steps), ~22 h at 4 epochs, ~11 h at 2 epochs. On 1 node the 8-epoch
rung is ~14 days, so node count is the wall-clock knob and nothing else — the global batch
is fixed by the trial, so steps, schedule and objective are identical at any gang size.

Eval costs ~7 minutes for the full 208-batch held-out pass. A checkpoint is ~96 GB (72 GB
Levanter state written sharded in ~15 s, plus a 24 GB HF export that takes ~2.5 min because
it gathers to a single host).

## Checkpoint Retention

Nothing under `MarinFold/` expires on its own: the bucket's only lifecycle rules cover
`tmp/ttl=Nd/` prefixes, and it has **no object versioning**, so every delete is permanent.
Retention is therefore an explicit step, never a background guarantee.

- **One HF export per run, at the final step.** The HF format is derived and regenerable
  from any Levanter checkpoint, so intermediate exports buy nothing; the sweep sets
  `hf_save_steps` past the run length, leaving the single end-of-run export.
- **Permanent checkpoints: final only, except the 8-epoch rung**, which keeps one per epoch
  because the R-precision-vs-tokens analysis (#117) needs the intermediate points. Cost per
  completed trial: ~0.09 TiB at the 2- and 4-epoch rungs, ~0.59 TiB at 8 epochs. The
  preview prints this figure for the point being launched.
- **Prune after a rung converges.** Drop all but the top-N runs per rung to final-only.
  Deleting is `aws s3 rm --recursive` against the run's version directory; always
  `--dryrun` first, read the enumeration, then verify the object-count and byte deltas
  afterwards. There is no undo.

Storing production checkpoints under `tmp/ttl=Nd/` and promoting winners later is **not
done**, deferred rather than ruled out. The storage side checks out — the bucket has enabled
30-day rules, and S3 reports a concrete expiry date and rule id for an object written under
`tmp/ttl=30d/` — but the code path that assigns a checkpoint's lifetime is not yet
understood well enough to stake real checkpoints on it:

- `temporary_checkpoint_base_path()` embeds the run's whole output path inside a *second*
  temp prefix, so pointing output at temp storage yields nested TTLs
  (`tmp/ttl=14d/checkpoints-temp/<bucket>/tmp/ttl=30d/...`). S3 matches the leftmost
  prefix, so those objects expire on the outer rule, not the one the run asked for.
- `TEMPORARY_CHECKPOINT_TTL_DAYS` is hardcoded to 14 in marin, and whether it clamps
  depends on a config that resolves differently than expected inside a CoreWeave pod
  (`MARIN_CLUSTER` is unset there, so `config/marin.yaml` governs, not
  `config/coreweave.yaml`).
- Setting `temporary_base_path=None` on the checkpointer does not survive:
  `resolve_checkpointer_output_path` reassigns it downstream.

Until those three are traced end to end, production checkpoints stay permanent under
`STORAGE_PREFIX` and are managed by explicit pruning. Only smoke and calibration runs use
temp storage, where a wrong lifetime costs nothing.

## Operator Directives

- Append `--user "$USERNAME"` to every Iris job submission and resubmission.
- **Submit every job with `--priority batch`.** MarinFold #108 requires batch priority for
  all CoreWeave work. The band on the driver is sufficient for the whole tree: the
  scheduler resolves a child job's unspecified band by walking the parent chain. Do not
  submit at `interactive` and rely on demotion, as the TPU sweeps did.
- The driver runs **inside** the target cluster (`--target-cluster <cluster>`). Iris only
  federates root jobs, so a driver cannot hand a gang to a different cluster; cluster
  choice happens at submission and nowhere else.
- The driver must stay alive for the life of its gang. Gangs are children of the driver
  job; if the driver exits, Iris finalizes (kills) them.
- Show me an assembled Iris job-run command and ask for review before the first submission.
- **NEVER parse W&B run IDs to recover run metadata. ALWAYS read epochs, LR, weight decay,
  batch size, placement, and other metadata from W&B tags/config or another structured
  dispatch record. Treat run IDs as opaque identity keys only.**

## Reviewed Assumptions

- Data uses hierarchical Feistel block shuffle with `data_seed=0`.
- Batch size changes optimizer-step count, warmup/decay cadence, and cumulative AdamW
  decay at fixed corpus epochs. Learning rate, weight decay, and batch size are therefore
  evaluated jointly.
- Grid edges may expand within the declared hard domains. Preserve the preferred
  transformed spacing unless evidence justifies and records a different step.
- Training is deterministic **at a fixed gang shape**; duplicate logical trials accomplish
  nothing. It is *not* bitwise reproducible across node counts: a different node count
  changes the data-axis width and the gradient-accumulation depth, so gradients are summed
  in a different order. Measured on the 2026-07-25 smokes, 1-node and 2-node runs of the
  same point diverge at step 2 (13.1391 vs 13.1310) and amplify chaotically through early
  training. The runs remain statistically equivalent and a relocated trial resumes its
  checkpoint correctly, but do not expect two placements of one trial to agree bitwise, and
  do not treat a small objective difference between placements as a defect.
- The architecture is fixed by #153 and is not a tuning axis.
- A SIGSEGV on a multi-node gang is treated as a preempted gang cosibling: retry in place,
  not as a code fault to investigate, absent a specific reason.
- **Liveness = the W&B run `state` (favored default).** `state=running` iff the trial is
  training; count/report "active" only from `state=running`. `crashed`/`failed`/`finished`
  mean NOT active — investigate and recover; do not casually reclassify a `crashed` run as
  a transient "flap." In Iris, parent and child `running` states are only scheduling gates;
  neither shows that training has started. Use `iris job summary` to drill from parent to
  child to tasks when a run genuinely needs deeper debugging; never infer liveness from
  parent or child job state, and favor W&B over reaching for the task-level gang view.
- A Zephyr worker pool reported as `killed` / "Terminated by user" after its stage
  completes is normal teardown, not a fault.
- **Heartbeats report two placement spans:** nodes, clusters, and gang shapes (a)
  **submitted** to Iris in any state and (b) **running per W&B** (`state=running`).
- **Iris job name = `<wandb_run_id>-<gang>-<unique>`; every submission is unique and there
  is no in-place "resubmit".** Any same-target restart or relocation stops the old Iris job
  and submits a new uniquely named one. Resume comes from the checkpoint, not the job name.
- **Invariant: at most one active dispatch per `(epochs, lr, wd, batch_size)`.** Otherwise
  two gangs can co-write the same checkpoint and corrupt it. This invariant is now global
  rather than per-region, and is the one hard constraint that cross-cluster relocation
  introduces: never leave the old gang running while starting the new one.
- Terminal failures retry immediately. Stall-based same-target restarts and relocations
  follow the recovery thresholds above.

## Commands

```bash
set -a; source ~/marin.env; set +a

# Preview a point. Builds and submits nothing; prints the resolved mesh and storage cost.
EPOCHS=8 LR=1e-3 WD=0.1 BATCH_SIZE=128 CLUSTER=cw-us-east-02a NODES=2 PREVIEW=yes \
  uv run python -m experiments.protein.exp153_sweep

# Tokenize once, before the first trial.
uv run iris --config lib/iris/config/marin.yaml job run --user "$USERNAME" \
  --target-cluster cw-us-east-02a --priority batch --memory 3GB --no-wait \
  -e TOKENIZE_ONLY yes -- python -m experiments.protein.exp153_sweep

# One trial. The driver runs inside the cluster; the gang inherits its batch band.
uv run iris --config lib/iris/config/marin.yaml job run --user "$USERNAME" \
  --target-cluster cw-us-east-02a --priority batch --memory 3GB --no-wait \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \
  -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
  -e EPOCHS 8 -e LR 1e-3 -e WD 0.1 -e BATCH_SIZE 128 \
  -e CLUSTER cw-us-east-02a -e NODES 2 -- python -m experiments.protein.exp153_sweep

# Browse or prune storage (see Checkpoint Retention before deleting).
scripts/ops/cw_s3.sh s3 ls s3://marin-us-east-02a/MarinFold/
```

Observability: `iris rpc controller list-peers` for live cluster load without kubectl;
`iris job logs <id>`; `iris job summary <id>` for per-task exit codes. Direct CoreWeave
controller URLs are IP-locked and return Forbidden from outside.
