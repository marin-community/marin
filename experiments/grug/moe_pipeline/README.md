# grug-moe-pipeline

Canonical pipeline-parallel Grug MoE implementation. This directory is a copy-paste
variant of [`experiments/grug/moe`](../moe/): its model and execution path are local
so pipeline-specific changes do not turn the ordinary MoE implementation into a
shared trainer framework.

[`pipeline.py`](./pipeline.py) splits the local transformer into stage pytrees and
builds automatic JaxPP ZeroBubble or DualPipeV optimizer steps. [`train.py`](./train.py)
owns model initialization, the training loop, and the Fray dispatch entry point.
[`benchmark.py`](./benchmark.py) maps environment variables onto that trainer for
repeatable performance runs. One logical stage maps to each physical pipeline rank with
`PIPELINE_SCHEDULE=automatic_zero_bubble`. Set `PIPELINE_SCHEDULE=automatic_dualpipe_v`,
`PIPELINE_STAGES=2P`, and `PIPELINE_PHYSICAL_STAGES=P` to fold two logical stages onto
each physical rank in JaxPP's V-shaped placement. DualPipeV requires at least `2P`
microbatches. `PIPELINE_LAYERS_PER_STAGE` contains one positive layer count per logical
stage.

The canonical loop uses a fixed AdamW optimizer and synthetic token rows. Set
`checkpoint_root` on `GrugPipelineTrainConfig` to resume the latest completed
checkpoint in that root, or initialize fresh parameters if none exists. Set
`checkpoint_every_steps` to a positive interval to save periodically and on clean
completion. An interval of zero disables saves but still permits restore. The
trainer rejects a positive interval without a checkpoint root. The
benchmark exposes these settings as `PIPELINE_CHECKPOINT_ROOT` and
`PIPELINE_CHECKPOINT_EVERY_STEPS`. `steps` is the total target number of optimizer
updates, including updates completed before restore.

Checkpoints work with both automatic schedules. Each save creates
`step-<12-digit-completed-step>-<unique-id>/` containing Levanter's TensorStore
Zarr3/OCDBT arrays and array manifest. `metadata.json` records the completed step,
model and optimizer settings, and array shapes and placements. It is written only
after every process commits its shards; directories without this marker are
ignored during restore. An atomically published `latest.json` points to the
committed checkpoint, so normal resumes do not list historical checkpoints.
An existing pointer is authoritative: interruption before publishing a newer
checkpoint leaves the previous checkpoint selected.
If the first pointer write was interrupted, restore discovers committed metadata
in the root. A committed checkpoint with incompatible metadata or
missing array data raises an error. Checkpoints are retained until explicitly
deleted. Use one writer gang per checkpoint root and a shared filesystem or
object-store root accessible to every process.

The state contains each logical stage's parameters, Adam moments and integer
count, and pending router-bias updates. Checkpoint I/O exposes existing local
device buffers without gathering model or optimizer arrays. Restore uses abstract
shape/sharding templates, reads only addressable shards, and reconstructs the
compiled step's MPMD placement. Each
process must own exactly one physical pipeline stage. Restore requires the same
model, precision, batch and microbatch settings, schedule, logical stage split,
mesh dimensions, and process-to-stage assignment. Changing pipeline, expert,
replica, or data parallelism requires a separate conversion; this path does not
reshard checkpoints across topologies.

[`checkpoint_smoke.py`](./checkpoint_smoke.py) tests continuation across fresh
four-process H100x8 gangs with PP2, EP8, and replica axis two. Run `--phase save`
and then `--phase resume` with the same `--checkpoint-root` and `--schedule
zero_bubble` (or `dualpipe_v`). The first gang saves step one and an uninterrupted
step-two reference; the second restores step one and compares its next step.
Every integer leaf must match exactly, and every floating state leaf and loss
must have relative L2 error at most 0.002. Adam counters must also equal the
recorded completed step. Only local shards and scalar error
reductions are read during comparison. Disable command buffers in both gangs:
`XLA_FLAGS='--xla_gpu_executable_terminate_timeout=300 --xla_gpu_enable_command_buffer='`.

For example, submit through the Iris hub to an H100 peer. Choose an unused job
name and checkpoint root in the peer's regional storage:

```bash
uv run iris --cluster=marin job run --no-wait \
  --enable-extra-resources --target-cluster cw-rno2a --priority batch \
  --gpu H100x8 --replicas 4 --cpu 32 --memory 256GB --disk 64GB \
  --timeout 3600 --extra pipeline --job-name <unique-save-job> \
  -e IRIS_PORT_JAX 32761 -e XLA_PYTHON_CLIENT_PREALLOCATE false \
  -e XLA_FLAGS '--xla_gpu_executable_terminate_timeout=300 --xla_gpu_enable_command_buffer=' \
  -- python -m experiments.grug.moe_pipeline.checkpoint_smoke \
  --checkpoint-root <shared-regional-root> --schedule zero_bubble --phase save
```

After that job succeeds, repeat with a new job name and `--phase resume`, keeping
the checkpoint root and schedule unchanged. Use a distinct root for the
`dualpipe_v` pair. Concurrent gangs need distinct `IRIS_PORT_JAX` values. A
successful resume emits `CHECKPOINT_SMOKE_PASSED` with the maximum per-leaf error.

The best validated Snowball 67B-A2B throughput point uses eight H100x8 replicas,
sixteen logical stages, batch 256, 32 microbatches, sequence length 8192, and layer
counts `1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,1`:

```bash
export XLA_FLAGS='--xla_gpu_executable_terminate_timeout=300 --xla_gpu_enable_command_buffer='
export PIPELINE_SCHEDULE=automatic_dualpipe_v
export PIPELINE_PHYSICAL_STAGES=8
export PIPELINE_STAGES=16
export PIPELINE_BATCH=256
export PIPELINE_MICROBATCHES=32
export PIPELINE_SEQ_LEN=8192
export PIPELINE_HIDDEN_DIM=2560
export PIPELINE_INTERMEDIATE_DIM=1280
export PIPELINE_SHARED_EXPERT_INTERMEDIATE_DIM=2560
export PIPELINE_LAYERS=26
export PIPELINE_LAYERS_PER_STAGE=1,2,2,2,2,2,2,2,1,1,1,1,2,2,2,1
export PIPELINE_EXPERTS=256
export PIPELINE_TOP_K=4
export PIPELINE_EXPERT_AXIS=8
export PIPELINE_HEADS=20
export PIPELINE_KV_HEADS=5
export PIPELINE_VOCAB_SIZE=128256
export PIPELINE_SLIDING_WINDOW=2048
export PIPELINE_QK_MULT=1.5703
export PIPELINE_MP=params=bfloat16,compute=bfloat16,output=bfloat16
export PIPELINE_ATTENTION=gpu_fa4_cute
export PIPELINE_MOE=ring
export PIPELINE_REMAT=recompute_all
uv run --extra pipeline python -m experiments.grug.moe_pipeline.benchmark
```

A 14-step run with four warmups measured 3.476 s median step time, 603,327 tokens/s,
19.46% analytic MFU, and 59.153 GB peak memory per device. A matched batch-128,
16-microbatch control measured 543,490 tokens/s and 17.53% analytic MFU. Both used
eight examples per microbatch. The larger batch improved median throughput by 11.0%
without increasing peak memory; final loss differed by 4.0e-5 after 14 synthetic-data
steps.
