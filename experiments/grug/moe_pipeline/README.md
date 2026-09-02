# grug-moe-pipeline

Canonical pipeline-parallel Grug MoE implementation. This directory is a copy-paste
variant of [`experiments/grug/moe`](../moe/): its model and execution path are local
so pipeline-specific changes do not turn the ordinary MoE implementation into a
shared trainer framework.

[`pipeline.py`](./pipeline.py) splits the local transformer into stage pytrees and
builds automatic JaxPP ZeroBubble or DualPipeV optimizer steps. [`benchmark.py`](./benchmark.py)
is the executable runner. One logical stage maps to each physical pipeline rank with
`PIPELINE_SCHEDULE=automatic_zero_bubble`. Set `PIPELINE_SCHEDULE=automatic_dualpipe_v`,
`PIPELINE_STAGES=2P`, and `PIPELINE_PHYSICAL_STAGES=P` to fold two logical stages onto
each physical rank in JaxPP's V-shaped placement. DualPipeV requires at least `2P`
microbatches. `PIPELINE_LAYERS_PER_STAGE` contains one positive layer count per logical
stage.

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
