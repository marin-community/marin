# Grug MoE FSDP hero

`moe_hero_fsdp` is the fixed one-rack GB200 configuration derived from W&B run
`gb200-d6144-64gpu-nomtp-noconv-bs1152-chunk4-v1`. It uses pure fully sharded
data parallelism: the expert mesh axis has size 1, and the replica axis has
size 1 so model state is sharded across all 64 GPUs.

The source research build measured 23.653% MFU and 283,560 tokens/s. Those
numbers were not measured on this branch. This variant has CPU-level config and
lowering coverage, but it has not been run on an accelerator.

## Resolved configuration

| Setting | Value |
|---|---:|
| Hidden dimension | 6144 |
| Layers | 48 |
| Attention heads / KV heads / head dimension | 48 / 12 / 128 |
| Routed experts / experts per token | 128 / 4 |
| Routed / aggregate shared intermediate dimension | 3072 / 6144 |
| Shared-expert count | 2 × 3072 |
| Sequence length / short-layer sliding window | 4096 / 512 |
| Short / long attention layers | 36 sliding-window / 12 full-causal |
| Attention implementation | `gpu_fa4_cute` |
| MoE implementation | `sonic_cute` |
| Capacity factor | 1.0 |
| Layer execution / rematerialization | array-stacked scan / `recompute_all` |
| Global batch / steps | 1152 / 120 |
| Topology | 16 workers × 4 GB200 |
| Processes per worker | 4 |
| Expert axis / replica axis | 1 / 1 |
| Mixed precision | fp32 parameters, bf16 compute and outputs |

The model hardcodes QB routing, GatedNorm, and the attention output gate. XSA
is a model config key and the launcher explicitly sets `xsa=True`; there are no
`gated_norm` or `attn_gate` launcher flags. MTP and SConv are absent, which
keeps both mechanisms off.

The two shared experts preserve the aggregate 6144 intermediate width and total
parameter count. Larry's experiment record explicitly sets
`SCALE_NUM_SHARED_EXPERTS=2`; the preceding unreplicated screen measured
23.17% to 23.46% MFU (+0.29 percentage points).

The optimizer is MuonH with peak learning rate 0.05, linear decay, minimum LR
ratio 0.05, 1% warmup, and no gradient clipping. The 0.05 value is the binding
cap from the source run. The source record gives the Adam/embedding group LR as
approximately 0.0251; this variant fixes it at 0.0251. Final-logit z-loss is
`1e-4`, router z-loss is 0, optimizer-state host offload is disabled, and EMA
is disabled.

The launcher trains for 120 steps on the pinned SlimPajama-6B tokenization,
uses the W&B `marin-community/marin_moe` project, disables parameter-watch
steps, and retains ten-minute checkpoints. The seed is 0.

## Submit

Set the allocator and XLA flags in the submitter environment. The Grug
dispatcher forwards `XLA_` and `JAX_` variables to the final accelerator
tasks.

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }\
--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false \
--xla_gpu_experimental_parallel_collective_overlap_limit=4 \
--xla_gpu_enable_latency_hiding_scheduler=true"
unset JAX_ENABLE_PGLE JAX_PGLE_PROFILING_RUNS

uv run python -m experiments.grug.moe_hero_fsdp.launch \
  --version dev \
  --run
```

`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is required at 64×GB200. The default
BFC allocator has caused fragmentation OOMs and silent collective deadlocks at
this topology.

The ragged all-to-all NCCL barrier flag is mandatory with JAX 0.11. Without
`--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, a
64-process run can compile and then segfault in `ncclDevCommCreate` before step
0. The overlap limit must remain 4; the setting is not monotone, and 2 measured
worse than the default value of 1.

Auto-PGLE must remain disabled for multi-host runs because its per-host
recompilation crashes. This variant ships no pinned PGLE profile.

Use a JAX 0.11 run as the control. The matched JAX 0.11 baseline measured 1.217
percentage points below the JAX 0.10.1-era baseline, so pre-0.11 results are
not valid controls for this template.
