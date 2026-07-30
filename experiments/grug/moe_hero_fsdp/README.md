# Grug MoE FSDP hero

`moe_hero_fsdp` fixes the one-rack 64×GB200 configuration from W&B run
`gb200-d6144-64gpu-nomtp-noconv-bs1152-chunk4-v1`. The expert and replica mesh
axes both have size 1, so model state is fully sharded across all 64 GPUs.

The source run reported 23.653% summary mean MFU and 283,560 final tokens/s.
Two reproductions on our stack and independent rack placements averaged
23.667% summary mean MFU and 280,876 final tokens/s. This branch has CPU config
and lowering coverage but has not run on an accelerator.

The 23.653% value is W&B `throughput/mean_mfu` over 24 samples from a 25-step
run. It is not a steady-state tail window. The step-20 parameter-watch sample
depresses the mean; the source run reached 3.32% MFU on that step. This template
keeps the 25-step schedule and watch interval 20 so its summary uses the same
methodology.

## Resolved configuration

| Setting | Value |
|---|---:|
| Hidden dimension / layers | 6144 / 48 |
| Attention heads / local KV heads / global KV heads | 48 / 12 / 6 |
| Head dimension | 128 |
| Routed experts / experts per token | 128 / 4 |
| Routed / aggregate shared intermediate dimension | 3072 / 6144 |
| Shared-expert count | 2 × 3072 |
| Sequence length / local sliding window | 4096 / 512 |
| Local / global attention layers | 40 sliding-window / 8 full-causal |
| Global-layer cadence | every sixth layer |
| RoPE | fused half-RoPE; disabled on global layers |
| Attention implementation | `gpu_fa4_cute` |
| MoE implementation / expert chunks | `sonic_cute` / 4 |
| Capacity factor / overflow reporting | 1.0 / enabled |
| Layer execution / rematerialization | array-stacked scan / `recompute_all` |
| Cross-entropy token chunk / backward scan unroll | 16,384 / 1 |
| Global batch / steps | 1152 / 25 |
| Topology | 16 workers × 4 GB200 |
| Processes per worker | 1 |
| Expert axis / replica axis | 1 / 1 |
| Mixed precision | fp32 parameters, bf16 compute and outputs |
| Tracker / parameter-watch interval | W&B / 20 |
| Checkpoint / eval | disabled / disabled |

QB routing, GatedNorm, and the attention output gate are structural in this
copy. XSA is explicit. MTP and SConv are absent.

MuonH uses a 0.05 peak learning rate, linear decay, a 0.05 minimum ratio, and
1% warmup. The recorded 25-step heuristic resolves
`adam_lr=0.02511723566133071`, `beta1=0.9062`,
`beta2=0.9646229185299474`, and
`epsilon=4.8379999999999997e-17`. The initializer standard deviation is
`0.5 / sqrt(6144) = 0.0063788795384978605`. Final-logit z-loss is `1e-4`,
router z-loss is 0, gradient clipping and EMA are disabled, and optimizer state
is offloaded to pinned host memory.

The d6144 reproduction used host offload. The rejected offload result applies
to the d5120 EP shape: it needed a 135 GiB pinned-host arena and measured
19.694% MFU. It does not justify disabling offload here.

The launcher installs the four `SCALE_MUON_*` values consumed by this branch.
The matched reproduction also used `SCALE_MUON_GROUP_NONEXPERT` and
`SCALE_MUON_GROUP_SHARED_ONLY` on a branch where they grouped same-shape
non-expert Newton–Schulz work. The current Levanter substrate does not implement
those mechanisms. This is a remaining configuration gap, not an
accelerator-verified equivalence.

The FSDP and EP headline numbers do not establish which strategy is faster.
They use different model shapes and measurement windows. In the available
same-shape comparison, the 0.942% margin is inside placement noise; repeated
draws on one placement agreed within 0.02%.

## Submit

The allocator and XLA flags remain external because they belong to the
accelerator runtime. The template installs `JAX_ENABLE_PGLE=1` and its fixed
Muon settings in the dispatched task.

```bash
run_id="moe-hero-fsdp-$(date -u +%Y%m%d-%H%M%S)"

.venv/bin/iris job run \
  --no-wait \
  --max-retries 50 \
  --cpu 2 \
  --memory 3GB \
  --extra cpu \
  --job-name "$run_id" \
  -e RUN_ID "$run_id" \
  -e XLA_PYTHON_CLIENT_ALLOCATOR cuda_async \
  -e XLA_FLAGS "--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false --xla_gpu_experimental_parallel_collective_overlap_limit=4" \
  -- python -m experiments.grug.moe_hero_fsdp.launch \
    --version dev \
    --run
```

`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` and both XLA flags match the two
successful reproductions. This FSDP graph has `expert_axis_size=1` and uses
`sonic_cute`, so the ragged-all-to-all failure and the EP all-to-all overlap
census do not establish that either flag is independently required here. They
are retained because the authoritative reproduction command used them.

Older project guidance says auto-PGLE must remain off for multi-host runs
because per-host recompilation can desynchronize processes. That guidance is
stale for this FSDP configuration: both independent 64-GPU reproductions ran
successfully with `JAX_ENABLE_PGLE=1`. The EP template remains a separate
configuration and keeps auto-PGLE disabled.
