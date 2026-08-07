# grug MoE FSDP hero

A minimal, **self-contained** FSDP variant of the grug MoE model for throughput and MFU
runs on one or more GB200 racks. Everything is inlined and opinionated: features that are options
elsewhere are hardwired on here, and the folder has **no dependency on `experiments/grug/moe`** —
it imports only the levanter substrate and its own modules.

Launch:

```bash
# dry-run: print the lowered plan locally, no GPUs
python -m experiments.grug.moe_hero_fsdp.launch \
  --run-id moe-hero-fsdp-test-1rack --dp-racks 1 --num-steps 25 --version dev

# submit one or more racks; each rack gets 16 GB200x4 nodes and batch 1024
RID="moe-hero-fsdp-test-2rack"
iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 5400 --job-name "${RID}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" \
  -- python -m experiments.grug.moe_hero_fsdp.launch \
    --run-id "$RID" --dp-racks 2 --num-steps 200 --version dev --run
```

W&B: `marin-community/marin_moe`, group `moe-hero-fsdp`, run name `--run-id`. Pass
`-e WANDB_PROJECT <project>` to the Iris coordinator command to use another W&B project.

Use `experiments.grug.moe_hero_fsdp.launch_supervised` instead of `launch` for NCCL diagnostics.
Each GPU task then runs training in a child process with XLA's 60-second per-execution deadman and
no in-pod restart. A deadman abort or another child failure fails the task immediately; the existing
15-minute progress watchdog remains the fallback.

`experiments.grug.moe_hero_fsdp.launch_failsafe_control` is the diagnostic control for the supervisor
itself: it keeps the same XLA deadman, progress tracking, and NCCL initialization timeout but runs the
trainer directly with no recovery parent and no task retry.

Checkpoint staging benchmark:

```bash
python -m experiments.grug.moe_hero_fsdp.checkpoint_benchmark \
  --run-id checkpoint-52b-1rack --dp-racks 1 --num-steps 12 \
  --checkpoint-every-steps 8 --version dev
```

This uses a 52.85B-total, approximately 1.71B-active top-1 MoE. It offloads the optimizer state,
writes a deterministic checkpoint at step 8 and another at clean completion, and records the
synchronous host-staging and asynchronous commit phases without enabling Python allocation tracing.
The entire artifact is pinned under `marin_temp_bucket(ttl_days=1)`, so it is disposable and covered
by the one-day lifecycle policy.

## Files

| file | contents |
| --- | --- |
| `model.py` | `GrugModelConfig` + the always-stacked Transformer (attention, MoE, SConv, scan) |
| `optimizer.py` | `GrugMoeMuonHConfig` — MuonH with three LR groups (muonh / adamh / adam) |
| `grugmuon_hero.py` | 4D-expert-aware distributed Newton-Schulz transform (the "Muon" of MuonH) |
| `adamh.py` | AdamH (Adam direction + Frobenius hyperball scale-invariant step) |
| `heuristic.py` | May Recipe compute-scaling LR refit; derives the optimizer from steps + batch |
| `train.py` | training loop, state init, dispatch, hero runtime env |
| `launch.py` | rack-scaled resources, DP/FSDP mesh, batch, tracker, dataset, and entry point |

## What's distinctive about this run

### Architecture
- **d6144, 48 layers, all MoE** (no dense layers). 48 heads, GQA with **12 local / 6 global KV
  heads**, head_dim 128.
- **128 experts, top-4** routing with **QB (query-bias) routing** (structural).
- **2 shared experts**, each `shared_expert_intermediate_dim=3072` → 6144 total shared width
  (the field is *per shared expert*).
- **GatedNorm**, **XSA** (Exclusive Self-Attention: subtract the per-head component of the output
  parallel to `v`), and a **headwise sigmoid attention gate** — all structural / always-on, no flags.
- **SConv** — Inkling-style depthwise causal 1-D conv at all four sites (`k`, `v`, `attn`, `mlp`),
  identity-init so it's inert at step 0.
- **Sliding-window attention (512)** with **every 6th layer global** (full-causal). Short layers use
  half-RoPE; global layers run **rope-free**. RoPE is the fused single-pass form (`rope_fused`).
- **Always array-stacked**: all blocks run through one compiled `lax.scan` body (the unrolled
  program OOMs). The per-layer sliding window rides in as precomputed FA4 bound arrays selected in
  the scan.

### Systems (FSDP)
- **One rack**: `expert_axis_size=1`, `replica_axis_size=1` → one 64-GPU `data` axis.
- **Two racks**: `expert_axis_size=1`, `replica_axis_size=2` → two DP replicas, each with a
  64-GPU `data` axis. Model parameters are replicated across `replica_dcn` and FSDP-sharded only
  on `data`. The embedding table is fully replicated so each device does a local lookup.
- **`expert_chunks=4`** — gather one quarter of the expert bank at a time (the largest MFU lever).
- **`sonic_cute`** MoE backend (QuACK SM100 grouped-GEMM) and **`gpu_fa4_cute`** attention
  (FA4 / CUTLASS 4.6).
- **Large-vocab cross-entropy**: the plain-XLA path with `BlockSizes(v_block_size=4096)` (v=4096 is
  the dominant lever for the 128k vocab; the SMEM-tiled `batched_xla` kernel caps the h*v weight
  tile at ~99KB and cannot take v=4096). No liger, no pure-JAX chunked CE.
- `offload_opt_state` to pinned host; `remat_mode="recompute_all"`.
- Runtime env baked in: `JAX_ENABLE_PGLE=1`, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.
- XLA GPU command buffers are disabled by default with `--xla_gpu_enable_command_buffer=`.
  See [#5675](https://github.com/marin-community/marin/issues/5675) for the CUDA graph failure and
  the plan to enable them again.

### Optimizer — MuonH
- **Muon direction** (Newton-Schulz orthogonalization) + a **Frobenius hyperball** scale-invariant
  step. 4D expert stacks are orthogonalized by a **distributed** Newton-Schulz (`grugmuon_hero`):
  always-NS, always-pad the 3D non-expert stacks, **intra-rack always-on** (never distributes over
  the cross-rack DCN axis), and **SYRK** (QuACK symmetric GEMM for `X @ Xᵀ`) as a config, defaulting
  on (`use_syrk=True`).
- Three LR groups: **muonh** (matrices + GatedNorms), **adamh** (`lm_head` / `output_proj`),
  **adam** (embeddings, router, attention gate, 1-D norm gains, the tiny SConv kernels).
- LR / beta / epsilon come from the **May Recipe compute-scaling heuristic** (issue #5951) for
  the selected global batch, 25 steps, and d6144. The schedule is linear, with **no gradient
  clipping** and no weight decay applied.
- CE logsumexp **z-loss = 1e-4**. Router z-loss is **logged only**, not added to the loss.

### Run
- **25 steps**, batch **1024 per rack**, seq **4096**, SlimPajama-6B, Llama-3 tokenizer, vocab
  128256.
- Mixed precision: params fp32, compute/output bf16.
- Eval is off. Training writes a resumable checkpoint every 30 minutes and at clean completion;
  a restarted gang resumes from the latest fully committed checkpoint.
- The process-local watchdog is disabled until one training hour has elapsed and the first step has
  completed. It then exits with code 124 when a later training step runs for 15 minutes or another
  lifecycle phase makes no progress for 60 minutes. This is a fallback for collectives that fail to
  honor XLA's 10-minute NCCL termination timeout; Iris treats it as a failure and retries the gang.
- FA4 metadata constants are explicitly replicated before batch sharding. This prevents the
  compiler from routing each attention metadata transfer through device 0 on a multi-rack run.
