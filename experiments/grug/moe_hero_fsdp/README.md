# grug MoE FSDP hero

A minimal, **self-contained** FSDP variant of the grug MoE model, fixed to one 64-GPU GB200 rack
(16 nodes × 4 GPUs) for a 25-step throughput / MFU reproduction. Everything is inlined and
opinionated: features that are options elsewhere are hardwired on here, and the folder has **no
dependency on `experiments/grug/moe`** — it imports only the levanter substrate and its own modules.

Launch:

```bash
# dry-run: print the lowered plan locally, no GPUs
python -m experiments.grug.moe_hero_fsdp.launch --version dev

# submit the run (coordinator dispatches the 64-GPU GB200 job)
RID=moe-hero-fsdp-test-1rack
iris --cluster=marin job run --no-wait --enable-extra-resources \
  --target-cluster cw-us-east-08a --priority interactive \
  --cpu 2 --memory 8GB --disk 32GB --timeout 5400 --job-name "${RID}-coord" \
  -e WANDB_API_KEY "$WANDB_API_KEY" -e RUN_ID "$RID" \
  -- python -m experiments.grug.moe_hero_fsdp.launch --version dev --run
```

W&B: `marin-community/marin_moe`, group `moe-hero-fsdp`, run name `$RUN_ID`.

## Files

| file | contents |
| --- | --- |
| `model.py` | `GrugModelConfig` + the always-stacked Transformer (attention, MoE, SConv, scan) |
| `optimizer.py` | `GrugMoeMuonHConfig` — MuonH with three LR groups (muonh / adamh / adam) |
| `grugmuon_hero.py` | 4D-expert-aware distributed Newton-Schulz transform (the "Muon" of MuonH) |
| `adamh.py` | AdamH (Adam direction + Frobenius hyperball scale-invariant step) |
| `heuristic.py` | May Recipe compute-scaling LR refit; derives the optimizer from steps + batch |
| `train.py` | training loop, state init, dispatch, hero runtime env |
| `launch.py` | inlined `HERO_MODEL`, resources, trainer, dataset, entry point |

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
- No MTP, no over-encoding, no PKO, no long-rope on global layers.

### Systems (FSDP)
- **Pure FSDP**: `expert_axis_size=1`, `replica_axis_size=1` → all 64 GPUs on the `data` axis.
- **`expert_chunks=4`** — gather one quarter of the expert bank at a time (the largest MFU lever).
- **`sonic_cute`** MoE backend (QuACK SM100 grouped-GEMM) and **`gpu_fa4_cute`** attention
  (FA4 / CUTLASS 4.6).
- **Large-vocab cross-entropy**: the fused `batched_xla` Pallas path with tuned
  `BlockSizes(1024, 512, 4096)` (v=4096 is the dominant lever vs the autotuned v=64). No liger,
  no pure-JAX chunked CE.
- `offload_opt_state` to pinned host; `remat_mode="recompute_all"`.
- Runtime env baked in: `JAX_ENABLE_PGLE=1`, `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`.

### Optimizer — MuonH
- **Muon direction** (Newton-Schulz orthogonalization) + a **Frobenius hyperball** scale-invariant
  step. 4D expert stacks are orthogonalized by a **distributed** Newton-Schulz (`grugmuon_hero`):
  always-NS, always-pad the 3D non-expert stacks, **intra-rack always-on** (never distributes over
  the cross-rack DCN axis), and **SYRK** (QuACK symmetric GEMM for `X @ Xᵀ`) as a config, defaulting
  on (`use_syrk=True`).
- Three LR groups: **muonh** (matrices + GatedNorms), **adamh** (`lm_head` / `output_proj`),
  **adam** (embeddings, router, attention gate, 1-D norm gains, the tiny SConv kernels).
- LR / beta / epsilon come from the **May Recipe compute-scaling heuristic** (issue #5951) at
  batch 1152 / 25 steps / d6144: `lr=0.05`, `adam_lr≈0.02512`, `beta1=0.9062`, `beta2≈0.96462`,
  `epsilon≈4.84e-17`, linear schedule, **no gradient clipping**, no weight decay applied.
- CE logsumexp **z-loss = 1e-4**. Router z-loss is **logged only**, not added to the loss.

### Run
- **25 steps**, batch **1152**, seq **4096**, SlimPajama-6B, Llama-3 tokenizer, vocab 128256.
- Mixed precision: params fp32, compute/output bf16.
- Checkpointing and eval are **off for this run** but the machinery is retained for later.
- Multi-rack wedge mitigations are intentionally **excluded** (this is a single-rack run).
