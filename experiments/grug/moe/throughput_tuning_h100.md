# Grug MoE throughput tuning on 8×H100 (cw-us-east-02a) — GQA & MLA

Reference for training-throughput (MFU / tok/s) of the d2048 / 24-layer / 64-expert
Grug MoE on a single 8×H100 node, for both the **GQA** attention block and the
**MLA** (Multi-head Latent Attention) block. Covers the tested configs, the tuning
levers, the code improvements made, and the memory/compile mechanics.

The chronological research log (including dead ends and corrections) lives in
`.agents/projects/mla-fa4-warp-specialized-forward.md`.

## TL;DR — best configs found

| block | best config | MFU (reported) | tok/s (per-GPU) | honest MFU* |
|---|---|---|---|---|
| **GQA** | FSDP scan, batch 256 | **28.16%** | 203k (25.4k) | ~24.4% (×0.868) |
| **MLA** | FSDP scan, batch 256 | **21.51%** | 155k (19.4k) | ~24.8% (×1.154) |
| **MLA parallel/ScMoE** | FSDP scan, batch ~208 | 11.3% | 164k (20.5k) | ~23.1% (×2.077) |

*Reported MFU uses the standard full-attention analytic (see "MFU accuracy"). GQA's
head_dim=128 is real so causal+SWA make it an over-count (×0.868); MLA's qk=192/v=128
+ latent projections are under-counted by the head_dim=64 analytic (×1.154). The
**parallel/ScMoE block** (LongCat-Flash, `SCALE_PARALLEL_BLOCK=1`) is undercounted
**×2.077**: the analytic sees only 12 blocks of (1 attn + MoE) and misses the dense
chain + 2nd attention. On tok/s/GPU (formula-free) it **beats plain MLA: 20.5k vs
19.4k**; honest MFU ~23.1% is on par (it OOMs at b224 before reaching the b256
optimum). See `.agents/projects/parallel-block-scmoe.md`.

Both use the same recipe: **`SCALE_SCAN_LAYERS=1 SCALE_WATCH=0`, FSDP mesh
(`SCALE_EXPERT_AXIS=1`), `ring` MoE, `gpu_fa4_cute`, `recompute_all`,
`muonh_heuristic`, big batch.** GQA is ~2× cheaper per token than MLA on this axis
and reaches a higher batch before OOM.

## Shared model config

d2048, 24 layers, 64 routed experts (top-4) + 1 shared expert (intermediate 2048),
routed intermediate 1024, seq 4096, sliding window 2048 on 18/24 layers (layers
{3,7,11,15,19,23} are full-causal), vocab 128256. MuonH optimizer (May-recipe
heuristic: LR/β/ε from tokens & dim). `mp = params=float32, compute=bf16, output=bf16`.

- **GQA block**: `num_heads=16, head_dim=128, num_kv_heads=4` (4:1 GQA).
- **MLA block** (`SCALE_MLA=1`): `num_heads=32` (=2·d/128), `qk_head_dim=192`
  (128 nope + 64 rope), `v_head_dim=128`, `q_lora_rank=1024`, `kv_lora_rank=512`.

## Results

### GQA — MFU vs batch (FSDP, ring MoE, MuonH, seq4096)
| batch | unrolled | **scan (`SCALE_SCAN_LAYERS=1`)** |
|---|---|---|
| 8 | 4.00% | — |
| 16 | 7.93% | — |
| 32 | 12.56% | — |
| 64 | 17.69% (marginal) | — |
| 128 | 21.45% | 24.99% |
| 256 | OOM | **28.16% / 203k (finished)** |
| 512 | — | OOM |

EP=8 ring baseline: b16 = 15.55% (OOMs b32+). So GQA ranking:
EP=8 15.55% < FSDP unrolled b128 21.45% < **FSDP scan b256 28.16%**.

### MLA — MFU vs batch (FSDP, ring MoE, scan on, MuonH, seq4096)
| batch | before sharding fix | **after MLA attn-sharding fix** |
|---|---|---|
| 32 | 14.71% (ceiling) | — |
| 64 | OOM | 18.03% |
| 128 | OOM | 19.77% |
| 256 | OOM (50 GiB) | **21.51% / 155k (finished)** |
| 512 | — | OOM |

### MLA — mesh / MoE-backend comparison (scan on)
| mesh / backend | batch | MFU | note |
|---|---|---|---|
| EP=8 ring, unrolled | 16 | 12.88% | OOMs b32+ |
| EP=8 ring, scan | 64 / 128 | 16.40% / 17.02% | fits (b256 OOM) |
| EP=8 ragged_all_to_all, scan | 64 / 128 | 5.27% / 5.79% | fits but slow (sort-chain overhead) |
| EP=8 deepep, scan | — | not run | needs `DEEPEP_SRC_ROOT` FFI setup |
| **FSDP (EP=1), scan** | 256 | **21.51%** | **best** |

MLA ranking (scan on): FSDP b256 **21.51%** ≥ EP ring+scan b128 **17.02%** >>
EP a2a+scan **5.8%**.

## Improvements made this session

1. **MLA attention-parameter sharding fix** (`experiments/grug/moe/model.py`,
   `MultiheadLatentAttention.init`) — **the key MLA improvement.** The latent/
   up-projection weights were sharded on the `model` (tensor-parallel) axis or
   replicated (`P(None,"model")` / `P(None,None)`). On a single-node run the `model`
   axis has size 1, so they were **replicated on all 8 GPUs** (no FSDP relief),
   ~10× the on-device attn-param footprint of GQA. Changed to shard over `data`:
   `w_dq/w_dkv/w_kr: P(None,None)→P("data",None)`; `w_uq/w_uk/w_uv:
   P(None,"model")→P("data","model")` (keeps `model` TP on the per-head output dim;
   RMSNorm unaffected — it acts on the gathered activation). **Moved the MLA FSDP
   ceiling b32→b256 and lifted MFU 14.71%→21.51%.** Committable.

2. **MLA implementation + fast 192/128 attention** (pulled in from the
   `dial_h100_dev` WIP / prior worktree; not authored here but required to run MLA):
   `MultiheadLatentAttention` block; the native SM90 warp-specialized backward
   un-gated for qk=192 MHA (`_fa4_cute_backend.py`); the asymmetric V-load predicate
   fix in `_fa4_cute_kernels.py`; `head_dim_v` plumbing in `_fa4_cute_config.py`.

3. **The tuning recipe** (launch knobs, no code change) — see below.

## Tuning levers, ranked by impact

1. **`SCALE_SCAN_LAYERS=1` — stacked `lax.scan` over layers (biggest lever).**
   Default is unrolled (`use_array_stacked_blocks=False`): XLA compiles the Block
   body 24× and lets all 24 layers' fp32 expert params/grads be live at once — a
   batch-independent ~45 GiB buffer that OOMs at b64+. Scan compiles one body and
   keeps one layer live → **~3-4× faster compile (~26→~7 min), collapses the 45 GiB
   buffer, and +3.5 MFU pts**. Requires `disable_pko=True` (default); mixed
   long/short attention stays homogeneous via `lax.cond` in the scan body.

2. **Push global batch up under FSDP.** MFU rises monotonically with batch (the fixed
   FSDP expert-param all-gather amortizes over more tokens). MFU-optimal that fits is
   b256 for both GQA and MLA; b512 OOMs.

3. **`SCALE_WATCH=0`.** With scan at large batch, the grad/param-norm watch callback
   unstacks the scanned layers around step 9-10 → OOM. Disable for throughput runs.

4. **FSDP mesh (`SCALE_EXPERT_AXIS=1`, data=8) over expert-parallel.** EP's `ring`
   all-gathers the full global token set per device (memory ∝ global batch) and OOMs
   at small batch; FSDP keeps the MoE dispatch local. With scan, EP ring+scan is
   viable (MLA 17% @ b128) but still below FSDP (21.5% @ b256). [2-D dp×ep, i.e.
   ring EP=2/4, is the next thing to try — see open items.]

5. **`ring` MoE backend** (the default). Do NOT use `ragged_all_to_all` (3× slower —
   heavy JAX sort/permute + double all-to-all + Python-unrolled capacity clip) or
   `deepep` (needs external FFI setup). See "MoE EP backends".

6. **`gpu_fa4_cute` attention** (native SM90 warp-specialized backward),
   **`recompute_all`** remat, **`muonh_heuristic`** optimizer,
   **`SCALE_CHECKPOINTS=local`** (disposable checkpoints for throughput runs).

## Launch command (MLA best; drop `SCALE_MLA` for GQA)

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
uv run iris --config=lib/iris/config/cw-us-east-02a.yaml job run --no-wait \
  --job-name mla-moe-d2048-b256 -e WANDB_API_KEY "$WANDB_API_KEY" -e RUN_ID mla-moe-d2048-b256 \
  -e SCALE_MLA 1 -e SCALE_SCAN_LAYERS 1 -e SCALE_WATCH 0 \
  -e SCALE_HIDDEN_DIM 2048 -e SCALE_NUM_LAYERS 24 -e SCALE_NUM_EXPERTS 64 -e SCALE_TOP_K 4 -e SCALE_SEQ_LEN 4096 \
  -e SCALE_EXPERT_AXIS 1 -e SCALE_REPLICA_AXIS 1 -e SCALE_GPU_REPLICAS 1 -e SCALE_BATCH 256 \
  -e SCALE_OPTIMIZER muonh_heuristic -e SCALE_ATTN_IMPL gpu_fa4_cute \
  -e SCALE_STEPS 60 -e SCALE_TRACKER wandb -e SCALE_CHECKPOINTS local \
  -- python -m experiments.grug.moe.launch_cw_scale
```

## MoE EP backends (`lib/levanter/src/levanter/grug/_moe/`)

- **`ring`** (default, `ep_ring.py`): all-gather all tokens over the expert axis →
  select this shard's assignments with one `top_k`+gather → expert `ragged_dot`s →
  scatter-add + reduce-scatter. Few ops (fast) but `[global_tokens, H]` memory. Best
  with scan.
- **`ragged_all_to_all`** (`ep_ragged_all_to_all.py`): sort tokens by destination
  expert → clip to capacity → all-to-all to owner shards → expert compute →
  all-to-all back → unsort/combine. `[local_capacity, H]` memory (lean) but ~6
  full-width argsort/sort passes + 2 all-to-all + an 8×8 Python-unrolled clip → 3×
  slower. MFU only credits the two matmuls, so utilization collapses to ~6%.
- **`deepep`** (`ep_deepep.py`): grug's DeepEP FFI — the TorchTitan-equivalent fused
  dispatch/combine (compact recv buffer, in-kernel permute). **Not runnable here
  without setup**: needs `DEEPEP_SRC_ROOT` at a DeepEP checkout (commit 7febc6e),
  `DEEPEP_CUDA_ARCH=sm_90` (H100), and `nvcc`; see `lib/levanter/docs/dev/DeepEP.md`.
  H100 is a supported target — it's a setup task, not a hardware limit.

## Memory mechanics (why things OOM)

- **Unrolled ~45 GiB buffer** = 24 layers × 64 experts × 3 × (2048×1024) ≈ 9.66B fp32
  params/grads live across all layers. Batch-independent (42-45 GiB @ b16/32/64),
  non-deterministic at the edge. Scan eliminates it.
- **MLA replicated attn params (pre-fix)** = latent/up-projections on the size-1
  `model` axis → ~10× GQA's per-device attn footprint; capped MLA at b32. Fixed by
  data-sharding (improvement #1).
- **FSDP b512 OOM** is a batch-scaling activation/dispatch buffer — the ceiling once
  the fixed buffers are removed. MFU-optimal sits at b256.
- **EP ring OOM** = `all_gather(x_local,"expert")` → `[global_tokens, H]` per device.
- Diagnostics: the BFC log `ran out of memory trying to allocate N GiB` is
  **non-fatal** (XLA retries); the fatal signal is
  `RESOURCE_EXHAUSTED: ... jit_train_step`, and a peer-rank OOM shows on other ranks
  as GRPC `connection refused`.

## MFU accuracy

`throughput/mfu` uses `flops_per_example = 3·lm_flops_per_token·seq` with the
standard-attention analytic (`levanter/utils/flop_utils.py`): `head_dim =
hidden/num_heads`, full `seq²`, no causal, no sliding window.

- **GQA**: head_dim=128 is correct, but causal + 18/24 windowed layers make the
  analytic an over-count → honest hardware MFU = **0.868×** reported.
- **MLA**: the analytic uses head_dim=64 (=2048/32) and omits qk=192/v=128 + latent
  projections, so it under-counts even net of the sliding window → honest MLA MFU =
  **1.154×** reported.
- `tokens_per_second` is exact and formula-independent — prefer it for comparison.
- Not-yet-done: an MLA-aware `lm_flops_per_token` so the MLA MFU is honest.

## Compile time

- Unrolled ~26 min (24 layer bodies + FA4 CuTe/cutlass JIT, not covered by the XLA
  cache); scanned ~7 min. No exhaustive/max autotune is enabled (XLA default level);
  the fused-CE pallas autotune is persistently cached.
- `JAX_COMPILATION_CACHE_DIR` is set to the region-local prefix by
  `iris/runtime/jax_init.py`, but `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`, so
  XLA autotune sub-results aren't persisted across runs (a compile-time win on the table).

## Open items / next experiments

- **2-D dp×ep mesh — ring EP=2 and EP=4** (`SCALE_EXPERT_AXIS=2|4`). Ring all-gathers
  only over the expert axis, so a smaller EP degree shrinks the gather (EP=2 → global/4)
  while `data` FSDP-shards the rest — may unlock batch past EP=8's b128 and beat FSDP.
- **DeepEP setup** — wire `DEEPEP_SRC_ROOT` + `DEEPEP_CUDA_ARCH=sm_90` + nvcc into the
  job to get the fused (TorchTitan-style) EP path; the only path that could top ring.
- **MLA-aware flop count** so reported MFU is honest.
- **Confirm b256 is MLA/GQA MFU-optimal** (vs b384) and profile the MLA b256 step to
  attribute the residual gap to GQA (attention core vs latent projections).
