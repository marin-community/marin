# Logbook: FP8 end-to-end grug MoE layer (expert GEMMs + FP8-over-the-wire dispatch/combine)

Research spike to derisk FP8 comms/collectives in the grug MoE layer. Goal: the MoE MLP layer
running end-to-end in FP8 — all expert GEMMs **and** the dispatch/combine (permute/unpermute)
collectives carrying FP8 over the wire — demonstrating **>= 1.4x fwd+bwd speedup vs the existing
bf16 configuration** at realistic operating parameters.

- **Branch:** `research/fp8-moe-comms` (based on `fp8-ragged-dot-mixed-fork`, which carries the
  mixed e4m3/e5m2 `Fp8RaggedDotOp` haliax ragged_dot on top of origin/main).
- **Deliverables (post-spike):** two review-optimized branches —
  `fp8-moe-mlp` (on `fp8-ragged-dot-mixed-fork`: FP8 wiring for the MoE MLP GEMMs) and
  `fp8-moe-mlp-comms` (on `fp8-moe-mlp`: FP8 over-the-wire dispatch/combine).
- **Experiment ID prefix:** `FP8MOE`.
- **Issue:** https://github.com/marin-community/marin/issues/6911
- **Hardware:** 8x H100 (one CoreWeave `cw-us-east-02a` node, NVLink), mixed-wgmma jaxlib fork
  (`mcwitt/jax@mixed-fp8-wgmma-0.10.0`, cached wheel — see `lib/haliax/scripts/mixed_fp8_fork_setup.sh`).

## Operating parameters (per David's message; fixed for all timing)

Model: D=2560, I=1280 (expert width), E=256, K=4 (top-k), L=26, B=8192·x tokens.

Bench mesh on 8 GPUs: `(replica_dcn=1, data=2, expert=4, model=1)` — reproduces the real EP4
layout. T_local=16384 tokens/device (B=8192·16 global). Per expert-group: 65536 tokens,
262144 assignments → **E_local=64, ~1024 avg tokens/expert, 65536 assignments/device**
(+capacity padding at 1.25 → 81920 GEMM rows for ring) — matches the profile-confirmed
per-device ragged-dot shape (`lhs[65536,2560] x rhs[64,2560,2560]`) from the d2560 config.

Non-uniform routing (same style as `bench_fp8_ragged_dot.py`) so group_sizes are realistic.

## Baseline facts (from the GEMM-level work)

- FP8 vs bf16-Triton per-GEMM fwd+bwd: w13 **1.31x**, w2 **1.19x** (mixed backward, GFP8-FORK-01).
- Expert-MLP layer (GEMMs+SwiGLU only, no collectives): **1.29x** at the operating point
  (GFP8-LAYER-01 in `grug-fp8-ragged.md`). So GEMM-only FP8 lands well short of 1.4x e2e once
  collectives/permutes are in the denominator — the extra headroom must come from FP8 wire
  (2x bytes) and cheaper permutes on 1-byte rows.

## Backend anatomy (read 2026-07-03, `lib/levanter/src/levanter/grug/_moe/`)

- **ring** (default): dispatch = `all_gather` of (x, selected_experts, combine_weights) over
  "expert" [replication]; local top-k capacity selection; two `ragged_dot`s; combine =
  scatter-add into the gathered token layout then `psum_scatter` [**wire reduction**].
- **ragged_all_to_all**: dispatch = local sort by global expert + `jax.lax.ragged_all_to_all`
  [permutation]; local permute by expert; two `ragged_dot`s; combine = **return**
  `ragged_all_to_all` [permutation] + local weighted sum (`_unpermute_from_global_expert`,
  f32 einsum). No wire reduction anywhere — natural FP8-wire home.
- **deepep**: intranode custom dispatch/combine (out of spike scope unless needed).
- GEMMs: both backends call `haliax.nn.ragged_dot(lhs, rhs, group_sizes)` — the `op=` seam
  (`Fp8RaggedDotOp`) exists on the base branch and bypasses backend dispatch.

Wire volume per device (fwd): ring = (ep-1)·T_local·D recv (AG) + (ep-1)/ep·T_ep·D send
(psum_scatter); a2a ≈ (1-1/ep)·K·T_local·D each way. At EP4/K=4 these are comparable;
a2a wins at higher EP.

## Hypothesis queue

- **H1 — FP8 wire on ring**: dispatch all_gather in e4m3 is numerically ~free (it is the same
  per-tensor delayed-scaling quantization the FP8 GEMM applies to its input anyway). The
  psum_scatter combine is a wire *reduction* — FP8-incompatible as-is, but decomposable into
  all_to_all(chunks in fp8) + local bf16 sum, preserving accumulation precision. Backward
  mirrors: AG(grad in e5m2) + a2a(grad chunks in e5m2)+local sum. Try first per user direction.
- **H2 — FP8 wire on a2a backend**: quantize the sorted rows before each `ragged_all_to_all`
  (e4m3 fwd, e5m2 bwd via custom_vjp around the wire segment), per-shard scales all_gathered as
  scalars; combine weighted-sum stays local f32. Structurally cleanest; fallback if ring fights.
- **H3 — permute-on-fp8 bonus**: once rows are 1 byte, the argsort gathers/takes and
  compact/expand row shuffles halve their bandwidth; possibly fuse quantize before the sort.
- **H4 — double-quantize elision**: dispatch-in-e4m3 output can feed the FP8 GEMM directly
  (skip dequant->requant) if the wire scale is the GEMM's delayed input scale. Optimization,
  not needed for feasibility.
- **H5 — amax/state under shard_map**: Fp8RaggedDotOp state (OverwriteWithGradient) enters
  shard_map replicated; per-shard amax cotangents must be pmax'ed (not psum'ed — shard_map's
  default transpose for replicated inputs) across the mesh. Needs explicit handling.

## Risks / open questions

- Does NCCL/XLA `ragged_all_to_all` / `all_to_all` accept fp8 dtypes? (bitcast to uint8 is the
  escape hatch.)
- Combine in FP8 quantizes the expert *outputs* pre-residual: numerics are a real change
  (unlike dispatch). Spike measures relfrob vs bf16 reference; scientific validation is out of
  scope (default stays bf16 regardless).
- 1.4x budget is tight if comms are a small fraction on NVLink — FP8MOE-001 profile decides
  where the time actually is.

## Experiment matrix

| ID | What | Status |
|----|------|--------|
| FP8MOE-001 | bf16 e2e MoE layer baseline (ring + a2a), fwd+bwd step time + comms fraction profile | planned |
| FP8MOE-002 | FP8 GEMMs only (wire still bf16), both backends | planned |
| FP8MOE-003 | FP8 wire dispatch+combine (H1 ring first, H2 fallback) | planned |
| FP8MOE-004 | numerics: relfrob of layer out + grads vs bf16 reference | planned |

---

## Entries
