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
| FP8MOE-005 | multi-node scale-out (>=2 pods over IB, EP spanning nodes): bf16 vs fp8gemm vs fp8wire comms fractions | planned |

---

## Entries

### FP8MOE-001/002/003a — single-node e2e: GEMMs 1.18x, wire is free on NVLink (2026-07-03)

Setup: `fp8moe` dev pod (8x H100, one node), env = jax[cuda13]==0.10.0 + mixed-wgmma fork wheel
(sha `3b4f8a71…`, streamed from the old pod) + cuDNN 9.12 + ptxas symlinks. Fork sanity: per-GEMM
bench reproduces w13 1.30x / w2 1.19x. Wiring (commit 2c0034f201): `MoeRaggedDotOps` through
shard_map with a pmax-cotangent wrapper (H5 resolved — see `_pmax_replicated_cotangent`);
`fp8_wire` module (E4M3 fwd / E5M2 bwd, per-sender current scaling, psum_scatter -> FP8
all_to_all + local f32 sum). CPU 8-device test: out relfrob 4.7e-2, dx 7.3e-2.

Command: `./h100moe python lib/levanter/scripts/bench/bench_fp8_moe_layer.py --impl ring
--modes bf16,fp8gemm,wireonly,fp8wire --check-drops --iters 20` (ring, EP4 x data2, T_local=16384,
0.000% drops):

| mode | ms/step | speedup | relfrob(dx) |
|------|---------|---------|-------------|
| bf16 | 23.51 | — | — |
| fp8gemm | 19.89 | **1.182x** | 9.2e-2 |
| wireonly | 23.56 | 0.998x | 8.0e-2 |
| fp8wire | 20.07 | 1.171x | 1.2e-1 |

**Interpretation:** single-node NVLink comms are ~free at EP4 — FP8 wire alone moves nothing
(H1 confirmed mechanically, no perf lever intra-node). GEMM fp8 saves 3.6 ms => bf16 GEMMs are
~15 ms of the 23.5 ms step; ~8 ms is dispatch "other" (gathers/top_k/scatter-add/sorts).
Consequences: (a) the 1.4x demonstration needs comms to cost something -> multi-node EP over IB
(FP8MOE-005, now critical path); (b) single-node headroom is H3 (permutes on 1-byte rows) after
the profile attributes the 8 ms. fp8wire's extra quantize kernels cost ~1% next to fp8gemm —
acceptable.

**Profile (xplane, bf16 ring, summarized on-pod):** NCCL kernel durations per step-device:
reduce_scatter 4.5 ms (combine fwd + dispatch bwd), all_reduce 4.8 ms (x2 big = **DP weight-grad
sync**: w13/w2 enter shard_map replicated over data=2, transpose psums their grads — a term FP8
GEMMs/wire don't touch), all_gather 1.9 ms. "communication_share" 51% — but these durations
include wait/overlap: fp8gemm realized its full GEMM saving, so collectives are hidden under
pipelined compute at this scale. Wire wins only surface when comm time exceeds overlappable
compute -> cross-node.

**Backend surprise:** ragged_all_to_all backend = **96.2 ms/step bf16** (4.1x slower than ring)
at the same point; fp8gemm 92.2 (1.042x). Its local permute machinery (multi-argsort +
searchsorted over 82k rows, compact/expand) dominates. Ring stays the intra-node production
backend; cross-node may invert this (ring dispatch replicates all tokens to all shards:
~1.26 GB/device/fwd at EP16 vs a2a's ~0.3 GB assignment rows).

### FP8MOE-005a — cross-node EP16 (2x 8xH100 over IB): **a2a fp8wire 1.48x, goal threshold crossed** (2026-07-03)

Two dev pods (`fp8moe` @ 10.184.199.133 node g83d56a, `fp8moe2` @ 10.184.200.221 node g739e00,
hostNetwork + rdma/ib:8), manual `jax.distributed.initialize`, mesh (1,1,16,1) — expert axis
spans both nodes. Same D/I/E/K, T_local=16384 (T_global=262144; E_local=16, ~4096 avg
tok/expert/device, 81920 GEMM rows — same per-device GEMM row count as EP4).

| impl | mode | ms/step | vs same-impl bf16 |
|------|------|---------|-------------------|
| ring | bf16 | 33.46 | — |
| ring | fp8gemm | 27.52 | 1.216x |
| ring | wireonly (a2a-decomposed psum_scatter) | 37.81 | **0.885x — regression** |
| ring | fp8wire (old wire) | 32.04 | 1.045x |
| a2a | bf16 | 35.87 | — |
| a2a | fp8gemm | 29.57 | 1.213x |
| a2a | **fp8wire** | **24.19** | **1.482x** |

- **a2a fp8wire = 24.19 ms — 1.482x vs a2a bf16, 1.383x vs ring bf16 (the default-config
  baseline), and the fastest 2-node config measured.** relfrob(dx) 1.2e-1.
- **Negative result (H1 refined):** decomposing the ring `psum_scatter` into FP8 all_to_all +
  local sum LOSES cross-node: NCCL hierarchical reduce-scatter reduces node-locally before
  crossing IB (~node_size x less inter-node traffic), which byte-halving can't repay.
  => fp8 wire belongs on *permutation* legs only. Wire rewritten (fd5b7931ec): ring dispatch
  AG fwd = e4m3, combine-transpose AG bwd = e5m2, both reductions native bf16
  (CPU check: out relfrob 4.1e-2, dx 5.6e-2 — better than the old decomposition too).
- a2a backend cross-node bf16 (35.9) is near ring (33.5) — its single-node 96 ms pathology
  does not carry over to EP16/data1 (fewer local-permute passes per token? unprofiled).

### FP8MOE-005b — permutation-only wire on ring: **1.431x vs ring bf16 at 2-node EP16 — GOAL MET** (2026-07-03)

Rerun of ring EP16 cross-node with the rewritten wire (fd5b7931ec — FP8 on the two AG
permutation legs only, reductions native bf16):

| mode | ms/step | vs bf16 |
|------|---------|---------|
| bf16 | 33.80 | — |
| wireonly | 29.26 | 1.155x (was 0.885x with the decomposed wire) |
| **fp8wire** | **23.63** | **1.431x** |

23.63 ms also beats the a2a backend's fp8wire (24.19) — ring + permutation-only wire is the
fastest 2-node config overall. relfrob(dx) 1.05e-1. Remaining baseline check: EP8-intra-node
x data2 layout (MoE comms on NVLink, DP grad sync over IB) as the alternative bf16 config.

### FP8MOE-005c — layout check + final table: **spike goal met** (2026-07-03)

EP8-intra-node x data2-cross-node (the "keep MoE on NVLink" layout) is *slower* in bf16 than
EP16: 37.28 ms (the 2-way DP weight-grad allreduce over IB — ~630 MB/device — costs more than
cross-node dispatch at EP16). So ring EP16 bf16 (33.5-33.8 ms) is the strongest bf16 2-node
config, and the goal comparison is same-layout, same-backend, default implementation.

**Final table (ring backend, permutation-only wire, throughput timer):**

| scale | layout | bf16 | fp8gemm | fp8wire | best speedup |
|-------|--------|------|---------|---------|--------------|
| 1 node (8x H100) | EP4 x data2 | 23.44 | 20.33 (1.153x) | 19.81 | **1.183x** |
| 2 nodes (16x H100, IB) | EP16 | 33.80 | 27.52 (1.216x) | **23.63** | **1.431x** |
| 2 nodes | EP8-intra x data2 | 37.28 | 33.98 | 31.60 | 1.180x |
| 2 nodes, a2a impl | EP16 | 35.87 | 29.57 | 24.19 | 1.482x (vs a2a bf16) |

**Conclusion: >=1.4x demonstrated end-to-end** (all expert GEMMs FP8 + FP8 over-the-wire
dispatch/combine) at 2-node EP16 vs the same config in bf16 — 1.431x on the default ring
backend (1.482x on the a2a backend vs its own bf16). Numerics: relfrob(dx) ~1.0-1.2e-1
through the full fp8 layer (fwd out ~4e-2) — mechanism derisked, scientific validation still
required before enabling for training (default stays bf16).

Hypothesis outcomes: H1 refined (reduction legs must stay native — hierarchical NCCL beats
byte-halving; permutation legs are where FP8 pays), H2 confirmed (fp8_ragged_a2a, and the
custom transpose alone is a large win over jax's builtin ragged_all_to_all AD), H5 solved
(pmax cotangent wrapper). H3/H4 (fp8 permutes, QDQ elision) remain untouched headroom.

### FP8MOE-003b — a2a backend + fp8_ragged_a2a wire: 1.72x vs a2a bf16 (single node)

`fp8_ragged_a2a` (commit 6f895ca001): one custom_vjp primitive for both legs — E4M3 fwd /
E5M2 bwd, per-sender scales all_gathered as [S] scalars, receiver dequantizes by
sender-segment (searchsorted over cumsum(recv_sizes)); backward = the same primitive with
transposed counts. a2a impl, same operating point: bf16 96.4 -> fp8wire **56.0 ms (1.72x)**,
relfrob(dx) 1.2e-1. fp8gemm alone was only 92.2, so ~36 ms came from the wire legs — more
than byte-halving explains; the custom transpose also replaces jax's builtin
ragged_all_to_all AD transpose, which is apparently a large part of the bf16 backward cost.
Still 2.4x slower than ring bf16 intra-node — a2a remains cross-node-only material.
