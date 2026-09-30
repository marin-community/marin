# mfu30-gemm: dense bf16 GEMM efficiency (agent C)

Branch `research/mcwitt/mfu30-gemm` (worktree `~/projects/marin.mfu30-gemm`), based on `research/mcwitt/mfu30`
(`49aa77d115`). Parent logbook: `.agents/logbooks/mfu30.md`. Toolkit: `autoresearch/loop-260930-mfu30-gemm/`.
Run ids `m30c-<short>-<NN>`, Iris ports 33300-33349.

## Scope

cuBLAS (nvjet) GEMMs take 4.7 s of the 13.86 s step on the Sep 24 baseline trace (rank 0, 3 steps), at
1.4-1.7 PF/s (57-68% of the 2.5 PF/s spec). Hypotheses: H-C1 ceiling (kernel vs contention vs power),
H-C2 shared-MLP gate/up fusion, H-C3 fused QKV, H-C4 contention with the ragged all-to-all.

## M30C-001 GEMM inventory (2026-09-30)

`gemm_configs.py` joins the HLO's cuBLAS custom-calls (shapes, layouts, dot dims) with the trace rows:
42 distinct configs, 4.71 s/step; 24 configs above 0.02 s/step cover 4.59 s (`hero_gemms.json`).
Per-instance in-situ rates (non-overlapped instances) run from P5 ~1.45-1.60 to a max of 1.80-1.85 PF/s;
medians 1.57-1.74. Wgrad GEMMs (K=65536) sit lowest (median 1.57-1.59).

XLA's `DotMerger` is already active: the MoE router ([6144,384]) is merged with one [6144,3072] dot that
shares `mlp_in` (latent-down in forward, a shared-expert gate in remat), giving the [6144,3456] GEMMs.
The 64 MB default `--xla_gpu_dot_merger_threshold_mb` stops the merge after the router seed. Q/K/V and
the shared-MLP gate/up dots also share operands (4-5 GEMMs per group per pass), so a larger threshold
would merge them without code changes. A threshold high enough to merge the shared-expert gate/up would
also merge them into the router GEMM, moving shared-expert work in front of the dispatch that it now hides.

## M30C-002 Contention with the ragged all-to-all is SM partitioning (H-C4)

The ragged all-to-all device kernel launches 32 blocks x 512 threads at 96 registers (49,152 registers per
block). Every nvjet GEMM CTA uses 256 threads x 255 registers (65,280, the whole register file), so a GEMM
CTA cannot share an SM with a transport block. While the transport runs, GEMMs get 116 of 148 SMs (-22%).
Overlapped shared-MLP instances have medians of 1.31-1.34 PF/s against 1.70-1.72 non-overlapped, a 22-23%
drop that matches the SM fraction. The GEMM grids are non-persistent (grid 2052-16416 CTAs), so the loss is
the SM share for the overlap time, with no static-schedule tail. The GEMM side has no lever here: shrinking
the transport grid slows the transport, and XLA exposes no cuBLASLt SM-count target.

## M30C-003 The GEMMs are power-bound (H-C1)

W&B system metrics (rank-0 node, 4 GPUs, 15 s samples, busy samples only):

| run | MFU | mean power (W) | SM clock p10/p50/p90 (MHz) |
|---|---|---|---|
| hero-nopdl-step108k | 22.3 | 1021-1048 | ~1400/1850-1950/2062 |
| hero-fa4sm100-nomask-step146k | 26.4 | 1068-1092 | ~1310-1390/1680-1785/2062 |
| gcab-control-20260930 | 28.1 | 1108-1120 | 1237-1329/1500-1665/1987-2062 |
| gcab-freeze-20260930 | 28.4 | 1108-1128 | 1232-1351/1496-1642/2010-2062 |

Enforced power limit 1200 W per GPU; max SM clock 2062 MHz (the 2.5 PF/s spec assumes it). As MFU rose,
average power rose and the median SM clock fell. NVML clock-event reason during GEMMs is 0x4 (SW power cap).

Single-GPU microbenchmark `m30c-gemmbench-02` (GB200x4 spare node, GPU 0 active, bf16 N(0,1) operands,
XLA's default cuBLAS path): burst (5 calls after 1 s idle) 1.65-1.81 PF/s; sustained (5 s loop, last
half) 1.40-1.55 PF/s at 1190-1450 MHz (mean of 50 ms NVML samples) and ~1170 W, reason SW power cap.
Rate tracks the sampled clock: scaled to 2062 MHz it lands at or slightly above the zero-operand rate
below, so the kernels lose nothing beyond the clock. In situ the same kernels reach 1.5-1.7 PF/s (real
data, interleaved lower-power phases). The kernels are not the limit; the 1200 W cap sets the clock
during GEMM phases.

Operand data sets the power. Same kernels, sustained 4 s loops (PF/s, SM clock):

| config | zeros | mant2 (2 mantissa bits) | sparse50 | N(0,1) | smooth N(0,1) | in situ median |
|---|---|---|---|---|---|---|
| [65536,6144]x[6144,3072] fwd | 2.25 @2062 (899 W, no cap) | 1.74 @1507 | 1.97 @1734 | 1.49 @1289 | 1.51 @1320 | 1.71 |
| [65536,3072]x[3072,6144] | 2.30 @2062 | 1.70 @1422 | 1.71 @1441 | 1.48 @1230 | 1.47 @1225 | 1.72 non-ovl |
| wgrad [3072,6144] K=65536 | 2.21 @2062 | 1.80 @1597 | 2.09 @1969 | 1.52 @1354 | 1.56 @1386 | 1.59 |
| Q proj [65536,6144]x[6144,6144] | 2.24 @2062 | 1.76 @1537 | 2.00 @1816 | 1.50 @1288 | 1.52 @1327 | 1.73 |
| K/V proj N=1536 | 2.08 @2062 | 1.66 @1533 | 1.87 @1811 | 1.44 @1331 | 1.45 @1347 | 1.56 |

With zero operands the GPU stays under the cap (830-940 W) at 2062 MHz and the kernels reach 2.08-2.30
PF/s, 83-92% of the spec: this is the kernel ceiling, 30-35% above in situ. Every non-trivial pattern hits
the 1200 W cap. In situ sits between N(0,1) and mant2. The profiler confirms the benchmark runs the same
nvjet kernels XLA picks in training (e.g. `256x192_64x4_2x2f_2cta_h_bz_NNT` for the shared gate/up).

In-situ signature: across 4,008 non-overlapped GEMM instances with >= 1 TFLOP, the rate falls as the
preceding window gets more tensor-heavy (cuBLAS, QuACK, FA4 kernels): correlation -0.09 over 2 ms, -0.42
over 10 ms, -0.66 over 50 ms (rate by quartile 1.71 / 1.65 / 1.60 / 1.47 PF/s); -0.25 within fixed
shape+kernel (slope -0.45 PF/s per unit busy fraction). The power controller integrates over tens of ms.

Fusion candidates under the same conditions (sustained N(0,1); zeros in brackets):
fused QKV forward 4,899 us vs Q + 2 x K/V 5,018 us (-2.4%) [3,270 vs 3,401, -3.9%]; both shared experts'
gate/up as one N=12288 GEMM 6,492 vs 4 x 1,661 = 6,644 us (-2.3%); both down projections as one K=6144
GEMM 3,318 vs 2 x 1,671 = 3,342 us (-0.7%). Before concat/slice costs, QKV fusion is worth ~0.02 s/step
and shared-MLP fusion ~0.03 s/step. H-C2 and H-C3 are below the campaign's detection threshold.

Kernel selection is not a lever (sustained N(0,1), same 8 configs): `--xla_gpu_enable_cublaslt=false`
matches the default within 2%; Triton only (`--xla_gpu_cublas_fallback=false`) is 8-20% slower;
`--xla_gpu_autotune_level=0` picks kernels 2-100x slower. All four GPUs of the node running concurrently
lose another ~6% (1.32-1.45 PF/s); a 30 s loop matches the 5 s one (1.494 PF/s), so the 5 s window is
steady state.

## M30C-004 Verdict and campaign implications (2026-09-30)

H-C1: the ceiling is power, not the kernel. The nvjet kernels XLA picks run at 2.1-2.3 PF/s (83-92% of
spec) when the operands draw little power. With realistic operands every hero GEMM holds the GPU at its
1200 W cap and the SM clock drops to 1200-1600 MHz. In situ they average 1.5-1.7 PF/s, above the N(0,1)
steady state because lower-power phases interleave. H-C2/H-C3 (fusion): -0.7% to -3.9% of the fused GEMMs'
time, ~0.02-0.03 s/step each before concat/slice costs. H-C4 (contention): SM partitioning by the
transport's register footprint, no GEMM-side lever. No rack job was submitted; nothing in this direction
clears the 0.15 MFU acceptance bar. Remaining GEMM-side upside: <= 0.05 s/step.

For the campaign:
- The ~9 s/step of tensor-core kernels (cuBLAS 4.7, QuACK 3.5, FA4 0.8) cannot get faster by kernel
  work under the fidelity rules. Only fewer FLOPs (remat) or lower energy per FLOP (fp8, excluded) move it.
- The 30% target has to come from non-tensor time: exposed collectives, exposed host copies, idle, and
  memory-bound passes. Memory-bound kernels are not clock-limited, so removing a pass saves its full time
  and also lowers the energy near adjacent GEMMs.
- Removing idle next to GEMM phases gives some time back to power: GEMMs slow by ~0.45 PF/s per unit of
  tensor-busy fraction over the preceding 50 ms. A rough allowance is 10-15% of the removed time.
- Mean power on the rank-0 node is ~1115 W at 28.4% MFU; 30% will push it toward the 1200 W cap.

(`m30c-gemmbench-01` was invalid: an np.float64 scale promoted operand B to f32, so it timed TF32-class
GEMMs at ~0.8 PF/s. Cancelled and fixed in `gemm_bench.py`.)

## M30C-005 Direction (a): non-MoE memory-bound passes (2026-09-30)

Inventory (Sep 24 trace, `fusion_detail.py`, `.remat` = XLA HloRematerialization clones that A's
host-offload flag may delete): norms bwd 0.346 s, norms remat 0.161 (+0.087 XLA-remat), norms fwd
0.148, short conv 0.187 (Pallas; bwd 0.119 at ~2.2 TB/s, 6.7 passes vs a 3-pass floor), attention
elementwise fwd 0.105 / bwd 0.095 / remat 0.075 (+0.148 XLA-remat), shared-MLP elementwise 0.05
(+0.11 XLA-remat). Total fusion time 1.36 s against 1.05 s at 7 TB/s for the bytes moved: kernel
inefficiency is worth ~0.3 s; the rest needs fewer passes.

Findings:
- The GatedNorm sigmoid runs as its own pass (`gemm_fusion_dot.*`, 0.066 s/step fwd+remat): XLA's
  Triton GemmFusion takes the sigmoid into the rank-128 GEMM's fusion, the autotuner falls back to
  cuBLAS, and the leftover epilogue stays a separate kernel. Binary epilogues (`x * gate`) are never
  fused toward users, so XLA cannot reach one pass here.
- The two RMSNorm backward fusions with the weight-gradient column reduction run at 2.7 TB/s
  (0.115 s/step; 0.071 s above roofline).
- The backward sums the input gradients of every projection that reads the same normalized input
  (4 full-width bf16 addends for attention, 6 for the MLP side) in one elementwise pass.

Component benchmark `block_bench.py` (one GB200, hero per-GPU shape 16x4096xd6144, real
`Block.__call__` with FA4, short convs, gated norms and shared experts; a stand-in replaces the routed
experts; checkpointed fwd+bwd), jobs `m30c-blockbench-02..04`:
- **Projection fusion (H-C2/H-C3 in code) is rejected.** Fused Q/K/V (+gate) and shared-expert gate/up
  keep the forward bitwise identical and cut the accumulation pass (-1.0 to -1.2 ms/layer), but the
  backward concatenation of the cotangents costs more (+1.6 ms `wrapped_concatenate`, +1 ms slices and
  multiplies; with the attention gate included XLA puts the concatenation on the reduction emitter,
  +6 ms). GEMM time did not drop. Net per layer: all fusions +1.5 to +2.5 ms, Q/K/V only +1.0 to +1.5,
  shared only +0 to +1. Kept as `model_fused_projections.py` for the record; model.py reverted.
- **`--xla_gpu_enable_triton_gemm=false`**: -0.85 ms/layer in three paired runs (123.9 -> 123.0 ms,
  run-to-run sd ~0.3), -0.98 in the earlier job. Memory-bound kernel time drops 1.9 ms/layer (the
  sigmoid fuses into the multiply; GemmRewriter turns the gradient accumulation into cuBLAS beta=1
  epilogues), GEMM and FA4 time rise ~1 ms (denser tensor work under the power cap, or the beta reads).
  At hero scale that is ~0.04 s/step: flag-only, but below the 0.15 MFU acceptance bar on its own.

Custom-kernel candidate (not started; needs orchestrator go-ahead): fused RMSNorm + GatedNorm forward
(one kernel per token block: pass 1 streams x for the row sum of squares and (x*w) @ W_down on tensor
cores; pass 2 re-reads x, computes sigmoid(silu(h) @ W_up) and writes the output, plus the gate for the
backward). Current fwd chain ~1.5 ms per norm call vs ~0.35-0.45 ms at roofline; 4 calls per layer
(attention and MLP norms, forward and remat) -> ~0.2 s/step. The backward keeps cuBLAS for the
weight-gradient GEMMs; fusing its elementwise passes adds maybe ~0.1 s. Effort: Pallas-Triton kernel
(Mosaic GPU fails layout inference on GB200 per the short-conv notes), custom_vjp saving (rstd, h, g),
GPU correctness tests, then a rack screen: ~1-2 days. Numerics change at bf16 rounding points
(allowed; needs the rack loss check). Short-conv backward with a register/SMEM carry: ~0.08 s/step,
needs CUDA or CuTe DSL, similar effort.
