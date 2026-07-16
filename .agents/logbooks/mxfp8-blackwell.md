---
topic: mxfp8-blackwell
issue: https://github.com/marin-community/marin/issues/7282
description: Blackwell-native MXFP8 (block-scaled) FP8 for grug MoE — dense and grouped GEMMs on B200/GB200
author: mcwitt
---

# MXFP8 on Blackwell: Task Logbook

## Scope

- Goal: implement and tune a Blackwell-native FP8 path for grug MoE, using MXFP8
  block scaling (32-element blocks, E8M0 scales) for both dense GEMMs
  (`jax.nn.scaled_dot_general` / cuDNN-cuBLASLt) and grouped expert GEMMs
  (SM100 block-scaled grouped kernel), replacing the Hopper delayed
  per-tensor-scaling recipe where profitable.
- Primary metric(s): layer-level fwd+bwd speedup vs bf16 at the canonical grug
  shape d2560/F1280/E256/K4; isolated dense GEMM speedup; end-to-end step
  time / MFU on 8x B200 vs the #7012 bf16 baseline; numerics (rel-Frobenius of
  grads vs bf16, finite loss over a smoke trajectory).
- Constraints: everything opt-in behind `GrugModelConfig.fp8` (defaults stay
  bf16); no regressions to the Hopper path on the shared code; quality
  validation itself is out of scope here (that is #7271, which consumes this).
- Coordinating issue/PR: https://github.com/marin-community/marin/issues/7282
  (implementation/tuning). Related: #7271 (compute-optimal MXFP8-vs-BF16
  quality check), #7012 (B200 MFU ladder), #6998 work stream, PR #7079 (base),
  #6880 (`Fp8RaggedDotOp`), #6660 (`Fp8DotGeneralOp`).
- Experiment ID prefix: `MXFP8-` (`MXFP8-001`, ...). Shared W&B tags:
  `mxfp8`, `7282`.

## Baseline

- Date: 2026-07-16
- Code refs: branch `research/mcwitt/7282-mxfp8-blackwell`, based on PR #7079
  head `2c1c53ea59` (`fp8-moe-mlp-comms`, stacked on `fp8-ragged-dot` /
  #6880).
- Baseline numbers:
  - Hopper per-tensor FP8 (the recipe MXFP8 must match or beat, measured on
    H100): full MoE layer (GEMMs + wire) 1.53x vs bf16 at 2-node EP16 ring
    (33.6 -> 22.0 ms/step), 1.65x on a2a; isolated ragged-dot w13 1.41x / w2
    1.27x; dense `Fp8DotGeneralOp` +19% vs bf16 at d4096 forward (#6660).
  - B200 bf16 (from #7012): best single-node d2560 step MFU 15.67%
    (slim-vjp + all_but_moe remat, ring_cute); 15.82% a2a_cute on the 32-GPU
    ladder. These are the e2e denominators for MXFP8 gains.
  - The Hopper FP8 GMM kernels are wgmma-based Mosaic (sm90-only) — they do
    NOT run on Blackwell; on B200 the current branch's FP8 coverage is dense
    (cuBLASLt f8 per-tensor) + wire only. So "MXFP8 grouped vs baseline"
    compares against bf16 ragged dot on B200, not against the Hopper kernels.

## Hypothesis Queue

### Active

- `MXFP8-H1`: dense GEMMs via `jax.nn.scaled_dot_general` (cuDNN block-scaled)
  beat bf16 >=1.2x at grug dense shapes on B200 and can slot in behind the
  `Fp8DotGeneralOp`-shaped interface. Next test: MXFP8-001 microbench.
- `MXFP8-H2`: an SM100 block-scaled grouped GEMM (CUTLASS via `cutlass_call`,
  reusing the fp8-ragged-cute FFI plumbing) beats bf16 ragged dot >=1.3x at
  canonical expert shapes. Next test: MXFP8-002 feasibility + bench.
- `MXFP8-H3`: MXFP8 quantization is stateless (per-block scales computed on
  the fly, no amax history), so the ops need none of the
  `OverwriteWithGradient` train-step machinery — strictly simpler than the
  per-tensor path. Next test: falls out of MXFP8-003 wiring.
- `MXFP8-H4`: the FP8 wire (per-token scaling, permutation legs only) carries
  over to B200/GB200 unchanged. Next test: enable during MXFP8-003 smoke.
- `MXFP8-H5`: quantize + scale-layout overhead (MXFP8 needs both a row-major
  and a transposed quantized copy for bwd, and cuBLASLt wants a tiled scale
  layout) can be fused or amortized like the Hopper cast-transpose kernels.
  Risk item. Next test: profile within MXFP8-004.

### Blocked

(none)

### Falsified / Dead End

(none)

### Promoted

(none)

## Experiment Matrix (v1)

| ID | What | Where |
|---|---|---|
| MXFP8-001 | Environment probe + dense microbench: jax/jaxlib/cuDNN versions on the B200 clusters, `jax.nn.scaled_dot_general` availability, mxfp8 vs bf16 vs per-tensor-f8 at grug dense shapes | 1x B200 |
| MXFP8-002 | Grouped GEMM feasibility: CUTLASS SM100 block-scaled grouped GEMM through `cutlass_call`; fall back to Mosaic tcgen05 if CUTLASS path stalls | 1x B200 |
| MXFP8-003 | Wire MXFP8 dense op behind `GrugFp8Config` (recipe knob per-tensor vs mxfp8), e2e train smoke incl. FP8 wire | 8x B200 |
| MXFP8-004 | MXFP8 grouped kernel integration; layer-level fwd+bwd bench vs bf16 at d2560/F1280/E256/K4 | 8x B200 |
| MXFP8-005 | Full-step MFU vs #7012 bf16 baseline; hand off numbers to #7271 | 8-32x B200 |

## Open Questions (foraging targets)

- `jax.nn.scaled_dot_general` / `jax.nn.scaled_matmul`: exact API, jax/jaxlib
  minimums, cuDNN version floor (>=9.7?), SM100 gating, and what the clusters
  currently ship. What does its VJP do (grad dtype/recipe)?
- MXFP8 training recipe details (per TE): fwd E4M3 x E4M3, grads E5M2 or E4M3?
  Block dims 1x32 along K for both operands; transposed quantized copies for
  dgrad/wgrad; E8M0 scale rounding mode.
- CUTLASS SM100 block-scaled *grouped* GEMM: does a ready example exist, and
  does the CuTe DSL / `cutlass_call` version we pin support it?
- Mosaic-GPU tcgen05 (Blackwell) support status in our jax pin — viable
  alternative for the grouped kernel and for the quantize/cast-transpose
  kernels?
- Scale-factor layout for cuBLASLt/cuDNN block-scaled matmul (128x4 tiled?):
  who produces it, kernel or XLA?
- Hardware access: Schmidt cluster B200/B300 Slurm partitions vs GB200
  (cw-us-east-08a, arm64 — does our wheel stack exist for aarch64?).

## Entry Log

### 2026-07-16 - MXFP8-000: kickoff

- Issue #7282 filed; branch `research/mcwitt/7282-mxfp8-blackwell` created at
  PR #7079 head `2c1c53ea59`.
- Context: #7079 completed the general FP8 wiring (grouped GEMMs via
  `Fp8RaggedDotOp`, dense via `Fp8DotGeneralOp`, FP8 wire, one config switch,
  train-step state). This thread makes the Blackwell-specific replacement:
  MXFP8 block scaling for dense (native `jax.nn.scaled_dot_general`) and
  grouped (SM100 kernel) GEMMs.
- Next action: background foraging on the open questions above, then
  MXFP8-001 environment probe.
