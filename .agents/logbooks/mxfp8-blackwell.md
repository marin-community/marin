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

### 2026-07-16 - MXFP8-000b: background research brief

- Effort: medium (4 parallel tracks: pinned-JAX source probe, TE recipe +
  scale layouts, SM100 grouped-GEMM options, internal prior work).
- Stop rule: stopped when all six open questions had source-backed answers and
  the ranked kernel-vehicle recommendation stabilized.
- Date: 2026-07-16

#### Current Marin context

- Branch is PR #7079 head + logbook; no MXFP8 code exists anywhere in the repo
  (only `scaled_matmul` grep hit is an unrelated SSD-kernel constant).
- #7012 step anatomy bounds expectations: MoE GEMMs are ~24% of the single-node
  step, so the MXFP8 e2e ceiling is ~1.1x there; at 32-way the step is
  collective-dominated (~48-50% of device-busy) and the ceiling shrinks
  further. Layer-level targets (grouped >=1.3x, dense >=1.2x) remain the gate;
  e2e MFU is reported against 15.67% (8x B200) / 15.04% (32-GPU d2560).
- Native XLA `ragged_dot` has no SM100 fast path (cuDNN enumerates zero plans,
  B200MFU-009/-010) — the grouped bf16 baseline on B200 is QuACK/Triton, not
  XLA.
- Prior related: #5816 (older B200 FP8 recipe issue, open, no results);
  TE 2.15 `te.jax.dense.grouped_dense` measured 1,449 TF/s bf16 at row-13
  shapes on 1x B200 (parity with tuned QuACK 1,470-1,560) in B200MFU-011 —
  NVIDIA's own EP design quantizes AFTER dispatch (`grouped_quantize ->
  grouped_gemm` seam), i.e. wire stays bf16 in their MXFP8 phase.
- Untracked harness `bench_grug_moe_mfu_fp8.py` (repo root, from the #7012
  session) is a row-13 MFU bench with `--fp8/--no-fp8-wire/--no-fp8-dense`
  flags built on exactly this branch stack — adopt for MXFP8-005.

#### Key findings (evidence map, condensed)

- **Claim: our jax 0.10.1 pin has the full dense MXFP8 API.** `jax.nn.
  scaled_dot_general` + `scaled_matmul` + `get_scaled_dot_general_config`
  (jax/_src/nn/functions.py:1268-1414; jax/_src/cudnn/
  scaled_matmul_stablehlo.py). Real custom_vjp: dgrad = g x rhs, wgrad =
  g x lhs, both as block-scaled matmuls with the cotangent quantized per
  `configs[2]`. custom_partitioning with K-shard psum logic; vmap batcher.
  Lowers to custom call `__op$block_scaled_dot` (cuDNN graph path — takes
  LINEAR row-major scales; no 128x4 swizzle needed on this path). No Python
  SM/cuDNN gate; enforcement is compile-time in `jax-cuda13-pjrt`. gpu-extra
  lockfile has cuDNN 9.19 (floor is 9.7.1 — 9.7.0 has a block-scaled
  concurrency bug). Confidence: high (installed source). Caveats:
  `configs=None` SILENTLY falls back to `lax.dot_general` (null-experiment
  footgun); output dtype must be f32/bf16/f16; residuals saved in FULL
  precision (no fp8 activation memory win; interacts with remat budget);
  `K % 32 == 0` required.
- **Claim: the correct MXFP8 training recipe is E4M3 everywhere (incl. grads),
  ceil-rounded E8M0 scales, dual rowwise+columnwise quantized copies.**
  NVIDIA arXiv 2506.08027 (8B/15T within 0.5% ppl of bf16; MoE 16B/1T within
  0.1% loss): E5M2 grads DEGRADE mxfp8 (block scale supplies range, element
  bits buy precision) — opposite of our Hopper per-tensor mixed finding, which
  stays correct for per-tensor. Scale rounding must be round-UP
  (`2^ceil(log2(amax/448))`); OCP v1.0's floor rule destabilizes training.
  Both copies must be quantized from the high-precision original (transposing
  fp8 data requires requantization). Our jax pin implements exactly this
  (e4m3 default `configs[2]`, `cast_to_e8m0_with_rounding_up`). TE
  `MXFP8BlockScaling` default = `Format.E4M3` both directions. Confidence:
  high (paper + TE source + jax source agree).
- **Claim: the grouped-GEMM vehicle exists in our pinned cutlass-dsl 4.5.2.**
  `examples/python/CuTeDSL/cute/blackwell/kernel/moe/torch_scaled_grouped_mm.py`
  (+ `moe_persistent_scheduler.py`, `moe_utils.py`, added 4.5.0): offs-ragged
  2Dx3D fwd `(sum_tokens,K) @ (E,K,N)` and 2Dx2D wgrad
  `(M,sum_tokens) @ (sum_tokens,N) -> (E,M,N)` — shape-for-shape our
  ragged_dot fwd/dgrad + wgrad decomposition. MXFP8 e4m3 + UE8M0 sf_vec 32;
  K must be % 128 (2560 OK, 1280 OK); expert tensormaps built by a ~2us helper
  kernel. NVIDIA's own B200 table: mxfp8 2Dx3D avg 1.29x, 2Dx2D avg 1.41x vs
  torch/cuBLAS grouped. Fallback: `blockscaled_grouped_gemm/` (general,
  experimental, fp4-adjacent bug reports #2737/#3102, MXF8 path unaffected so
  far). C++ reference: CUTLASS example 92
  (`92_blackwell_moe_gemm_blockscaled_rcgrouped.cu`, 4.3.1). Confidence:
  high on existence/shape-fit (source); perf numbers are NVIDIA's, unreplicated.
- **Claim: Mosaic-GPU in our jaxlib fully supports tcgen05 block-scaled MMA.**
  `jax/experimental/mosaic/gpu/tcgen05.py`: `mma(..., a_scale/b_scale:
  TMEMRef)`, e8m0->block-32 (mxfp8), f32 accumulator required, N % 32,
  scales in TMEM with required layout; Pallas exposes `tcgen05_mma` +
  `async_copy_scales_to_tmem`; `pallas/ops/gpu/blackwell_ragged_dot_mgpu.py`
  is a bf16 tcgen05 ragged-dot example to extend. Viable second entrant;
  most kernel-engineering. Confidence: high (installed source).
- **Claim: scale-factor layout is the main integration trap on the raw
  cuBLASLt/CUTLASS path.** Hardware wants 128x4-tile `((32,4),4)` interleaved
  UE8M0 tiles (512B per 128x128 data area), zero-padded to full tiles,
  IMMUTABLE under operand transpose. TE stores row-major and swizzles just
  before GEMM (fused kernels); naive quantize/dequant can dominate (fal.ai:
  dequant 1.76x the matmul; quantizer ~40% of step if unfused). The cuDNN
  graph path (jax dense) takes linear scales — trap applies to our custom
  grouped kernel + its quantizer only. Confidence: high on layout, medium on
  overhead magnitudes (blog-level).
- **Risk: MXFP8 instability at scale is documented** (Kempner arXiv
  2506.20752: loss spikes/grad-norm blowups in larger longer-trained models,
  unrecoverable) — recipe- and scale-conditional; NVIDIA parity claims used
  the exact recipe above. Belongs to #7271's gate; log grad-norms + per-site
  quantization toggles there.
- **Negative/failed leads**: DeepGEMM = DeepSeek 1x128/128x128 f32-scale
  recipe, not MX block-32 — wrong semantics, torch-only. Triton
  `tl.dot_scaled` on Blackwell has open layout/crash issues (#8648, #8431,
  #7550) and lost prior bake-offs here. TE-from-JAX for grouped = re-plumbing
  cuBLASLt grouped through our own FFI; keep TE as a torch perf REFERENCE
  only. jax has no scaled ragged/grouped op at all (zero "ragged" hits in
  jax/_src/cudnn). `jax.lax.scaled_dot` composite is inference-shaped (no
  transpose rule) — not our path.

#### Decisions taken from the brief

1. Grouped vehicle: vendor the DSL MoE scaled-grouped kernel (4.5.2, already
   pinned) behind the `fp8-ragged-cute` `cutlass_call` adapter pattern
   (arch `sm_100a`; tensormap/TMA/persistent-scheduler machinery already
   proven through the FFI on Hopper). Mosaic tcgen05 ragged is the bake-off
   second entrant, not the first.
2. Dense vehicle: `jax.nn.scaled_dot_general` with default mxfp8 configs
   (e4m3/e4m3/e4m3, ceil-e8m0) behind a NEW STATELESS op class implementing
   the `DotGeneralOp` protocol; recipe knob on `GrugFp8Config` dispatches
   per-tensor (Hopper) vs mxfp8 (Blackwell).
3. Grad dtype: e4m3 (paper default), do NOT port the Hopper e5m2-mixed rule to
   mxfp8. Revisit only if #7271 shows divergence.
4. FP8 wire stays bf16 during the MXFP8 phase (NVIDIA EP quantizes after
   dispatch); `fp8_wire` remains an independent flag to re-test later.
5. cutlass-dsl pin: consider bump `<4.6` -> `<4.7` (4.5.3 compile-time fix,
   4.6.1 multi-FFI-registration for JAX) once the vendored kernel works.

#### Hypothesis queue update

- Revise `MXFP8-H1`: strengthened by source probe (full custom_vjp +
  partitioning exist in-pin); add the `configs=None` footgun and
  full-precision-residual memory caveat to the test plan.
- Revise `MXFP8-H2`: primary kernel = DSL MoE scaled-grouped (not the general
  blockscaled-grouped example); baseline = QuACK/Triton bf16 (XLA has no SM100
  ragged path). NVIDIA's 1.29-1.41x makes >=1.3x plausible but tile-M
  {128,256} padding at small per-expert M is the named risk.
- Confirm `MXFP8-H3` shape: stateless op, no OverwriteWithGradient — API
  confirmed stateless in-pin; but note residuals stay high-precision.
- Revise `MXFP8-H4` -> deprioritized: wire stays bf16 in the MXFP8 phase by
  design; re-test after MXFP8-005.
- Sharpen `MXFP8-H5`: the trap is quantize+swizzle for the CUSTOM grouped
  kernel (dense/cuDNN path takes linear scales); plan a fused
  dual-write+swizzle quantizer like the Hopper CT kernels.
- Add `MXFP8-H6`: e4m3 cotangents (with block scales) are
  numerically adequate for training — per NVIDIA paper; falsified if #7271
  wall-time-matched loss diverges and e5m2-grad arm rescues it.

#### Source ledger (abridged; full citations in the agent reports)

| Source | Type | Claim used for |
|---|---|---|
| jax 0.10.1 installed source (functions.py, scaled_matmul_stablehlo.py, tcgen05.py) | Marin code / installed pkg | dense API, VJP, partitioning, footguns, Mosaic tcgen05 |
| NVIDIA arXiv 2506.08027 | paper | e4m3-everywhere recipe, ceil e8m0, dual copies, parity results |
| TE source + MXFP8 feature docs | official docs | recipe defaults, swizzle-before-GEMM, %32 constraints |
| cuBLAS docs 3.1.4.3 + Colfax/torchao reconstructions | official docs (layout table unverified verbatim) | 128x4 SF layout |
| cuDNN 9.7.x release notes | official docs | version floor, 9.7.0 concurrency bug |
| cutlass v4.5.2 tag: CuTeDSL moe/ + blockscaled_grouped_gemm/ + example 92 | external code | grouped vehicle, constraints, NVIDIA perf table |
| cutlass issues #2737 / PR #3102 | GitHub issue | experimental-kernel risk |
| Kempner arXiv 2506.20752 | paper | instability risk at scale |
| #7012 logbook B200MFU-007/-009/-010/-011/-030 | logbook | e2e ceiling, XLA no-SM100-ragged, TE grouped_dense parity, EP seam |
| fal.ai MXFP8 quantizer blog | blog | quantizer overhead magnitude (medium confidence) |

#### Revised experiment matrix (v2, supersedes v1)

| ID | What | Where |
|---|---|---|
| MXFP8-001 | Env probe + dense microbench: verify `scaled_dot_general` compiles/runs on B200 (gpu-extra sync, cuDNN 9.19); numerics vs bf16 reference; perf vs bf16 einsum AND vs per-tensor `Fp8DotGeneralOp` at grug dense shapes (qkv/o d2560, shared-expert) | 1x B200 |
| MXFP8-002 | Vendor DSL MoE scaled-grouped kernel behind `cutlass_call` (sm_100a); correctness vs emulated reference; bench vs QuACK bf16 at row-13 per-device shapes; measure quantize+swizzle overhead separately | 1x B200 |
| MXFP8-003 | Stateless `MxFp8DotGeneralOp` + `GrugFp8Config` recipe knob; dense-mxfp8 e2e train smoke (wire off) | 8x B200 |
| MXFP8-004 | MXFP8 ragged op (new op class per the `Fp8RaggedDotOp` docstring contract) + fused dual-write quantizer; layer-level fwd+bwd bench vs bf16 | 8x B200 |
| MXFP8-005 | Full-step MFU via `bench_grug_moe_mfu_fp8.py` (adopt onto branch) vs 15.67%/15.04% bf16 baselines; hand off to #7271 | 8-32x B200 |

- Next action: MXFP8-001 on a B200 node (Schmidt cluster or GB200 east-08a;
  needs `--extra gpu` sync per cluster toolchain notes).
