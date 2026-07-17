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

- `MXFP8-H2` (GEMM-level CONFIRMED for all three products, MXFP8-002/002b:
  fwd 2130/2027, dgrad 2041/2112, wgrad 2209/2214 TF/s w13/w2 — ~1.4-1.5x vs
  bf16; errors 3e-6..1e-5): the LAYER-level verdict now hangs entirely on H5.
- `MXFP8-H3`: MXFP8 quantization is stateless (per-block scales computed on
  the fly, no amax history), so the ops need none of the
  `OverwriteWithGradient` train-step machinery — strictly simpler than the
  per-tensor path. Next test: falls out of MXFP8-003 wiring.
- `MXFP8-H4`: the FP8 wire (per-token scaling, permutation legs only) carries
  over to B200/GB200 unchanged. Next test: enable during MXFP8-003 smoke.
- `MXFP8-H5` (SHARPENED by MXFP8-002b — now the gating hypothesis): a
  fusion-grade dual-write quantize+swizzle producer (<=0.7 ms marginal for
  the four activation tensors) turns the honest layer-quad from 0.58x into
  ~1.25x; XLA producers max out at ~1.6-2.2 TB/s (7.06 ms total, breakeven
  is 2.09 ms); an HBM-ideal standalone kernel only reaches ~1.11x. Next
  test: MXFP8-002c quantizer prototype (CuTe or Mosaic, Hopper CT analog).

### Blocked

(none)

### Falsified / Dead End

- `MXFP8-H1` (as stated): naive `jax.nn.scaled_dot_general` does NOT beat
  bf16 dense on sm100 — 0.64-1.00x, while per-tensor `Fp8DotGeneralOp` gets
  1.35-1.61x at the same shapes. Why stopped: MXFP8-001 (logbook entry
  2026-07-16, job mxfp8-001-dense-r3). Superseded by `MXFP8-H1b`.
- `MXFP8-H1b` (resolved, split verdict -> dense mxfp8 shelved): the
  `__cudnn$blockScaledDot` GEMM alone is healthy (1.11-1.33x) but XLA's
  block-quantize kernel costs ~a full GEMM per operand (~1 TB/s effective),
  and even a free quantizer only TIES per-tensor fp8. Decision: dense stays
  on per-tensor `Fp8DotGeneralOp`; revisit only if #7271 shows per-tensor
  numerics insufficient. Evidence: MXFP8-001b (2026-07-16, job
  mxfp8-001b-dense-r2).

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

### 2026-07-16 - MXFP8-001: dense microbench on GB200 — mxfp8 loses to bf16; per-tensor fp8 wins big

- Hypothesis: MXFP8-H1 (dense `scaled_dot_general` >=1.2x vs bf16 on sm100).
- Commit Hash: 6d3cf1a02 (harness), branch `research/mcwitt/7282-mxfp8-blackwell`.
- Command: `iris --controller-url=http://localhost:10000 job run --user mwittmann
  --gpu GB200x1 --enable-extra-resources --cpu 16 --memory 64g --extra gpu
  --job-name mxfp8-001-dense-r3 -- python
  experiments/grug/moe/standalone/bench_mxfp8_dense.py --out /tmp/bench_mxfp8_dense.json`
  (cw-us-east-08a via kubectl port-forward tunnel; jobs r1/r2 were env/assert
  fixes — r1 hit the `--offline` cuDNN reinstall bug, fixed by cherry-picking
  #7031 = 0f3dae95d; r2's assert was too strict: XLA rewrites the lowered
  `__op$block_scaled_dot` into `__cudnn$blockScaledDot` in the compiled HLO).
- Config: 1x GB200 (sm100), jax 0.10.1, cuDNN 9.19.0.56, T=65536 tokens,
  bf16 inputs, median of 50 iters after 10 warmup. fwd+bwd = grad wrt (x, w)
  of sum(out^2).
- Result (speedup vs bf16; bf16 abs at qkvo: 1471 TF/s fwd):

  | shape | arm | fwd | fwd+bwd | err out/gx/gw |
  |---|---|---|---|---|
  | qkvo 2560x2560 | fp8_tensor | **1.352x** (1990 TF/s) | **1.607x** (2363 TF/s) | 4.3e-2/5.7e-2/3.5e-2 |
  | qkvo 2560x2560 | mxfp8 | 0.996x | 0.813x | 3.8e-2/4.3e-2/2.8e-2 |
  | shared_up 2560x1280 | fp8_tensor | 1.166x | 1.474x | |
  | shared_up 2560x1280 | mxfp8 | 0.815x | 0.642x | |
  | shared_down 1280x2560 | fp8_tensor | 1.321x | 1.493x | |
  | shared_down 1280x2560 | mxfp8 | 0.966x | 0.666x | |

- Interpretation (`exploratory`, single GPU/single run):
  - The native mxfp8 path WORKS on GB200 (lowers to `__cudnn$blockScaledDot`,
    errors slightly better than per-tensor fp8) but is SLOWER than bf16
    everywhere, catastrophically so on bwd. The compiled HLO materializes
    `f8e4m3fn[1,T,K/32,32]` block tensors — XLA emits standalone
    quantize/scale kernels around the cuDNN GEMM; bwd re-quantizes g twice
    (dgrad + wgrad legs). Overhead, not the GEMM, is the prime suspect.
  - Per-tensor `Fp8DotGeneralOp` (the #7079 dense path, cuBLASLt) is
    EXCELLENT on sm100: 1.35x fwd / 1.61x fwd+bwd at qkvo — better than its
    H100 result (+19% at d4096). The already-merged dense path is the
    Blackwell dense incumbent to beat.
  - MXFP8-H1 as stated is FALSIFIED for the naive call; the open question is
    GEMM-only throughput of `__cudnn$blockScaledDot` with precomputed scales
    (`jax.nn.scaled_matmul`), which decides between "fix with fused
    quantizer" and "cuDNN dense mxfp8 dead end".
- Ops note: job output JSON lives on the pod's /tmp and dies with it — the
  logs carry all numbers; future harness runs should print JSON to stdout.
- Next action: MXFP8-001b — add a `mxfp8_prequant` arm (`scaled_matmul` with
  precomputed quantized operands + scales, fwd-only) to isolate GEMM-only
  throughput; keep per-tensor fp8 as the dense reference.

### 2026-07-16 - MXFP8-001b: GEMM-only isolate — blockScaledDot fine, XLA quantize is the killer; dense stays per-tensor

- Hypothesis: MXFP8-H1b (GEMM-only beats bf16; slowdown is quantize overhead).
- Commit Hash: harness at branch tip (001b arm + ascontiguousarray fix on top
  of 6d3cf1a02).
- Command: same submit as MXFP8-001, job `/mwittmann/mxfp8-001b-dense-r2` on
  cw-us-east-08a (1x GB200, jax 0.10.1, cuDNN 9.19).
- Result (fwd speedup vs bf16; T=65536):

  | shape | fp8_tensor | mxfp8 e2e | mxfp8 GEMM-only | quantize_x alone |
  |---|---|---|---|---|
  | qkvo 2560x2560 | 1.350x | 1.024x | 1.332x | 0.346 ms (~71% of GEMM) |
  | shared_up 2560x1280 | 1.195x | 0.881x | 1.114x | 0.358 ms |
  | shared_down 1280x2560 | 1.270x | 1.005x | 1.139x | 0.262 ms |

  fwd+bwd unchanged from MXFP8-001 (fp8_tensor ~1.5x, mxfp8 e2e 0.65-0.79x).
- Interpretation (`exploratory`):
  - H1b split verdict: the `__cudnn$blockScaledDot` GEMM itself is healthy
    (1.11-1.33x) — but XLA's emitted block-quantize kernel runs at roughly
    1 TB/s effective (0.35 ms for a 336 MB bf16 activation read on ~8 TB/s
    HBM), one full GEMM's worth of time per operand. On bwd the cotangent is
    quantized twice (dgrad + wgrad legs), hence the 0.65x.
  - Even with a FREE quantizer, mxfp8 GEMM-only only ties per-tensor fp8
    (1.33 vs 1.35 at qkvo, worse at narrow shapes). Per-tensor's delayed
    scaling has no amax reduction in the hot path and its scalar
    multiply+cast fuses into neighboring XLA ops — structurally cheaper.
  - DECISION: dense GEMMs stay on per-tensor `Fp8DotGeneralOp` (the #7079
    path) on Blackwell. Native-mxfp8 dense is shelved unless/until a fused
    quantizer exists AND #7271's quality gate shows per-tensor numerics are
    insufficient (mxfp8's rel-err is only marginally better: 3.7e-2 vs
    4.3e-2). MXFP8 effort concentrates on the grouped expert GEMMs, where
    sm100 has NO working fp8 path at all and the CUTLASS DSL kernel brings
    its own fused scale-factor handling.
  - Recipe note for #7271: this implies a mixed recipe on Blackwell
    (per-tensor dense + mxfp8 grouped). Per-tensor dense was already
    exercised on H100 trainings; document the split in any quality run.
- Next action: MXFP8-002 — vendor the CuTeDSL MoE scaled-grouped-GEMM kernel
  (cutlass-dsl 4.5.2) behind the fp8-ragged-cute `cutlass_call` adapter,
  correctness first, then bench vs QuACK bf16 at row-13 expert shapes.

### 2026-07-16 - MXFP8-002: CuTeDSL MXFP8 grouped GEMM working via cutlass_call on GB200 — 2.2 PF/s, gate met on fwd

- Hypothesis: MXFP8-H2 (SM100 block-scaled grouped GEMM >=1.3x vs bf16).
- Commit Hash: 42f7d9fa2 (vendored kernel + adapter + bench, pushed).
- Command: `python experiments/grug/moe/standalone/bench_mxfp8_grouped.py` via
  the usual 1x GB200 iris submit; jobs `/mwittmann/mxfp8-002-g1..g9` (g8 first
  green, g9 = tile ablation green).
- Config: cutlass v4.5.2 MoE scaled-grouped kernel (tcgen05 block-scaled MMA,
  e4m3 + e8m0 sf_vec 32), stock aarch64 nvidia-cutlass-dsl 4.5.2 wheel, no
  jaxlib fork, no torch. M=262144 tokens, E=64; uniform + Dirichlet-0.5
  skewed routing incl. zero-token experts; tile (128,256,128).
- Result (median of 50):

  | shape | arm | ms | TF/s | note |
  |---|---|---|---|---|
  | w13 K2560 N1280 | mxfp8_kernel | 0.781 | **2200** | 1.42x vs bf16 dense yardstick (1546) |
  | w13 skewed | mxfp8_kernel | 0.790 | 2173 | skew costs ~1% |
  | w2 K1280 N2560 | mxfp8_kernel | 0.838 | **2050** | 1.32x vs bf16 dense (1552) |
  | w2 skewed | mxfp8_kernel | 0.846 | 2031 | |
  | either | bf16 XLA ragged_dot | ~80 | 21-33 | strawman only (no sm100 path) |

  Correctness: 3.4e-06..9.9e-06 rel-Frobenius vs dequantized-same-operands
  reference (gate <1e-3 passed by 2 orders); ~3.7e-2 vs unquantized = expected
  mxfp8 noise. Tile (128,256,128) beats (128,128,128) by ~8%.
- Interpretation (`exploratory`, single GPU):
  - MXFP8-H2 CONFIRMED on the forward product: vs the honest tuned-bf16
    grouped baseline (QuACK 1,449-1,560 TF/s at these shapes, B200MFU-011),
    2,200 TF/s is ~1.41-1.52x. 2.2 PF/s is ~49% of B200 dense-FP8 peak on a
    ragged problem. The persistent scheduler absorbs routing imbalance.
  - Remaining before layer-level claims: dgrad + wgrad products (vendored
    kernel's 2Dx2D scenario with accumulate_on_output covers wgrad), and the
    quantize+swizzle producer — currently bench-only per-group gather/concat;
    MXFP8-001b already showed XLA quantize (~1 TB/s) can eat the entire win.
    The fused dual-write quantize+swizzle kernel is now THE critical path
    (MXFP8-H5).
- New gotchas (durable):
  1. `nvidia-cutlass-dsl-libs-cu13` 4.5.2 differs from libs-base at the same
     version: its `cute.make_ptr` ignores the requested memspace for llvm.ptr
     values, so cutlass_call tensors arrive in GENERIC address space, not gmem
     (the Hopper NOTES "genuine gmem pointers" claim holds only on
     base/cu12). Fix at the launcher boundary: rebuild each tensor via
     `iterator.toint()` + `make_ptr(dtype, int, AddressSpace.gmem,
     assumed_align=...)` — make_ptr honors memspace for integer values.
  2. Any `cute.recast_tensor`/`recast_ptr` on cutlass_call tensors also lands
     in generic space — avoid recasts; `cutlass.jax` maps f8e8m0fnu/uint8
     natively and `TensorSpec(mode=(0,2,1))` expresses layout permutations.
  3. CPU-only local repro: `build_function_spec` + `get_or_compile_kernel`
     with CUTE_DSL_ARCH=sm_100a compiles the full kernel on a GPU-less x86
     box — fast iteration; but never install libs-base AND libs-cu13 in one
     venv (silent wheel shadowing masked gotcha 1 locally).
  4. f32 `jax.lax.ragged_dot` reference at these shapes RESOURCE_EXHAUSTEDs
     (triton autotune profiles a ~160 GiB fusion); use per-expert dense
     matmuls as reference. bf16 ragged_dot on sm100 is 21-33 TF/s.
  5. The vendored kernel needed a 4th file (`moe_sched_extension.py`) beyond
     the three surveyed; kernel HardwareInfo needs a live CUDA ctx — hardcode
     148 SMs for GB200.
- Next action: dgrad/wgrad products + fused quantize+swizzle kernel, then
  layer-level fwd+bwd bench (MXFP8-004).

### 2026-07-16 - MXFP8-002b: dgrad + wgrad green; honest layer-quad is 0.58x — producers gate everything

- Hypothesis: MXFP8-H2 (bwd legs) + MXFP8-H5 (producer cost).
- Commit Hash: 92f7af921 + b6fa1f9de (adapters, fused dual producer, bench).
- Command: `python experiments/grug/moe/standalone/bench_mxfp8_grouped.py`, job
  `/mwittmann/mxfp8-002b-g1` (1x GB200; single job, first-attempt green —
  CPU-only sm_100a compile validated the 2Dx2D launcher locally in 2 s).
- Result (M=262144, E=64, tile (128,256,128), median of 50; err vs
  dequantized-same-operands ref, gate <1e-3):

  | product | w13 | w2 | note |
  |---|---|---|---|
  | fwd | 0.806 ms / 2130 TF/s | 0.847 / 2027 | |
  | dgrad | 0.842 / 2041 | 0.814 / 2112 | transposed-weight view via TensorSpec mode, no materialized transpose |
  | wgrad (2Dx2D) | 0.778 / **2209** | 0.776 / **2214** | fastest product; skew costs ~15% here (~1% elsewhere) |

  All errors 3e-6..1e-5. bf16 dense yardstick 1475-1531 TF/s.
- Producer costs (jitted XLA, per tensor): fused dual-orientation producer
  (both quantized copies + swizzled SFs in one jit; bit-exact, asserted
  on-device) buys ~1.3x over naive but lands at 1.72 ms for act (M,2560) /
  0.95 ms (M,1280) / ~0.86 ms per weight — effective ~1.6-2.2 TB/s. The
  QUANTIZE itself is the bottleneck (MXFP8-001b redux), not the swizzle.
- **Layer-quad honest total** (fwd+dgrad+wgrad, w13+w2, vs 3x bf16 dense =
  6.95 ms): GEMMs only 4.86 ms -> **1.43x**; + XLA producers (7.06 ms) ->
  **0.58x**. Break-even producer budget: 2.09 ms. HBM-ideal (~7 TB/s)
  standalone dual-write quantizer ~1.4 ms -> ~1.11x; fusion-grade (marginal
  cost = fp8 writes only, quantize fused into ops already touching the
  tensors) ~0.7 ms -> **~1.25x**. The layer-level win exists ONLY with
  fusion-grade producers (`exploratory`, but internally consistent with
  MXFP8-001b's independent quantize measurement).
- New gotchas: 2Dx2D wgrad runs at full speed in natural layouts (m-major A /
  n-major B via TensorSpec mode) and per-expert token slices stay TMA-aligned
  for any group sizes; the 2Dx2D SF chunk order needs an atom-block
  permutation ([expert][row_block][col_block]), unlike the 2Dx3D plain
  swizzle; zero-token experts safe only with accumulate_on_output=False
  (epilogue stores zeros — True would need pre-zeroed FFI outputs we can't
  provide); time bench arms BEFORE reference churn (BFC OOM warning is
  non-fatal but pollutes timing).
- Next action: MXFP8-002c — prototype the fused/dual-write MXFP8 quantizer
  (CuTe or Mosaic, Hopper CT-kernel analog) targeting <=0.7 ms marginal for
  the four activation tensors; this is the 0.58x-vs-1.25x decision point.
  Op wiring (MXFP8-004) proceeds in parallel once the producer verdict is in.

### 2026-07-16 - MXFP8-002c: CuTe dual-write quantizer — 2.5x over XLA, layer-quad lands at break-even

- Hypothesis: MXFP8-H5 (fusion-grade producer turns 0.58x into ~1.25x).
- Commit Hash: 68bbe7d5a / 764ee12c2 / 48e363421 / 189c3faf0
  (`mxfp8_grouped/quantize_cute.py`, `bench_mxfp8_quantizer.py`, probe).
- Command: usual 1x GB200 submit; jobs `/mwittmann/mxfp8-002c-g1..g11`
  (g11 = final green bench; g3-g8 were a libNVVM bisect, see gotchas).
- Config: CuTe DSL via cutlass_call. One bf16 read of x[M,K]; 8x8 sub-tile
  per thread, warp-shuffle amax for both orientations, integer-only e8m0
  round-up (exhaustively validated on all 32,640 finite |bf16| amax values),
  inline-PTX `cvt.rn.satfinite.e4m3x2.f32`. No transpose stage needed —
  wgrad consumes the columnwise copy in natural (M,K) storage. Group-dependent
  SF swizzles stay on the XLA path.
- Result:
  - Bit-exact vs the (corrected) XLA reference on all outputs, incl.
    adversarial cases (all-zero blocks, denormals, 2^120, powers of two).
  - **Found a real bug in the XLA reference**: `e8m0_to_f32` via `jnp.exp2`
    is inexact on 217/256 exponent bytes on GPU — the reference was
    quantizing with wrong scales at extreme exponents. Fixed in adapter.py
    (bit-constructed decode). Any e8m0 decode must never use exp2.
  - Perf: (262144,2560) 0.573 ms kernel / 0.658 ms with swizzle (4.8 TB/s
    effective total) vs 1.698 ms XLA = 2.58x; (262144,1280) 0.337/0.379 vs
    0.896 = 2.36x. Producer total for 4 activation tensors: **2.074 ms**
    (was 5.19 XLA; break-even 2.09).
  - **Layer quad: 1.002x** with weight producers amortized; 0.802x if the
    XLA weight producers run every microbatch. Threshold <=1.0 ms NOT met;
    this design's op-count ceiling is ~1.4 ms (issue/latency bound, 25%
    occupancy). <=1.0 ms needs TMA pipelining + in-kernel swizzle; ~0.7 ms
    (=> ~1.25x) realistically needs epilogue fusion.
  - Epilogue-fusion probe: feasible but a targeted rewrite (~1-2 weeks):
    single elementwise point at `torch_scaled_grouped_mm.py:1995-2004`; needs
    w13 column interleave (offline relayout), shuffle-consistent amaxes
    within the epi subtile (both block orientations fit), SIMT side-stores
    bypassing TMA incl. the k_tile_cnt==0 zero-fill path. No EVT multi-output
    mechanism exists in the vendored kernel.
- Interpretation (`exploratory`): H5 partially confirmed — the standalone
  producer removes quantization as the DOMINANT cost (0.58x -> 1.00x) but
  standalone-optimal is structurally short of fusion-grade. MXFP8 grouped is
  now at parity, not a win. Two ranked paths to a real layer win:
  (a) epilogue fusion (1-2 wk kernel work, ~1.25x), and/or
  (b) NEW `MXFP8-H7`: a per-tensor-scaled fp8 grouped kernel on sm100 (vendor
  the non-scaled DSL grouped GEMM, delayed scaling like the dense path) —
  per-tensor quantize is a scalar multiply+cast that XLA fuses for free, so
  GEMM-level ~1.4x could translate to layer-level nearly intact. Same lesson
  as dense (MXFP8-001b): block scaling pays a producer tax per-tensor doesn't.
- New gotchas (durable): GB200 nodes are HETEROGENEOUS for libNVVM —
  `nvvm.cvt.packfloat` fails on most nodes but passes on some (same wheel);
  inline PTX bypasses it. CPU-only sm_100a compile does NOT exercise the
  pod's libNVVM/SASS stage. `iris job logs` tails 1000 lines by default
  (`--max-lines --no-tail` for compile errors). `cutlass.range_constexpr`
  only inside @cute.kernel AST-preprocessed code.
- Next action (decision point for the thread): pick between epilogue fusion
  (a), per-tensor grouped (b), or wiring MXFP8-004 with the current
  break-even producer while (a)/(b) are explored.
