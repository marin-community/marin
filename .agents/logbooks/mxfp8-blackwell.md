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
- `MXFP8-H3` (CONFIRMED, MXFP8-004c @ af2110ac5): the op is fully stateless —
  threaded as a static shard_map closure, no OverwriteWithGradient, no pmax
  cotangent, no optimizer/EMA special-casing. e2e smoke green, loss tracks
  bf16 to ~2e-3/step.
- `MXFP8-H4`: the FP8 wire (per-token scaling, permutation legs only) carries
  over to B200/GB200 unchanged. Next test: enable during MXFP8-003 smoke.
- `MXFP8-H5` (RESOLVED: fusion is the only route, and it's importable): the
  standalone-quantizer branch is closed by adversarial math (even a perfect
  6.2 TB/s kernel leaves ~1.6 ms > 0.7 target; MXFP8-000c). The fusion-grade
  producer exists upstream as MIT-licensed cudnn-frontend fused grouped
  kernels. Superseded by `MXFP8-H8`.

- `MXFP8-H7` (NEW, from the 001b+002c pattern): a per-tensor-scaled fp8
  grouped kernel on sm100 (vendor the non-scaled DSL grouped GEMM, delayed
  scaling like the dense path) beats mxfp8 at LAYER level — per-tensor
  quantize is a scalar multiply+cast XLA fuses for free, so GEMM-level ~1.4x
  survives nearly intact. Numerics: uniform-e4m3 bwd approximation (as in
  `Fp8RaggedDotOp`) or true mixed e5m2xe4m3 (tcgen05 supports mixed A/B
  natively — no wgmma same-dtype restriction on sm100!). Next test: vendor +
  bench; #7271 arbitrates recipe quality vs mxfp8.

### Blocked

(none)

### Falsified / Dead End

- MXFP8-on-Hopper (user question re tokamax): DEAD. tokamax has no MXFP8 on
  any arch (sm90 = weight-only dequant-to-bf16 wgmma; sm100 = W4A8 inference,
  float subchannel scales); TE gates MXFP8 to cc>=10; no public sm90
  1x32-e8m0 kernel; arithmetic ceiling ~bf16 peak (rescale FMA stream ~0.92x
  MMA time at K=32/wgmma). Evidence: MXFP8-000c (2026-07-16).

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

- `MXFP8-H8`: fused cudnn-frontend kernels -> layer 1.39x (MXFP8-004a) ->
  wired op (004c) -> **e2e win: mxfp8 15.51-15.65% MFU vs bf16 14.34% /
  per-tensor 14.57% at B64 GB200x4, loss tracks bf16 to ~4e-4/step,
  `replicated` across 3 nodes** (MXFP8-005 @ bd98e7d0d). Thread goal
  ("mxfp8 at least similar to per-tensor") met and exceeded e2e. Remaining
  headroom items tracked in the MXFP8-005 ranked-gaps list.

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

### 2026-07-16 - Direction decision (mcwitt): push MXFP8 to per-tensor-class efficiency

- User decision on the MXFP8-002c three-way: do NOT pivot to per-tensor
  grouped (H7 deprioritized, kept in queue) — MXFP8's numerical-stability
  advantages justify pushing on performance until mxfp8 performs at least
  similarly to per-tensor fp8. Third-party kernels are in scope to import or
  vendor (explicit question: does tokamax `ragged_dot_mgpu` support mxfp8 on
  Hopper?).
- New work items: (a) third-party kernel survey — tokamax (pinned 0.0.6 in
  levanter `kernels` extra), TE JAX grouped MXFP8 surface, QuACK, triton
  `dot_scaled`, public fused mxfp8 quantizers (fal.ai?); (b) continue the
  engineering paths from 002c: epilogue fusion (SwiGLU+dual-quantize) and/or
  TMA-pipelined quantizer with in-kernel swizzle.

### 2026-07-16 - MXFP8-000c: third-party kernel survey (brief #2) — NVIDIA ships our epilogue fusion, MIT-licensed

- Effort: medium (2 tracks: tokamax source deep-dive; web survey TE/QuACK/CUTLASS/triton/DeepGEMM/fal.ai + adversarial ceiling).
- Date: 2026-07-16. Full agent reports promoted here in condensed form.

#### Headline findings

1. **cudnn-frontend >=1.21 ships open-source (MIT) CuTeDSL fused MoE grouped
   kernels** (`python/cudnn/grouped_gemm/`) — including
   `grouped_gemm_swiglu_quant` (blockscaled grouped GEMM -> SwiGLU ->
   dual-orientation MXFP8 quantize, emitting d + d_col + swizzled
   sfd_row/sfd_col + per-expert amax), `grouped_gemm_dswiglu_quant` (bwd),
   `grouped_gemm_quant` (plain quantizing epilogue), and
   `grouped_gemm_wgrad`. Same file lineage as our vendored kernel (they
   include newer revisions of moe_persistent_scheduler/moe_utils/
   moe_sched_extension). This IS the 002c epilogue fusion, already written:
   the "interleaved" dense variant solves the w13 column-interleave problem
   the probe identified. NVIDIA blog: 1.3x fwd / 2.1x bwd kernel-level on
   GB200. Constraints: SM100+, sf_vec 16/32, m_aligned=256, experts <=1024.
2. **Standalone quantizers cannot reach the 0.7 ms target — adversarial math
   closes that path.** Even a perfect 6.2 TB/s dual-orientation quantizer
   (Cursor's, closed-source) only takes our 4-tensor producer 2.07 -> ~1.6 ms.
   Everyone credible (Cursor, NVIDIA, CUTLASS ex.92, QuACK) gets
   producer-free via epilogue fusion. (Our SIMT kernel's ~4.8 TB/s sits at
   the known cp.async plateau; fal recipe = one bulk-TMA per CTA tile +
   packed 4-byte scale stores; torchao's 5.9 TB/s single-orientation kernels
   are BSD-3 and vendorable if a standalone is ever needed.)
3. **MXFP8-on-Hopper is DEAD** (answers the tokamax question): tokamax has no
   MXFP8 anywhere (0.0.6/0.0.12/HEAD; sm90 path is weight-only
   dequant-to-bf16 before wgmma — no fp8 TC use on Hopper at all; sm100 quant
   kernels are W4A8 inference, float subchannel-128 scales, no sm100 wgrad).
   TE hard-gates MXFP8 to cc>=10.0. No public sm90 1x32-e8m0 kernel exists,
   and the arithmetic says why: fp8 wgmma is K=32/instruction, so per-block
   rescale costs ~0.92x the MMA time in CUDA-core FMAs -> ceiling ~1030 TF/s
   ~= bf16 peak. DeepSeek's 1x128 (0.23x overhead) is the sm90 frontier.
4. **Our GEMM is at ~80-84% of the honest ceiling, not 50%.** Measured cuBLAS
   mxfp8 dense tops at ~2.5-2.7 PF/s (not the 4.5 spec); Cursor's grouped
   kernel ~2650 TF/s at wide-N shapes. Verdict: ~10-20% GEMM headroom, best
   captured by the fused epilogues + scheduler tuning, not mainloop heroics.
5. Other options graded: TE common lib (Apache-2.0, torch-free
   libtransformer_engine.so) exposes nvte_swiglu-with-MXFP8-output,
   nvte_group_quantize (dual-orientation one-pass), swizzle, and
   nvte_grouped_gemm (cuBLAS 13.3+ for MXFP8 grouped; kMaxGroups=64 = our E);
   te.jax grouped_dense MXFP8 landed in v2.15 — cheapest honest baseline.
   QuACK (Apache-2.0, JAX bindings): blockscaled epilogue with SF-direction
   knob on bf16-input GEMMs (varlen_m MoE fwd + varlen_k wgrad) — cleanest
   for making UPSTREAM dense GEMMs emit mxfp8. Triton dense-only 2378 TF/s,
   no grouped MX (low EV). DeepGEMM sm100 impl can express gran_k=32 but
   default recipe is 1x128 (medium-low EV). jax 0.10.1 Pallas already
   exposes tcgen05_mma with scale refs (pure-JAX fallback).
6. Prerequisite for most of the above: bump `nvidia-cutlass-dsl` pin past
   `<4.6` (lib/levanter/pyproject.toml:89); the v4.6 drop also updated our
   vendored torch_scaled_grouped_mm.py (diff it) and 4.6.1 improves JAX FFI
   registration. Re-run 002c bit-exactness after the bump.

#### Plan (supersedes the 002c three-way)

- MXFP8-004a (primary): vendor cudnn-frontend fused grouped kernels behind
  our cutlass_call adapter; wire fwd w13 (SwiGLU+quant), w2 (quant epilogue),
  dSwiGLU+quant, wgrad. Gate: layer-quad >=1.2x vs bf16. Fallback: same
  fusion in C++ CUTLASS ex.92 via a small FFI .so.
- MXFP8-004b (parallel, cheap): bench TE 2.15+ te.jax grouped_dense MXFP8 at
  our shapes on a pod — the NVIDIA-tuned narrow-N datapoint; time-boxed
  (arm64 install risk).

### 2026-07-16 - MXFP8-004b: TE grouped_dense MXFP8 baseline — our kernels at parity or ahead of cuBLAS

- Hypothesis: reference datapoint for H8 (NVIDIA-tuned narrow-N grouped MXFP8).
- Commit Hash: 97cf4ac0b (bench_te_grouped.py + run_te_bench.sh install recipe
  + bench_te_grouped_gb200.json).
- Command: jobs `/mwittmann/mxfp8-004b-g1..g6` (g6 clean); TE 2.16.0 on
  aarch64 = cu13 wheel + metapackage + jax-glue sdist compiled on-pod (~105 s,
  needs synthetic CUDA_HOME from pip header wheels + libnccl link; 2.16.0 is
  the LAST version with an aarch64 cu13 wheel). Pod: cuBLASLt 13.4.1 (>=13.3
  MXFP8-grouped gate met), CUDA 13.0, cuDNN 9.19.
- Result (M=262144, E=64 uniform, median 50):

  | arm | w13 | w2 |
  |---|---|---|
  | TE mxfp8 gemm-only (pre-quantized) | 2085 TF/s | 1950 TF/s |
  | TE mxfp8 fwd incl. quantize | 1230 | 1414 |
  | TE mxfp8 fwd+bwd | 1499 | 1606 |
  | TE bf16 fwd / fwd+bwd | 1310 / 1172 | 1323 / 1218 |

  Numerics: mxfp8 fwd 3.7e-2 vs f32 ref; dgrad/wgrad 5.7e-2 vs TE bf16 grads
  (TE uses an e5m2 cotangent per its HYBRID-compatible path).
- Interpretation (`exploratory`):
  - Our CuTeDSL kernels (2130/2027 fwd, 2041/2112 dgrad, 2209/2214 wgrad) are
    at PARITY on w13 fwd and 4-12% AHEAD elsewhere vs NVIDIA-tuned cuBLAS
    grouped MXFP8 at our narrow-N shapes — no free GEMM upside left in TE;
    cuBLAS-via-TE is a legitimate fallback, not an upgrade.
  - TE e2e mxfp8 fwd+bwd = 1.28-1.32x its own bf16, with the fwd leg
    quantize-bound (mxfp8 fwd barely beats bf16 fwd) — independent
    confirmation of the fusion thesis.
  - Constraint worth remembering: TE V2 MXFP8 grouped requires K, N AND every
    group size %128 — real skewed routing violates this; our kernel has no
    such restriction.
- Next action: awaiting MXFP8-004a (cudnn-frontend fused kernels).

### 2026-07-16 - MXFP8-004a: cudnn-frontend fused kernels green — layer 1.39x vs full bf16, gate met

- Hypothesis: MXFP8-H8 (vendored fused kernels reach layer-quad >=1.2x).
- Commit Hash: aa998a01f (vendor + adapter) .. cf979bb4c (bench/baselines);
  all pushed. Vendored from NVIDIA/cudnn-frontend @ 3041f3e88 (MIT) into
  `experiments/grug/moe/standalone/mxfp8_fused/`: grouped_gemm_swiglu_quant,
  grouped_gemm_dswiglu_quant, grouped_gemm_quant, wgrad + newer scheduler/
  utils revisions, torch stripped, driven via cutlass_call.
- Command: jobs `/mwittmann/mxfp8-004a-g1..g11` (g11 definitive full quad).
- Pin bump NOT needed: a 6-site `atomicrmw_compat` shim runs upstream code on
  stock 4.5.2 — and the arm64 cu13 wheel already has the 4.6-style signature
  AT version 4.5.2 (wheel heterogeneity across arches at the same version!).
- Correctness (3 configs): swiglu_quant h/h_col/sfh_row/sfh_col BIT-EXACT vs
  our corrected reference (fastmath-silu concern did not materialize);
  dswiglu_quant likewise bit-exact, dSwiGLU cross-validated vs jax.vjp; quant/
  wgrad legs 6e-6..3.6e-5 rel-frob. Fused discrete col-SF layout is
  byte-identical to our build_sf_wgrad layout — fused outputs feed wgrad with
  zero glue.
- Result (g11, M=262144, E=64, REAL gated shapes D2560/F1280 — w13 output is
  2F=2560 wide, so numbers are NOT comparable to the earlier half-width
  6.95 ms baseline):

  | leg | ms | TF/s |
  |---|---|---|
  | fwd w13 SwiGLU+dual-quant | 1.500 | 2291 |
  | fwd w2 (bf16 out) | 0.929 | 1849 |
  | dgrad-w2 + dSwiGLU + dual-quant | 1.381 | 1244 |
  | dgrad w13 (bf16 out) | 1.327 | 2589 |
  | wgrad w13 / w2 | 1.427 / 0.784 | 2408 / 2191 |
  | + remaining producers (x, g2 dual-quant, CuTe) | 1.429 | |
  | **total** | **8.777** | |

  Baselines (same node): bf16 6-GEMM 9.893 ms; full bf16 layer incl. XLA
  SwiGLU/dSwiGLU 12.220 ms. **Speedups: 1.39x vs full bf16 layer / 1.13x vs
  strict 6-GEMM / 1.35x gemm-only.** Gate >=1.2x MET on the honest full-layer
  accounting (the fused pipeline subsumes the elementwise work bf16 pays
  separately); 1.13-1.18x across nodes vs the GEMM-only yardstick.
- Remaining costs, ranked: (1) the dswiglu leg (1244 TF/s — C-read +
  dual-store epilogue), (2) the x/g2 dual-quant producers (1.43 ms — foldable
  upstream into attention/dgrad epilogues), (3) best-tiler notes: swiglu/
  dswiglu/gemm (256,256) c(2,1), wgrad (256,256) c(2,2) [2868/2452 TF/s].
- New gotchas (durable): arm64 vs x86 cutlass-dsl wheels DIFFER at the same
  4.5.2 version (smoke both); cutlass_call compile failures on the FFI thread
  are process-fatal — probe risky kernels in a subprocess (002c quantizer
  fails libNVVM on ~3/4 of nodes; the fused kernels compiled everywhere —
  all-inline-PTX); same kernel measures ~25% faster after compile-idle than
  in-sequence (report in-sequence); quant kernel needs d_col=d even with
  generate_sfd=False.
- Interpretation (`exploratory`): H8 CONFIRMED. MXFP8 grouped now delivers a
  real layer-level win in the per-tensor class, with identified headroom
  (dswiglu epilogue, upstream producer folding).
- Next action: MXFP8-004c — wire `MxFp8RaggedDotOp` on the fused kernels
  (256-padded routing) into the grug MoE op path; then MXFP8-005 full-step
  MFU.

### 2026-07-16 - MXFP8-004c: op wired, e2e train smoke green — mxfp8 loss tracks bf16

- Hypothesis: MXFP8-H3 (stateless op, no train-step machinery) + integration.
- Commit Hash: af2110ac5 (pushed; tree clean).
- Command: jobs `/mwittmann/mxfp8-004c-g1..g5` (g2 op-level, g3/g4/g5 smoke
  arms on GB200x4).
- Design (details in commit + report): new `MoeExpertMlpOp` Protocol in
  levanter `_moe/common.py`, threaded through `moe_mlp` into ring AND a2a EP
  backends as a static shard_map closure (stateless — H3 CONFIRMED: no amax
  history, no OverwriteWithGradient, no pmax cotangent). Concrete
  `MxFp8MoeMlpOp` lives in `experiments/grug/moe/mxfp8.py` (whole expert MLP
  w13->SwiGLU->w2 as ONE custom_vjp — forced by the fused kernel boundary);
  levanter never imports the vendored package. Residuals: x_col/h_col+SFs +
  bf16 c13; dgrad weight copies re-quantized in bwd (~16 GB residual savings
  at row-13). 256-aligned repack with traced SF layouts (wgrad atom-block
  perm verified bit-equal to host layout). `producer="auto"` probes the CuTe
  quantizer in a SUBPROCESS (FFI compile is process-fatal) with XLA fallback.
  Config: `GrugFp8Config(recipe="per_tensor"|"mxfp8", grouped, mxfp8_producer)`;
  mxfp8 requires wire=False, grouped=True; dense stays per-tensor; non-sm100
  rejected at trace time. Tilers pinned to 004a winners.
- Result:
  - g2 op-level (skewed routing, 2 zero-token experts): dequant-ref checks
    7.4e-6..2.7e-5 (gate 1e-3), quantizer checks 3-5e-5, blackbox vs bf16
    6.6-6.7e-2 (chained fwd+bwd double-quantization noise class), zero-token
    wgrads exactly 0. ALL PASSED.
  - Train smoke (GB200x4, EP4 ring, d2560/F1280/E64/K4, L13, B32, seq4096,
    MuonH, 20 steps): bf16 / per-tensor-dense / mxfp8 all finite, no NaN.
    Loss step19: 10.846 / 10.847 / 10.848 — mxfp8 tracks bf16 to ~2e-3 per
    step. Steady median s/step 1.2608 / 1.2559 / 1.2718 (mxfp8 ~1% slower at
    this collective-dominated smoke config; weights re-quantized every
    microbatch and XLA producer in play — perf verdict is MXFP8-005's job).
  - CPU tests: 31 passed (experiments) + 6 passed/1 GPU-skip (levanter seam);
    existing ragged-dot-op tests still green.
- Caveat: both GPU jobs landed on bad-libNVVM nodes, so the e2e ladder
  exercised the XLA producer; the CuTe producer path in the op is untested
  e2e (rests on 002c bit-exactness) — MXFP8-005 item.
- New gotchas (durable): the nix-profile `iris` CLI is Python 3.13 and pins
  the pod interpreter -> dep sync breaks (resiliparse cp312-only); submit
  with `.venv/bin/iris`. `./infra/pre-commit.py --fix` reformats the vendored
  mxfp8_fused/ kernels — revert those after repo-wide fix passes. A global
  ~/.config/git/ignore has `lib/` — `git add -f` for new files under
  lib/levanter/tests.
- Next action: MXFP8-005 — row-13 full-config MFU on GB200x4 (comparable to
  the #7012 4-GPU numbers: EP1 17.01% / ring EP4 16.32% / a2a 15.82%
  B200-conv) with per-phase timing; weight-quantize amortization across
  microbatches; exercise CuTe producer on a good node; one a2a arm.

### 2026-07-17 - MXFP8-005: e2e MFU — mxfp8 beats bf16 (+1.17pp) and per-tensor (+0.94pp); best 15.65%

- Hypothesis: H8 e2e leg + H5 fusion thesis at step level.
- Commit Hash: bd98e7d0d (save-qweights knob, --profile-steps,
  trace_phases.py). Jobs `/mwittmann/mxfp8-005-g1..g13` (results from
  g7-g13; g1-g6 were shell-quoting/B128-OOM casualties).
- Config: GB200x4, EP4, d2560/L26/E64/K4/seq4096, MuonH, recompute_all,
  20 steps, steady-median, B200-conv MFU. **B64** — B128 does NOT fit this
  branch (182.8 GiB/dev vs 133.6 budget; #7012 ran B128 only via
  slim-vjp/all_but_moe, not yet on this stack), so #7012's 17.01/16.32/15.82%
  references are NOT directly comparable (2x batch + QuACK-cute bf16 there).
- Result (same-node 4-leg = g7; ladder internally comparable):

  | arm | s/step | MFU | loss@19 |
  |---|---|---|---|
  | bf16 ring | 3.4635 | 14.34% | 10.5299 |
  | per-tensor dense-only | 3.4094 | 14.57% | 10.5374 |
  | **mxfp8 ring (XLA producer)** | **3.2020** | **15.51%** | 10.5385 |
  | mxfp8 + save-qweights (g8) | 3.1725 | **15.65%** | 10.5386 |
  | mxfp8 forced CuTe producer (g9) | 3.3941 | 14.63% | 10.5385 |
  | mxfp8 a2a B64 | OOM (XLA ragged_a2a 165-168 GiB alloc, even mf 0.92) | | |
  | mxfp8 a2a B32 (+one-shot-off flag) | 2.2403 | 11.08% | vs ring B32 12.43% |

  mxfp8 loss tracks bf16 to 8.5e-3 over 20 steps (~4e-4/step); identical
  across 5 nodes and XLA-vs-CuTe producers to <=7e-5. Ring MFU replicated
  across 3 nodes (15.51/15.57/15.58) — label `replicated` for the ring arm.
- Per-phase attribution (xplane trace, ms/GPU/step, mxfp8 arm): MuonH NS
  optimizer ~833 (26%, dominant at B64), fa4 attention ~542, EP collectives
  ~287, **fused MXFP8 expert kernels 200.8** (vs bf16's ~1,100-1,150 ms
  MoE-MLP block — the whole block went 1.1 s -> 0.49 s), XLA producers ~185,
  dispatch/pad ~205, fp8 dense ~143, transposes incl. w13 interleave ~75.
  Producers+repack (~290) cost 1.4x the fused GEMMs — fusion thesis holds e2e.
- Weight-quantize amortization: train.py has NO grad-accumulation loop (1
  microbatch/step), so the duplication was remat recompute; fixed via
  `checkpoint_name` on the 4 fwd qweight outputs +
  `GrugFp8Config.mxfp8_save_qweights` remat-policy extension: -13.9 ms/step
  (+0.07pp), loss delta 5.8e-5 (gate passed).
- CuTe producer ran e2e for the first time (g9, numerics match to 5e-5) but
  on a different node measured slower than XLA-producer arms — slow-node vs
  genuinely-slower unresolved; needs same-node A/B.
- Ranked gaps to bf16+1.5-2pp (measured +1.31pp with save-qweights):
  1. MuonH NS optimizer 26% share -> amortize via B128 (needs #7012 slim-vjp
     port) or bf16/sharded NS.
  2. Producer folding (~185 ms) + pre-interleaved w13 storage (~75 ms):
     ~0.5-0.8pp.
  3. B128 memory wall = slim-vjp port (also halves optimizer share).
  4. bf16 denominator is soft (triton+XLA loop, not QuACK-cute) — port QuACK
     bf16 grouped for the honest comparison.
  5. dswiglu epilogue (1244 TF/s), ~10-15 ms/step.
  6. a2a B64 memory pathology is upstream XLA (single 165-168 GiB alloc in
     ragged_all_to_all lowering); a2a op-threading itself validated (loss
     matches ring to 3e-5). Always set the one-shot-kernel-off flag.
- New gotchas: this Bash tool doesn't word-split unquoted vars (iris args must
  be literal or in `bash -c`); B64 is the 4-GPU ceiling for recompute_all
  L26; GPU trace events carry only XlaModule metadata — classify by kernel
  symbol (MuonH NS identifiable by count E_local*L*2*5); logs snapshotted in
  scratch/mxfp8-005/ (gitignored).
- Next action: thread goal MET (mxfp8 > per-tensor e2e). Follow-ups ranked
  above; biggest single lever is the slim-vjp/B128 port, then producer
  folding. [Superseded same-day by the production-viability re-rank below.]

### 2026-07-17 - Direction decision (mcwitt): production-viability filter — slim-vjp/B128 port dropped, follow-ups re-ranked

- Trigger: user asked whether slim-vjp/`all_but_moe` are viable for a
  production run (>120B total / >15B active params) and directed: do not
  spend benchmark time on configs that are not production-viable.
- Evidence (from the #7012 branch logbook,
  `research/mcwitt/7012-b200-mfu:.agents/logbooks/7012-b200-moe-mfu.md`,
  entries B200MFU-015/-029/-032 — that thread has advanced past our last
  sync): at the production reference config (d5120 L48 E64 top4 b1024,
  64xGB200 rep2 DDP, 16 seq/GPU ≈ the >120B/~14B-active scale asked about),
  `all_but_moe` wants 439-627 GiB and `none` 1.43-1.62 TiB of step
  temporaries vs the 138 GiB pool — 3-12x over budget. **`recompute_all` is
  the only viable remat mode at production scale**, and slim residuals
  standalone are a measured −0.28pp tax under recompute_all. Also: the
  MuonH-NS 26% share at our B64 4-GPU grid is a small-bench artifact —
  optimizer share at the reference config is ~4% (B200MFU-007/-032), so the
  "amortize NS via B128" motivation evaporates too.
- Consequence: **MXFP8-005's B64 `recompute_all` ladder was already the
  production-representative mode**; the mxfp8 win (+1.17pp vs bf16, +0.94pp
  vs per-tensor) needs no remat-mode caveat. What is NOT yet
  production-representative is the *shape/scale* (d2560 L26 4-GPU vs d5120
  L48 multi-node, where #7012 measured the step becoming
  collective-dominated at >=32-way — ~48% collectives at d2560 EP4 32-GPU,
  which compresses any GEMM-side win).
- New ranked follow-ups (production filter applied):
  1. **MXFP8-006: production-shape ladder** — mxfp8 vs per-tensor vs bf16 at
     d5120/L48/E64/K4, 16 seq/GPU, recompute_all, multi-node EP on GB200
     (as close to the B200MFU-032 reference as our node budget allows).
     Includes fused-kernel tile re-tune at d5120/F2560 shapes,
     save-qweights memory re-probe under the tighter budget, and the
     one-shot-off a2a flag. This is now the only benchmark that decides
     whether the +1.17pp survives where it matters.
  2. Producer folding (~185 ms) + pre-interleaved w13 storage (~75 ms):
     production-viable and double-dips under recompute_all (producers run in
     both the forward and the remat re-run).
  3. dswiglu epilogue headroom (1244 TF/s leg): mode/scale-independent
     kernel work.
  4. QuACK bf16 grouped denominator port: the production driver runs
     QuACK-class bf16, so the honest comparison arm needs it at (1)'s scale.
  5. CuTe-producer same-node A/B: only if (2) doesn't subsume the producers.
  6. Quality gate hand-off to #7271 (compute-optimal wall-time-matched
     MXFP8 vs BF16): unchanged, runs in parallel — it's the numerics case
     for mxfp8 and is scale-relevant regardless of the perf ladder.
- Dropped (production filter):
  - slim-vjp/`all_but_moe`/B128 port: falsified at production scale by
    B200MFU-032 (3-12x memory over budget; slim standalone is a perf tax).
    Was only ever a small-grid comparability item.
  - a2a-at-B64 XLA pathology chase: 4-GPU bench artifact; production EP
    collective work (EP16/32 CUBIN bug, EP32 SPMD-remat OOM) lives on
    #7012/#7279, not this thread.
- Next action: launch MXFP8-006 (production-shape ladder).

### 2026-07-17 - MXFP8-006 prep: d5120 tile tune clean + first multi-node gang green

- Hypothesis: the fused kernels and the wired op carry to the production
  shape (d5120/F2560, E_local=8 under EP8) and to multi-node gangs.
- Commit Hash: e1f400e13 (gang init via `iris.runtime.jax_init.initialize_jax`
  in the MFU harness; no-op single-process), 4ff2d1198
  (`--d-model/--f-dim/--experts` on bench_mxfp8_fused.py).
- Command: jobs `/mwittmann/mxfp8-006-tune1` (GB200x1, `--d-model 5120
  --f-dim 2560 --experts 8 --tokens 262144 --ablate`) and
  `/mwittmann/mxfp8-006-smoke2n` (2x GB200x4 gang, d2560/L13/B64 EP8 ring,
  bf16 + mxfp8 chained, 10 steps).
- Result (tune1, M=262144): all six legs pass correctness at the new shape
  (relfrob 1.2e-5..9.9e-4); swiglu 2434 / gemm_w2 2440 / dswiglu 1648 /
  gemm_w13 2484 / wgrad_w13 2393 / wgrad_w2 2554 TF/s — throughput UP vs the
  d2560 shape (bigger K); ablation confirms the 004a default tilers
  ((256,256) c(2,1); wgrad c(2,2)) remain best — no re-pin needed. dswiglu
  is still the laggard leg.
- Result (smoke2n): gang init works (process 0/2 + 1/2, 4 local devices
  each); cross-node EP8 ring clean, NO CUBIN error at this size; bf16
  10.93% vs mxfp8 11.50% MFU (B200-conv) even at the smoke config; mxfp8
  loss tracks bf16 to ~5e-3 over 10 steps (10.9 loss class). mxfp8 arm
  compile ~15 min (916 s first step).
- Interpretation: no kernel-side blocker for the production shape; the
  EP>=8-ring CUBIN hazard (#7012 B200MFU-032) did not appear at 2 nodes —
  still unprobed at 8-node buffer sizes.
- Next action: `/mwittmann/mxfp8-006-ladder` submitted — 8x GB200x4 (32
  GPUs), d5120/L48/E64/K4, B512 (16 seq/GPU), EP8 ring, recompute_all,
  MuonH, arms bf16(+trace) -> mxfp8(+trace) -> per-tensor-dense -> mxfp8+
  save-qweights, 20 steps each.

### 2026-07-17 - MXFP8-006 ladder attempt 1: step-0 OOM (851.61 GiB temp arena) — killed, diagnosing off-gang

- Hypothesis: the MXFP8-005 stack runs unchanged at the production shape
  (d5120/L48/E64/K4, B512 = 16 seq/GPU, 8x GB200x4, EP8 ring,
  recompute_all).
- Commit Hash: 0d728ba89 (compile_probe.py added after the failure).
- Command: `/mwittmann/mxfp8-006-ladder` (8 replicas GB200x4, 4 chained
  arms). Full log harvested post-mortem (job killed 21:40 UTC).
- Result: NOT a compile hang. ARM-bf16 compiled in ~30 min (18:57->19:29
  UTC incl. init) then died at step 0:
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 851.61GiB`
  (the whole jit_train_step temp arena; pool is 138 GiB at the 0.75 mem
  fraction). The watched "3 h compile" was ARM-mxfp8 (chain continued past
  the failure); its compile emitted nothing for 83 min before the kill —
  mxfp8-arm compile length at 32-way is its own open question (CuTe
  subprocess probes run at trace time). All four arms would have OOM'd.
- Off-gang diagnosis so far (compile_probe.py — AOT lower+compile with
  abstract avals + real output shardings, prints memory_analysis, never
  executes; runs on CPU fake devices or small GPU gangs):
  - CPU probe (32 fake devices, reference attention): temp 627.99 GiB at
    B32/L48, 163.07 GiB at B32/L12 — linear in L (12.9 GiB/layer),
    near-flat in B. BUT the buffer dump shows the CPU arena is dominated
    by `f32[40,4096,4096]` reference-attention score matrices — a probe
    artifact (GPU runs FA4, no S^2 buffers). CPU probes of this graph
    CANNOT stand in for GPU memory behavior unless attention is excluded;
    recorded as a probe-fidelity gotcha.
  - GPU probes in flight on 2-node gangs (real FA4/backend, B128 = same
    16 seq/GPU): `/mwittmann/mxfp8-006-probe2n` (EP8 default vs
    latency-hiding-scheduler-off A/B) and `/mwittmann/mxfp8-006-probe2nb`
    (EP2/EP4 — at 8 GPUs EP8 degenerates to data=1, but the failing
    32-GPU mesh was (data=4, expert=8) and the SPMD "involuntary full
    rematerialization" reshards fire on the data axis).
- Structural facts recorded while reading the branch model: the layer
  stack is a Python for-loop of 48 individually-`eqx.filter_checkpoint`ed
  blocks (NO lax.scan on this branch); nothing data-forces remat-forwards
  to serialize in the backward, so scheduler-driven concurrent remat is a
  live hypothesis alongside SPMD reshard synthesis. Suspect list also
  includes MuonH NS whitening (must gather full weight matrices; no data
  dependence between per-weight updates -> XLA free to gather all at
  once; batch-independent, ∝ params ≈ the observed arena class).
- Ops notes: parallel session switched the shared checkout to
  research/mcwitt/7331-nccl-ep mid-day — this thread now works from a git
  worktree at ~/projects/marin-7282 with PYTHONPATH prepended over the
  venv editables (same class of hazard as B200MFU-014's stale-editable
  incident). Log reads against RUNNING jobs kept light per the 2026-07-17
  wedge finding (heavy reads freeze the training process); `--no-tail`
  reads from the log HEAD, not the tail — use plain `--max-lines N` for
  tails.
- Next action: read GPU probe memory_analysis; identify the arena owner;
  fix; re-probe; only then relaunch the ladder.

### 2026-07-17 19:40 - MXFP8-006 OOM root cause: FA4 bounds built per-block on device 0 (#7012's poison, ported fix)
- Hypothesis chain closed out this session:
  - `eqx.error_if` (callback→device-0) — **falsified**. probe2ng reran the
    792.47 GiB baseline config with `EQX_ON_ERROR=nan` (verified locally:
    nan mode lowers to a pure `lax.cond`, no callback; env var confirmed
    delivered in the job command): temp arena **byte-identical 792.47 GiB**.
    On inspection none of the grug `error_if` sites even trace in this
    config (`same_segment_ids` is True; no `max_segments`/THD metadata).
  - Re-read of the probe2nf allocation dump: the arena is dominated by
    **full-global-batch activations materialized per GPU** — 387 values of
    bf16[524288,5120] (524288 = 128×4096 = the whole global batch, 5 GiB
    each, ≈ 48 layers × 8) plus 432 half-batch values. Classic
    "involuntary full rematerialization" (replicate-then-partition through
    device 0). The `pure_callback.384` first-value line in the table was a
    red herring for attribution (an 8-byte tuple slot), but the callback
    itself is real and FA4-only: the CPU/reference dump has **zero**
    callbacks, so it lives in the FA4-cute/cutlass bridge.
- Root cause (matches #7012's B200MFU work exactly): the FA4 per-token
  metadata (lower_bounds, valid) is computed **inside every one of the 48
  `eqx.filter_checkpoint`ed blocks**. The `jnp.arange`/`jnp.zeros`
  constants it's built from are placed on `{maximal device=0}`, making the
  [B,S] bounds device-0; the downstream reshard to the batch-sharded spec
  becomes a full-batch replicate-then-partition **per layer** (and again in
  each backward recompute under recompute_all). At d2560 the arena stayed
  under the pool so it was invisible; at d5120/L48 it is ~16.3 GiB/layer.
- Fix: ported #7012's attention diff verbatim (it is the entire
  attention-dir delta between the branches): replicated `P(None,None)`
  pins on the metadata constants, `AttentionMask.fa4_bounds` +
  `with_fa4_bounds`, exported `fa4_cute_segment_bounds`, metadata sharding
  follows q's actual batch axes; bench model now precomputes short/long
  bounds ONCE outside the layer loop and threads them through the masks
  (plus batch-reshard pin on segment_ids). Local CPU sanity: bounds
  compute + mask plumbing + model import green.
- Commit Hash: 0ef1d613c
- Command: /mwittmann/mxfp8-006-probe2ni — 2-node AOT probe chain, bf16
  arm + mxfp8 arm at the ladder shape (d5120/L48/E64/top4, B128≙16
  seq/GPU, ring EP8, recompute_all). Also answers "does mxfp8 cost extra
  temp memory at production shape" (mcwitt asked whether fp8 memory
  behavior threatens production runs — answer so far: the OOM was
  bf16-first and dtype-agnostic; this probe gives the honest fp8 delta).
- Next action: if temp collapses (needs ≲138 GiB pool at B512-equivalent
  scaling) → relaunch the 8-node ladder on the fixed commit; append fp8
  temp delta here; issue comment with the root cause + fix.

### 2026-07-17 21:10 - MXFP8-006 probes: FA4-bounds fix ALSO falsified; arena is activation-class
- Correction to the 19:40 entry: the ported #7012 FA4-bounds fix is NOT the
  arena owner. probe2ni/probe2nj (markers `levanter_fa4_bounds=True
  model_fa4_bounds=True` prove the pods ran the fixed code; probe2ni's
  526-580 s train_step compile = cache miss = the fix did change the HLO)
  still report temp 792.47 GiB at d5120/L48/B128 ring-EP8. The fix is kept
  (it is correct and matches #7012 production), but the dominant term is
  elsewhere.
- probe2nj B64 arm: temp 340.11 GiB (0.43x of B128's 792.47) — the arena
  scales ~linearly with GLOBAL batch => activation-class full-batch
  materialization, not gradient/optimizer memory. Even B64 (8 seq/GPU) is
  2.5x the 138 GiB pool: no batch retreat rescues the unrolled model.
- Census attempt failed benignly: the fixed program hit the XLA compile
  cache (24.8 s), and cache hits write no dump — the census parsed a
  stray jit_multiply module. Next census run must disable the JAX
  compilation cache for the dump arm.
- External datapoint (#7012 issue, 32-way records): d5120 ring_cute EP8
  ran CLEAN at 16.2% MFU on 32 GPUs — on the scan/`use_array_stacked_blocks`
  bench. Prime remaining suspect: our 48 individually-checkpointed
  UNROLLED blocks let the XLA scheduler keep ~16.3 GiB/layer live
  concurrently (~26x the residual-stream save); EP1 pure-DP probe at
  581 GiB says it is not MoE-machinery-specific.
- Commits: 0a3eda269 (probe markers + instruction-keyed census + mxfp8
  wire default). Jobs: /mwittmann/mxfp8-006-probe2n{g,i,j}.
- Next action: harvest mxfp8 arm delta; rerun dump arm with
  JAX_ENABLE_COMPILATION_CACHE=false for the instruction census; decision
  point after census: sharding-annotation fix vs scan-over-blocks port.

### 2026-07-17 22:35 - MXFP8-006 census lands: arena = MoE dispatch/combine across unrolled layers; scan port in validation
- probe2nk (cache-off dump, bf16 B128 ring-EP8 d5120/L48) instruction-keyed
  census of the 792.47 GiB arena (footprint by shape, defining ops):
  - bf16[524288,5120] x336 (1680 GiB): input_scatter_fusion x96 (2/layer,
    fwd+bwd MoE combine scatter at FULL GLOBAL token width) +
    loop_broadcast_fusion.remat variants x86 — full-batch materialization.
  - bf16[262144,5120] x336 (840 GiB): pallas_call x240 (5/layer dispatch/
    sort kernels at topk-expanded width 65536 local tokens x top4) +
    concatenates.
  - Attention appears only at per-GPU shapes (bf16[16,4096,40,128] x888,
    555 GiB class) — NOT the driver. Neither is MuonH (0 optimizer shapes
    in the top classes).
  - Verdict: the arena is the ring-MoE dispatch/combine + rematerialized
    broadcast buffers of ALL 48 unrolled layers kept concurrently live by
    the scheduler. Explains B-linearity (340 GiB at B64) and EP1's
    581 GiB (dispatch buffers exist under pure DP too).
- Scan port (e8e105d4e + train-state fix): use_array_stacked_blocks
  behind a config flag; Bool[L] mask schedule + precomputed FA4 bounds via
  jnp.where; traced disable_rope; 4D expert stacks; optimizer masks route
  stacked norm gains->adam, 4D->muonh; grugmuon adopted from
  mcwitt/moe-standalone-ep tip (distributed 4D NS). CPU checks green
  (unrolled regression, stacked init/initial_state).
- probe2ns/2nt failures during validation: (a) train.py iterated
  params.blocks (fixed, stacked-aware qb-betas); (b) autotuning OOM'd
  allocating 75.00 GiB = bf16[48,64,5120,2560] — _newtonschulz_4d_distributed's
  (L,E) merge reshape drops expert sharding onto P(None,"data","model"),
  which REPLICATES on the 2-node probe mesh (data=1). Probe-mesh
  degeneracy, not a production blocker (data=8 keeps it 8-way sharded at
  ~9.4 GiB/chip). probe2nu re-running with SCALE_MUON_NO_NS=1 (grugmuon's
  own memory-probe knob) to measure the scan model's fwd/bwd arena.
- Jobs: /mwittmann/mxfp8-006-probe2n{k,s,t,u}.
- Next action: probe2nu memory analysis -> if arena collapses under the
  138 GiB pool, relaunch the ladder in scan mode (32 GPUs, data=4 keeps
  4D NS sharded); then loss-parity smoke unrolled-vs-scan at d2560.

### 2026-07-17 23:45 - MXFP8-007: #7201-best ports + smoke PASS; 128-GPU comparison blocked by gang churn, serial driver armed
- Direction (mcwitt): base the production test on the #7201 current best
  (ClassicLarry issuecomment-5010023509: MuonH d5120 5-of-128 i2560 sh2560
  48L GQA40/10 sw2048, 19.0% MFU / 591,330 tok/s at 128 GPUs on the GB200
  2.5 PF dense convention, branch grug/embedding-gather-shard-map); port
  its wins, smoke the mxfp8 config, then apples-to-apples.
- 007a ports (6a23eb79b): replica-local embedding gather + replicated
  embed table (8-rack wedge fix), lm_head intra-rack sharding, liger
  chunked CE (CE_IMPL=liger), SCALE_MUON_SYRK QuACK NS (env-gated, NOT
  enabled: quack-kernels pins cutlass-dsl 4.6, our mxfp8 kernels are
  validated on 4.5.2 — small known baseline advantage, NS is ~4% share),
  --num-kv-heads GQA flag, mfu_gb200 readout (2.5 PF), --replica-axis.
- 007b smoke (2-node EP8, d5120/L8/E128top5/GQA/scan, B64): fused kernels
  at E_local=16 all-legs correct (relfrob<=9.5e-4, 1836-2646 TF/s, 1.588x
  vs bf16 layer); e2e bf16 21.0% GB200-MFU / 205k tok/s vs mxfp8 24.25% /
  237k (+15.5%); loss parity ~6e-3 over 15 steps. Two harness bumps fixed:
  bench --run-id/--output-dir required; bench wire default for mxfp8.
  Sizing note: L13 OOMs at 8 GPUs (state ~86 GB/GPU: E128 experts shard
  8-way but attn/embed replicate at data=1) -> smoke at L8.
- 007c baseline replication: Larry's launcher runs on THIS controller
  (data cache in place); env reconstruction VERIFIED exact via the train
  job's hparams dump (scan on, muonh lr 0.05, replica 2, watch@20).
  Three baseline attempts + one treatment died 1-5 min in with NO in-pod
  error: NVLink-domain gang scheduling (16-node domains; each 32-node gang
  needs 2) recomposes gangs when the parallel session (#7331, same
  mwittmann user) admits its own 32-node gangs — it is cycling
  ep128-ring4/a2a8/ring8 every few minutes, and each admission killed
  whatever was mid-placement (incl. its own muonprobe-fix). Cluster is 201
  GB200 nodes / 804 GPUs, so it is churn, not capacity.
- Mitigation: serial auto-retry driver (waits for <1 other large job, max
  3 baseline + 2 treatment attempts, treatment with --max-retries 2);
  jobs mx7201-base{4..6}-coord, mx7201-treat{2,3}.
- Jobs: /mwittmann/mxfp8-007-smoke{,2,3,4}, /mwittmann/mx7201-{base,base2,base3}-coord, /mwittmann/mx7201-treat.
- Next: harvest baseline steady tok/s (validate vs 591,330), then
  treatment bf16-control + mxfp8 arms; report all on GB200 2.5 PF conv;
  comment on #7201 if significant.

### 2026-07-18 01:45 - MXFP8-007c: gang-death root cause isolated — log reads against RUNNING 32-task jobs kill the gang
- Forensics across 6 failed baseline/treatment attempts (base..base7, treat,
  b64g1): every 32-task train gang died 1-4 min after a log read (mine or a
  monitor's) hit the RUNNING job — including base7, killed 4 min after a
  2-line log peek, with all 32 tasks healthy mid-tokenizer-copy and the
  controller dropping "late update for terminal attempt (attempt_state=11
  reported=5)" — i.e. the controller declared attempts dead while workers
  still reported running. Mechanism (extends the 2026-07-17 wedge finding):
  at 32-task scale even a capped `job logs` read stalls the task process
  long enough to miss heartbeats -> controller fails the attempt -> gang
  kill cascade -> coordinator JobFailedError. NOT scheduler churn (base7
  died on a quiet cluster), NOT capacity (201 nodes / 804 GPUs), NOT bad
  nodes (base3/base4 node sets disjoint), NOT priority (all
  iris-interactive). Larry's identical runs survive because nothing reads
  their logs mid-run; 16-task (b64g1 reached its own real OOM) and 2-task
  jobs tolerate reads.
- Also real: b64g1 showed the 64-GPU r1 variant of the #7201 config is
  infeasible (318.93 GiB single alloc — 4D-NS/momentum class at r1) —
  descaling was the wrong move; the published 128-GPU r2 point stands.
- New discipline: NO log reads (any cap) against RUNNING >=16-task jobs;
  drivers/monitors poll controller SQL state only; logs harvested at
  terminal state.
- Driver v3 (SQL-only polling) is retrying: base8 next, then treat{4,5}.

### 2026-07-18 01:55 - CORRECTION: gang deaths were real 318 GiB OOMs, not log reads; SYRK env is prime suspect
- Retraction of the 01:45 entry's log-read theory: the parallel session's
  post-mortem (memory iris-log-reads-wedge-training) had already
  architecturally exonerated log reads (finelog serves from parquet via a
  sidecar; never touches the task), and the true error was found in the
  controller's task_attempts.error (NOT in job logs): base7's originating
  task died of RESOURCE_EXHAUSTED allocating 317.75 GiB in jit_train_step;
  the other 31 tasks were then stamped TASK_STATE_COSCHED_FAILED (state
  11) and killed — which is why every gang "died silently mid-line".
  b64g1's 318.93 GiB OOM is the same allocation class: the full stacked
  expert tensor [48,128,5120,2560] f32 (+pad), materialized UNSHARDED.
- Suspect: SCALE_MUON_SYRK=1 on the branch tip (the QuACK symmetric-GEMM
  NS path, commit b0d484ddb, landed ~when Larry's successful e128top5 run
  started; his exact running code state is ambiguous). My runs set it; the
  318 GiB smells like the SYRK path skipping the (L,E)-merge distribution.
- Test in flight: mx7201-base8-coord = identical config MINUS
  SCALE_MUON_SYRK and CE_IMPL. SQL-state-only monitoring (logs only after
  terminal). If it survives past step 0, bisect SYRK vs liger later; the
  comparison proceeds no-SYRK on both baselines (noting the published
  19.0% had syrk in the name).

### 2026-07-18 02:30 - MXFP8-007c: baseline REPLICATED (19.63% MFU); root cause of all failures was missing env
- The controller's job_config.environment_json for Larry's winning run
  revealed the real config: **EP1 + sonic_cute** (not EP8), THREE extra NS
  distribution knobs (SCALE_MUON_INTRA_RACK=1, SCALE_MUON_DIST_NONEXPERT=1,
  SCALE_MUON_PAD_NONEXPERT=1 — whose absence caused the 203-318 GiB
  replicated NS gathers), NCCL_SOCKET_IFNAME='^ibs,ibp,lo,docker,veth,
  cilium,lxc' (whose absence caused treat6's multi-node init barrier
  deadlock), XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async, s3 JAX compile cache,
  CE liger. His run was dirty (base b0d484ddb + uncommitted) but the
  pushed tip + exact env reproduces.
- mx7201-base9 (32 nodes, exact env, json_logger): **609,310 tok/s /
  19.63% MFU**, loss 5.40 @ step 50 — vs Larry's published 591,330 /
  19.0%. Baseline anchored on our own cluster/day. (~3% faster than his,
  warm s3 compile cache + noise.)
- mx7201-treat7 launched: our bench, 128 GPUs, r2 EP8 ring scan, bf16
  control then mxfp8, with the operative env (NCCL ifname, cuda_async,
  MUON knobs, s3 cache, liger).

### 2026-07-18 07:35 - MXFP8-007c overnight: baseline anchored; treatment blocked on the B200MFU-036 CUBIN flake at 32 nodes
- Standing results: baseline (Larry's exact env, our rerun) 609,310 tok/s /
  19.63% GB200-MFU at 128 GPUs; smoke (8 GPUs, #7201 arch, scan):
  mxfp8 +15.5% throughput over bf16 control with loss parity.
- Treatment attempts at 128 GPUs (our bench, r2 EP8 ring scan, NO_NS):
  - treat7: passed init+CUBIN load, died on the EP 4D-NS OOM (since fixed
    twice: reshard variant -> XLA involuntary-full-remat compile hang
    (treat8/9, killed at 70+ min); shard_map explicit-collective variant
    is committed and CPU-verified but not yet field-validated — NO_NS
    sidesteps it for the MFU comparison, caveat: NS excluded in treatment
    arms only, ~4% share class, cancels in the bf16-vs-mxfp8 delta).
  - treat10 (NO_NS): B200MFU-036 CUBIN flake ("Failed to load in-memory
    CUBIN") at runtime start — the known no-mitigation load-call fault
    (#7331 is fighting the same; their ncclep logbook shows 0/5 at r4).
  - treat11 (in-job retries x4): retries die on
    "GetKeyValue() timed out ... clique" — in-job gang retries cannot
    re-bootstrap jax.distributed's KV store (sibling's retry pattern works
    only for their single-node benches). In-job retry design abandoned.
- Driver v4 running: up to 3 FRESH-job attempts (treat12..14) — fresh
  launches re-bootstrap cleanly and empirically pass CUBIN load ~2/3.
- Jobs: /mwittmann/mx7201-treat{6..11}, mx7201-probe32, mx7201-base{7,8,9}.

### 2026-07-18 10:45 - MXFP8-007c: 64-GPU pivot; baseline64 landed; clique wedges traced to gang scheduling class
- 128->64 GPU descale (3/3 fresh 32-node treatment attempts lost to
  CUBIN/clique flakes; 248B model floor is 64 GPUs — the 32-GPU probe
  OOM'd at 132.79 GiB state).
- **Baseline64 landed** (mx7201-b64g2, their branch, Larry's exact env,
  16 nodes B1024 r1 EP1 sonic_cute): **313,666 tok/s / 20.21% GB200-MFU**
  — better per-GPU than the 128-GPU rerun (4,901 vs 4,760 tok/s/GPU),
  consistent with less cross-rack traffic. `exploratory` (single run).
- Treatment arms then went 0/4 without ever finishing NCCL clique init:
  - t64g1: transient wheel-download timeout (resiliparse via
    marin.community) during env sync.
  - t64g2: B200MFU-036 CUBIN load fault at jit_train_step.
  - t64g3: clique-init rendezvous wedge (leader deadlocked in the
    ncclCommInitRank callback; log-silent 20 min; killed).
  - t64g4: all 16 nodes timed out on GetKeyValue(cuda:root_process:0)
    after 10m — root process wedged in comm init despite a 600s cooldown,
    so this is NOT (only) the fast-restart taint.
- Root-cause lead: **gang scheduling class differs between arms.** The
  fray-submitted baseline gangs use `coscheduling_group_by=leafgroup`
  (soft IB-level colocation; fray iris_backend.py always uses leafgroup
  for GPU gangs), while `iris job run --gpu GB200x4 --replicas 16`
  hard-binds to `nvlink.domain` (cli/job.py resolve_multinode_defaults).
  Every fray/leafgroup training gang today initialized cleanly; every
  CLI/nvlink.domain treatment gang wedged or flaked. Correlation, not yet
  causation — t64g7+ tests it directly.
- Fix in flight: python submitter (tmp/submit_treat_leafgroup.py) clones
  the CLI submit path but sets CoschedulingConfig(group_by="leafgroup");
  also makes placement apples-to-apples with the baseline. Driver v3
  (attempts t64g7..9) adds log-staleness wedge autokill (>900s -> kill)
  and 600s inter-attempt cooldowns.
- Jobs: /mwittmann/mx7201-{b64g1,b64g2,b32g}-coord, mx7201-t64g{1..4},
  mx7201-t32g1.

### 2026-07-18 11:25 - MXFP8-007c: leafgroup hypothesis falsified; treatment arms moved onto the fray coordinator path
- t64g7 (leafgroup gang via python submitter) wedged identically to the
  nvlink.domain gangs: all 15 non-root ranks timed out after 10 min on
  GetKeyValue(cuda:root_process:0) — root never published the clique ID.
  Gang scheduling class is NOT the discriminator. Score: direct-CLI bench
  gangs 0/5 today (1 env-download flake, 1 CUBIN, 3 clique-init wedges);
  fray-coordinator gangs (baselines) N/N clean.
- Rank-0 forensics: after "process 0/16" at init, rank 0 logged nothing for
  15 min (python stdout buffering makes silence weak evidence); sibling
  ranks reached jit_train_step execution. Root wedged either in the NCCL
  comm-init callback (fast-restart taint) or in a rank-skewed compile;
  undetermined — and not worth more 64-GPU attempts to bisect.
- Pivot: ported their launch_cw_scale wholesale onto our branch
  (commit above) + SCALE_FP8 knob wiring GrugFp8Config with the MFU-bench
  defaults (mxfp8: wire=False dense grouped, producer auto); dispatch.py
  now forwards XLA_/CE_/SCALE_MUON_ like the source branch. Dry-build
  verified: d5120/L48/E128 top-5, GQA 40/10, scan, ring, mxfp8, muonh.
- Treatment arms now run through the SAME fray coordinator machinery as
  the baseline (driver v4): bf16 control (ring EP8 r1 NO_NS) then mxfp8,
  B1024/16 nodes, real trainer + slimpajama data. This also upgrades the
  comparison: all three arms share trainer/data/optimizer machinery, so
  C-vs-B isolates mxfp8 kernels and B-vs-A isolates ring-EP8+NO_NS vs
  EP1+sonic_cute+NS.

### 2026-07-18 11:50 - MXFP8-007c: CUBIN fault is EP>=8-correlated (per #7331); fresh-cache falsified; treatment descales to EP4
- Fray-path bf16 arm attempts: tf-bf16 (shared S3 compile cache) and
  tg-bf16 (FRESH cache namespace compilation-cache-mx7282-fresh1) both
  died on the same "Failed to load in-memory CUBIN" at jit_train_step —
  launch path AND cache-poisoning both falsified as causes. tg-bf16-r2
  wedged (log stale 1000s+, autokilled by the driver).
- The #7331 logbook already characterizes this: **the EP>=8 CUBIN-load
  failure (B200MFU-035) is intermittent with no known workaround, and
  EP16/32/64 arms all fail (B200MFU-033)**. The baseline is EP1 ->
  immune; every failed treatment arm today is EP8 ring -> exposed. That
  single fact explains the baseline/treatment asymmetry that drove the
  (falsified) scheduling-class and cache theories.
- Score for EP8 at 16 nodes today: 0/7 (4 CUBIN-class, 3 clique wedges —
  plausibly the same fault surfacing on the root rank).
- Decision: treatment arms descale to **EP4** (E_local=32), both arms,
  driver v6 (mx7201-th-*, 3 attempts per arm). EP4 was #7331's most
  reliable ring config (ring_cute EP4 20.83% was their best bf16 EP
  result). The bf16-vs-mxfp8 delta stays internally consistent; EP8 is
  now a documented deployment risk owned by B200MFU-035/#7331 rather
  than a blocker for this comparison.

### 2026-07-18 12:05 - MXFP8-007c: today's 0/9 re-diagnosed as a cluster incident window; standing down until recovery
- EP4 (th-bf16) died on the same CUBIN load fault as the EP8 arms — EP-size
  theory joins launch-path, cache-poisoning, coscheduling-class, and
  driver-skew (all 201 nodes uniform 595.71.05/CUDA 13.2) in the falsified
  pile.
- The #7012 logbook (B200MFU-034..039, parallel session) already
  characterizes both failure modes precisely:
  - CUBIN INVALID_VALUE: intermittent fault of cuModuleLoadData itself
    during the ~5.6k per-fusion module load storm; per-ALLOCATION
    correlated pass rate (0%/20%/75% across node-sets, stochastic within);
    every env/flag workaround (EAGER, compile-parallelism) REFUTED.
  - Clique-init deadlock: deterministic on fast gang restart (22/22;
    leaders in ib_uverbs_event_read; every NCCL knob immune). Cold
    compiles are protected by the 15-25 min gap; warm-cache runs hit it.
    Today's warm retries (t64g3/4/7, tg-bf16-r2) fit exactly.
- Decisive observation: **zero multi-node successes cluster-wide since
  16:27 UTC (09:27 PDT)** — our 9 failures, larry's 16:18 failure (his
  own 293-GiB OOM, but he stopped submitting after), and rav has been
  cycling 32-node debug jobs named r2l48{diag,cap,nccl,fix,mnnvl} since
  ~16:00 UTC — i.e. the cluster owner-side is actively debugging an
  NCCL/MNNVL-class incident in this exact window. Our failure cluster
  starting ~09:30 PDT is the same window.
- Actions: killed th-bf16-r2 (no GPUs held into a degraded cluster; all
  our jobs terminal), stopped the retry driver, armed a zero-GPU
  recovery watch (controller-DB poll for the next multi-node state=4
  job cluster-wide). On recovery: relaunch driver v6 arms (EP4 first,
  then EP8 if the allocation is cold) via the fray path.
- Standing results unaffected: baseline64 313,666 tok/s / 20.21%
  GB200-MFU; 128-GPU baseline 609,310 / 19.63%; smoke mxfp8 +15.5%.

### 2026-07-18 14:15 - MXFP8-008: Hopper-cruft audit + arch-agnostic fp8 recipe (recipe="auto")
- Hypothesis (from mcwitt): the branch may carry Hopper workarounds the
  Blackwell path no longer needs (e.g. the fused cast-transpose kernel that
  works around sm90 wgmma's no-transpose-on-read for 1-byte operands).
- Audit result: the Blackwell op path is already clean. `MxFp8MoeMlpOp`
  (mxfp8.py + vendored mxfp8_fused CuTeDSL kernels) never touches
  fp8_cast_transpose / ragged_dot_mgpu / fp8_ragged; its dual-orientation
  activation quantize is an MXFP8 *recipe* requirement (E8M0 scale blocks
  must lie along the contracting dim of each GEMM orientation), not a layout
  workaround. e5m2/wgmma grep hits inside mxfp8_fused/ are vendored upstream
  dtype-support tables, not code paths. Nothing vendored is dead: every
  module in mxfp8_fused/ and mxfp8_grouped/ is consumed by the op, the GPU
  test ladder, or the MXFP8-001..004 benches the logbook links.
- Interface change (per mcwitt's direction): fp8 recipe selection is now
  architecture-agnostic, following the `ragged_dot(implementation="auto")`
  convention. `GrugFp8Config.recipe` defaults to `"auto"`, resolved once at
  model init on the accelerator task by `resolve_fp8_config` (sm100+ ->
  mxfp8 fused kernels, sm90 -> per_tensor Fp8RaggedDotOp; both impls stay
  functional). `wire` is now `bool | None`, `None` following the resolved
  recipe (per_tensor -> fp8 wire, mxfp8 -> bf16 collectives per MXFP8-000b);
  explicit `wire=True` + mxfp8 still rejected. The CPU dispatcher ships
  configs unresolved (no device probe off-accelerator). `SCALE_FP8` and the
  bench/probe `--fp8-recipe` accept `auto` (now the bench/probe default);
  unknown values fail fast.
- Remat-policy site no longer needs the recipe: `mxfp8_save_qweights` gates
  the save-name alone (only the mxfp8 op tags it; a saved-but-never-tagged
  name is a no-op under other recipes).
- Command: `JAX_PLATFORMS=cpu pytest experiments/grug/moe/test_mxfp8.py -q`
  -> 33 passed (incl. new resolve_fp8_config cases); dispatcher dry-build of
  SCALE_FP8=auto/mxfp8/bogus OK; `./infra/pre-commit.py --files <changed>` OK.
- Scope note: a first pass deleted the Hopper stack outright; mcwitt
  clarified the Hopper impl must stay functional — reverted before commit,
  no history impact.

### 2026-07-18 18:40 - MXFP8-007c: bf16 control GREEN at 64 GPUs; mxfp8 arm blocked by k8s pod-creation outage
- **tj-bf16 (arm B) SUCCEEDED**: fray path, ring EP4, muonh NO_NS, scan,
  B1024, 50 steps — **330,897 tok/s steady ≈ 21.3% GB200-MFU** (derived via
  the 2.5 PF/GPU convention using the baseline harvest's tok/s↔MFU factor);
  final loss 8.121@step 49. Submitted 15:05 PDT on a fresh allocation minutes
  after the incident window lifted (first larger-gang successes ~14:44);
  cleared clique init and trained clean.
- Chained mxfp8 arm (tj-mx, submitted ~1 min after tj-bf16 teardown) wedged
  in clique init — textbook B200MFU-038 fast-restart deadlock; the rendezvous
  spam kept logs fresh on the ti attempt but tj-mx went log-stale and the
  driver autokill caught it at 17 min.
- Cool-down retries tk-mx / tl-mx / tm-mx all blocked by a NEW cluster
  failure mode starting ~16:00 PDT: k8s admission webhook `mpod.kb.io`
  returns 500 on pod creation (iris burns 130-300 placement attempts/job at
  ~1/s), and post-outage pods that do get admitted stall in 'starting'
  indefinitely (no container start). Even 1-cpu `echo ok` probe jobs place
  but never complete. All treatment jobs killed promptly; zero GPUs held.
  Standing down with a completion-probe sentinel (webhook-probe2 must reach
  state 4) before the next mx attempt.
- Comparison state: A (#7201 best, their stack) 313,666 tok/s / 20.21%;
  B (our stack, bf16) 330,897 tok/s / ≈21.3% (+5.5% tok/s — caveat: B runs
  SCALE_MUON_NO_NS, so B-vs-A conflates stack wins with skipped
  Newton-Schulz); C (mxfp8) pending — C-vs-B is the clean mxfp8 read.

### 2026-07-19 02:20 - MXFP8-007c COMPLETE: three-way comparison landed — mxfp8 +27.8% over bf16, +34.8% over #7201 best
- **C (mxfp8 arm, tq-mx) SUCCEEDED**: 422,832 tok/s steady ≈ 27.2% GB200-MFU
  (2.5 PF convention), loss 7.935@49, 10.1 s/step, 50/50 steps, ckpt saved.
  Same stack/flags as B except SCALE_FP8=mxfp8 SCALE_FP8_PRODUCER=xla.
- Three-way at 64 GPUs (16 GB200 nodes, B1024, d5120/48L/E128 top-5, seq4096):

  | arm | config | tok/s | GB200-MFU |
  |---|---|---|---|
  | A | #7201 best (EP1+sonic_cute, full MuonH) | 313,666 | 20.21% |
  | B | our stack bf16 (ring EP4, scan, NO_NS) | 330,897 | ≈21.3% |
  | C | B + mxfp8 fused kernels (xla producer) | 422,832 | ≈27.2% |

  **C/B = 1.278** (isolated mxfp8 effect, apples-to-apples) — replicates the
  8-GPU smoke ratio (1.155 there; better here, GEMM-heavier at EP4-local
  E=32/shard). **C/A = 1.348** over the #7201 production best. `replicated`
  (smoke + 64-GPU, distinct allocations).
- Caveats: (1) B and C both run SCALE_MUON_NO_NS, so B-vs-A conflates stack
  wins with skipped Newton-Schulz; C-vs-B is the clean mxfp8 read. (2) C's
  final loss 7.935 vs B's 8.121 at step 49 — check whether the data order is
  run-id-seeded before reading anything into per-run loss levels; smoke-scale
  numerics parity (~2e-3 tracking) is the standing evidence. (3) 50-step
  runs; steady tok/s variance <0.01%.
- **NEW DEFECT + root cause (exploratory, 2 wedges + 1 clean A/B): the CuTe
  producer compile-probe subprocess wedges multi-node startup.** tn-mx and
  tp-mx (mxfp8_producer=auto) hung SILENTLY pre-compile (no rendezvous spam,
  no errors; tp killed after 80+ min of nothing). tq-mx with
  SCALE_FP8_PRODUCER=xla went submit→training in ~10 min. Suspect
  `cute_producer_available()`'s per-process subprocess (jax init on GPUs the
  parent already holds, 16 nodes) — needs a real fix before producer=auto is
  usable at scale (probe on proc 0 only, or file-cache the probe verdict).
  Knob landed as 1bad90b96.
- Timeline note: attempts tn/tk/tl/tm were spread across an ~8h cluster
  pod-creation outage (webhook 500s; probes wedged in ContainerCreating);
  recovery detected 23:55 by an end-to-end 1-cpu probe job completing.
- Driver lesson recorded: 900s log-staleness autokill fires mid-compile for
  cold mxfp8 programs (tn killed wrongly); v13+ uses 2400s + rendezvous-spam
  detection is still the only reliable deadlock tell.

### 2026-07-19 10:45 - MXFP8-009: "libNVVM node heterogeneity" ROOT-CAUSED — cutlass-dsl wheel shadowing, fixed in lockfile; producer probe deleted

- Hypothesis: pinpoint what makes CuTe DSL compiles fail on "some nodes"
  (mcwitt directive: bisect the failing construct + survey the fleet-facing
  toolchain for heterogeneity).
- Commit Hash: 166696dfb (survey script), d064dc173 (lockfile fix),
  f4cac5066 (probe deletion). Jobs `/mwittmann/nvvm-survey-s1..s3`,
  `/mwittmann/mxfp8-cute-prod-g1`.
- Method: `standalone/nvvm_node_survey.py` — per-node census (toolchain
  binary hashes + loaded-.so maps + installed-tree fingerprints) + the 002c
  compile ladder, run as a 32-replica GB200x4 job (one pod per node), twice
  on the old lock and once on the fixed lock. Detailed per-node results are
  in local session files only (not for publication).
- Result (`replicated`, 96 pods total):
  - NOT a kernel construct: failing pods fail EVERY ladder variant, incl. a
    trivial inline-asm-only kernel AND the native `.to(Float8E4M3FN)`
    intrinsic control; passing pods pass everything.
  - NOT node heterogeneity: half the nodes FLIPPED verdict between two
    identical survey draws minutes apart. Driver, kernel, ptxas, libnvvm,
    libcuda are hash-identical on passing and failing pods.
  - ACTUAL cause: `nvidia-cutlass-dsl` requires `-libs-base` unconditionally
    and the `[cu13]` extra merely ADDS `-libs-cu13`; the two wheels ship 99
    identical paths with different CUDA-12/CUDA-13 builds of the entire DSL
    (frontend + `_cutlass_ir` MLIR compiler). uv silently lets one clobber
    the other, re-rolled per `uv sync` (~coin flip). Venv tree fingerprints:
    always a clean sweep of one variant; variant predicted the verdict 64/64
    (base -> NVVM "unsupported operation" on every compile, cu13 -> all
    green). Big training gangs dodged it because they delta-install into a
    pre-baked venv (one shared draw); per-pod-synced bench jobs re-rolled
    every submit — the source of every "works on some nodes" observation
    since 002c.
  - Fix: uv `override-dependencies` excludes libs-base (impossible marker).
    Confirmation draw on the fixed lock: 32/32 pods pass ALL ladder
    variants, venvs pure cu13.
- Fallout corrections (stale-claims cleanup, f4cac5066):
  - `cute_producer_available()` subprocess probe DELETED (mxfp8.py + the
    bench copy); `producer="auto"` now simply means the CuTe kernel, `xla`
    stays as an explicit A/B knob. This also removes the MXFP8-007c
    multi-node startup-wedge vector for good.
  - The 002c inline-PTX `cvt` bypass is KEPT (bit-exactness validated) but
    its rationale comment corrected: the intrinsic compiles fine on a clean
    cu13 install; the failures were base-variant compilers.
  - 002c/004c gotcha entries claiming "GB200 nodes are HETEROGENEOUS for
    libNVVM" are superseded by this entry. The 004c "probe fails on ~3/4 of
    nodes while fused kernels compile" observation should also be re-read
    with suspicion: probe-subprocess GPU-init failures were
    indistinguishable from compile failures in that setup.
- Validation in flight: `mxfp8-cute-prod-g1` runs the GPU op ladder with the
  CuTe producer as default (first e2e exercise of the in-op CuTe producer —
  closes the 004c caveat).
- Next action: fold the fix into the production PR set; consider reporting
  the packaging conflict upstream (nvidia-cutlass-dsl metadata) and a uv
  feature request for loud file-clobber warnings.

### 2026-07-19 10:55 - MXFP8-009 postscript: CuTe-producer op ladder green

- `/mwittmann/mxfp8-cute-prod-g1` (GB200x1, producer=auto -> cute): contract
  ok; dequant refs 7.4e-6..5.3e-5 (gate 1e-3); blackbox vs bf16 6.6-6.7e-2
  (unchanged class vs 004c). ALL CHECKS PASSED — the in-op CuTe producer is
  e2e-validated; 004c caveat closed.

### 2026-07-19 11:40 - MXFP8-009b: inline-PTX cvt re-justified on merit — native intrinsic bit-exact but ~12% slower

- Hypothesis (mcwitt): with the wheel-shadowing root cause fixed, the 002c
  inline-PTX e4m3 cvt may be a defunct-workaround artifact — revert to the
  idiomatic DSL-native `.to(Float8E4M3FN)` unless it has a standing
  justification.
- Method: patched the quantizer store path to the native conversion; ran the
  full 002c bit-exact suite + GPU op ladder (`mxfp8-nativecvt-g1`), then a
  same-pod interleaved 2-round A/B native-vs-asm (`mxfp8-cvt-ab-g1`) to
  remove the documented ~25% cross-job timing variance.
- Result (`replicated` within-pod):
  - Numerics: native is fully bit-exact (8/8 cases incl. adversarial
    denormal/2^120 blocks, both rounds; 32/32 BIT-EXACT lines; op ladder
    green, producer=cute).
  - Perf: asm wins consistently — d2560 kernel 0.559/0.568 ms vs native
    0.634/0.644 (~12%); 4-tensor producer total 2.113/2.135 vs 2.312/2.318
    (~9%). The producer budget vs XLA is 2.09 ms: asm sits AT break-even,
    native is clearly past it.
- Decision: KEEP the inline asm, justification rewritten in-code to the
  measured perf (not the defunct compile workaround); native patch reverted.
  Commit 3ddb2360b. Everything else from the misdiagnosis era already
  removed in f4cac5066 (probe) / d064dc173 (lockfile).

### 2026-07-19 16:10 - MXFP8-010: NS-enabled 64-GPU pair COMPLETE — honest mxfp8 win 1.31x clean / 1.25x vs #7201 best; CuTe producer breaks 16-node executable load

- Hypothesis (mcwitt): rerun the 007c comparison with Newton-Schulz enabled
  (the NO_NS control was unrealistic) and matched configs.
- Prep findings: (1) hparams-dump diff showed the 007c arms ALSO differed on
  expert_axis (B=EP4 vs C=EP8) — the old C/B=1.278 was confounded; (2) the
  shard_map 4D NS path (e2be05f4b) field-validated on a 2-node smoke pair
  (both arms trained; only failure was a 2-node-only checkpoint host-RAM
  OOM, exit 137 during final save — state/host is 8x the 64-GPU case);
  (3) loss divergence between same-seed arms is pure numerics: data order
  is seed-keyed (seed=0 both), nothing keys off run_id.
- Command: jobs `/mwittmann/mx7201-ns64-{bf16,mx,mx2,mx3,mx4}`, drivers
  tmp/ns64_driver.sh + retries. Config = 007c COMMON_ENV minus
  SCALE_MUON_NO_NS, both arms ring EP8, 16xGB200x4, B1024, 50 steps.
- Result (`exploratory`, single run per arm; GB200-MFU via tok/s ratio to
  the 007c baseline anchor 313,666 = 20.21%):

  | arm | config | tok/s | MFU | loss@49 |
  |---|---|---|---|---|
  | A | #7201 best (EP1 sonic, full MuonH) | 313,666 | 20.21% | — |
  | B2 (bf16+NS) | bf16 + NS, ring EP8 | 299,894 | ~19.3% | 5.449 |
  | C2 (mx+NS) | mxfp8 + NS, ring EP8, xla producer | **392,287** | **~25.3%** | 5.383 |

  **C2/B2 = 1.308 (clean mxfp8 read, all else matched); C2/A = 1.251
  (honest vs #7201 best, both full MuonH).** NS cost on the mx stack
  (tq-mx vs mx4, only NS differs): 7.2%. B2 < A by 4.4% — the old "B beats
  A" was the NO_NS+EP4 artifact.
- **NEW DEFECT (replicated 3/3 vs 2/2): producer=cute deterministically
  breaks 16-node executable load.** mx/mx2/mx3 (cute, via the new auto
  default from f4cac5066) all died at jit_train_step load with "Failed to
  load in-memory CUBIN (compiled for a different GPU?)" /
  CUDA_ERROR_INVALID_VALUE on ALL 16 hosts — across different node sets
  AND a fresh compilation cache (mx3), so neither node flake nor stale
  cache. mx4 (identical but SCALE_FP8_PRODUCER=xla) trained clean, as did
  bf16+NS and the 2-node cute smoke. Reframes at least part of
  "B200MFU-036 CUBIN flake" as graph-content-dependent. Mechanism OPEN
  (DSL kernels are runtime-registered FFI, not embedded in the XLA
  executable — so the load failure path is not understood). Mitigation
  shipped: producer auto -> xla (693124f9b); cute stays opt-in.
- Interpretation: mxfp8 remains a large, now-honest win at production
  config; the CuTe producer perf (002c 2.5x quantizer) is stranded until
  the load defect is root-caused (candidate follow-up: XLA flag bisect
  starting with --xla_gpu_enable_command_buffer=, HLO dump diff cute-vs-xla).
- Next action: post corrected numbers to #7282/#7201; producer-load
  root-cause as a separate work item.

### 2026-07-20 15:15 - MXFP8-U000: uniform-dense baseline replicated; oracle design approved

- Hypothesis: native `jax.nn.scaled_dot_general` still loses to delayed
  per-tensor FP8 because of block-quantization producers, while prequantized
  `blockScaledDot` remains competitive.
- Commit Hash: `0a3785463` for the benchmark. The approved design spec remains
  local and uncommitted.
- Command: `/home/marin/projects/marin/.venv/bin/iris
  --cluster=cw-us-east-08a job run --no-wait --user mwittmann --job-name
  mxfp8-uniform-baseline-r1 --gpu GB200x1 --enable-extra-resources --cpu 16
  --memory 64g --extra gpu -- python
  experiments/grug/moe/standalone/bench_mxfp8_dense.py --out
  /tmp/mxfp8-uniform-baseline-r1.json`.
- Config: one GB200, JAX 0.10.1, cuDNN 9.19, `M=65536`, BF16 inputs, 10
  warmups, median of 50 iterations. Job
  `/mwittmann/mxfp8-uniform-baseline-r1` succeeded in 43.65 seconds.
- Result (`exploratory`, one job):

  | shape | per-tensor fwd+bwd | native mxfp8 fwd+bwd | native / per-tensor time | prequant mxfp8 fwd | per-tensor fwd |
  |---|---:|---:|---:|---:|---:|
  | 2560x2560 | 1.084 ms | 2.180 ms | 2.01x | 0.430 ms | 0.433 ms |
  | 2560x1280 | 0.648 ms | 1.479 ms | 2.28x | 0.308 ms | 0.313 ms |
  | 1280x2560 | 0.681 ms | 1.524 ms | 2.24x | 0.331 ms | 0.299 ms |

  Native MXFP8 lowered to `__cudnn$blockScaledDot` for every shape. Its
  output/input-gradient/weight-gradient relative errors were lower than the
  per-tensor arm, so the loss is a producer-cost result, not a fallback or
  numerical failure.
- Interpretation: the original MXFP8-001/001b conclusion replicates on a
  fresh GB200. Forward-only prequantized MXFP8 is within -1% to +11% of
  per-tensor FP8 depending on shape; a three-product prequantized
  forward+dgrad+wgrad oracle is required before deciding whether fusion can
  make uniform MXFP8 competitive.
- Next action: implement MXFP8-U001 from
  `docs/superpowers/specs/2026-07-20-uniform-mxfp8-design.md`: measure the
  zero-producer oracle on d2560 and production d5120 shapes before adding any
  model integration.

### 2026-07-20 15:37 - MXFP8-U001: zero-producer dense oracle passes, narrowly

- Hypothesis: once all block quantization and transpose work is removed,
  three cuDNN block-scaled matmuls (forward, dgrad, and wgrad) can match the
  delayed per-tensor FP8 forward+backward path on the weighted production
  dense mix.
- Commit Hash: `21e129ca58d2b5c0eec67d99ae9e8ed01f7f339f`.
- Command (run twice, changing `--job-name` from `mix-r1` to `mix-r2`):
  `/home/marin/projects/marin/.venv/bin/iris --cluster=cw-us-east-08a job run
  --no-wait --user mwittmann --job-name mxfp8-uniform-oracle-mix-r1 --gpu
  GB200x1 --enable-extra-resources --cpu 16 --memory 96g --extra gpu -- python
  experiments/grug/moe/standalone/bench_mxfp8_dense.py --git-sha
  21e129ca58d2b5c0eec67d99ae9e8ed01f7f339f --shape
  q_o_shared_5120x5120 --shape kv_5120x1280 --tokens 65536 --warmup 10
  --iters 50 --out /dev/stdout`.
- Config: one GB200 per replication, JAX 0.10.1, CUDA 13, cuDNN 9.19,
  `XLA_FLAGS=""`. Each replication ran both production shapes sequentially
  on the same GPU. The mix weights five square projections and two K/V
  projections. All oracle rows compiled exactly three
  `__cudnn$blockScaledDot` call sites.
- Result (`replicated`, two independent jobs):

  | replication | shape | per-tensor fwd+bwd (median +/- MAD) | prequant MXFP8 oracle (median +/- MAD) | oracle / per-tensor |
  |---|---|---:|---:|---:|
  | mix-r1 | 5120x5120 | 4.029 +/- 0.122 ms | 4.086 +/- 0.212 ms | 1.0140x |
  | mix-r1 | 5120x1280 | 1.223 +/- 0.028 ms | 1.159 +/- 0.027 ms | 0.9481x |
  | mix-r2 | 5120x5120 | 4.061 +/- 0.078 ms | 3.978 +/- 0.033 ms | 0.9796x |
  | mix-r2 | 5120x1280 | 1.257 +/- 0.060 ms | 1.157 +/- 0.013 ms | 0.9207x |
  | mix-r1 | weighted 5:2 | - | - | **1.0068x** |
  | mix-r2 | weighted 5:2 | - | - | **0.9731x** |

  Median complete-mix ratio: **0.9900x**. Both replications pass the oracle
  gate (`<=1.01x`). The slower run's 1.4% square-shape deficit is smaller
  than that row's timing dispersion and does not hide a K/V regression.
  Output/dgrad/wgrad relative errors were respectively
  `3.72e-2/2.66e-2/1.03e-2` (square) and
  `3.72e-2/3.32e-2/1.03e-2` (K/V), all lower than the per-tensor arm.
- Producer bound: the deliberately unfused six-quantization JAX path cost
  2.491/2.505 ms on square and 1.647/1.666 ms on K/V. Those costs are not
  part of the oracle but show that native JAX producers cannot meet the
  final gate; fused dual-orientation production and reuse are required.
- Harness failures excluded from performance evidence:
  `/mwittmann/mxfp8-uniform-oracle-{square,kv}-r1` failed because Iris bundles
  omit `.git`; `--git-sha` is now explicit. Square `r2` failed because the
  first validator counted shared StableHLO helper definitions rather than
  compiled call sites; K/V `r2` completed and agreed with the retained runs.
- Artifacts: Iris jobs `/mwittmann/mxfp8-uniform-oracle-mix-r1` and
  `/mwittmann/mxfp8-uniform-oracle-mix-r2`; full machine-readable JSON is in
  each job's durable task log.
- Interpretation: the block-scaled GEMMs themselves do not rule out uniform
  MXFP8, so stopping criterion 1 does not apply. The available margin is only
  about 1%, making producer reuse decisive.
- Next action: MXFP8-U002 measures the existing CuTe dual-orientation
  producer in the dense linear-scale layout, first unshared and then reusing
  each high-precision `x`, `w`, and cotangent producer across the three
  oracle products and across Q/K/V or gate/up projections.

### 2026-07-20 16:09 - MXFP8-U002: optimized CuTe producers miss the dense gate

- Hypothesis: the existing CuTe dual-orientation MXFP8 quantizer can recover
  the zero-producer oracle by producing rowwise and columnwise tensors once
  per operand and reusing activation producers across adjacent projections.
- Commit Hashes: `8f6f1d55e` for the replicated forward+backward production
  mix and `b7771837b` for the projection-reuse benchmark.
- Command (run twice for each benchmark, changing `r1` to `r2`):
  `/home/marin/projects/marin/.venv/bin/iris --cluster=cw-us-east-08a job run
  --no-wait --user mwittmann --job-name mxfp8-uniform-cute-mix-r1 --gpu
  GB200x1 --enable-extra-resources --cpu 16 --memory 96g --extra gpu -- python
  experiments/grug/moe/standalone/bench_mxfp8_dense.py --git-sha
  8f6f1d55e --producer cute --shape q_o_shared_5120x5120 --shape
  kv_5120x1280 --tokens 65536 --warmup 10 --iters 50 --out /dev/stdout`.
  The projection jobs used commit `b7771837b`, job name
  `mxfp8-uniform-projection-reuse-r1`, and added `--projection-reuse`.
- Config: one GB200 per job, JAX 0.10.1, CUDA 13, cuDNN 9.19, BF16 inputs,
  10 warmups, median and MAD of 50 iterations. Each replication ran both
  production shapes on the same GPU. The CuTe forward+backward arm quantized
  each of `x`, `w`, and the output cotangent once into both orientations,
  then issued the same three block-scaled matmuls as the oracle.
- Producer validation: the dense layouts are bit-exact with Marin's grouped
  MXFP8 reference on normal and adversarial inputs at `M=8192` and `M=65536`.
  They intentionally differ from JAX's internal dense quantizer on some
  values because JAX applies scaling in BF16 while the grouped/CuTe reference
  uses F32 power-of-two scaling. Relative errors against the F32 reference
  remained lower than delayed per-tensor FP8. The corrected recheck job was
  `/mwittmann/mxfp8-uniform-quantizer-recheck-r2`; an earlier smoke failed a
  too-strong native-JAX-bit-match assertion and is excluded.
- Result (`replicated`, two independent production-mix jobs):

  | replication | shape | per-tensor fwd+bwd | prequant oracle | CuTe producer + reuse | standalone producer |
  |---|---|---:|---:|---:|---:|
  | mix-r1 | 5120x5120 | 4.033 ms | 4.013 ms | 4.847 +/- 0.085 ms | 1.206 ms |
  | mix-r1 | 5120x1280 | 1.240 ms | 1.151 ms | 1.722 +/- 0.031 ms | 0.854 ms |
  | mix-r2 | 5120x5120 | 4.114 ms | 4.147 ms | 4.874 +/- 0.142 ms | 1.088 ms |
  | mix-r2 | 5120x1280 | 1.277 ms | 1.107 ms | 1.723 +/- 0.039 ms | 0.743 ms |

  On the weighted 5:2 dense mix, actual CuTe producer+GEMM time was
  **1.2221x** and **1.2030x** the per-tensor baseline; the median ratio was
  **1.2125x**, well outside the `<=1.01x` gate. The benchmark's emitted
  `weighted_production_ratio` in these jobs describes the prequantized oracle,
  not the CuTe arm; the ratios above are recomputed from the retained rows.
- Forward projection reuse (`replicated`, two independent jobs):

  | projection bundle | per-tensor r1/r2 | CuTe unshared r1/r2 | CuTe reused r1/r2 | reused / per-tensor |
  |---|---:|---:|---:|---:|
  | Q/K/V | 1.948 / 2.003 ms | 2.739 / 2.913 ms | 2.266 / 2.401 ms | 1.163x / 1.199x |
  | shared gate/up | 2.443 / 2.719 ms | 3.013 / 3.188 ms | 2.761 / 3.048 ms | 1.130x / 1.121x |

  Reusing the activation quantization saved 0.47-0.51 ms for Q/K/V and
  0.14-0.25 ms for gate/up, confirming that reuse works. It still left a
  12-20% forward deficit. Backward cotangents are projection-specific, so
  they cannot receive the same reuse.
- Artifacts: `/mwittmann/mxfp8-uniform-cute-mix-r1`,
  `/mwittmann/mxfp8-uniform-cute-mix-r2`,
  `/mwittmann/mxfp8-uniform-projection-reuse-r1`, and
  `/mwittmann/mxfp8-uniform-projection-reuse-r2`. Full JSON is in each
  durable task log.
- Interpretation: a correct standalone dual-orientation producer cannot
  recover the oracle margin at these shapes, even with the two material
  cross-projection reuse opportunities. This rules out the available
  Pallas/CuTe producer architecture for the 99% full-step target. It does
  not prove that a future producer fused into GEMM epilogues is impossible.
- Next action: benchmark NVIDIA Transformer Engine 2.16's supported JAX
  `MXFP8BlockScaling` dense path against the same per-tensor baseline. This
  provides an independent implementation control before concluding that
  only unavailable epilogue fusion could meet the gate.

### 2026-07-20 16:27 - MXFP8-U003: TE reaches the projected step gate; in-repo CuTe does not

- Hypothesis: an independent optimized MXFP8 implementation, optionally with
  fused Q/K/V and shared gate/up parameter layouts, can reduce producer and
  launch costs enough to make uniform MXFP8 plausible at full-step scale.
- Commit Hashes: `29071f8ee` for the Transformer Engine separate-projection
  control, `9e7b7653e` for its fused-projection control, and `db8b919f5` for
  the equivalent Marin CuTe fused-projection benchmark.
- Transformer Engine command (two replications, changing `r1` to `r2`):
  `/home/marin/projects/marin/.venv/bin/iris --cluster=cw-us-east-08a job run
  --no-wait --user mwittmann --job-name mxfp8-uniform-te-fused-r1 --gpu
  GB200x1 --enable-extra-resources --cpu 16 --memory 96g --extra gpu -- bash
  experiments/grug/moe/standalone/run_te_bench.sh python
  experiments/grug/moe/standalone/bench_te_dense.py --git-sha 9e7b7653e
  --shape q_o_shared_5120x5120 --shape kv_5120x1280
  --projection-fusion --tokens 65536 --warmup 10 --iters 50 --out
  /dev/stdout`. The runner installs TE 2.16.0 JAX against CUDA 13.0 and
  cuDNN 9.19. NVIDIA's documented `make_dot_general_cls(MXFP8BlockScaling())`
  path is used directly.
- CuTe command: the same Iris resources and timing protocol, running
  `bench_mxfp8_dense.py --git-sha db8b919f5 --producer cute
  --projection-fusion` under job names `mxfp8-uniform-cute-fused-r1/r2`.
- Separate-projection TE control (`replicated`):

  | replication | 5120x5120 TE / per-tensor | 5120x1280 TE / per-tensor | weighted TE / per-tensor |
  |---|---:|---:|---:|
  | r1 | 4.480 / 4.105 ms | 1.455 / 1.234 ms | 1.1009x |
  | r2 | 4.115 / 3.855 ms | 1.352 / 1.112 ms | 1.0829x |

  TE lowered to `te_gemm_v2_ffi` and `te_dbias_quantize_ffi`, not JAX's
  native producer or a BF16 fallback. Output/dgrad/wgrad errors were about
  `3.71e-2/3.71e-2/3.80e-2`, lower than the per-tensor control.
- Fused representation: Q/K/V are one `5120x7680` parameter/GEMM and shared
  gate/up are one `5120x10240` parameter/GEMM. O and shared down remain two
  separate square projections. The baseline splits the same combined BF16
  parameter and runs the current per-tensor operator independently, so the
  comparison charges all forward, dgrad, and wgrad work without a parameter
  concatenation in either arm.
- Result (`replicated`, two independent jobs per implementation):

  | implementation | replication | fused QKV MX / tensor | fused gate/up MX / tensor | complete dense MX / tensor |
  |---|---|---:|---:|---:|
  | TE 2.16 | r1 | 6.592 / 6.492 ms | 8.445 / 8.078 ms | **1.0608x** |
  | TE 2.16 | r2 | 6.218 / 6.184 ms | 8.173 / 7.716 ms | **1.0539x** |
  | Marin CuTe | r1 | 6.885 / 6.356 ms | 8.927 / 7.971 ms | **1.1345x** |
  | Marin CuTe | r2 | 7.074 / 6.454 ms | 9.120 / 8.139 ms | **1.1426x** |

  Every fused CuTe row compiled three `__cudnn$blockScaledDot` calls plus
  `CuteDSLRT_NvJaxCutlassCall`; its errors matched the validated MXFP8 class.
  Fusion improves the earlier CuTe 1.2125x median dense ratio but does not
  remove enough producer cost.
- Full-step projection: the matched hybrid production run processes
  `1024*4096` tokens at 392,287 tok/s, or 10.6919 s/step. At 48 layers, the
  measured dense deltas project as follows. This is an additive kernel-time
  estimate, not a replacement for a matched full-step run.

  | implementation | replication | added dense time/step | projected hybrid throughput ratio | projected tok/s |
  |---|---:|---:|---:|---:|
  | TE fused | r1 | 66.0 ms | **0.9939x** | 389,881 |
  | TE fused | r2 | 56.0 ms | **0.9948x** | 390,242 |
  | CuTe fused | r1 | 145.6 ms | **0.9866x** | 387,017 |
  | CuTe fused | r2 | 156.1 ms | **0.9856x** | 386,643 |

- Artifacts: `/mwittmann/mxfp8-uniform-te-dense-r1/r2`,
  `/mwittmann/mxfp8-uniform-te-fused-r1/r2`, and
  `/mwittmann/mxfp8-uniform-cute-fused-r1/r2`; all six succeeded with exit 0
  and retain full JSON in durable task logs.
- Interpretation: uniform MXFP8 is not physically impossible: NVIDIA's
  optimized producer plus fused parameter layout clears the 99% *projected*
  full-step gate in both replications. The currently integrated CuTe producer
  misses it in both replications, and its known 16-node executable-load failure
  independently blocks a production run. The TE result cannot yet be called a
  uniform Grug demonstration because TE is not packaged in Marin and the
  projection has not been tested through Grug's sharding/remat/full-step graph.
- Next action: prototype TE's stateless `dense.dense` call without Flax state,
  then stop if packaging or sharding prevents a bounded Grug smoke. If the
  direct call works, add an experiment-only fused QKV/gate-up layout and apply
  the 4-8 GPU full-step gate before considering a 64-GPU run.

### 2026-07-20 16:40 - MXFP8-U004: stateless TE control matches the Flax path

- Hypothesis: Transformer Engine's public stateless `dense.dense` API uses the
  same MXFP8 kernels and numerics as its documented Flax wrapper, without
  adding Flax variables to the Grug model state.
- Commit Hash: `5648dcffc`.
- Command: `/home/marin/projects/marin/.venv/bin/iris
  --cluster=cw-us-east-08a job run --no-wait --user mwittmann --job-name
  mxfp8-uniform-te-direct-r1 --gpu GB200x1 --enable-extra-resources --cpu 16
  --memory 96g --extra gpu -- bash
  experiments/grug/moe/standalone/run_te_bench.sh python
  experiments/grug/moe/standalone/bench_te_dense.py --git-sha 5648dcffc
  --shape kv_5120x1280 --direct-api --tokens 65536 --warmup 10 --iters 50
  --out /dev/stdout`.
- Config: one GB200, TE 2.16.0, JAX 0.10.1, CUDA 13, cuDNN 9.19; BF16
  inputs; 10 warmups and 50 measured iterations.
- Result (`exploratory`): direct stateless TE took
  `1.396 +/- 0.044 ms`, versus `1.349 +/- 0.037 ms` for the Flax wrapper and
  `1.119 +/- 0.035 ms` for delayed per-tensor FP8. Direct and Flax TE had the
  same output/dgrad/wgrad relative errors (`3.706e-2/3.706e-2/3.808e-2`) and
  both lowered to `te_gemm_v2_ffi` plus `te_dbias_quantize_ffi`.
- Artifact: Iris job `/mwittmann/mxfp8-uniform-te-direct-r1`, succeeded with
  exit 0 in 148 seconds.
- Interpretation: the stateless call is suitable for an experiment-only Grug
  integration. It does not improve the separate-projection result; projection
  fusion remains necessary for the projected 99% gate.
- Next action: add `dense_recipe="mxfp8"` without changing the hybrid default,
  then compile a complete distributed train step.

### 2026-07-20 17:48 - MXFP8-U005: no available dense path meets the distributed full-step gate

- Hypothesis: TE's stateless MXFP8 dense op can replace every attention and
  shared-expert `Fp8DotGeneralOp` while Grug's existing fused MXFP8 expert op
  remains unchanged, producing a uniform recipe within 1% of the hybrid
  step time.
- Commit Hashes: `5a4e285bc` adds the uniform recipe and `5c782b91c` adds the
  required TE mesh-resource context. `GrugFp8Config(recipe="mxfp8")` remains
  the exact hybrid default; uniform is selected explicitly with
  `dense_recipe="mxfp8"`.
- Representative command (run twice, changing `dense_recipe` between
  `mxfp8` and `per_tensor`):
  `/home/marin/projects/marin/.venv/bin/iris --cluster=cw-us-east-08a job run
  --no-wait --user mwittmann --job-name mxfp8-uniform-fullstep-two-r1 --gpu
  GB200x2 --enable-extra-resources --cpu 24 --memory 192g --extra gpu -- bash
  experiments/grug/moe/standalone/run_te_bench.sh python
  experiments/grug/moe/standalone/bench_grug_moe_mfu_fp8.py --run-id
  mxfp8-uniform-fullstep-two-r1 --output-dir
  /tmp/mxfp8-uniform-fullstep-two-r1 --fp8 --no-fp8-wire --fp8-recipe
  mxfp8 --fp8-dense-recipe mxfp8 --mxfp8-producer xla --steps 15
  --warmup-steps 7 --batch-size 32 --seq-len 4096 --hidden-dim 5120
  --num-layers 4 --num-experts 32 --num-experts-per-token 5 --head-dim 128
  --num-kv-heads 10 --num-gpus 2 --moe-implementation ring
  --expert-parallelism 2 --replica-axis 1 --attention-implementation
  gpu_fa4_cute --remat-mode recompute_all --stacked-blocks`.
- Config: TE 2.16.0, JAX 0.10.1, CUDA 13, cuDNN 9.19; d5120, KV1280,
  shared intermediate 2560, seq4096, top-5 routing, `recompute_all`. The final
  EP2 pair used 65,536 tokens and 16 experts per device, matching the
  production run's per-device token count and local-expert count.
- Result:

  | arm | outcome | steady median |
  |---|---|---:|
  | hybrid EP2, L4 | succeeded, finite loss through step 14 | **48,157 tok/s; 2.7221 s; 13.05% B200-convention MFU** |
  | uniform EP2, L4 | XLA GPU backend abort before execution | `AllReduceThunk::CheckImplementable(): reduction_kind.has_value()` |
  | uniform EP4, L4 | same XLA GPU backend abort before execution | no timing |

  The EP2 uniform failure reproduced the EP4 failure at the same backend
  phase, after TE import, model tracing, custom partitioning, and several
  minutes of compilation. The process exits 134 from a C++ fatal check, so
  Python cannot catch or downgrade it. The paired EP2 hybrid job succeeded
  with exit 0. A one-GPU fallback is not a valid uniform test because Grug's
  MXFP8 whole-expert-MLP op requires an expert axis larger than one.
- Integration failures excluded from timing evidence: the first EP4 attempt
  lacked TE's process-global `MeshResource` and was fixed; an eight-layer EP4
  attempt exhausted memory while autotuning a 25 GiB expert transpose; the
  four-layer EP4 hybrid control hit the pre-existing in-memory CUBIN load
  fault. Reducing to the minimum valid EP2 graph removed those confounds and
  retained the TE all-reduce abort only in the uniform arm.
- Local verification: the focused config/state/import contracts pass
  (`3 passed`); required exact-file pre-commit checks pass; pyrefly reports
  zero errors on `mxfp8_dense.py` and `train.py`. A broader Grug suite reached
  24 passes and 14 skips, then hit the existing base-variant checkpoint
  serialization timeout.
- Artifacts: `/mwittmann/mxfp8-uniform-fullstep-{u-r1,u-r1b,u-r1c,two-r1}`
  and `/mwittmann/mxfp8-hybrid-fullstep-{h-r1c,two-r1}`. Full traces and the
  hybrid JSON summary are retained in Iris task logs.
- Interpretation: uniform MXFP8 is not physically impossible. The TE fused
  microbenchmark projects to 99.39-99.48% of hybrid throughput. No available
  implementation can realize that result in the current Grug train graph:
  Marin's correct CuTe producer is 21.25% slower on the weighted dense mix
  (13.45-14.26% slower with projection fusion), while TE's faster op aborts
  deterministically on the minimum valid distributed graph and is not a Marin
  dependency. The current hybrid recipe is therefore the only demonstrated
  production-viable choice.
- Decision: keep dense matmuls on delayed per-tensor FP8 and ragged expert
  matmuls on MXFP8. Resume uniform work when either TE supports Grug's
  multi-axis distributed backward or a dense MXFP8 producer can be fused into
  GEMM epilogues without the measured standalone producer tax.
