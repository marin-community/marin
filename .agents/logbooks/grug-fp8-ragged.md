# Logbook: FP8 mixed-precision MoE ragged-dot optimization

Optimizing the FP8 (E4M3/E5M2) Mosaic-GPU ragged-dot kernel replacement for Grug MoE training on
H100, measured against the tuned bf16 Triton baseline. Lightweight process: this in-branch logbook +
the metric harness (`scripts/bench/`, see `FP8_AUTOTUNE_RUNBOOK.md`). No GitHub issue / W&B yet —
promote upward only once results are worth publishing.

- **Branch:** `grug-fp8-fork` (port wins to `grug-fp8-shim` later).
- **Metric:** best-tuned fp8 fwd+bwd vs best-tuned bf16, ratio-of-medians + bootstrap 95% CI, per
  shape. Defined in `FP8_AUTOTUNE_RUNBOOK.md`. Cached wheel makes a run ~3 min.
- **Experiment ID prefix:** `GFP8-OPT`.

## North star

**Lift the realistic operating point** — EP4–8, ~512–1024 tokens/expert, the memory-limited regime
where real d2560 training runs. fp8 already wins everywhere on the real model (no regressions); the
win is throttled at small per-device batch by fp8 fixed overhead. Big-batch ceiling (2k–4k
tok/expert, already 1.27–1.35×) is a nice-to-have, not the target.

## Baseline (trusted, from the validated metric)

Real d2560 model (D2560 / F1280 / E256 / top_k4, true-ragged). Best-vs-best fp8 speedup:

| point | tokens/expert | E_local | speedup |
|-------|---------------|---------|---------|
| `d2560_e32_t512` | 512  | 32 | 1.073× |
| `d2560_e32_t1k`  | 1024 | 32 | 1.161× ← **realistic operating point** |
| `d2560_e64_t1k`  | 1024 | 64 | 1.166× |
| `d2560_e32_t2k`  | 2048 | 32 | 1.270× |
| `d2560_e32_t4k`  | 4096 | 32 | 1.347× |

tokens/expert (per-device batch) is the dominant lever; E_local saturates ~32. The 512→4096 curve is
the fixed-overhead amortization curve — the loop's job is to flatten its low end. Profile of a real
GM2560-B16-EP8 run showed the expert kernel at 18.75% occupancy (separate headroom).

## Hypothesis queue

Ranked; re-rank after the GFP8-OPT-P01 profile attributes fp8's time.

- **H1 — wgrad cast-transpose tax dominates at low tok/expert** → bf16-wgrad may beat fp8-wgrad at the
  realistic point. Test: free A/B (`--mosaic-wgrad bf16` vs `fp8`). _Status: running (P02)._
- **H2 — grad recipe** → does `e4m3` grad shift the low-batch curve vs `e5m2`? Free A/B
  (`--grad-dtype`). _Status: running (P02)._
- **H3 — block configs curated for old F=5632** → re-tune candidate set for F=1280 to lift the
  18.75% occupancy. _Status: queued (needs profile to confirm occupancy is the limiter)._
- **H4 — quant/dequant + delayed-scaling fixed cost is the floor at 512 tok/expert** → fuse quant /
  simplify scaling. _Status: queued (needs profile attribution)._
- **H5 — fwd vs bwd attribution** → which pass does fp8 help/hurt? Folded into P01 (capture both).

## Experiment log

### GFP8-OPT-P01 — profile the realistic point (attribution)

**Goal:** attribute the fp8 fwd+bwd e2e time at `d2560_e32_t1k` into Mosaic GEMM kernels (fwd / dlhs /
wgrad) vs fp8 fixed overhead (quantize/scale, cast-transpose, dequant). Captures bf16 too for contrast.

**Method:** new `--profile` mode in `bench_ragged_fp8_autotune.py` (default GFP8 block config = the
held d2560 winner) writes jax profiler traces to `<out>/profiler/{fp8,bf16}/`; analyzed in-job with
`lib/marin/tools/profile_summary.py` (top ops by exclusive time = per-kernel on the GPU device plane);
raw traces tarred to `s3://marin-na/marin/grug-fp8/profiles/` as backstop.

**Job:** `/matt/iris-run-job-20260628-230523` (SUCCEEDED). Trace tarball:
`s3://marin-na/marin/grug-fp8/profiles/d2560_e32_t1k.tgz`. Device-exclusive op breakdown (xplane top
ops, 30 captured steps at the GFP8 default config; arbitrary-but-consistent units, relative shares are
what matter):

| | fp8 | share | bf16 | share |
|---|---|---|---|---|
| GEMM kernels | `body_mosaic_gpu_kernel` ×270 | **71.4** | 5× `_lambda_*` Triton | 81.8 |
| cast-transpose (wgrad) | `input_convert_transpose_fusion` | **13.8** | 2× `wrapped_transpose` (plain) | 10.5 |
| quant converts | `loop_convert_fusion` + `wrapped_convert` | **6.6** | — | 0 |
| scaling | `loop_broadcast_fusion` + `loop_multiply_fusion` | **5.0** | (broadcast+multiply) | 4.4 |
| concat (gated MLP) | `input_concatenate_fusion` | 3.1 | `input_concatenate_fusion` | 3.3 |
| **sum (same units)** | **102.9** | | **123.8** | |

**Findings (this reframes the loop):**
- The fp8 **GEMMs alone are ~1.38× faster** than the bf16 GEMMs (73.5 vs 101.3 same units). But fp8
  carries **~28% fixed overhead** (cast-transpose 13.8 + converts 6.6 + scaling 5.0 + misc), dragging
  the net device-busy ratio to **~1.20×** (102.9 vs 123.8) — consistent with the ~1.16× wall headline.
  **So the GEMM win is real and large; overhead is what we're leaving on the table.**
- Highest-leverage target: **`input_convert_transpose_fusion` (13.8%)** — the fused
  quantize+transpose feeding the wgrad GEMM. bf16's equivalent plain transpose is 10.5%, so the *net*
  fp8 tax here is ~3.3pts, but the absolute 13.8% is the biggest single non-GEMM op. (H1 tests whether
  dropping fp8 wgrad entirely is a net win — but that also forfeits the wgrad GEMM's fp8 speedup.)
- **Quant converts (6.6%) are pure fp8 tax with no bf16 analog** → candidate for fusing the
  bf16→fp8 cast into the Mosaic kernel prologue (quantize-on-load) instead of a separate XLA fusion.
- **Scaling (5.0%) ≈ bf16's broadcast+multiply (4.4%)** → small net tax; deprioritize H4's scaling arm.

_Re-ranked queue → see updated Hypothesis queue above (H1/converts promoted, scaling demoted)._

_Status: DONE._

### GFP8-OPT-P02 — free A/B: wgrad mode × grad dtype at realistic shapes

**Goal:** test H1 + H2 with zero kernel change — both flags already thread through the orchestrator,
and the bf16 baseline is invariant to them, so fp8 absolute medians and speedups are directly
comparable across the 2×2 grid.

**Matrix:** `{mosaic_wgrad ∈ fp8,bf16} × {grad_dtype ∈ e5m2,e4m3}` on
`d2560_e32_t512, d2560_e32_t1k, d2560_e64_t1k`.

**Job:** `/matt/iris-run-job-20260628-230534` (SUCCEEDED). Speedup vs tuned bf16:

| shape (tok/exp) | fp8-wgrad e5m2 | fp8-wgrad e4m3 | bf16-wgrad e5m2 | bf16-wgrad e4m3 |
|---|---|---|---|---|
| t512 (512) | 1.060 | 1.055 | **1.084** | **1.092** |
| t1k (1024) | **1.156** | **1.163** | 1.104 | 1.111 |
| e64_t1k (1024) | **1.145** | **1.147** | 1.092 | 1.096 |

**H1 (wgrad mode) — RESOLVED, crossover:** bf16-wgrad wins at t512 (+3pts, the cast-transpose tax
outweighs the small wgrad GEMM's fp8 speedup); fp8-wgrad wins at t1k+ (the wgrad GEMM is big enough
that its fp8 speedup dominates). **→ select wgrad mode by tokens/expert** (bf16 at ≤512, fp8 above) —
a free config heuristic, no kernel change. Crossover sits between 512 and 1024 tok/expert.

**H2 (grad dtype) — RESOLVED:** speed-neutral (within CI everywhere, as expected — same fp8 GEMM
cost). e4m3 grad gives marginally *better* numerics (rel_frob 0.082 vs 0.115) at no speed cost, but
conflicts with the [[grug-fp8-ragged-mixed-required]] invariant → flag, don't adopt.

_Status: DONE._

## Re-ranked hypothesis queue (post P01+P02)

The profile reframed the goal: there is a genuine **1.38× fp8 GEMM win**; ~28% fixed overhead drags net
to ~1.20×. Free recipe wins (H1/H2) recover ~3pts at t512 only. The structural prize is the overhead.

**Ordering update (user call + autonomy granted 2026-06-28):** R4 promoted ahead of R1; from here I
run the loop autonomously (decide experiments/ordering, report results). R1 deferred — its forward
kernel is ~stock, quantize-on-load needs a second f8 smem buffer + in-pipeline cast for ~2–3% with a
bandwidth tradeoff. R4 targets the bigger 13.8% op and the wgrad kernel is already vendored.

### GFP8-OPT-R4a — disambiguate the 13.8% `input_convert_transpose_fusion`

**Why:** the fused Mosaic cast-transpose kernel (`fp8_cast_transpose_mgpu`) has clean imports → likely
*is* engaged in-container (shapes 32768/2560 both %128==0 conform). So the XLA `input_convert_transpose`
is probably NOT the wgrad cast-transpose but the **forward weight prep**: `_mosaic_pallas_call` does
`jnp.swapaxes(rhs, 1, 2)` on the quantized f8 weights every step (K-contiguous layout for f8 wgmma) —
XLA fuses that convert+transpose. Must confirm before optimizing.

**Method:** profile the fp8 fwd+bwd at t1k in BOTH wgrad modes; the wgrad cast-transpose exists only in
fp8-wgrad mode, the forward weight prep is in both. If `input_convert_transpose_fusion` persists in
bf16-wgrad → it's the forward weight swapaxes (retarget R4 there); if it vanishes → it's the wgrad
cast-transpose. _Status: DONE._

**Job:** `/matt/iris-run-job-20260629-000709` (SUCCEEDED). `_has_pallas_mosaic=True` (fused cast-
transpose IS engaged for activations). `input_convert_transpose_fusion` = **14.21µs in BOTH modes**
(fp8-wgrad 13.8% / bf16-wgrad 13.9%) — **persists** → it is the **forward weight quantize+swapaxes**,
not the wgrad cast-transpose. (bf16-wgrad cross-check: wgrad becomes Triton `_lambda_`, mosaic count
270→90, weight-prep op unchanged.) The activation cast-transpose is absorbed into mosaic kernels (no
XLA op) precisely because it routes through `cast_transpose_mgpu`.

### GFP8-OPT-R4b — route forward weights through the fused cast-transpose

**Target:** the forward weight prep in `fp8_ragged.fp8_scaled_ragged_dot` — `in_q(rhs)` + the XLA
`jnp.swapaxes(rhs,1,2)` inside `_mosaic_pallas_call` (DEFAULT branch). **Plan:** produce both
`q_rhs[G,K,N]` (dlhs) and `q_rhs_t[G,N,K]` (forward) from one per-expert fused `cast_transpose`
(vmap over G), thread `q_rhs_t` through `quantized_ragged_dot` (mirroring the existing `q_lhs_t`
activation plumbing), and call the forward GEMM with the pre-transposed weights so the XLA swapaxes is
gone. Bit-identical (cast_transpose == quantize+.T). No rebuild (haliax edit). Validate via the
numerics gate + profile delta (the XLA op should vanish). _Status: designing._

**Implemented.** New `in_q_transpose_3d` (fp8_cast_transpose.py): per-tensor scale + `vmap(cast_transpose)`
over the expert axis → `(q_rhs[G,K,N], q_rhs_t[G,N,K], new_scale)` with the same overwrite-gradient
custom_vjp as `in_q_transpose`. Threaded `q_rhs_t` through `quantized_ragged_dot` (custom_vjp arity
+1, nondiff_argnums 10–13→11–14, bwd None slot); added `rhs_pretransposed` to `_mosaic_pallas_call` /
`_ragged_dot_layout` so the forward DEFAULT branch skips the XLA swapaxes when given the pre-transposed
weights. Gated on `implementation == "mosaic"` (triton/xla keep `in_q` + natural layout). CPU: 29
haliax fp8 tests pass; xla fp8 fwd+grad runs clean with the new plumbing. **Open risk:** whether
`vmap` over the Mosaic `cast_transpose_mgpu` pallas kernel lowers on H100 — only the cluster can say.

**Validation job:** `/matt/iris-run-job-20260629-004930` (SUCCEEDED). **WIN.**

| shape | baseline | R4b | Δ |
|---|---|---|---|
| t512 | 1.060 | **1.127** | +6.7pts |
| t1k | 1.156 | **1.212** | +5.6pts |
| e64_t1k | 1.145 | **1.220** | +7.5pts |

- **Numerics identical:** grad_rel_frob_vs_bf16 = 0.115 (== baseline), bit-for-bit. Gate passed.
- **XLA op gone:** `input_convert_transpose_fusion` (13.8%) replaced by `batched_body_mosaic_gpu_kernel`
  (the vmapped cast-transpose, ×60). Device-busy sum 103→98µs.
- **Why wall gained more than device-busy:** the old XLA fusion's cost was largely a host/launch/
  materialization gap (the ~4pt wall-vs-device-busy overhead); folding it into a Mosaic kernel
  recovered it. Device-busy ratio 1.20→1.27, wall 1.16→1.21. `vmap` over `cast_transpose_mgpu`
  lowered fine on H100 (the open risk is resolved — no 3D kernel rewrite needed).
- Plateau counter reset to 0 (improvement). **Realistic operating point now ~1.21×** (ceiling ~1.38×).

**Remaining overhead (R4b profile, device-busy):** GEMM 74.9%, batched cast-transpose 13.5% (now the
top non-GEMM, intrinsic to f8 K-contiguity — must transpose weights each step), converts ~3%, scaling
~5%, concat 3.3%. Plus a residual ~6pt wall-vs-device-busy host-overhead gap (1.21 wall vs 1.27 busy).

### GFP8-OPT-R4c — tune the cast-transpose mosaic block (next, cheap probe)

The batched cast-transpose is the #1 remaining non-GEMM (13.5%) and runs at a hardcoded 128×128
(`_MOSAIC_BLOCK`). Probe block sensitivity (e.g. 256) at t1k to decide if full block-config plumbing is
worth it. _Status: DONE — NEGATIVE._

**Job:** `/matt/iris-run-job-20260629-005928`. block=128 (1.213/1.222) is optimal; **256 crashes**
(`cast_transpose_mgpu` doesn't support 256 tiles — workers rc=1, gate None), **64 regresses**
(1.094/1.099). The cast-transpose tile is already best; no block-tuning gain. (Plateau counter: 1 of 5.)
`FP8_CAST_TRANSPOSE_BLOCK` left env-overridable (default 128) for future probes.

## Loop status (after R4b win, R2/R4c negatives)

**Realistic operating point: 1.156× → 1.212× (R4b), +5.6pts, numerics-identical, no rebuild.** This
closed roughly half the device-busy gap to the ceiling (1.20→1.27; wall 1.16→1.21; ceiling ~1.38).

**Easy/structural wins are now exhausted:**
- GEMM (75%): block-saturated (R2 negative) — not improvable by tuning.
- Cast-transpose (13.5%): now a Mosaic kernel, tile-optimal (R4c negative), and **intrinsic** — f8
  wgmma needs K-contiguous weights, so they must be re-transposed every training step. No cheap cut.
- Converts (~3%, `wrapped_convert`): R4b already absorbed the weight quant into the cast-transpose;
  the remainder is the activation/grad quant. R1 (quantize-on-load) could target it but is hard Mosaic
  surgery (2nd f8 smem buffer + in-pipeline cast) for ~2pt — low ROI.
- Scaling (~5%): ≈ bf16's broadcast+multiply — near-wash.
- Host-overhead gap (~6pt wall-vs-busy): partly a measurement artifact (traced profile vs wall metric);
  likely shared with bf16 / mostly cancels in the ratio. Not a clean lever.

**Assessment:** R4b is the milestone. Remaining levers are hard and small (R1 ≈2pt, host-overhead
shaky). Recommend consolidating: commit R4b, port to `grug-fp8-shim`, optionally pursue R1 only if a
further ~2pt is worth the Mosaic surgery. R4b change is purely additive and gated on mosaic, so it's
safe to keep regardless.

### GFP8-OPT-R1 — investigated; target subsumed by R4b

**Job:** `/matt/iris-run-job-20260629-024653` (profile of the committed R4b path at t1k). Residual
non-GEMM ops: `batched_body_mosaic_gpu_kernel` (cast-transpose) 13.5% — intrinsic; `wrapped_convert`
3.1% — the only remaining foldable fp8-specific convert (likely the forward f8→accumulator-dtype
astype); `input_concatenate_fusion` 3.2% / `loop_broadcast` 3.0% / `loop_multiply` 2.2% — **shared with
bf16** (gated MLP + scaling: bf16 had 3.3 / 2.9 / 2.5), so they do NOT affect the fp8/bf16 ratio.

**Conclusion:** R4b folded the weight+activation quantizes into the Mosaic cast-transposes, so the GEMM
receives f8 directly — R1's original 6.6% convert tax fell to a single 3.1% `wrapped_convert`. R1's
realistic ceiling is now ~1–3pt for hard kernel work (a fork-kernel patch so the forward GEMM emits its
accumulator dtype directly instead of f8→astype — would also improve forward numerics). Low ROI vs the
banked R4b win. Plateau counter: 2 of 5, but the remaining levers are all intrinsic/shared/small.

### GFP8-OPT-P03 — isolated single-GEMM efficiency probe (why not 2×?)

**Question (raised post-loop):** the e2e win is 1.21×; David estimated 40–50% for the gmm; the
hardware peak ratio is 2×. Is the ~1.38× GEMM ceiling a *kernel-quality* gap (mgpu leaving headroom)
or a *hardware/size* limit? Decompose first: e2e speedup factors **exactly** as
`2.0 × (fp8 %peak ÷ bf16 %peak) = 2.0 × (28.2 ÷ 48.5) = 1.16×` — bf16 runs at 48.5% of bf16-peak,
fp8 at 28.2% of fp8-peak (clean FLOP count `1.93 PFLOP/step` ÷ wall medians 4.025 / 3.460 ms). So the
whole question is *why fp8 reaches a lower fraction of its (2×-higher) peak*.

**Method:** `bench_fp8_gemm_efficiency.py` — isolated forward single GEMM (G=1, pre-quantized operands,
no transpose/quant in the timed region), four variants: `fp8_mosaic` (`_mosaic_ragged_dot`, the
production fwd kernel), `bf16_triton` (`ragged_dot` auto, the baseline), `fp8_dense`/`bf16_dense`
(`jax.lax.dot_general` → cuBLASLt, the *mature-kernel* upper bound). Achieved TFLOP/s and % of H100 SXM
peak. M-sweep at K=N=2560 plus an 8192³ calibration. **Job:** `/matt/iris-run-job-20260629-042330`
(SUCCEEDED, H100x1).

**Caveat:** at M≤2048 the isolated single GEMM is **launch-overhead-bound** (flat ~25–29 µs floor
across t512/t1k/t2k despite 2–4× FLOP), so small-M absolute %-peak is contaminated; the **compute-bound
large shapes are the clean signal**, plus the *ratios* and trends.

| shape (M,K,N) | fp8_mosaic | bf16_triton | fp8_dense (cuBLAS) | bf16_dense (cuBLAS) |
|---|---|---|---|---|
| t2k (2048·2560²) | 47.5% | 45.7% | **39.5%** | 65.1% |
| t4k (4096·2560²) | 57.0% | 52.8% | **50.5%** | 66.5% |
| t8k (8192·2560²) | 58.6% | 47.2% | **59.8%** | 68.4% |
| 8192³ | 62.8% | 42.0% | **63.7%** | 70.2% |

**Findings — the kernel-headroom hypothesis is FALSIFIED, reframing the conclusion:**
1. **The mgpu fp8 kernel is at cuBLASLt parity when compute-bound** (58.6% vs 59.8% at t8k; 62.8% vs
   63.7% at 8192³). There is **no kernel-quality headroom** to chase — R2's negative was right, and now
   we know *why*: the kernel is already as good as cuBLAS. (My earlier "mgpu ~40% vs achievable ~65%"
   guess was wrong.)
2. **fp8 < 2× at the realistic point is a fundamental size/occupancy effect.** Even cuBLASLt fp8 reaches
   only **39.5%** of fp8-peak at M=2048 (vs bf16 65% of bf16-peak), because the 2×-faster fp8 cores
   can't be fed at K=2560, M~1024–2048. So the *mature-vs-mature* achievable fp8/bf16 ratio at these
   per-expert sizes is **~1.2–1.5×, not 2×**; it only clears ~1.75–1.8× at M≥8192. **Triangulation:** the
   e2e-apportioned fp8 GEMM efficiency (~40% of peak) ≈ cuBLASLt's 39.5% at t2k → the real grouped kernel
   is already at the hardware-achievable limit for this size. This is the same signal as the real-run
   18.75% expert-kernel occupancy.
3. **bf16 triton sits well below the *dense* cuBLASLt ceiling (49% vs 65–73%)** — but this is the
   **raggedness tax, shared by every ragged kernel**, NOT a triton-specific deficiency. ⚠️ **CORRECTED
   (see P03b + the `bf16-ragged-mosaic-vs-triton` memory):** I originally wrote this as "triton is
   undertuned → our fp8 win is honest-to-generous." **That framing was wrong.** A separate bf16-kernel
   investigation (branch `grug-bf16-mosaic-ragged`) tuned JAX-mgpu and tokamax on the same forward and
   found them *all* clustered at ~48–50% of bf16-peak — i.e. **at triton's level**. The gap to dense is
   structural (one fat GEMM + shared weight + static shape), unreachable by any deployable ragged kernel.
   The 8192³ "2.99×" is triton's large-M block-config *droop* (off the operating point), not the realistic
   regime. So there is **no faster deployable bf16 baseline to tune toward**; triton is at the ragged
   ceiling, and our fp8 win vs it (1.90× fwd / 1.21× e2e) is the *real* number, not "generous."

**Reframed conclusion:** we are **essentially at the achievable GEMM ceiling for the realistic operating
point** (1.21× e2e vs a ~1.2–1.4× mature-GEMM ceiling at 1024–2048 tok/expert). David's 40–50% is real
but lives at **larger per-expert batch** (t4k+ where the GEMM ratio is ~1.5–2.1× and e2e is 1.35×). The
high-leverage lever is therefore **not** kernel or overhead work — it is the **operating point**: raise
tokens/expert (bigger microbatch / lower EP / grad-accum / expert grouping), and fp8's own activation-
memory savings can *fund* that bigger microbatch. _Status: DONE. Supersedes the "kernel saturated /
1.38× ceiling" framing below._

### GFP8-OPT-P03b — faithful grouped sweep + batched-dense references (apples-to-apples)

**Why:** the P03 single GEMM is launch-bound at M≤2048, and its single-fat-GEMM `*_dense` ceiling is
optimistic (one shared weight, M=E·tok). Re-ran **grouped** (ragged, E=32 balanced groups, K=N=2560,
forward only, pre-quantized) so launch amortizes over experts — the faithful MoE setting — and added a
**batched** `dot_general` reference (distinct per-expert weights, equal groups → strided-batched).
**Jobs:** `/matt/iris-run-job-20260629-044730` (grouped), `/matt/iris-run-job-20260629-133259`
(+batched, +HLO check). Plots in session.

**Grouped forward ratio (fp8_mosaic / bf16_triton) climbs with tokens/expert:** 128→1.49×, 512→1.66×,
**1024→1.90× (operating point)**, 2048→1.95×, 4096→2.08×, 8192→2.59×. At 1024 both kernels sit at the
same fraction of their dtype peak: **bf16_triton 48.6% vs fp8_mosaic 46.1%** → the 1.90× is essentially
`2.0 × (46.1/48.6)` = pure hardware-peak ratio, **not** bf16 weakness. The fat forward GEMM is already
near-2×; the e2e shortfall to 1.21× is dot2 (skinny K=1280), the wgrad (contraction = tokens), and the
~28% fp8 overhead.

**Batched-dense HLO check — fp8 does NOT lower to cuBLAS.** `HLO-CHECK fp8_batched: cublas_custom_calls=0
triton_mentions=1` — XLA routes batched **fp8** `dot_general` to a Triton emitter (15–18% of peak at the
operating point, *worse than bf16*; fp8_mosaic beats it 2.5× → endorses the mgpu kernel). So there is **no
clean batched-cuBLAS fp8 ceiling via `dot_general`** (would need cuBLASLt's fp8 grouped-GEMM API direct).
bf16 batched *did* lower cleanly (~68%), giving a valid bf16 grouped ceiling.

**bf16 ceiling hierarchy (each step a real, unrecoverable structural cost):** dense single fat GEMM
(shared weight, static) ~73% → batched cuBLAS (distinct weights, *static equal* groups) ~68% → **ragged
kernels (dynamic groups): triton ≈ mgpu ≈ tokamax ~49%** = deployable reality. The 68% batched number is
NOT reachable by a deployable ragged kernel (needs static equal groups), so the orange "vs ideal bf16
1.35×" comparison I drew in-session is *not* the deployed scenario — the real fp8 win is vs triton
(1.90× fwd / 1.21× e2e). **Cross-thread reconciliation (with `grug-bf16-mosaic-ragged`):** "triton is
below peak" is true only vs *dense*; vs achievable-ragged it's at the ceiling, all ragged kernels tie,
and the fp8/bf16 ratio reflects fp8's 2× peak, not bf16 slack → no easy bf16 win exists. _Status: DONE._

### GFP8-OPT-P04 — wgrad-mode crossover vs tokens/expert (full 2^7–2^13 sweep)

**Why:** P02 found the fp8-wgrad vs bf16-wgrad crossover sits between 512 and 1024 tok/expert at three
points; this maps the full power-of-two curve (128…8192) to confirm the operating regime favors fp8 and
to quantify how the margin scales. Two best-vs-best orchestrator runs differing only in `--mosaic-wgrad`,
E_local=32 (EP8), real d2560 (D2560/F1280), n=40.
**Jobs:** `/matt/iris-run-job-20260630-163824` (fp8), `/matt/iris-run-job-20260630-163830` (bf16).

| tok/expert | 128 | 256 | 512 | **1024** | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| fp8-wgrad  | 1.010 | 1.061 | 1.115 | **1.207** | 1.305 | 1.380 | 1.480 |
| bf16-wgrad | 1.097 | 1.117 | 1.171 | 1.159 | 1.137 | 1.151 | 1.188 |

**fp8-wgrad rises monotonically** (the token-axis wgrad GEMM grows → its fp8 2× dominates the fixed
per-expert cast-transpose tax); **bf16-wgrad is flat** ~1.10–1.19 (its wgrad never gets fp8 large-GEMM
scaling — only the shared fwd+dlhs is fp8). Crossover between 512 and 1024; fp8-wgrad wins at the
operating point (≥1024) and the margin only widens (+5pt @1024 → +23pt @4096). **Confirms the R3
decision** to drop the bf16-fallback crossover: production (EP4–8, ≥1024) is squarely fp8-wgrad's regime.
t1k fp8-wgrad 1.207 matches the post-R4b figure (1.212) → this sweep includes the fused-cast-transpose win.

**Setup fix:** the fork-setup dlhs build-patch had gone stale — `mcwitt/jax` now carries that relaxation
itself (commit 7ee3f26, `_is_mixed_fp8` form), so the anchor no longer matched and the fail-fast tripped;
made the patch recognize the fork's committed form and skip (now a fallback for older refs).

**Reproducible artifacts (new convention — prior plots were lost as "in session"):** raw data + a
reusable plot script + provenance README are committed under
`lib/levanter/scripts/bench/results/wgrad_tokens_per_expert/`. Regenerate the PNG from committed data
(no cluster) via `plot_speedup_vs_tokens.py` — see that README. _Status: DONE._

## Loop conclusion

**Result: realistic operating point 1.156× → 1.212× (R4b), validated, committed (`7f5268deb7`).** The
loop closed ~half the device-busy distance to the ~1.38× ceiling; the rest is intrinsic (the f8
K-contiguity cast-transpose, 13.5%) or shared-with-bf16 (doesn't move the ratio). R2/R4c/R1 are
negatives/subsumed that bound the remaining space. Recommend banking R4b and porting to
`grug-fp8-shim`; the one optional further lever is the forward-output-dtype kernel patch (~1–3pt + a
numerics improvement). Kill the 6.6% pure-quant convert tax (`loop_convert`
  + `wrapped_convert`, no bf16 analog) by fusing the bf16→fp8 cast into the Mosaic GEMM prologue
  instead of a standalone XLA fusion. Helps *every* shape (overhead on every GEMM input). _Queued._
- **R2 (kernel) — re-tune block configs for F=1280.** The GEMM is 71% of fp8 time; the candidate set
  (`_MOSAIC_RAW`/`_WGRAD_RAW`) was curated for the old F=5632. Re-curate for F=1280 to lift GEMM
  throughput / SM occupancy (real GM2560 profile showed 18.75%). _Queued._
- ~~R3 (config) — per-shape wgrad-mode crossover~~ **DROPPED from productionization.** The bf16-wgrad
  win is only at t512, which is the small-batch *GPU-profiling* regime, not realistic full-scale
  training (EP4 sits at ≥1024 tok/expert where fp8-wgrad already wins). And the crossover is just a
  workaround for the cast-transpose overhead that R1/R4 are meant to remove — shipping it would add a
  config branch that the kernel work obsoletes. **Kept only as a diagnostic:** after R1/R4, the t512
  crossover should *vanish* (fp8-wgrad should win there too) — a falsifiable check that the overhead
  cut worked.
- **R4 (kernel, hard) — cast-transpose fusion.** The 13.8% `input_convert_transpose_fusion` for wgrad;
  fold convert+transpose into the wgrad kernel prologue. Largest single non-GEMM op but structurally
  hardest (transpose is intrinsic to the wgrad layout). _Queued, after R1._
- ~~H4 scaling~~ — demoted: fp8 scaling (5.0%) ≈ bf16 broadcast+multiply (4.4%), near-wash.

### GFP8-OPT-R2 — re-tune block configs for F=1280

**Goal:** the fwd/dlhs GEMM is M-skinny per expert (~512–1024 tokens), N=2F=2560, K=2560/1280; the
wgrad GEMM has small K=tokens/expert. The candidate set was curated for F=5632. Added 8 mosaic tiles
(wide block_n, small block_m) + 5 wgrad tiles (small block_k) — mosaic 19→23, wgrad 11→15, all
smem-feasible. Coordinate descent still sees the old 128/128/128 winner, so R2 only "wins" if a new
tile beats it. Fast-tier: no rebuild (pure data in `fp8_autotune_configs.py`).

**Baseline to beat (A/B fp8-wgrad e5m2):** t512 1.060× / t1k 1.156× / e64_t1k 1.145×.

**Job:** `/matt/iris-run-job-20260628-232829` (SUCCEEDED, fp8-wgrad e5m2, realistic shapes).

**Result — NEGATIVE (informative):** the winning mosaic tile is still **128/128/128/6/4** at all three
shapes; none of the 8 new F=1280 tiles won. Speedups t512 1.050 / t1k 1.154 / e64 1.164 — all within
CI of the A/B baseline (1.060 / 1.156 / 1.145). No new winner → no profile delta (winner == the
already-profiled config). **Rules out GEMM block-tuning as a lever:** the 71%-of-time GEMM is already
at its tuned best, so the 18.75% occupancy headroom is *not* reachable via block tiling. → net speedup
now lives entirely in the overhead (R1/R4). (Loop plateau counter: 1 of 5.)

## Loop economics — CORRECTED (R1/R4 need no rebuild)

The forward/dlhs Mosaic kernel (`jax.experimental.pallas.ops.gpu.ragged_dot_mgpu.ragged_dot`) and the
wgrad kernel (`haliax._src.transposed_ragged_dot_mgpu`, already vendored) are **pure-Python Pallas
kernels**. R1 (quantize-on-load) and R4 (cast-transpose fusion) only need ops the cached jaxlib already
supports (scale-multiply, `.astype(f8)`, existing `wgmma`/transpose) → **no 11-min jaxlib rebuild**.
Vendor the forward kernel into haliax, edit, and the job bundles the working-tree copy + cached wheel
(~3.5-min iteration). A rebuild is needed *only* if a change requires a new Mosaic/jaxlib C++ primitive
(e.g. a new wgmma variant) — R1/R4 do not.

**R1 is a bandwidth tradeoff, not a free 6.6% kill.** Quantize-on-load removes the XLA convert *and* the
intermediate f8 materialization (write-f8 + reread-f8 HBM round-trip), but the in-kernel load then
reads bf16 (2× that operand's load bandwidth). → win for **lhs** (read once), risky for **rhs**
(reread per M-tile). Scope R1 to **lhs quantize-on-load, rhs stays pre-quantized**.

## Finalized loop spec

- **North star:** lift the realistic operating point (t1k, 1024 tok/expert); big-batch ceiling is a
  nice-to-have. Tracking metric: **net speedup at t1k and t512, fp8-best vs tuned-bf16.** Today
  ~1.16× / ~1.09×. Ceiling ≈ **1.38×** (overhead → 0).
- **Per-iteration protocol (rigor = metric + profile delta):** edit → rebuild forked wheel *only if
  kernel source changed* → run the metric on the realistic shapes (CIs) → before/after profile to
  confirm the targeted op shrank → logbook entry. Numerics gate (rel_frob vs bf16) is built into the
  metric.
- **Queue order:** R2 (fast, in flight) → R1 (quantize-on-load, first kernel change) → R4
  (cast-transpose fusion). R3 dropped. Re-rank after each profile delta.
- **Stop criterion:** realistic-point speedup plateaus across **5** consecutive changes, or ~80% of
  the overhead gap closed (≈1.33×).
- **Process:** lightweight — this logbook drives the queue; commit/snapshot at milestones; promote to
  GitHub/W&B only once results are worth publishing.
