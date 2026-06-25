# Plan: Pallas FP8 cast-transpose kernel (TE-style) to land the f8 wgrad win

Status: **M1 + M2 DONE on jax 0.10.0 (no bump). M3 (e2e wiring) is next.** Decision: Mosaic backend,
wgrad-only, amax deferred (per user). Branch: `research/grug-fp8-h100`.

## Outcome (2026-06-25, GFP8-033)
- **M1 ✅** `haliax/_src/fp8_cast_transpose.py` reference + 19 CPU bit-exact tests (vs `(quantize,
  quantize.T)`), committed.
- **M2 ✅ WORKS on jax 0.10.0 — no bump needed.** The fused Mosaic kernel lowers and is **bit-exact**
  (q_exact/qt_exact on all wgrad shapes). The earlier "blocked, needs a jax bump" call was wrong: it used
  the wrong idiom (a *memref* transpose of the swizzled store). The working idiom is a **register layout
  cast** (JAX's `test_transposed_load_store`): transpose the **f32** quant via
  `layout_cast(qf, WGMMA_TRANSPOSED)` (same-dtype only; f8's WGMMA_8BIT won't convert), `astype(f8)` at
  the store, into a **plain (non-swizzled)** `transpose_ref(qt_smem,(1,0))`. All machinery
  (`Layout.WGMMA_TRANSPOSED`, `WGMMA_8BIT`, `layout_cast`, `handle_transposes`, `memref_transpose`) is in
  0.10.0 already (April-2025). Perf: across the 4 wgrad cast-transposes ~0.61–0.68 ms (fused) vs ~0.95 ms
  (quantize+swapaxes) → **saves ~0.27 ms** → flips f8 wgrad from a ~0.12 ms e2e loss to ~1.34×. Kernel
  cleaned + wired into the public `cast_transpose` (H100 + 128-tileable → Mosaic, else reference).
- **M3 ✅ DONE — f8 wgrad e2e 1.333× (beats the bf16 hybrid).** Wired `cast_transpose` into `fp8_ragged`:
  forward `in_q_transpose` → `q_lhs`/`q_lhs_t` from one read; backward `cast_transpose(g)` → `q_g`/`q_g_t`;
  the f8 wgrad calls `_mosaic_wgrad_transposed(q_lhs_t, q_g_t)` directly (no `swapaxes`). 3-arm e2e (H100,
  grad_dtype=e4m3): bf16 3.738 ms; mosaic bf16-wgrad 2.926 ms (1.278×); **mosaic f8-wgrad 2.805 ms
  (1.333×)**, dw13/dw2 7.16e-2 / 6.42e-2 (in-band). Gate (< 2.94 ms, in-band) met. Reverses the
  GFP8-028/029 "f8 wgrad is a net e2e loss" verdict. Gated on `RAGGED_F8_WGRAD` (mosaic only); 33 CPU tests
  green. Full evidence: logbook GFP8-033 M3.
- **Remaining (M3 close-out, pending user sign-off):** flip the mosaic wgrad default to f8 + retire the
  `RAGGED_F8_WGRAD` env toggle (a ~4%-gain-vs-complexity call; the bf16 hybrid stays simpler).

---

(original plan below — superseded by the outcome above)

Experiment IDs: **GFP8-033+**.

## Problem & goal

Hopper f8 `wgmma` needs the contraction axis contiguous, so the f8 weight-gradient must materialize
token-contiguous (transposed) f8 operands — a transpose bf16 never pays (16-bit wgmma transposes
in-kernel). The f8 wgrad *kernel* already beats bf16 (1.2–1.3×, GFP8-029), but the standalone transpose
sinks the full op (e2e 3.06ms vs the bf16-wgrad hybrid 2.94ms). GFP8-030 tried to fuse the transpose into
the quant cast via `swapaxes(quantize(x))`, but that is a **second, independent cast** that re-reads the
bf16 source (1 read + 2 writes becomes 2 reads + 2 writes), so the transposed copy cost a full re-cast
(~0.10ms) — worse than the standalone f8→f8 transpose (~0.075ms) it replaced. No win.

**Transformer Engine's fix:** one fused `cast_transpose` kernel reads the high-precision tensor **once**
and writes **both** f8 layouts (rowwise + columnwise). Marginal cost of the transposed copy ≈ one extra
f8 write. XLA will not synthesize this (it keeps the two casts separate), so we must write the Pallas
kernel ourselves. (Capability confirmed: a Pallas kernel emitting `(q[M,K], qT[K,M])` from one input is
bit-exact `qT==q.T` in interpret mode.)

**Win condition:** f8-wgrad e2e **< 2.94ms** (beats the shipped bf16-wgrad hybrid). Target ~2.7–2.8ms
(**~1.34–1.38×**, up from 1.27×). Numerics unchanged from the existing f8 wgrad (dw13/dw2 ~7e-2, in-band).

## Public API (per add-pallas-kernel)

```python
# haliax/_src/fp8_cast_transpose.py  (NEW)
def cast_transpose(
    x: jax.Array,          # [M, K] high-precision (bf16/f32)
    scale: jax.Array,      # per-tensor delayed-scaling scale (scalar)
    *,
    out_dtype: jnp.dtype = jnp.float8_e4m3fn,
    compute_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jax.Array, jax.Array]:   # (q[M,K] rowwise, qT[K,M] transposed), both out_dtype
    """One HBM read of x -> both the rowwise and transposed f8 quantizations (TE cast-transpose).

    Bit-identical to (quantize(x, out_dtype, scale, compute_dtype),
                      swapaxes(quantize(...), 0, 1)) — the transpose is produced in-kernel, not by a
    second cast. GPU (Pallas) fast path; reference fallback elsewhere."""
```

- **Reference (correctness oracle):** `q = quantize(x, out_dtype, scale, compute_dtype); return q, q.T`
  using the existing `haliax._src.fp8.quantize`. This is the bit-exact target.
- **Wrapper / fallback:** Pallas-Triton on GPU; the vanilla reference on CPU/TPU or when Pallas is
  unavailable (same guard pattern as `ragged_dot`'s `_has_pallas_triton`).
- **Scale is an INPUT** (delayed scaling: the step's scale comes from the *previous* amax history, known
  before the cast). Amax-for-next-step stays a separate reduction in `_new_scale_and_history` for v1
  (fusing it is a stretch goal — see M4).

## Kernel design (Pallas-Triton primary)

Grid over `(M//block_m, K//block_k)`; each program:
```python
def _cast_transpose_kernel(x_ref, scale_ref, q_ref, qt_ref, *, fp8_max):
    x = x_ref[...].astype(jnp.float32)            # [bm, bk]
    s = jnp.clip(x / scale_ref[0], -fp8_max, fp8_max)
    q = s.astype(out_dtype)                        # [bm, bk]
    q_ref[...] = q                                 # rowwise store  [bm, bk] @ (i, j)
    qt_ref[...] = q.T                              # transposed store [bk, bm] @ (j, i)
# out_specs: BlockSpec((bm, bk), lambda i, j: (i, j))   for q   [M, K]
#            BlockSpec((bk, bm), lambda i, j: (j, i))   for qT  [K, M]
```
Key perf detail: the transposed store must stay coalesced. Triton lowers the in-register/SMEM tile
transpose (`q.T`); square-ish tiles (e.g. 64×64 / 128×128) give a clean SMEM transpose. **M2 must
confirm the transposed store is coalesced on H100** — if Triton's transpose-store underperforms, fall
back to (a) a Mosaic-GPU kernel whose TMA does transposed stores natively, or (b) accept a slightly
sub-peak transpose that is still cheaper than the redundant re-cast (the bar is the GFP8-030 baseline,
not peak bandwidth).

- `cost_estimate=`: IO = read M·K·`bf16` + write 2·M·K·`f8`; ~0 FLOPs (memory-bound). Use
  `pl.estimate_cost` on a body-equivalent JAX fn + `with_io_bytes_accessed`.
- Backend decision criterion recorded in M2; default Triton unless its transpose store is >1.3× the
  read-bound floor.

## Integration into the f8 ragged path

`fp8_ragged.py`, mosaic + f8-wgrad branch (the path currently behind `RAGGED_F8_WGRAD`):

- **Forward** (`quantized_ragged_dot_fwd`): the activations `lhs` (x) feed the forward (rowwise) AND the
  wgrad (transposed). Replace `q_lhs = in_q(...)` for x with:
  ```python
  new_lhs_scale = _new_scale_and_history(lhs, e4m3, lhs_scale, lhs_amax_history)[0]
  q_lhs, q_lhs_t = cast_transpose(lhs, new_lhs_scale)   # one read -> both layouts
  ```
  Store `q_lhs_t` as a residual (f8 [K, M], ~16MB). Forward GEMM still consumes `q_lhs`.
- **Backward** (`quantized_ragged_dot_bwd`): the grad `g` (dout) feeds dlhs (rowwise) AND wgrad
  (transposed): `q_g, q_g_t = cast_transpose(g, new_out_grad_scale)`. dlhs uses `q_g`; wgrad uses
  `q_g_t`.
- **Wgrad**: `grad_rhs = _transposed_ragged_dot(q_lhs_t, q_g_t, group_sizes, out_dtype=pref)` directly —
  no `swapaxes` (operands already token-contiguous). Then dequantize by `lhs_scale * new_out_grad_scale`.
- This makes the f8 wgrad the proven default for the mosaic path: replace the `RAGGED_F8_WGRAD` env
  toggle with a clean decision (default on **iff** the e2e win is confirmed in M3; else stays a config).

Net traffic vs the current bf16-wgrad path: +1 f8 write each for `q_lhs_t`/`q_g_t` (the rowwise copies
replace the casts we already did), and the bf16 wgrad GEMM (~1.16ms) → f8 wgrad (~0.91ms). Expected e2e
~2.74ms.

## Milestones

- **M1 — reference + correctness harness (CPU).** `cast_transpose` reference; pytest under
  `lib/haliax/tests/` checking bit-exact vs `(quantize, swapaxes)` over a shape/dtype grid (e4m3/e5m2,
  odd shapes, the real wgrad shapes [8192,2048]/[8192,11264]); interpret-mode Pallas == reference.
- **M2 — Pallas kernel + GPU correctness + standalone perf (H100).** Bit-exact vs reference on GPU.
  Time the kernel vs three baselines: (a) GFP8-030 redundant re-cast `swapaxes(quantize(x))`, (b)
  standalone f8 transpose of pre-cast f8, (c) the read-bound floor (M·K·bf16 read). Decide Triton vs
  Mosaic. Target: both outputs ≈ one cast + ~one extra f8 write (clearly < (a)).
- **M3 — wire into fp8_ragged + e2e (H100).** 3-arm parity rerun (bf16 / bf16-wgrad hybrid /
  f8-wgrad-via-cast-transpose). **Gate:** f8 arm < 2.94ms AND dw13/dw2 in the ~6–8e-2 band. If win,
  flip the mosaic wgrad default to f8 and remove the env toggle; commit.
- **M4 — stretch.** (i) Apply the same kernel to the **forward weight transpose** (GFP8-031, ~0.21ms /
  ~8% e2e): cast-transpose the weight to keep both [G,K,N] and [G,N,K] f8 layouts. (ii) Fuse the
  **amax reduction** as a third output (read x once for amax+cast+transpose, saving the separate amax
  read in `_new_scale_and_history`). Each measured independently; keep only if it earns its complexity.
- **Autotune + tuned table** for block_m/block_k per the perf-workflow; checked-in defaults +
  autotune-on-miss.

## Correctness, perf, risks (per add-pallas-kernel Definition of Done)

- **Numerics:** the kernel must match XLA's f8 round-to-nearest-even bit-for-bit (M1/M2 assert
  `array_equal`). Risk: Pallas/Triton f8 cast rounding differs from XLA → drift. Mitigation: test it;
  if it differs, match XLA's rounding explicitly in the kernel.
- **Transpose-store efficiency** is THE perf risk (uncoalesced writes). Mitigation: square SMEM-transposed
  tiles; M2 measures vs the read-bound floor; Mosaic-TMA fallback.
- **Residual memory:** +`q_lhs_t` (f8 [K,M]) residual per quantized activation; small vs activations.
- **DoD:** values bit-exact on the grid; e2e f8 wgrad < bf16 hybrid with in-band numerics; cost_estimate
  attached; bench emits machine-readable rows; tuned block table checked in; CPU fallback tested.

## Open decisions for the user

1. **Backend default** — Triton (recommended; memory op, existing infra) vs Mosaic (native TMA transpose
   but heavier; the wgrad consumer is already Mosaic). Decided empirically in M2; which do you prefer as
   the starting point?
2. **Scope of the first pass** — wgrad only (M1–M3), or include the forward weight transpose (M4-i) in the
   same push? Wgrad-only is the cleaner win to validate the kernel first.
3. **Amax fusion (M4-ii)** — pursue now (saves another full read of x) or defer until the kernel lands?
