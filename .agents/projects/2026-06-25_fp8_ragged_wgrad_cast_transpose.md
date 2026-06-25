# FP8 ragged weight-gradient (wgrad) via cast-transpose — design & implementation plan

**Status:** revised after codex + glm-5.2 review (v2)
**Owner:** Grug FP8-on-H100 trial, S5
**Branch:** `grug-fp8-s5`
**Logbook:** GFP8-022 (mosaic f8 fwd), GFP8-024 (f8 wgrad MEASURED `FAIL` at the transpose wall),
GFP8-025 (field uses out-of-kernel cast-transpose), GFP8-027 (autotune → e2e 1.06× fwd+bwd; wgrad still bf16)

> **v2 correction (from review).** v1 wrongly proposed forking `ragged_dot_mgpu` and authoring a
> ragged-contraction kernel from scratch. The correct wgrad kernel already exists —
> `jax.experimental.pallas.ops.gpu.transposed_ragged_dot_mgpu` — and a prior session already vendored +
> f8-patched it (`lib/levanter/scripts/bench/_transposed_ragged_dot_f8.py`). It is efficient for 16-bit
> grads but **fails for f8** because it does an in-kernel `wgmma(acc, transpose_ref(lhs), rhs)`, and
> Hopper f8 wgmma forbids operand transposes (`mosaic/gpu/wgmma.py:147` `supports_transpose =
> bytewidth==2`). The fix is to **adapt that kernel** to consume **physically cast-transposed**
> token-contiguous operands so the in-kernel transpose is dropped — not to write a new kernel.

---

## 1. Problem

M0 hybrid runs fwd + dgrad in f8, **wgrad in bf16** (`fp8_ragged.py:188-196`); after autotune it nets
1.06× fwd+bwd (GFP8-027). wgrad is the largest remaining bf16 fraction.

`drhs[g,k,n] = Σ_{m∈g} lhs[m,k]·dout[m,n]` contracts the **ragged token axis** → dense `[G,K,N]` output.
Its kernel is `transposed_ragged_dot_mgpu` (ref dim-numbers `(((0,),(0,)),((),())), lhs_ragged=[0],
rhs_group=[]` == our `_DRHS_DIM_NUMS`). On Hopper f8 it **fails at the wgmma operand transpose** — a
hardware limit (f8 tensor core consumes only the TN/K-contiguous layout). Measured: GFP8-024 wgrad f8
entries = `FAIL`. No library hands us a Hopper f8 wgrad (GFP8-025: flax dense-only; qwix→XLA (no f8
ragged); tokamax Hopper is bf16, f8 only on Blackwell). TE/DeepSeek solve it with an **out-of-kernel
cast-transpose**, which is this plan.

## 2. Mechanism — cast-transpose (confirmed necessary)

Materialize **token-contiguous f8 copies** of both wgrad operands so the contraction (token) axis is
innermost, making the per-group GEMM a standard f8 wgmma with **no in-kernel transpose**:

```
q_lhs [M=tok, K=hid]  ──cast-transpose──▶  q_lhsT [K=hid, M=tok]   (tok contiguous)
q_g   [M=tok, N=out]  ──cast-transpose──▶  q_gT   [N=out, M=tok]   (tok contiguous)
drhs[g] = q_lhsT[:,grp_g] @ q_gT[:,grp_g]^T  →  [K,N],  contraction = tok (contiguous) ✓ TN-legal
```

Both operands must be transposed (not just lhs): the stock kernel feeds rhs without `transpose_ref`
relying on 16-bit's layout freedom; for f8 the rhs's token axis must also be physically contiguous.

**Staging:** M1 materializes `q_lhsT`/`q_gT` with `jnp.swapaxes` (XLA f8 transpose — a bandwidth pass,
lossless). M3 fuses the transpose into the quant sites if it profiles as a meaningful fraction
(codex/glm both flag: **measure the transpose-only cost**; the whole prize is ~5%, GFP8-024).

## 3. Design decisions

### D1 — Adapt the vendored transposed kernel; drop the in-kernel transpose

Start from `_transposed_ragged_dot_f8.py` (already has: group head/tail boundary masking, empty-group
skip `@pl.when(group_size>0)`, and the f8 boundary-mask-via-f32 patch — all inherited, no rework).
**Change:** feed pre-transposed operands `lhs_t [K,M_tok-contiguous]`, `g_t [N,M_tok-contiguous]` and
replace `wgmma(acc, transpose_ref(lhs_smem,(1,0)), rhs_smem)` with a **no-transpose** `wgmma(acc,
lhs_smem, g_smem)` over the (now token-contiguous) operands. Restructure the BlockSpecs so the pipelined
axis (block over tokens) is the contiguous axis of both inputs. Keep `nd_loop((g, grid_m*grid_n))`,
`emit_pipeline` over the group's token blocks, and the **dense** SMEM→GMEM store (NOT the log TMA-store
ladder — that's a ragged-*output* artifact; review [SHOULD-FIX]).

New module: `lib/haliax/src/haliax/_src/transposed_ragged_dot_mgpu.py` (promote the vendored f8 kernel
into the library, with the `# Adapted from jax…transposed_ragged_dot_mgpu` provenance header).

### D2 — Explicit `out_dtype` (review [SHOULD-FIX])

Accumulate f32, **store `preferred_element_type`** (not f8-then-cast): `out_shape=ShapeDtypeStruct(
(G,K,N), out_dtype)`, `o_smem[...] = acc.astype(out_dtype)`. The wgrad dequant
(`fp8_ragged.py:207-209`) assumes the result is already in `preferred_element_type`.

### D3 — All-E4M3 (unchanged); same-dtype operands; scaling reuses the existing f8 branch

grad_dtype=e4m3 already forced. Once `_DRHS` is mosaic-supported, the mosaic branch falls through to the
existing unified f8 `drhs` path (`fp8_ragged.py:198-209`, dequant by `lhs_scale·new_out_grad_scale`).

### D4 — `emit_pipeline` dynamic grid (glm key risk)

The inner pipeline grid is `(group_num_blocks_gmem[g_i],)` — **data-dependent**. The vendored kernel
already uses this and compiled in the prior bench, so it is supported; M1 re-confirms on H100 first.
Fallback if it regresses: pad each group's token count up to a `block_tok` multiple (fixed grid) — glm's
padding-to-fixed alternative; costs padded-token waste, keep as contingency only.

## 4. Code changes

1. **New:** `lib/haliax/src/haliax/_src/transposed_ragged_dot_mgpu.py` — adapted kernel (D1/D2), with a
   `WgradBlockConfig` frozen dataclass + autotuned default (filled by M2).
2. **`nn/ragged_dot.py`** — wire the `_DRHS_DIM_NUMS` branch of `_mosaic_pallas_call` (currently raises,
   `:459-463`): cast-transpose both operands, call the kernel, return `[G,K,N]` in `out_dtype`.
3. **`_src/fp8_ragged.py`** — delete the bf16 wgrad fallback (`:188-196`) + `_WGRAD_FALLBACK_DTYPE`
   (`:38-41`) **only after** the f8 kernel passes GPU parity (codex [CONSIDER]); update docstrings.

## 5. Milestones

- **M1 — wire + verify (small).** Promote the vendored kernel, apply D1 (physical transpose, drop
  in-kernel transpose) + D2 (out_dtype). Validate f8 lowers on H100 and matches the f32 reference within
  ~6–8% rel-frob. This is a *wiring + small kernel edit*, not a from-scratch kernel.
- **M2 — autotune + e2e.** Tune `block_m/n/k/steps/grid_block_n` at the real Grug shape; bake winner;
  re-run `bench_ragged_mosaic_hybrid_e2e.py`. Add a **transpose-only timing row**. Delete the bf16
  fallback. Target: e2e fwd+bwd above 1.06× (GFP8-024 projects ~1.38× regime-relative; expect a small
  but real increment — confirm it exceeds the transpose cost).
- **M3 — fuse (conditional).** If transpose cost is material, fuse cast-transpose into the quant sites.

## 6. Validation

- **Reference oracle:** `jax.lax.ragged_dot_general(..., _DRHS_DIM_NUMS, preferred_element_type=f32)` on
  dequantized operands — max/mean abs-diff + rel-frob.
- **GPU parity:** f8 wgrad vs f32 ref and vs current bf16 drhs — rel-frob in the ~6–8% band (matches
  GFP8-027 dw13 6.38e-2 / dw2 6.10e-2).
- **Full-VJP:** `bench_ragged_mosaic_hybrid_e2e.py` already checks dx/dw13/dw2 — re-run; grads stay in
  band, timing improves.
- **pytest** `lib/levanter/tests/kernels/` (small CPU shapes + H100 path); read `TESTING.md` first.
- **`cost_estimate=`** on the `pallas_call` (skill requirement).

## 7. Open questions — resolved by review

- **Q1 (in-kernel vs physical transpose):** physical cast-transpose is **mandatory** — confirmed HW
  (`wgmma.py:147`), not an assumption. ✓
- **Q2 (masking):** head+tail+empty-group masking is **already implemented** in the vendored kernel
  (`_transposed_ragged_dot_f8.py:101-144`); inherit it. ✓
- **Q3 (occupancy):** `(g,ki,nj)` tiling saturates 132 SMs (G·K·N gives 2048→11264 tiles at 128²);
  split-K is a **post-profile contingency**, not an M2 gate. ✓
- **Q4 (transpose cost):** unknown until measured — **M2 adds a transpose-only row**; do not assume M3 is
  optional. ⚠ measure.
- **Q5 (dynamic grid):** `emit_pipeline` data-dependent grid is supported (vendored kernel compiled);
  padding-to-fixed is the fallback. ⚠ re-confirm in M1.

## 8. Out of scope

e5m2 grads for mosaic; Blackwell/sm100; fine-grained scaling; the forward K-major weight-store
(~+0.2× fwd) lever.
