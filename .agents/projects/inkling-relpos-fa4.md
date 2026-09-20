# Inkling relative-position embedding in fast_track (fused fa4_cute)

Issue: TBD (fast_track experiment). Branch: `fast_track_inkling_relpos`.

## Goal
Replace RoPE in fast_track attention with Inkling's content-dependent relative-position
bias, added to the pre-softmax logits, implemented **efficiently** (fused into fa4_cute,
no S² materialization) and MFU-checked. Then run at d512/d768 in two variants:
MLA on all layers, and GQA on all layers (baselines already exist with RoPE).

## The Inkling mechanism (confirmed from HF transformers `modeling_inkling.py`)
- `r_proj: hidden -> num_heads * d_rel` gives a per-token, per-head relative feature `R`
  (`d_rel=16`). RMSNorm on q/k before attention (fast_track already does this).
- Learned bank `proj` shape `[d_rel, rel_extent]` — **shared across heads** (no head axis).
- `rel_logits = relative_states @ proj` -> `[B, H, S, rel_extent]`, then **gather by integer
  distance `q_pos - k_pos`**, add to the score. `rel_extent` = 512 (local) / 1024 (global);
  beyond the extent there is no explicit relative term.

So: `bias[b,h,i,j] = R[b,i,h,:] · proj[:, i-j]` for `0 <= i-j < rel_extent`, else 0.

## Design: pass precomputed A = R@proj to the kernel (minimal kernel surface)
- JAX computes `A[b,h,i,d] = einsum(R, proj)` -> `[B,H,S,rel_extent]`. Autodiff handles
  dR, dproj from dA, so the kernel never differentiates r_proj/proj.
- Kernel FWD: when scoring (i,j), if `0 <= i-j < L` add `A[b,h,i, i-j]`.
- Kernel BWD: accumulate `dA[b,h,i, i-j] += dS[i,j]`. Each dA entry is written by exactly
  one (q-tile,k-tile) pair (fixed i, unique j=i-d), so no atomics across tiles.

## Stages
1. **JAX foundation + reference oracle (CPU-testable) — DO FIRST**
   - `InklingRelPos` eqx.Module in model.py: `r_proj`, `proj` bank, `rel_extent`, `d_rel`.
     Produces `A` in `[B,H,S,L]` (q layout aware).
   - Extend `attention()` / `reference_attention` in `_core.py` with optional per-head
     `rel_bias` (compact A + rel_extent); reference expands by gather and adds to scores
     under the existing mask.
   - Wire into `CausalSelfAttention`: gate `use_inkling_relpos`; when on, skip RoPE, add the
     bias; per-layer rel_extent (local vs global). Keep XSA, attn-gate, sconv, latent attn.
   - Config flags: `inkling_relpos: bool`, `rel_dim` (=16), `rel_extent_local`,
     `rel_extent_global`.
   - CPU test: forward+grad finite; reference bias matches hand-computed small case;
     gather/masking correct with causal + sliding window + segments.
2. **fa4_cute FWD gather-add** (GPU). Thread `A` + `rel_extent` through the JAX wrapper
   (`gpu_fa4_cute_attention`), backend custom_vjp (nondiff rel_extent), spec builders (add
   input buffer), and the cute forward kernel (SMEM-stage the A tile for the m-block, add by
   distance in the score loop). Test fwd output vs reference on GPU.
3. **fa4_cute BWD scatter** (GPU). Produce `dA`; add to custom_vjp bwd outputs; cute bwd
   accumulates dS into dA by distance. Gradient-check vs reference on GPU.
4. **MFU probe** at d512 (`--num-steps 20 --no-eval`); compare to the RoPE baseline MFU.
   Iterate on tiling/SMEM if the hit is material. This is the gate before real runs.
5. **Runs**: d512 + d768, variants {MLA all layers, GQA all layers}. Compare to existing
   MLA / GQA RoPE baselines. Report Paloma macro loss + MFU.

## Kernel-edit notes (precise, from reading the code — for the GPU stage)

Layout: `cutlass_call` reorders JAX axes via `TensorSpec(mode=...)`. q [B,S,Hq,D] uses
`mode=(1,3,2,0)` → kernel `mQ` is [S,D,Hq,B], indexed `mQ[None,None,q_head,batch_idx]`. So pass the
compact bias A as JAX `[B,Hq,S,L]` with `rel_bias_spec = TensorSpec(mode=(2,3,1,0), static=True)` →
kernel `mRelBias` is `[S, L, Hq, B]`, scalar-indexed `mRelBias[query_idx, delta, q_head, batch_idx]`.

Forward (`_fa4_cute_kernels.py`):
- `_cutlass_attention_forward_specs`: append `rel_bias_spec` to the input tuple (after the two
  metadata specs). `segmented_flash_attention_forward(..., rel_bias)`: pass as last input to `call(...)`.
- Launcher `__call__`/`kernel`/`_launch_...forward`: add `mRelBias` param in input position (after
  `mValid`, before `mO`); thread into `basic_params`.
- `softmax_rescale_O` inner (r,c) loop already computes `query_idx` and `key_idx`. Add, before the
  mask sets -inf: `delta = query_idx - key_idx; if 0 <= delta < L: acc_S_mn[r,c] += A/softmax_scale`
  where `A = mRelBias[query_idx, delta, q_head, batch_idx]`. **Divide by softmax_scale** because the
  kernel multiplies `acc_S` by `softmax_scale_log2` inside `exp2` (so acc_S is in raw-qk units; adding
  A/scale makes the exponent `scale*qk + A`, matching the reference and the dA helper). L = mRelBias.shape[1].

Backward: FlashAttentionBackwardSm90/Sm80/Sm120 recompute S from Q,K,lse. Add the SAME
`A/softmax_scale` term where they form the score tile (mirror the fwd add; read-only). Thread
`mRelBias` through `_launch_segmented_flash_attention_backward*` and the bwd specs. dQ/dK/dV then correct.
dA does NOT come from the kernel — the custom_vjp computes it via `_inkling_relpos.rel_bias_backward`
(verified). So the bwd kernel needs only the read, no scatter/extra output.

custom_vjp (`_fa4_cute_backend.py`): add `rel_bias` as a differentiable arg (NOT in nondiff_argnums,
which must shift to (6,7)). fwd saves `rel_bias` in residuals; bwd returns
`(dq, dk, dv, drel_bias, None, None)` where `drel_bias = rel_bias_backward(q,k,v,out,cot,lse,rel_bias,
lower_bounds,valid, softmax_scale=...)`. Wrapper `gpu_fa4_cute_attention(...rel_bias=A)`: transpose the
model's A `[B,Hq,S,L]` to whatever `_fa4_cute_attention_forward_sharded`'s shard_map expects (mirror
q's [B,S,Hq,D] handling; A's seq axis must stay unsharded like q's), drop the NotImplementedError.

GPU validation order: (1) fwd vs reference (`attention(...,implementation="reference",rel_bias=A)`),
(2) grads vs reference `jax.grad`, (3) d512 MFU probe `--num-steps 20 --no-eval`.

## Constraints / risks
- Cute kernels are GPU-only: every kernel test is an iris GPU job (minutes/iter). The
  reference oracle (stage 1) is the CPU-side ground truth to diff against.
- fwd kernel currently hardcodes `score_mod=None`; bwd has partial score_mod plumbing.
- head_dim stays 128 (fa4 fast path) — the bias adds no head channels.
- rel_extent (512/1024) is set independent of the sliding window (2048).
