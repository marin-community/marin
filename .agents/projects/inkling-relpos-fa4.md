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

## Constraints / risks
- Cute kernels are GPU-only: every kernel test is an iris GPU job (minutes/iter). The
  reference oracle (stage 1) is the CPU-side ground truth to diff against.
- fwd kernel currently hardcodes `score_mod=None`; bwd has partial score_mod plumbing.
- head_dim stays 128 (fa4 fast path) — the bias adds no head channels.
- rel_extent (512/1024) is set independent of the sliding window (2048).
