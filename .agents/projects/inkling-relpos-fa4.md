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

## STATUS UPDATE (GPU)
- Root cause of all "no-traceback crashes": OOM at 1GB default RAM. Bare iris kernel jobs need
  `--gpu H100x1 --extra pipeline --cpu 8 --memory 96GB --disk 200GB`.
- FORWARD fused bias kernel VALIDATED vs reference oracle on H100: all cases PASS
  (max_abs 1.5-1.8e-2, rel ~6e-3), incl. sliding window & multiple L. Design simplified to a single
  always-real-bias path (non-Inkling callers pass zero [B,Hq,S,1]); no Constexpr/dummy.
- NEXT: backward. H100 uses the sm90 native bwd path (qhead_per_kvhead>1, head_dim=128 ->
  segmented_flash_attention_backward_sm90_native / FlashAttentionBackwardSm90). Add the bias to its
  S-recompute via the existing score_mod + aux_tensors hook (aux = lower_bounds, valid, + bias);
  score_mod adds bias[b,h,q, clamp(q-k)] for 0<=q-k<L; score_mod_bwd = identity (dA is computed
  separately by rel_bias_backward in the custom_vjp, not the kernel). Then wire the custom_vjp
  (rel_bias as differentiable arg; bwd returns dq,dk,dv,dA) and the wrapper.

---

## sm90-native backward fork — validated design (2026-09-20)

### Why (data)
d512 H100 b28, zero-dA isolation probes:
- RoPE (sm90-native bwd): **8.04% MFU** (2.30M tps)
- Inkling fwd + fused segmented dq/dk/dv, **no dA**: 6.49% (rel_extent=1024), 7.13% (rel_extent=256)
- Inkling full correct (blocked dA): 5.25%

Two independent costs:
1. **Fixed segmented-vs-sm90 overhead** ~1.55pt (extrapolated ~7.38% floor even at zero band/no-dA). The rel_bias forces the bwd OFF the external Hopper WGMMA kernel onto our Ampere segmented kernel.
2. **dA** ~1.24pt (a standalone banded attention-sized recompute).
Neither alone reaches the <10%-impact target (>7.24% MFU). Must fix BOTH.

### Key facts discovered
- The external `flash_attn.cute.flash_bwd_sm90.FlashAttentionBackwardSm90` already exposes FlexAttention-style hooks: `score_mod`, `score_mod_bwd`, `has_aux_tensors`, `aux_data=AuxData(tensors=(...))`. The grug sm90 launcher currently passes `score_mod=None`. **No source fork needed** — inject via callbacks.
- Wheel `flash_attn_4-4.0.0b28-py3-none-any.whl` is pure-python; source extracted at `../fa4wheel/extracted/flash_attn/cute/` for reference (softmax.py apply_score_mod_inner/bwd_inner, flash_bwd_sm90.py apply_score_mod/apply_score_mod_bwd, utils.py scalar_to_ssa/ssa_to_scalar/AuxData/compute_softmax_scale_log2).
- **score_mod signature**: `score_mod(score_ssa, batch_idx, head_idx, q_idx=?, kv_idx=?, seqlen_info=?, aux_tensors=?)` → modified score_ssa. All are size-`vec_size` SSA vectors; batch/head are broadcast-constant per tile. `call_score_mod` passes them as kwargs.
- **score convention with score_mod present**: `compute_softmax_scale_log2` returns `(LOG2E, softmax_scale)` — softmax_scale is applied to raw qk BEFORE score_mod, and score_mod sees NATURAL-unit scores. So an additive logit bias A is added directly (no /softmax_scale conversion, unlike the Ampere fwd). exp2(x*LOG2E - lse).
- **Additive-bias backward is identity for dq/dk/dv**: S=qk+A ⇒ d/d(qk)=1 ⇒ score_mod_bwd returns grad unchanged.
- **dA is free**: dA[b,h,i,delta] = dS[b,h,i,i-delta] (in band). The kernel computes `grad_val = P*(dP - dPsum)` = dS right before calling score_mod_bwd. Each (q,kv)→(q,delta) is unique ⇒ score_mod_bwd can SCATTER grad_val into a dA output aux tensor with no races, and return grad unchanged.

### Implementation steps
**STEP 1 (fast dq/dk/dv, dA still standalone):**
- Dispatch (`_fa4_cute_backend.py` ~L174): allow sm90-native path when rel_bias is not None (drop the `rel_bias is None` guard; still require sm90_backward/GQA/d128).
- `segmented_flash_attention_backward_sm90_native`: add rel_bias as an operand + input spec (mode like fwd: A[B,Hq,S,L] -> kernel [S,L,Hq,B]).
- `segmented_flash_attention_backward_sm90_launcher`: build an Inkling `score_mod` (adds A gathered by (q_meta, delta=q-kv), gated 0<=delta<rel_extent, using the fwd scalar idiom per-lane over vec_size) and pass `score_mod=_inkling_score_mod, score_mod_bwd=_identity, has_aux_tensors=True, aux_data.tensors=(lower_bounds, valid, rel_bias)`. Keep `_grug_segment_mask_mod` for the causal/segment mask (uses aux[0],aux[1]).
- Keep dA via existing standalone `rel_bias_backward`. Measure MFU (expect ~6.8-7%).

**STEP 2 (free dA): eliminate standalone pass**
- Add a dA output tensor to the sm90 launcher's cutlass_call output specs (zero-filled).
- score_mod_bwd scatters grad_val into dA[b,h,q,delta] per-lane (in band); returns grad unchanged.
- Drop the standalone rel_bias_backward when on sm90 path. Measure (target <10% impact = >7.24%).

### Risks
- CuTe DSL per-lane aux gather/scatter inside a @cute.jit score_mod (vec_size lanes): extract `q_idx_ssa[j]`, build result via rmem tensor + `.load()`. Idiom mirrors apply_score_mod_inner's per-lane index extraction; validate with a GPU grad probe FIRST (correctness before MFU).
- fwd (Ampere, materialized A) and bwd (score_mod reading same A) must add identical A. Both use same aux A, same band gate ⇒ P matches.

## STEP 1 debugging (2026-09-20) — sm90 score_mod NaN root cause
Bisected on GPU (grad probe vs reference, B2 S256 Hq4 Hkv1 D128 L128, FUSED_BWD=1):
- NO-BIAS sm90 path: PASS (dispatch/plumbing correct; base Hopper bwd unchanged).
- score_mod=identity: dq/dk/dv FINITE (score_mod presence + LOG2E scale switch OK).
- score_mod=const (per-lane loop, no gmem read): FINITE (rmem-loop / if-in-band / tensor-write OK).
- score_mod=head0 (real read, head hardcoded 0): FINITE (read mechanism OK; in-range).
- => NaN was an **OOB gmem read**: head_idx[j] out of range.
ROOT CAUSE: `apply_score_mod_inner`/`floor_if_packed` (flash_attn.cute b28 softmax.py) UNCONDITIONALLY
assume Pack-GQA when qhead_per_kvhead>1: they hand score_mod `q_idx = realQ // rep` and
`head_idx = trueQhead*rep + (realQ % rep)`. But `segmented_flash_attention_backward_sm90` tiles over
real q-heads (head_idx_kv = head_idx//rep; mQ indexed by head_idx directly) and is NON-packed, so the
index tensor holds real Q. The packed head overshoots Hq -> OOB -> NaN.
FIX (in _grug_inkling_score_mod): invert the packing with rep=qhead_per_kvhead (build-time const):
  q_head = head_idx[j] // rep ;  q_pos = q_idx[j]*rep + (head_idx[j] % rep) ;  delta = q_pos - kv_idx[j]
  bias = rel_bias_t[q_pos, delta_safe, q_head, b0]   # kernel layout [S,L,Hq,B]
Debug modes retained behind FAST_TRACK_INKLING_SM_DEBUG=identity|const|head0 (remove before final).
dA (standalone rel_bias_backward) PASSES throughout (rel ~5e-3).

## STATUS 2026-09-20 (fully-fused free-dA + scale finding)
MFU (d512, fully-fused sm90 vendored bwd, in-kernel free dA):
- rel_extent=1024: 6.03% (25% impact vs RoPE 8.04)
- rel_extent=512: 6.55% (18.5%)
- no-dA ceiling 1024: 6.75%
STEP 2 (in-kernel dA scatter) done + grad-verified: 5.39 -> 6.03 (scatter costs ~0.72pt vs standalone 1.36pt).
d1280 (rel_extent 512): ink ~11.5% vs ROPE ~15.7% => ~27% impact -- IMPACT GROWS WITH SCALE
(A/dA [B,Hq,S,L] bandwidth scales with heads). => on-the-fly low-rank is essential, more so at scale.

## NEXT: on-the-fly low-rank bias (eliminate [B,Hq,S,L] materialization)
Model passes R[B,S,H,rel_dim=16] + proj[16,L] instead of materialized A. Kernel computes
A = R.proj on the fly in fwd score-add and bwd S-recompute (16-mul/elem, proj in smem; R reused per row).
Hybrid (simpler, gets the ~1.3pt read gap): keep dA[B,H,S,L] scatter output, compute dR=einsum(dA,proj)
& dproj=einsum(dA,R) in the custom_vjp (cheap). Full (also kills dA traffic): kernel emits dR (16-wide
scatter, tiny) + dproj (smem partial + atomic reduce) -- harder. Do hybrid first, measure, then full.

## On-the-fly low-rank implementation plan (ready to execute)
Goal: stop materializing/streaming A[B,Hq,S,L]; compute A = sum_r R[q,h,r]*proj[r,delta] in-kernel.
Wins the ~1.3pt fwd+bwd read gap + the model-side A write; grows with scale (d1280 impact 27%).

Layouts: mR kernel [S, rel_dim, Hq, B] (spec mode (2,3,1,0) on R[B,S,H,rel_dim] -> actually R is [B,S,H,r];
pick spec so kernel indexes mR[q, r, head, batch]); mProj [rel_dim, L] shared (no batch/head).

Edits:
1. Forward kernel (_fa4_cute_kernels segmented_flash_attention_forward_launcher): replace mRelBias operand
   with mR + mProj through __call__(256)/kernel(385)/mma(544)/_launch(952); at the bias-add site (~832):
   rel_extent=mProj.shape[1]; rel_dim=mProj.shape[0]; acc=0; for r in range(rel_dim): acc += mR[q_meta,r,q_head,b].to(f32)*mProj[r,delta_safe].to(f32); add acc*LOG2E/softmax_scale_log2 when in_band.
2. Backward vendored apply_score_mod (_fa4_cute_flash_bwd_sm90): read R (aux) + proj (aux) instead of
   aux[2] A; compute A on the fly (same loop); keep dA scatter unchanged (aux[3]).
3. Backend: forward specs + segmented_flash_attention_forward take (R, proj). sm90 native backward passes
   R, proj as aux[2],aux[4]... (reindex: aux=(lb,valid,R,dA,proj)) OR aux=(lb,valid,proj,dA) with R as a
   separate operand. Simplest: aux tensors (lb, valid, R, proj, dA); apply_score_mod reads aux[2]=R,aux[3]=proj;
   apply_score_mod_bwd writes aux[4]=dA.
4. custom_vjp: differentiable inputs become (q,k,v,R,proj) [drop A]. fwd saves R,proj. bwd: kernel returns
   dA (scatter); compute dR=einsum("bhql,rl->bqhr" ... dA,proj) and dproj=einsum(dA,R) OUTSIDE (cheap);
   return dq,dk,dv,dR,dproj. (Hybrid: dA still materialized as kernel output + read once by these einsums.)
   FULL variant (later): kernel emits dR (16-wide scatter) + dproj (smem partial + atomic) -> no dA traffic.
5. Model InklingRelPos/CausalSelfAttention: compute R=einsum(x,r_proj)->[B,S,H,rel_dim]; pass (R, proj) to
   attention instead of A. reference_attention: materialize A=einsum(R,proj) for the oracle.
6. Tests: standalone forward on-the-fly vs reference; grad check (dq/dk/dv/dR/dproj) vs reference; MFU d512+d1280.
Risk: ~6 files/2 kernels, several GPU iterations. Hybrid keeps dA traffic (gets ~1.3pt); FULL removes it (needs reduction).

## Milestone 1 (on-the-fly A read, hybrid dA) DONE + grad-verified (2026-09-20)
Grad check (d512 GQA, vs reference): NO-BIAS + WITH-BIAS all PASS incl dR (3.5e-3) and dproj (3.1e-3).
Interface changed: attention() takes rel_r[B,S,Hq,rel_dim]+rel_proj[rel_dim,L]; custom_vjp returns
dR,dproj. Kernels compute A=R.proj on the fly (fwd + bwd S-recompute). Model no longer materializes A.
Reference/TPU/MLA-fallback materialize A internally. dA still scattered in-kernel; VJP maps dA->dR,dproj.
MFU pending (d512 1024/512, d1280 512) vs materialized (6.03/6.55/11.34, rope d1280 15.44).
Milestone 2 (optional): emit dR (16-wide) + dproj (reduction) in-kernel to also kill dA[B,Hq,S,L] traffic.

## On-the-fly low-rank: FAILED, REVERTED (2026-09-20)
Milestone 1 was grad-correct (dq/dk/dv/dR/dproj all PASS) but MFU CRATERED:
- d512 rel_extent=1024: 2.48% (vs materialized 6.03)   -- 2.4x SLOWER
- d512 rel_extent=512:  2.48% (vs 6.55)
- d1280 rel_extent=512: 4.22% (vs 11.34)
Root cause: computing A=sum_r R[q,r]*proj[r,delta] per score element (16 FMAs + 16 R-reads +
16 proj-reads, in BOTH fwd and bwd scalar loops) is ~2.5x costlier than ONE coalesced load of the
materialized A. The materialized A read was already cheap (cached/coalesced); the model-side A einsum
is a well-optimized matmul. The bandwidth premise was wrong -- these kernels are not A-bandwidth-bound
in the way assumed; the per-element scalar 16-dim dot dominates. Reverted to the materialized kernel
(HEAD after 19cce3c0fc). BEST STATE = materialized fully-fused free-dA: d512 6.03(1024)/6.55(512),
d1280 11.34 vs rope 15.44. <10% MFU impact is NOT reachable with these approaches.
Remaining micro-lever: bf16 dA output (~0.3pt). Otherwise ship materialized + run the science.

## bf16 dA: banked (2026-09-20) -- FINAL optimized materialized kernel
Grad still PASS (dA rel 5.3e-3). MFU:
- d512 rel_extent=1024: 6.03 -> 6.33 (21% impact)
- d512 rel_extent=512:  6.55 -> 6.96 (13.4% impact)  <- near target
- d1280 rel_extent=512: 11.34 -> 12.75 (27% -> 17.4% impact)   [rope d1280 15.44]
bf16 dA helps MORE at scale (dA[B,Hq,S,L] traffic scales with heads): +1.41pt at d1280.
Campaign total: correct fused Inkling backward 5.25 -> 6.96 (d512-512); d1280 impact 27% -> 17.4%.

Kernel is at its practical floor for the materialized approach. Remaining fundamental cost = the A
read (fwd + bwd, ~1.3pt) inherent to a per-position bias. Micro-opts (acc_S_pre skip, scale fold) are
<0.1pt (compute, not memory; model is memory-bound) -> not worth the vendored-kernel risk.
On-the-fly per-element FAILED (2.4x slower). The ONE remaining big lever: precompute the A TILE via a
small tensor-core GEMM into smem per (m_block,n_block) (R_tile[128,16] @ proj_slice[16,~192]) in both
fwd and bwd prologues -- amortizes the 16-dim dot via tensor cores + smem reads (unlike the failed
scalar per-element on-the-fly), removing the global A HBM materialization+reads. High effort/risk;
needs the R+proj interface (reverted) back + smem staging. Deferred.
