# Stage 2 — CuTe FA4 relative-position-bias kernel (#7426)

Stage 1 (reference path + model mechanism) is done, validated, committed on `grug/moe-abl/rel-pos-bias`,
and the de-risk (seq-1024 reference run) confirmed the mechanism trains. Stage 2 extends the CUTLASS
**Python CuTe DSL** FA4 kernel to consume the compact bias band so the real d1024/seq-4096 run works.

## Interface (from Stage 1)
Model passes `bias_band: [B, S, num_heads, window]`, compute dtype. Convention: `bias_band[b,i,h,r]` is
the additive **scaled-logit** bias between query `i` and key `i-r` (r=0=self). Applies only for
`0 <= i-j < window` and `j <= i` (causal). Zero-init `rel_pos_b` ⇒ all-zero band ⇒ kernel is a no-op.

## Forward (`_fa4_cute_kernels.py`, the mask+softmax col loop ~L817-839)
The loop already computes `query_idx` (row), `key_idx` (col), `query_meta_idx`, and the valid/causal
predicate. Inject the bias there, before the row `exp2` (L850):
- `offset = query_idx - key_idx`; when `in_window = (0 <= offset) and (offset < window)` **and** the
  existing valid/causal predicate holds: `acc_S_mn[r,c] += mBias[batch, query_meta_idx, q_head, offset] * inv_softmax_scale`.
- **Scaling subtlety:** `acc_S` is raw QK; the kernel multiplies by `softmax_scale` inside `exp2`
  (`acc_S * softmax_scale_log2`). The reference biases the *scaled* logits, so add
  `bias * (1/softmax_scale)` to `acc_S` (equivalently pass `inv_softmax_scale = 1/softmax_scale`).
- v1: index `mBias` per (r,c) directly (per-element GMEM load — slow but correct, for validation).
  v2 opt: per N-tile, preload the ~`n_block_size` band values for each query row into registers/SMEM.

## Plumbing
- Kernel launcher signature (L247+): add `mBias: cute.Tensor` operand + pass `inv_softmax_scale` (or
  reuse softmax_scale and reciprocate in-kernel).
- `_fa4_cute_backend.py::segmented_flash_attention_forward`: add `bias` arg, add to `input_spec`
  (`_cutlass_attention_forward_specs`), thread into `cutlass_call(...)` and `call(q,k,v,lb,valid,bias)`.
- `attention()` dispatch (`_core.py`): currently raises NotImplementedError for bias on non-reference;
  route `bias` (as the band, not the dense `[B,H,Q,K]`) to the fa4 forward when
  `implementation == gpu_fa4_cute`. Model already builds `bias_band`; stop densifying for the fa4 path.

## Backward (harder half)
`segmented_flash_attention_backward*` computes dQ/dK/dV. Need an extra output `d_bias_band [B,S,H,window]`.
Since the bias adds 1:1 to the scaled score, `d_bias_band[b,i,h,r] = dScaledScore[b,h,i,i-r]` = the
softmax-input gradient at the banded position (×`inv_softmax_scale` to undo the scale, mirroring fwd).
Flash bwd forms `dS = P ∘ (dP - rowsum(dP∘P))` per tile; scatter that into `d_bias_band` at the banded
cols. The model's autodiff then gets `dA`/`dB` from `d_bias_band` via the `einsum` VJP (kernel need not
touch A/B). Wire `d_bias_band` through the custom_vjp so JAX composes it.

## Validation (incremental, GPU)
1. **Forward-only**: tiny d1024 shape on 4-GPU (or a unit harness), compare fa4-with-bias output vs the
   Stage-1 reference (`attention_implementation=reference`) on the same q/k/v + a random nonzero band.
   Assert allclose. (Bias=0 must match the un-biased fa4 exactly.)
2. **Backward**: `jax.grad` through both paths; compare `d_bias_band` (and dQ/dK/dV) fa4 vs reference.
3. Then the full d1024/seq-4096 ablation run (`SCALE_REL_POS_BIAS=1`, `SCALE_ATTN_IMPL=gpu_fa4_cute`).

## Risks
- CuTe SMEM/register pressure from the band load (mitigate with the per-N-tile preload).
- The d6144/64-GPU CUBIN/#7407 issue is NOT in scope (d1024/4-GPU is the target and is the reliable regime).
- Each kernel edit needs a GPU compile/run to verify — this is multi-step, not one-shot.
