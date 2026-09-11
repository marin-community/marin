# Native Attention-Style MIMO XLA

## Problem

The new attention-style MIMO API in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L453) is currently an adapter over the older chunked ranked kernel, not a native implementation.

Today it pays for:

- Q/K regrouping and chunk packing in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L509)
- V/Z chunk packing in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L517)
- rank-weight reshaping and rank expansion in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L533)
- output repacking and `final_state` / `final_k` packaging in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L582)

The underlying XLA kernel in [xla.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/xla.py#L144) still expects chunked ranked tensors and still builds the TPU-hostile local block:

```python
bc = jnp.einsum("gktnv,gksnu->gktsuv", ...)
local_output = jnp.einsum("gkts,gktsuv,gkusp->gkvtp", ...)
```

That mismatch is why the new public API is slower than `chunked_public` even though the kernel math is unchanged. On the current `v5p-8` slice measurements, the attention-style wrapper costs roughly `14-17%` forward throughput and `4-12%` backward throughput.

## Goals

- Make `mamba3_mimo_attentionish_forward(...)` the native XLA path for real-valued MIMO.
- Preserve the current structured decomposition:
  - local block
  - chunk summary
  - `lax.scan`
  - prefix emit
  - diagonal correction / gate / collapse
- Remove the full adapter tax from [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L509).
- Keep `R` out of the two minor-most dimensions outside the big fused contractions.
- Keep SISO and the existing chunked MIMO API working throughout.

Non-goals:

- Pallas
- complex / RoPE beyond the current real-valued subset
- a custom VJP in the first pass
- deleting the current chunked MIMO API

## Proposed Solution

Add a native XLA attention-style forward core that accepts the public attention-ish signature directly and performs the chunked SSD-style algorithm internally, instead of converting to the legacy chunked ranked API first.

Core shape choice:

- external API stays attention-like:
  - `q, k: [B, S, R, G, N]`
  - `v, z: [B, S, H, P]`
  - `mimo_v / mimo_z / mimo_o: [H, R, P]`
- internal per-head flattened batch:
  - `U = B * H`
- canonical layouts outside big matmuls:
  - `q_rank / k_rank: [U, K, R, C, N]`
  - `psi_v_rank / o_rank: [U, K, R, C, P]`
  - state carry: `[U, N, P]`
- fused views only for big contractions:
  - `q_f / k_f: [U, K, C*R, N]`
  - `psi_v_f: [U, K, C*R, P]`

Minimal native core:

```python
def mamba3_mimo_attentionish_forward_xla_batched(...):
    q_rank, k_rank, psi_v_rank, trap_scale, gamma = _prepare_attentionish_inputs(...)
    q_f = _fuse_chunk_rank(q_rank)
    k_f = _fuse_chunk_rank(k_rank)
    psi_v_f = _fuse_chunk_rank(psi_v_rank)

    chunk_summary = jnp.einsum("ukxn,ukxp->uknp", k_state_f, psi_v_f, preferred_element_type=jnp.float32)
    incoming_state, final_state = _scan_chunk_states(chunk_decay_last, chunk_summary)
    inter_rank = _emit_prefix(q_f, incoming_state, exp_da)
    offdiag_rank = _emit_strict_offdiag(q_f, k_f, psi_v_f, segsum)
    diag_rank = _emit_same_step(q_rank, k_rank, psi_v_rank, gamma, d)
    return _gate_and_reduce_rank_major(inter_rank + offdiag_rank + diag_rank, z, mimo_z, mimo_o)
```

This keeps the current decomposition but changes where the layout boundary lives:

- today:
  - attention-ish API -> adapter -> old chunked ranked kernel
- proposed:
  - attention-ish API -> native attention-ish XLA core

## Implementation Outline

1. Add a native XLA attention-style core, likely in a new module such as `mamba3/xla_attentionish.py`, with direct `q/k/v` inputs and `[N, P]` carry.
2. Move the current adapter logic in [api.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/api.py#L509) into private input-prep helpers used by the new core only where unavoidable.
3. Replace the local block from [xla.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/xla.py#L157) with a fused `C*R` formulation that avoids materializing `[C, C, R, R]`.
4. Keep same-step diagonal interaction and rank collapse outside the big fused contractions, with rank-major `[R, C, P]` cleanup layouts.
5. Preserve the existing `mamba3_mimo_chunked_forward(...)` path as a fallback and parity oracle while the new attention-style path lands.
6. Benchmark the native attention-style path against `chunked_public` on the three canonical TPU shapes and profile wrapper-vs-kernel time again.

## Notes

- The current chunk summary in [xla.py](/Users/dlwh/.codex/worktrees/03f9/marin/lib/levanter/src/levanter/kernels/pallas/mamba3/xla.py#L135) already has the right structure; the main redesign is the local block and the public entry boundary.
- The carry should move from the old `[P, N]` convention to `[N, P]` for the native attention-style path. That matches the upstream Tilelang kernel more naturally and makes prefix emit a direct `Q @ state`.
- The diagonal same-step path should stay separate. Upstream also handles the `R x R` same-step interaction outside the large fused causal matmul.
- The first native version should still use pure JAX/XLA and `lax.scan`. The failed `associative_scan` experiment is still a hard no.
- Backward can rely on ordinary JAX differentiation initially. If the native path wins materially but backward residuals get too large, a recompute-heavy custom VJP can be the next optimization.
- Backwards compatibility is straightforward:
  - keep `mamba3_mimo_chunked_forward(...)`
  - keep the public attention-style API
  - change only which implementation that public API dispatches to

## Future Work

- Add rotary / nonzero `angles` natively instead of only permitting `None` or zero.
- Add `segsum` as a real input path instead of deriving only from the local Mamba schedules.
- Add a custom VJP once the forward structure is stable and benchmarked.
- Revisit a Pallas local kernel only if the native attention-style XLA path still leaves `local_output` dominant.
