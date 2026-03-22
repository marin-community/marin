# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal, TypeAlias, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .config import (
    BlockSizes,
    HybridModeConfig,
    Mamba3Mode,
    mamba3_tpu_default_chunk_size,
)
from .reference import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    mamba3_chunk_state_reference_batched,
    mamba3_mimo_apply_gate_and_collapse_chunked,
    mamba3_mimo_chunked_forward_ranked_reference_batched,
    mamba3_mimo_chunked_forward_reference_batched,
    mamba3_mimo_direct_recurrence_reference_batched,
    mamba3_mimo_rank_collapse_chunked,
    mamba3_mimo_rank_expand_chunked,
    mamba3_chunked_forward_reference_batched,
    mamba3_chunked_sequential_reference_batched,
    mamba3_direct_recurrence_reference_batched,
    mamba3_intra_chunk_reference_batched,
    prepare_mamba3_chunked_scales,
    prepare_mamba3_scales,
)
from .xla import (
    mamba3_chunk_state_xla_batched,
    mamba3_chunked_forward_xla_batched,
    mamba3_intra_chunk_xla_batched,
    mamba3_mimo_chunked_forward_ranked_xla_batched,
    mamba3_mimo_chunked_forward_xla_batched,
)


Implementation: TypeAlias = Literal["xla", "reference"]


IMPLEMENTATIONS = {
    "reference": mamba3_intra_chunk_reference_batched,
    "xla": mamba3_intra_chunk_xla_batched,
}
_DEFAULT_IMPLEMENTATIONS: tuple[Implementation, ...] = ("xla",)


def _flatten_intra_chunk_inputs(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    out_correction: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[int, ...]]:
    if a_log_cumsum.ndim < 1:
        raise ValueError(f"`a_log_cumsum` must be at least rank-1, got {a_log_cumsum.shape}.")
    if src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` and `out_correction` must match `a_log_cumsum`.")
    if b.ndim < 2 or c.ndim < 2 or x.ndim < 2:
        raise ValueError("`b`, `c`, and `x` must be at least rank-2.")

    leading_shape = a_log_cumsum.shape[:-1]
    chunk_size = a_log_cumsum.shape[-1]
    if b.shape[:-2] != leading_shape or c.shape[:-2] != leading_shape or x.shape[:-2] != leading_shape:
        raise ValueError("All inputs must share the same leading batch/group axes.")
    if b.shape[-2] != chunk_size or c.shape[-2] != chunk_size or x.shape[-2] != chunk_size:
        raise ValueError("All inputs must share the same chunk axis.")

    groups = math.prod(leading_shape) if leading_shape else 1
    return (
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        out_correction.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        c.reshape(groups, chunk_size, c.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    ), leading_shape


def mamba3_intra_chunk(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    out_correction: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    c: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> Float[Array, "... chunk value"]:
    """Dispatch the quadratic intra-chunk Mamba-3 block to the requested backend."""

    flat_inputs, leading_shape = _flatten_intra_chunk_inputs(a_log_cumsum, src_scale, out_correction, b, c, x)
    if implementation is None:
        impls: Sequence[Implementation] = _DEFAULT_IMPLEMENTATIONS
    elif isinstance(implementation, Sequence) and not isinstance(implementation, (str, bytes)):
        impls = cast(Sequence[Implementation], implementation)
    else:
        impls = (cast(Implementation, implementation),)

    for impl in impls:
        fn = IMPLEMENTATIONS.get(impl)
        if fn is None:
            raise ValueError(f"Unsupported Mamba-3 implementation: {impl}.")
        y = fn(*flat_inputs)
        return y.reshape(leading_shape + y.shape[-2:])

    raise ValueError("No Mamba-3 implementation was provided.")


def mamba3_chunk_state(
    a_log_cumsum: Float[Array, "... chunk"],
    src_scale: Float[Array, "... chunk"],
    b: Float[Array, "... chunk state"],
    x: Float[Array, "... chunk value"],
) -> Float[Array, "... value state"]:
    """Compute chunk-end transformed state accumulation."""

    if a_log_cumsum.ndim < 1:
        raise ValueError(f"`a_log_cumsum` must be at least rank-1, got {a_log_cumsum.shape}.")
    leading_shape = a_log_cumsum.shape[:-1]
    chunk_size = a_log_cumsum.shape[-1]
    if src_scale.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` must match `a_log_cumsum`.")
    if b.shape[:-2] != leading_shape or x.shape[:-2] != leading_shape:
        raise ValueError("`b` and `x` must share the same leading axes as `a_log_cumsum`.")
    if b.shape[-2] != chunk_size or x.shape[-2] != chunk_size:
        raise ValueError("`b` and `x` must share the same chunk axis as `a_log_cumsum`.")

    groups = math.prod(leading_shape) if leading_shape else 1
    y = mamba3_chunk_state_xla_batched(
        a_log_cumsum.reshape(groups, chunk_size),
        src_scale.reshape(groups, chunk_size),
        b.reshape(groups, chunk_size, b.shape[-1]),
        x.reshape(groups, chunk_size, x.shape[-1]),
    )
    return y.reshape(leading_shape + y.shape[-2:])


def mamba3_chunked_forward_from_transformed(
    a_log_cumsum: Float[Array, "... chunks chunk"],
    src_scale: Float[Array, "... chunks chunk"],
    out_correction: Float[Array, "... chunks chunk"],
    b: Float[Array, "... chunks chunk state"],
    c: Float[Array, "... chunks chunk state"],
    x: Float[Array, "... chunks chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Chunked Mamba-3 forward pass on transformed inputs."""

    del block_sizes, interpret, backend
    if a_log_cumsum.ndim < 2 or src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("Expected transformed inputs with shape `[..., chunks, chunk]`.")
    leading_shape = a_log_cumsum.shape[:-2]
    groups = math.prod(leading_shape) if leading_shape else 1
    num_chunks, chunk_size = a_log_cumsum.shape[-2:]
    flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
    flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
    flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)
    flat_b = b.reshape(groups, num_chunks, chunk_size, b.shape[-1])
    flat_c = c.reshape(groups, num_chunks, chunk_size, c.shape[-1])
    flat_x = x.reshape(groups, num_chunks, chunk_size, x.shape[-1])

    if implementation not in (None, "xla", "reference"):
        raise ValueError(f"Unsupported Mamba-3 implementation: {implementation}.")

    if implementation == "reference":
        y, final_state = mamba3_chunked_forward_reference_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
        )
    else:
        y, final_state = mamba3_chunked_forward_xla_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
        )

    return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])


def mamba3_chunked_forward(
    dt: Float[Array, "... chunks chunk"],
    lam: Float[Array, "... chunks chunk"],
    a: Float[Array, "... chunks chunk"] | Float[Array, "... chunks"],
    b: Float[Array, "... chunks chunk state"],
    c: Float[Array, "... chunks chunk state"],
    x: Float[Array, "... chunks chunk value"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Stable Mamba-3 entrypoint on native chunked inputs."""

    if implementation == "reference":
        leading_shape = dt.shape[:-2]
        groups = math.prod(leading_shape) if leading_shape else 1
        y, final_state = mamba3_direct_recurrence_reference_batched(
            dt.reshape(groups, dt.shape[-2], dt.shape[-1]),
            lam.reshape(groups, lam.shape[-2], lam.shape[-1]),
            a.reshape(groups, a.shape[-2], a.shape[-1]) if a.shape == dt.shape else a.reshape(groups, a.shape[-1]),
            b.reshape(groups, b.shape[-3], b.shape[-2], b.shape[-1]),
            c.reshape(groups, c.shape[-3], c.shape[-2], c.shape[-1]),
            x.reshape(groups, x.shape[-3], x.shape[-2], x.shape[-1]),
        )
        return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])

    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    return mamba3_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        backend=backend,
    )


def mamba3_mimo_chunked_forward_from_transformed(
    a_log_cumsum: Float[Array, "... chunks chunk"],
    src_scale: Float[Array, "... chunks chunk"],
    out_correction: Float[Array, "... chunks chunk"],
    b: Float[Array, "... chunks chunk state rank"],
    c: Float[Array, "... chunks chunk state rank"],
    x_base: Float[Array, "... chunks chunk value"],
    z_base: Float[Array, "... chunks chunk value"],
    w_x: Float[Array, "... value rank"] | Float[Array, "value rank"],
    w_z: Float[Array, "... value rank"] | Float[Array, "value rank"],
    w_o: Float[Array, "... value rank"] | Float[Array, "value rank"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Chunked real-valued MIMO Mamba-3 forward pass on transformed schedules."""

    del block_sizes, interpret, backend
    if implementation not in (None, "xla", "reference"):
        raise ValueError("MIMO is currently available only for the XLA and reference implementations.")
    if a_log_cumsum.ndim < 2 or src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("Expected transformed inputs with shape `[..., chunks, chunk]`.")

    leading_shape = a_log_cumsum.shape[:-2]
    groups = math.prod(leading_shape) if leading_shape else 1
    num_chunks, chunk_size = a_log_cumsum.shape[-2:]
    flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
    flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
    flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)
    flat_b = b.reshape(groups, num_chunks, chunk_size, b.shape[-2], b.shape[-1])
    flat_c = c.reshape(groups, num_chunks, chunk_size, c.shape[-2], c.shape[-1])
    flat_x = x_base.reshape(groups, num_chunks, chunk_size, x_base.shape[-1])
    flat_z = z_base.reshape(groups, num_chunks, chunk_size, z_base.shape[-1])

    if implementation == "reference":
        y, final_state = mamba3_mimo_chunked_forward_reference_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
            flat_z,
            w_x,
            w_z,
            w_o,
        )
    else:
        y, final_state = mamba3_mimo_chunked_forward_xla_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x,
            flat_z,
            w_x,
            w_z,
            w_o,
        )

    return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])


def mamba3_mimo_chunked_forward(
    dt: Float[Array, "... chunks chunk"],
    lam: Float[Array, "... chunks chunk"],
    a: Float[Array, "... chunks chunk"] | Float[Array, "... chunks"],
    b: Float[Array, "... chunks chunk state rank"],
    c: Float[Array, "... chunks chunk state rank"],
    x_base: Float[Array, "... chunks chunk value"],
    z_base: Float[Array, "... chunks chunk value"],
    w_x: Float[Array, "... value rank"] | Float[Array, "value rank"],
    w_z: Float[Array, "... value rank"] | Float[Array, "value rank"],
    w_o: Float[Array, "... value rank"] | Float[Array, "value rank"],
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Stable MIMO Mamba-3 entrypoint on native chunked inputs."""

    if implementation == "reference":
        leading_shape = dt.shape[:-2]
        groups = math.prod(leading_shape) if leading_shape else 1
        y, final_state = mamba3_mimo_direct_recurrence_reference_batched(
            dt.reshape(groups, dt.shape[-2], dt.shape[-1]),
            lam.reshape(groups, lam.shape[-2], lam.shape[-1]),
            a.reshape(groups, a.shape[-2], a.shape[-1]) if a.shape == dt.shape else a.reshape(groups, a.shape[-1]),
            b.reshape(groups, b.shape[-4], b.shape[-3], b.shape[-2], b.shape[-1]),
            c.reshape(groups, c.shape[-4], c.shape[-3], c.shape[-2], c.shape[-1]),
            x_base.reshape(groups, x_base.shape[-3], x_base.shape[-2], x_base.shape[-1]),
            z_base.reshape(groups, z_base.shape[-3], z_base.shape[-2], z_base.shape[-1]),
            w_x,
            w_z,
            w_o,
        )
        return y.reshape(leading_shape + y.shape[-3:]), final_state.reshape(leading_shape + final_state.shape[-2:])

    src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    return mamba3_mimo_chunked_forward_from_transformed(
        a_log_cumsum,
        src_scale,
        out_correction,
        b,
        c,
        x_base,
        z_base,
        w_x,
        w_z,
        w_o,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        backend=backend,
    )


def _validate_attentionish_mimo_inputs(
    q: Float[Array, "batch seq rank qk_groups state"],
    k: Float[Array, "batch seq rank qk_groups state"],
    v: Float[Array, "batch seq heads value"],
    *,
    chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    if q.shape != k.shape:
        raise ValueError(f"`q` and `k` must have the same shape, got {q.shape} and {k.shape}.")
    if q.ndim != 5 or v.ndim != 4:
        raise ValueError("Expected `q/k` with shape `[B, S, R, G, N]` and `v` with shape `[B, S, H, P]`.")
    batch, seq_len, rank, qk_groups, state_dim = q.shape
    if v.shape[:2] != (batch, seq_len):
        raise ValueError("`v` must share batch and sequence axes with `q/k`.")
    heads = v.shape[2]
    if chunk_size <= 0 or seq_len % chunk_size != 0:
        raise ValueError(f"`chunk_size` must divide the sequence length exactly, got {chunk_size} for {seq_len}.")
    if heads % qk_groups != 0:
        raise ValueError(f"`qk_groups` must divide the number of heads, got {qk_groups} for {heads}.")
    return batch, seq_len, heads, rank, qk_groups, state_dim


def _validate_attentionish_siso_inputs(
    q: Float[Array, "batch seq heads state"],
    k: Float[Array, "batch seq heads state"],
    v: Float[Array, "batch seq heads value"],
    *,
    chunk_size: int,
) -> tuple[int, int, int, int]:
    if q.shape != k.shape:
        raise ValueError(f"`q` and `k` must have the same shape, got {q.shape} and {k.shape}.")
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected `q/k` with shape `[B, S, H, N]` and `v` with shape `[B, S, H, P]`.")
    batch, seq_len, heads, state_dim = q.shape
    if v.shape[:3] != (batch, seq_len, heads):
        raise ValueError("`v` must share batch, sequence, and head axes with `q/k`.")
    if chunk_size <= 0 or seq_len % chunk_size != 0:
        raise ValueError(f"`chunk_size` must divide the sequence length exactly, got {chunk_size} for {seq_len}.")
    return batch, seq_len, heads, state_dim


def _require_none_or_zero(name: str, value: jax.Array | None) -> None:
    if value is None:
        return
    if not bool(jnp.all(value == 0)):
        raise NotImplementedError(
            f"`{name}` is only supported as `None` or all-zero in the current real-valued attention-ish MIMO path."
        )


def _reshape_attentionish_weights(
    weights: Float[Array, "heads rank value"],
    *,
    batch: int,
) -> Float[Array, "batch heads value rank"]:
    return jnp.broadcast_to(
        jnp.swapaxes(weights, -1, -2)[None, ...], (batch,) + (weights.shape[0], weights.shape[2], weights.shape[1])
    )


def _grouped_qk_to_chunked_state_rank(
    tensor: Float[Array, "batch seq rank qk_groups state"],
    *,
    num_heads: int,
    chunk_size: int,
) -> Float[Array, "batch heads chunks chunk state rank"]:
    heads_per_group = num_heads // tensor.shape[3]
    head_to_group = jnp.arange(num_heads) // heads_per_group
    by_head = jnp.take(tensor, head_to_group, axis=3)
    state_rank = by_head.transpose(0, 3, 1, 4, 2)
    num_chunks = tensor.shape[1] // chunk_size
    return state_rank.reshape(tensor.shape[0], num_heads, num_chunks, chunk_size, tensor.shape[-1], tensor.shape[2])


def _chunk_value_by_head(
    tensor: Float[Array, "batch seq heads value"],
    *,
    chunk_size: int,
) -> Float[Array, "batch heads chunks chunk value"]:
    num_chunks = tensor.shape[1] // chunk_size
    return tensor.reshape(tensor.shape[0], num_chunks, chunk_size, tensor.shape[2], tensor.shape[-1]).transpose(
        0, 3, 1, 2, 4
    )


def _chunk_state_by_head(
    tensor: Float[Array, "batch seq heads state"],
    *,
    chunk_size: int,
) -> Float[Array, "batch heads chunks chunk state"]:
    num_chunks = tensor.shape[1] // chunk_size
    return tensor.reshape(tensor.shape[0], num_chunks, chunk_size, tensor.shape[2], tensor.shape[-1]).transpose(
        0, 3, 1, 2, 4
    )


def _materialize_chunked_layout(tensor: jax.Array) -> jax.Array:
    """Force a concrete chunk-major layout before entering the hot kernel."""

    return jnp.copy(tensor)


def _package_attentionish_outputs(
    output: jax.Array,
    *,
    final_state: jax.Array | None,
    final_k: jax.Array | None,
    return_final_state: bool,
    return_final_k: bool,
) -> jax.Array | tuple[jax.Array, ...]:
    if return_final_state and return_final_k:
        return output, cast(jax.Array, final_state), cast(jax.Array, final_k)
    if return_final_state:
        return output, cast(jax.Array, final_state)
    if return_final_k:
        return output, cast(jax.Array, final_k)
    return output


def mamba3_attentionish_forward_prepacked_from_transformed(
    q_chunked: Float[Array, "batch heads chunks chunk state"],
    k_chunked: Float[Array, "batch heads chunks chunk state"],
    v_chunked: Float[Array, "batch heads chunks chunk value"],
    *,
    a_log_cumsum: Float[Array, "batch heads chunks chunk"],
    src_scale: Float[Array, "batch heads chunks chunk"],
    out_correction: Float[Array, "batch heads chunks chunk"],
    q_bias: Float[Array, "heads state"] | None = None,
    k_bias: Float[Array, "heads state"] | None = None,
    d: Float[Array, "heads"] | None = None,
    return_final_state: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """Run the attention-style SISO kernel from already chunk-packed transformed inputs.

    This is the thin TPU-facing entrypoint for model code that can precompute the
    chunk-major `q/k/v` layout and transformed scalar schedules outside the hot
    kernel path, avoiding the sequence-major packing overhead of the convenience
    attention-style wrapper.
    """

    if q_chunked.shape != k_chunked.shape:
        raise ValueError(
            f"`q_chunked` and `k_chunked` must have the same shape, got {q_chunked.shape} and {k_chunked.shape}."
        )
    if q_chunked.ndim != 5 or v_chunked.ndim != 5:
        raise ValueError(
            "Expected `q_chunked/k_chunked` with shape `[B, H, K, C, N]` and `v_chunked` with shape `[B, H, K, C, P]`."
        )
    batch, heads, num_chunks, chunk_size, state_dim = q_chunked.shape
    if v_chunked.shape[:4] != (batch, heads, num_chunks, chunk_size):
        raise ValueError("`v_chunked` must share batch, head, chunk, and token axes with `q_chunked`.")
    if a_log_cumsum.shape != (batch, heads, num_chunks, chunk_size):
        raise ValueError("`a_log_cumsum` must have shape `[B, H, K, C]`.")
    if src_scale.shape != a_log_cumsum.shape or out_correction.shape != a_log_cumsum.shape:
        raise ValueError("`src_scale` and `out_correction` must match `a_log_cumsum`.")
    if q_bias is not None and q_bias.shape != (heads, state_dim):
        raise ValueError(f"`q_bias` must have shape `[H, N]`, got {q_bias.shape}.")
    if k_bias is not None and k_bias.shape != (heads, state_dim):
        raise ValueError(f"`k_bias` must have shape `[H, N]`, got {k_bias.shape}.")
    if d is not None and d.shape != (heads,):
        raise ValueError(f"`d` must have shape `[H]`, got {d.shape}.")
    if implementation not in (None, "xla", "reference"):
        raise ValueError(
            "Prepacked attention-style SISO is currently available only for the XLA and reference implementations."
        )

    with jax.named_scope("mamba3_attentionish_siso_prepacked"):
        if q_bias is not None:
            q_chunked = q_chunked + q_bias[None, :, None, None, :]
        if k_bias is not None:
            k_chunked = k_chunked + k_bias[None, :, None, None, :]

        groups = batch * heads
        flat_c = q_chunked.reshape(groups, num_chunks, chunk_size, state_dim)
        flat_b = k_chunked.reshape(groups, num_chunks, chunk_size, state_dim)
        flat_v = v_chunked.reshape(groups, num_chunks, chunk_size, v_chunked.shape[-1])
        flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
        flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
        flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)

        with jax.named_scope("chunked_core"):
            if implementation == "reference":
                output_chunked, final_state = mamba3_chunked_forward_reference_batched(
                    flat_a_log_cumsum,
                    flat_src_scale,
                    flat_out_correction,
                    flat_b,
                    flat_c,
                    flat_v,
                )
            else:
                output_chunked, final_state = mamba3_chunked_forward_xla_batched(
                    flat_a_log_cumsum,
                    flat_src_scale,
                    flat_out_correction,
                    flat_b,
                    flat_c,
                    flat_v,
                )

        if d is not None:
            flat_d = jnp.broadcast_to(d[None, :], (batch, heads)).reshape(groups)
            output_chunked = output_chunked + flat_d[:, None, None, None] * flat_v

        output = output_chunked.reshape(batch, heads, num_chunks, chunk_size, v_chunked.shape[-1]).transpose(
            0, 2, 3, 1, 4
        )
        output = output.reshape(batch, num_chunks * chunk_size, heads, v_chunked.shape[-1])
        if not return_final_state:
            return output

        formatted_final_state = final_state.reshape(
            batch, heads, final_state.shape[-2], final_state.shape[-1]
        ).transpose(0, 1, 3, 2)
        return output, formatted_final_state


def mamba3_attentionish_forward_from_transformed(
    q: Float[Array, "batch seq heads state"],
    k: Float[Array, "batch seq heads state"],
    v: Float[Array, "batch seq heads value"],
    *,
    q_bias: Float[Array, "heads state"] | None = None,
    k_bias: Float[Array, "heads state"] | None = None,
    d: Float[Array, "heads"] | None = None,
    angles: Float[Array, "batch seq heads rot"] | None = None,
    da_cs: Float[Array, "batch heads seq"] | None = None,
    da_cs_rev: Float[Array, "batch heads seq"] | None = None,
    dt: Float[Array, "batch heads seq"] | None = None,
    trap: Float[Array, "batch heads seq"] | None = None,
    segsum: Float[Array, "batch heads chunks chunk chunk"] | None = None,
    chunk_size: int,
    return_final_state: bool = False,
    return_final_k: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Attention-like real-valued SISO entrypoint mapped onto the chunked Mamba-3 kernel family."""

    del block_sizes, interpret, backend, da_cs_rev
    _require_none_or_zero("angles", angles)
    _require_none_or_zero("segsum", segsum)
    batch, seq_len, heads, state_dim = _validate_attentionish_siso_inputs(q, k, v, chunk_size=chunk_size)
    if da_cs is None or dt is None or trap is None:
        raise ValueError("`da_cs`, `dt`, and `trap` are required for the transformed attention-ish SISO API.")
    if da_cs.shape != (batch, heads, seq_len) or dt.shape != da_cs.shape or trap.shape != da_cs.shape:
        raise ValueError("`da_cs`, `dt`, and `trap` must all have shape `[B, H, S]`.")
    if q_bias is not None and q_bias.shape != (heads, state_dim):
        raise ValueError(f"`q_bias` must have shape `[H, N]`, got {q_bias.shape}.")
    if k_bias is not None and k_bias.shape != (heads, state_dim):
        raise ValueError(f"`k_bias` must have shape `[H, N]`, got {k_bias.shape}.")
    if d is not None and d.shape != (heads,):
        raise ValueError(f"`d` must have shape `[H]`, got {d.shape}.")
    if implementation not in (None, "xla", "reference"):
        raise ValueError("Attention-ish SISO is currently available only for the XLA and reference implementations.")

    with jax.named_scope("mamba3_attentionish_siso"):
        num_chunks = seq_len // chunk_size
        with jax.named_scope("prepare_qk"):
            q_chunked = _chunk_state_by_head(q, chunk_size=chunk_size)
            k_chunked = _chunk_state_by_head(k, chunk_size=chunk_size)
            if q_bias is not None:
                q_chunked = q_chunked + q_bias[None, :, None, None, :]
            if k_bias is not None:
                k_chunked = k_chunked + k_bias[None, :, None, None, :]
            q_chunked = _materialize_chunked_layout(q_chunked)
            k_chunked = _materialize_chunked_layout(k_chunked)
        with jax.named_scope("prepare_v"):
            v_chunked = _chunk_value_by_head(v, chunk_size=chunk_size)
            v_chunked = _materialize_chunked_layout(v_chunked)
        with jax.named_scope("prepare_scales"):
            a_log_cumsum = da_cs.reshape(batch, heads, num_chunks, chunk_size)
            lam = jax.nn.sigmoid(trap)
            src_scale, out_correction = prepare_mamba3_chunked_scales(
                dt.reshape(batch, heads, num_chunks, chunk_size),
                lam.reshape(batch, heads, num_chunks, chunk_size),
            )
            a_log_cumsum = _materialize_chunked_layout(a_log_cumsum)
            src_scale = _materialize_chunked_layout(src_scale)
            out_correction = _materialize_chunked_layout(out_correction)

        groups = batch * heads
        flat_c = q_chunked.reshape(groups, num_chunks, chunk_size, state_dim)
        flat_b = k_chunked.reshape(groups, num_chunks, chunk_size, state_dim)
        flat_v = v_chunked.reshape(groups, num_chunks, chunk_size, v.shape[-1])
        flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
        flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
        flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)

        with jax.named_scope("chunked_core"):
            if implementation == "reference":
                output_chunked, final_state = mamba3_chunked_forward_reference_batched(
                    flat_a_log_cumsum,
                    flat_src_scale,
                    flat_out_correction,
                    flat_b,
                    flat_c,
                    flat_v,
                )
            else:
                output_chunked, final_state = mamba3_chunked_forward_xla_batched(
                    flat_a_log_cumsum,
                    flat_src_scale,
                    flat_out_correction,
                    flat_b,
                    flat_c,
                    flat_v,
                )

        with jax.named_scope("diagonal_skip"):
            if d is not None:
                flat_d = jnp.broadcast_to(d[None, :], (batch, heads)).reshape(groups)
                output_chunked = output_chunked + flat_d[:, None, None, None] * flat_v

        with jax.named_scope("package_output"):
            output = output_chunked.reshape(batch, heads, num_chunks, chunk_size, v.shape[-1]).transpose(0, 2, 3, 1, 4)
            output = output.reshape(batch, seq_len, heads, v.shape[-1])
            formatted_final_state = (
                final_state.reshape(batch, heads, final_state.shape[-2], final_state.shape[-1]).transpose(0, 1, 3, 2)
                if return_final_state
                else None
            )
            formatted_final_k = None
            if return_final_k:
                final_k = jnp.take(k, seq_len - 1, axis=1)
                if k_bias is not None:
                    final_k = final_k + k_bias[None, ...]
                formatted_final_k = final_k

        return _package_attentionish_outputs(
            output,
            final_state=formatted_final_state,
            final_k=formatted_final_k,
            return_final_state=return_final_state,
            return_final_k=return_final_k,
        )


def mamba3_attentionish_forward(
    q: Float[Array, "batch seq heads state"],
    k: Float[Array, "batch seq heads state"],
    v: Float[Array, "batch seq heads value"],
    *,
    q_bias: Float[Array, "heads state"] | None = None,
    k_bias: Float[Array, "heads state"] | None = None,
    d: Float[Array, "heads"] | None = None,
    angles: Float[Array, "batch seq heads rot"] | None = None,
    da_cs: Float[Array, "batch heads seq"] | None = None,
    da_cs_rev: Float[Array, "batch heads seq"] | None = None,
    dt: Float[Array, "batch heads seq"] | None = None,
    trap: Float[Array, "batch heads seq"] | None = None,
    segsum: Float[Array, "batch heads chunks chunk chunk"] | None = None,
    chunk_size: int,
    return_final_state: bool = False,
    return_final_k: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Native attention-like SISO Mamba-3 entrypoint."""

    return mamba3_attentionish_forward_from_transformed(
        q,
        k,
        v,
        q_bias=q_bias,
        k_bias=k_bias,
        d=d,
        angles=angles,
        da_cs=da_cs,
        da_cs_rev=da_cs_rev,
        dt=dt,
        trap=trap,
        segsum=segsum,
        chunk_size=chunk_size,
        return_final_state=return_final_state,
        return_final_k=return_final_k,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        backend=backend,
    )


def mamba3_mimo_attentionish_forward_from_transformed(
    q: Float[Array, "batch seq rank qk_groups state"],
    k: Float[Array, "batch seq rank qk_groups state"],
    v: Float[Array, "batch seq heads value"],
    mimo_v: Float[Array, "heads rank value"],
    mimo_o: Float[Array, "heads rank value"],
    *,
    q_bias: Float[Array, "heads rank state"] | None = None,
    k_bias: Float[Array, "heads rank state"] | None = None,
    z: Float[Array, "batch seq heads value"] | None = None,
    d: Float[Array, "heads"] | None = None,
    mimo_z: Float[Array, "heads rank value"] | None = None,
    angles: Float[Array, "batch seq heads rot"] | None = None,
    da_cs: Float[Array, "batch heads seq"] | None = None,
    da_cs_rev: Float[Array, "batch heads seq"] | None = None,
    dt: Float[Array, "batch heads seq"] | None = None,
    trap: Float[Array, "batch heads seq"] | None = None,
    segsum: Float[Array, "batch heads chunks chunk chunk"] | None = None,
    chunk_size: int,
    reduce_o: bool = True,
    return_final_state: bool = False,
    return_final_k: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Attention-like real-valued MIMO entrypoint mapped onto the current chunked Mamba-3 kernel family.

    This currently supports the real-valued no-RoPE/no-segsum subset of the spec. `angles` and `segsum`
    must therefore be `None` or all-zero, and `da_cs`/`dt`/`trap` are required.
    """

    del block_sizes, interpret, backend, da_cs_rev
    _require_none_or_zero("angles", angles)
    _require_none_or_zero("segsum", segsum)
    batch, seq_len, heads, rank, _, state_dim = _validate_attentionish_mimo_inputs(q, k, v, chunk_size=chunk_size)
    if da_cs is None or dt is None or trap is None:
        raise ValueError("`da_cs`, `dt`, and `trap` are required for the transformed attention-ish MIMO API.")
    if da_cs.shape != (batch, heads, seq_len) or dt.shape != da_cs.shape or trap.shape != da_cs.shape:
        raise ValueError("`da_cs`, `dt`, and `trap` must all have shape `[B, H, S]`.")
    if q_bias is not None and q_bias.shape != (heads, rank, state_dim):
        raise ValueError(f"`q_bias` must have shape `[H, R, N]`, got {q_bias.shape}.")
    if k_bias is not None and k_bias.shape != (heads, rank, state_dim):
        raise ValueError(f"`k_bias` must have shape `[H, R, N]`, got {k_bias.shape}.")
    if mimo_v.shape[:2] != (heads, rank) or mimo_o.shape[:2] != (heads, rank):
        raise ValueError("`mimo_v` and `mimo_o` must have shape `[H, R, P]`.")
    if z is not None and z.shape != v.shape:
        raise ValueError("`z` must have the same shape as `v`.")
    if mimo_z is not None and mimo_z.shape != mimo_v.shape:
        raise ValueError("`mimo_z` must have the same shape as `mimo_v`.")
    if z is None and mimo_z is not None:
        raise ValueError("`mimo_z` requires `z`.")
    if z is not None and mimo_z is None:
        raise ValueError("`z` requires `mimo_z`.")

    if implementation not in (None, "xla", "reference"):
        raise ValueError("Attention-ish MIMO is currently available only for the XLA and reference implementations.")

    num_chunks = seq_len // chunk_size
    q_chunked = _grouped_qk_to_chunked_state_rank(q, num_heads=heads, chunk_size=chunk_size)
    k_chunked = _grouped_qk_to_chunked_state_rank(k, num_heads=heads, chunk_size=chunk_size)
    if q_bias is not None:
        q_chunked = q_chunked + jnp.swapaxes(q_bias, -1, -2)[None, :, None, None, :, :]
    if k_bias is not None:
        k_chunked = k_chunked + jnp.swapaxes(k_bias, -1, -2)[None, :, None, None, :, :]

    v_chunked = _chunk_value_by_head(v, chunk_size=chunk_size)
    z_chunked = _chunk_value_by_head(z, chunk_size=chunk_size) if z is not None else None
    a_log_cumsum = da_cs.reshape(batch, heads, num_chunks, chunk_size)
    lam = jax.nn.sigmoid(trap)
    src_scale, out_correction = prepare_mamba3_chunked_scales(
        dt.reshape(batch, heads, num_chunks, chunk_size),
        lam.reshape(batch, heads, num_chunks, chunk_size),
    )

    groups = batch * heads
    flat_c = q_chunked.reshape(groups, num_chunks, chunk_size, state_dim, rank)
    flat_b = k_chunked.reshape(groups, num_chunks, chunk_size, state_dim, rank)
    flat_x = v_chunked.reshape(groups, num_chunks, chunk_size, v.shape[-1])
    flat_a_log_cumsum = a_log_cumsum.reshape(groups, num_chunks, chunk_size)
    flat_src_scale = src_scale.reshape(groups, num_chunks, chunk_size)
    flat_out_correction = out_correction.reshape(groups, num_chunks, chunk_size)
    flat_w_v = _reshape_attentionish_weights(mimo_v, batch=batch).reshape(groups, v.shape[-1], rank)
    flat_w_o = _reshape_attentionish_weights(mimo_o, batch=batch).reshape(groups, v.shape[-1], rank)
    flat_x_ranked = mamba3_mimo_rank_expand_chunked(flat_x, flat_w_v)

    if implementation == "reference":
        y_ranked, final_state = mamba3_mimo_chunked_forward_ranked_reference_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x_ranked,
        )
    else:
        y_ranked, final_state = mamba3_mimo_chunked_forward_ranked_xla_batched(
            flat_a_log_cumsum,
            flat_src_scale,
            flat_out_correction,
            flat_b,
            flat_c,
            flat_x_ranked,
        )

    if d is not None:
        if d.shape != (heads,):
            raise ValueError(f"`d` must have shape `[H]`, got {d.shape}.")
        flat_d = jnp.broadcast_to(d[None, :], (batch, heads)).reshape(groups)
        y_ranked = y_ranked + flat_d[:, None, None, None] * flat_x_ranked

    if z_chunked is not None:
        flat_z = z_chunked.reshape(groups, num_chunks, chunk_size, z.shape[-1])
        flat_w_z = _reshape_attentionish_weights(cast(jax.Array, mimo_z), batch=batch).reshape(
            groups, z.shape[-1], rank
        )
        if reduce_o:
            output = mamba3_mimo_apply_gate_and_collapse_chunked(
                y_ranked, mamba3_mimo_rank_expand_chunked(flat_z, flat_w_z), flat_w_o
            )
        else:
            y_ranked = y_ranked.astype(jnp.float32) * jax.nn.silu(
                mamba3_mimo_rank_expand_chunked(flat_z, flat_w_z).astype(jnp.float32)
            )
            output = y_ranked.astype(flat_x.dtype)
    else:
        output = mamba3_mimo_rank_collapse_chunked(y_ranked, flat_w_o) if reduce_o else y_ranked

    if reduce_o:
        output = (
            output.reshape(batch, heads, num_chunks, chunk_size, v.shape[-1])
            .transpose(0, 2, 3, 1, 4)
            .reshape(batch, seq_len, heads, v.shape[-1])
        )
    else:
        output = output.reshape(batch, heads, num_chunks, rank, chunk_size, v.shape[-1]).transpose(0, 2, 4, 3, 1, 5)
        output = output.reshape(batch, seq_len, rank, heads, v.shape[-1])

    formatted_final_state = (
        final_state.reshape(batch, heads, final_state.shape[-2], final_state.shape[-1]).transpose(0, 1, 3, 2)
        if return_final_state
        else None
    )
    formatted_final_k = None
    if return_final_k:
        last_k = jnp.take(k, seq_len - 1, axis=1)
        heads_per_group = heads // q.shape[3]
        head_to_group = jnp.arange(heads) // heads_per_group
        final_k = jnp.take(last_k, head_to_group, axis=2)
        if k_bias is not None:
            final_k = final_k + k_bias.transpose(1, 0, 2)[None, ...]
        formatted_final_k = final_k

    return _package_attentionish_outputs(
        output,
        final_state=formatted_final_state,
        final_k=formatted_final_k,
        return_final_state=return_final_state,
        return_final_k=return_final_k,
    )


def mamba3_mimo_attentionish_forward(
    q: Float[Array, "batch seq rank qk_groups state"],
    k: Float[Array, "batch seq rank qk_groups state"],
    v: Float[Array, "batch seq heads value"],
    mimo_v: Float[Array, "heads rank value"],
    mimo_o: Float[Array, "heads rank value"],
    *,
    q_bias: Float[Array, "heads rank state"] | None = None,
    k_bias: Float[Array, "heads rank state"] | None = None,
    z: Float[Array, "batch seq heads value"] | None = None,
    d: Float[Array, "heads"] | None = None,
    mimo_z: Float[Array, "heads rank value"] | None = None,
    angles: Float[Array, "batch seq heads rot"] | None = None,
    da_cs: Float[Array, "batch heads seq"] | None = None,
    da_cs_rev: Float[Array, "batch heads seq"] | None = None,
    dt: Float[Array, "batch heads seq"] | None = None,
    trap: Float[Array, "batch heads seq"] | None = None,
    segsum: Float[Array, "batch heads chunks chunk chunk"] | None = None,
    chunk_size: int,
    reduce_o: bool = True,
    return_final_state: bool = False,
    return_final_k: bool = False,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> jax.Array | tuple[jax.Array, ...]:
    """Native attention-like MIMO Mamba-3 entrypoint.

    The current implementation is real-valued and expects precomputed `da_cs` along with `dt` and `trap`.
    """

    return mamba3_mimo_attentionish_forward_from_transformed(
        q,
        k,
        v,
        mimo_v,
        mimo_o,
        q_bias=q_bias,
        k_bias=k_bias,
        z=z,
        d=d,
        mimo_z=mimo_z,
        angles=angles,
        da_cs=da_cs,
        da_cs_rev=da_cs_rev,
        dt=dt,
        trap=trap,
        segsum=segsum,
        chunk_size=chunk_size,
        reduce_o=reduce_o,
        return_final_state=return_final_state,
        return_final_k=return_final_k,
        implementation=implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        backend=backend,
    )


def mamba3_hybrid_chunked_forward_from_transformed(
    a_log_cumsum: Float[Array, "... chunks chunk"],
    src_scale: Float[Array, "... chunks chunk"],
    out_correction: Float[Array, "... chunks chunk"],
    b: Float[Array, "..."],
    c: Float[Array, "..."],
    x: Float[Array, "... chunks chunk value"],
    *,
    mode: Mamba3Mode = "siso",
    z: Float[Array, "... chunks chunk value"] | None = None,
    w_x: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    w_z: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    w_o: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Dispatch transformed Mamba-3 inputs to the stable SISO or MIMO API."""

    if mode == "siso":
        if any(arg is not None for arg in (z, w_x, w_z, w_o)):
            raise ValueError("SISO hybrid mode does not accept MIMO-only arguments `z`, `w_x`, `w_z`, or `w_o`.")
        return mamba3_chunked_forward_from_transformed(
            a_log_cumsum,
            src_scale,
            out_correction,
            cast(Float[Array, "... chunks chunk state"], b),
            cast(Float[Array, "... chunks chunk state"], c),
            x,
            implementation=implementation,
            block_sizes=block_sizes,
            interpret=interpret,
            backend=backend,
        )

    if mode == "mimo":
        if z is None or w_x is None or w_z is None or w_o is None:
            raise ValueError("MIMO hybrid mode requires `z`, `w_x`, `w_z`, and `w_o`.")
        return mamba3_mimo_chunked_forward_from_transformed(
            a_log_cumsum,
            src_scale,
            out_correction,
            cast(Float[Array, "... chunks chunk state rank"], b),
            cast(Float[Array, "... chunks chunk state rank"], c),
            x,
            z,
            w_x,
            w_z,
            w_o,
            implementation=implementation,
            block_sizes=block_sizes,
            interpret=interpret,
            backend=backend,
        )

    raise ValueError(f"Unsupported hybrid Mamba-3 mode: {mode}.")


def mamba3_hybrid_chunked_forward(
    dt: Float[Array, "... chunks chunk"],
    lam: Float[Array, "... chunks chunk"],
    a: Float[Array, "... chunks chunk"] | Float[Array, "... chunks"],
    b: Float[Array, "..."],
    c: Float[Array, "..."],
    x: Float[Array, "... chunks chunk value"],
    *,
    mode: Mamba3Mode = "siso",
    z: Float[Array, "... chunks chunk value"] | None = None,
    w_x: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    w_z: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    w_o: Float[Array, "... value rank"] | Float[Array, "value rank"] | None = None,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: BlockSizes | None = None,
    interpret: bool = False,
    backend: str | None = None,
) -> tuple[Float[Array, "... chunks chunk value"], Float[Array, "... value state"]]:
    """Dispatch native Mamba-3 inputs to the stable SISO or MIMO API."""

    if mode == "siso":
        if any(arg is not None for arg in (z, w_x, w_z, w_o)):
            raise ValueError("SISO hybrid mode does not accept MIMO-only arguments `z`, `w_x`, `w_z`, or `w_o`.")
        return mamba3_chunked_forward(
            dt,
            lam,
            a,
            cast(Float[Array, "... chunks chunk state"], b),
            cast(Float[Array, "... chunks chunk state"], c),
            x,
            implementation=implementation,
            block_sizes=block_sizes,
            interpret=interpret,
            backend=backend,
        )

    if mode == "mimo":
        if z is None or w_x is None or w_z is None or w_o is None:
            raise ValueError("MIMO hybrid mode requires `z`, `w_x`, `w_z`, and `w_o`.")
        return mamba3_mimo_chunked_forward(
            dt,
            lam,
            a,
            cast(Float[Array, "... chunks chunk state rank"], b),
            cast(Float[Array, "... chunks chunk state rank"], c),
            x,
            z,
            w_x,
            w_z,
            w_o,
            implementation=implementation,
            block_sizes=block_sizes,
            interpret=interpret,
            backend=backend,
        )

    raise ValueError(f"Unsupported hybrid Mamba-3 mode: {mode}.")


__all__ = [
    "BlockSizes",
    "HybridModeConfig",
    "IMPLEMENTATIONS",
    "Implementation",
    "Mamba3Mode",
    "intra_chunk_log_alpha_cumsum",
    "local_log_alpha",
    "mamba3_attentionish_forward",
    "mamba3_attentionish_forward_from_transformed",
    "mamba3_hybrid_chunked_forward",
    "mamba3_hybrid_chunked_forward_from_transformed",
    "mamba3_tpu_default_chunk_size",
    "mamba3_chunk_state",
    "mamba3_mimo_attentionish_forward",
    "mamba3_mimo_attentionish_forward_from_transformed",
    "mamba3_mimo_chunked_forward",
    "mamba3_mimo_chunked_forward_from_transformed",
    "mamba3_chunk_state_reference_batched",
    "mamba3_chunked_forward",
    "mamba3_chunked_forward_from_transformed",
    "mamba3_mimo_chunked_forward_reference_batched",
    "mamba3_mimo_direct_recurrence_reference_batched",
    "mamba3_chunked_forward_reference_batched",
    "mamba3_chunked_sequential_reference_batched",
    "mamba3_direct_recurrence_reference_batched",
    "mamba3_intra_chunk",
    "mamba3_intra_chunk_reference_batched",
    "prepare_mamba3_chunked_scales",
    "prepare_mamba3_scales",
]
