# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API and backend selection for the depthwise causal short convolution."""

import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

from levanter.sharding import partitioning_axes, partition_spec_of

from .reference import OOB_SEGMENT, short_conv_reference
from .triton_gpu import (
    SEQUENCE_MULTIPLE as TRITON_SEQUENCE_MULTIPLE,
    short_conv_triton_bwd_local,
    short_conv_triton_fwd_local,
    triton_short_conv_available,
    triton_short_conv_shapes_supported,
)

logger = logging.getLogger(__name__)

Implementation: TypeAlias = Literal["reference", "triton_gpu"]

#: Mesh axes the activation batch is sharded over in the grug MoE models. The kernel is
#: shard-local along every one of them. The sequence may be sharded over one further axis
#: (a halo exchange covers the taps that cross shards); the channel axis must stay whole.
DEFAULT_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _default_implementations() -> tuple[Implementation, ...]:
    if triton_short_conv_available():
        return ("triton_gpu", "reference")
    return ("reference",)


def _as_sequence(
    implementation: Implementation | Sequence[Implementation] | None,
) -> tuple[Implementation, ...]:
    if implementation is None:
        return _default_implementations()
    if isinstance(implementation, str):
        return (implementation,)  # explicit single choice: fail fast, never silently fall back
    return tuple(implementation)


def _active_batch_axes(mesh, batch_axes: Sequence[str]) -> tuple[str, ...]:
    return tuple(axis for axis in batch_axes if axis in mesh.shape)


def _partitioned_dims(array: jax.Array, mesh) -> tuple[tuple[str, ...], ...] | None:
    """Per-dimension partitioning mesh axes of ``array``, or None when it carries no spec."""
    spec = partition_spec_of(array)
    if spec is None:
        return None
    return tuple(partitioning_axes(entry, mesh) for entry in spec)


def _assert_local_axes(name: str, array: jax.Array, axes: Sequence[int], mesh) -> None:
    """Reject partitioned sequence/channel dimensions to prevent implicit all-gathers."""
    dims = _partitioned_dims(array, mesh)
    if dims is None:
        return
    for axis in (a for a in axes if a < len(dims)):
        if dims[axis]:
            raise ValueError(
                f"short_conv requires an unsharded {'sequence' if axis == 1 else 'channel'} axis "
                f"for {name}; got {partition_spec_of(array)}."
            )


# The whole cost of this op is in the backward, so the backward is the kernel's own rather than
# whatever reverse-mode AD would derive. Residuals are the primal inputs only.
@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _short_conv_triton_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    exact_reference_rounding: bool,
) -> Float[Array, "B S C"]:
    return short_conv_triton_fwd_local(weight, x, segment_ids, exact_reference_rounding=exact_reference_rounding)


def _short_conv_triton_local_fwd(weight, x, segment_ids, exact_reference_rounding):
    out = short_conv_triton_fwd_local(weight, x, segment_ids, exact_reference_rounding=exact_reference_rounding)
    return out, (weight, x, segment_ids)


def _short_conv_triton_local_bwd(exact_reference_rounding, residuals, dy):
    weight, x, segment_ids = residuals
    dx, dw_partials = short_conv_triton_bwd_local(
        weight, x, segment_ids, dy, exact_reference_rounding=exact_reference_rounding
    )
    return jnp.sum(dw_partials, axis=0).astype(weight.dtype), dx, None


_short_conv_triton_local.defvjp(_short_conv_triton_local_fwd, _short_conv_triton_local_bwd)


def _triton_local_call(
    weight: jax.Array,
    x: jax.Array,
    segment_ids: jax.Array | None,
    *,
    exact_reference_rounding: bool,
) -> jax.Array:
    if segment_ids is None:
        segment_ids = jnp.zeros(x.shape[:2], jnp.int32)
    return _short_conv_triton_local(weight, x, segment_ids, exact_reference_rounding)


def _sequence_shard_axis(array: jax.Array, mesh) -> str | None:
    """The one mesh axis partitioning the sequence across devices, or None when it is whole."""
    dims = _partitioned_dims(array, mesh)
    if dims is None or len(dims) < 2:
        return None
    axes = dims[1]
    if len(axes) > 1:
        raise ValueError(f"short_conv supports one sequence-sharding mesh axis; got {axes}")
    return axes[0] if axes else None


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


LocalCall: TypeAlias = Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array]


def _call_right_padded(
    local_call: LocalCall, weight: jax.Array, x: jax.Array, segment_ids: jax.Array | None, padded_seq: int
) -> jax.Array:
    """``local_call`` on ``x`` right-padded to ``padded_seq`` rows.

    The padded rows carry the out-of-range segment id, so they read and contribute as the zeros past
    the end of a sequence do. Causality keeps them from affecting earlier outputs; the caller drops
    their own outputs.
    """
    tail = padded_seq - x.shape[1]
    if not tail:
        return local_call(weight, x, segment_ids)
    if segment_ids is None:
        segment_ids = jnp.zeros(x.shape[:2], jnp.int32)
    x = jnp.pad(x, ((0, 0), (0, tail), (0, 0)))
    segment_ids = jnp.pad(segment_ids, ((0, 0), (0, tail)), constant_values=OOB_SEGMENT)
    return local_call(weight, x, segment_ids)


def _short_conv_sharded(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"] | None,
    *,
    local_call: LocalCall,
    mesh,
    batch_axes: Sequence[str],
    seq_axis: str | None,
    padded_local_seq: int,
) -> Float[Array, "B S C"]:
    """Run ``local_call`` inside an explicit ``shard_map``.

    Sequence shards prepend a left halo. The local block is right-padded to ``padded_local_seq``
    before convolution, and the halo and padding outputs are discarded. An unsharded sequence needs
    no communication.
    """
    seq_len = x.shape[1]
    if mesh is None:
        return _call_right_padded(local_call, weight, x, segment_ids, padded_local_seq)[:, :seq_len, :]
    # An axis that shards the sequence cannot also shard the batch of the same array.
    active = tuple(axis for axis in _active_batch_axes(mesh, batch_axes) if axis != seq_axis)
    if seq_axis is None and not active:
        return _call_right_padded(local_call, weight, x, segment_ids, padded_local_seq)[:, :seq_len, :]

    # The sequence axis is sharded by design on the halo path; any other sharded axis would
    # be a hidden all-gather, including a batch axis the caller did not name.
    _assert_local_axes("x", x, axes=(2,) if seq_axis else (1, 2), mesh=mesh)
    dims = _partitioned_dims(x, mesh)
    unnamed = tuple(axis for axis in (dims[0] if dims else ()) if axis not in active)
    if unnamed:
        raise ValueError(
            f"short_conv would all-gather x's batch axis over {unnamed}, which is not in "
            f"batch_axes {tuple(batch_axes)}; got {partition_spec_of(x)}."
        )
    batch_spec = active or None
    x_spec = P(batch_spec, seq_axis, None)
    seg_spec = P(batch_spec, seq_axis)
    x = reshard(x, x_spec)
    if segment_ids is not None:
        segment_ids = reshard(segment_ids, seg_spec)
    weight = reshard(weight, P(None, None))
    halo = weight.shape[0] - 1 if seq_axis else 0

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None, None), x_spec, seg_spec),
        out_specs=x_spec,
        check_vma=False,
    )
    def _local(weight_local, x_local, segment_ids_local):
        local_seq = x_local.shape[1]
        x_block, seg_block = x_local, segment_ids_local
        if halo:
            shards = jax.lax.axis_size(seq_axis)
            # Rank r sends its tail to r+1; rank 0 is nobody's destination, so ppermute zeroes
            # its halo, which reads as the zeros before the start of the sequence.
            shift = [(src, src + 1) for src in range(shards - 1)]
            x_halo = jax.lax.ppermute(x_local[:, local_seq - halo :, :], seq_axis, shift)
            x_block = jnp.concatenate([x_halo, x_local], axis=1)
            if seg_block is not None:
                seg_halo = jax.lax.ppermute(seg_block[:, local_seq - halo :], seq_axis, shift)
                first = jax.lax.axis_index(seq_axis) == 0
                seg_halo = jnp.where(first, jnp.full_like(seg_halo, OOB_SEGMENT), seg_halo)
                seg_block = jnp.concatenate([seg_halo, seg_block], axis=1)
        out = _call_right_padded(local_call, weight_local, x_block, seg_block, padded_local_seq)
        return out[:, halo : halo + local_seq, :]

    # pyrefly: ignore[bad-argument-count]  # jax.shard_map decorator erases _local's real signature
    return _local(weight, x, segment_ids)


def short_conv(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"] | None = None,
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    exact_reference_rounding: bool = True,
    batch_axes: Sequence[str] = DEFAULT_BATCH_AXES,
) -> Float[Array, "B S C"]:
    """Depthwise causal 1-D convolution over the sequence axis.

    ``out[b,t,c] = sum_lag weight[lag,c] * x[b,t-lag,c]``, with taps that would reach into
    a previous packed document dropped and positions before the sequence start read as
    zero. Each channel is independent.

    A sequence sharded over one mesh axis exchanges a left halo before local
    convolution. Each shard must contain at least ``kernel_size - 1`` tokens.
    With ``exact_reference_rounding=True``, the bf16 forward matches the unsharded
    reference bitwise, and so does the Triton kernel's fp32 forward. On the reference
    path, XLA fusion can change fp32 results.
    Input gradients near shard boundaries combine separately rounded local and
    neighboring contributions, so they can differ from the unsharded reference.

    Args:
      weight: ``[kernel_size, channels]`` taps; ``weight[0]`` is the current token.
      x: ``[batch, seq_len, channels]`` activations.
      segment_ids: ``[batch, seq_len]`` packed-document ids, or None for an unpacked batch.
      implementation: a single name (fail fast if unsupported) or an ordered sequence to
        try in turn. Defaults to the Triton kernel on GPU and the reference elsewhere, or when
        the kernel cannot take the call: it needs kernel size 4, a channel count that is a
        multiple of 64, and bfloat16 or float32 inputs.
      exact_reference_rounding: keep the reference's per-op bf16 rounding, which makes the
        forward and, with the sequence whole, ``dx`` bit-identical to ``short_conv_reference``.
        Setting False keeps a single fp32 accumulator across taps -- more accurate, not
        bit-comparable.
      batch_axes: mesh axes the batch is sharded over.
    """
    if weight.dtype != x.dtype:
        # The reference promotes mixed dtypes via standard JAX rules (fp32 for fp32 weight /
        # bf16 x) while the kernel outputs x.dtype, so mixed inputs would give
        # backend-dependent dtypes and values. Normalise at the boundary instead.
        raise ValueError(
            f"short_conv requires weight and x to share a dtype; got weight={weight.dtype}, "
            f"x={x.dtype}. Cast to a common dtype before calling."
        )

    requested = _as_sequence(implementation)
    explicit_single = isinstance(implementation, str)

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        mesh = None
    seq_axis = None if mesh is None else _sequence_shard_axis(x, mesh)
    halo = weight.shape[0] - 1
    local_seq = x.shape[1]
    if seq_axis is not None:
        shards = mesh.shape[seq_axis]
        if local_seq % shards:
            raise ValueError(f"seq_len {local_seq} is not divisible by the '{seq_axis}' axis ({shards})")
        local_seq //= shards
        if halo > local_seq:
            raise ValueError(f"short_conv halo size {halo} exceeds the local sequence length {local_seq}")
    # The kernel walks the local sequence, plus the halo when sharded, in whole chunks.
    triton_local_seq = _round_up(local_seq + halo if seq_axis else local_seq, TRITON_SEQUENCE_MULTIPLE)
    sharded = functools.partial(
        _short_conv_sharded,
        weight,
        x,
        segment_ids,
        mesh=mesh,
        batch_axes=batch_axes,
        seq_axis=seq_axis,
    )

    errors: list[str] = []
    for name in requested:
        if name == "reference":
            if seq_axis is None:
                return short_conv_reference(weight, x, segment_ids)
            return sharded(local_call=short_conv_reference, padded_local_seq=local_seq + halo)
        if name == "triton_gpu":
            if not triton_short_conv_available():
                reason = "Triton backend unavailable or not running on a GPU"
            else:
                reason = triton_short_conv_shapes_supported(
                    weight.shape, (x.shape[0], triton_local_seq, x.shape[2]), x.dtype, exact_reference_rounding
                )
            if reason is not None:
                if explicit_single:
                    raise RuntimeError(f"short_conv implementation 'triton_gpu' is unusable: {reason}")
                errors.append(f"triton_gpu: {reason}")
                warnings.warn(f"short_conv falling back from 'triton_gpu' ({reason})", stacklevel=2)
                continue
            return sharded(
                local_call=functools.partial(_triton_local_call, exact_reference_rounding=exact_reference_rounding),
                padded_local_seq=triton_local_seq,
            )
        raise ValueError(f"Unknown short_conv implementation {name!r}")

    raise RuntimeError("No usable short_conv implementation: " + "; ".join(errors))
