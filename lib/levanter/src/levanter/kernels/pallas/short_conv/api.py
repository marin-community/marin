# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Public API and backend selection for the depthwise causal short convolution."""

import functools
import logging
import warnings
from collections.abc import Sequence
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from jaxtyping import Array, Float, Int

from .config import ShortConvBlockSizes
from .pallas_gpu import (
    pallas_short_conv_available,
    short_conv_pallas_bwd_local,
    short_conv_pallas_fwd_local,
    short_conv_shapes_supported,
)
from .reference import short_conv_reference

logger = logging.getLogger(__name__)

Implementation: TypeAlias = Literal["reference", "pallas_gpu"]

#: Mesh axes the activation batch is sharded over in the grug MoE models. The kernel is
#: shard-local along every one of them; the sequence and channel axes must stay whole.
DEFAULT_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _default_implementations() -> tuple[Implementation, ...]:
    if pallas_short_conv_available():
        return ("pallas_gpu", "reference")
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


def _mesh_sizes(mesh) -> dict[str, int]:
    try:
        return dict(mesh.shape)
    except (AttributeError, TypeError):
        return {}


def _assert_local_axes(name: str, array: jax.Array, rank: int, mesh=None) -> None:
    """Reject a sequence or channel axis that is *actually* sharded.

    The caller reshards to ``P(active, None, None)`` immediately after this check, so the point is
    not to reject a spec -- it is to refuse to paper over a real all-gather with a silent reshard.

    A spec entry naming a mesh axis of size 1 shards nothing. That is not a corner case here: the
    EP64 hero mesh is (replica_dcn=1, data=1, expert=64, model=1), and the attention projections
    are ``P(_FSDP_AXES, "model")``, so every k/v activation carries "model" on its channel axis.
    Rejecting on the name alone fails the hero while the reshard it is guarding is a no-op.
    """
    sharding = jax.typeof(array).sharding if isinstance(array, jax.core.Tracer) else array.sharding
    spec = getattr(sharding, "spec", None)
    if spec is None:
        return
    sizes = _mesh_sizes(mesh) if mesh is not None else {}
    for axis in range(1, min(rank, len(spec))):
        entry = spec[axis]
        if entry is None:
            continue
        names = (entry,) if isinstance(entry, str) else tuple(entry)
        if sizes and all(sizes.get(n, 1) == 1 for n in names):
            continue
        raise ValueError(
            f"short_conv requires an unsharded {'sequence' if axis == 1 else 'channel'} axis "
            f"for {name}; got {spec}."
        )


# --------------------------------------------------------------------------------------
# custom_vjp around the Pallas kernels.
#
# JAX must never autodiff *through* a pallas_call, and -- much more to the point here --
# we specifically do not want reverse-mode AD to invent the backward. The whole cost of
# this op lives in the backward, so the backward is hand-written.
# --------------------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _short_conv_pallas_local(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    block_sizes: ShortConvBlockSizes,
    exact_reference_rounding: bool,
) -> Float[Array, "B S C"]:
    return short_conv_pallas_fwd_local(
        weight,
        x,
        segment_ids,
        block_sizes=block_sizes,
        exact_reference_rounding=exact_reference_rounding,
    )


def _short_conv_pallas_local_fwd(weight, x, segment_ids, block_sizes, exact_reference_rounding):
    out = short_conv_pallas_fwd_local(
        weight,
        x,
        segment_ids,
        block_sizes=block_sizes,
        exact_reference_rounding=exact_reference_rounding,
    )
    # Residuals are the primal inputs only: `x` is needed for dw, `segment_ids` for both
    # masks, `weight` for dx. Nothing shifted or masked is saved -- that is the 4.84 GB of
    # fp32 scratch the XLA backward allocates and this one does not.
    return out, (weight, x, segment_ids)


def _short_conv_pallas_local_bwd(block_sizes, exact_reference_rounding, residuals, dy):
    weight, x, segment_ids = residuals
    dx, dw_partials = short_conv_pallas_bwd_local(
        weight,
        x,
        segment_ids,
        dy,
        block_sizes=block_sizes,
        exact_reference_rounding=exact_reference_rounding,
    )
    dw = jnp.sum(dw_partials, axis=0).astype(weight.dtype)
    return dw, dx, None  # segment_ids is integer metadata: no cotangent


_short_conv_pallas_local.defvjp(_short_conv_pallas_local_fwd, _short_conv_pallas_local_bwd)


def _sequence_shard_axes(array: jax.Array, mesh) -> tuple[str, ...]:
    """Mesh axes the sequence dim of ``array`` is actually split over.

    An entry naming a length-1 axis splits nothing, so it does not count: that mesh keeps the
    shard-local path unchanged.
    """
    sharding = jax.typeof(array).sharding if isinstance(array, jax.core.Tracer) else array.sharding
    spec = getattr(sharding, "spec", None)
    if spec is None or len(spec) < 2 or spec[1] is None:
        return ()
    entry = spec[1]
    names = (entry,) if isinstance(entry, str) else tuple(entry)
    sizes = _mesh_sizes(mesh)
    return tuple(name for name in names if sizes.get(name, 1) > 1)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


#: Segment id the first context shard reads for its (empty) left halo. It has to be the same
#: sentinel ``short_conv_reference`` pads with, and it cannot be 0: a real document numbered 0
#: would then match across the front of the sequence and keep a tap the unsharded kernel drops.
_FIRST_SHARD_SEGMENT_ID = -1


def _short_conv_sequence_sharded(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    *,
    local_call,
    seq_axis: str,
    batch_axes: tuple[str, ...],
    mesh,
    width: int,
    padded_local_seq: int,
) -> Float[Array, "B S C"]:
    """Run the shard-local kernel over a context-parallel sequence, with a left halo.

    The convolution reaches ``width - 1`` tokens to the left, so a sequence split across the
    context axis needs that many tokens from the left neighbour to reproduce the unsharded result.
    Each shard fetches them with one ``ppermute``, prepends them to its own block, runs the
    unchanged local kernel, and drops the halo outputs. The kernel stays shard-local and keeps
    rejecting a sharded sequence axis; this wrapper is the sanctioned path above it, and it wraps
    the reference and Pallas bodies alike.

    The first shard reads zeros and segment id ``-1``, which is what ``short_conv_reference`` sees
    before position 0. The zeros alone are not enough: a halo segment id of 0 would match a real
    document numbered 0 and keep a tap the unsharded kernel drops.

    ``padded_local_seq`` right-pads the concatenated block, because the Pallas kernel needs a
    sequence divisible by its block size. A causal convolution never reads to the right, so the
    padding cannot change an output that is kept.
    """
    halo = width - 1
    batch_spec = batch_axes or None
    x_spec = P(batch_spec, seq_axis, None)
    seg_spec = P(batch_spec, seq_axis)

    x = reshard(x, x_spec)
    segment_ids = reshard(segment_ids, seg_spec)
    weight = reshard(weight, P(None, None))

    @functools.partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(None, None), x_spec, seg_spec),
        out_specs=x_spec,
        check_vma=False,
    )
    def _local(weight_local, x_local, segment_ids_local):
        local_seq = x_local.shape[1]
        if halo == 0:
            return local_call(weight_local, x_local, segment_ids_local)
        shards = jax.lax.axis_size(seq_axis)
        # Rank r sends its tail to r+1; rank 0 is nobody's destination, so ppermute zeroes it.
        shift = [(src, src + 1) for src in range(shards - 1)]
        x_halo = jax.lax.ppermute(x_local[:, local_seq - halo :, :], seq_axis, shift)
        seg_halo = jax.lax.ppermute(segment_ids_local[:, local_seq - halo :], seq_axis, shift)
        first = jax.lax.axis_index(seq_axis) == 0
        x_halo = jnp.where(first, jnp.zeros_like(x_halo), x_halo)
        seg_halo = jnp.where(first, jnp.full_like(seg_halo, _FIRST_SHARD_SEGMENT_ID), seg_halo)

        x_block = jnp.concatenate([x_halo, x_local], axis=1)
        seg_block = jnp.concatenate([seg_halo, segment_ids_local], axis=1)
        tail = padded_local_seq - x_block.shape[1]
        if tail:
            x_block = jnp.pad(x_block, ((0, 0), (0, tail), (0, 0)))
            seg_block = jnp.pad(seg_block, ((0, 0), (0, tail)), constant_values=-1)
        return local_call(weight_local, x_block, seg_block)[:, halo : halo + local_seq, :]

    # pyrefly: ignore[bad-argument-count]  # jax.shard_map decorator erases _local's real signature
    return _local(weight, x, segment_ids)


def _short_conv_pallas_sharded(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"],
    *,
    block_sizes: ShortConvBlockSizes,
    exact_reference_rounding: bool,
    batch_axes: Sequence[str],
) -> Float[Array, "B S C"]:
    """Enter an explicit ``shard_map`` before touching the kernel.

    The op is shard-local by construction (depthwise, causal along the sequence, no
    cross-channel term), so the manual region needs no collectives at all. Making the
    boundary explicit stops XLA inferring a sharding for an opaque call and is the house
    rule for every Pallas/FFI kernel in this repo.
    """
    call = functools.partial(
        _short_conv_pallas_local,
        block_sizes=block_sizes,
        exact_reference_rounding=exact_reference_rounding,
    )

    mesh = get_abstract_mesh()
    if mesh is None or mesh.empty:
        return call(weight, x, segment_ids)

    active = _active_batch_axes(mesh, batch_axes)
    if not active:
        return call(weight, x, segment_ids)

    _assert_local_axes("x", x, rank=3, mesh=mesh)
    x = reshard(x, P(active, None, None))
    segment_ids = reshard(segment_ids, P(active, None))
    weight = reshard(weight, P(None, None))

    @functools.partial(shard_map, mesh=mesh, out_specs=P(active, None, None), check_vma=False)
    def _local(weight_local, x_local, segment_ids_local):
        return call(weight_local, x_local, segment_ids_local)

    # pyrefly: ignore[bad-argument-count]  # jax.shard_map decorator erases _local's real signature
    return _local(weight, x, segment_ids)


def short_conv(
    weight: Float[Array, "W C"],
    x: Float[Array, "B S C"],
    segment_ids: Int[Array, "B S"] | None = None,
    *,
    implementation: Implementation | Sequence[Implementation] | None = None,
    block_sizes: ShortConvBlockSizes | None = None,
    exact_reference_rounding: bool = True,
    batch_axes: Sequence[str] = DEFAULT_BATCH_AXES,
) -> Float[Array, "B S C"]:
    """Depthwise causal 1-D convolution over the sequence axis.

    ``out[b,t,c] = sum_lag weight[lag,c] * x[b,t-lag,c]``, with taps that would reach into
    a previous packed document dropped and positions before the sequence start read as
    zero. Shard-local: no collectives, no cross-channel term.

    Args:
      weight: ``[kernel_size, channels]`` taps; ``weight[0]`` is the current token.
      x: ``[batch, seq_len, channels]`` activations.
      segment_ids: ``[batch, seq_len]`` packed-document ids, or None for an unpacked batch.
      implementation: a single name (fail fast if unsupported) or an ordered sequence to
        try in turn. Defaults to the Pallas kernel on GPU, the reference elsewhere.
      block_sizes: GPU tile configuration.
      exact_reference_rounding: keep the reference's per-op bf16 rounding, which makes the
        forward and ``dx`` bit-identical to ``short_conv_reference``. Setting False keeps a
        single fp32 accumulator across taps -- more accurate, not bit-comparable.
      batch_axes: mesh axes the batch is sharded over.
    """
    if weight.dtype != x.dtype:
        # The reference promotes mixed dtypes via standard JAX rules (fp32 for fp32 weight /
        # bf16 x) while the Pallas kernel outputs x.dtype, so mixed inputs would give
        # backend-dependent dtypes and values. Normalise at the boundary instead.
        raise ValueError(
            f"short_conv requires weight and x to share a dtype; got weight={weight.dtype}, "
            f"x={x.dtype}. Cast to a common dtype before calling."
        )

    block_sizes = block_sizes or ShortConvBlockSizes.get_default()
    requested = _as_sequence(implementation)
    explicit_single = isinstance(implementation, str)

    # A context-parallel sequence takes the halo wrapper, which prepends `kernel_size - 1` tokens
    # from the left neighbour and then calls the same shard-local body on a whole block. The shape
    # every backend has to support is that block, not the caller's global sequence.
    mesh = get_abstract_mesh()
    seq_axes = () if mesh is None or mesh.empty else _sequence_shard_axes(x, mesh)
    if len(seq_axes) > 1:
        raise ValueError(
            f"short_conv supports one sequence-sharding mesh axis; got {seq_axes}. The halo is a "
            "single left ppermute and has no meaning over a product of axes."
        )
    halo = weight.shape[0] - 1
    local_shape = x.shape
    padded_local_seq = x.shape[1]
    if seq_axes:
        shards = _mesh_sizes(mesh)[seq_axes[0]]
        if x.shape[1] % shards:
            raise ValueError(f"seq_len {x.shape[1]} is not divisible by the '{seq_axes[0]}' axis ({shards})")
        padded_local_seq = _round_up(x.shape[1] // shards + halo, block_sizes.s_block_size)
        local_shape = (x.shape[0], padded_local_seq, x.shape[2])

    errors: list[str] = []
    for name in requested:
        if name == "reference":
            if not seq_axes:
                return short_conv_reference(weight, x, segment_ids)
            seg = segment_ids if segment_ids is not None else jnp.zeros(x.shape[:2], jnp.int32)
            return _short_conv_sequence_sharded(
                weight,
                x,
                seg,
                local_call=short_conv_reference,
                seq_axis=seq_axes[0],
                batch_axes=_active_batch_axes(mesh, batch_axes),
                mesh=mesh,
                width=weight.shape[0],
                # The reference has no block constraint, so the block is exactly halo + local.
                padded_local_seq=x.shape[1] // _mesh_sizes(mesh)[seq_axes[0]] + halo,
            )
        if name != "pallas_gpu":
            raise ValueError(f"Unknown short_conv implementation {name!r}")

        reason = None
        if not pallas_short_conv_available():
            reason = "Pallas Triton backend unavailable or not running on a GPU"
        else:
            reason = short_conv_shapes_supported(weight.shape, local_shape, block_sizes)
        if reason is not None:
            if explicit_single:
                raise RuntimeError(f"short_conv implementation 'pallas_gpu' is unusable: {reason}")
            errors.append(f"pallas_gpu: {reason}")
            warnings.warn(f"short_conv falling back from 'pallas_gpu' ({reason})", stacklevel=2)
            continue

        seg = segment_ids
        if seg is None:
            # One kernel handles both cases; a constant segment id makes every tap valid.
            # [batch, seq_len] int32 is ~0.03% of the activation traffic at hero shape.
            seg = jnp.zeros(x.shape[:2], jnp.int32)
        if seq_axes:
            return _short_conv_sequence_sharded(
                weight,
                x,
                seg,
                local_call=functools.partial(
                    _short_conv_pallas_local,
                    block_sizes=block_sizes,
                    exact_reference_rounding=exact_reference_rounding,
                ),
                seq_axis=seq_axes[0],
                batch_axes=_active_batch_axes(mesh, batch_axes),
                mesh=mesh,
                width=weight.shape[0],
                padded_local_seq=padded_local_seq,
            )
        return _short_conv_pallas_sharded(
            weight,
            x,
            seg,
            block_sizes=block_sizes,
            exact_reference_rounding=exact_reference_rounding,
            batch_axes=batch_axes,
        )

    raise RuntimeError("No usable short_conv implementation: " + "; ".join(errors))
