# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the AMD June model's weight gradients are reduced across devices, and where ZeRO-1 gathers its weights.

The model and the trainer choose when each cross-device reduction runs by marking weights ``reduced`` over the batch
axes: a marked weight's gradient is ``unreduced`` (each device's partial sum), and the reduction waits until the
caller reshards it. This module holds the markings and the custom VJPs that move a reduction or a gather into the
layer scan:

- ``mark_replicated_reduced``: with one microbatch, the small replicated block weights (norms, GatedNorm, router) are
  reduced once after the layer scan instead of in every layer;
- ``mark_batch_reduced``: with several microbatches, every weight is reduced once per step, after the last one;
- ``LayerRouting``, read by ``Transformer``'s layer scan: per-layer reductions with the earlier microbatches' sums
  added (``grad_offsets``), reduce-scatters into the ZeRO-1 shard (``grad_sinks``), all-gathers of the ZeRO-1 shard
  into the compute copy (``gather_sinks``, ``gather_buffers``) and gradient sums added in the backward of the scan
  (``grad_sums``).
"""

import functools
import math
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from levanter.grug.sharding import _axis_names, _mesh_axis_size

from experiments.june_tpu_67b_a2b.moe.model import _BATCH_AXES

if TYPE_CHECKING:
    from experiments.amd.june_67b_a2b.model import Block

EXPERT_AXIS = "expert"
REPLICA_AXIS = "replica_dcn"
# The largest replicated block weight, per layer, whose gradient waits for one reduction after the layer scan. Those
# reductions are latency-bound; at d=2560 this covers the norms, GatedNorm, router and attention gate and leaves the
# attention and shared-expert weights, whose reductions overlap the backward, in the scan.
REPLICATED_GRAD_MAX_BYTES = 4 << 20


class LayerRouting(NamedTuple):
    """How ``Transformer``'s layer scan routes the stacked block weights' gradients and gathers, each a stacked
    ``Block`` with None for the weights it leaves alone.

    ``grad_offsets``: float32 sums of the earlier microbatches' gradients as ``unreduced_to_local(sum, leading=1)``;
    each layer's backward adds its gradient to its slice and reduces the total at once (``_reduce_with_offset``).
    ``grad_sinks``: zeros in a sharding finer than the weight's (ZeRO-1), through whose gradient that reduction leaves
    as a reduce-scatter (``_reduce_into_sink``). ``gather_sinks`` and ``gather_buffers`` (both or neither): each layer
    all-gathers its slice of the weight, cast to the sink's dtype, to the sink's sharding (``_gather_into_sink``) and
    writes it into the buffer, a full-size stack the scan carries and returns. ``grad_sums``: float32 sums like
    ``grad_offsets`` that the scan carries and whose backward adds each layer's gradient in place
    (``_grad_into_sum``); differentiate with ``jax.vjp``, seeding the returned sums' cotangent with the sums.
    """

    grad_offsets: "Block | None" = None
    grad_sinks: "Block | None" = None
    gather_sinks: "Block | None" = None
    gather_buffers: "Block | None" = None
    grad_sums: "Block | None" = None


def _is_router_bias(path) -> bool:
    return any(getattr(k, "name", None) == "router_bias" for k in path)


def _without_unit_axes(spec: P) -> P | None:
    """``spec`` with size-1 mesh axes dropped, or None when it shards over a batch axis of size > 1."""
    mesh = get_abstract_mesh()
    out = []
    for entry in spec:
        axes = tuple(a for a in _axis_names(entry) if _mesh_axis_size(mesh, a) > 1)
        if set(axes) & set(_BATCH_AXES):
            return None
        out.append(axes[0] if len(axes) == 1 else (axes or None))
    return P(*out)


def after_scan_reduced_spec(path, x) -> P | None:
    """The ``reduced`` spec ``mark_replicated_reduced`` gives the stacked block weight ``x`` at ``path``, or None if it
    leaves ``x`` alone: not replicated over the batch axes (a size-1 ``expert`` axis counts as replicated), over
    ``REPLICATED_GRAD_MAX_BYTES`` per layer, already marked by the caller, or ``router_bias``, which has no gradient."""
    if not isinstance(x, jax.Array) or _is_router_bias(path):
        return None
    if x.size * x.dtype.itemsize // x.shape[0] > REPLICATED_GRAD_MAX_BYTES:  # the leading axis is the layer
        return None
    if jax.typeof(x).sharding.spec.reduced:
        return None
    spec = _without_unit_axes(jax.typeof(x).sharding.spec)
    if spec is None:
        return None
    # Every batch axis, size-1 ones included: a dot's unreduced output names its contracting dimension's axes.
    return P(*spec, reduced=frozenset(_BATCH_AXES))


def mark_replicated_reduced(blocks):
    """Mark the stacked block weights that ``after_scan_reduced_spec`` picks ``reduced`` over the batch axes.

    Their cotangents stay unreduced through the layer scan, and the transpose of this reshard reduces each once, after
    the scan, instead of one all-reduce per weight in every layer. Weights the caller marked already (every weight
    with several microbatches) are left as they are."""

    def mark(path, x):
        spec = after_scan_reduced_spec(path, x)
        return x if spec is None else reshard(x, spec)

    return jax.tree_util.tree_map_with_path(mark, blocks)


def batch_reduced_spec(spec: P) -> P:
    """``spec`` with the batch axes other than ``expert`` dropped from its partitions and every batch axis it is not
    sharded over marked ``reduced`` (``mark_batch_reduced``)."""
    partitions = []
    for entry in spec.partitions:
        kept = tuple(a for a in _axis_names(entry) if a not in _BATCH_AXES or a == EXPERT_AXIS)
        partitions.append(kept[0] if len(kept) == 1 else (kept or None))
    kept_axes = {a for entry in partitions for a in _axis_names(entry)}
    # Every batch axis not kept, size-1 ones included: a dot's unreduced output names its contracting dim's axes.
    return P(*partitions, reduced=frozenset(a for a in _BATCH_AXES if a not in kept_axes))


def mark_batch_reduced(params):
    """Mark every weight ``reduced`` over the batch axes it is not sharded over by ``expert``.

    Its gradient is then ``unreduced`` over them, each device's partial sum over its own tokens, which the caller
    reduces when it chooses. A weight sharded over another batch axis (the embedding and output projection over
    ``replica_dcn`` and ``data``, FSDP weights over ``data``) is all-gathered first, so its gradient is full size on
    every device. The expert weights keep their ``expert`` sharding, since only their own shard's tokens reach them.
    ``router_bias`` is left alone: it has no gradient."""

    def mark(path, x):
        if not isinstance(x, jax.Array) or _is_router_bias(path):
            return x
        return reshard(x, batch_reduced_spec(jax.typeof(x).sharding.spec))

    return jax.tree_util.tree_map_with_path(mark, params)


def batch_shards(mesh) -> int:
    return math.prod(mesh.shape[axis] for axis in _BATCH_AXES)


def zero1_spec(spec: P) -> P:
    """``spec`` with its ``expert`` dimension also split over the replicas, for a weight sharded over ``expert`` and
    not already over the replicas; any other spec unchanged.

    ``replica_dcn`` goes after ``expert``, so it splits each device's expert shard: the replicas that hold the same
    experts (one per node) each keep a part, and the reduce-scatter and all-gather run among them, across nodes only."""
    partitions = [_axis_names(entry) for entry in spec]
    if any(REPLICA_AXIS in axes for axes in partitions):
        return spec
    for i, axes in enumerate(partitions):
        if EXPERT_AXIS in axes:
            partitions[i] = (*axes, REPLICA_AXIS)
            return P(*(axes[0] if len(axes) == 1 else (axes or None) for axes in partitions))
    return spec


def zero1_sharded(params):
    """``params`` with the expert weights sharded over the replicas as well (``zero1_spec``)."""
    return jax.tree.map(
        lambda x: reshard(x, zero1_spec(jax.typeof(x).sharding.spec)) if isinstance(x, jax.Array) else x, params
    )


def is_zero1_shard(x) -> bool:
    """Whether ``x`` has a dimension sharded over both ``expert`` and ``replica_dcn``: a ZeRO-1 shard."""
    if not isinstance(x, jax.Array):
        return False
    return any(EXPERT_AXIS in _axis_names(e) and REPLICA_AXIS in _axis_names(e) for e in jax.typeof(x).sharding.spec)


def resharded_like(tree, like):
    """``tree`` resharded leaf by leaf to the shardings of ``like``, a tree of the same structure."""
    return jax.tree.map(
        lambda x, target: reshard(x, jax.typeof(target).sharding) if isinstance(x, jax.Array) else x, tree, like
    )


def per_layer_reduced(params):
    """True for the stacked block weights sharded over ``expert`` (the expert weights) that the layer scan does not
    mark itself (``after_scan_reduced_spec``, for a size-1 ``expert`` axis): the last microbatch reduces their
    gradients one layer at a time inside its backward. False everywhere else."""

    def per_layer(path, x):
        if not isinstance(x, jax.Array) or EXPERT_AXIS not in jax.typeof(x).sharding.spec:
            return False
        return after_scan_reduced_spec(path, x) is None

    return _stacked_flags(params, per_layer)


def summed_in_scan(params):
    """True for the stacked block weights whose gradient sums the layer scan carries (``LayerRouting.grad_sums``):
    all of them but ``router_bias``, which has no gradient. False everywhere else."""
    return _stacked_flags(params, lambda path, x: isinstance(x, jax.Array) and not _is_router_bias(path))


def _stacked_flags(params, flag):
    """A tree like ``params``: ``flag(path, x)`` for the stacked block weights, False elsewhere."""
    flags = jax.tree.map(lambda _: False, params)
    stacked = jax.tree_util.tree_map_with_path(flag, params.stacked_blocks.stacked)
    return eqx.tree_at(lambda t: t.stacked_blocks.stacked, flags, stacked)


def local_to_unreduced(local: jax.Array, *, leading: int = 0) -> jax.Array:
    """[..., n, ...] with dimension ``leading`` sharded over some mesh axes, each device's own partial sum -> [...]
    ``unreduced`` over them, the same per-device data. The inverse of ``unreduced_to_local``; moves no data."""
    spec = jax.typeof(local).sharding.spec
    parts = tuple(spec.partitions) + (None,) * (local.ndim - len(spec.partitions))
    axes = _axis_names(parts[leading])
    rest = (*parts[:leading], *parts[leading + 1 :])
    return jax.shard_map(
        lambda a: jax.lax.pcast(jnp.squeeze(a, leading), axes, to="unreduced"),
        in_specs=P(*parts[:leading], axes, *parts[leading + 1 :]),
        out_specs=P(*rest, unreduced=frozenset(axes)),
    )(local)


def unreduced_to_local(x: jax.Array, *, leading: int = 0) -> jax.Array:
    """``x`` ``unreduced`` over some mesh axes -> each device's partial sum as one row of a new dimension at
    ``leading``, sharded over those axes: an ordinary array, which can be sliced (no ``dynamic_slice`` rule exists for
    unreduced arrays). Moves no data."""
    spec = jax.typeof(x).sharding.spec
    axes = tuple(a for a in get_abstract_mesh().axis_names if a in spec.unreduced)
    parts = tuple(spec.partitions) + (None,) * (x.ndim - len(spec.partitions))
    return jax.shard_map(
        lambda a: jnp.expand_dims(a, leading),
        in_specs=P(*parts, unreduced=frozenset(axes)),
        out_specs=P(*parts[:leading], axes, *parts[leading:]),
        check_vma=False,
    )(x)


def _offset_total(offset: jax.Array, cotangent: jax.Array) -> jax.Array:
    return (local_to_unreduced(offset) + cotangent.astype(jnp.float32)).astype(cotangent.dtype)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _reduce_with_offset(weight: jax.Array, offset: jax.Array, reduced_spec: P, weight_spec: P) -> jax.Array:
    """``weight`` (one layer's slice, sharded as ``weight_spec``) marked ``reduced`` as ``reduced_spec``; in the
    backward its gradient plus ``offset`` is reduced across devices into ``weight_spec``, inside the backward of the
    layer scan.

    ``offset`` is the float32 sum of the earlier microbatches' unreduced gradients for this layer. The backward adds
    the layer's unreduced gradient to it in float32, casts the total to the gradient's dtype and reshards it to
    ``weight_spec``: an all-reduce over the reduced axes. ``offset`` gets no gradient."""
    del offset, weight_spec
    return reshard(weight, reduced_spec)


def _reduce_with_offset_fwd(weight, offset, reduced_spec, weight_spec):
    del weight_spec
    return reshard(weight, reduced_spec), offset


def _reduce_with_offset_bwd(reduced_spec, weight_spec, offset, cotangent):
    del reduced_spec
    return reshard(_offset_total(offset, cotangent), weight_spec), None


_reduce_with_offset.defvjp(_reduce_with_offset_fwd, _reduce_with_offset_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _reduce_into_sink(weight: jax.Array, offset: jax.Array, sink: jax.Array, reduced_spec: P, sink_spec: P):
    """``_reduce_with_offset`` with the reduced gradient sharded as ``sink_spec``, which shards more than the weight
    does (ZeRO-1: the expert dimension over the replicas as well), so the reduction is a reduce-scatter.

    A custom VJP's backward must return a cotangent of its input's type, so the gradient cannot leave as the weight's:
    it leaves as that of ``sink``, a zero input sharded as ``sink_spec`` that the forward ignores; the weight's
    cotangent is zero. The caller differentiates with respect to the sinks."""
    del offset, sink, sink_spec
    return reshard(weight, reduced_spec)


def _reduce_into_sink_fwd(weight, offset, sink, reduced_spec, sink_spec):
    del sink, sink_spec
    return reshard(weight, reduced_spec), offset


def _reduce_into_sink_bwd(reduced_spec, sink_spec, offset, cotangent):
    del reduced_spec
    return None, None, reshard(_offset_total(offset, cotangent), sink_spec)


_reduce_into_sink.defvjp(_reduce_into_sink_fwd, _reduce_into_sink_bwd)


def _apply_grad_offsets(layer, offsets, sinks):
    """Route the layer weights that have an offset in ``offsets`` (None elsewhere) through ``_reduce_with_offset``, or
    through ``_reduce_into_sink`` where ``sinks`` has a sink for them."""

    def apply(offset, weight, sink=None):
        if offset is None:
            return weight
        # The offset is ``unreduced_to_local`` of the weight's gradient sum: dimension 0 over the axes the weight is
        # marked reduced over, then the marked weight's partitions.
        offset_spec = jax.typeof(offset).sharding.spec
        reduced_spec = P(*offset_spec.partitions[1:], reduced=frozenset(_axis_names(offset_spec.partitions[0])))
        if sink is None:
            return _reduce_with_offset(weight, offset, reduced_spec, jax.typeof(weight).sharding.spec)
        return _reduce_into_sink(weight, offset, sink, reduced_spec, jax.typeof(sink).sharding.spec)

    if sinks is None:
        return jax.tree.map(apply, offsets, layer, is_leaf=lambda x: x is None)
    return jax.tree.map(apply, offsets, layer, sinks, is_leaf=lambda x: x is None)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _gather_into_sink(shard: jax.Array, sink: jax.Array, gathered_spec: P) -> jax.Array:
    """All-gather ``shard``, one layer of a weight sharded over the replicas (ZeRO-1), to ``gathered_spec``.

    ``gathered_spec`` is the weight's sharding marked ``reduced`` over the batch axes, so the gradient is each device's
    unreduced partial sum. The transpose of the gather would reduce-scatter it into the shard at once; instead it
    leaves unreduced through ``sink``, a zero input of the gathered type that the forward ignores, for the caller to
    accumulate and reduce once per step. The shard's own cotangent is zero."""
    del sink
    return reshard(shard, gathered_spec)


def _gather_into_sink_fwd(shard, sink, gathered_spec):
    del sink
    return reshard(shard, gathered_spec), None


def _gather_into_sink_bwd(gathered_spec, _, cotangent):
    del gathered_spec
    return None, cotangent


_gather_into_sink.defvjp(_gather_into_sink_fwd, _gather_into_sink_bwd)


def apply_gathers(layer, sinks, buffers, index: jax.Array):
    """Gather the layer weights that have a sink in ``sinks`` (None elsewhere), cast to the sink's dtype, to the sink's
    sharding with ``_gather_into_sink``, and write each into layer ``index`` of its buffer in ``buffers``, the stacked
    destinations in the model's sharding. Returns the layer with the gathered weights in place, and the buffers. The
    write carries the value only; the gradient leaves through the sink."""

    def gather(sink, weight):
        if sink is None:
            return None
        return _gather_into_sink(weight.astype(sink.dtype), sink, jax.typeof(sink).sharding.spec)

    def write(gathered, buffer):
        if gathered is None:
            return buffer
        # Unmarked, in the sharding of one layer of the buffer: dropping ``reduced`` moves no data, and a weight also
        # sharded over ``data`` (FSDP) keeps this device's slice.
        layer_spec = P(*jax.typeof(buffer).sharding.spec[1:])
        value = reshard(jax.lax.stop_gradient(gathered).astype(buffer.dtype), layer_spec)
        return jax.lax.dynamic_update_index_in_dim(buffer, value, index, 0)

    gathered = jax.tree.map(gather, sinks, layer, is_leaf=lambda x: x is None)
    layer = jax.tree.map(lambda g, w: w if g is None else g, gathered, layer, is_leaf=lambda x: x is None)
    return layer, jax.tree.map(write, gathered, buffers, is_leaf=lambda x: x is None)


@jax.custom_vjp
def _grad_into_sum(weight: jax.Array, total: jax.Array, index: jax.Array) -> tuple[jax.Array, jax.Array]:
    """``(weight, total)`` unchanged; the backward adds ``weight``'s gradient into layer ``index`` of ``total``'s.

    ``weight`` is one layer of a stacked block weight marked ``reduced``, so its gradient is ``unreduced``. ``total``
    is the float32 sum of that stack's gradients as ``unreduced_to_local(sum, leading=1)``, ``[L, S, ...]``. Threaded
    through the layer scan with its cotangent seeded with the running sum, the backward of the scan carries the sum and
    adds each layer's gradient into its row, ``row + gradient.astype(float32)``; ``weight``'s own gradient is zero, so
    no stacked per-microbatch gradient is built."""
    del index
    return weight, total


def _grad_into_sum_fwd(weight, total, index):
    return (weight, total), index


def _grad_into_sum_bwd(index, cotangents):
    weight_ct, total_ct = cotangents
    gradient = unreduced_to_local(weight_ct).astype(total_ct.dtype)
    # A one-row scatter-add rather than a dynamic slice, add and dynamic-update-slice: XLA:GPU fuses a dynamic slice of
    # the updated buffer into the in-place update only within one fusion, so the latter takes a separate read-add
    # kernel; the scatter reads and writes the row in place in one.
    total_ct = total_ct.at[index].add(gradient, indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds")
    return None, total_ct, None


_grad_into_sum.defvjp(_grad_into_sum_fwd, _grad_into_sum_bwd)


def _sum_block_grads(layer, sums, index: jax.Array):
    """Route the layer weights that have a sum in ``sums`` (None elsewhere) through ``_grad_into_sum``: the layer, and
    the sums to carry to the next layer."""
    totals, treedef = jax.tree.flatten(sums, is_leaf=lambda x: x is None)
    weights = treedef.flatten_up_to(layer)
    routed = [w if t is None else _grad_into_sum(w, t, index) for w, t in zip(weights, totals, strict=True)]
    layer_out = treedef.unflatten([r if t is None else r[0] for r, t in zip(routed, totals, strict=True)])
    sums_out = treedef.unflatten([None if t is None else r[1] for r, t in zip(routed, totals, strict=True)])
    return layer_out, sums_out


def call_routed(layer, sums, offsets, sinks, index: jax.Array, *args):
    """``layer(*args)`` with its weights in ``sums`` routed through ``_grad_into_sum`` (then the sums appended to the
    outputs) and those in ``offsets`` through ``_reduce_with_offset`` or ``_reduce_into_sink``. Called inside the
    layer's checkpoint, so the checkpoint's inputs are the layer's own slices, which the backward reads in place."""
    if sums is not None:
        layer, sums = _sum_block_grads(layer, sums, index)
    if offsets is not None:
        layer = _apply_grad_offsets(layer, offsets, sinks)
    outputs = layer(*args)
    return outputs if sums is None else (*outputs, sums)
