# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Microbatched (GPipe) pipeline-parallel training of the PRODUCTION grug-MoE Transformer.

The whole model -- embed, the stacked production ``Block``s, the real sparse ring-EP
``moe_mlp``, and the fused-CE head -- runs inside a SINGLE ``shard_map`` that manualizes
ALL five mesh axes (``stage``, ``replica_dcn``, ``data``, ``expert``, ``model``). The
load-bearing axes are ``stage`` (the pipeline), ``expert`` (expert parallelism), and
``data`` (FSDP / batch sharding); ``replica_dcn`` / ``model`` are size-1 but still
manualized so NO GSPMD axis touches any operand. That is required on TPU: ``ragged_dot``
lowers to the Mosaic/Pallas megablox GMM, which GSPMD cannot auto-partition, and the XLA
SPMD partitioner processes every GSPMD axis for every op (including size-1 ones the
operands are merely replicated over), so even one residual Auto axis trips "Mosaic
kernels cannot be automatically partitioned."

Layout:

- The production ``blocks`` tuple is stacked into a leading-axis pytree and reshaped to
  ``[stage, layers_per_stage, ...]`` (:func:`stack_blocks_for_stages`). The shard_map
  slices the ``stage`` dim; each device squeezes its size-1 stage shard and ``lax.scan``s
  its ``layers_per_stage`` blocks (:func:`_run_stage_blocks`).
- The weights additionally shard the expert dim over ``expert`` (ring EP) and a feature
  dim over ``data`` (ZeRO-3 FSDP); the body :func:`_fsdp_all_gather`s each ``data``-sharded
  leaf whole before use, and the gather's pinned ``psum_scatter`` backward keeps the weight
  grad ``/data``-sharded (the full weight cotangent is never materialized).
- The MoE runs the real sparse ring EP INLINE (:func:`_inline_ring_moe_mlp`:
  ``all_gather`` dispatch + ``ragged_dot`` megablox GMM + ``psum_scatter`` collect over the
  manual ``expert`` axis), with no nested EP shard_map.
- A GPipe microbatch schedule (``T = num_microbatches + num_stages - 1`` steps) ripples the
  microbatches through the stages: stage 0 injects microbatch ``t`` at step ``t``, each
  stage runs its blocks on its current buffer, and the activation is ``ppermute``d
  downstream stage->stage+1. The last stage drains each microbatch's hidden; the fused-CE
  head (:func:`_ep_cross_entropy`) scores them all in one pass after the sweep, so the
  bubble shrinks to ``(S-1)/(M+S-1)``.

The body is differentiated by whole-program ``jax.value_and_grad`` (no manual backward):
the forward ``ppermute`` transposes to a reverse ``ppermute`` and the inline ring's
``all_gather`` / ``psum_scatter`` transpose cleanly because every axis is manual.
"""

from __future__ import annotations

import contextlib
import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.grug import grug_moe
from levanter.grug import loss as grug_loss
from levanter.grug import sharding as grug_sharding
from levanter.grug._moe import ep_ring
from levanter.grug.attention import AttentionMask

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer

EXPERT_AXIS = "expert"
DATA_AXIS = "data"
STAGE_AXIS = "stage"

# The inline ring-EP pipeline manualizes both batch-sharding axes (expert + data) so
# the megablox GMM kernel -- which GSPMD cannot auto-partition -- sees only Manual axes
# on its operands. Token reductions (cross-entropy mean, router z-loss mean) span the
# full batch, which is split across both manual axes, so they psum over both.
_EP_BATCH_AXES = (EXPERT_AXIS, DATA_AXIS)


def stack_blocks_for_stages(transformer: Transformer, num_stages: int) -> tuple[eqx.Module, eqx.Module]:
    """Split a Transformer's block tuple into (stacked-array tree, static tree).

    Returns ``(arrays, static)`` where ``arrays`` has every block-array leaf
    stacked along a leading ``[num_stages, layers_per_stage, ...]`` axis and
    ``static`` carries the (shared) non-array structure. ``eqx.combine(arrays,
    static)`` rebuilds a single ``Block`` whose leaves carry that leading axis;
    indexing the leading dims yields the per-stage / per-layer block.
    """
    num_layers = len(transformer.blocks)
    if num_layers % num_stages != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by num_stages={num_stages}")
    layers_per_stage = num_layers // num_stages

    per_block = [eqx.partition(block, eqx.is_array) for block in transformer.blocks]
    block_arrays = [arrays for arrays, _ in per_block]
    static = per_block[0][1]
    # Stack the per-block array pytrees along a new leading layer axis, then split
    # that axis into (stage, layers_per_stage).
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *block_arrays)
    stacked = jax.tree_util.tree_map(lambda x: x.reshape((num_stages, layers_per_stage, *x.shape[1:])), stacked)
    return stacked, static


def build_layer_masks(transformer: Transformer, num_stages: int, seq_len: int) -> jax.Array:
    """Materialize each layer's attention mask as a boolean ``[Q, K]`` array.

    The production rule selects the long sliding window on every 4th layer
    (``i % 4 == 3``) and the short window otherwise. ``sliding_window`` is static
    structure on ``AttentionMask`` (so two masks have different treedefs and can't
    be ``jnp.where``'d), so we materialize the per-layer choice to a traced array
    that scans alongside the stacked block params. Returns ``[stage,
    layers_per_stage, Q, K]``; the attention path broadcasts it over the batch.
    """
    cfg = transformer.config
    base = AttentionMask.causal()
    short = base.with_sliding_window(cfg.sliding_window // 2).materialize_mask(seq_len, seq_len)
    long = base.with_sliding_window(cfg.sliding_window).materialize_mask(seq_len, seq_len)
    per_layer = [long if (i % 4 == 3) else short for i in range(cfg.num_layers)]
    masks = jnp.stack(per_layer, axis=0)
    layers_per_stage = cfg.num_layers // num_stages
    return masks.reshape((num_stages, layers_per_stage, seq_len, seq_len))


def _run_stage_blocks(
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    hidden: jax.Array,
    stage_masks: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Scan this stage's ``layers_per_stage`` production ``Block``s over ``hidden``.

    ``stage_block_arrays`` has leaves shaped ``[layers_per_stage, ...]`` and
    ``stage_masks`` is ``[layers_per_stage, Q, K]`` (size-1 stage shard already
    squeezed). Returns ``(hidden, router_z_loss_sum)``; the router z-loss is summed
    across this stage's layers so the pipeline can aggregate it across stages.
    """

    def step(carry_hidden: jax.Array, layer: tuple[eqx.Module, jax.Array]) -> tuple[jax.Array, jax.Array]:
        layer_arrays, mask = layer
        block = eqx.combine(layer_arrays, block_static)
        new_hidden, router_stats = block(carry_hidden, mask)
        return new_hidden, router_stats["router_z_loss"].astype(jnp.float32)

    final_hidden, z_losses = jax.lax.scan(step, hidden, (stage_block_arrays, stage_masks))
    return final_hidden, jnp.sum(z_losses)


def _next_token_labels(tokens: jax.Array) -> jax.Array:
    """Left-shift tokens by one for next-token prediction; last position labelled 0."""
    return jnp.concatenate([tokens[:, 1:], tokens[:, :1] * 0], axis=1).astype(jnp.int32)


# --- Real sparse ring-EP inline in the {stage, expert, data}-manual pipeline shard_map
#
# The PRODUCTION ring EP (``_moe_mlp_ep_ring_local``: all_gather dispatch + ragged_dot
# GMM + psum_scatter collect) runs INLINE in the same manual region as the pipeline
# ppermute -- one shard_map manualizing ``stage``, ``expert``, AND ``data``, no nested
# EP shard_map.
#
# Mesh: ALL five axes are Explicit and the outer shard_map manualizes every one. The
# load-bearing axes are ``stage`` (pipeline), ``expert`` (EP), and ``data`` (the
# dispatched tokens carry a ``data``-sharded batch dim); ``replica_dcn`` / ``model``
# are size-1 but still manualized so NO GSPMD axis survives. This is required on TPU:
# ``ragged_dot`` lowers to the Mosaic/Pallas ``_gmm_megablox`` kernel, which GSPMD
# CANNOT auto-partition -- and the XLA SPMD partitioner processes every GSPMD axis for
# every op, including size-1 ones the operands are merely replicated over, so even one
# residual Auto axis trips "Mosaic kernels cannot be automatically partitioned."
# Manualizing all five leaves zero GSPMD axis touching the kernel, so megablox lowers;
# and with no live Explicit axis inside the manual region the xla ragged_dot path
# lowers too. The replicated weights' grad reduce is inserted by the single shard_map's
# transpose: with exactly one shard_map the transpose is clean
# (ppermute<->reverse-ppermute, all_gather<->psum_scatter).


def ep_pipeline_mesh(*, stage: int, expert: int, replica: int, data: int, model: int = 1) -> Mesh:
    """Mesh for the inline ring-EP pipeline: ALL axes Explicit (fully manualized).

    Every axis is ``AxisType.Explicit`` so the outer shard_map can manualize all five.
    The load-bearing axes are ``stage`` (pipeline), ``expert`` (EP), and ``data`` (the
    dispatched tokens are batch-sharded over it); ``replica_dcn`` / ``model`` are size-1
    but still manualized. This is required on TPU: the megablox GMM (``ragged_dot`` ->
    ``_gmm_megablox``) is a Mosaic/Pallas kernel GSPMD cannot auto-partition, and the
    SPMD partitioner processes every GSPMD axis for every op (including size-1 ones the
    operands are merely replicated over), so even one residual Auto axis trips the
    Mosaic auto-partition error. Manualizing all five leaves zero GSPMD axis on the
    kernel; with no live Explicit axis inside, the xla ragged_dot path also lowers.
    """
    shape = (stage, replica, data, expert, model)
    if int(np.prod(shape)) != jax.device_count():
        raise ValueError(f"mesh shape {shape} (prod={int(np.prod(shape))}) must use all {jax.device_count()} devices")
    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    # ALL axes Explicit so the outer shard_map can manualize EVERY axis. The megablox
    # GMM (a Mosaic kernel) cannot survive ANY GSPMD partition attempt -- the XLA SPMD
    # partitioner processes every GSPMD axis for every op, including size-1 ones the
    # operands are merely replicated over, so a single residual Auto axis is enough to
    # trip "Mosaic kernels cannot be automatically partitioned." Manualizing all five
    # leaves zero GSPMD axis touching the kernel. With no live Explicit axis inside the
    # manual region the xla ragged_dot path lowers too (no Explicit-axis sharding-rule
    # error), so this mesh works under both RAGGED_DOT_IMPL=xla and megablox.
    axis_types = (AxisType.Explicit,) * 5
    return Mesh(devices, grug_sharding._GRUG_MESH_AXIS_NAMES, axis_types=axis_types)


def _inline_ring_moe_mlp(
    x,
    selected_experts,
    combine_weights,
    w_up_gate,
    w_down,
    *,
    activation=grug_moe.ActivationFunctionEnum.silu,
    implementation=None,
    mesh=None,
    capacity_factor=grug_moe._DEFAULT_EP_CAPACITY_FACTOR,
    report_capacity_overflow=False,
):
    """Production ring EP run INLINE -- ``_moe_mlp_ep_ring_local`` with no shard_map.

    ``expert`` is already manual in the enclosing pipeline shard_map and the expert
    weights enter sharded over it (each shard holds ``E / expert_size`` experts), so
    the routed path runs directly: ``all_gather(..., "expert")`` reconstitutes the
    token set, ``ragged_dot`` does the grouped expert GMM, ``psum_scatter(...,
    "expert")`` returns each shard's token slice. ``x`` is this (expert, data)-shard's
    local token slice ``[TL, D]``; the all-gather makes it global over ``expert``.

    ``data`` is Manual here too (see :func:`ep_pipeline_mesh`): the megablox GMM the
    ``ragged_dot`` lowers to cannot be GSPMD-partitioned, so no live GSPMD axis may
    touch its token / weight operands. The ``dropped`` diagnostic is discarded; its
    psum is scoped to the manual ``expert`` axis by :func:`_inline_ring_moe_mlp_context`.

    ``w_up_gate`` is this shard's LOCAL slice (``E / expert_size`` experts), so the
    full (static) ``num_experts`` -- which the ring's routing offsets need -- is read
    from the enclosing :func:`_inline_ring_moe_mlp_context`.
    """
    activation_fn = activation.to_jax_fn() if isinstance(activation, grug_moe.ActivationFunctionEnum) else activation
    num_experts = _EP_NUM_EXPERTS[0]
    if num_experts is None:
        raise RuntimeError("_inline_ring_moe_mlp must run inside _inline_ring_moe_mlp_context")
    out, dropped = ep_ring._moe_mlp_ep_ring_local(
        x,
        selected_experts,
        combine_weights,
        w_up_gate,
        w_down,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )
    if report_capacity_overflow:
        return out, dropped
    return out


# The full (static) expert count for the inline ring; set by the context manager
# because inside the manual region ``w_up_gate`` is only this shard's local slice.
_EP_NUM_EXPERTS: list[int | None] = [None]


@contextlib.contextmanager
def _inline_ring_moe_mlp_context(num_experts: int, data_is_manual: bool = True):
    """Wire the production forward to run the real ring EP inline under the EP mesh.

    Patches, all restored on exit:

    1. ``grug_moe.moe_mlp`` -> :func:`_inline_ring_moe_mlp` (skip the nested EP
       shard_map; ``expert`` is already manual).
    2. ``_EP_NUM_EXPERTS`` -> ``num_experts`` (the full count; the local weight slice
       only reveals ``E / expert_size``).
    3. ``ep_ring._batch_axes`` -> ``("expert",)`` so the ring's ``dropped``
       diagnostic ``psum`` reduces over the manual ``expert`` axis only (the ring
       all_gather / psum_scatter operate within each data group, so the per-shard
       drop count is summed over ``expert``; ``data`` is a separate batch group).
    4. The router QB-beta ``shard_map`` -> a local threshold (``qb_beta`` is
       metrics-only; its production ``shard_map`` is invalid under the EP mesh).

    ``data_is_manual`` is ``True`` for the TPU all-manual path (tokens are locally
    sliced, so the QB ``top_k`` runs as-is) and ``False`` for the GPU path (``data`` is
    a GSPMD axis, so the QB body's input token axis is ``data``-sharded -- it is
    replicated over ``data`` before the ``top_k``, which needs that axis unsharded; the
    metric is discarded so the replication is harmless).

    The reshard / ``out_sharding`` neutralization for the inline EP mesh is supplied
    separately by :func:`_neutralize_reshards_auto`.
    """
    moe_original = grug_moe.moe_mlp
    batch_axes_original = ep_ring._batch_axes
    qb_shard_map_original = grug_model.shard_map
    num_experts_original = _EP_NUM_EXPERTS[0]

    def _qb_local_shard_map(fn, *_args, **_kwargs):
        # The only ``grug_model.shard_map`` the forward reaches is the router QB-beta
        # call (the outer pipeline uses ``jax.shard_map`` bound in this module). It
        # manualizes the batch axes and ``pmean``s -- invalid under the EP mesh -- to
        # produce the metrics-only ``qb_beta``. Run its body locally with no collective.
        def _local(*args):
            if not data_is_manual:
                # The QB input is ``data``-sharded on its token axis; the body's ``top_k``
                # requires that axis unsharded, so replicate over ``data`` first.
                args = tuple(reshard(a, P(*((None,) * a.ndim))) for a in args)
            with _pmean_is_identity():
                return fn(*args)

        return _local

    try:
        grug_moe.moe_mlp = _inline_ring_moe_mlp
        _EP_NUM_EXPERTS[0] = num_experts
        ep_ring._batch_axes = lambda _mesh: (EXPERT_AXIS,)
        grug_model.shard_map = _qb_local_shard_map
        yield
    finally:
        grug_moe.moe_mlp = moe_original
        _EP_NUM_EXPERTS[0] = num_experts_original
        ep_ring._batch_axes = batch_axes_original
        grug_model.shard_map = qb_shard_map_original


@contextlib.contextmanager
def _pmean_is_identity():
    """Make ``jax.lax.pmean`` a no-op in scope (for the dead QB-beta local body)."""
    original = jax.lax.pmean
    jax.lax.pmean = lambda x, axis_name=None, **_kw: x
    try:
        yield
    finally:
        jax.lax.pmean = original


# Modules whose module-level ``reshard`` name the production forward reaches:
# ``grug_model`` (direct ``reshard`` + ``_batch_reshard``), ``grug_sharding``
# (``_reshard_for_init`` / ``_reshard_for_shard_map`` / ``unshard`` all call the
# module-level ``reshard``), and ``grug_loss`` (the fused-CE head's reshards).
_RESHARD_PATCHED_MODULES = (grug_model, grug_sharding, grug_loss)


@contextlib.contextmanager
def _neutralize_reshards_auto():
    """Drop the production forward's batch-axis sharding calls for the inline EP mesh.

    Inside the ``{stage, expert, data}``-manual shard_map the model's batch axes
    (``data`` / ``expert``) are Manual and ``replica_dcn`` / ``model`` are size-1 Auto,
    so NO in-body ``reshard`` / ``out_sharding=`` may name the batch axes. This drops
    every in-body ``reshard`` to identity (the local shard already holds its slice) and
    rewrites ``grug_model._batch_spec`` to ``P()`` so the model's ``out_sharding=
    _batch_spec()`` einsums replicate over the surviving Auto axes instead of naming the
    now-manual batch axes. Restored on exit.
    """
    reshard_originals = [(m, m.reshard) for m in _RESHARD_PATCHED_MODULES]
    batch_spec_original = grug_model._batch_spec
    batch_reshard_original = grug_model._batch_reshard

    def _identity_reshard(x, _sharding):
        return x

    try:
        for m, _ in reshard_originals:
            m.reshard = _identity_reshard
        grug_model._batch_spec = lambda: P()
        grug_model._batch_reshard = lambda x: x
        yield
    finally:
        for m, fn in reshard_originals:
            m.reshard = fn
        grug_model._batch_spec = batch_spec_original
        grug_model._batch_reshard = batch_reshard_original


def _stage_in_specs(stage_block_arrays: eqx.Module) -> eqx.Module:
    """In-specs for the stacked block arrays: ZeRO-3 FSDP over ``data`` + EP over ``expert``.

    Every block leaf is sharded over ``stage`` (leading dim). On top of that:

    - The expert MLP weights ``mlp.expert_mlp.w_gate_up`` ``[stage, layers, E, D, I2]``
      and ``...w_down`` ``[stage, layers, E, I, D]`` shard their expert dim (array axis
      2) over the manual ``expert`` axis (so each shard holds ``E / expert_size``
      experts, as :func:`_moe_mlp_ep_ring_local` expects) AND shard a feature dim over
      ``data`` for FSDP -- ``I2`` (axis 4) of ``w_gate_up`` and ``I`` (axis 3) of
      ``w_down``.
    - Every other large block weight (attention projections, the shared dense MLP, the
      router, and the gated-norm factor matrices) shards its largest feature dim over
      ``data``.

    The body :func:`_fsdp_all_gather` reconstructs each weight over ``data`` before use;
    the autodiff transpose turns that all-gather into a reduce-scatter, so the returned
    weight grads (hence the optimizer state) stay sharded ``/data`` -- true ZeRO-3.
    Per-device weight+optimizer memory then scales as ``total/(stage*expert*data)``.
    """
    default_spec = P(STAGE_AXIS)
    specs = jax.tree_util.tree_map(lambda _: default_spec, stage_block_arrays)

    # Stacked arrays carry a leading [stage, layers_per_stage, ...]; the feature axis to
    # FSDP-shard over ``data`` is named by absolute position in the stacked array.
    def _set(specs_tree, accessor, spec):
        return eqx.tree_at(accessor, specs_tree, spec, is_leaf=lambda x: x is None)

    # Expert MLP: expert dim (axis 2) over ``expert``; a non-expert feature dim over ``data``.
    expert_mlp = specs.mlp.expert_mlp
    expert_mlp = eqx.tree_at(
        lambda m: (m.w_gate_up, m.w_down),
        expert_mlp,
        (
            P(STAGE_AXIS, None, EXPERT_AXIS, None, DATA_AXIS),  # w_gate_up [stage,layers,E,D,I2] -> shard I2
            P(STAGE_AXIS, None, EXPERT_AXIS, DATA_AXIS, None),  # w_down    [stage,layers,E,I,D] -> shard I
        ),
    )
    specs = eqx.tree_at(lambda s: s.mlp.expert_mlp, specs, expert_mlp)

    # Router [stage,layers,D,E] -> shard D (axis 2); router_bias [stage,layers,E] stays stage-only (tiny).
    specs = _set(specs, lambda s: s.mlp.router, P(STAGE_AXIS, None, DATA_AXIS, None))

    # Attention projections [stage,layers,*,*] -> shard the output feature dim over ``data``.
    specs = _set(specs, lambda s: s.attn.w_q, P(STAGE_AXIS, None, None, DATA_AXIS))
    specs = _set(specs, lambda s: s.attn.w_k, P(STAGE_AXIS, None, None, DATA_AXIS))
    specs = _set(specs, lambda s: s.attn.w_v, P(STAGE_AXIS, None, None, DATA_AXIS))
    specs = _set(specs, lambda s: s.attn.w_o, P(STAGE_AXIS, None, None, DATA_AXIS))
    specs = _set(specs, lambda s: s.attn.attn_gate, P(STAGE_AXIS, None, DATA_AXIS, None))

    # Gated norms (attn + mlp): w_down [stage,layers,D,R] -> shard D; w_up [stage,layers,R,D] -> shard D.
    for gn in (lambda s: s.attn_gated_norm, lambda s: s.mlp_gated_norm):
        specs = _set(specs, lambda s, _gn=gn: _gn(s).w_down, P(STAGE_AXIS, None, DATA_AXIS, None))
        specs = _set(specs, lambda s, _gn=gn: _gn(s).w_up, P(STAGE_AXIS, None, None, DATA_AXIS))

    # Shared dense MLP: all three matrices shard the shared-intermediate dim over ``data``.
    if specs.shared is not None:
        specs = _set(specs, lambda s: s.shared.w_gate, P(STAGE_AXIS, None, None, DATA_AXIS))
        specs = _set(specs, lambda s: s.shared.w_up, P(STAGE_AXIS, None, None, DATA_AXIS))
        specs = _set(specs, lambda s: s.shared.w_down, P(STAGE_AXIS, None, DATA_AXIS, None))

    return specs


def _data_gather_axis(spec: P) -> int | None:
    """Absolute array axis a ``P`` shards over ``data``, or ``None`` if it does not."""
    for axis, name in enumerate(spec):
        if name == DATA_AXIS:
            return axis
    return None


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _fsdp_gather_leaf(leaf: jax.Array, axis: int) -> jax.Array:
    """All-gather one ``data``-sharded weight leaf whole over the manual ``data`` axis.

    The backward is pinned to an explicit reduce-scatter (see the custom VJP below):
    autodiff through the surrounding ``check_vma=False`` shard_map does NOT recognise
    this all-gather's transpose as a reduce-scatter, so without the pin XLA
    materialises the full (un-``/data``-sharded) weight cotangent -- the buffer that
    OOMs the 40B on H100. The pin reduce-scatters the cotangent over ``data`` directly,
    so the weight grad stays ``/data``-sharded (ZeRO-3) and the full cotangent is never
    built. ``all_gather`` forward / ``psum_scatter`` backward is the exact mathematical
    transpose, so grads are unchanged.
    """
    return jax.lax.all_gather(leaf, DATA_AXIS, axis=axis, tiled=True)


def _fsdp_gather_leaf_fwd(leaf: jax.Array, axis: int):
    return _fsdp_gather_leaf(leaf, axis), None


def _fsdp_gather_leaf_bwd(axis: int, _residual, cotangent: jax.Array):
    return (jax.lax.psum_scatter(cotangent, DATA_AXIS, scatter_dimension=axis, tiled=True),)


_fsdp_gather_leaf.defvjp(_fsdp_gather_leaf_fwd, _fsdp_gather_leaf_bwd)


def _fsdp_all_gather(arrays: eqx.Module, specs: eqx.Module) -> eqx.Module:
    """All-gather each ``data``-sharded leaf over ``data`` so the body sees the full weight.

    ``arrays`` are this stage's local stacked block leaves (still carrying the leading
    ``[stage, ...]`` shard) and ``specs`` their in-specs from :func:`_stage_in_specs`.
    For every leaf whose spec names ``data`` on a feature axis, gather that axis over the
    manual ``data`` axis with ``tiled=True`` to rebuild the full weight (the per-expert
    GMM and the dense ops need it whole). Leaves with no ``data`` in their spec pass
    through. The gather's transpose is pinned to a reduce-scatter over ``data`` (see
    :func:`_fsdp_gather_leaf`), so the weight grads come back ``/data``-sharded without
    ever materialising the full weight cotangent -- the ZeRO-3 win.
    """

    def _gather(leaf, spec):
        axis = _data_gather_axis(spec)
        if axis is None:
            return leaf
        return _fsdp_gather_leaf(leaf, axis)

    return jax.tree_util.tree_map(_gather, arrays, specs)


_EMBED_FSDP_SPECS = (
    P(DATA_AXIS, None),  # token_embed [V, D] -> shard V
    P(),  # embed_norm
    P(),  # embed_gated_norm
    P(),  # final_norm
    P(),  # final_gated_norm
    P(None, DATA_AXIS),  # output_proj [D, V] -> shard V
)


def _embed_in_specs(embed_arrays: tuple) -> tuple:
    """FSDP in-specs for the embed/head tuple: shard the vocab dim over ``data``.

    ``token_embed`` ``[V, D]`` and ``output_proj`` ``[D, V]`` are the non-trivial
    embed/head weights (``V`` is large); shard their vocab dim over ``data``. The small
    norm / gated-norm leaves stay replicated. The :func:`_fsdp_all_gather` body call
    rebuilds them before the embed gather / head matmul; the transpose reduce-scatters
    the grads back ``/data``.
    """
    return tuple(
        jax.tree_util.tree_map(lambda _l, _s=group_spec: _s, group)
        for group_spec, group in zip(_EMBED_FSDP_SPECS, embed_arrays, strict=True)
    )


def _is_pspec(x: object) -> bool:
    """Treat a ``PartitionSpec`` as a tree leaf (it is itself a tuple-pytree)."""
    return isinstance(x, P)


def _constrain_to_specs(grads: eqx.Module, specs: eqx.Module, mesh: jax.sharding.Mesh) -> eqx.Module:
    """Reshard each grad leaf to its param's in-spec on the SAME run mesh.

    ``jax.value_and_grad`` of the stage-manual ``shard_map`` returns the weight grads
    stage-REPLICATED -- the transpose of the stage-sharded forward produces a
    ``P(None, ...)`` cotangent on the stage axis even though the param is
    ``P("stage", ...)``. Left as-is, the partitioner only discovers the mismatch when
    optax combines the grad with the stage-sharded param / opt-state, and reshards
    stage-replicated -> stage-sharded by involuntary full rematerialization: it
    materializes the entire stacked weight grad un-sharded on one device. Resharding
    each grad leaf to the param's ``NamedSharding(mesh, spec)`` here -- the same mesh,
    so it is an intra-mesh slice, never a cross-``Mesh`` materialization -- pins the
    target sharding at grad-production time so the grad exits already stage-sharded and
    optax sees matching shardings. ``reshard`` (not ``with_sharding_constraint``) is
    required for the Explicit axes: ``with_sharding_constraint`` acts as an assert that the
    array is ALREADY in the spec, which the replicated grad is not. Any ``AxisType.Auto``
    axis cannot be named by ``reshard``, so its projection is constrained separately with
    ``with_sharding_constraint``; on the all-Explicit pipeline mesh there are no Auto axes,
    so only the ``reshard`` path runs.
    """
    auto_axes = frozenset(name for name, at in zip(mesh.axis_names, mesh.axis_types, strict=True) if at == AxisType.Auto)

    def _constrain(g, spec):
        explicit = P(*(None if name in auto_axes else name for name in spec))
        g = reshard(g, NamedSharding(mesh, explicit))
        if any(name in auto_axes for name in spec):
            auto = P(*(name if name in auto_axes else None for name in spec))
            g = jax.lax.with_sharding_constraint(g, NamedSharding(mesh, auto))
        return g

    return jax.tree_util.tree_map(_constrain, grads, specs, is_leaf=_is_pspec)


def _ep_cross_entropy(
    final_hidden: jax.Array,
    output_proj: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
    *,
    batch_axes: tuple[str, ...],
) -> jax.Array:
    """Mean next-token cross-entropy whose token reduction spans the batch-sharding axes.

    ``final_hidden`` is this shard's local token slice. The cross-entropy is a weighted
    mean over ALL tokens; sum the per-shard NLL and weight, ``psum`` each over the MANUAL
    batch axes (``batch_axes``), then divide -- so every shard returns the full-batch
    mean.

    ``batch_axes`` is ``(expert, data)`` when both are Manual (the TPU all-manual path)
    and ``(expert,)`` when ``data`` is a GSPMD axis (the GPU path): there the residual
    ``data`` reduction of the separate NLL / weight sums is inserted by GSPMD when the
    replicated ``P()`` loss is materialized -- it all-reduces each sum over ``data``
    before the division, so the global weighted mean is exact (the division is applied
    to the two already-data-reduced sums, not per-data-group then averaged).
    """
    logits = jnp.einsum("bsd,dv->bsv", final_hidden, output_proj).astype(jnp.float32)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    label_logit = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = jnp.sum((log_z - label_logit) * weight)
    wsum = jnp.sum(weight)
    nll = jax.lax.psum(nll, batch_axes)
    wsum = jax.lax.psum(wsum, batch_axes)
    return nll / wsum


# --- Microbatched (GPipe-scheduled) inline ring-EP pipeline ---------------------
#
# The loss below splits the batch into ``num_microbatches`` microbatches and runs the
# GPipe schedule: stage 0 injects microbatch ``tick`` at timestep ``tick``, every stage
# runs its blocks on its current buffer each tick, and the activation is ``ppermute``d
# downstream stage->stage+1. The last stage collects each microbatch's drained hidden
# and the head scores them in one fused pass after the sweep, so the bubble shrinks to
# ``(S-1)/(M+S-1)``.
#
# The body is differentiated by whole-program ``jax.value_and_grad`` (no manual
# backward): the forward ppermute transposes to a reverse ppermute and the inline
# ring's all_gather/psum_scatter transpose cleanly because ``expert`` and ``data`` are
# both manual (no live GSPMD axis under the manual region).


def _ep_pipeline_loss_microbatched(
    embed_arrays: eqx.Module,
    stage_block_arrays: eqx.Module,
    embed_static: eqx.Module,
    block_static: eqx.Module,
    transformer: Transformer,
    token_microbatches: jax.Array,
    weight_microbatches: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
    num_microbatches: int,
) -> jax.Array:
    """Microbatched (GPipe) next-token loss with the real ring EP inline; ``{stage, expert, data}`` manual.

    Runs the GPipe microbatch schedule inside the single ``{stage, expert, data}``-manual
    ``shard_map``. ``token_microbatches`` / ``weight_microbatches`` are
    ``[num_microbatches, microbatch, seq]``; each microbatch's tokens shard their
    per-microbatch batch dim over BOTH the manual ``expert`` and ``data`` axes
    (``P(None, (EXPERT_AXIS, DATA_AXIS), None)``). The loss is the full-batch weighted
    next-token CE (one fused score over all microbatches' drained hiddens) plus the
    router z-aux averaged over microbatches.
    """
    num_layers = len(transformer.blocks)
    cfg = transformer.config
    seq_len = token_microbatches.shape[-1]

    layer_masks = build_layer_masks(transformer, num_stages, seq_len)

    stage_spec = P(STAGE_AXIS)
    stage_in_specs = _stage_in_specs(stage_block_arrays)
    embed_in_specs = _embed_in_specs(embed_arrays)
    # Each microbatch's tokens shard their per-microbatch batch axis over both expert
    # and data; the leading num_microbatches axis is replicated.
    token_spec = P(None, _EP_BATCH_AXES, None)

    def body(stage_arrays, embed, masks, tokens, weights):
        sid = jax.lax.axis_index(STAGE_AXIS)
        S = num_stages
        M = num_microbatches
        T = M + S - 1
        fwd_perm = [(i, i + 1) for i in range(S - 1)]
        is_first = sid == 0
        is_last = sid == (S - 1)

        # FSDP: all-gather every ``data``-sharded weight over ``data`` before use; the
        # transpose reduce-scatters the grads back ``/data``.
        stage_arrays = _fsdp_all_gather(stage_arrays, stage_in_specs)
        embed = _fsdp_all_gather(embed, embed_in_specs)

        token_embed, embed_norm, embed_gated_norm, final_norm, final_gated_norm, output_proj = eqx.combine(
            embed, embed_static
        )
        stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays)
        stage_masks = masks[0]

        # Activation buffer for one microbatch's local token slice. ``expert`` and
        # ``data`` are both manual, so ``tokens`` is already this shard's slice -- the
        # local per-microbatch row count is ``tokens.shape[1]`` (= microbatch /
        # (expert_size * data_size)).
        local_microbatch = tokens.shape[1]
        hidden_shape = (local_microbatch, seq_len, cfg.hidden_dim)
        buf = jnp.zeros(hidden_shape, jnp.float32)
        z_total = jnp.zeros((), jnp.float32)
        # Last stage drains microbatch m into h_final[m]; every other stage / invalid
        # slot adds zero, so each microbatch is written exactly once.
        h_final = jnp.zeros((M, *hidden_shape), jnp.float32)

        # Remat the per-stage block-scan: across the T = M+S-1 schedule steps,
        # whole-program value_and_grad would otherwise save every (step x layer) forward
        # (~T * layers_per_stage activations). Recompute each step's stage forward in the
        # backward so activation memory scales with one stage, not the whole schedule.
        run_stage = eqx.filter_checkpoint(_run_stage_blocks)
        for t in range(T):
            m = t - sid
            valid = (m >= 0) & (m < M)
            m_clip = jnp.clip(m, 0, M - 1)
            tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)

            embedded = token_embed[tok_m]
            embedded = embed_gated_norm(embed_norm(embedded))
            stage_in = jnp.where(is_first, embedded, buf)

            stage_out, z_local = run_stage(stage_blocks, block_static, stage_in, stage_masks)
            # Per-stage router z-loss is local to this stage's layers; count it once
            # per microbatch (every valid slot processes a distinct microbatch).
            z_total = z_total + jnp.where(valid, z_local, 0.0)

            contrib = jnp.where(is_last & valid, stage_out, jnp.zeros_like(stage_out))
            prev = jax.lax.dynamic_index_in_dim(h_final, m_clip, axis=0, keepdims=False)
            h_final = jax.lax.dynamic_update_index_in_dim(h_final, prev + contrib, m_clip, axis=0)

            buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

        # Replicate each microbatch's last-stage hidden onto every stage (others held
        # zero), then score the whole global batch in ONE fused CE -- a single weighted
        # mean over all microbatches' tokens, identical to the non-pipelined oracle.
        h_final = jax.lax.psum(h_final, STAGE_AXIS)
        flat_batch = M * local_microbatch
        final_hidden = final_gated_norm(final_norm(h_final.reshape(flat_batch, seq_len, cfg.hidden_dim)))
        tokens_flat = tokens.reshape(flat_batch, seq_len)
        weights_flat = weights.reshape(flat_batch, seq_len)
        labels = _next_token_labels(tokens_flat)
        ce = _ep_cross_entropy(
            final_hidden, output_proj, labels, weights_flat.astype(jnp.float32), batch_axes=_EP_BATCH_AXES
        )

        # Router z-loss: per-(stage,microbatch) it is a token mean over this
        # (expert, data) shard's slice. The full-batch mean is the mean over stages'
        # layers, (expert, data) shards, AND microbatches -- so sum over stages, average
        # over both manual batch axes, and divide by both num_layers and num_microbatches.
        z_total = jax.lax.psum(z_total, STAGE_AXIS)
        z_total = jax.lax.psum(z_total, _EP_BATCH_AXES) / jax.lax.psum(1, _EP_BATCH_AXES)
        aux = cfg.router_z_loss_coef * (z_total / num_layers / num_microbatches)
        return ce + aux

    def _place(x, spec):
        return reshard(x, NamedSharding(mesh, spec))

    stage_block_arrays = jax.tree_util.tree_map(_place, stage_block_arrays, stage_in_specs)
    embed_arrays = jax.tree_util.tree_map(_place, embed_arrays, embed_in_specs)
    layer_masks = _place(layer_masks, stage_spec)
    token_microbatches = _place(token_microbatches, token_spec)
    weight_microbatches = _place(weight_microbatches, token_spec)

    return shard_map(
        body,
        mesh=mesh,
        in_specs=(stage_in_specs, embed_in_specs, stage_spec, token_spec, token_spec),
        out_specs=P(),
        axis_names=frozenset(grug_sharding._GRUG_MESH_AXIS_NAMES),
        check_vma=False,
    )(stage_block_arrays, embed_arrays, layer_masks, token_microbatches, weight_microbatches)


def pipeline_value_and_grad_ep_microbatched(
    transformer: Transformer,
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
    num_microbatches: int,
) -> tuple[jax.Array, eqx.Module, eqx.Module]:
    """``(loss, embed_grads, stage_grads)`` for the microbatched ring-EP inline pipeline.

    Differentiates :func:`_ep_pipeline_loss_microbatched` with whole-program
    ``jax.value_and_grad`` under the inline-ring + reshard-neutralization patches.
    ``token_ids`` / ``loss_weight`` are ``[num_microbatches, microbatch, seq]``; ``mesh``
    must be an :func:`ep_pipeline_mesh`. ``embed_grads`` is the embed/norm/head array tree
    (vocab dims ``/data``-sharded) and ``stage_grads`` the ``[stage, layers_per_stage,
    ...]`` block grad (each stage's shard, sharded ``/(expert*data)``).
    """
    embed_arrays, embed_static = eqx.partition(
        (
            transformer.token_embed,
            transformer.embed_norm,
            transformer.embed_gated_norm,
            transformer.final_norm,
            transformer.final_gated_norm,
            transformer.output_proj,
        ),
        eqx.is_array,
    )

    def loss_fn(embed_arrays, stage_block_arrays):
        return _ep_pipeline_loss_microbatched(
            embed_arrays,
            stage_block_arrays,
            embed_static,
            block_static,
            transformer,
            token_ids,
            loss_weight,
            mesh=mesh,
            num_stages=num_stages,
            num_microbatches=num_microbatches,
        )

    with _inline_ring_moe_mlp_context(transformer.config.num_experts), _neutralize_reshards_auto():
        loss, (g_embed, g_stage) = jax.value_and_grad(loss_fn, argnums=(0, 1))(embed_arrays, stage_block_arrays)
    # The value_and_grad returns the weight grads stage-replicated; pin them to the
    # param in-specs so the optax update never reshards stage-replicated -> stage-sharded
    # by involuntary full rematerialization (see :func:`_constrain_to_specs`).
    g_embed = _constrain_to_specs(g_embed, _embed_in_specs(embed_arrays), mesh)
    g_stage = _constrain_to_specs(g_stage, _stage_in_specs(stage_block_arrays), mesh)
    return loss, g_embed, g_stage
