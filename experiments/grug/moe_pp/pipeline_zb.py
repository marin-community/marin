# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Device-group microbatched pipeline for the grug-MoE (toward zero-bubble).

Unlike :mod:`experiments.grug.moe_pp.pipeline_manual` (logical stages on one
shared mesh, no overlap), here each stage owns a DISJOINT group of devices, so
different stages run concurrently on different hardware -- the prerequisite for
any pipeline speedup. There is no ``stage`` mesh axis, so the stage-stacked
weight-grad that OOMs the GPU partitioner never forms.

Each stage's forward and backward are compiled ONCE (``jax.jit``) and the Python
scheduler calls those compiled functions per microbatch under the stage's
sub-mesh. Compiled stage calls on disjoint device slices dispatch asynchronously,
so the runtime overlaps them. Activations cross stage boundaries by an explicit
``jax.device_put`` to the next stage's sub-mesh; cotangents flow back the same
way. Each stage's blocks are rematerialized (via :func:`_stage_forward`), so only
stage-boundary activations are held and the backward recomputes block internals.

This module implements the GPipe order (all microbatches forward, then all
backward) with the combined per-stage backward -- the correctness + placement
baseline. The zero-bubble W/B split layers on top of the same jitted stage fns.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.sharding import _GRUG_MESH_AXIS_NAMES

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer
from experiments.grug.moe_pp.pipeline_manual import (
    _embed_forward,
    _embed_head_tuple,
    _head_forward,
    _stage_forward,
)

_REPLICATED = P()


def _stage_submesh(group_devices, *, expert: int, data: int) -> Mesh:
    """A grug sub-mesh ``(stage, replica_dcn, data, expert, model)`` over one device slice."""
    arr = np.array(group_devices, dtype=object).reshape(1, 1, data, expert, 1)
    return Mesh(arr, _GRUG_MESH_AXIS_NAMES, axis_types=tuple(AxisType.Explicit for _ in _GRUG_MESH_AXIS_NAMES))


def _put_params(tree, mesh: Mesh):
    """Replicate a param pytree onto ``mesh`` (fully replicated within the stage slice)."""
    return jax.device_put(tree, NamedSharding(mesh, _REPLICATED))


def _put_act(x: jax.Array, mesh: Mesh) -> jax.Array:
    """Transport an activation/cotangent onto ``mesh`` with the grug batch sharding."""
    return jax.device_put(x, NamedSharding(mesh, grug_model._batch_spec()))


def _make_stage_fns(block_static, masks):
    """Jitted ``(forward, backward)`` for one stage.

    Called by the scheduler under the stage's sub-mesh (``set_mesh`` cannot live
    inside ``jax.jit``). ``forward(params, x) -> (y, z)``;
    ``backward(params, x, dy, dz) -> (dparams, dx)``, the latter recomputing the
    stage forward (block remat) from ``x``.
    """

    @jax.jit
    def forward(params, x):
        return _stage_forward(params, block_static, x, masks)

    @jax.jit
    def backward(params, x, dy, dz):
        _, vjp = jax.vjp(lambda p, h: _stage_forward(p, block_static, h, masks), params, x)
        return vjp((dy, dz))

    return forward, backward


def zb_build(
    transformer: Transformer,
    *,
    num_stages: int,
    num_microbatches: int,
    expert_per_stage: int = 1,
    data_per_stage: int = 1,
):
    """Place params on per-stage sub-meshes and compile the stage fns ONCE.

    Returns a ``step(token_ids, loss_weight) -> (loss, embed_head_grads, block_grads)``
    closure that runs the device-group GPipe schedule, reusing the placed params and
    compiled stage forward/backward across calls. Hoisting this setup out of the step
    is what makes the pipeline timeable (and usable in a real training loop) instead
    of recompiling every iteration.

    Stages own disjoint device slices of ``expert_per_stage * data_per_stage`` devices
    each (so ``num_stages * expert_per_stage * data_per_stage == device_count``).
    """
    devices = jax.devices()
    dps = expert_per_stage * data_per_stage
    if num_stages * dps != len(devices):
        raise ValueError(f"num_stages*{dps} ({num_stages * dps}) must equal device_count ({len(devices)})")
    submeshes = [
        _stage_submesh(devices[s * dps : (s + 1) * dps], expert=expert_per_stage, data=data_per_stage)
        for s in range(num_stages)
    ]

    cfg = transformer.config
    num_layers = cfg.num_layers
    if num_layers % num_stages != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by num_stages={num_stages}")
    lps = num_layers // num_stages
    coef = cfg.router_z_loss_coef

    base_mask = grug_model.AttentionMask.causal()
    short_mask, long_mask = grug_model._layer_attention_masks(base_mask, sliding_window=cfg.sliding_window)
    per_layer_masks = [long_mask if (i % 4 == 3) else short_mask for i in range(num_layers)]
    stage_masks = [tuple(per_layer_masks[s * lps : (s + 1) * lps]) for s in range(num_stages)]

    # --- params: embed/head tuple on stage 0 (embed) + last stage (head); blocks per stage ---
    embed_head = _embed_head_tuple(transformer)
    eh_arrays, eh_static = eqx.partition(embed_head, eqx.is_array)
    eh0 = _put_params(eh_arrays, submeshes[0])
    ehL = _put_params(eh_arrays, submeshes[-1])

    block_static = eqx.partition(transformer.blocks[0], eqx.is_array)[1]
    block_arrays = [eqx.partition(b, eqx.is_array)[0] for b in transformer.blocks]
    stage_params = [_put_params(block_arrays[s * lps : (s + 1) * lps], submeshes[s]) for s in range(num_stages)]

    # --- jitted stage / embed / head fns (compiled once, called per microbatch) ---
    stage_fns = [_make_stage_fns(block_static, stage_masks[s]) for s in range(num_stages)]
    mesh0, meshL = submeshes[0], submeshes[-1]

    @jax.jit
    def embed_fwd(params, tok):
        return _embed_forward(params, eh_static, tok)

    @jax.jit
    def embed_bwd(params, tok, dh):
        _, vjp = jax.vjp(lambda p: _embed_forward(p, eh_static, tok), params)
        return vjp(dh)

    @jax.jit
    def head_fwd(params, h, labels, w):
        return _head_forward(params, eh_static, h, labels, w)

    @jax.jit
    def head_bwd(params, h, labels, w, scale):
        _, vjp = jax.vjp(lambda p, hh: _head_forward(p, eh_static, hh, labels, w), params, h)
        return vjp(scale)

    inv_m = 1.0 / num_microbatches
    dz = jnp.asarray(inv_m * coef / num_layers, jnp.float32)

    def step(token_ids: jax.Array, loss_weight: jax.Array) -> tuple[jax.Array, tuple, list]:
        """Run one GPipe forward+backward over the global batch; returns ``(loss, g_eh, g_blocks)``.

        The loss/grads match the non-pipelined oracle over the same global batch by
        construction (same embed / masks / blocks / head / router z-loss, averaged over
        microbatches), to float reassociation tolerance.
        """
        global_batch = token_ids.shape[0]
        if global_batch % num_microbatches != 0:
            raise ValueError(f"global_batch={global_batch} must divide by num_microbatches={num_microbatches}")
        mb = global_batch // num_microbatches
        weight = loss_weight.astype(jnp.float32)

        # --- forward sweep ---
        tok_mb: list = [None] * num_microbatches
        labels_mb: list = [None] * num_microbatches
        weight_mb: list = [None] * num_microbatches
        saved_x: list = [[None] * num_stages for _ in range(num_microbatches)]
        head_h: list = [None] * num_microbatches
        per_mb_loss: list = [None] * num_microbatches

        for m in range(num_microbatches):
            tok = _put_act(token_ids[m * mb : (m + 1) * mb], mesh0)
            labels = _put_act(jnp.concatenate([tok[:, 1:], tok[:, :1] * 0], axis=1).astype(jnp.int32), meshL)
            w = _put_act(weight[m * mb : (m + 1) * mb], meshL)
            tok_mb[m], labels_mb[m], weight_mb[m] = tok, labels, w

            with set_mesh(mesh0):
                h = embed_fwd(eh0, tok)
            z_stages = []
            for s in range(num_stages):
                if s > 0:
                    h = _put_act(h, submeshes[s])
                saved_x[m][s] = h
                with set_mesh(submeshes[s]):
                    h, z_s = stage_fns[s][0](stage_params[s], h)
                z_stages.append(z_s)
            head_h[m] = h
            with set_mesh(meshL):
                ce = head_fwd(ehL, h, labels, w)
            z_total = jnp.sum(jnp.stack([jax.device_put(z, ce.sharding) for z in z_stages]))
            per_mb_loss[m] = ce + coef * (z_total / num_layers)

        loss = jnp.mean(jnp.stack([jax.device_put(pl, per_mb_loss[0].sharding) for pl in per_mb_loss]))

        # --- backward sweep (GPipe): seed 1/M per microbatch, accumulate grads ---
        g_eh = None
        g_blocks: list = [None] * num_layers
        for m in range(num_microbatches):
            with set_mesh(meshL):
                g_head_m, d_hidden = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
            for s in reversed(range(num_stages)):
                if s < num_stages - 1:
                    d_hidden = _put_act(d_hidden, submeshes[s])
                with set_mesh(submeshes[s]):
                    g_slice, d_hidden = stage_fns[s][1](stage_params[s], saved_x[m][s], d_hidden, dz)
                base = s * lps
                for j, g in enumerate(g_slice):
                    cur = g_blocks[base + j]
                    g_blocks[base + j] = g if cur is None else jax.tree_util.tree_map(jnp.add, cur, g)

            d_hidden = _put_act(d_hidden, mesh0)
            with set_mesh(mesh0):
                (g_embed_m,) = embed_bwd(eh0, tok_mb[m], d_hidden)
            g_head_on0 = jax.device_put(g_head_m, NamedSharding(mesh0, _REPLICATED))
            g_eh_m = jax.tree_util.tree_map(jnp.add, g_embed_m, g_head_on0)
            g_eh = g_eh_m if g_eh is None else jax.tree_util.tree_map(jnp.add, g_eh, g_eh_m)

        return loss, g_eh, g_blocks

    return step


def zb_value_and_grad(
    transformer: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    num_stages: int,
    num_microbatches: int,
    expert_per_stage: int = 1,
    data_per_stage: int = 1,
) -> tuple[jax.Array, tuple, list]:
    """One-shot ``(loss, embed_head_grads, block_grads)`` (builds + steps once).

    Convenience for tests; a training/perf loop should call :func:`zb_build` once and
    reuse the returned ``step`` across iterations.
    """
    step = zb_build(
        transformer,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        expert_per_stage=expert_per_stage,
        data_per_stage=data_per_stage,
    )
    return step(token_ids, loss_weight)
