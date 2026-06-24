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

The backward is split (zero-bubble W/B): a chained B-pass computes input-gradients
only (the cheap critical path) and a fanned-out W-pass computes weight-gradients per
(stage, microbatch) independently, so the expensive weight-grad no longer gates each
stage's ``dx``. ``wb_split=False`` falls back to the GPipe combined ``vjp`` backward
for comparison. The forward is still all-microbatches-forward then backward.
"""

from __future__ import annotations

import os
import time

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

# Diagnostic: when MOE_PP_TRACE=1, step() blocks after the forward sweep and again after
# the backward sweep and logs each wall time, to see whether a sweep overlaps stages or runs
# serially. Off by default (the block points would otherwise kill cross-step pipelining).
_TRACE = os.environ.get("MOE_PP_TRACE") == "1"
_t: dict = {}


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


class _StageFns:
    """Jitted forward + the three backward flavours for one stage, compiled once.

    All are called by the scheduler under the stage's sub-mesh (``set_mesh`` cannot
    live inside ``jax.jit``). Each backward recomputes the stage forward (block
    remat) from ``x`` before differentiating.

    * ``forward(params, x) -> (y, z)``
    * ``backward(params, x, dy, dz) -> (dparams, dx)`` -- combined (GPipe baseline)
    * ``b(params, x, dy, dz) -> dx`` -- INPUT-gradient only (the pipeline critical
      path; ``params`` held constant so no weight-grad matmul is computed)
    * ``w(params, x, dy, dz) -> dparams`` -- WEIGHT-gradient only (deferrable off the
      critical path; ``x`` held constant)

    Splitting B from W is the zero-bubble move: the combined backward gates each
    stage's ``dx`` on the expensive weight-grad, serializing the backward chain;
    computing ``dx`` alone keeps the chain cheap and the ``w`` work fills bubbles.
    """

    def __init__(self, block_static, masks, remat: bool = True):
        fwd = lambda params, x: _stage_forward(params, block_static, x, masks, remat)  # noqa: E731

        @jax.jit
        def forward(params, x):
            return fwd(params, x)

        @jax.jit
        def backward(params, x, dy, dz):
            _, vjp = jax.vjp(lambda p, h: fwd(p, h), params, x)
            return vjp((dy, dz))

        @jax.jit
        def b(params, x, dy, dz):
            _, vjp = jax.vjp(lambda h: fwd(params, h), x)
            (dx,) = vjp((dy, dz))
            return dx

        @jax.jit
        def w(params, x, dy, dz):
            _, vjp = jax.vjp(lambda p: fwd(p, x), params)
            (dparams,) = vjp((dy, dz))
            return dparams

        self.forward, self.backward, self.b, self.w = forward, backward, b, w


def zb_build(
    transformer: Transformer,
    *,
    num_stages: int,
    num_microbatches: int,
    expert_per_stage: int = 1,
    data_per_stage: int = 1,
    wb_split: bool = True,
    remat: bool = True,
):
    """Place params on per-stage sub-meshes and compile the stage fns ONCE.

    Returns a ``step(token_ids, loss_weight) -> (loss, embed_head_grads, block_grads)``
    closure that runs the device-group schedule, reusing the placed params and compiled
    stage fns across calls. Hoisting this setup out of the step is what makes the
    pipeline timeable (and usable in a real training loop) instead of recompiling every
    iteration.

    With ``wb_split`` (the zero-bubble path), the backward is split into a B-pass
    (input-gradient, the chained critical path) followed by a W-pass (weight-gradient,
    independent across every microbatch/stage, so it fans out over all devices instead
    of gating each stage's ``dx`` on the expensive weight-grad). With ``wb_split=False``
    the backward is the GPipe combined ``vjp`` -- the correctness baseline.

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
    stage_fns = [_StageFns(block_static, stage_masks[s], remat=remat) for s in range(num_stages)]
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
        if _TRACE:
            _t["start"] = time.perf_counter()

        # --- forward sweep (one chained microbatch at a time; reductions deferred so the
        # hot dispatch loop never stalls feeding the per-stage GPUs) ---
        tok_mb: list = [None] * num_microbatches
        labels_mb: list = [None] * num_microbatches
        weight_mb: list = [None] * num_microbatches
        saved_x: list = [[None] * num_stages for _ in range(num_microbatches)]
        head_h: list = [None] * num_microbatches
        ce_mb: list = [None] * num_microbatches
        z_mb: list = [None] * num_microbatches

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
                    h, z_s = stage_fns[s].forward(stage_params[s], h)
                z_stages.append(z_s)
            head_h[m] = h
            with set_mesh(meshL):
                ce_mb[m] = head_fwd(ehL, h, labels, w)
            z_mb[m] = z_stages

        per_mb_loss = []
        for m in range(num_microbatches):
            z_total = jnp.sum(jnp.stack([jax.device_put(z, ce_mb[m].sharding) for z in z_mb[m]]))
            per_mb_loss.append(ce_mb[m] + coef * (z_total / num_layers))
        loss = jnp.mean(jnp.stack([jax.device_put(pl, per_mb_loss[0].sharding) for pl in per_mb_loss]))

        if _TRACE:
            jax.block_until_ready((loss, head_h, saved_x))
            _t["fwd"] = time.perf_counter()

        # --- backward sweep: seed 1/M per microbatch, accumulate grads ---
        g_eh = None
        g_blocks: list = [None] * num_layers

        def _accum_embed_head(g_embed_m, g_head_m, prev):
            g_head_on0 = jax.device_put(g_head_m, NamedSharding(mesh0, _REPLICATED))
            g_eh_m = jax.tree_util.tree_map(jnp.add, g_embed_m, g_head_on0)
            return g_eh_m if prev is None else jax.tree_util.tree_map(jnp.add, prev, g_eh_m)

        def _accum_blocks(base, g_slice):
            for j, g in enumerate(g_slice):
                cur = g_blocks[base + j]
                g_blocks[base + j] = g if cur is None else jax.tree_util.tree_map(jnp.add, cur, g)

        if not wb_split:
            # GPipe combined backward: each stage's vjp yields (dweights, dx) together,
            # so dx (the chain) waits on the weight-grad. Correctness/perf baseline.
            for m in range(num_microbatches):
                with set_mesh(meshL):
                    g_head_m, d_hidden = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
                for s in reversed(range(num_stages)):
                    if s < num_stages - 1:
                        d_hidden = _put_act(d_hidden, submeshes[s])
                    with set_mesh(submeshes[s]):
                        g_slice, d_hidden = stage_fns[s].backward(stage_params[s], saved_x[m][s], d_hidden, dz)
                    _accum_blocks(s * lps, g_slice)
                d_hidden = _put_act(d_hidden, mesh0)
                with set_mesh(mesh0):
                    (g_embed_m,) = embed_bwd(eh0, tok_mb[m], d_hidden)
                g_eh = _accum_embed_head(g_embed_m, g_head_m, g_eh)
        else:
            # Zero-bubble W/B split. B-pass: chained input-grad only (cheap critical path),
            # saving each stage's output cotangent. W-pass: weight-grad per (stage, microbatch),
            # all independent -> fans out across every device instead of gating the chain.
            saved_dy: list = [[None] * num_stages for _ in range(num_microbatches)]
            g_embed_mb: list = [None] * num_microbatches
            g_head_mb: list = [None] * num_microbatches
            for m in range(num_microbatches):
                with set_mesh(meshL):
                    g_head_mb[m], d_hidden = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
                for s in reversed(range(num_stages)):
                    if s < num_stages - 1:
                        d_hidden = _put_act(d_hidden, submeshes[s])
                    saved_dy[m][s] = d_hidden
                    with set_mesh(submeshes[s]):
                        d_hidden = stage_fns[s].b(stage_params[s], saved_x[m][s], d_hidden, dz)
                d_hidden = _put_act(d_hidden, mesh0)
                with set_mesh(mesh0):
                    (g_embed_mb[m],) = embed_bwd(eh0, tok_mb[m], d_hidden)
            for m in range(num_microbatches):
                g_eh = _accum_embed_head(g_embed_mb[m], g_head_mb[m], g_eh)

            if _TRACE:
                jax.block_until_ready((g_eh, saved_dy))
                _t["bpass"] = time.perf_counter()

            # W-pass: mb-major (round-robin) dispatch so every device gets one call per round and
            # the stages overlap; accumulate AFTER the sweep -- an in-loop tree_map stalls the
            # single Python dispatch thread (flattening big grad pytrees) and serializes the GPUs.
            w_grads: list = [[None] * num_stages for _ in range(num_microbatches)]
            for m in range(num_microbatches):
                for s in range(num_stages):
                    with set_mesh(submeshes[s]):
                        w_grads[m][s] = stage_fns[s].w(stage_params[s], saved_x[m][s], saved_dy[m][s], dz)
            for s in range(num_stages):
                base = s * lps
                acc = list(w_grads[0][s])
                for m in range(1, num_microbatches):
                    acc = [jax.tree_util.tree_map(jnp.add, a, b) for a, b in zip(acc, w_grads[m][s], strict=True)]
                for j, g in enumerate(acc):
                    g_blocks[base + j] = g

        if _TRACE:
            jax.block_until_ready((loss, g_eh, g_blocks))
            now = time.perf_counter()
            fwd, bwd = _t["fwd"] - _t["start"], now - _t["fwd"]
            split = ""
            if wb_split:
                bpass, wpass = _t["bpass"] - _t["fwd"], now - _t["bpass"]
                split = f" [B-pass={bpass * 1e3:.0f}ms W-pass={wpass * 1e3:.0f}ms]"
            print(
                f"TRACE fwd_sweep={fwd * 1e3:.0f}ms bwd_sweep={bwd * 1e3:.0f}ms{split} "
                f"(M={num_microbatches} P={num_stages} wb_split={wb_split})"
            )

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
    wb_split: bool = True,
    remat: bool = True,
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
        wb_split=wb_split,
        remat=remat,
    )
    return step(token_ids, loss_weight)
