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

With ``wb_split`` (the zero-bubble path) the whole step runs from one interleaved
schedule (:func:`zb_schedule`): F, B (input-grad) and W (weight-grad) ops wavefront
across the stages, the backward split so deferred W work fills the bubble the last
stages leave while the backward drains to stage 0. ``wb_split=False`` is the GPipe
baseline: a full forward sweep, then a combined-``vjp`` backward (B and W fused, so no
work fills the bubble) -- the reference the wavefront is measured against.
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


# Op kinds in the zero-bubble schedule. F = stage forward, B = input-grad (the
# critical path, threaded upstream), W = weight-grad (deferrable, fills bubbles).
_F, _B, _W = "F", "B", "W"
_PRIORITY = {_B: 0, _F: 1, _W: 2}


def zb_schedule(num_stages: int, num_microbatches: int) -> list[tuple[str, int, int]]:
    """Zero-bubble (ZB-H1) op order: a list of ``(kind, microbatch, stage)`` to dispatch.

    The schedule is the wavefront heuristic. Each stage is a resource that runs one op
    per time slot; an op is eligible once its data deps are met:

    * ``F(m,s)`` needs ``F(m,s-1)`` (the upstream activation),
    * ``B(m,s)`` needs ``F(m,s)`` and ``B(m,s+1)`` (the downstream input-grad),
    * ``W(m,s)`` needs ``B(m,s)`` (it reuses the same cotangent ``B`` consumed).

    Per slot every free stage greedily takes its highest-priority eligible op with
    ``B > F > W``: advance the backward critical path first, otherwise push a forward,
    and only when neither is ready spend the slot on a deferred weight-grad. Because the
    last stage drains its ``B`` early and idles while the backward marches down to stage
    0, its ``W`` work slides into that tail -- the bubble the plain forward-then-B-then-W
    order leaves empty. Splitting ``W`` out of ``B`` is what frees those slots; with
    ``M >> P`` every stage stays busy across F+B+W and the bubble fraction goes to ~0.

    Forwards are issued eagerly (no run-ahead cap), so all ``M`` activations stay live --
    the memory cost the device-group pipeline already pays; this targets the bubble, not
    activation memory (a bounded ZB-2p variant would cap concurrent forwards).
    """

    def deps(op: tuple[str, int, int]) -> list[tuple[str, int, int]]:
        kind, m, s = op
        if kind == _F:
            return [(_F, m, s - 1)] if s > 0 else []
        if kind == _B:
            return [(_F, m, s)] + ([(_B, m, s + 1)] if s < num_stages - 1 else [])
        return [(_B, m, s)]

    remaining = {(k, m, s) for k in (_F, _B, _W) for m in range(num_microbatches) for s in range(num_stages)}
    done: set[tuple[str, int, int]] = set()
    schedule: list[tuple[str, int, int]] = []
    while remaining:
        eligible = [op for op in remaining if all(d in done for d in deps(op))]
        picked = []
        for s in range(num_stages):
            cand = [op for op in eligible if op[2] == s]
            if cand:
                picked.append(min(cand, key=lambda op: (_PRIORITY[op[0]], op[1])))
        if not picked:
            raise RuntimeError("zero-bubble schedule deadlocked (dependency cycle)")
        schedule.extend(picked)
        remaining.difference_update(picked)
        done.update(picked)
    return schedule


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
    schedule = zb_schedule(num_stages, num_microbatches) if wb_split else []

    def step(token_ids: jax.Array, loss_weight: jax.Array) -> tuple[jax.Array, tuple, list]:
        """Run one pipelined forward+backward over the global batch; returns ``(loss, g_eh, g_blocks)``.

        With ``wb_split`` the F/B/W ops follow the zero-bubble ``schedule``; otherwise a
        GPipe forward sweep then combined backward. Either way the loss/grads match the
        non-pipelined oracle over the same global batch by construction (same embed /
        masks / blocks / head / router z-loss, averaged over microbatches), to float
        reassociation tolerance.
        """
        global_batch = token_ids.shape[0]
        if global_batch % num_microbatches != 0:
            raise ValueError(f"global_batch={global_batch} must divide by num_microbatches={num_microbatches}")
        mb = global_batch // num_microbatches
        weight = loss_weight.astype(jnp.float32)
        if _TRACE:
            _t["start"] = time.perf_counter()

        # Per-microbatch inputs on the terminal stages' meshes (tokens feed stage 0;
        # labels/loss-weight feed the head on the last stage).
        tok_mb: list = [None] * num_microbatches
        labels_mb: list = [None] * num_microbatches
        weight_mb: list = [None] * num_microbatches
        for m in range(num_microbatches):
            tok = _put_act(token_ids[m * mb : (m + 1) * mb], mesh0)
            labels_mb[m] = _put_act(jnp.concatenate([tok[:, 1:], tok[:, :1] * 0], axis=1).astype(jnp.int32), meshL)
            weight_mb[m] = _put_act(weight[m * mb : (m + 1) * mb], meshL)
            tok_mb[m] = tok

        # Activation/cotangent buffers, indexed [microbatch][stage]. ``saved_x`` is each
        # stage's block INPUT (embed output for stage 0); ``saved_dy`` the cotangent fed
        # into its backward (head seed for the last stage). Both are kept for every
        # microbatch so the W ops can reuse them whenever the schedule defers them.
        saved_x: list = [[None] * num_stages for _ in range(num_microbatches)]
        saved_dy: list = [[None] * num_stages for _ in range(num_microbatches)]
        head_h: list = [None] * num_microbatches
        ce_mb: list = [None] * num_microbatches
        z_mb: list = [[None] * num_stages for _ in range(num_microbatches)]
        w_grads: list = [[None] * num_stages for _ in range(num_microbatches)]
        g_embed_mb: list = [None] * num_microbatches
        g_head_mb: list = [None] * num_microbatches
        g_eh = None
        g_blocks: list = [None] * num_layers

        def _accum_embed_head(g_embed_m, g_head_m, prev):
            g_head_on0 = jax.device_put(g_head_m, NamedSharding(mesh0, _REPLICATED))
            g_eh_m = jax.tree_util.tree_map(jnp.add, g_embed_m, g_head_on0)
            return g_eh_m if prev is None else jax.tree_util.tree_map(jnp.add, prev, g_eh_m)

        if wb_split:
            # Zero-bubble wavefront: one interleaved dispatch following ``schedule``. F/B/W
            # ops stream across the stages; reductions are deferred so the single Python
            # dispatch thread never stalls (an in-loop tree_map serializes the GPUs).
            for kind, m, s in schedule:
                if kind == _F:
                    if s == 0:
                        with set_mesh(mesh0):
                            saved_x[m][0] = embed_fwd(eh0, tok_mb[m])
                    with set_mesh(submeshes[s]):
                        h_out, z_mb[m][s] = stage_fns[s].forward(stage_params[s], saved_x[m][s])
                    if s < num_stages - 1:
                        saved_x[m][s + 1] = _put_act(h_out, submeshes[s + 1])
                    else:
                        head_h[m] = h_out
                        with set_mesh(meshL):
                            ce_mb[m] = head_fwd(ehL, h_out, labels_mb[m], weight_mb[m])
                elif kind == _B:
                    if s == num_stages - 1:
                        with set_mesh(meshL):
                            g_head_mb[m], dy = head_bwd(ehL, head_h[m], labels_mb[m], weight_mb[m], inv_m)
                        saved_dy[m][s] = dy
                    with set_mesh(submeshes[s]):
                        dx = stage_fns[s].b(stage_params[s], saved_x[m][s], saved_dy[m][s], dz)
                    if s > 0:
                        saved_dy[m][s - 1] = _put_act(dx, submeshes[s - 1])
                    else:
                        with set_mesh(mesh0):
                            (g_embed_mb[m],) = embed_bwd(eh0, tok_mb[m], dx)
                else:  # _W
                    with set_mesh(submeshes[s]):
                        w_grads[m][s] = stage_fns[s].w(stage_params[s], saved_x[m][s], saved_dy[m][s], dz)

            if _TRACE:
                jax.block_until_ready((ce_mb, z_mb, g_head_mb, g_embed_mb, w_grads))
                _t["sched"] = time.perf_counter()

            for m in range(num_microbatches):
                g_eh = _accum_embed_head(g_embed_mb[m], g_head_mb[m], g_eh)
            for s in range(num_stages):
                base = s * lps
                acc = list(w_grads[0][s])
                for m in range(1, num_microbatches):
                    acc = [jax.tree_util.tree_map(jnp.add, a, b) for a, b in zip(acc, w_grads[m][s], strict=True)]
                for j, g in enumerate(acc):
                    g_blocks[base + j] = g
        else:
            # GPipe baseline: full forward sweep, then a combined-vjp backward (B and W
            # fused, so dx waits on the weight-grad and nothing fills the bubble).
            for m in range(num_microbatches):
                with set_mesh(mesh0):
                    h = embed_fwd(eh0, tok_mb[m])
                for s in range(num_stages):
                    if s > 0:
                        h = _put_act(h, submeshes[s])
                    saved_x[m][s] = h
                    with set_mesh(submeshes[s]):
                        h, z_mb[m][s] = stage_fns[s].forward(stage_params[s], h)
                head_h[m] = h
                with set_mesh(meshL):
                    ce_mb[m] = head_fwd(ehL, h, labels_mb[m], weight_mb[m])

            if _TRACE:
                jax.block_until_ready((head_h, saved_x))
                _t["fwd"] = time.perf_counter()

            def _accum_blocks(base, g_slice):
                for j, g in enumerate(g_slice):
                    cur = g_blocks[base + j]
                    g_blocks[base + j] = g if cur is None else jax.tree_util.tree_map(jnp.add, cur, g)

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

        # Loss (deferred reduction; the backward seeds are constants, so this never gates it).
        per_mb_loss = []
        for m in range(num_microbatches):
            z_total = jnp.sum(jnp.stack([jax.device_put(z, ce_mb[m].sharding) for z in z_mb[m]]))
            per_mb_loss.append(ce_mb[m] + coef * (z_total / num_layers))
        loss = jnp.mean(jnp.stack([jax.device_put(pl, per_mb_loss[0].sharding) for pl in per_mb_loss]))

        if _TRACE:
            jax.block_until_ready((loss, g_eh, g_blocks))
            now = time.perf_counter()
            total = (now - _t["start"]) * 1e3
            if wb_split:
                sched = (_t["sched"] - _t["start"]) * 1e3
                print(
                    f"TRACE zb_wavefront dispatch+exec={sched:.0f}ms reduce={total - sched:.0f}ms total={total:.0f}ms "
                    f"(M={num_microbatches} P={num_stages} ops={len(schedule)})"
                )
            else:
                fwd, bwd = (_t["fwd"] - _t["start"]) * 1e3, (now - _t["fwd"]) * 1e3
                print(
                    f"TRACE gpipe fwd_sweep={fwd:.0f}ms bwd_sweep={bwd:.0f}ms total={total:.0f}ms "
                    f"(M={num_microbatches} P={num_stages})"
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
