# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel primitives for grug, with a zero-bubble schedule.

This is the pipeline analogue of Levanter's micro-batching
(`levanter.grad_accum.microbatched`): micro-batching splits the global batch
into microbatches and folds a forward+backward over them on one set of devices;
here the same microbatch loop is *distributed across pipeline stages* so each
stage owns a contiguous slice of the model's layers and the microbatches stream
through the stages.

Two backends compute the same gradients:

- ``pipeline_value_and_grad`` runs the forward pipeline inside a ``shard_map``
  over a ``stage`` mesh axis (activations shipped stage->stage+1 with
  ``jax.lax.ppermute``) and lets ``jax.value_and_grad`` build the backward. The
  reverse-mode transpose of the forward ppermute ships cotangents stage->stage-1,
  so this is a correct GPipe schedule. It is the reference oracle.

- ``zero_bubble_value_and_grad`` runs an explicit schedule that splits each
  stage's backward into B (gradient w.r.t. the stage input, which unblocks the
  upstream stage) and W (gradient w.r.t. the stage weights, which blocks
  nothing), and slots the W work into the warmup/cooldown bubbles. It must match
  the oracle's gradients bit-for-tolerance.

The model enters as three SPMD-identical callables so a single program runs on
every stage:

- ``embed_fn(embed_params, token_microbatch) -> hidden`` (used on stage 0),
- ``stage_fn(stage_params, hidden_in) -> hidden_out`` (every stage),
- ``head_loss_fn(head_params, hidden, target_microbatch) -> scalar`` (stage N-1).

``stage_params`` carries a leading ``layers_per_stage`` axis; ``stage_fn``
applies that stage's blocks in order. ``embed_params``/``head_params`` are
replicated across stages (only stages 0 / N-1 consume them) so the single
program type-checks everywhere.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import PartitionSpec as P

STAGE_AXIS = "stage"

PyTree = Any


class Schedule(enum.Enum):
    """Pipeline schedule. GPIPE is all-forward-then-all-backward (large bubble);
    ZERO_BUBBLE splits the backward into B/W and fills the bubble with W."""

    GPIPE = enum.auto()
    ZERO_BUBBLE = enum.auto()


@dataclasses.dataclass(frozen=True)
class PipelineModel:
    """The three SPMD-identical model pieces the pipeline drives."""

    embed_fn: Callable[[PyTree, jax.Array], jax.Array]
    stage_fn: Callable[[PyTree, jax.Array], jax.Array]
    head_loss_fn: Callable[[PyTree, jax.Array, jax.Array], jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PipelineParams:
    """Pipeline-partitioned parameters.

    ``stage`` has a leading ``num_stages`` axis sharded over the stage mesh axis;
    ``embed``/``head`` are replicated (consumed only on the first / last stage).
    """

    embed: PyTree
    stage: PyTree
    head: PyTree


def _forward_pipeline_loss(
    params: PipelineParams,
    tokens: jax.Array,
    targets: jax.Array,
    *,
    model: PipelineModel,
    num_stages: int,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    hidden_dtype: jnp.dtype,
) -> jax.Array:
    """Forward pipeline body, run *inside* a shard_map over the stage axis.

    Each device holds one stage. At timestep ``t`` stage ``s`` works on
    microbatch ``m = t - s``: stage 0 embeds ``tokens[m]``, every other stage
    consumes the activation ppermute delivered from stage ``s-1`` last step, and
    stage ``N-1`` accumulates the loss for microbatch ``m``. Invalid (warmup /
    cooldown) slots are masked to a zero loss contribution, so they produce zero
    gradient and never corrupt a valid slot (whenever stage ``s`` has a valid
    microbatch at ``t``, stage ``s-1`` produced exactly that microbatch at
    ``t-1``).
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    S = num_stages
    M = num_microbatches
    T = M + S - 1
    fwd_perm = [(i, i + 1) for i in range(S - 1)]

    # shard_map shards (does not unbind) the mapped axis: with one stage per
    # device the leading stage-shard axis is size 1. Drop it so stage_fn sees a
    # clean [layers_per_stage, ...] pytree. Autodiff re-expands it for the grad.
    stage_params = jax.tree_util.tree_map(lambda x: x[0], params.stage)

    buf = jnp.zeros(hidden_shape, hidden_dtype)
    total_loss = jnp.zeros((), jnp.float32)

    is_first = sid == 0
    is_last = sid == (S - 1)

    for t in range(T):
        m = t - sid
        valid = (m >= 0) & (m < M)
        m_clip = jnp.clip(m, 0, M - 1)

        tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)
        embedded = model.embed_fn(params.embed, tok_m)
        stage_in = jnp.where(is_first, embedded, buf)

        stage_out = model.stage_fn(stage_params, stage_in)

        tgt_m = jax.lax.dynamic_index_in_dim(targets, m_clip, axis=0, keepdims=False)
        loss_m = model.head_loss_fn(params.head, stage_out, tgt_m)
        total_loss = total_loss + jnp.where(is_last & valid, loss_m, 0.0)

        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    # Only the last stage accumulated loss; broadcast it to every stage.
    total_loss = jax.lax.psum(total_loss, STAGE_AXIS) / M
    return total_loss


def pipeline_value_and_grad(
    params: PipelineParams,
    tokens: jax.Array,
    targets: jax.Array,
    *,
    model: PipelineModel,
    mesh: jax.sharding.Mesh,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    hidden_dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, PipelineParams]:
    """GPipe-schedule (loss, grads) via autodiff through the forward pipeline.

    ``tokens``/``targets`` are ``[num_microbatches, microbatch, ...]`` and are
    replicated across stages. ``hidden_shape`` is the per-microbatch activation
    shape that flows between stages (``[microbatch, ...]``).
    """
    num_stages = mesh.shape[STAGE_AXIS]
    stage_spec = P(STAGE_AXIS)
    repl = P()
    in_specs = (
        PipelineParams(embed=repl, stage=stage_spec, head=repl),
        repl,
        repl,
    )

    def value_and_grad_body(p, toks, tgts):
        return jax.value_and_grad(
            lambda pp: _forward_pipeline_loss(
                pp,
                toks,
                tgts,
                model=model,
                num_stages=num_stages,
                num_microbatches=num_microbatches,
                hidden_shape=hidden_shape,
                hidden_dtype=hidden_dtype,
            )
        )(p)

    loss, grads = shard_map(
        value_and_grad_body,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(repl, PipelineParams(embed=repl, stage=stage_spec, head=repl)),
    )(params, tokens, targets)
    return loss, grads


def _tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _zero_bubble_body(
    params: PipelineParams,
    tokens: jax.Array,
    targets: jax.Array,
    *,
    model: PipelineModel,
    num_stages: int,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    hidden_dtype: jnp.dtype,
) -> tuple[jax.Array, PipelineParams]:
    """Explicit forward + split-backward pipeline, run inside a shard_map.

    The backward is split into two passes that the schedule decouples:

    - **B pass** (reverse time): per timestep compute ``dx`` (cotangent w.r.t.
      the stage input) and ppermute it upstream. This is the critical path that
      unblocks stage ``s-1``; it also seeds the embed gradient on stage 0.
    - **W pass**: replay the stored output cotangents through ``vjp_p`` to get the
      weight gradient ``dp``. W depends on nothing downstream, so it is fully
      deferred here (and on hardware is slotted into the warmup/cooldown bubbles).

    B and W use *separate* vjp closures so the deferred W is genuine compute, not
    a buffered add — that decoupling is what makes the bubble fillable.
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    S = num_stages
    M = num_microbatches
    T = M + S - 1
    fwd_perm = [(i, i + 1) for i in range(S - 1)]
    bwd_perm = [(i + 1, i) for i in range(S - 1)]
    is_first = sid == 0
    is_last = sid == (S - 1)

    stage_params = jax.tree_util.tree_map(lambda x: x[0], params.stage)

    # Forward sweep: keep per-timestep vjp closures + outputs (static-keyed by t).
    vjp_x_by_t: list[Any] = [None] * T
    vjp_p_by_t: list[Any] = [None] * T
    embed_vjp_by_t: list[Any] = [None] * T
    out_by_t: list[Any] = [None] * T
    tgt_by_t: list[Any] = [None] * T
    valid_by_t: list[Any] = [None] * T

    buf = jnp.zeros(hidden_shape, hidden_dtype)
    total_loss = jnp.zeros((), jnp.float32)
    for t in range(T):
        m = t - sid
        valid = (m >= 0) & (m < M)
        m_clip = jnp.clip(m, 0, M - 1)
        tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)
        tgt_m = jax.lax.dynamic_index_in_dim(targets, m_clip, axis=0, keepdims=False)

        # Loop vars are bound as lambda defaults: jax.vjp traces each immediately,
        # so the closures capture this iteration's tensors (B023-safe).
        embedded, embed_vjp = jax.vjp(lambda e, _t=tok_m: model.embed_fn(e, _t), params.embed)
        stage_in = jnp.where(is_first, embedded, buf)
        stage_out, vjp_x = jax.vjp(lambda x, _sp=stage_params: model.stage_fn(_sp, x), stage_in)
        _, vjp_p = jax.vjp(lambda p, _si=stage_in: model.stage_fn(p, _si), stage_params)

        loss_m = model.head_loss_fn(params.head, stage_out, tgt_m)
        total_loss = total_loss + jnp.where(is_last & valid, loss_m, 0.0)

        vjp_x_by_t[t] = vjp_x
        vjp_p_by_t[t] = vjp_p
        embed_vjp_by_t[t] = embed_vjp
        out_by_t[t] = stage_out
        tgt_by_t[t] = tgt_m
        valid_by_t[t] = valid

        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    total_loss = jax.lax.psum(total_loss, STAGE_AXIS) / M

    # Backward B pass (reverse time): dx upstream + head/embed grads + store dy.
    g_embed = jax.tree_util.tree_map(jnp.zeros_like, params.embed)
    g_head = jax.tree_util.tree_map(jnp.zeros_like, params.head)
    dy_by_t: list[Any] = [None] * T
    dbuf = jnp.zeros(hidden_shape, hidden_dtype)
    for t in reversed(range(T)):
        valid = valid_by_t[t]
        # Cotangent on this microbatch's loss is 1/M, but only the last stage's
        # valid slots actually feed the loss; everything else contributes nothing.
        last_valid = is_last & valid
        loss_w = jnp.where(last_valid, 1.0 / M, 0.0)
        head_vjp = jax.vjp(lambda h, o, _tg=tgt_by_t[t]: model.head_loss_fn(h, o, _tg), params.head, out_by_t[t])[1]
        g_head_t, dout = head_vjp(loss_w)
        dy = jnp.where(is_last, dout, dbuf)
        dy_by_t[t] = dy
        g_head = _tree_add(g_head, g_head_t)

        (dx,) = vjp_x_by_t[t](dy)
        # embed grad lives only on stage 0's valid slots.
        first_valid = is_first & valid
        (g_embed_t,) = embed_vjp_by_t[t](jnp.where(first_valid, dx, jnp.zeros_like(dx)))
        g_embed = _tree_add(g_embed, g_embed_t)
        dbuf = jax.lax.ppermute(dx, STAGE_AXIS, bwd_perm)

    # Backward W pass (deferred): replay stored cotangents through vjp_p.
    g_stage = jax.tree_util.tree_map(jnp.zeros_like, stage_params)
    for t in range(T):
        (dp,) = vjp_p_by_t[t](dy_by_t[t])
        g_stage = _tree_add(g_stage, dp)

    # embed/head are replicated, so a vjp w.r.t. them inside shard_map already
    # psums the cotangent across the stage axis: g_embed/g_head come back correct
    # on every device, no manual psum. stage params are sharded over `stage`, so
    # each device keeps its own stage grad; re-add the leading shard axis.
    g_stage = jax.tree_util.tree_map(lambda x: x[None], g_stage)
    return total_loss, PipelineParams(embed=g_embed, stage=g_stage, head=g_head)


def zero_bubble_value_and_grad(
    params: PipelineParams,
    tokens: jax.Array,
    targets: jax.Array,
    *,
    model: PipelineModel,
    mesh: jax.sharding.Mesh,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    hidden_dtype: jnp.dtype = jnp.float32,
) -> tuple[jax.Array, PipelineParams]:
    """Zero-bubble-schedule (loss, grads) via an explicit B/W-split backward.

    Gradients must match :func:`pipeline_value_and_grad` (the autodiff oracle).
    """
    num_stages = mesh.shape[STAGE_AXIS]
    stage_spec = P(STAGE_AXIS)
    repl = P()

    def body(p, toks, tgts):
        return _zero_bubble_body(
            p,
            toks,
            tgts,
            model=model,
            num_stages=num_stages,
            num_microbatches=num_microbatches,
            hidden_shape=hidden_shape,
            hidden_dtype=hidden_dtype,
        )

    return shard_map(
        body,
        mesh=mesh,
        in_specs=(PipelineParams(embed=repl, stage=stage_spec, head=repl), repl, repl),
        out_specs=(repl, PipelineParams(embed=repl, stage=stage_spec, head=repl)),
    )(params, tokens, targets)
