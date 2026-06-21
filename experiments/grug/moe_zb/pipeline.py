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

from experiments.grug.moe_zb.schedule import Op, backward_order

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
        # Embed is a cheap gather, so it stays compute-everywhere + masked.
        embedded = model.embed_fn(params.embed, tok_m)
        stage_in = jnp.where(is_first, embedded, buf)

        stage_out, stage_aux = model.stage_fn(stage_params, stage_in)
        # Per-stage aux (e.g. router z-loss) is local to a stage's own layers and
        # summed across stages — count it on every valid slot, not just the last.
        total_loss = total_loss + jnp.where(valid, stage_aux, 0.0)

        tgt_m = jax.lax.dynamic_index_in_dim(targets, m_clip, axis=0, keepdims=False)
        # GPipe is the autodiff oracle: its backward would put the head-grad psum
        # inside a divergent cond (deadlock), so the head stays compute-everywhere +
        # masked here. The head-skip optimization lives in the zero-bubble backend.
        loss_m = model.head_loss_fn(params.head, stage_out, tgt_m)
        total_loss = total_loss + jnp.where(is_last & valid, loss_m, 0.0)

        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    # Sum loss+aux across stages (head loss on last stage, aux on all), mean over microbatches.
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

    # Only `stage` is manual (ppermute runs over it). Any other mesh axes
    # (data/expert for FSDP/EP) stay auto so GSPMD partitions the model inside.
    loss, grads = shard_map(
        value_and_grad_body,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=(repl, PipelineParams(embed=repl, stage=stage_spec, head=repl)),
        axis_names=frozenset({STAGE_AXIS}),
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
    # The head runs on only the last stage (lax.cond), so its vjp is recomputed
    # under cond in the backward rather than stored on every stage.
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
        # so the closures capture this iteration's tensors (B023-safe). Embed is a
        # cheap gather, so it stays compute-everywhere + masked, with its vjp stored.
        embedded, embed_vjp = jax.vjp(lambda e, _t=tok_m: model.embed_fn(e, _t), params.embed)
        stage_in = jnp.where(is_first, embedded, buf)
        # stage_fn returns (hidden, aux); vjp closures take a (d_hidden, d_aux) cotangent.
        (stage_out, stage_aux), vjp_x = jax.vjp(lambda x, _sp=stage_params: model.stage_fn(_sp, x), stage_in)
        _, vjp_p = jax.vjp(lambda p, _si=stage_in: model.stage_fn(p, _si), stage_params)

        total_loss = total_loss + jnp.where(valid, stage_aux, 0.0)
        # Head ([D, V] projection) only on the last stage — the ~2x redundant op.
        loss_m = jax.lax.cond(
            is_last,
            lambda out, tgt: model.head_loss_fn(params.head, out, tgt),
            # A stage-varying zero (cheap reduction, no [D, V] matmul): the cond
            # branches must share the same shard_map vma, and the head loss varies
            # over `stage` through `out`, so the skip branch must too.
            lambda out, tgt: jnp.sum(out) * 0.0,
            stage_out,
            tgt_m,
        )
        total_loss = total_loss + jnp.where(valid, loss_m, 0.0)

        vjp_x_by_t[t] = vjp_x
        vjp_p_by_t[t] = vjp_p
        embed_vjp_by_t[t] = embed_vjp
        out_by_t[t] = stage_out
        tgt_by_t[t] = tgt_m
        valid_by_t[t] = valid

        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    total_loss = jax.lax.psum(total_loss, STAGE_AXIS) / M

    # Interleaved zero-bubble backward: one op stream per `schedule.backward_order`
    # that runs B (input-grad, the ppermute critical path) in reverse time and
    # slots each deferred W (weight-grad) into the next B's upstream-ppermute
    # stall. B and W of the same timestep stay ordered (W(t) follows B(t)), so the
    # gradients are identical to the two-phase backward; only execution interleaves.
    zeros_embed = jax.tree_util.tree_map(jnp.zeros_like, params.embed)
    # Stage-local view of the replicated head params (same value, vma={stage}): a
    # vjp w.r.t. it yields a *stage-local* head grad with no in-cond all-reduce
    # (that would deadlock — only the last stage enters the cond). The local grads
    # are summed across stages by one unconditional psum after the backward loop.
    sv_one = 1.0 + 0.0 * sid.astype(jnp.float32)
    head_sv = jax.tree_util.tree_map(lambda h: h * sv_one, params.head)
    zeros_head_sv = jax.tree_util.tree_map(lambda h: h * (sv_one * 0.0), params.head)
    g_embed = zeros_embed
    g_head_local = zeros_head_sv
    g_stage = jax.tree_util.tree_map(jnp.zeros_like, stage_params)
    dy_by_t: list[Any] = [None] * T
    dbuf = jnp.zeros(hidden_shape, hidden_dtype)

    def b_op(t: int, dbuf: jax.Array) -> jax.Array:
        valid = valid_by_t[t]
        # Cotangent on this microbatch's loss is 1/M; invalid (warmup/cooldown)
        # slots feed nothing. cond restricts the head vjp to the last stage, where
        # the head actually runs — no S-fold redundant [D, V] backward.
        loss_w = jnp.where(valid, 1.0 / M, 0.0)
        # Head backward only on the last stage. The vjp is taken w.r.t. head_sv (the
        # stage-local view), so both outputs are stage-local — no all-reduce inside
        # the cond. dout is the cotangent w.r.t. the per-stage `out`; the skip branch
        # returns stage-varying zeros so the cond branches share the same vma.
        g_head_local_t, dout = jax.lax.cond(
            is_last,
            lambda out, tgt, w: jax.vjp(lambda h, o: model.head_loss_fn(h, o, tgt), head_sv, out)[1](w),
            lambda out, tgt, w: (zeros_head_sv, out * 0.0),
            out_by_t[t],
            tgt_by_t[t],
            loss_w,
        )
        dy = jnp.where(is_last, dout, dbuf)
        dy_by_t[t] = dy
        nonlocal g_head_local, g_embed
        g_head_local = _tree_add(g_head_local, g_head_local_t)

        # Aux (stage-local) is part of the loss on every valid slot; cotangent 1/M.
        aux_cot = jnp.where(valid, 1.0 / M, 0.0)
        (dx,) = vjp_x_by_t[t]((dy, aux_cot))
        # embed grad lives only on stage 0's valid slots (cheap masked vjp).
        first_valid = is_first & valid
        (g_embed_t,) = embed_vjp_by_t[t](jnp.where(first_valid, dx, jnp.zeros_like(dx)))
        g_embed = _tree_add(g_embed, g_embed_t)
        # Pin the B->ppermute->B critical chain so XLA keeps it serialized and lets
        # the dependency-free W ops fill the cross-stage stalls rather than collapsing
        # the interleave back into two monolithic phases.
        dx = jax.lax.optimization_barrier(dx)
        return jax.lax.ppermute(dx, STAGE_AXIS, bwd_perm)

    def w_op(t: int) -> None:
        aux_cot = jnp.where(valid_by_t[t], 1.0 / M, 0.0)
        (dp,) = vjp_p_by_t[t]((dy_by_t[t], aux_cot))
        nonlocal g_stage
        g_stage = _tree_add(g_stage, dp)

    for op, t in backward_order(S, M):
        if op is Op.B:
            dbuf = b_op(t, dbuf)
        else:
            w_op(t)

    # One unconditional all-reduce shares the last stage's (stage-local) head grad
    # to all stages, matching the replicated head params — the collective the cond
    # could not contain. embed is replicated too, so its in-loop vjp already psummed
    # the cotangent across stage (no manual psum). stage params are sharded over
    # `stage`, so each device keeps its own grad; re-add the leading shard axis.
    g_head = jax.lax.psum(g_head_local, STAGE_AXIS)
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

    # Only `stage` is manual (ppermute runs over it). Any other mesh axes
    # (data/expert for FSDP/EP) stay auto so GSPMD partitions the model inside.
    return shard_map(
        body,
        mesh=mesh,
        in_specs=(PipelineParams(embed=repl, stage=stage_spec, head=repl), repl, repl),
        out_specs=(repl, PipelineParams(embed=repl, stage=stage_spec, head=repl)),
        axis_names=frozenset({STAGE_AXIS}),
    )(params, tokens, targets)
