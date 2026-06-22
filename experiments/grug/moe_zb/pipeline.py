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


def _tree_add(a: PyTree, b: PyTree) -> PyTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _vocab_parallel_ce(logits_local: jax.Array, targets: jax.Array, denom: float, num_stages: int) -> jax.Array:
    """Next-token cross-entropy with the vocab axis ``V`` sharded over ``stage``.

    Run *inside* a ``shard_map`` (so ``stage`` is manual). ``logits_local`` is this
    stage's ``[B, S, V/S]`` vocab slice and ``targets`` the global ``[B, S]`` token
    ids; ``denom`` matches ``head_loss_fn`` ((seq-1)*batch). The stabilizing max
    must come from ``all_gather`` + ``jnp.max`` + ``stop_gradient`` (``lax.pmax``
    has no shard_map vjp rule). The loss is psum-replicated, but the ``all_gather``
    leaves JAX unable to *infer* that, so the final ``psum(loss)/num_stages`` (a
    no-op on a replicated value) re-asserts replication to the shard_map's
    varying-manual-axis tracker, which lets the wrapping shard_map keep
    ``check_vma=True`` (required so auto data/expert axes still partition inside).
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    v_local = logits_local.shape[-1]
    labels = targets[:, 1:]
    pred = logits_local[:, :-1, :]  # [B, S-1, V_local]
    # Global max over the full (stage-sharded) vocab, for numerical stability; detached.
    allmax = jax.lax.all_gather(pred.max(-1), STAGE_AXIS, axis=0, tiled=False)
    gmax = jax.lax.stop_gradient(jnp.max(allmax, axis=0))  # [B, S-1]
    gsumexp = jax.lax.psum(jnp.exp(pred - gmax[..., None]).sum(-1), STAGE_AXIS)
    logZ = jnp.log(gsumexp) + gmax
    vstart = sid * v_local
    loc = labels - vstart
    in_range = (loc >= 0) & (loc < v_local)
    ll_local = jnp.take_along_axis(pred, jnp.clip(loc, 0, v_local - 1)[..., None], -1)[..., 0]
    label_logit = jax.lax.psum(jnp.where(in_range, ll_local, 0.0), STAGE_AXIS)
    loss = -jnp.sum(label_logit - logZ) / denom
    return jax.lax.psum(loss, STAGE_AXIS) / num_stages


def _vocab_parallel_head_grads(
    head_normalize_fn: Callable[[PyTree, jax.Array], jax.Array],
    head_project_fn: Callable[[PyTree, jax.Array], jax.Array],
    head_params: PyTree,
    hidden: jax.Array,
    targets: jax.Array,
    denom: float,
    num_stages: int,
) -> tuple[jax.Array, PyTree, jax.Array]:
    """Vocab-parallel next-token CE returning ``(loss, g_head, d_hidden)`` directly.

    The zero-bubble backend runs the backward by hand, so it cannot let autodiff
    transpose the vocab collectives: a manual ``jax.vjp`` through the internal psum
    yields a hidden-state cotangent missing the cross-vocab reduce. Instead this
    computes the CE gradient w.r.t. the logits in closed form
    (``softmax_local - onehot_local``) and backprops it through the *collective-free*
    norm/project split. ``output_proj``'s grad is purely local (vocab-sharded); the
    hidden-state cotangent needs the full (sharded) vocab contraction reduced
    across stages, which under ``check_vma=True`` the shard_map inserts for free at
    the replicated ``head_normalize_fn`` boundary (an explicit ``psum`` there would
    double-count). The returned loss is re-asserted as replicated via a psum
    identity so the enclosing shard_map can keep ``check_vma=True``.
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    normed, norm_vjp = jax.vjp(lambda hp, h: head_normalize_fn(hp, h), head_params, hidden)
    logits_local, proj_vjp = jax.vjp(lambda hp, n: head_project_fn(hp, n), head_params, normed)
    v_local = logits_local.shape[-1]
    labels = targets[:, 1:]
    pred = logits_local[:, :-1, :]
    allmax = jax.lax.all_gather(pred.max(-1), STAGE_AXIS, axis=0, tiled=False)
    gmax = jax.lax.stop_gradient(jnp.max(allmax, axis=0))
    shifted = jnp.exp(pred - gmax[..., None])
    gsumexp = jax.lax.psum(shifted.sum(-1), STAGE_AXIS)
    logZ = jnp.log(gsumexp) + gmax
    vstart = sid * v_local
    loc = labels - vstart
    in_range = (loc >= 0) & (loc < v_local)
    loc_clip = jnp.clip(loc, 0, v_local - 1)
    ll_local = jnp.take_along_axis(pred, loc_clip[..., None], -1)[..., 0]
    label_logit = jax.lax.psum(jnp.where(in_range, ll_local, 0.0), STAGE_AXIS)
    loss = -jnp.sum(label_logit - logZ) / denom
    # psum identity re-asserts replication past the all_gather so the enclosing
    # shard_map can keep check_vma=True (auto data/expert axes partition inside).
    loss = jax.lax.psum(loss, STAGE_AXIS) / num_stages

    # CE gradient w.r.t. the (next-token) logits, per local vocab slice.
    softmax_local = shifted * jnp.exp(gmax - logZ)[..., None]  # exp(pred - logZ)
    onehot_local = jnp.where(in_range[..., None], jax.nn.one_hot(loc_clip, v_local, dtype=pred.dtype), 0.0)
    d_pred = (softmax_local - onehot_local) / denom
    # Last position has no next-token label -> zero grad there.
    d_logits = jnp.zeros_like(logits_local).at[:, :-1, :].set(d_pred)

    # The projection contracts over the full (sharded) vocab; under check_vma=True
    # the cross-stage reduce is inserted automatically where d_normed feeds the
    # replicated head_normalize_fn, so no explicit psum here.
    g_head_proj, d_normed = proj_vjp(d_logits)
    g_head_norm, d_hidden = norm_vjp(d_normed)
    # head_project_fn reads only output_proj and head_normalize_fn only final_norm,
    # so each vjp zeros the other's field -> the sum is the full head grad.
    g_head = _tree_add(g_head_proj, g_head_norm)
    return loss, g_head, d_hidden


class Schedule(enum.Enum):
    """Pipeline schedule. GPIPE is all-forward-then-all-backward (large bubble);
    ZERO_BUBBLE splits the backward into B/W and fills the bubble with W."""

    GPIPE = enum.auto()
    ZERO_BUBBLE = enum.auto()


@dataclasses.dataclass(frozen=True)
class PipelineModel:
    """The SPMD-identical model pieces the pipeline drives.

    ``head_loss_fn`` (the CE-fused head) drives the replicated fallback path used
    by the dense parity test. When ``head_logits_fn`` is set (the grug model), the
    pipeline instead runs the *vocab-parallel* head: ``head_logits_fn`` produces a
    ``[B, S, V/S]`` logit slice from the stage-sharded ``output_proj`` and the
    pipeline folds the sharded vocab into one cross-entropy via ``_vocab_parallel_ce``.

    The GPipe oracle differentiates ``head_logits_fn`` via autodiff over the whole
    shard_map. The zero-bubble backend instead needs the vocab-reduce inserted
    *between* the projection and the norm (a manual vjp through the internal vocab
    collectives gives the wrong hidden-state cotangent), so it consumes the
    collective-free ``head_normalize_fn`` / ``head_project_fn`` split and a
    closed-form CE gradient. ``head_vp_specs`` is the ``PartitionSpec`` tree (vocab
    axis over ``stage``) for the head params on the vocab-parallel path; all four
    head fields must be set together.
    """

    embed_fn: Callable[[PyTree, jax.Array], jax.Array]
    stage_fn: Callable[[PyTree, jax.Array], jax.Array]
    head_loss_fn: Callable[[PyTree, jax.Array, jax.Array], jax.Array]
    head_logits_fn: Callable[[PyTree, jax.Array], jax.Array] | None = None
    head_normalize_fn: Callable[[PyTree, jax.Array], jax.Array] | None = None
    head_project_fn: Callable[[PyTree, jax.Array], jax.Array] | None = None
    head_vp_specs: PyTree | None = None


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


def _pipeline_forward_hidden(
    params_stage: PyTree,
    params_embed: PyTree,
    tokens: jax.Array,
    *,
    model: PipelineModel,
    num_stages: int,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    hidden_dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array]:
    """Forward pipeline carrying only hidden states, run *inside* a shard_map.

    Each device holds one stage. At timestep ``t`` stage ``s`` works on microbatch
    ``m = t - s``: stage 0 embeds ``tokens[m]`` (a cheap gather, kept here +
    masked), every other stage consumes the activation ``ppermute`` delivered from
    stage ``s-1`` last step. The head is *not* scored here — the final stage's
    valid output for microbatch ``m`` is written into ``h_final[m]`` so the
    vocab-parallel head scores every microbatch once after the sweep, still inside
    the same shard_map.

    Returns ``(h_final, aux_total)`` — ``h_final`` is ``[num_microbatches,
    *hidden_shape]`` and ``aux_total`` the summed per-stage aux; both are reduced
    onto every stage (``psum``) so the caller reads a stage-replicated value.
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    S = num_stages
    M = num_microbatches
    T = M + S - 1
    fwd_perm = [(i, i + 1) for i in range(S - 1)]
    is_first = sid == 0
    is_last = sid == (S - 1)

    stage_params = jax.tree_util.tree_map(lambda x: x[0], params_stage)

    buf = jnp.zeros(hidden_shape, hidden_dtype)
    aux_total = jnp.zeros((), jnp.float32)
    h_final = jnp.zeros((M, *hidden_shape), hidden_dtype)
    for t in range(T):
        m = t - sid
        valid = (m >= 0) & (m < M)
        m_clip = jnp.clip(m, 0, M - 1)
        tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)
        embedded = model.embed_fn(params_embed, tok_m)
        stage_in = jnp.where(is_first, embedded, buf)
        stage_out, stage_aux = model.stage_fn(stage_params, stage_in)
        # Per-stage aux (e.g. router z-loss) is local to a stage's own layers and
        # summed across stages — count it on every valid slot, not just the last.
        aux_total = aux_total + jnp.where(valid, stage_aux, 0.0)
        # Last stage's valid output for microbatch m is its final hidden; add it
        # into slot m (other stages / invalid slots add zero — each m written once).
        contrib = jnp.where(is_last & valid, stage_out, jnp.zeros_like(stage_out))
        prev = jax.lax.dynamic_index_in_dim(h_final, m_clip, axis=0, keepdims=False)
        h_final = jax.lax.dynamic_update_index_in_dim(h_final, prev + contrib, m_clip, axis=0)
        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    h_final = jax.lax.psum(h_final, STAGE_AXIS)
    aux_total = jax.lax.psum(aux_total, STAGE_AXIS)
    return h_final, aux_total


def _gpipe_vp_loss_body(
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
    """GPipe forward + vocab-parallel head as a single scalar loss, inside shard_map.

    Runs the forward pipeline (collecting each microbatch's psum-replicated
    final-stage hidden), then scores every microbatch *once* with the head whose
    ``output_proj`` is vocab-sharded over ``stage``: each stage computes its
    ``[B, S, V/S]`` logit slice and the vocab-parallel CE folds the sharded vocab
    into one cross-entropy. ``jax.value_and_grad`` over this body yields the head
    grad already sharded over ``stage`` (only cheap ``[B, S-1]`` reductions cross
    stages — no ``[D, V]`` all-reduce).
    """
    assert model.head_logits_fn is not None, "vocab-parallel GPipe body needs head_logits_fn"
    h_final, aux_total = _pipeline_forward_hidden(
        params.stage,
        params.embed,
        tokens,
        model=model,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
        hidden_shape=hidden_shape,
        hidden_dtype=hidden_dtype,
    )
    microbatch, seq = hidden_shape[0], hidden_shape[1]
    d = hidden_shape[2]
    h_reshaped = h_final.reshape(num_microbatches * microbatch, seq, d)
    targets_reshaped = targets.reshape(num_microbatches * microbatch, seq)
    denom = (seq - 1) * num_microbatches * microbatch
    logits_local = model.head_logits_fn(params.head, h_reshaped)
    head_loss = _vocab_parallel_ce(logits_local, targets_reshaped, denom, num_stages)
    return head_loss + aux_total / num_microbatches


def _gpipe_replicated_loss_body(
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
    """GPipe forward + replicated CE-fused head as one scalar loss, inside shard_map.

    Fallback for models without a vocab-parallel head (the dense MSE parity test):
    the head is scored per-timestep on the last stage and masked, so its replicated
    grad reduces over ``stage`` via ``value_and_grad``. Sequences stay whole on
    their stage, so the next-token shift in ``head_loss_fn`` is intact.
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    S = num_stages
    M = num_microbatches
    T = M + S - 1
    fwd_perm = [(i, i + 1) for i in range(S - 1)]
    is_first = sid == 0
    is_last = sid == (S - 1)

    stage_params = jax.tree_util.tree_map(lambda x: x[0], params.stage)

    buf = jnp.zeros(hidden_shape, hidden_dtype)
    total_loss = jnp.zeros((), jnp.float32)
    for t in range(T):
        m = t - sid
        valid = (m >= 0) & (m < M)
        m_clip = jnp.clip(m, 0, M - 1)
        tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)
        tgt_m = jax.lax.dynamic_index_in_dim(targets, m_clip, axis=0, keepdims=False)
        embedded = model.embed_fn(params.embed, tok_m)
        stage_in = jnp.where(is_first, embedded, buf)
        stage_out, stage_aux = model.stage_fn(stage_params, stage_in)
        total_loss = total_loss + jnp.where(valid, stage_aux, 0.0)
        loss_m = model.head_loss_fn(params.head, stage_out, tgt_m)
        total_loss = total_loss + jnp.where(is_last & valid, loss_m, 0.0)
        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    return jax.lax.psum(total_loss, STAGE_AXIS) / M


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
    """(loss, grads) for the GPipe schedule via autodiff. The reference oracle.

    The whole loss — forward pipeline + head — runs inside one ``shard_map`` over
    ``stage`` wrapped in ``jax.value_and_grad``, so the GPipe-schedule backward and
    the head grad fall out of autodiff.

    When ``model.head_logits_fn`` is set (the grug model), the head is
    vocab-parallel: ``output_proj`` is vocab-sharded over ``stage`` (in/out spec
    ``model.head_vp_specs``), each microbatch's final-stage hidden is psummed to
    replicate it, and the ``[D, V]`` projection runs once across stages with the
    grad staying stage-sharded. Otherwise the replicated CE-fused ``head_loss_fn``
    is scored per-timestep on the last stage (head grad replicated, ``out_specs``
    ``P()``).

    ``tokens``/``targets`` are ``[num_microbatches, microbatch, seq]``;
    ``hidden_shape`` is the per-microbatch activation shape ``[microbatch, seq, D]``.
    """
    num_stages = mesh.shape[STAGE_AXIS]
    M = num_microbatches
    stage_spec = P(STAGE_AXIS)
    repl = P()
    vocab_parallel = model.head_logits_fn is not None

    if vocab_parallel:
        head_spec = model.head_vp_specs
        body = _gpipe_vp_loss_body
    else:
        head_spec = repl
        body = _gpipe_replicated_loss_body

    def full_loss(p: PipelineParams) -> jax.Array:
        in_specs = PipelineParams(embed=repl, stage=stage_spec, head=head_spec)
        # check_vma stays True so auto data/expert axes keep partitioning inside;
        # the vocab-parallel loss re-asserts its replication via a psum identity.
        return shard_map(
            lambda pp, tk, tg: body(
                pp,
                tk,
                tg,
                model=model,
                num_stages=num_stages,
                num_microbatches=M,
                hidden_shape=hidden_shape,
                hidden_dtype=hidden_dtype,
            ),
            mesh=mesh,
            in_specs=(in_specs, repl, repl),
            out_specs=repl,
            axis_names=frozenset({STAGE_AXIS}),
        )(p, tokens, targets)

    return jax.value_and_grad(full_loss)(params)


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

    When ``model.head_logits_fn`` is set (the grug model), the head is computed
    *once* after the single forward sweep via :func:`_vocab_parallel_ce` over the
    psum-replicated final-stage hiddens: a ``jax.vjp`` yields the (stage-sharded)
    head grad and the cotangents ``dh_final`` w.r.t. each microbatch's last-stage
    output, which the B pass uses to seed the last stage's ``dy`` — no per-timestep
    head recompute, no two-pass. Otherwise the replicated CE-fused ``head_loss_fn``
    is differentiated per-timestep inside the B pass.
    """
    sid = jax.lax.axis_index(STAGE_AXIS)
    S = num_stages
    M = num_microbatches
    T = M + S - 1
    fwd_perm = [(i, i + 1) for i in range(S - 1)]
    bwd_perm = [(i + 1, i) for i in range(S - 1)]
    is_first = sid == 0
    is_last = sid == (S - 1)
    vocab_parallel = model.head_logits_fn is not None

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
    # Vocab-parallel head: gather each microbatch's last-stage final hidden into
    # h_final[m] (masked; non-last stages / invalid slots add zero) so a single
    # head scores them after the sweep.
    h_final = jnp.zeros((M, *hidden_shape), hidden_dtype)
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
        # stage_fn returns (hidden, aux); vjp closures take a (d_hidden, d_aux) cotangent.
        (stage_out, stage_aux), vjp_x = jax.vjp(lambda x, _sp=stage_params: model.stage_fn(_sp, x), stage_in)
        _, vjp_p = jax.vjp(lambda p, _si=stage_in: model.stage_fn(p, _si), stage_params)

        total_loss = total_loss + jnp.where(valid, stage_aux, 0.0)
        if vocab_parallel:
            contrib = jnp.where(is_last & valid, stage_out, jnp.zeros_like(stage_out))
            prev = jax.lax.dynamic_index_in_dim(h_final, m_clip, axis=0, keepdims=False)
            h_final = jax.lax.dynamic_update_index_in_dim(h_final, prev + contrib, m_clip, axis=0)
        else:
            loss_m = model.head_loss_fn(params.head, stage_out, tgt_m)
            total_loss = total_loss + jnp.where(is_last & valid, loss_m, 0.0)

        vjp_x_by_t[t] = vjp_x
        vjp_p_by_t[t] = vjp_p
        embed_vjp_by_t[t] = embed_vjp
        out_by_t[t] = stage_out
        tgt_by_t[t] = tgt_m
        valid_by_t[t] = valid

        buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

    g_embed = jax.tree_util.tree_map(jnp.zeros_like, params.embed)
    g_head = jax.tree_util.tree_map(jnp.zeros_like, params.head)
    g_stage = jax.tree_util.tree_map(jnp.zeros_like, stage_params)

    # Vocab-parallel head computed once: psum h_final to replicate it on every
    # stage, then the closed-form CE gradient gives the stage-sharded head grad and
    # dh_final (the cotangent w.r.t. each microbatch's last-stage hidden) that seeds
    # the B pass. A plain manual vjp through the vocab collectives would give the
    # wrong hidden cotangent, so _vocab_parallel_head_grads inserts the vocab-reduce
    # by hand (see its docstring).
    dh_final = None
    if vocab_parallel:
        assert model.head_normalize_fn is not None and model.head_project_fn is not None
        microbatch, seq, d = hidden_shape
        targets_reshaped = targets.reshape(M * microbatch, seq)
        denom = (seq - 1) * M * microbatch
        h_final = jax.lax.psum(h_final, STAGE_AXIS)
        loss_head, g_head, dh_reshaped = _vocab_parallel_head_grads(
            model.head_normalize_fn,
            model.head_project_fn,
            params.head,
            h_final.reshape(M * microbatch, seq, d),
            targets_reshaped,
            denom,
            S,
        )
        dh_final = dh_reshaped.reshape(M, microbatch, seq, d)
        total_loss = jax.lax.psum(total_loss, STAGE_AXIS) / M + loss_head
    else:
        total_loss = jax.lax.psum(total_loss, STAGE_AXIS) / M

    # Interleaved zero-bubble backward: one op stream per `schedule.backward_order`
    # that runs B (input-grad, the ppermute critical path) in reverse time and
    # slots each deferred W (weight-grad) into the next B's upstream-ppermute
    # stall. B and W of the same timestep stay ordered (W(t) follows B(t)), so the
    # gradients are identical to the two-phase backward; only execution interleaves.
    dy_by_t: list[Any] = [None] * T
    dbuf = jnp.zeros(hidden_shape, hidden_dtype)

    def b_op(t: int, dbuf: jax.Array) -> jax.Array:
        valid = valid_by_t[t]
        nonlocal g_head, g_embed
        if vocab_parallel:
            # The single head vjp already produced this microbatch's cotangent in
            # dh_final[m]; seed the last stage's dy from it (others use dbuf). Zero
            # out invalid (warmup/cooldown) slots, whose clamped index would
            # otherwise pull a spurious dh_final row into dy.
            m_clip = jnp.clip(t - sid, 0, M - 1)
            dout = jax.lax.dynamic_index_in_dim(dh_final, m_clip, axis=0, keepdims=False)
            dout = jnp.where(valid, dout, jnp.zeros_like(dout))
        else:
            # Cotangent on this microbatch's loss is 1/M, but only the last stage's
            # valid slots actually feed the loss; everything else contributes nothing.
            last_valid = is_last & valid
            loss_w = jnp.where(last_valid, 1.0 / M, 0.0)
            head_vjp = jax.vjp(lambda h, o, _tg=tgt_by_t[t]: model.head_loss_fn(h, o, _tg), params.head, out_by_t[t])[1]
            g_head_t, dout = head_vjp(loss_w)
            g_head = _tree_add(g_head, g_head_t)
        dy = jnp.where(is_last, dout, dbuf)
        dy_by_t[t] = dy

        # Aux (stage-local) is part of the loss on every valid slot; cotangent 1/M.
        aux_cot = jnp.where(valid, 1.0 / M, 0.0)
        (dx,) = vjp_x_by_t[t]((dy, aux_cot))
        # embed grad lives only on stage 0's valid slots.
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

    # embed is replicated, so a vjp w.r.t. it inside shard_map already psums the
    # cotangent across the stage axis: g_embed comes back correct on every device.
    # The head grad is stage-sharded for the vocab-parallel path (out_spec
    # head_vp_specs) and replicated for the fallback; either way the single head
    # vjp / the replicated per-timestep vjp leaves it correct, no manual psum.
    # stage params are sharded over `stage`, so each device keeps its own stage
    # grad; re-add the leading shard axis.
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
    vocab_parallel = model.head_logits_fn is not None
    # Vocab-parallel: output_proj is vocab-sharded over stage on both the in (each
    # stage holds [D, V/S]) and out (the head grad stays stage-sharded).
    head_spec = model.head_vp_specs if vocab_parallel else repl

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
    # (data/expert for FSDP/EP) stay auto so GSPMD partitions the model inside;
    # check_vma stays True for that, with the vocab-parallel loss re-asserting its
    # replication via a psum identity.
    return shard_map(
        body,
        mesh=mesh,
        in_specs=(PipelineParams(embed=repl, stage=stage_spec, head=head_spec), repl, repl),
        out_specs=(repl, PipelineParams(embed=repl, stage=stage_spec, head=head_spec)),
        axis_names=frozenset({STAGE_AXIS}),
    )(params, tokens, targets)
