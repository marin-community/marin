# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel forward of the PRODUCTION grug-MoE Transformer.

This wraps the real ``experiments.grug.moe.model`` (``Transformer``/``Block``,
ring-EP ``moe_mlp``, fused-CE head) in an outer ``shard_map`` that manualizes
only the ``stage`` mesh axis. Inside that stage-manual region the model keeps its
real sharding -- FSDP over ``data``, expert-parallel over ``expert`` (itself a
nested ``shard_map`` that manualizes ``expert``), vocab-TP over ``model``.

The forward lowers and matches the non-pipelined oracle for ``PP x FSDP``,
``PP x EP``, AND ``PP x FSDP x EP`` (``data > 1`` and ``expert > 1`` together).
The toy ``moe_zb`` documented the two-GSPMD-axes-under-a-manual-axis XLA
partitioner crash (``spmd_partitioner_util.cc:497``); it is avoided here because
every ``shard_map`` input is PRE-PLACED on the mesh explicitly (params enter
replicated over ``data``/``expert``, sharded only over ``stage``). Feeding params
with their init-time ``(data, expert)`` weight shardings instead -- so weight
collectives and activation collectives compound across two GSPMD axes under the
manual ``stage`` -- is what trips the partitioner.

Layout:

- The production ``blocks`` tuple is stacked into a leading-axis pytree and
  reshaped to ``[stage, layers_per_stage, ...]``. The outer ``shard_map`` slices
  the ``stage`` dim via ``P("stage", ...)``; each device sees its own
  ``[1, layers_per_stage, ...]`` shard, squeezes the size-1 stage dim, and
  ``lax.scan``s its ``layers_per_stage`` blocks.
- Stage 0 embeds the tokens (masked by ``axis_index("stage") == 0``); every other
  stage consumes the activation ``ppermute``'d from ``stage - 1``.
- The final stage runs the fused-CE head and produces the scalar loss; other
  stages produce a zero loss. A trailing ``psum`` over ``stage`` makes the loss
  stage-replicated.

Only a single forward (one global batch, no microbatching) runs here -- this is a
de-risk that the real model *lowers and forwards* inside the stage-manual
shard_map, not a throughput schedule.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import inspect

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
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer, _batch_spec

EXPERT_AXIS = "expert"
_GRUG_MESH_AXIS_NAMES = ("stage", "replica_dcn", "data", "expert", "model")

STAGE_AXIS = "stage"

# The production model's inner EP / fused-CE / QB shard_maps default to
# ``check_vma=False``. That is fine for the forward and for top-level autodiff (where
# the surviving Explicit data/expert axes get their replicated-weight grad reduce
# inserted by GSPMD). But the pipeline differentiates them BY HAND inside the outer
# ``stage``-manual ``check_vma=False`` shard_map, and a ``check_vma=False`` inner
# shard_map's reverse-mode transpose drops the cross-(data, expert) reduction for its
# replicated weights -- the expert/router grads come back as un-summed per-shard
# partials. Re-tracking VMA on just those inner shard_maps (``check_vma=True``) makes
# the transpose insert the reduce; it is forward-identical and leaves the sharding
# untouched. We scope the override to the gradient trace rather than editing the
# production source.
_VMA_PATCHED_MODULES = (grug_moe, grug_model)


def _force_check_vma_true(fn):
    # The grug modules bind two different shard_map entry points: the canonical
    # ``jax.shard_map`` (takes ``check_vma``) and the deprecated
    # ``jax.experimental.shard_map`` (takes ``check_rep``). Set whichever the wrapped
    # callable accepts to its VMA-tracking value.
    params = inspect.signature(fn).parameters
    vma_kwarg = "check_vma" if "check_vma" in params else ("check_rep" if "check_rep" in params else None)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if vma_kwarg is not None and "stage" not in (kwargs.get("axis_names") or frozenset()):
            kwargs[vma_kwarg] = True
        return fn(*args, **kwargs)

    return wrapped


@contextlib.contextmanager
def _inner_shard_maps_track_vma():
    """Force the production model's inner shard_maps to ``check_vma=True`` in scope.

    Patches the ``shard_map`` name bound in the grug model / EP modules and the
    fused-CE head's ``jax.shard_map`` so their reverse-mode transpose reduces
    replicated-weight cotangents over the Explicit data/expert axes. Restored on exit.
    """
    originals = [(m, m.shard_map) for m in _VMA_PATCHED_MODULES]
    jax_original = jax.shard_map
    try:
        for m, fn in originals:
            m.shard_map = _force_check_vma_true(fn)
        # loss.py calls ``jax.shard_map`` directly.
        jax.shard_map = _force_check_vma_true(jax_original)
        grug_loss.jax.shard_map = jax.shard_map
        yield
    finally:
        for m, fn in originals:
            m.shard_map = fn
        jax.shard_map = jax_original
        grug_loss.jax.shard_map = jax_original


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


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


def pipeline_forward_loss(
    transformer: Transformer,
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
) -> jax.Array:
    """Run the production Transformer forward as a stage-pipelined ``shard_map``.

    ``stage_block_arrays`` is the stacked ``[stage, layers_per_stage, ...]`` block
    array tree from :func:`stack_blocks_for_stages`; ``block_static`` its shared
    static structure. ``transformer`` supplies the (replicated) embed / norm / head
    params. Returns the scalar next-token loss (stage-replicated).
    """
    num_layers = len(transformer.blocks)
    cfg = transformer.config
    seq_len = token_ids.shape[1]

    # Embed / final-norm / head params are replicated across stages (each stage's
    # program references them but only stage 0 / last stage consume them).
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

    layer_masks = build_layer_masks(transformer, num_stages, seq_len)

    stage_spec = P(STAGE_AXIS)
    repl = P()

    # Params enter replicated over the auto (data/expert/model) axes; the stacked
    # block arrays additionally carry the leading `stage` (manual) dim. The model
    # reshards weights to their canonical specs internally where it needs them
    # (e.g. ``moe_mlp`` reshards the expert weights to ``P("expert", ...)`` before
    # its EP shard_map, and the fused-CE head reshards ``output_proj``), so feeding
    # replicated weights keeps the math correct while leaving only the activation
    # reshards to exercise the auto-axis GSPMD partitioning.
    embed_in_specs = jax.tree_util.tree_map(lambda _: repl, embed_arrays)
    stage_in_specs = jax.tree_util.tree_map(lambda _: stage_spec, stage_block_arrays)

    def body(stage_arrays, embed, masks, tokens, weight):
        sid = jax.lax.axis_index(STAGE_AXIS)
        is_first = sid == 0
        is_last = sid == (num_stages - 1)

        token_embed, embed_norm, embed_gated_norm, final_norm, final_gated_norm, output_proj = eqx.combine(
            embed, embed_static
        )

        # Squeeze the size-1 stage shard: [1, layers_per_stage, ...] -> [layers_per_stage, ...].
        stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays)
        stage_masks = masks[0]

        # Track the production model's inter-layer token sharding exactly (so the
        # embed gather here matches what every Block reshards to internally).
        batch_spec = _batch_spec()
        # Stage 0 embeds; later stages receive the ppermute'd activation.
        embedded = token_embed.at[tokens].get(out_sharding=batch_spec)
        embedded = embed_gated_norm(embed_norm(embedded))

        # Single-microbatch GPipe forward: S sequential steps. At step t the stage
        # with sid == t is active: it runs its block-scan on its current input and
        # ships the result to stage t+1 via ppermute. Stage 0 seeds with the embed;
        # other stages start from the buffer the ppermute fills.
        fwd_perm = [(i, i + 1) for i in range(num_stages - 1)]
        buf = jnp.where(is_first, embedded, jnp.zeros_like(embedded))
        z_total = jnp.zeros((), jnp.float32)
        for t in range(num_stages):
            active = sid == t
            stage_out, z_local = _run_stage_blocks(stage_blocks, block_static, buf, stage_masks)
            z_total = z_total + jnp.where(active, z_local, 0.0)
            # Keep the active stage's fresh output; inactive stages hold their buffer.
            buf = jnp.where(active, stage_out, buf)
            if t < num_stages - 1:
                buf = jax.lax.ppermute(buf, STAGE_AXIS, fwd_perm)

        # After S steps the last stage's buffer holds the fully-processed activation.
        final_hidden = final_gated_norm(final_norm(buf))
        labels = jnp.concatenate([tokens[:, 1:], tokens[:, :1] * 0], axis=1).astype(jnp.int32)
        ce = fused_linear_softmax_cross_entropy_loss(
            final_hidden,
            output_proj,
            labels,
            weight=weight.astype(jnp.float32),
            reduction="mean",
            dtype=jnp.float32,
        )
        # The CE lives only on the last stage; sum it once across stages (zero
        # elsewhere). The router z-loss is per-layer and already stage-local, so
        # psum it once over all stages then average over layers -- matching the
        # production `next_token_loss` (sum over layers / num_layers).
        ce_total = jax.lax.psum(jnp.where(is_last, ce, 0.0), STAGE_AXIS)
        z_total = jax.lax.psum(z_total, STAGE_AXIS)
        aux = cfg.router_z_loss_coef * (z_total / num_layers)
        return ce_total + aux

    # Place every input on the mesh per its declared in_spec (shard_map requires the
    # input sharding to match in_specs exactly).
    def _place(x, spec):
        return reshard(x, NamedSharding(mesh, spec))

    stage_block_arrays = jax.tree_util.tree_map(_place, stage_block_arrays, stage_in_specs)
    embed_arrays = jax.tree_util.tree_map(_place, embed_arrays, embed_in_specs)
    layer_masks = _place(layer_masks, stage_spec)
    token_ids = _place(token_ids, repl)
    loss_weight = _place(loss_weight, repl)

    # check_vma=False keeps the surviving (data/expert/model) axes Explicit inside
    # the body, which the production model requires -- it reshards activations with
    # Explicit-axis specs (``out_sharding=``/``reshard``) throughout. With the
    # default check_vma=True those axes become Auto inside the shard_map and the
    # model's Explicit reshards raise a context-mesh mismatch. The cost is that
    # ``jax.value_and_grad`` cannot transpose this shard_map (see the module note in
    # ``check_grad_blocked.py``); the forward lowers and matches the oracle exactly.
    return shard_map(
        body,
        mesh=mesh,
        in_specs=(stage_in_specs, embed_in_specs, stage_spec, repl, repl),
        out_specs=repl,
        axis_names=frozenset({STAGE_AXIS}),
        check_vma=False,
    )(stage_block_arrays, embed_arrays, layer_masks, token_ids, loss_weight)


# --- Gradients: manual GPipe backward inside the stage-manual shard_map -------
#
# ``jax.value_and_grad`` of ``pipeline_forward_loss`` is structurally blocked: the
# production model's in-body Explicit reshards make the outer ``check_vma=False``
# shard_map non-transposable (see ``check_grad_blocked.py``). The fix is to run the
# backward BY HAND inside the same shard_map. ``jax.vjp`` of one stage's forward
# *does* work inside the manual region (the reshards are differentiated in place,
# never transposing the outer shard_map -- proven by ``check_stage_vjp.py``), so we
# capture a per-(stage, microbatch) ``jax.vjp`` closure on the forward sweep and run
# a backward sweep that seeds the loss cotangent on the last stage and ``ppermute``s
# activation-cotangents upstream stage->stage-1, accumulating weight grads locally.


def _embed_tokens(
    token_embed: jax.Array,
    embed_norm,
    embed_gated_norm,
    tokens: jax.Array,
    batch_spec: P,
) -> jax.Array:
    """Embed + post-embed gated-norm for one microbatch (stage-0 forward piece).

    Mirrors the embed prefix of ``pipeline_forward_loss.body`` /
    ``Transformer.__call__``: gather with the production batch sharding, then
    ``embed_gated_norm(embed_norm(.))``.
    """
    embedded = token_embed.at[tokens].get(out_sharding=batch_spec)
    return embed_gated_norm(embed_norm(embedded))


def _head_loss(
    final_norm,
    final_gated_norm,
    output_proj: jax.Array,
    hidden: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Final-norm + fused-CE head for one microbatch's last-stage hidden.

    Mirrors the tail of ``pipeline_forward_loss.body``: final gated-norm, then the
    fused linear softmax cross-entropy against next-token ``labels``. Returns the
    per-microbatch scalar CE (mean reduction over its own tokens).
    """
    final_hidden = final_gated_norm(final_norm(hidden))
    return fused_linear_softmax_cross_entropy_loss(
        final_hidden,
        output_proj,
        labels,
        weight=weight.astype(jnp.float32),
        reduction="mean",
        dtype=jnp.float32,
    )


def _next_token_labels(tokens: jax.Array) -> jax.Array:
    """Left-shift tokens by one for next-token prediction; last position labelled 0."""
    return jnp.concatenate([tokens[:, 1:], tokens[:, :1] * 0], axis=1).astype(jnp.int32)


def pipeline_value_and_grad(
    transformer: Transformer,
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    token_microbatches: jax.Array,
    weight_microbatches: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
    num_microbatches: int,
) -> tuple[jax.Array, eqx.Module, eqx.Module]:
    """``(loss, embed_grads, stage_grads)`` for the production Transformer via a
    manual GPipe backward inside the stage-manual ``shard_map``.

    The whole model -- embed (stage 0), the stacked production ``Block``s (per
    stage), and the fused-CE head (last stage) -- is differentiated without ever
    transposing the outer shard_map: a forward sweep captures per-(stage,
    microbatch) ``jax.vjp`` closures, then a backward sweep seeds the loss cotangent
    on the last stage and ``ppermute``s activation-cotangents stage->stage-1.

    ``token_microbatches`` / ``weight_microbatches`` are ``[num_microbatches,
    microbatch, seq]``. Returns ``(loss, embed_grads, stage_grads)`` where
    ``embed_grads`` is the array tree of the replicated embed/norm/head params (the
    cotangent psummed onto every stage; only stages 0 / last contribute) and
    ``stage_grads`` is the ``[stage, layers_per_stage, ...]`` block grad (each
    stage keeps its own shard).
    """
    num_layers = len(transformer.blocks)
    cfg = transformer.config
    seq_len = token_microbatches.shape[-1]

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

    layer_masks = build_layer_masks(transformer, num_stages, seq_len)

    stage_spec = P(STAGE_AXIS)
    repl = P()
    embed_in_specs = jax.tree_util.tree_map(lambda _: repl, embed_arrays)
    stage_in_specs = jax.tree_util.tree_map(lambda _: stage_spec, stage_block_arrays)

    def body(stage_arrays, embed, masks, tokens, weights):
        sid = jax.lax.axis_index(STAGE_AXIS)
        S = num_stages
        M = num_microbatches
        T = M + S - 1
        fwd_perm = [(i, i + 1) for i in range(S - 1)]
        bwd_perm = [(i + 1, i) for i in range(S - 1)]
        is_first = sid == 0
        is_last = sid == (S - 1)
        batch_spec = _batch_spec()

        token_embed, embed_norm, embed_gated_norm, final_norm, final_gated_norm, output_proj = eqx.combine(
            embed, embed_static
        )
        stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays)
        stage_masks = masks[0]

        hidden_shape = (token_microbatches.shape[1], seq_len, cfg.hidden_dim)
        # Every activation buffer (ppermuted carry, last-stage collection, backward
        # cotangent carry) lives in the Explicit batch sharding the production blocks
        # reshard to, so the wavefront selects / dynamic-updates all agree.
        zero_hidden = reshard(jnp.zeros(hidden_shape, jnp.float32), batch_spec)

        # --- Forward sweep: capture per-timestep vjp closures + last-stage hiddens.
        vjp_x_by_t: list = [None] * T
        vjp_p_by_t: list = [None] * T
        embed_vjp_by_t: list = [None] * T
        valid_by_t: list = [None] * T

        buf = zero_hidden
        z_total = jnp.zeros((), jnp.float32)
        h_final = jnp.broadcast_to(zero_hidden, (M, *hidden_shape))
        for t in range(T):
            m = t - sid
            valid = (m >= 0) & (m < M)
            m_clip = jnp.clip(m, 0, M - 1)
            tok_m = jax.lax.dynamic_index_in_dim(tokens, m_clip, axis=0, keepdims=False)

            # Loop vars bound as lambda defaults so jax.vjp captures this iteration's
            # tensors immediately (B023-safe).
            embedded, embed_vjp = jax.vjp(
                lambda te, en, eg, _t=tok_m: _embed_tokens(te, en, eg, _t, batch_spec),
                token_embed,
                embed_norm,
                embed_gated_norm,
            )
            # Both select branches must share a sharding: the embed gather carries the
            # Explicit batch spec, buf (ppermuted activation) starts replicated, so
            # reshard both to the batch spec before choosing.
            stage_in = jnp.where(is_first, reshard(embedded, batch_spec), reshard(buf, batch_spec))

            (stage_out, z_local), vjp_x = jax.vjp(
                lambda h, _sb=stage_blocks: _run_stage_blocks(_sb, block_static, h, stage_masks), stage_in
            )
            (_, _), vjp_p = jax.vjp(
                lambda sb, _si=stage_in: _run_stage_blocks(sb, block_static, _si, stage_masks), stage_blocks
            )

            z_total = z_total + jnp.where(valid, z_local, 0.0)
            contrib = jnp.where(is_last & valid, stage_out, jnp.zeros_like(stage_out))
            prev = jax.lax.dynamic_index_in_dim(h_final, m_clip, axis=0, keepdims=False)
            h_final = jax.lax.dynamic_update_index_in_dim(h_final, prev + contrib, m_clip, axis=0)

            vjp_x_by_t[t] = vjp_x
            vjp_p_by_t[t] = vjp_p
            embed_vjp_by_t[t] = embed_vjp
            valid_by_t[t] = valid

            buf = jax.lax.ppermute(stage_out, STAGE_AXIS, fwd_perm)

        # --- Head: scored once after the sweep on the last stage's collected hiddens.
        # h_final is psummed so the last stage's per-microbatch hidden is visible on
        # every stage (every other stage held zero), making the head vjp run identically
        # on all stages -- so its (replicated) grad is already correct without a psum.
        h_final = jax.lax.psum(h_final, STAGE_AXIS)
        # Use the in-body (stage-Manual mesh) token/weight args -- the outer closure
        # vars carry the all-Explicit mesh aval and a reshape under them clashes.
        flat_batch = M * hidden_shape[0]
        labels = _next_token_labels(tokens.reshape(flat_batch, seq_len))
        weights_flat = weights.reshape(flat_batch, seq_len)
        # Merging the replicated microbatch axis M into the batch-sharded microbatch
        # axis is ambiguous to the partitioner, so name the flattened batch's
        # sharding explicitly (the merged [M*microbatch] axis carries the batch spec).
        flat_hidden_spec = P(batch_spec[0], None, None)
        h_final_flat = jax.lax.reshape(h_final, (flat_batch, seq_len, cfg.hidden_dim), out_sharding=flat_hidden_spec)

        # Score the whole global batch (all microbatches) in ONE fused-CE call with a
        # single mean reduction -- identical to the non-pipelined oracle, which means
        # over all batch*seq tokens at once (not a mean of per-microbatch means).
        def _head_mean(fn, fgn, op, h):
            return _head_loss(fn, fgn, op, h, labels, weights_flat)

        ce_loss, head_vjp = jax.vjp(_head_mean, final_norm, final_gated_norm, output_proj, h_final_flat)
        g_final_norm, g_final_gated, g_output_proj, dh_final = head_vjp(jnp.ones((), jnp.float32))

        z_total = jax.lax.psum(z_total, STAGE_AXIS)
        # Each microbatch's per-layer z-loss is a token mean over its own rows; the
        # full-batch mean (what the oracle computes) is the mean over microbatches of
        # those, so divide the per-(stage,microbatch) sum by M as well as num_layers.
        aux = cfg.router_z_loss_coef * (z_total / num_layers / num_microbatches)
        total_loss = ce_loss + aux

        # --- Backward sweep: reverse-time B pass. dy seeds from dh_final on the last
        # stage, dbuf (ppermuted upstream) elsewhere; dx ships upstream; embed/stage
        # grads accumulate. Aux z-loss cotangent is router_z_loss_coef / num_layers
        # on every valid slot (matching how z_total enters the loss).
        g_embed_token = jax.tree_util.tree_map(jnp.zeros_like, token_embed)
        g_embed_norm = jax.tree_util.tree_map(jnp.zeros_like, embed_norm)
        g_embed_gated = jax.tree_util.tree_map(jnp.zeros_like, embed_gated_norm)
        g_stage = jax.tree_util.tree_map(jnp.zeros_like, stage_blocks)

        aux_cot_scale = cfg.router_z_loss_coef / num_layers / num_microbatches
        microbatch = hidden_shape[0]
        dbuf = zero_hidden
        for t in reversed(range(T)):
            valid = valid_by_t[t]
            m_clip = jnp.clip(t - sid, 0, M - 1)
            # dh_final is the flat [M*microbatch, seq, D] head cotangent; microbatch m's
            # slice is rows [m*microbatch : (m+1)*microbatch].
            dout = jax.lax.dynamic_slice_in_dim(dh_final, m_clip * microbatch, microbatch, axis=0)
            dout = jnp.where(valid, dout, jnp.zeros_like(dout))
            # dout (from psummed dh_final) and dbuf (ppermuted upstream cotangent) may
            # carry different shardings; reshard both to the batch spec before the select
            # and before feeding the stage vjp (which expects the batch-sharded cotangent).
            dy = jnp.where(is_last, reshard(dout, batch_spec), reshard(dbuf, batch_spec))

            z_cot = jnp.where(valid, aux_cot_scale, 0.0)
            (dx,) = vjp_x_by_t[t]((dy, z_cot))
            (dp,) = vjp_p_by_t[t]((dy, z_cot))
            g_stage = _tree_add(g_stage, dp)

            first_valid = is_first & valid
            dx_embed = jnp.where(first_valid, dx, jnp.zeros_like(dx))
            g_te, g_en, g_eg = embed_vjp_by_t[t](dx_embed)
            g_embed_token = _tree_add(g_embed_token, g_te)
            g_embed_norm = _tree_add(g_embed_norm, g_en)
            g_embed_gated = _tree_add(g_embed_gated, g_eg)

            dx = reshard(dx, batch_spec)
            dbuf = jax.lax.ppermute(dx, STAGE_AXIS, bwd_perm)

        # The embed grad is genuinely stage-local (gated to stage 0's valid slots, zero
        # elsewhere), so psum over stage sums the single contribution onto every device
        # to match the replicated out_spec P(). The head grad is NOT psummed: the head
        # runs on the psum-replicated h_final, so every stage already holds the IDENTICAL
        # full head grad -- psumming it would multiply by num_stages.
        g_embed = (
            jax.lax.psum(g_embed_token, STAGE_AXIS),
            jax.lax.psum(g_embed_norm, STAGE_AXIS),
            jax.lax.psum(g_embed_gated, STAGE_AXIS),
            g_final_norm,
            g_final_gated,
            g_output_proj,
        )
        # Stage block grads stay stage-sharded -- re-add the leading shard axis.
        g_stage = jax.tree_util.tree_map(lambda x: x[None], g_stage)
        return total_loss, g_embed, g_stage

    def _place(x, spec):
        return reshard(x, NamedSharding(mesh, spec))

    stage_block_arrays = jax.tree_util.tree_map(_place, stage_block_arrays, stage_in_specs)
    embed_arrays = jax.tree_util.tree_map(_place, embed_arrays, embed_in_specs)
    layer_masks = _place(layer_masks, stage_spec)
    token_microbatches = _place(token_microbatches, repl)
    weight_microbatches = _place(weight_microbatches, repl)

    # The inner-shard_map VMA override must be live while the outer shard_map's body is
    # traced (that is when the per-stage jax.vjp closures -- and the inner EP/head
    # transposes -- are built).
    with _inner_shard_maps_track_vma():
        loss, g_embed, g_stage = shard_map(
            body,
            mesh=mesh,
            in_specs=(stage_in_specs, embed_in_specs, stage_spec, repl, repl),
            out_specs=(repl, embed_in_specs, stage_in_specs),
            axis_names=frozenset({STAGE_AXIS}),
            check_vma=False,
        )(stage_block_arrays, embed_arrays, layer_masks, token_microbatches, weight_microbatches)
    return loss, g_embed, g_stage


# --- Autodiff backward: whole-program value_and_grad of the pipeline FORWARD ----
#
# The manual-backward path above forces the production EP / QB / fused-CE inner
# shard_maps to ``check_vma=True`` so their reverse-mode transpose reduces
# replicated-weight cotangents. That transpose path dies on TPU
# (``manual_axis_type on jax.ShapeDtypeStruct must not be None`` inside the
# check_vma=True shard_map transpose). The autodiff path below drops the manual
# backward AND the check_vma override entirely: it differentiates the pipeline
# FORWARD with ``jax.value_and_grad`` and lets whole-program GSPMD insert the
# replicated-weight grad reductions over the Explicit data/expert axes -- the same
# way top-level FSDP training transposes the ring-EP. The only thing that blocks
# ``value_and_grad`` through the outer ``check_vma=False`` shard_map is the
# production model's in-body Explicit ``reshard`` calls (``reshard out_specs must
# refer to a manual axis``); the context manager below neutralizes them.


class ReshardNeutralization(enum.Enum):
    """How the in-body ``reshard`` calls are neutralized for autodiff transpose.

    ``SPEC`` rewrites ``reshard(x, NamedSharding(mesh, spec))`` to ``reshard(x,
    spec)`` -- the bare-``PartitionSpec`` form resolves the sharding against the
    *context* (Manual-stage) mesh instead of pinning the all-Explicit aval mesh, so
    the value's aval mesh matches the shard_map body and ``jax.value_and_grad`` can
    transpose it. ``IDENTITY`` drops the reshard entirely (fully GSPMD-inferred,
    mirroring the reshard-free toy); it leaves the model's ``out_sharding=`` einsums
    to place activations.

    ``jax.lax.with_sharding_constraint`` is NOT an option here: it rejects
    Explicit-axis specs (it only constrains Auto axes), which is exactly what the
    production model's batch specs name.
    """

    SPEC = enum.auto()
    IDENTITY = enum.auto()


# Modules whose module-level ``reshard`` name the production forward reaches:
# ``grug_model`` (direct ``reshard`` + ``_batch_reshard``), ``grug_sharding``
# (``_reshard_for_init`` / ``_reshard_for_shard_map`` / ``unshard`` all call the
# module-level ``reshard``), and ``grug_loss`` (the fused-CE head's reshards).
_RESHARD_PATCHED_MODULES = (grug_model, grug_sharding, grug_loss)


def _bare_spec(sharding) -> P:
    """Extract the ``PartitionSpec`` from a ``NamedSharding`` (or pass a spec through).

    The production forward calls ``reshard`` with either a ``NamedSharding`` (the
    ``_reshard_for_*`` helpers) or a bare ``PartitionSpec`` (direct model reshards);
    both reduce to a spec here.
    """
    spec = getattr(sharding, "spec", sharding)
    return spec if isinstance(spec, P) else P(*spec)


def _reshard_replacement(mode: ReshardNeutralization):
    """Build a drop-in ``reshard(x, sharding)`` that ``value_and_grad`` can transpose."""
    if mode is ReshardNeutralization.IDENTITY:

        def _identity(x, _sharding):
            return x

        return _identity

    def _spec_reshard(x, sharding):
        return reshard(x, _bare_spec(sharding))

    return _spec_reshard


def _local_moe_mlp(
    x,
    selected_experts,
    combine_weights,
    w_up_gate,
    w_down,
    *,
    activation=grug_moe.ActivationFunctionEnum.silu,
    implementation=None,
    mesh=None,
    capacity_factor=1.0,
    report_capacity_overflow=False,
):
    """``moe_mlp`` without the inner EP ``shard_map`` -- a dense GSPMD-partitioned MoE.

    The production ``moe_mlp`` wraps its expert compute in a ``check_vma=False``
    ``shard_map`` over the ``expert`` axis. Nested inside the outer stage-manual
    ``shard_map``, that inner shard_map's reverse-mode transpose mis-scales the
    expert-weight cotangent by ``1/num_stages`` (the same failure the fused-CE head
    shows). This replacement runs every expert densely over all tokens with plain
    einsums (no token sort, no scatter, no collective): the ``expert`` axis stays an
    Explicit GSPMD axis on the weights and the token (``data``) axis on the
    activations, so whole-program autodiff inserts the expert-weight grad reduction
    over ``data`` itself -- exactly as the bet predicts.

    This dense path does NOT model the ring EP's per-shard capacity drop
    (``capacity_factor=1.0`` discards assignments past ``ceil(assignments/ep_size)``
    per shard). With no expert axis (``ep_size == 1``) the production fallback also
    drops nothing, so the dense path is forward-identical and grads are exact. With
    ``ep_size > 1`` the ring drops a few percent of assignments that this path keeps,
    so the EP forward (and hence grads) differ by the capacity-drop fraction.
    """
    activation_fn = activation.to_jax_fn() if isinstance(activation, grug_moe.ActivationFunctionEnum) else activation
    intermediate_dim = int(w_down.shape[1])

    # Replicate the expert weights over the ``expert`` axis (bare-spec reshard, so
    # the all-gather's transpose is the expert-grad reduce GSPMD inserts). Every
    # shard then holds all experts and the dense einsum needs no expert collective.
    w_up_gate = reshard(w_up_gate, P(None, None, None))
    w_down = reshard(w_down, P(None, None, None))

    # Dense expert compute: every expert applied to every token.
    # w_up_gate [E, D, 2I]; x [T, D] -> [T, E, 2I].
    w13 = jnp.einsum("td,edi->tei", x, w_up_gate)
    gate, up = grug_moe.split_moe_w13_output(w13, intermediate_dim=intermediate_dim, interleaved=False)
    hidden = activation_fn(gate) * up  # [T, E, I]
    # w_down [E, I, D]; hidden [T, E, I] -> [T, E, D].
    expert_out = jnp.einsum("tei,eid->ted", hidden, w_down)

    # One-hot dispatch: gather each token's selected experts and combine.
    # selected_experts [T, K] -> per-token-per-slot expert one-hot [T, K, E].
    num_experts = int(w_up_gate.shape[0])
    one_hot = jax.nn.one_hot(selected_experts, num_experts, dtype=expert_out.dtype)  # [T, K, E]
    # For each (token, slot) pick its expert's output, weight by combine_weights, sum over slots.
    selected_out = jnp.einsum("tke,ted->tkd", one_hot, expert_out)  # [T, K, D]
    out = jnp.einsum("tk,tkd->td", combine_weights.astype(expert_out.dtype), selected_out)

    if report_capacity_overflow:
        return out, jnp.zeros((), jnp.int32)
    return out


@contextlib.contextmanager
def _neutralize_reshards(mode: ReshardNeutralization):
    """Patch the production forward's ``reshard`` (and EP ``moe_mlp``) for autodiff.

    The production model reshards inter-layer activations with Explicit-axis specs
    via the module-level ``reshard`` name (directly, and through
    ``_batch_reshard`` / ``_reshard_for_shard_map`` / ``_reshard_for_init`` /
    ``unshard``). A ``reshard`` whose target is a ``NamedSharding`` pins the
    all-Explicit aval mesh, which clashes with the Manual-stage context mesh under
    transpose and makes the outer ``check_vma=False`` shard_map non-transposable.
    Rebinding ``reshard`` in scope to the bare-spec form (or identity) keeps the
    forward numerically identical while letting ``jax.value_and_grad`` transpose the
    outer shard_map.

    The EP ``moe_mlp`` is additionally rebound to :func:`_local_moe_mlp`: its inner
    ``shard_map`` over ``expert`` mis-scales the expert-weight cotangent by
    ``1/num_stages`` under the outer transpose, so the autodiff path runs a
    shard-map-free dense MoE instead. All restored on exit.
    """
    replacement = _reshard_replacement(mode)
    originals = [(m, m.reshard) for m in _RESHARD_PATCHED_MODULES]
    moe_original = grug_moe.moe_mlp
    try:
        for m, _ in originals:
            m.reshard = replacement
        grug_moe.moe_mlp = _local_moe_mlp
        yield
    finally:
        for m, fn in originals:
            m.reshard = fn
        grug_moe.moe_mlp = moe_original


def _reference_cross_entropy(
    final_hidden: jax.Array,
    output_proj: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Mean next-token cross-entropy WITHOUT an inner ``shard_map``.

    The production fused-CE head wraps its compute in a ``jax.shard_map`` (its own
    ``out_specs=P()`` reduction). Nested inside the outer stage-manual
    ``check_vma=False`` shard_map, that inner shard_map's reverse-mode transpose
    mis-scales the activation cotangent by ``1/num_stages`` while leaving its
    directly-consumed ``output_proj`` at full scale -- so whole-program autodiff
    gives non-uniform grads. This reference head computes the same loss with plain
    ops, letting the data/expert-axis token reduction happen via GSPMD on the
    Explicit axes (the activation is batch-sharded), which transposes exactly. The
    mean over the batch reduces over the Explicit data/expert shards automatically.
    """
    logits = jnp.einsum("bsd,dv->bsv", final_hidden, output_proj).astype(jnp.float32)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    label_logit = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = (log_z - label_logit) * weight
    return jnp.sum(nll) / jnp.sum(weight)


def pipeline_loss(
    embed_arrays: eqx.Module,
    stage_block_arrays: eqx.Module,
    embed_static: eqx.Module,
    block_static: eqx.Module,
    transformer: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
) -> jax.Array:
    """Scalar pipelined next-token loss as a function of the differentiable params.

    Runs the production Transformer forward as a stage-pipelined ``shard_map``
    (``check_vma=False``) -- single global batch, GPipe forward, ppermute carry,
    embed on stage 0, reference-CE head on the last stage -- and returns the mean
    loss. ``embed_arrays`` / ``stage_block_arrays`` are the differentiable leaves
    (replicated embed/norm/head, stacked ``[stage, layers_per_stage, ...]`` blocks);
    ``embed_static`` / ``block_static`` / ``transformer`` carry the static structure
    and config. No manual vjp, no ``check_vma=True`` -- ``jax.value_and_grad`` of
    this builds the whole backward.

    The head uses :func:`_reference_cross_entropy` rather than the production fused
    CE: the fused head's inner shard_map breaks whole-program-autodiff cotangent
    scaling under the outer stage-manual shard_map (see that helper's docstring).
    """
    num_layers = len(transformer.blocks)
    cfg = transformer.config
    seq_len = token_ids.shape[1]

    layer_masks = build_layer_masks(transformer, num_stages, seq_len)

    stage_spec = P(STAGE_AXIS)
    repl = P()
    embed_in_specs = jax.tree_util.tree_map(lambda _: repl, embed_arrays)
    stage_in_specs = jax.tree_util.tree_map(lambda _: stage_spec, stage_block_arrays)

    def body(stage_arrays, embed, masks, tokens, weight):
        sid = jax.lax.axis_index(STAGE_AXIS)
        is_first = sid == 0
        is_last = sid == (num_stages - 1)

        token_embed, embed_norm, embed_gated_norm, final_norm, final_gated_norm, output_proj = eqx.combine(
            embed, embed_static
        )

        stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays)
        stage_masks = masks[0]

        # Embed via a plain gather, then place the activation on the production
        # inter-layer batch sharding with the bare-spec reshard (transposable under
        # value_and_grad inside a check_vma=False shard_map; the NamedSharding /
        # ``out_sharding=`` forms are not). This makes the scan carry enter already
        # batch-sharded, matching what every Block reshards its output to -- so the
        # scan carry input/output types agree.
        batch_spec = _batch_spec()
        embedded = reshard(token_embed[tokens], batch_spec)
        embedded = embed_gated_norm(embed_norm(embedded))

        fwd_perm = [(i, i + 1) for i in range(num_stages - 1)]
        buf = jnp.where(is_first, embedded, jnp.zeros_like(embedded))
        z_total = jnp.zeros((), jnp.float32)
        for t in range(num_stages):
            active = sid == t
            stage_out, z_local = _run_stage_blocks(stage_blocks, block_static, buf, stage_masks)
            z_total = z_total + jnp.where(active, z_local, 0.0)
            buf = jnp.where(active, stage_out, buf)
            if t < num_stages - 1:
                buf = jax.lax.ppermute(buf, STAGE_AXIS, fwd_perm)

        final_hidden = final_gated_norm(final_norm(buf))
        labels = _next_token_labels(tokens)
        ce = _reference_cross_entropy(final_hidden, output_proj, labels, weight.astype(jnp.float32))
        ce_total = jax.lax.psum(jnp.where(is_last, ce, 0.0), STAGE_AXIS)
        z_total = jax.lax.psum(z_total, STAGE_AXIS)
        aux = cfg.router_z_loss_coef * (z_total / num_layers)
        return ce_total + aux

    def _place(x, spec):
        return reshard(x, NamedSharding(mesh, spec))

    stage_block_arrays = jax.tree_util.tree_map(_place, stage_block_arrays, stage_in_specs)
    embed_arrays = jax.tree_util.tree_map(_place, embed_arrays, embed_in_specs)
    layer_masks = _place(layer_masks, stage_spec)
    token_ids = _place(token_ids, repl)
    loss_weight = _place(loss_weight, repl)

    return shard_map(
        body,
        mesh=mesh,
        in_specs=(stage_in_specs, embed_in_specs, stage_spec, repl, repl),
        out_specs=repl,
        axis_names=frozenset({STAGE_AXIS}),
        check_vma=False,
    )(stage_block_arrays, embed_arrays, layer_masks, token_ids, loss_weight)


def pipeline_value_and_grad_autodiff(
    transformer: Transformer,
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
    reshard_mode: ReshardNeutralization = ReshardNeutralization.SPEC,
) -> tuple[jax.Array, eqx.Module, eqx.Module]:
    """``(loss, embed_grads, stage_grads)`` via whole-program ``jax.value_and_grad``.

    Differentiates :func:`pipeline_loss` (the stage-manual ``shard_map`` forward)
    with ``jax.value_and_grad`` under :func:`_neutralize_reshards`, so GSPMD inserts
    the replicated-weight grad reductions over the Explicit data/expert axes. No
    manual GPipe backward, no ``check_vma=True``. Return grouping matches
    :func:`pipeline_value_and_grad`: ``embed_grads`` is the replicated
    embed/norm/head array tree, ``stage_grads`` the ``[stage, layers_per_stage,
    ...]`` block grad.

    ``reshard_mode`` selects how the production model's in-body Explicit reshards
    are neutralized for the transpose (``SPEC`` rewrites them to the
    transposable bare-spec form; ``IDENTITY`` drops them). A single global batch
    (no microbatching).
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
        return pipeline_loss(
            embed_arrays,
            stage_block_arrays,
            embed_static,
            block_static,
            transformer,
            token_ids,
            loss_weight,
            mesh=mesh,
            num_stages=num_stages,
        )

    with _neutralize_reshards(reshard_mode):
        loss, (g_embed, g_stage) = jax.value_and_grad(loss_fn, argnums=(0, 1))(embed_arrays, stage_block_arrays)
    return loss, g_embed, g_stage


# --- Real sparse ring-EP inline in the {stage, expert}-manual pipeline shard_map -
#
# The dense ``_local_moe_mlp`` above keeps ``expert`` a GSPMD axis and runs every
# expert over every token. The path below runs the PRODUCTION ring EP
# (``_moe_mlp_ep_ring_local``: all_gather dispatch + ragged_dot GMM + psum_scatter
# collect) INLINE in the same manual region as the pipeline ppermute -- one
# shard_map manualizing BOTH ``stage`` and ``expert``, no nested EP shard_map.
#
# Mesh: ``stage`` / ``expert`` are Explicit (manualized by the outer shard_map);
# ``data`` / ``replica_dcn`` / ``model`` are Auto (GSPMD / FSDP). Auto -- not
# Explicit -- is load-bearing: ``ragged_dot_general`` has no Explicit-axis sharding
# rule (it only lowers when its axes are Auto or Manual), and an Auto ``data`` axis
# lets whole-program ``value_and_grad`` insert the FSDP expert-weight grad reduction
# exactly as top-level training does. Because ``expert`` is MANUAL the 3-way
# PP x FSDP x EP has a single GSPMD axis (``data``), avoiding the two-GSPMD-under-
# manual partitioner crash; because there is exactly one shard_map the transpose is
# clean (ppermute<->reverse-ppermute, all_gather<->psum_scatter) with no
# ``check_vma=True`` and no manual backward.


def ep_pipeline_mesh(*, stage: int, expert: int, replica: int, data: int, model: int = 1) -> Mesh:
    """Mesh for the inline ring-EP pipeline: ``stage``/``expert`` Explicit, rest Auto.

    ``stage`` and ``expert`` are ``AxisType.Explicit`` so the outer shard_map can
    manualize them; ``replica_dcn`` / ``data`` / ``model`` are ``AxisType.Auto`` so
    ``ragged_dot_general`` lowers and GSPMD inserts the FSDP weight-grad reduce.
    """
    shape = (stage, replica, data, expert, model)
    if int(np.prod(shape)) != jax.device_count():
        raise ValueError(f"mesh shape {shape} (prod={int(np.prod(shape))}) must use all {jax.device_count()} devices")
    devices = np.array(jax.devices(), dtype=object).reshape(shape)
    axis_types = (AxisType.Explicit, AxisType.Auto, AxisType.Auto, AxisType.Explicit, AxisType.Auto)
    return Mesh(devices, _GRUG_MESH_AXIS_NAMES, axis_types=axis_types)


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
    "expert")`` returns each shard's token slice. ``x`` is this expert-shard's local
    token slice ``[TL, D]`` (sharded over ``data`` by GSPMD); the all-gather makes it
    global over ``expert``.

    ``ragged_dot_general`` has no Explicit-axis sharding rule, so the remaining GSPMD
    axes (``data`` / ``replica_dcn`` / ``model``) must be Auto here -- see
    :func:`ep_pipeline_mesh`. The ``dropped`` diagnostic is discarded; its psum is
    scoped to the manual ``expert`` axis by :func:`_inline_ring_moe_mlp_context`.

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
def _inline_ring_moe_mlp_context(num_experts: int):
    """Wire the production forward to run the real ring EP inline under the EP mesh.

    Patches, all restored on exit:

    1. ``grug_moe.moe_mlp`` -> :func:`_inline_ring_moe_mlp` (skip the nested EP
       shard_map; ``expert`` is already manual).
    2. ``_EP_NUM_EXPERTS`` -> ``num_experts`` (the full count; the local weight slice
       only reveals ``E / expert_size``).
    3. ``ep_ring._batch_axes`` -> ``("expert",)`` so the ring's ``dropped``
       diagnostic ``psum`` reduces over the manual ``expert`` axis only (``data`` /
       ``replica_dcn`` are Auto here and would be unbound).
    4. The router QB-beta ``shard_map`` -> a local threshold (``qb_beta`` is
       metrics-only; its production ``shard_map`` is invalid under the EP mesh).

    The reshard / ``out_sharding`` neutralization for the Auto mesh is supplied
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


@contextlib.contextmanager
def _neutralize_reshards_auto():
    """Drop the production forward's Explicit-axis sharding calls for the Auto EP mesh.

    Under the EP mesh ``data`` / ``replica_dcn`` / ``model`` are Auto, so NO
    ``reshard`` / ``out_sharding=`` may name them. This drops every in-body
    ``reshard`` to identity (the model's batch axes are GSPMD-inferred under Auto) and
    rewrites ``grug_model._batch_spec`` to ``P()`` so the model's ``out_sharding=
    _batch_spec()`` einsums replicate (valid under Auto) instead of naming the batch
    axes. Restored on exit.
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


def _ep_pipeline_loss(
    embed_arrays: eqx.Module,
    stage_block_arrays: eqx.Module,
    embed_static: eqx.Module,
    block_static: eqx.Module,
    transformer: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
) -> jax.Array:
    """Pipelined next-token loss with the real ring EP inline; ``{stage, expert}`` manual.

    Mirrors :func:`pipeline_loss` but (1) manualizes ``{stage, expert}`` instead of
    ``{stage}``, (2) shards the expert MLP weights' leading expert dim over the manual
    ``expert`` axis (stacked under ``stage``), (3) shards each microbatch's tokens
    over ``expert`` too, and (4) uses only Auto-safe sharding ops in its own body
    (the model's Explicit reshards are neutralized by :func:`_neutralize_reshards_auto`).
    """
    num_layers = len(transformer.blocks)
    cfg = transformer.config
    seq_len = token_ids.shape[1]

    layer_masks = build_layer_masks(transformer, num_stages, seq_len)

    stage_spec = P(STAGE_AXIS)
    # Stacked block arrays carry a leading [stage, layers_per_stage, ...]; the expert
    # MLP weights additionally carry the expert dim at array-position 2 (after stage,
    # layers_per_stage), so they are sharded P("stage", None, "expert", ...) and every
    # other block leaf is P("stage", ...). Tokens are sharded over expert.
    stage_in_specs = _stage_in_specs(stage_block_arrays)
    embed_in_specs = jax.tree_util.tree_map(lambda _: P(), embed_arrays)
    token_spec = P(EXPERT_AXIS, None)

    def body(stage_arrays, embed, masks, tokens, weight):
        sid = jax.lax.axis_index(STAGE_AXIS)
        is_first = sid == 0
        is_last = sid == (num_stages - 1)

        token_embed, embed_norm, embed_gated_norm, final_norm, final_gated_norm, output_proj = eqx.combine(
            embed, embed_static
        )

        stage_blocks = jax.tree_util.tree_map(lambda x: x[0], stage_arrays)
        stage_masks = masks[0]

        # Embed this expert-shard's local token slice; Auto GSPMD places activations.
        embedded = token_embed[tokens]
        embedded = embed_gated_norm(embed_norm(embedded))

        fwd_perm = [(i, i + 1) for i in range(num_stages - 1)]
        buf = jnp.where(is_first, embedded, jnp.zeros_like(embedded))
        z_total = jnp.zeros((), jnp.float32)
        for t in range(num_stages):
            active = sid == t
            stage_out, z_local = _run_stage_blocks(stage_blocks, block_static, buf, stage_masks)
            z_total = z_total + jnp.where(active, z_local, 0.0)
            buf = jnp.where(active, stage_out, buf)
            if t < num_stages - 1:
                buf = jax.lax.ppermute(buf, STAGE_AXIS, fwd_perm)

        final_hidden = final_gated_norm(final_norm(buf))
        labels = _next_token_labels(tokens)
        # Each expert-shard holds its token slice; sum NLL / weight over the local
        # slice then reduce over the manual expert axis to score the full batch.
        ce = _ep_cross_entropy(final_hidden, output_proj, labels, weight.astype(jnp.float32))
        ce_total = jax.lax.psum(jnp.where(is_last, ce, 0.0), STAGE_AXIS)
        # Each layer's router z-loss is a token mean over this expert-shard's slice; the
        # full-batch mean (what the oracle computes) is the mean over the equal-size
        # expert shards, so average ``z_total`` over the manual expert axis as well as
        # summing it over stages.
        z_total = jax.lax.psum(z_total, STAGE_AXIS)
        z_total = jax.lax.psum(z_total, EXPERT_AXIS) / jax.lax.psum(1, EXPERT_AXIS)
        aux = cfg.router_z_loss_coef * (z_total / num_layers)
        return ce_total + aux

    def _place(x, spec):
        return reshard(x, NamedSharding(mesh, spec))

    stage_block_arrays = jax.tree_util.tree_map(_place, stage_block_arrays, stage_in_specs)
    embed_arrays = jax.tree_util.tree_map(_place, embed_arrays, embed_in_specs)
    layer_masks = _place(layer_masks, stage_spec)
    token_ids = _place(token_ids, token_spec)
    loss_weight = _place(loss_weight, token_spec)

    return shard_map(
        body,
        mesh=mesh,
        in_specs=(stage_in_specs, embed_in_specs, stage_spec, token_spec, token_spec),
        out_specs=P(),
        axis_names=frozenset({STAGE_AXIS, EXPERT_AXIS}),
        check_vma=False,
    )(stage_block_arrays, embed_arrays, layer_masks, token_ids, loss_weight)


def _stage_in_specs(stage_block_arrays: eqx.Module) -> eqx.Module:
    """In-specs for the stacked block arrays: shard expert MLP weights over ``expert``.

    Every block leaf is sharded over ``stage`` (leading dim). The expert MLP weights
    (``mlp.expert_mlp.w_gate_up`` ``[stage, layers, E, D, I2]`` and ``...w_down``
    ``[stage, layers, E, I, D]``) additionally shard their leading expert dim (array
    axis 2) over the manual ``expert`` axis, so each shard holds ``E / expert_size``
    experts -- exactly what :func:`_moe_mlp_ep_ring_local` expects.
    """
    default_spec = P(STAGE_AXIS)
    specs = jax.tree_util.tree_map(lambda _: default_spec, stage_block_arrays)
    expert_spec = P(STAGE_AXIS, None, EXPERT_AXIS)
    expert_mlp = specs.mlp.expert_mlp
    expert_mlp = eqx.tree_at(lambda m: (m.w_gate_up, m.w_down), expert_mlp, (expert_spec, expert_spec))
    return eqx.tree_at(lambda s: s.mlp.expert_mlp, specs, expert_mlp)


def _ep_cross_entropy(
    final_hidden: jax.Array,
    output_proj: jax.Array,
    labels: jax.Array,
    weight: jax.Array,
) -> jax.Array:
    """Mean next-token cross-entropy whose token reduction spans the manual expert axis.

    ``final_hidden`` is this expert-shard's local token slice. The cross-entropy is a
    weighted mean over ALL tokens; with tokens sharded over the manual ``expert``
    axis, sum the per-shard NLL and weight then ``psum`` both over ``expert`` before
    dividing -- so every shard returns the full-batch mean.
    """
    logits = jnp.einsum("bsd,dv->bsv", final_hidden, output_proj).astype(jnp.float32)
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    label_logit = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    nll = jnp.sum((log_z - label_logit) * weight)
    wsum = jnp.sum(weight)
    nll = jax.lax.psum(nll, EXPERT_AXIS)
    wsum = jax.lax.psum(wsum, EXPERT_AXIS)
    return nll / wsum


def pipeline_value_and_grad_ep(
    transformer: Transformer,
    stage_block_arrays: eqx.Module,
    block_static: eqx.Module,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    mesh: jax.sharding.Mesh,
    num_stages: int,
) -> tuple[jax.Array, eqx.Module, eqx.Module]:
    """``(loss, embed_grads, stage_grads)`` for the REAL ring-EP inline pipeline.

    Differentiates :func:`_ep_pipeline_loss` with whole-program ``jax.value_and_grad``
    under the inline-ring + Auto-reshard patches. ``mesh`` must be an
    :func:`ep_pipeline_mesh` (``stage``/``expert`` Explicit, rest Auto). Return
    grouping matches :func:`pipeline_value_and_grad_autodiff`.
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
        return _ep_pipeline_loss(
            embed_arrays,
            stage_block_arrays,
            embed_static,
            block_static,
            transformer,
            token_ids,
            loss_weight,
            mesh=mesh,
            num_stages=num_stages,
        )

    with _inline_ring_moe_mlp_context(transformer.config.num_experts), _neutralize_reshards_auto():
        loss, (g_embed, g_stage) = jax.value_and_grad(loss_fn, argnums=(0, 1))(embed_arrays, stage_block_arrays)
    return loss, g_embed, g_stage
