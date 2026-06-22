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
import functools
import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import NamedSharding, reshard
from jax.sharding import PartitionSpec as P
from levanter.grug import grug_moe
from levanter.grug import loss as grug_loss
from levanter.grug.attention import AttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer, _batch_spec

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
