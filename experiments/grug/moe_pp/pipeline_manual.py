# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manual-threaded pipeline parallelism for the production grug-MoE.

There is NO ``stage`` mesh axis. The pipeline schedule is driven from Python by
composing per-stage ``jax.vjp`` closures: the forward sweep runs each stage (a
contiguous slice of the real ``Block``s) and captures its vjp; the backward sweep
plays the vjps back in reverse, seeding the loss/aux cotangents and threading the
activation cotangent upstream. Each stage's weight grad is produced by its OWN
vjp, so the grad is never stacked across stages -- the ``[num_stages, ...]``
buffer that whole-program autodiff through a ``stage``-manual ``shard_map``
materializes (and that OOMs the GPU partitioner at scale) cannot form here.

Because there is no outer ``shard_map``, EP/FSDP run exactly as in production:
each stage is the real ``Block`` forward on the normal grug mesh, so the model's
in-body ``reshard`` calls and the ring-EP nested ``shard_map`` need no
neutralization. This is the single-batch correctness baseline (all-forward then
all-backward, identical grads to any valid GPipe order); the microbatched
zero-bubble op-order layers on top of the same per-stage vjp primitives.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from experiments.grug.moe import model as grug_model
from experiments.grug.moe.model import Transformer

# The embed/head leaves, in the order the rest of the pipeline names them. The first
# three are consumed by the embed prefix (stage 0), the last three by the head (last
# stage); both stages differentiate the whole tuple and the two grads are summed.
_EMBED_HEAD_FIELDS = (
    "token_embed",
    "embed_norm",
    "embed_gated_norm",
    "final_norm",
    "final_gated_norm",
    "output_proj",
)


def _embed_head_tuple(transformer: Transformer) -> tuple:
    return tuple(getattr(transformer, f) for f in _EMBED_HEAD_FIELDS)


def _embed_forward(embed_head_arrays: tuple, embed_head_static: tuple, token_ids: jax.Array) -> jax.Array:
    """Stage-0 prefix: embed lookup + embed norm, mirroring ``Transformer.__call__``."""
    token_embed, embed_norm, embed_gated_norm, _fn, _fgn, _op = eqx.combine(embed_head_arrays, embed_head_static)
    hidden = token_embed.at[token_ids].get(out_sharding=grug_model._batch_spec())
    return embed_gated_norm(embed_norm(hidden))


def _stage_forward(
    block_arrays_slice, block_static, hidden: jax.Array, stage_masks, remat: bool = True
) -> tuple[jax.Array, jax.Array]:
    """Run one stage's slice of real ``Block``s; return ``(hidden, router_z_sum)``.

    ``block_arrays_slice`` is the tuple of this stage's per-block array pytrees and
    ``stage_masks`` the matching per-layer ``AttentionMask`` objects. The router
    z-loss is summed across this stage's layers so the driver can aggregate it.

    With ``remat`` each block is wrapped in ``jax.checkpoint``: the backward recomputes
    the block forward (attention scores, the MoE dispatch/GMM intermediates) instead of
    saving them, so only the residual stream at block boundaries is held -- required at
    the memory-bound (40B GPU) scale. With ``remat=False`` the forward residuals are
    kept, so a single combined backward differentiates without recompute (FSDP-parity
    FLOPs); use this when memory allows. Remat is value-identical, so grads are unchanged.
    """
    z = jnp.zeros((), jnp.float32)
    for block_arrays, mask in zip(block_arrays_slice, stage_masks, strict=True):

        def _apply_block(b_arrays, h, _mask=mask):
            block = eqx.combine(b_arrays, block_static)
            return block(h, _mask)

        apply = jax.checkpoint(_apply_block) if remat else _apply_block
        hidden, router_stats = apply(block_arrays, hidden)
        z = z + router_stats["router_z_loss"].astype(jnp.float32)
    return hidden, z


def _head_forward(
    embed_head_arrays: tuple, embed_head_static: tuple, hidden: jax.Array, labels: jax.Array, weight: jax.Array
) -> jax.Array:
    """Last-stage suffix: final norm + fused-CE, mirroring ``Transformer.next_token_loss``."""
    _te, _en, _egn, final_norm, final_gated_norm, output_proj = eqx.combine(embed_head_arrays, embed_head_static)
    hidden = final_gated_norm(final_norm(hidden))
    return fused_linear_softmax_cross_entropy_loss(
        hidden, output_proj, labels, weight=weight, reduction="mean", logsumexp_weight=None, dtype=jnp.float32
    )


def manual_pp_value_and_grad(
    transformer: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array,
    *,
    num_stages: int,
) -> tuple[jax.Array, tuple, list]:
    """``(loss, embed_head_grads, block_grads)`` via a manually threaded GPipe backward.

    Splits the block stack into ``num_stages`` contiguous stages and threads
    ``jax.vjp`` per stage (no ``stage`` mesh axis, no outer ``shard_map``). ``loss``
    equals the production ``Transformer.next_token_loss`` over the same batch by
    construction (same embed / masks / blocks / final-norm / fused-CE / router
    z-loss); the per-stage vjp composition is the chain rule, so the grads match the
    non-pipelined autodiff to float reassociation tolerance.

    Returns ``embed_head_grads`` (the 6-tuple grad, embed + head contributions
    summed) and ``block_grads`` (a length-``num_layers`` list of per-block array
    grads, in layer order).
    """
    cfg = transformer.config
    num_layers = cfg.num_layers
    if num_layers % num_stages != 0:
        raise ValueError(f"num_layers={num_layers} must be divisible by num_stages={num_stages}")
    layers_per_stage = num_layers // num_stages

    # Per-layer attention masks: the long sliding window on every 4th layer, short
    # otherwise -- the production rule from ``Transformer.__call__``.
    base_mask = grug_model.AttentionMask.causal()
    short_mask, long_mask = grug_model._layer_attention_masks(base_mask, sliding_window=cfg.sliding_window)
    per_layer_masks = [long_mask if (i % 4 == 3) else short_mask for i in range(num_layers)]

    embed_head = _embed_head_tuple(transformer)
    eh_arrays, eh_static = eqx.partition(embed_head, eqx.is_array)

    block_static = eqx.partition(transformer.blocks[0], eqx.is_array)[1]
    block_arrays = [eqx.partition(b, eqx.is_array)[0] for b in transformer.blocks]

    labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
    weight = loss_weight.astype(jnp.float32)

    # --- forward sweep: capture per-stage vjp closures, thread the activation ---
    hidden, embed_vjp = jax.vjp(lambda a: _embed_forward(a, eh_static, token_ids), eh_arrays)

    stage_vjps = []
    z_stages = []
    for s in range(num_stages):
        sl = slice(s * layers_per_stage, (s + 1) * layers_per_stage)
        arrays_slice = block_arrays[sl]
        masks_slice = per_layer_masks[sl]
        (hidden, z_s), vjp_s = jax.vjp(
            lambda a, h, _m=masks_slice: _stage_forward(a, block_static, h, _m), arrays_slice, hidden
        )
        stage_vjps.append(vjp_s)
        z_stages.append(z_s)

    ce, head_vjp = jax.vjp(lambda a, h: _head_forward(a, eh_static, h, labels, weight), eh_arrays, hidden)

    z_total = jnp.sum(jnp.stack(z_stages))
    loss = ce + cfg.router_z_loss_coef * (z_total / num_layers)

    # --- backward sweep: seed cotangents, thread the activation grad upstream ---
    # d(loss)/d(ce) = 1; d(loss)/d(z_s) = router_z_loss_coef / num_layers for every stage.
    g_head, d_hidden = head_vjp(1.0)
    dz = jnp.asarray(cfg.router_z_loss_coef / num_layers, jnp.float32)

    block_grads: list = [None] * num_layers
    for s in reversed(range(num_stages)):
        g_arrays_slice, d_hidden = stage_vjps[s]((d_hidden, dz))
        block_grads[s * layers_per_stage : (s + 1) * layers_per_stage] = list(g_arrays_slice)

    (g_embed,) = embed_vjp(d_hidden)

    # The embed prefix and the head each differentiate the whole 6-tuple (the other
    # half gets zero cotangents); sum to the full embed/head grad.
    embed_head_grads = jax.tree_util.tree_map(lambda a, b: a + b, g_embed, g_head)
    return loss, embed_head_grads, block_grads
