# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate 11 forward replay with explicit sublayer intervention sites.

The original model owns every attention and expert computation. This module
exposes its residual history so Boost/Cut can measure loss responses at those
sites without invoking a backward pass. Only the candidate 11 recipe is covered.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from levanter.grug.attention import AttentionMask, fa4_cute_segment_bounds

from experiments.grug.fast_track import model as native


class SublayerCapture(NamedTuple):
    completed_sources: tuple[jax.Array, ...]
    partial_before: jax.Array | None
    input: jax.Array
    output: jax.Array
    partial: jax.Array


class ForwardCapture(NamedTuple):
    token_rows: jax.Array
    bigram_ids: jax.Array
    bigram_rows: jax.Array
    embedding: jax.Array
    bigram_source: jax.Array
    stages: tuple[SublayerCapture, ...]
    final_sources: tuple[jax.Array, ...]
    final_stream: jax.Array
    shared_preactivations: tuple[jax.Array, ...]
    segment_ids: jax.Array | None
    routing_selected: tuple[jax.Array, ...]
    routing_weights: tuple[jax.Array, ...]
    routing_logits: tuple[jax.Array, ...]
    routed_inputs: tuple[jax.Array, ...]


def capture_forward(
    model: native.Transformer,
    token_ids: jax.Array,
    mask: AttentionMask,
    *,
    interventions: tuple[jax.Array, ...] = (),
) -> tuple[jax.Array, ForwardCapture]:
    """Return final normalized hidden states and the clean forward history.

    Interventions, when supplied, add to each attention/MLP branch output in
    execution order. Downstream computation is replayed in full. This replay
    is an intervention oracle; it is not yet Cut's frozen-context carrier.
    """
    cfg = model.config
    layers = model.layers()
    if interventions and len(interventions) != 2 * len(layers):
        raise ValueError("One intervention per attention/MLP branch is required")

    token_rows = native._embedding_gather(model.token_embed, token_ids)
    embedding = native._activation_boundary(model.embed_norm(token_rows))
    segment_ids = None
    if mask.segment_ids is not None:
        segments = native._batch_reshard(mask.segment_ids[0])
        segment_ids = (segments, segments)
    else:
        segments = None
    long_mask = AttentionMask(is_causal=True, segment_ids=segment_ids)
    short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
    batch_size, seq_len = token_ids.shape
    long_bounds, valid = fa4_cute_segment_bounds(long_mask, batch_size=batch_size, seq_len=seq_len, sliding_window=None)
    short_bounds, _ = fa4_cute_segment_bounds(
        short_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        sliding_window=cfg.sliding_window,
    )
    valid = native._batch_reshard(valid)
    long_mask = long_mask.with_fa4_bounds(native._batch_reshard(long_bounds), valid)
    short_mask = AttentionMask(is_causal=True, segment_ids=segment_ids).with_fa4_bounds(
        native._batch_reshard(short_bounds), valid
    )

    bigram_ids = native._bigram_hash_ids(token_ids, segments, cfg.embed2_rows, cfg.embed2_ngram)
    bigram_rows = native._table2_gather(cfg)(model.token_embed2, bigram_ids)
    bigram_source, _ = native._content_gate(
        embedding,
        model.embed2_norm(bigram_rows),
        model.bigram_gate_w,
        model.bigram_gate_b,
        model.bigram_gate_a_lr,
        model.bigram_gate_b_lr,
        act=cfg.bigram_gate_act,
        logit_scale=cfg.bigram_gate_logit_scale,
    )

    queries = jnp.stack(
        [query for layer in layers for query in (layer.attn_res_query_attn, layer.attn_res_query_mlp)]
        + [model.attn_res_query_final]
    )
    eps = cfg.layer_norm_eps
    blocks = (bigram_source,)
    block_logits = (native._attn_res_source_logits(bigram_source, queries, eps),)
    partial = embedding
    captures = []
    shared_preactivations = []
    routing_selected = []
    routing_weights = []
    routing_logits = []
    routed_inputs = []
    segment_size = max(1, cfg.num_layers // cfg.attn_res_num_blocks)
    for index, layer in enumerate(layers):
        if index % segment_size == 0 and index // segment_size < cfg.attn_res_num_blocks:
            blocks = (*blocks, partial)
            block_logits = (
                *block_logits,
                native._attn_res_source_logits(partial, queries[2 * index :], eps),
            )
            partial = None
        use_long = native._is_long_layer(index, cfg.num_layers, cfg.global_every, cfg.global_layers)
        layer_mask = long_mask if use_long else short_mask
        partial_before = partial
        attn_input, _, _ = native._attn_res_mix(blocks, block_logits, partial, queries, 2 * index, eps)
        attn_output, _ = layer.attn_branch(attn_input, layer_mask, use_long, use_long, token_ids)
        if interventions:
            attn_output = (attn_output.astype(jnp.float32) + interventions[2 * index]).astype(attn_output.dtype)
        partial = attn_output if partial is None else partial + attn_output
        captures.append(SublayerCapture(blocks, partial_before, attn_input, attn_output, partial))

        partial_before = partial
        mlp_input, _, _ = native._attn_res_mix(blocks, block_logits, partial, queries, 2 * index + 1, eps)
        mlp_output, mlp_stats = layer.mlp_branch(mlp_input, layer_mask)
        shared_preactivations.append(mlp_stats[native._SHARED_PRE])
        routing_selected.append(mlp_stats[native._ROUTING_SELECTED])
        routing_weights.append(mlp_stats[native._ROUTING_WEIGHTS])
        routing_logits.append(mlp_stats[native._ROUTING_LOGITS])
        routed_inputs.append(mlp_stats[native._ROUTED_INPUT])
        if interventions:
            mlp_output = (mlp_output.astype(jnp.float32) + interventions[2 * index + 1]).astype(mlp_output.dtype)
        partial = partial + mlp_output
        captures.append(SublayerCapture(blocks, partial_before, mlp_input, mlp_output, partial))

    final_index = len(queries) - 1
    final_sources = (*blocks, partial)
    logits = [native._block_logit(logits, queries, final_index) for logits in block_logits]
    logits.append(native._attn_res_source_logits(partial, queries[-1:], eps)[0])
    _, final_stream = native._softmax_mix(logits, list(final_sources))
    final_stream = native._activation_boundary(native._batch_reshard(final_stream.astype(embedding.dtype)))
    return model.final_norm(final_stream), ForwardCapture(
        token_rows,
        bigram_ids,
        bigram_rows,
        embedding,
        bigram_source,
        tuple(captures),
        final_sources,
        final_stream,
        tuple(shared_preactivations),
        segments,
        tuple(routing_selected),
        tuple(routing_weights),
        tuple(routing_logits),
        tuple(routed_inputs),
    )
