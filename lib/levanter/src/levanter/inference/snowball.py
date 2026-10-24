# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Paged Snowball decode with replicated attention and expert-parallel MoE."""

import dataclasses
import math

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from jax.sharding import PartitionSpec as P

from levanter.grug.grug_moe import moe_mlp
from levanter.grug.sharding import _current_mesh, _mesh_axis_size, _reshard_for_init
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.layers.attention import ragged_paged_attention
from levanter.layers.kv_cache import KvPageCache, PageCache

from levanter.models.snowball import SnowballLMHeadModel, long_attention_layer_mask, rms_norm


class SnowballCache(PageCache):
    # Keep the scan's layer dimension in storage so donated cache buffers can
    # alias its result without concatenating and splitting every decode call.
    kv_pages: NamedArray
    dropped_assignments: jax.Array

    def reset(self):
        return SnowballCache(hax.zeros_like(self.kv_pages), jnp.zeros_like(self.dropped_assignments))

    def copy_page(self, src_page: int, dst_page: int):
        pages = self.kv_pages.at["page", dst_page].set(self.kv_pages["page", src_page])
        return dataclasses.replace(self, kv_pages=pages)


def initial_cache(model: "SnowballLMHeadModel", spec: PageTableSpec, *, dtype) -> SnowballCache:
    cfg = model.config
    pages = hax.zeros(
        {
            "layer": cfg.num_layers,
            "page": spec.num_pages,
            "slot": spec.page_size,
            "kv_head": 2 * cfg.num_kv_heads,
            "head_size": cfg.inferred_head_dim,
        },
        dtype=dtype,
    )
    # Attention is replicated; the expert mesh only partitions expert weights and MoE tokens.
    pages = jax.tree.map(lambda x: _reshard_for_init(x, P()), pages)
    return SnowballCache(pages, jnp.zeros((2,), dtype=jnp.int32))


def _rotate(x, positions, theta):
    rotary_size = x.shape[-1] // 2
    frequencies = 1.0 / (theta ** (jnp.arange(rotary_size // 2, dtype=jnp.float32) / (rotary_size // 2)))
    angles = positions[:, None, None].astype(jnp.float32) * frequencies[None, None, :]
    a, b = jnp.split(x[..., :rotary_size], 2, axis=-1)
    rotated = jnp.concatenate(
        [a * jnp.cos(angles) - b * jnp.sin(angles), b * jnp.cos(angles) + a * jnp.sin(angles)], axis=-1
    ).astype(x.dtype)
    return jnp.concatenate([rotated, x[..., rotary_size:]], axis=-1)


def _linear(x, weight):
    return jnp.einsum("td,dh->th", x, weight, out_sharding=P())


def _attention(attn, x, cache, batch_info, positions, *, use_long):
    cfg = attn.cfg
    tokens = x.shape[0]
    dim = cfg.inferred_head_dim
    q = rms_norm((_linear(x, attn.w_q)).reshape(tokens, cfg.num_heads, dim))
    k = rms_norm((_linear(x, attn.w_k)).reshape(tokens, cfg.num_kv_heads, dim))
    v = (_linear(x, attn.w_v)).reshape(tokens, cfg.num_kv_heads, dim)
    if not use_long:
        q, k = _rotate(q, positions, cfg.rope.theta), _rotate(k, positions, cfg.rope.theta)
    q = q * cfg.qk_mult
    axes = (Axis("position", tokens), Axis("kv_head", cfg.num_kv_heads), Axis("head_size", dim))
    cache = cache.update(batch_info, hax.named(k, axes), hax.named(v, axes))
    groups = cfg.num_heads // cfg.num_kv_heads
    queries = hax.named(
        q.reshape(tokens, cfg.num_kv_heads, groups, dim), (*axes[:2], Axis("q_heads_per_group", groups), axes[-1])
    )
    with hax.axis_mapping({}):
        output = (
            ragged_paged_attention(
                queries,
                cache.kv_pages,
                batch_info.seq_lens,
                batch_info.page_indices,
                batch_info.cu_q_lens,
                batch_info.num_seqs,
                sm_scale=1.0 / math.sqrt(dim),
                sliding_window=None if use_long else cfg.sliding_window,
                allow_fallback=False,
            )
            .array.reshape(tokens, cfg.num_heads, dim)
            .astype(x.dtype)
        )
    aligned_v = jnp.repeat(v, groups, axis=1)
    dot = jnp.sum(output * aligned_v, axis=-1, keepdims=True)
    norm = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
    output = output - dot / (norm + 1e-6) * aligned_v
    output = output * (2 * jax.nn.sigmoid(_linear(x, attn.attn_gate)))[..., None]
    return _linear(output.reshape(tokens, -1), attn.w_o), cache


def _moe(mlp, x, valid):
    cfg = mlp.cfg
    ep = _mesh_axis_size(_current_mesh(), "expert")
    padding = (-x.shape[0]) % ep
    x = jnp.pad(x, ((0, padding), (0, 0)))
    valid = jnp.pad(valid, ((0, padding),))
    # Split a single global token pack; EP gathers disjoint shards, not replicated copies.
    x = _reshard_for_init(x, P("expert", None))
    valid = _reshard_for_init(valid, P("expert"))
    logits = (x @ mlp.router).astype(jnp.float32)
    _, selected = jax.lax.top_k(logits + _reshard_for_init(mlp.router_bias, P()), cfg.num_experts_per_token + 1)
    selected = selected[:, :-1]
    weights = jax.nn.sigmoid(jnp.take_along_axis(logits, selected, axis=-1))
    weights = (weights * (2.5 / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9))).astype(x.dtype)
    experts = mlp.expert_mlp
    output, counts = moe_mlp(
        x,
        selected.astype(jnp.int32),
        weights,
        jnp.concatenate([experts.w_gate, experts.w_up], axis=-1),
        experts.w_down,
        implementation="ring",
        activation=jax.nn.silu,
        token_valid=valid,
        capacity_factor=float(ep),
        report_capacity_overflow=True,
    )
    output = _reshard_for_init(output, P())
    if padding:
        output = output[:-padding]
    return output, jnp.stack([counts.sender_dropped, counts.receiver_dropped])


def decode(
    model: "SnowballLMHeadModel",
    input_ids: NamedArray,
    cache: SnowballCache,
    batch_info: PageBatchInfo,
    pos_ids: NamedArray,
):
    cfg = model.config
    mesh = _current_mesh()
    if any(_mesh_axis_size(mesh, axis) != 1 for axis in ("data", "context", "model", "replica_dcn")):
        raise ValueError(
            "Snowball paged inference currently supports expert parallelism with replicated attention only"
        )
    tf = model.transformer
    ids = _reshard_for_init(input_ids.array, P())
    positions = _reshard_for_init(pos_ids.array, P())
    if ids.ndim != 1:
        raise ValueError("Snowball paged decode requires a packed one-dimensional token buffer")
    valid = jnp.arange(ids.size) < batch_info.num_new_tokens
    hidden = tf.embed_gated_norm(tf.embed_norm(tf.token_embed.at[ids].get(out_sharding=P())))
    stacked_blocks = jax.tree.map(lambda *blocks: jnp.stack(blocks), *tf.blocks)
    stacked_pages = cache.kv_pages.array
    page_axes = cache.kv_pages.axes[1:]
    schedule = long_attention_layer_mask(cfg.num_layers)

    def layer_step(carry, layer_inputs):
        hidden, drops = carry
        block, pages, use_long = layer_inputs
        attention_input = block.attn_gated_norm(block.rms_attn(hidden))
        layer_cache = KvPageCache(hax.named(pages, page_axes))
        attn_output, layer_cache = jax.lax.cond(
            use_long,
            lambda _: _attention(block.attn, attention_input, layer_cache, batch_info, positions, use_long=True),
            lambda _: _attention(block.attn, attention_input, layer_cache, batch_info, positions, use_long=False),
            operand=None,
        )
        hidden = hidden + attn_output
        mlp_input = block.mlp_gated_norm(block.rms_mlp(hidden))
        routed, layer_drops = _moe(block.mlp, mlp_input, valid)
        shared = jax.nn.silu(_linear(mlp_input, block.shared.w_gate)) * _linear(mlp_input, block.shared.w_up)
        hidden = hidden + (routed + _linear(shared, block.shared.w_down))
        return (hidden, drops + layer_drops), layer_cache.kv_pages.array

    (hidden, drops), stacked_pages = jax.lax.scan(
        layer_step, (hidden, cache.dropped_assignments), (stacked_blocks, stacked_pages, schedule)
    )
    hidden = tf.final_gated_norm(tf.final_norm(hidden))
    logits = _linear(hidden, tf.output_proj)
    return hax.named(logits, (*input_ids.axes, model.Vocab)), SnowballCache(
        hax.named(stacked_pages, cache.kv_pages.axes), drops
    )
