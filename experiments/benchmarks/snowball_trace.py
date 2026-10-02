# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded diagnostic copy of Snowball paged decode with auxiliary JIT outputs."""

import dataclasses

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard
from levanter.grug.sharding import _current_mesh, _mesh_axis_size, unshard
from levanter.layers.kv_cache import KvPageCache, ListCache
from levanter.models.snowball import (
    _BATCH_AXES,
    _ROUTING_RENORM_SUM,
    _activation_reshard,
    _activation_spec,
    _long_attention_schedule,
)


def _moe_with_trace(mlp, hidden, token_valid):
    # Match SnowballMoEMLP's routing and actual expert dispatch. The selected
    # arrays returned here are the same arrays consumed by expert_mlp below.
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = jnp.einsum("td,de->te", flat, reshard(mlp.router, P(None, None))).astype(jnp.float32)
    biased = logits + unshard(mlp.router_bias)
    _, selected = jax.lax.top_k(biased, mlp.cfg.num_experts_per_token + 1)
    selected = selected[:, :-1]
    weights = jax.nn.sigmoid(jnp.take_along_axis(logits, selected, axis=-1))
    weights = (weights * (_ROUTING_RENORM_SUM / (jnp.sum(weights, axis=-1, keepdims=True) + 1e-9))).astype(hidden.dtype)
    output = mlp.expert_mlp(
        flat,
        selected.astype(jnp.int32),
        weights,
        token_valid=token_valid,
        mesh=get_abstract_mesh(),
        report_capacity_overflow=False,
    )
    return _activation_reshard(output.reshape(hidden.shape)), logits, selected, weights


def decode_with_trace(model, input_ids, kv_cache, batch_info, pos_ids):
    """Return actual logits, caches, and stage tensors together in one diagnostic JIT.

    This mirrors the paged decoder's layer scan and uses its attention and expert
    implementations. Retaining auxiliary outputs can change XLA fusion; callers
    must report the difference from the ordinary decoder's logits.
    """
    tokens = reshard(input_ids.array[:, None], P(_BATCH_AXES, None))
    hidden = model.transformer.token_embed.at[tokens].get(out_sharding=_activation_spec())
    trace = {"embedding": hidden[:, 0]}
    hidden = model.transformer.embed_gated_norm(model.transformer.embed_norm(hidden))
    trace["post_embed_norm_gate"] = hidden[:, 0]
    token_valid = jnp.arange(tokens.shape[0]) < batch_info.num_new_tokens
    stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *model.transformer.blocks)
    expert_size = _mesh_axis_size(_current_mesh(), "expert")
    if expert_size > 1:
        experts = dataclasses.replace(stacked.mlp.expert_mlp, implementation="ring", capacity_factor=float(expert_size))
        stacked = eqx.tree_at(lambda block: block.mlp.expert_mlp, stacked, experts)
    pages = jnp.stack([cache.kv_pages.array for cache in kv_cache])
    cache_axes = kv_cache[0].kv_pages.axes

    def layer_step(hidden, layer_data):
        block, pages, use_long = layer_data
        cache = KvPageCache(hax.named(pages, cache_axes))
        attention_input = block.attn_gated_norm(block.rms_attn(hidden))
        attention_output, cache = jax.lax.cond(
            use_long,
            lambda _: block.attn.decode(attention_input, cache, batch_info, pos_ids.array, use_long=True),
            lambda _: block.attn.decode(attention_input, cache, batch_info, pos_ids.array, use_long=False),
            operand=None,
        )
        residual = hidden + attention_output
        mlp_input = block.mlp_gated_norm(block.rms_mlp(residual))
        routed, logits, selected, weights = _moe_with_trace(block.mlp, mlp_input, token_valid)
        shared = block.shared(mlp_input)
        output = residual + (routed + shared)
        stages = {
            "layer_input": hidden[:, 0],
            "attention_input": attention_input[:, 0],
            "attention_output": attention_output[:, 0],
            "post_attention_residual": residual[:, 0],
            "mlp_input": mlp_input[:, 0],
            "routed_output": routed[:, 0],
            "shared_output": shared[:, 0],
            "layer_output": output[:, 0],
            "router_logits": logits,
            "expert_ids": selected,
            "expert_weights": weights,
        }
        return output, (cache.kv_pages.array, stages)

    hidden, (pages, stages) = jax.lax.scan(layer_step, hidden, (stacked, pages, _long_attention_schedule(len(kv_cache))))
    trace.update(stages)
    trace["pre_final_norm"] = hidden[:, 0]
    hidden = model.transformer.final_norm(hidden)
    trace["post_final_norm"] = hidden[:, 0]
    hidden = model.transformer.final_gated_norm(hidden)
    trace["post_final_gate"] = hidden[:, 0]
    logits = jnp.einsum("bsd,dv->bsv", hidden, model.transformer.output_proj, out_sharding=_activation_spec("model"))
    caches = ListCache(tuple(KvPageCache(hax.named(pages[i], cache_axes)) for i in range(len(kv_cache))))
    return hax.named(logits[:, 0], (input_ids.axes[0], model.Vocab)), caches, trace
