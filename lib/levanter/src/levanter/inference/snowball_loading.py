# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound the extra device memory needed to load a Snowball HF snapshot."""

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.partitioning import ResourceMapping

from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.models.snowball import (
    SnowballBlock,
    SnowballConfig,
    SnowballLMHeadModel,
    snowball_block_from_state_dict,
    snowball_embeddings_from_state_dict,
    snowball_final_from_state_dict,
)


# Normalize the key so conversion reuses one compiled function across layer indices.
_LOCAL_LAYER_INDEX = 0


def load_snowball_serving_model(
    converter: HFCheckpointConverter,
    ref: RepoRef,
    config: SnowballConfig,
    *,
    dtype: jnp.dtype,
    axis_mapping: ResourceMapping,
    pack_expert_weights: bool = False,
) -> SnowballLMHeadModel:
    """Load a Snowball snapshot with device conversion memory bounded to one layer.

    Strict replay stores gate and up in their grouped-matmul layout at load time.
    """
    state_dict = converter.load_state_dict(ref, dtype)
    prefix = converter.ignore_prefix
    if prefix and not any(name.startswith(f"{prefix}.") for name in state_dict):
        prefix = None
    vocab = Axis("vocab", config.vocab_size)
    template = eqx.filter_eval_shape(SnowballLMHeadModel.init, vocab, config, key=jax.random.PRNGKey(0))
    pack_expert_weights = pack_expert_weights or config.replay_precision == "zero_kl"
    with hax.axis_mapping(axis_mapping):
        return _convert_layers(template, state_dict, prefix, pack_expert_weights=pack_expert_weights)


def _convert_layers(
    template: SnowballLMHeadModel,
    state_dict: dict[str, jax.Array],
    prefix: str | None,
    *,
    pack_expert_weights: bool = False,
) -> SnowballLMHeadModel:
    """Convert a resident HF state dict one layer at a time, releasing the inputs."""
    layer_prefix = f"{prefix}.model.layers." if prefix else "model.layers."
    weights_by_layer: list[dict[str, jax.Array]] = [{} for _ in template.transformer.blocks]
    for name in tuple(state_dict):
        if name.startswith(layer_prefix):
            index, _, suffix = name.removeprefix(layer_prefix).partition(".")
            layer_index = int(index)
            if not 0 <= layer_index < len(weights_by_layer) or not suffix:
                raise ValueError(f"Checkpoint has an unexpected Snowball layer key {name!r}")
            weights_by_layer[layer_index][f"{layer_prefix}{_LOCAL_LAYER_INDEX}.{suffix}"] = state_dict.pop(name)

    transformer = eqx.filter_jit(snowball_embeddings_from_state_dict)(template.transformer, state_dict, prefix)
    transformer = eqx.filter_jit(snowball_final_from_state_dict)(transformer, state_dict, prefix)
    jax.block_until_ready(transformer)
    state_dict.clear()

    convert_block = eqx.filter_jit(snowball_block_from_state_dict)
    pack_block = eqx.filter_jit(SnowballBlock.with_packed_experts)
    blocks = []
    for block, weights in zip(transformer.blocks, weights_by_layer, strict=True):
        block = jax.block_until_ready(convert_block(block, weights, _LOCAL_LAYER_INDEX, prefix))
        if pack_expert_weights:
            block = jax.block_until_ready(pack_block(block))
        blocks.append(block)
        weights.clear()
    transformer = eqx.tree_at(lambda value: value.blocks, transformer, tuple(blocks))
    return eqx.tree_at(lambda model: model.transformer, template, transformer)
