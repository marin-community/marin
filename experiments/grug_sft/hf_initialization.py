# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load the public Snowball HF export into the native June Grug trainer."""

import dataclasses
from collections.abc import Sequence
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.nn import ArrayStacked
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel, SnowballTransformer

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer


def _copy_matching_parameters(source: Any, target: Any) -> Any:
    """Copy array leaves from a structurally matching module into a target template."""
    if isinstance(target, eqx.Module):
        values = {}
        for field in dataclasses.fields(target):
            target_value = getattr(target, field.name)
            if field.metadata.get("static", False) or not hasattr(source, field.name):
                values[field.name] = target_value
            else:
                values[field.name] = _copy_matching_parameters(getattr(source, field.name), target_value)
        return type(target)(**values)
    if isinstance(target, tuple) and isinstance(source, Sequence):
        if len(source) != len(target):
            raise ValueError(f"Module sequence length mismatch: {len(source)} != {len(target)}")
        return tuple(
            _copy_matching_parameters(source_item, target_item)
            for source_item, target_item in zip(source, target, strict=True)
        )
    if isinstance(target, jax.ShapeDtypeStruct) or eqx.is_array(target):
        if not eqx.is_array(source):
            raise TypeError(f"Expected an array source for target {target}")
        if source.shape != target.shape:
            raise ValueError(f"Parameter shape mismatch: {source.shape} != {target.shape}")
        return source
    return target


def _stack_blocks(blocks: tuple, template: ArrayStacked) -> ArrayStacked:
    stacked = jax.tree.map(lambda *layers: jnp.stack(layers), *blocks)
    return dataclasses.replace(template, stacked=stacked)


def vendored_transformer_from_snowball(
    source: SnowballTransformer,
    target_config: GrugModelConfig,
    *,
    key: jax.Array,
) -> Transformer:
    """Convert a loaded Snowball snapshot to the native training model layout."""
    shape_fields = (
        "vocab_size",
        "hidden_dim",
        "intermediate_dim",
        "shared_expert_intermediate_dim",
        "num_experts",
        "num_experts_per_token",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "inferred_head_dim",
    )
    mismatches = {
        field: (getattr(source.config, field), getattr(target_config, field))
        for field in shape_fields
        if getattr(source.config, field) != getattr(target_config, field)
    }
    if mismatches:
        raise ValueError(f"Snowball HF architecture does not match the training model: {mismatches}")

    unstacked_config = dataclasses.replace(target_config, use_array_stacked_blocks=False)
    unstacked_template = eqx.filter_eval_shape(Transformer.init, unstacked_config, key=key)
    converted = cast(Transformer, _copy_matching_parameters(source, unstacked_template))
    if not target_config.use_array_stacked_blocks:
        return converted

    assert converted.blocks is not None
    stacked_template_model = eqx.filter_eval_shape(Transformer.init, target_config, key=key)
    assert stacked_template_model.stacked_blocks is not None
    return Transformer(
        token_embed=converted.token_embed,
        embed_norm=converted.embed_norm,
        embed_gated_norm=converted.embed_gated_norm,
        output_proj=converted.output_proj,
        blocks=None,
        stacked_blocks=_stack_blocks(converted.blocks, stacked_template_model.stacked_blocks),
        final_norm=converted.final_norm,
        final_gated_norm=converted.final_gated_norm,
        config=target_config,
    )


def load_vendored_transformer_from_hf(
    target_config: GrugModelConfig,
    checkpoint_path: str,
    *,
    key: jax.Array,
    dtype: jnp.dtype,
) -> Transformer:
    """Load and shard the pinned HF weights, then convert them to the training layout."""
    converter = SnowballConfig(reference_checkpoint=checkpoint_path, tokenizer=checkpoint_path).hf_checkpoint_converter()
    loaded = converter.load_pretrained(
        SnowballLMHeadModel,
        ref=checkpoint_path,
        axis_mapping={},
        dtype=dtype,
    )
    if not isinstance(loaded, SnowballLMHeadModel):
        raise TypeError(f"Expected SnowballLMHeadModel, got {type(loaded).__name__}")
    return vendored_transformer_from_snowball(loaded.transformer, target_config, key=key)


def pending_qb_betas_from_export(model: Transformer) -> jax.Array:
    """Recover QB state from the effective router bias stored in the HF export."""
    if model.stacked_blocks is not None:
        router_bias = model.stacked_blocks.stacked.mlp.router_bias
    else:
        assert model.blocks is not None
        router_bias = jnp.stack([block.mlp.router_bias for block in model.blocks])
    return -router_bias
