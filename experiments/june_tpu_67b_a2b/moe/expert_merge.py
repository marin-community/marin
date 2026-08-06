# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-pair expert-bank conversion for the array-stacked June Grug MoE."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn import ArrayStacked
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import validate_bijection
from experiments.june_tpu_67b_a2b.moe.model import MoEMLP, Transformer


def _require_stacked_model(model: Transformer) -> tuple[ArrayStacked, ArrayStacked]:
    if model.stacked_blocks is None or not isinstance(model.expert_banks, ArrayStacked):
        raise ValueError("June one-pair conversion currently requires array-stacked blocks and expert banks")
    return model.stacked_blocks, model.expert_banks


def _take_leading_axis(value: jax.Array, indices: tuple[int, ...]) -> jax.Array:
    retained = jnp.asarray(indices, dtype=jnp.int32)
    selected = value.at[retained].get()
    sharding = getattr(value, "sharding", None)
    if sharding is not None:
        selected = jax.sharding.reshard(selected, sharding)
    return selected


def _replace_stacked_bank(
    stacked_banks: MoEExpertMlp,
    bank_index: int,
    replacement: MoEExpertMlp,
) -> MoEExpertMlp:
    if jax.tree.structure(stacked_banks) != jax.tree.structure(replacement):
        raise ValueError("replacement bank must have the same pytree structure as the stacked bank")

    def replace(stacked: jax.Array, new_value: jax.Array) -> jax.Array:
        if stacked.shape[1:] != new_value.shape:
            raise ValueError(
                f"replacement bank leaf shape {new_value.shape} does not match stacked leaf shape {stacked.shape[1:]}"
            )
        return stacked.at[bank_index].set(new_value)

    return jax.tree.map(replace, stacked_banks, replacement)


def _permute_stacked_router(
    router: MoEMLP,
    *,
    layer_index: int,
    source_to_shared: np.ndarray,
) -> MoEMLP:
    shared_to_source = jnp.asarray(np.argsort(source_to_shared), dtype=jnp.int32)
    layer_router = router.router[layer_index].at[:, shared_to_source].get()
    layer_bias = router.router_bias[layer_index].at[shared_to_source].get()
    return dataclasses.replace(
        router,
        router=router.router.at[layer_index].set(layer_router),
        router_bias=router.router_bias.at[layer_index].set(layer_bias),
    )


def permute_pending_qb_beta(
    pending_qb_beta: jax.Array,
    *,
    layer_index: int,
    source_to_shared: np.ndarray | jax.Array,
) -> jax.Array:
    """Rename one layer's pending QB slots with the same expert permutation as its router."""
    if pending_qb_beta.ndim != 2:
        raise ValueError(f"pending_qb_beta must have shape [layers, experts], got {pending_qb_beta.shape}")
    if not 0 <= layer_index < pending_qb_beta.shape[0]:
        raise IndexError(f"layer_index must be in [0, {pending_qb_beta.shape[0]}), got {layer_index}")
    permutation = validate_bijection(source_to_shared, int(pending_qb_beta.shape[1]))
    shared_to_source = jnp.asarray(np.argsort(permutation), dtype=jnp.int32)
    permuted = pending_qb_beta[layer_index].at[shared_to_source].get()
    return pending_qb_beta.at[layer_index].set(permuted)


def convert_one_expert_pair(
    model: Transformer,
    *,
    representative_layer: int,
    source_layer: int,
    source_to_shared: np.ndarray | jax.Array,
    shared_bank: MoEExpertMlp | None = None,
) -> Transformer:
    """Merge one source layer into a representative bank while preserving its router by ID renaming."""
    stacked_blocks, expert_banks = _require_stacked_model(model)
    if representative_layer == source_layer:
        raise ValueError("representative_layer and source_layer must be different")
    if not 0 <= representative_layer < model.config.num_layers:
        raise IndexError(f"representative_layer must be in [0, {model.config.num_layers}), got {representative_layer}")
    if not 0 <= source_layer < model.config.num_layers:
        raise IndexError(f"source_layer must be in [0, {model.config.num_layers}), got {source_layer}")

    permutation = validate_bijection(source_to_shared, model.config.num_experts)
    bank_mapping = model.config.resolved_expert_bank_for_layer
    representative_bank = bank_mapping[representative_layer]
    source_bank = bank_mapping[source_layer]
    if representative_bank == source_bank:
        raise ValueError("the selected layers already share an expert bank")
    if bank_mapping.count(representative_bank) != 1 or bank_mapping.count(source_bank) != 1:
        raise ValueError("the initial one-pair conversion requires both source banks to be used by one layer")

    retained_old_banks = tuple(bank for bank in range(expert_banks.num_layers) if bank != source_bank)
    old_to_new = {old_bank: new_bank for new_bank, old_bank in enumerate(retained_old_banks)}
    representative_new_bank = old_to_new[representative_bank]
    converted_mapping = tuple(
        representative_new_bank if layer_index == source_layer else old_to_new[bank_id]
        for layer_index, bank_id in enumerate(bank_mapping)
    )
    converted_config = dataclasses.replace(model.config, expert_bank_for_layer=converted_mapping)

    stacked_bank_values = expert_banks.stacked
    if shared_bank is not None:
        stacked_bank_values = _replace_stacked_bank(stacked_bank_values, representative_bank, shared_bank)
    retained_bank_values = jax.tree.map(
        lambda value: _take_leading_axis(value, retained_old_banks),
        stacked_bank_values,
    )
    converted_banks = dataclasses.replace(
        expert_banks,
        stacked=retained_bank_values,
        num_layers=len(retained_old_banks),
    )

    converted_router = _permute_stacked_router(
        stacked_blocks.stacked.mlp,
        layer_index=source_layer,
        source_to_shared=permutation,
    )
    converted_router = dataclasses.replace(converted_router, cfg=converted_config)
    converted_attention = dataclasses.replace(stacked_blocks.stacked.attn, cfg=converted_config)
    converted_block_values = dataclasses.replace(
        stacked_blocks.stacked,
        mlp=converted_router,
        attn=converted_attention,
    )
    converted_blocks = dataclasses.replace(stacked_blocks, stacked=converted_block_values)
    return dataclasses.replace(
        model,
        stacked_blocks=converted_blocks,
        expert_banks=converted_banks,
        config=converted_config,
    )


__all__ = ["convert_one_expert_pair", "permute_pending_qb_beta"]
