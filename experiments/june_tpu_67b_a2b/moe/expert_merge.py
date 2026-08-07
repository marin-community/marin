# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-pair expert-bank conversion for the array-stacked June Grug MoE."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn import ArrayStacked
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MoEExpertMlp

from experiments.grug.moe.expert_merge import MoeLayerTrace, validate_bijection
from experiments.june_tpu_67b_a2b.moe.model import MoeBlockTrace, MoEMLP, Transformer

_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _require_stacked_model(model: Transformer) -> tuple[ArrayStacked, ArrayStacked]:
    if model.stacked_blocks is None or not isinstance(model.expert_banks, ArrayStacked):
        raise ValueError("June one-pair conversion currently requires array-stacked blocks and expert banks")
    return model.stacked_blocks, model.expert_banks


def stacked_expert_bank_at(expert_banks: ArrayStacked, bank_index: jax.Array | int) -> MoEExpertMlp:
    """Select one bank from an array-stacked expert container with execution sharding."""
    stacked = expert_banks.stacked
    return dataclasses.replace(
        stacked,
        w_gate=stacked.w_gate.at[bank_index].get(out_sharding=P("expert", "data", "model")),
        w_up=stacked.w_up.at[bank_index].get(out_sharding=P("expert", "data", "model")),
        w_down=stacked.w_down.at[bank_index].get(out_sharding=P("expert", "model", "data")),
    )


def expert_bank_for_layer(model: Transformer, layer_index: int) -> MoEExpertMlp:
    """Return one layer's explicit expert bank with its execution sharding restored."""
    _, expert_banks = _require_stacked_model(model)
    if not 0 <= layer_index < model.config.num_layers:
        raise IndexError(f"layer_index must be in [0, {model.config.num_layers}), got {layer_index}")
    bank_index = model.config.resolved_expert_bank_for_layer[layer_index]
    return stacked_expert_bank_at(expert_banks, bank_index)


def _sample_block_trace(trace: MoeBlockTrace, token_indices: jax.Array | None) -> MoeLayerTrace:
    mlp_input = trace.mlp_input.reshape(-1, trace.mlp_input.shape[-1])
    routed_output = trace.routed_output.reshape(-1, trace.routed_output.shape[-1])
    selected_experts = trace.selected_experts
    combine_weights = trace.combine_weights
    if token_indices is not None:
        mesh = get_abstract_mesh()
        batch_shards = int(np.prod([mesh.shape[axis] for axis in _BATCH_AXES]))
        if token_indices.ndim != 1 or token_indices.shape[0] % batch_shards != 0:
            raise ValueError(
                "token_indices must be one-dimensional with a length divisible by the batch mesh size; "
                f"got shape {token_indices.shape} for {batch_shards} shards"
            )
        vector_sharding = jax.sharding.NamedSharding(mesh, P(_BATCH_AXES, None))
        mlp_input = mlp_input.at[token_indices].get(out_sharding=vector_sharding)
        routed_output = routed_output.at[token_indices].get(out_sharding=vector_sharding)
        selected_experts = selected_experts.at[token_indices].get(out_sharding=vector_sharding)
        combine_weights = combine_weights.at[token_indices].get(out_sharding=vector_sharding)
    return MoeLayerTrace(
        mlp_input=mlp_input,
        selected_experts=selected_experts,
        combine_weights=combine_weights,
        routed_output=routed_output,
    )


def forward_with_moe_traces(
    model: Transformer,
    token_ids: jax.Array,
    *,
    target_layers: tuple[int, ...],
    token_indices: jax.Array | None = None,
    mask: AttentionMask | jax.Array | None = None,
) -> tuple[jax.Array, dict[int, MoeLayerTrace], jax.Array]:
    """Run the stacked model and return sampled MoE traces plus per-layer capacity overflow."""
    stacked_blocks, expert_banks = _require_stacked_model(model)
    if mask is None:
        mask = AttentionMask.causal()
    if len(set(target_layers)) != len(target_layers):
        raise ValueError(f"target_layers contains duplicates: {target_layers}")
    if any(layer < 0 or layer >= model.config.num_layers for layer in target_layers):
        raise IndexError(f"target_layers must lie in [0, {model.config.num_layers}), got {target_layers}")
    ordered_targets = tuple(sorted(target_layers))

    cfg = model.config
    segment_ids = mask.segment_ids if isinstance(mask, AttentionMask) else None
    short_mask = AttentionMask(is_causal=True, sliding_window=cfg.sliding_window, segment_ids=segment_ids)
    long_mask = AttentionMask(is_causal=True, sliding_window=None, segment_ids=segment_ids)
    mask_schedule = ((jnp.arange(cfg.num_layers) % 4) == 3) | (jnp.arange(cfg.num_layers) == cfg.num_layers - 1)
    bank_schedule = jnp.asarray(cfg.resolved_expert_bank_for_layer, dtype=jnp.int32)

    def scan_layer(carry: jax.Array, scan_inputs):
        layer, use_long_mask, bank_index = scan_inputs
        expert_bank = expert_banks.get_layer(bank_index)
        block_trace = layer.forward_with_moe_trace(
            carry,
            short_mask,
            long_mask,
            use_long_mask,
            expert_bank,
            False,
            cfg.disable_long_rope,
        )
        sampled_trace = _sample_block_trace(block_trace, token_indices)
        trace_arrays = (
            sampled_trace.mlp_input,
            sampled_trace.selected_experts,
            sampled_trace.combine_weights,
            sampled_trace.routed_output,
        )
        return block_trace.hidden, (trace_arrays, block_trace.router_stats["capacity_overflow"])

    hidden, (stacked_trace_arrays, capacity_overflow) = jax.lax.scan(
        scan_layer,
        model.embed_inputs(token_ids),
        xs=(stacked_blocks.stacked, mask_schedule, bank_schedule),
    )
    traces = {
        target_layer: MoeLayerTrace(*(value[target_layer] for value in stacked_trace_arrays))
        for target_layer in ordered_targets
    }
    return model.finalize_hidden(hidden), traces, capacity_overflow


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


__all__ = [
    "convert_one_expert_pair",
    "expert_bank_for_layer",
    "forward_with_moe_traces",
    "permute_pending_qb_beta",
    "stacked_expert_bank_at",
]
