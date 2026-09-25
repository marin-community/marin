# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Topology-independent checkpoints for the Hero pipeline variant."""

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
from levanter import mpmd_checkpoint
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.checkpoint import save_checkpoint as save_levanter_checkpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_pipeline.pipeline import GrugMoeAutomaticPipelineState, GrugMoePipelineStage


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class HeroPipelineCheckpointState:
    """Full Hero state with layers independent of their pipeline partition."""

    params: GrugMoePipelineStage
    opt_state: optax.OptState
    pending_qb_betas: tuple[jax.Array, ...]


def _merge_stages(stages: tuple[GrugMoePipelineStage, ...]) -> GrugMoePipelineStage:
    first, last = stages[0], stages[-1]
    return GrugMoePipelineStage(
        token_embed=first.token_embed,
        embed_norm=first.embed_norm,
        embed_gated_norm=first.embed_gated_norm,
        output_proj=last.output_proj,
        blocks=tuple(block for stage in stages for block in stage.blocks),
        final_norm=last.final_norm,
        final_gated_norm=last.final_gated_norm,
        config=first.config,
        start_layer=0,
        end_layer=last.end_layer,
    )


def _split_stage(stage: GrugMoePipelineStage, layer_counts: tuple[int, ...]) -> tuple[GrugMoePipelineStage, ...]:
    result = []
    start = 0
    for index, count in enumerate(layer_counts):
        end = start + count
        result.append(
            GrugMoePipelineStage(
                token_embed=stage.token_embed if index == 0 else None,
                embed_norm=stage.embed_norm if index == 0 else None,
                embed_gated_norm=stage.embed_gated_norm if index == 0 else None,
                output_proj=stage.output_proj if index == len(layer_counts) - 1 else None,
                blocks=stage.blocks[start:end],
                final_norm=stage.final_norm if index == len(layer_counts) - 1 else None,
                final_gated_norm=stage.final_gated_norm if index == len(layer_counts) - 1 else None,
                config=stage.config,
                start_layer=start,
                end_layer=end,
            )
        )
        start = end
    if start != len(stage.blocks):
        raise ValueError(f"destination stages cover {start} of {len(stage.blocks)} Hero layers")
    return tuple(result)


def _unstack_pending(value: jax.Array) -> tuple[jax.Array, ...]:
    spec = (*value.sharding.spec, None, None)
    assert spec[0] is None
    sharding = NamedSharding(value.sharding.mesh, P(spec[1]))
    return tuple(
        jax.make_array_from_single_device_arrays(
            value.shape[1:], sharding, [shard.data[index] for shard in value.addressable_shards], dtype=value.dtype
        )
        for index in range(value.shape[0])
    )


def _stack_pending(values: tuple[jax.Array, ...], sharding: NamedSharding) -> jax.Array:
    buffers = [{shard.device: shard.data for shard in value.addressable_shards} for value in values]
    return jax.make_array_from_single_device_arrays(
        (len(values), *values[0].shape),
        sharding,
        [
            jnp.stack([buffer[device] for buffer in buffers])
            for device in sharding.mesh.devices.flat
            if device in buffers[0]
        ],
        dtype=values[0].dtype,
    )


def checkpoint_state(state: GrugMoeAutomaticPipelineState) -> HeroPipelineCheckpointState:
    """Expose stage-local parameters, optimizer state, and router updates by layer."""
    arrays = mpmd_checkpoint.checkpoint_arrays(state)
    optimizer = jax.tree.map(
        lambda *values: _merge_stages(values) if isinstance(values[0], GrugMoePipelineStage) else values[0],
        *arrays.opt_state,
        is_leaf=lambda value: isinstance(value, GrugMoePipelineStage),
    )
    return HeroPipelineCheckpointState(
        params=_merge_stages(arrays.trainable_params),
        opt_state=optimizer,
        pending_qb_betas=tuple(beta for stage in arrays.pending_qb_betas for beta in _unstack_pending(stage)),
    )


def _pipeline_state(
    canonical: HeroPipelineCheckpointState, exemplar: GrugMoeAutomaticPipelineState
) -> GrugMoeAutomaticPipelineState:
    layer_counts = tuple(len(stage.blocks) for stage in exemplar.trainable_params)
    params = _split_stage(canonical.params, layer_counts)
    optimizer = tuple(
        jax.tree.map(
            lambda value, index=index: (
                _split_stage(value, layer_counts)[index] if isinstance(value, GrugMoePipelineStage) else value
            ),
            canonical.opt_state,
            is_leaf=lambda value: isinstance(value, GrugMoePipelineStage),
        )
        for index in range(len(layer_counts))
    )
    pending = tuple(
        _stack_pending(canonical.pending_qb_betas[stage.start_layer : stage.end_layer], target.sharding)
        for stage, target in zip(params, exemplar.pending_qb_betas, strict=True)
    )
    return GrugMoeAutomaticPipelineState(params, optimizer, pending)


def save_checkpoint(root: str, state: GrugMoeAutomaticPipelineState, *, step: int, contract: dict) -> str:
    """Publish a complete Hero pipeline checkpoint using Levanter's step layout."""
    path = prefix_join(root, f"step-{step}")
    return save_levanter_checkpoint(checkpoint_state(state), step, path, metadata={"training_config": contract})


def restore_checkpoint(
    root: str, state: GrugMoeAutomaticPipelineState, shardings: PyTree, *, contract: dict
) -> tuple[GrugMoeAutomaticPipelineState, int]:
    """Restore the latest complete Hero checkpoint into the compiled stage layout."""
    candidates = discover_checkpoint_candidates(root)
    if not candidates:
        return state, 0
    checkpoint = candidates[-1]
    saved_config = checkpoint.metadata.get("training_config")
    if saved_config is not None and saved_config != json.loads(json.dumps(contract)):
        raise ValueError("Hero checkpoint training configuration does not match")
    arrays = mpmd_checkpoint.checkpoint_arrays(state)
    canonical = checkpoint_state(arrays)
    loaded = mpmd_checkpoint.restore_checkpoint(
        canonical, checkpoint.path, jax.tree.map(lambda value: value.sharding, canonical)
    )
    restored = _pipeline_state(loaded, arrays)
    return mpmd_checkpoint.wrap_checkpoint_arrays(restored, shardings), checkpoint.step
