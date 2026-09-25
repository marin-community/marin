# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Topology-independent checkpoints for the Hero pipeline variant."""

import json
from dataclasses import dataclass

import jax
import optax
from jaxtyping import PyTree
from levanter import mpmd_checkpoint
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.checkpoint import save_checkpoint as save_levanter_checkpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_pipeline.pipeline import (
    GrugMoeAutomaticPipelineState,
    GrugMoePipelineStage,
    split_pipeline_stage,
)


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
        pending_qb_betas=tuple(
            beta for stage in arrays.pending_qb_betas for beta in mpmd_checkpoint.unstack_checkpoint_layers(stage)
        ),
    )


def _pipeline_state(
    canonical: HeroPipelineCheckpointState, exemplar: GrugMoeAutomaticPipelineState
) -> GrugMoeAutomaticPipelineState:
    layer_counts = tuple(len(stage.blocks) for stage in exemplar.trainable_params)
    params = split_pipeline_stage(canonical.params, len(layer_counts), layer_counts=layer_counts)
    optimizer = tuple(
        jax.tree.map(
            lambda value, index=index: (
                split_pipeline_stage(value, len(layer_counts), layer_counts=layer_counts)[index]
                if isinstance(value, GrugMoePipelineStage)
                else value
            ),
            canonical.opt_state,
            is_leaf=lambda value: isinstance(value, GrugMoePipelineStage),
        )
        for index in range(len(layer_counts))
    )
    pending = tuple(
        mpmd_checkpoint.stack_checkpoint_layers(
            canonical.pending_qb_betas[stage.start_layer : stage.end_layer], target.sharding
        )
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
