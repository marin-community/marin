# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical model and optimizer checkpoints shared by FSDP and pipeline training."""

import json
from dataclasses import dataclass

import jax
import optax
from jaxtyping import PyTree
from levanter import mpmd_checkpoint
from levanter.checkpoint import discover_checkpoint_candidates
from levanter.checkpoint import save_checkpoint as save_levanter_checkpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_pipeline.model import Transformer
from experiments.grug.moe_pipeline.pipeline import (
    GrugMoeAutomaticPipelineState,
    GrugMoePipelineStage,
    split_transformer,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GrugMoeCheckpointState:
    """Unsplit trainable model, optimizer, and one pending router update per layer.

    FSDP can save or load this tree with ordinary Levanter checkpoint APIs.
    Router biases are excluded from params and optimizer moments; the pending
    updates supply them at the beginning of the next training step.
    """

    params: Transformer
    opt_state: optax.OptState
    pending_qb_betas: tuple[jax.Array, ...]


def _merge_stages(stages: tuple[GrugMoePipelineStage, ...]) -> Transformer:
    return Transformer(
        token_embed=stages[0].token_embed,
        embed_norm=stages[0].embed_norm,
        embed_gated_norm=stages[0].embed_gated_norm,
        output_proj=stages[-1].output_proj,
        blocks=tuple(block for stage in stages for block in stage.blocks),
        final_norm=stages[-1].final_norm,
        final_gated_norm=stages[-1].final_gated_norm,
        config=stages[0].config,
    )


def checkpoint_state(state: GrugMoeAutomaticPipelineState) -> GrugMoeCheckpointState:
    """Assemble stage-local buffers under global model and optimizer paths."""
    arrays = mpmd_checkpoint.checkpoint_arrays(state)
    optimizer = jax.tree.map(
        lambda *values: _merge_stages(values) if isinstance(values[0], GrugMoePipelineStage) else values[0],
        *arrays.opt_state,
        is_leaf=lambda value: isinstance(value, GrugMoePipelineStage),
    )
    return GrugMoeCheckpointState(
        params=_merge_stages(arrays.trainable_params),
        opt_state=optimizer,
        pending_qb_betas=tuple(
            beta for stage in arrays.pending_qb_betas for beta in mpmd_checkpoint.unstack_checkpoint_layers(stage)
        ),
    )


def _pipeline_state(
    canonical: GrugMoeCheckpointState, exemplar: GrugMoeAutomaticPipelineState
) -> GrugMoeAutomaticPipelineState:
    layer_counts = tuple(len(stage.blocks) for stage in exemplar.trainable_params)
    stages = len(layer_counts)
    params = split_transformer(canonical.params, stages, layer_counts=layer_counts)
    optimizer = tuple(
        jax.tree.map(
            lambda value, index=index: (
                split_transformer(value, stages, layer_counts=layer_counts)[index]
                if isinstance(value, Transformer)
                else value
            ),
            canonical.opt_state,
            is_leaf=lambda value: isinstance(value, Transformer),
        )
        for index in range(stages)
    )
    pending = tuple(
        mpmd_checkpoint.stack_checkpoint_layers(
            canonical.pending_qb_betas[stage.start_layer : stage.end_layer], target.sharding
        )
        for stage, target in zip(params, exemplar.pending_qb_betas, strict=True)
    )
    return GrugMoeAutomaticPipelineState(params, optimizer, pending)


def save_checkpoint(root: str, state: GrugMoeAutomaticPipelineState, *, step: int, contract: dict) -> str:
    """Save canonical state with ordinary Levanter checkpoint publication."""
    path = prefix_join(root, f"step-{step}")
    return save_levanter_checkpoint(checkpoint_state(state), step, path, metadata={"training_config": contract})


def restore_checkpoint(
    root: str, state: GrugMoeAutomaticPipelineState, shardings: PyTree, *, contract: dict
) -> tuple[GrugMoeAutomaticPipelineState, int]:
    """Load an FSDP or PP checkpoint into the destination pipeline partition.

    Args:
        root: Directory searched for the latest completed checkpoint.
        state: Destination pipeline state used as a shape and structure template.
        shardings: Compiled step input shardings with the same tree as state.
        contract: Model, precision, and optimizer configuration to validate
            against training_config metadata when the checkpoint supplies it.
    """
    candidates = discover_checkpoint_candidates(root)
    if not candidates:
        return state, 0
    checkpoint = candidates[-1]
    saved_config = checkpoint.metadata.get("training_config")
    if saved_config is not None and saved_config != json.loads(json.dumps(contract)):
        raise ValueError("Checkpoint training configuration does not match")
    arrays = mpmd_checkpoint.checkpoint_arrays(state)
    canonical = checkpoint_state(arrays)
    loaded = mpmd_checkpoint.restore_checkpoint(
        canonical, checkpoint.path, jax.tree.map(lambda value: value.sharding, canonical)
    )
    restored = _pipeline_state(loaded, arrays)
    return mpmd_checkpoint.wrap_checkpoint_arrays(restored, shardings), checkpoint.step
