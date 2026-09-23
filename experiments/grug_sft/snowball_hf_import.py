# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import canonical Snowball HF weights into the current stacked Grug trainer model."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from fray.types import ResourceConfig
from haliax.nn import ArrayStacked
from haliax.partitioning import set_mesh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballBlock, SnowballConfig, SnowballTransformer
from levanter.utils.jax_utils import use_cpu_device
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from marin.utils import get_directory_friendly_name
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.model import Block, GrugModelConfig, Transformer

_CHECKPOINT_STEP = 0
_CONVERSION_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


def _stack_blocks(blocks: Sequence[SnowballBlock], template: ArrayStacked[Block]) -> ArrayStacked[Block]:
    """Stack loaded per-layer leaves while retaining the trainer template's static metadata."""
    if len(blocks) != template.num_layers:
        raise ValueError(f"Expected {template.num_layers} blocks, got {len(blocks)}")

    source_leaves = [jax.tree.leaves(block) for block in blocks]
    template_leaves, template_treedef = jax.tree.flatten(template.stacked)
    if any(len(leaves) != len(template_leaves) for leaves in source_leaves):
        raise ValueError("Snowball blocks do not match the stacked trainer pytree")

    stacked_leaves = []
    for leaf_index, template_leaf in enumerate(template_leaves):
        stacked = jnp.stack([leaves[leaf_index] for leaves in source_leaves])
        if stacked.shape != template_leaf.shape:
            raise ValueError(f"Stacked leaf {leaf_index} has shape {stacked.shape}; expected {template_leaf.shape}")
        stacked_leaves.append(jax.sharding.reshard(stacked, template_leaf.sharding))

    stacked_block = jax.tree.unflatten(template_treedef, stacked_leaves)
    return eqx.tree_at(lambda value: value.stacked, template, stacked_block)


def import_snowball_hf_weights(
    snowball_config: SnowballConfig,
    trainer_config: GrugModelConfig,
    state_dict: Mapping[str, Any],
    *,
    key: jax.Array,
) -> tuple[Transformer, jax.Array]:
    """Build current Grug weights and QB state from canonical Snowball HF tensors.

    Snowball HF exports contain the *effective* router biases after the source checkpoint's pending
    QB update was applied. The trainer applies ``pending_qb_betas`` at the start of every
    step, so the inverse bias is reconstructed here. Centering is functionally exact because adding a
    per-layer scalar to every expert logit changes neither top-k selection nor combine weights.
    """
    load_template = eqx.filter_eval_shape(SnowballTransformer.init, snowball_config, key=key)
    loaded = load_template.from_state_dict(dict(state_dict))

    target = eqx.filter_eval_shape(Transformer.init, trainer_config, key=key)
    stacked_blocks = _stack_blocks(loaded.blocks, target.stacked_blocks)

    router_bias = stacked_blocks.stacked.mlp.router_bias
    centered_router_bias = router_bias - jnp.mean(router_bias, axis=-1, keepdims=True)
    stacked_blocks = eqx.tree_at(
        lambda value: value.stacked.mlp.router_bias,
        stacked_blocks,
        centered_router_bias,
    )
    pending_qb_betas = -centered_router_bias

    model = eqx.tree_at(
        lambda value: (
            value.token_embed,
            value.embed_norm.weight,
            value.embed_gated_norm.w_down,
            value.embed_gated_norm.w_up,
            value.output_proj,
            value.stacked_blocks,
            value.final_norm.weight,
            value.final_gated_norm.w_down,
            value.final_gated_norm.w_up,
        ),
        target,
        (
            jax.sharding.reshard(loaded.token_embed, target.token_embed.sharding),
            jax.sharding.reshard(loaded.embed_norm.weight, target.embed_norm.weight.sharding),
            jax.sharding.reshard(loaded.embed_gated_norm.w_down, target.embed_gated_norm.w_down.sharding),
            jax.sharding.reshard(loaded.embed_gated_norm.w_up, target.embed_gated_norm.w_up.sharding),
            jax.sharding.reshard(loaded.output_proj, target.output_proj.sharding),
            stacked_blocks,
            jax.sharding.reshard(loaded.final_norm.weight, target.final_norm.weight.sharding),
            jax.sharding.reshard(loaded.final_gated_norm.w_down, target.final_gated_norm.w_down.sharding),
            jax.sharding.reshard(loaded.final_gated_norm.w_up, target.final_gated_norm.w_up.sharding),
        ),
    )
    return model, pending_qb_betas


@dataclass(frozen=True)
class SnowballHfToGrugConfig:
    hf_id: str
    hf_revision: str
    model_config: dict[str, Any]
    output_path: str
    resources: ResourceConfig


@dataclass(frozen=True)
class SnowballHfToGrugCheckpoint:
    step: ArtifactStep[LevanterCheckpoint]
    model: GrugModelConfig


def _run_snowball_hf_to_grug(config: SnowballHfToGrugConfig) -> None:
    model_config = draccus.decode(GrugModelConfig, config.model_config)
    ref = RepoRef(config.hf_id, config.hf_revision)
    converter = SnowballConfig().hf_checkpoint_converter().replaced(reference_checkpoint=ref)
    source_config = converter.config_from_hf_checkpoint(ref)
    exact_fields = (
        "vocab_size",
        "hidden_dim",
        "intermediate_dim",
        "shared_expert_intermediate_dim",
        "num_experts",
        "num_experts_per_token",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "sliding_window",
        "layer_norm_eps",
        "initializer_std",
        "qk_mult",
    )
    mismatches = {
        name: (getattr(source_config, name), getattr(model_config, name))
        for name in exact_fields
        if getattr(source_config, name) != getattr(model_config, name)
    }
    if mismatches:
        raise ValueError(f"Pinned HF config does not match the requested Grug architecture: {mismatches}")
    if model_config.max_seq_len > source_config.max_seq_len:
        raise ValueError(
            f"Training max_seq_len={model_config.max_seq_len} exceeds HF max_seq_len={source_config.max_seq_len}"
        )

    with use_cpu_device(), set_mesh(compact_grug_mesh(expert_axis_size=1)):
        state_dict = converter.load_state_dict(ref, dtype=jnp.bfloat16)
        model, pending_qb_betas = import_snowball_hf_weights(
            source_config,
            model_config,
            state_dict,
            key=jax.random.key(0),
        )
        manager = GlobalAsyncCheckpointManager()
        save_checkpoint(
            {"params": model, "pending_qb_betas": pending_qb_betas},
            step=_CHECKPOINT_STEP,
            checkpoint_path=prefix_join(
                prefix_join(config.output_path, "checkpoints"),
                f"step-{_CHECKPOINT_STEP}",
            ),
            manager=manager,
            is_temporary=False,
        )
        manager.wait_until_finished()

    tokenizer = load_tokenizer(config.hf_id, revision=config.hf_revision)
    with tempfile.TemporaryDirectory(prefix="snowball-grug-tokenizer-") as tokenizer_dir:
        tokenizer.save_pretrained(tokenizer_dir)
        for name in os.listdir(tokenizer_dir):
            if not name.startswith("."):
                StoragePath(prefix_join(config.output_path, name)).upload_from(os.path.join(tokenizer_dir, name))


def _convert_job(config: SnowballHfToGrugConfig) -> None:
    remote(_run_snowball_hf_to_grug, resources=config.resources, env_vars=_CONVERSION_ENV)(config)


def snowball_hf_to_grug(
    hf_id: str,
    *,
    hf_revision: str,
    model: GrugModelConfig,
    version: str,
    resources: ResourceConfig,
) -> SnowballHfToGrugCheckpoint:
    """Materialize one immutable HF export as a native stacked Grug weights checkpoint."""
    name = f"checkpoints/hf-to-stacked-grug/{get_directory_friendly_name(hf_id)}"

    def build_config(ctx: StepContext) -> SnowballHfToGrugConfig:
        return SnowballHfToGrugConfig(
            hf_id=hf_id,
            hf_revision=hf_revision,
            model_config=draccus.encode(model),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg("convert_resources"),
        )

    step: ArtifactStep[LevanterCheckpoint] = ArtifactStep(
        name=name,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=_convert_job,
        build_config=build_config,
        runtime_args={"convert_resources": resources},
    )
    return SnowballHfToGrugCheckpoint(step=step, model=model)
