# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import canonical Snowball HF weights into the vendored stacked Grug trainer model."""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
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
from levanter.models.snowball import SnowballConfig, snowball_from_state_dict
from levanter.utils.jax_utils import use_cpu_device
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from marin.utils import get_directory_friendly_name
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.june_tpu_67b_a2b.moe.model import Block, GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.sft.launcher import SFTSpec

_CHECKPOINT_STEP = 0


def _conversion_checkpoint_path(output_path: str) -> str:
    checkpoint_root = prefix_join(output_path, "checkpoints")
    return prefix_join(checkpoint_root, f"step-{_CHECKPOINT_STEP}")


def _stack_blocks(blocks: Sequence[Block], template: ArrayStacked[Block]) -> ArrayStacked[Block]:
    """Stack loaded per-layer leaves while retaining the trainer template's static metadata."""
    if len(blocks) != template.num_layers:
        raise ValueError(f"Expected {template.num_layers} blocks, got {len(blocks)}")

    source_leaves = [jax.tree.leaves(block) for block in blocks]
    template_leaves, template_treedef = jax.tree.flatten(template.stacked)
    if any(len(leaves) != len(template_leaves) for leaves in source_leaves):
        raise ValueError("HF-loaded blocks do not match the vendored stacked trainer pytree")

    stacked_leaves = []
    for leaf_index, template_leaf in enumerate(template_leaves):
        stacked = jnp.stack([leaves[leaf_index] for leaves in source_leaves])
        if stacked.shape != template_leaf.shape:
            raise ValueError(f"Stacked leaf {leaf_index} has shape {stacked.shape}; expected {template_leaf.shape}")
        stacked_leaves.append(jax.sharding.reshard(stacked, template_leaf.sharding))

    stacked_block = jax.tree.unflatten(template_treedef, stacked_leaves)
    return eqx.tree_at(lambda value: value.stacked, template, stacked_block)


def import_snowball_hf_weights(
    config: GrugModelConfig,
    state_dict: Mapping[str, Any],
    *,
    key: jax.Array,
) -> tuple[Transformer, jax.Array]:
    """Build the historical stacked Grug model and its QB state from canonical HF tensors.

    Snowball HF exports contain the *effective* router biases after the source checkpoint's pending
    QB update was applied. The historical trainer applies ``pending_qb_betas`` at the start of every
    step, so the inverse bias is reconstructed here. Centering is functionally exact because adding a
    per-layer scalar to every expert logit changes neither top-k selection nor combine weights.
    """
    if not config.use_array_stacked_blocks:
        raise ValueError("Snowball SFT import requires use_array_stacked_blocks=True")

    unstacked_config = dataclasses.replace(config, use_array_stacked_blocks=False)
    # Shape-only templates avoid allocating two additional random FP32 copies of the model. At 67B,
    # each such copy is roughly 268 GB and would dominate the conversion worker's memory budget.
    load_template = eqx.filter_eval_shape(Transformer.init, unstacked_config, key=key)
    loaded = snowball_from_state_dict(load_template, dict(state_dict))
    if loaded.blocks is None:
        raise ValueError("HF import did not produce unstacked blocks")

    target = eqx.filter_eval_shape(Transformer.init, config, key=key)
    if target.stacked_blocks is None:
        raise ValueError("Stacked trainer template did not produce stacked blocks")
    stacked_blocks = _stack_blocks(loaded.blocks, target.stacked_blocks)

    router_bias = stacked_blocks.stacked.mlp.router_bias
    centered_router_bias = router_bias - jnp.mean(router_bias, axis=-1, keepdims=True)
    stacked_blocks = eqx.tree_at(
        lambda value: value.stacked.mlp.router_bias,
        stacked_blocks,
        centered_router_bias,
    )
    pending_qb_betas = -centered_router_bias

    model = Transformer(
        token_embed=loaded.token_embed,
        embed_norm=loaded.embed_norm,
        embed_gated_norm=loaded.embed_gated_norm,
        output_proj=loaded.output_proj,
        blocks=None,
        stacked_blocks=stacked_blocks,
        final_norm=loaded.final_norm,
        final_gated_norm=loaded.final_gated_norm,
        config=config,
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
            model_config,
            state_dict,
            key=jax.random.key(0),
        )
        manager = GlobalAsyncCheckpointManager()
        save_checkpoint(
            {"params": model, "pending_qb_betas": pending_qb_betas},
            step=_CHECKPOINT_STEP,
            checkpoint_path=_conversion_checkpoint_path(config.output_path),
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
    remote(_run_snowball_hf_to_grug, resources=config.resources)(config)


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


@dataclass(frozen=True)
class ConvertedSnowballGrugModel:
    """Historical Grug SFT source backed by a pinned HF-to-native conversion artifact."""

    conversion: SnowballHfToGrugCheckpoint
    tokenizer_path: str
    expert_parallel: int
    init_from: ArtifactStep[LevanterCheckpoint] | None = None
    model_axis: int = 1
    replica_axis: int = 1
    per_device_parallelism: int = -1
    mp: str = "params=float32,compute=bfloat16,output=bfloat16"

    def tokenizer_cache_key(self) -> str:
        return self.tokenizer_path

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        del ctx
        return self.tokenizer_path

    @property
    def run(self) -> Callable[..., None]:
        return self._model("", "").run

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        if self.init_from is None or self.init_from is self.conversion.step:
            return (self.conversion.step,)
        return (self.conversion.step, self.init_from)

    def _model(self, tokenizer_path: str, init_from: str) -> GrugModel:
        return GrugModel(
            model=self.conversion.model,
            tokenizer_path=tokenizer_path,
            init_from=init_from,
            expert_parallel=self.expert_parallel,
            model_axis=self.model_axis,
            replica_axis=self.replica_axis,
            per_device_parallelism=self.per_device_parallelism,
            mp=self.mp,
        )

    def build_train_config(
        self,
        ctx: StepContext,
        spec: SFTSpec,
        data_config,
        resources: ResourceConfig,
        num_train_steps: int,
    ):
        init_artifact = self.conversion.step if self.init_from is None else self.init_from
        init_from = prefix_join(ctx.artifact_path(init_artifact), "checkpoints")
        return self._model(self.resolve_tokenizer(ctx), init_from).build_train_config(
            ctx,
            spec,
            data_config,
            resources,
            num_train_steps,
        )


__all__ = [
    "ConvertedSnowballGrugModel",
    "SnowballHfToGrugCheckpoint",
    "import_snowball_hf_weights",
    "snowball_hf_to_grug",
]
