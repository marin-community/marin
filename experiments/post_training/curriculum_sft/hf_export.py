# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a stacked Grug SFT checkpoint for the evaluation vLLM backend."""

import dataclasses
import tempfile
from dataclasses import dataclass

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from fray.types import ResourceConfig
from haliax.partitioning import set_mesh
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe.model import GrugModelConfig as ExportModelConfig
from experiments.grug.moe.model import Transformer as ExportTransformer
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer
from experiments.post_training.curriculum_sft.grug_pipeline import GRUG_CHECKPOINTS_DIR

_EXPORT_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


@dataclass(frozen=True)
class GrugHfExportConfig:
    checkpoint_path: str
    tokenizer: str
    model_config: dict
    output_path: str
    resources: ResourceConfig


def _export_model(params: Transformer, config: ExportModelConfig) -> ExportTransformer:
    assert params.stacked_blocks is not None
    return ExportTransformer(
        token_embed=params.token_embed,
        embed_norm=params.embed_norm,
        embed_gated_norm=params.embed_gated_norm,
        output_proj=params.output_proj,
        blocks=tuple(params.stacked_blocks.unstacked()),
        final_norm=params.final_norm,
        final_gated_norm=params.final_gated_norm,
        config=config,
    )


def export_grug_checkpoint(config: GrugHfExportConfig) -> Artifact:
    """Write BF16 HF shards with the effective QB router bias applied."""
    model_config = draccus.decode(GrugModelConfig, config.model_config)
    export_fields = {field.name for field in dataclasses.fields(ExportModelConfig)}
    export_config = draccus.decode(
        ExportModelConfig,
        {name: value for name, value in config.model_config.items() if name in export_fields},
    )
    mesh = compact_grug_mesh(expert_axis_size=1)
    with set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, model_config, key=jax.random.PRNGKey(0))
        state = load_checkpoint(
            {
                "params": template,
                "pending_qb_betas": jax.ShapeDtypeStruct(
                    (model_config.num_layers, model_config.num_experts), jnp.float32
                ),
            },
            latest_checkpoint_path(config.checkpoint_path),
            mesh=mesh,
        )
        params = state["params"]
        pending_qb_betas = state["pending_qb_betas"]
        assert params.stacked_blocks is not None
        router_bias = -pending_qb_betas
        router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
        params = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, params, router_bias)
        params = jax.tree.map(
            lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
            params,
        )
        jax.block_until_ready(params)
        converter = (
            export_config.hf_checkpoint_converter()
            .replaced(tokenizer=load_tokenizer(config.tokenizer))
            .with_config_overrides({"dtype": "bfloat16"})
        )
        with tempfile.TemporaryDirectory(prefix="curriculum-grug-hf-") as export_dir:
            converter.save_pretrained(_export_model(params, export_config), export_dir, dtype=jnp.bfloat16)
            StoragePath(config.output_path).upload_from(export_dir + "/", recursive=True)
    return Artifact(path=config.output_path)


def _run_export(config: GrugHfExportConfig) -> Artifact:
    return remote(export_grug_checkpoint, resources=config.resources, env_vars=_EXPORT_ENV)(config)


def grug_hf_export(
    checkpoint: ArtifactStep[LevanterCheckpoint],
    *,
    model: GrugModelConfig,
    tokenizer: str,
    version: str,
    resources: ResourceConfig,
) -> ArtifactStep[Artifact]:
    """Export one native Grug SFT checkpoint to a vLLM-readable HF directory."""

    def build_config(ctx: StepContext) -> GrugHfExportConfig:
        return GrugHfExportConfig(
            checkpoint_path=prefix_join(ctx.artifact_path(checkpoint), GRUG_CHECKPOINTS_DIR),
            tokenizer=tokenizer,
            model_config=draccus.encode(model),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg("export_resources"),
        )

    return ArtifactStep(
        name=user_owned_name("models/curriculum-sft/grug-hf"),
        version=version,
        artifact_type=Artifact,
        run=_run_export,
        build_config=build_config,
        deps=(checkpoint,),
        runtime_args={"export_resources": resources},
    )
