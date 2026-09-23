# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import canonical Snowball HF weights into the current stacked Grug trainer model."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from fray.types import ResourceConfig
from haliax.partitioning import set_mesh
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballTransformer
from levanter.utils.jax_utils import use_cpu_device
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint
from marin.utils import get_directory_friendly_name
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.model import GrugModelConfig, Transformer

_CHECKPOINT_STEP = 0
_CONVERSION_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


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
    model_config.validate_snowball_config(source_config)

    with use_cpu_device(), set_mesh(compact_grug_mesh(expert_axis_size=1)):
        state_dict = converter.load_state_dict(ref, dtype=jnp.bfloat16)
        key = jax.random.key(0)
        source_template = eqx.filter_eval_shape(SnowballTransformer.init, source_config, key=key)
        source = source_template.from_state_dict(state_dict)
        target = eqx.filter_eval_shape(Transformer.init, model_config, key=key)
        model, pending_qb_betas = target.with_snowball_weights(source)
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
