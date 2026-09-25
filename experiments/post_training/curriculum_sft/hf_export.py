# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a stacked Grug SFT checkpoint for the evaluation vLLM backend."""

import dataclasses
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
from fray.types import ResourceConfig
from levanter.checkpoint import latest_checkpoint_path
from levanter.tokenizers import load_tokenizer
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe.model import GrugModelConfig as ExportModelConfig
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.post_training.curriculum_sft.grug_pipeline import GRUG_CHECKPOINTS_DIR
from experiments.post_training.curriculum_sft.hf_export_streaming import export_checkpoint_streaming

_EXPORT_ENV = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}


@dataclass(frozen=True)
class GrugHfExportConfig:
    checkpoint_path: str
    tokenizer: str
    model_config: dict[str, Any]
    output_path: str
    resources: ResourceConfig


def export_grug_checkpoint(config: GrugHfExportConfig) -> Artifact:
    """Write BF16 HF shards with the effective QB router bias applied."""
    model_config = draccus.decode(GrugModelConfig, config.model_config)
    export_fields = {field.name for field in dataclasses.fields(ExportModelConfig)}
    export_config = draccus.decode(
        ExportModelConfig,
        {name: value for name, value in config.model_config.items() if name in export_fields},
    )
    if not model_config.use_array_stacked_blocks:
        raise ValueError("Grug HF export requires stacked blocks")
    with tempfile.TemporaryDirectory(prefix="curriculum-grug-hf-") as export_dir:
        export_checkpoint_streaming(
            latest_checkpoint_path(config.checkpoint_path),
            config.output_path,
            export_config,
            load_tokenizer(config.tokenizer),
            Path(export_dir),
        )
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
