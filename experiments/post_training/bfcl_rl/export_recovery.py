# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a saved recovery policy without loading its optimizer or reference."""

from dataclasses import replace

import click
from fray.types import ResourceConfig
from levanter.models.snowball import SnowballConfig
from levanter.trainer import TrainerConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.export.levanter_checkpoint import ConvertCheckpointStepConfig, convert_checkpoint_to_hf
from marin.rl.cli import rl_build_options
from marin.training.training import LevanterCheckpoint
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS


@click.command(help=__doc__)
@click.option("--recovery-version", required=True)
@click.option("--checkpoint-step", type=click.IntRange(min=0), required=True)
@rl_build_options
def main(recovery_version: str, checkpoint_step: int) -> ArtifactStep:
    source = MODELS["student"]
    recovery_name = user_owned_name("models/bfcl-rl-recovery-dpo-full")
    checkpoint = ArtifactStep.adopt(
        f"{recovery_name}-export-input",
        recovery_version,
        f"{recovery_name}/{recovery_version}/checkpoints/step-{checkpoint_step}",
        kind=LevanterCheckpoint,
    )
    initial_student = ArtifactStep.adopt(
        user_owned_name("models/bfcl-rl-student"),
        source.version,
        source.uri,
        kind=LevanterCheckpoint,
        config={"model": source.model, "revision": source.revision},
    )

    def build_config(ctx: StepContext) -> ConvertCheckpointStepConfig:
        return ConvertCheckpointStepConfig(
            checkpoint_path=ctx.artifact_path(checkpoint),
            checkpoint_subpath="model/policy",
            trainer=TrainerConfig(),
            model=SnowballConfig(
                max_seq_len=262144,
                qk_mult=1.75,
                initializer_std=0.009882117688026186,
                attention_implementation="gpu_fa4_cute",
                moe_implementation="ring",
                reference_checkpoint=ctx.artifact_path(initial_student),
                tokenizer=f"{source.model}@{source.revision}",
            ),
            tokenizer=f"{source.model}@{source.revision}",
            output_path=prefix_join(ctx.output_path, f"hf/step-{checkpoint_step}"),
            export_dtype="bfloat16",
            resources=ResourceConfig.with_cpu(cpu=48, ram="1024Gi", disk="1024Gi"),
            use_cpu=True,
        )

    name = user_owned_name("models/bfcl-rl-recovery-policy-export")
    step = ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=LevanterCheckpoint,
        run=convert_checkpoint_to_hf,
        build_config=build_config,
        deps=(checkpoint, initial_student),
    )
    return replace(step, runtime_args={"execution": COLLECTION_EXECUTION})


if __name__ == "__main__":
    main()
