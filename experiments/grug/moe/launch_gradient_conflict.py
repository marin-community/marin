# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Central1 launch graph for the preregistered direct shared-bank gradient diagnostic."""

import dataclasses
import os

import click
from fray.cluster import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import grug_moe_training_datasets, grug_moe_validation_datasets
from experiments.grug.moe.merge_artifacts import GradientConflictArtifact
from experiments.grug.moe.merge_jobs import (
    GradientConflictArtifactReference,
    GradientConflictCheckpointConfig,
    GradientConflictJobConfig,
    SourceCheckpointConfig,
    run_gradient_conflict,
)
from experiments.grug.moe.optimizer import TiedExpertLrScale

_RESOURCES_KEY = "merge_resources"
_EXPERIMENT_REGION = "us-central1"
_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=[_EXPERIMENT_REGION])
_BUDGET = 3.82e17
_HIDDEN_DIM = 512
_TARGET_STEPS = 2**14
_SEQUENCE_LENGTH = 4096
_ARTIFACT_VERSION = "2026.08.06"
_TEACHER_ARTIFACT_NAME = "grug/tied_experts/d512/full/baseline"
_STAGE_A_ARTIFACT_NAME = "grug/expert_merge/d512/native_local_ce_kl/stage-a"
_CONTROL_ARTIFACT_NAME = "grug/expert_merge/d512/native_local_ce_kl_adapter_control/stage-b"
_TEACHER_ARTIFACT_FINGERPRINT = "38d1fe9b"
_STAGE_A_ARTIFACT_FINGERPRINT = "62564720"
_CONTROL_ARTIFACT_FINGERPRINT = "59969d51"
_TEACHER_COMMIT = "884b213ff4"
_STAGE_A_ARTIFACT_ROOT = "gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl/stage-a/2026.08.06"
_CONTROL_ARTIFACT_ROOT = (
    "gs://marin-us-central1/grug/expert_merge/d512/native_local_ce_kl_adapter_control/stage-b/2026.08.06"
)


def _checkpoint_path(artifact_root: str, step: int) -> str:
    return prefix_join(prefix_join(artifact_root, "checkpoints"), f"step-{step}")


def build_gradient_conflict_diagnostic(
    teacher: ArtifactStep[LevanterCheckpoint],
    stage_a: ArtifactStep[LevanterCheckpoint],
    control: ArtifactStep[LevanterCheckpoint],
    *,
    version: str | None = None,
    resources: ResourceConfig = _RESOURCES,
    teacher_commit: str | None = None,
) -> ArtifactStep[GradientConflictArtifact]:
    """Build one sequential worker over the three preregistered checkpoints."""
    expected_artifacts = (
        (teacher, _TEACHER_ARTIFACT_NAME),
        (stage_a, _STAGE_A_ARTIFACT_NAME),
        (control, _CONTROL_ARTIFACT_NAME),
    )
    for artifact, expected_name in expected_artifacts:
        if artifact.name != expected_name or artifact.version != _ARTIFACT_VERSION:
            raise ValueError(
                f"gradient-conflict input must be {expected_name}@{_ARTIFACT_VERSION}, "
                f"got {artifact.name}@{artifact.version}"
            )
    if teacher_commit != _TEACHER_COMMIT:
        raise ValueError(f"gradient-conflict teacher commit must be {_TEACHER_COMMIT}, got {teacher_commit!r}")
    base_model, base_optimizer, _, source_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != 6:
        raise ValueError(f"d512 gradient conflict requires six layers, got {base_model.num_layers}")
    base_model = dataclasses.replace(base_model, expert_bank_for_layer=tuple(range(base_model.num_layers)))
    base_optimizer = dataclasses.replace(
        base_optimizer,
        expert_bank_group_sizes=base_model.expert_bank_group_sizes,
        tied_expert_lr_scale=TiedExpertLrScale.UNSCALED,
        schedule_horizon_steps=source_steps,
    )
    train = grug_moe_training_datasets()
    validation = grug_moe_validation_datasets()
    data_deps: tuple[ArtifactStep[TokenizedCache], ...] = (*train, *validation)
    diagnostic_name = "grug/expert_merge/d512/shared_bank_gradient_conflict/diagnostic"
    diagnostic_version = resolve_version(diagnostic_name, version)

    def diagnostic_config(ctx: StepContext) -> GradientConflictJobConfig:
        stage_a_root = ctx.artifact_path(stage_a)
        control_root = ctx.artifact_path(control)
        teacher_root = ctx.artifact_path(teacher)
        stage_a_artifact = GradientConflictArtifactReference(
            name=_STAGE_A_ARTIFACT_NAME,
            version=_ARTIFACT_VERSION,
            root=stage_a_root,
            fingerprint=_STAGE_A_ARTIFACT_FINGERPRINT,
        )
        control_artifact = GradientConflictArtifactReference(
            name=_CONTROL_ARTIFACT_NAME,
            version=_ARTIFACT_VERSION,
            root=control_root,
            fingerprint=_CONTROL_ARTIFACT_FINGERPRINT,
        )
        return GradientConflictJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=prefix_join(ctx.artifact_path(teacher), "checkpoints"),
                source_commit=teacher_commit,
            ),
            teacher_artifact=GradientConflictArtifactReference(
                name=_TEACHER_ARTIFACT_NAME,
                version=_ARTIFACT_VERSION,
                root=teacher_root,
                fingerprint=_TEACHER_ARTIFACT_FINGERPRINT,
            ),
            data=mixture(ctx, train, validation=validation),
            checkpoints=(
                GradientConflictCheckpointConfig(
                    label="selected_stage_a",
                    artifact=stage_a_artifact,
                    checkpoint_path=_checkpoint_path(stage_a_root, 382),
                    expected_step=382,
                    continuation_tokens=0,
                ),
                GradientConflictCheckpointConfig(
                    label="shared_control_midpoint",
                    artifact=control_artifact,
                    checkpoint_path=_checkpoint_path(control_root, 191),
                    expected_step=191,
                    continuation_tokens=25_034_752,
                ),
                GradientConflictCheckpointConfig(
                    label="shared_control_endpoint",
                    artifact=control_artifact,
                    checkpoint_path=_checkpoint_path(control_root, 382),
                    expected_step=382,
                    continuation_tokens=50_069_504,
                ),
            ),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-shared-bank-gradient-conflict-d512-l2-l3",
            affected_layers=(2, 3),
            batch_size=32,
            num_batches=16,
            loader_start_step=382,
            bootstrap_samples=10_000,
            bootstrap_seed=8_032,
            seed=0,
        )

    return ArtifactStep(
        name=user_namespaced_name(diagnostic_name, diagnostic_version),
        version=diagnostic_version,
        artifact_type=GradientConflictArtifact,
        run=run_gradient_conflict,
        build_config=diagnostic_config,
        deps=(teacher, stage_a, control, *data_deps),
        runtime_args={_RESOURCES_KEY: resources},
    )


def diagnostic_from_environment(*, version: str | None = None) -> ArtifactStep[GradientConflictArtifact]:
    teacher_root = os.environ.get("GRUG_MERGE_TEACHER")
    if not teacher_root:
        raise ValueError("GRUG_MERGE_TEACHER must name the regional teacher artifact root")
    teacher = ArtifactStep.adopt(
        _TEACHER_ARTIFACT_NAME,
        _ARTIFACT_VERSION,
        teacher_root,
        kind=LevanterCheckpoint,
    )
    stage_a = ArtifactStep.adopt(
        _STAGE_A_ARTIFACT_NAME,
        _ARTIFACT_VERSION,
        _STAGE_A_ARTIFACT_ROOT,
        kind=LevanterCheckpoint,
    )
    control = ArtifactStep.adopt(
        _CONTROL_ARTIFACT_NAME,
        _ARTIFACT_VERSION,
        _CONTROL_ARTIFACT_ROOT,
        kind=LevanterCheckpoint,
    )
    return build_gradient_conflict_diagnostic(
        teacher,
        stage_a,
        control,
        version=version,
        teacher_commit=os.environ.get("GRUG_MERGE_TEACHER_COMMIT"),
    )


@click.command()
@build_options
def main() -> ArtifactStep[GradientConflictArtifact]:
    return diagnostic_from_environment()


if __name__ == "__main__":
    main()
