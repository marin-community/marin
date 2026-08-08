# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the preregistered d512 correspondence-free joint refactor screen."""

import dataclasses
from dataclasses import dataclass
from enum import StrEnum

import click
from fray.cluster import ResourceConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.data import mixture
from marin.training.training import LevanterCheckpoint
from rigging.filesystem import prefix_join

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import grug_moe_training_datasets, grug_moe_validation_datasets
from experiments.grug.moe.merge_artifacts import ExpertCalibrationArtifact
from experiments.grug.moe.merge_jobs import (
    JointRefactorJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
    run_joint_refactor,
    run_recovery,
)
from experiments.grug.moe.merge_recovery import (
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)
from experiments.grug.moe.optimizer import TiedExpertLrScale

_VERSION = "2026.08.08"
_RESOURCES_KEY = "merge_resources"
_REGION = "us-central1"
_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=[_REGION])
_TEACHER_NAME = "grug/expert_merge/d512/teacher"
_TEACHER_VERSION = "2026.08.06"
_TEACHER_ROOT = "gs://marin-us-central1/grug/tied_experts/d512/full/baseline/2026.08.06"
_TEACHER_COMMIT = "884b213ff4"
_CALIBRATION_NAME = "grug/expert_merge/d512/calibration-layers-2-3"
_CALIBRATION_VERSION = "2026.08.06"
_CALIBRATION_ROOT = "gs://marin-us-central1/grug/expert_merge/d512/calibration-layers-2-3/2026.08.06"
_OFFLINE_NAME = "grug/expert_merge/d512/joint-refactor-layers-2-3"
_SCREEN_NAME = "grug/expert_merge/d512/joint_refactor/screen-25m"
_BUDGET = 3.82e17
_HIDDEN_DIM = 512
_TARGET_STEPS = 2**14
_SEQUENCE_LENGTH = 4_096
_SCREEN_TOKENS = 25_034_752
_SCREEN_STEP = 191


class JointRefactorStage(StrEnum):
    OFFLINE = "offline"
    SCREEN = "screen"


@dataclass(frozen=True)
class JointRefactorPipeline:
    offline: ArtifactStep[LevanterCheckpoint]
    screen: ArtifactStep[LevanterCheckpoint]


def _checkpoint_dir(path: str) -> str:
    return prefix_join(path, "checkpoints")


def build_joint_refactor_pipeline(
    *,
    version: str = _VERSION,
    resources: ResourceConfig = _RESOURCES,
) -> JointRefactorPipeline:
    """Build a central1-only graph with no expert-matching dependency."""
    teacher = ArtifactStep.adopt(
        _TEACHER_NAME,
        _TEACHER_VERSION,
        _TEACHER_ROOT,
        kind=LevanterCheckpoint,
    )
    calibration = ArtifactStep.adopt(
        _CALIBRATION_NAME,
        _CALIBRATION_VERSION,
        _CALIBRATION_ROOT,
        kind=ExpertCalibrationArtifact,
    )
    base_model, base_optimizer, _, source_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != 6 or base_model.num_experts != 256:
        raise ValueError(
            f"joint refactor requires the six-layer, 256-expert d512 model, got "
            f"layers={base_model.num_layers}, experts={base_model.num_experts}"
        )
    base_model = dataclasses.replace(base_model, expert_bank_for_layer=tuple(range(base_model.num_layers)))
    base_optimizer = dataclasses.replace(
        base_optimizer,
        expert_bank_group_sizes=base_model.expert_bank_group_sizes,
        tied_expert_lr_scale=TiedExpertLrScale.UNSCALED,
        schedule_horizon_steps=source_steps,
    )
    source = SourceCheckpointConfig(
        model=base_model,
        optimizer=base_optimizer,
        training_steps=source_steps,
        checkpoint_dir=_checkpoint_dir(_TEACHER_ROOT),
        source_commit=_TEACHER_COMMIT,
    )

    def offline_config(ctx: StepContext) -> JointRefactorJobConfig:
        return JointRefactorJobConfig(
            source=dataclasses.replace(source, checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher))),
            calibration_path=ctx.artifact_path(calibration),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-010-joint-refactor-offline-d512-l2-l3",
        )

    offline = ArtifactStep(
        name=_OFFLINE_NAME,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_joint_refactor,
        build_config=offline_config,
        deps=(teacher, calibration),
        runtime_args={_RESOURCES_KEY: resources},
    )

    train = grug_moe_training_datasets()
    validation = grug_moe_validation_datasets()
    data_deps = (*train, *validation)

    def screen_config(ctx: StepContext) -> RecoveryJobConfig:
        return RecoveryJobConfig(
            source=dataclasses.replace(source, checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher))),
            data=mixture(ctx, train, validation=validation),
            matching_path=None,
            init_checkpoint_dir=_checkpoint_dir(ctx.artifact_path(offline)),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-010-joint-refactor-screen-d512-l2-l3",
            stage=RecoveryStage.PRESERVATION,
            trainable_scope=RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS,
            initialization=RecoveryInitialization.JOINT_REFACTORIZATION,
            assignment_mode=AssignmentMode.IDENTITY,
            prefit_applied=False,
            training_tokens=_SCREEN_TOKENS,
            cross_entropy_weight=1.0,
            moe_loss_weight=1.0,
            logit_kl_weight=0.1,
            checkpoint_every=_SCREEN_STEP,
            checkpoint_token_milestones=(_SCREEN_TOKENS,),
        )

    screen = ArtifactStep(
        name=_SCREEN_NAME,
        version=version,
        artifact_type=LevanterCheckpoint,
        run=run_recovery,
        build_config=screen_config,
        deps=(teacher, offline, *data_deps),
        runtime_args={_RESOURCES_KEY: resources},
    )
    return JointRefactorPipeline(offline=offline, screen=screen)


@click.command()
@click.option(
    "--stage",
    type=click.Choice([stage.value for stage in JointRefactorStage]),
    default=JointRefactorStage.SCREEN.value,
)
@build_options
def main(stage: str) -> ArtifactStep[LevanterCheckpoint]:
    pipeline = build_joint_refactor_pipeline()
    return pipeline.offline if stage == JointRefactorStage.OFFLINE.value else pipeline.screen


if __name__ == "__main__":
    main()
