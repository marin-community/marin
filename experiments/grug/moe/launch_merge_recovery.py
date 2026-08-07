# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy artifact graph for one-pair Grug expert matching and recovery."""

import dataclasses
import os
from dataclasses import dataclass
from enum import StrEnum

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

from experiments.grug.moe.expert_merge import AssignmentMode
from experiments.grug.moe.expert_prefit import PrefitObjective
from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import grug_moe_training_datasets, grug_moe_validation_datasets
from experiments.grug.moe.merge_artifacts import (
    ExpertCalibrationArtifact,
    ExpertMatchingArtifact,
)
from experiments.grug.moe.merge_jobs import (
    CalibrationJobConfig,
    ConversionJobConfig,
    MatchingJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
    run_calibration,
    run_conversion,
    run_matching,
    run_prefit,
    run_recovery,
)
from experiments.grug.moe.merge_recovery import RecoveryStage
from experiments.grug.moe.optimizer import TiedExpertLrScale

_RESOURCES_KEY = "merge_resources"
_EXPERIMENT_REGION = "us-central1"
_MERGE_RESOURCES = ResourceConfig.with_tpu("v5p-8", regions=[_EXPERIMENT_REGION])
_BUDGET = 3.82e17
_HIDDEN_DIM = 512
_TARGET_STEPS = 2**14
_SEQUENCE_LENGTH = 4096


class MergeBranchName(StrEnum):
    IDENTITY = "identity"
    NATIVE = "native"
    SPECTRAL = "spectral"
    SPECTRAL_PREFIT = "spectral_prefit"
    NATIVE_AGGREGATE_PREFIT = "native_aggregate_prefit"


@dataclass(frozen=True)
class MergeRecoveryBranch:
    name: MergeBranchName
    assignment_mode: AssignmentMode
    prefit_applied: bool
    converted: ArtifactStep[LevanterCheckpoint]
    stage_a: ArtifactStep[LevanterCheckpoint]
    stage_b: ArtifactStep[LevanterCheckpoint]


@dataclass(frozen=True)
class MergeRecoveryPipeline:
    teacher: ArtifactStep[LevanterCheckpoint]
    calibration: ArtifactStep[ExpertCalibrationArtifact]
    matching: ArtifactStep[ExpertMatchingArtifact]
    prefit: ArtifactStep[LevanterCheckpoint]
    native_aggregate_prefit: ArtifactStep[LevanterCheckpoint]
    branches: tuple[MergeRecoveryBranch, ...]


def _checkpoint_dir(path: str) -> str:
    return prefix_join(path, "checkpoints")


def build_merge_recovery_pipeline(
    teacher: ArtifactStep[LevanterCheckpoint],
    *,
    version: str | None = None,
    resources: ResourceConfig = _MERGE_RESOURCES,
    teacher_commit: str | None = None,
) -> MergeRecoveryPipeline:
    base_model, base_optimizer, _, source_steps = build_from_heuristic(
        budget=_BUDGET,
        hidden_dim=_HIDDEN_DIM,
        target_steps=_TARGET_STEPS,
        seq_len=_SEQUENCE_LENGTH,
    )
    if base_model.num_layers != 6:
        raise ValueError(f"one-pair d512 merge requires six layers, got {base_model.num_layers}")
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

    calibration_name = "grug/expert_merge/d512/calibration-layers-2-3"
    calibration_version = resolve_version(calibration_name, version)

    def calibration_config(ctx: StepContext) -> CalibrationJobConfig:
        return CalibrationJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            data=mixture(ctx, train, validation=validation),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-calibration-d512-l2-l3",
        )

    calibration = ArtifactStep(
        name=user_namespaced_name(calibration_name, calibration_version),
        version=calibration_version,
        artifact_type=ExpertCalibrationArtifact,
        run=run_calibration,
        build_config=calibration_config,
        deps=(teacher, *data_deps),
        runtime_args={_RESOURCES_KEY: resources},
    )

    matching_name = "grug/expert_merge/d512/matching-layers-2-3"
    matching_version = resolve_version(matching_name, version)

    def matching_config(ctx: StepContext) -> MatchingJobConfig:
        return MatchingJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            calibration_path=ctx.artifact_path(calibration),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-matching-d512-l2-l3",
        )

    matching = ArtifactStep(
        name=user_namespaced_name(matching_name, matching_version),
        version=matching_version,
        artifact_type=ExpertMatchingArtifact,
        run=run_matching,
        build_config=matching_config,
        deps=(teacher, calibration),
        runtime_args={_RESOURCES_KEY: resources},
    )

    prefit_name = "grug/expert_merge/d512/spectral-prefit-layers-2-3"
    prefit_version = resolve_version(prefit_name, version)

    def prefit_config(ctx: StepContext) -> PrefitJobConfig:
        return PrefitJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            calibration_path=ctx.artifact_path(calibration),
            matching_path=ctx.artifact_path(matching),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-spectral-prefit-d512-l2-l3",
            assignment_mode=AssignmentMode.SPECTRAL,
            objective=PrefitObjective.PER_EXPERT,
        )

    prefit = ArtifactStep(
        name=user_namespaced_name(prefit_name, prefit_version),
        version=prefit_version,
        artifact_type=LevanterCheckpoint,
        run=run_prefit,
        build_config=prefit_config,
        deps=(teacher, calibration, matching),
        runtime_args={_RESOURCES_KEY: resources},
    )

    native_aggregate_prefit_name = "grug/expert_merge/d512/native-aggregate-prefit-layers-2-3"
    native_aggregate_prefit_version = resolve_version(native_aggregate_prefit_name, version)

    def native_aggregate_prefit_config(ctx: StepContext) -> PrefitJobConfig:
        return PrefitJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            calibration_path=ctx.artifact_path(calibration),
            matching_path=ctx.artifact_path(matching),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-native-aggregate-prefit-d512-l2-l3",
            assignment_mode=AssignmentMode.NATIVE,
            objective=PrefitObjective.AGGREGATE_ROUTED,
        )

    native_aggregate_prefit = ArtifactStep(
        name=user_namespaced_name(native_aggregate_prefit_name, native_aggregate_prefit_version),
        version=native_aggregate_prefit_version,
        artifact_type=LevanterCheckpoint,
        run=run_prefit,
        build_config=native_aggregate_prefit_config,
        deps=(teacher, calibration, matching),
        runtime_args={_RESOURCES_KEY: resources},
    )

    branch_specs = (
        (MergeBranchName.IDENTITY, AssignmentMode.IDENTITY, None),
        (MergeBranchName.NATIVE, AssignmentMode.NATIVE, None),
        (MergeBranchName.SPECTRAL, AssignmentMode.SPECTRAL, None),
        (MergeBranchName.SPECTRAL_PREFIT, AssignmentMode.SPECTRAL, prefit),
        (
            MergeBranchName.NATIVE_AGGREGATE_PREFIT,
            AssignmentMode.NATIVE,
            native_aggregate_prefit,
        ),
    )
    branches = []
    for branch_name, assignment_mode, branch_prefit in branch_specs:
        prefit_applied = branch_prefit is not None
        conversion_name = f"grug/expert_merge/d512/{branch_name.value}/converted"
        conversion_version = resolve_version(conversion_name, version)

        def conversion_config(
            ctx: StepContext,
            *,
            branch_name=branch_name,
            assignment_mode=assignment_mode,
            prefit_applied=prefit_applied,
            branch_prefit=branch_prefit,
        ) -> ConversionJobConfig:
            return ConversionJobConfig(
                source=SourceCheckpointConfig(
                    model=base_model,
                    optimizer=base_optimizer,
                    training_steps=source_steps,
                    checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                    source_commit=teacher_commit,
                ),
                calibration_path=ctx.artifact_path(calibration),
                matching_path=ctx.artifact_path(matching),
                prefit_path=_checkpoint_dir(ctx.artifact_path(branch_prefit)) if branch_prefit is not None else None,
                output_path=ctx.output_path,
                resources=ctx.runtime_arg(_RESOURCES_KEY),
                run_id=f"grug-xem-{branch_name.value}-convert-d512-l2-l3",
                assignment_mode=assignment_mode,
            )

        conversion_deps = (
            (teacher, calibration, matching, branch_prefit)
            if branch_prefit is not None
            else (teacher, calibration, matching)
        )
        converted = ArtifactStep(
            name=user_namespaced_name(conversion_name, conversion_version),
            version=conversion_version,
            artifact_type=LevanterCheckpoint,
            run=run_conversion,
            build_config=conversion_config,
            deps=conversion_deps,
            runtime_args={_RESOURCES_KEY: resources},
        )

        def recovery_handle(
            stage: RecoveryStage,
            init_from: ArtifactStep[LevanterCheckpoint],
            *,
            branch_name=branch_name,
            assignment_mode=assignment_mode,
            prefit_applied=prefit_applied,
        ):
            stage_label = "stage-a" if stage is RecoveryStage.LOCAL else "stage-b"
            recovery_name = f"grug/expert_merge/d512/{branch_name.value}/{stage_label}"
            recovery_version = resolve_version(recovery_name, version)
            training_tokens = 50_000_000 if stage is RecoveryStage.LOCAL else 200_000_000

            def recovery_config(
                ctx: StepContext,
                *,
                stage=stage,
                branch_name=branch_name,
                assignment_mode=assignment_mode,
                prefit_applied=prefit_applied,
                training_tokens=training_tokens,
            ) -> RecoveryJobConfig:
                return RecoveryJobConfig(
                    source=SourceCheckpointConfig(
                        model=base_model,
                        optimizer=base_optimizer,
                        training_steps=source_steps,
                        checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                        source_commit=teacher_commit,
                    ),
                    data=mixture(ctx, train, validation=validation),
                    matching_path=ctx.artifact_path(matching),
                    init_checkpoint_dir=_checkpoint_dir(ctx.artifact_path(init_from)),
                    output_path=ctx.output_path,
                    resources=ctx.runtime_arg(_RESOURCES_KEY),
                    run_id=f"grug-xem-{branch_name.value}-{stage_label}-d512-l2-l3",
                    stage=stage,
                    assignment_mode=assignment_mode,
                    prefit_applied=prefit_applied,
                    training_tokens=training_tokens,
                    logit_kl_weight=0.1 if stage is RecoveryStage.PRESERVATION else 0.0,
                )

            return ArtifactStep(
                name=user_namespaced_name(recovery_name, recovery_version),
                version=recovery_version,
                artifact_type=LevanterCheckpoint,
                run=run_recovery,
                build_config=recovery_config,
                deps=(teacher, matching, init_from, *data_deps),
                runtime_args={_RESOURCES_KEY: resources},
            )

        stage_a = recovery_handle(RecoveryStage.LOCAL, converted)
        stage_b = recovery_handle(RecoveryStage.PRESERVATION, stage_a)
        branches.append(
            MergeRecoveryBranch(
                name=branch_name,
                assignment_mode=assignment_mode,
                prefit_applied=prefit_applied,
                converted=converted,
                stage_a=stage_a,
                stage_b=stage_b,
            )
        )

    return MergeRecoveryPipeline(
        teacher=teacher,
        calibration=calibration,
        matching=matching,
        prefit=prefit,
        native_aggregate_prefit=native_aggregate_prefit,
        branches=tuple(branches),
    )


def pipeline_from_environment(*, version: str | None = None) -> MergeRecoveryPipeline:
    source = os.environ.get("GRUG_MERGE_TEACHER")
    if not source:
        raise ValueError("GRUG_MERGE_TEACHER must name the regional teacher artifact root")
    teacher_name = "grug/expert_merge/d512/teacher"
    teacher = ArtifactStep.adopt(
        teacher_name,
        resolve_version(teacher_name, version),
        source,
        kind=LevanterCheckpoint,
    )
    return build_merge_recovery_pipeline(
        teacher,
        version=version,
        teacher_commit=os.environ.get("GRUG_MERGE_TEACHER_COMMIT"),
    )


@click.command()
@click.option("--branch", type=click.Choice([branch.value for branch in MergeBranchName]), default="spectral_prefit")
@build_options
def main(branch: str) -> ArtifactStep[LevanterCheckpoint]:
    pipeline = pipeline_from_environment()
    selected = next(candidate for candidate in pipeline.branches if candidate.name.value == branch)
    return selected.stage_b


if __name__ == "__main__":
    main()
