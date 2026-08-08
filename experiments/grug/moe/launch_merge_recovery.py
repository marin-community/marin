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
    CapacityOracleSplitJobConfig,
    ConversionJobConfig,
    LayerAdapterAugmentJobConfig,
    MatchingJobConfig,
    PrefitJobConfig,
    RecoveryJobConfig,
    SourceCheckpointConfig,
    run_calibration,
    run_capacity_oracle_split,
    run_conversion,
    run_layer_adapter_augment,
    run_matching,
    run_prefit,
    run_recovery,
)
from experiments.grug.moe.merge_recovery import (
    RecoveryCheckpointSelection,
    RecoveryInitialization,
    RecoveryStage,
    RecoveryTrainableScope,
)
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
    NATIVE_LOCAL_SELECTED = "native_local_selected"
    NATIVE_LOCAL_CE = "native_local_ce"
    NATIVE_LOCAL_KL = "native_local_kl"
    NATIVE_LOCAL_CE_KL = "native_local_ce_kl"
    NATIVE_LOCAL_CE_KL_BANK_ONLY = "native_local_ce_kl_bank_only"
    NATIVE_LOCAL_CE_KL_MLP_NORMS = "native_local_ce_kl_mlp_norms"
    NATIVE_LOCAL_CE_KL_CAPACITY_ORACLE = "native_local_ce_kl_capacity_oracle"
    NATIVE_LOCAL_CE_KL_ADAPTER_CONTROL = "native_local_ce_kl_adapter_control"
    NATIVE_LOCAL_CE_KL_ADAPTER_R8 = "native_local_ce_kl_adapter_r8"
    NATIVE_JOINT = "native_joint"


@dataclass(frozen=True)
class MergeRecoveryBranch:
    name: MergeBranchName
    assignment_mode: AssignmentMode
    prefit_applied: bool
    converted: ArtifactStep[LevanterCheckpoint]
    stage_a: ArtifactStep[LevanterCheckpoint]
    stage_b: ArtifactStep[LevanterCheckpoint]


@dataclass(frozen=True)
class DirectPreservationBranch:
    """A conversion followed directly by preservation recovery."""

    name: MergeBranchName
    assignment_mode: AssignmentMode
    converted: ArtifactStep[LevanterCheckpoint]
    stage_b: ArtifactStep[LevanterCheckpoint]


@dataclass(frozen=True)
class RecoveryUnlockBranch:
    """A one-factor recovery diagnostic from the selected local checkpoint."""

    name: MergeBranchName
    trainable_scope: RecoveryTrainableScope
    recovery: ArtifactStep[LevanterCheckpoint]


@dataclass(frozen=True)
class MergeRecoveryPipeline:
    teacher: ArtifactStep[LevanterCheckpoint]
    calibration: ArtifactStep[ExpertCalibrationArtifact]
    matching: ArtifactStep[ExpertMatchingArtifact]
    prefit: ArtifactStep[LevanterCheckpoint]
    native_aggregate_prefit: ArtifactStep[LevanterCheckpoint]
    branches: tuple[MergeRecoveryBranch, ...]
    native_joint: DirectPreservationBranch
    capacity_oracle_split: ArtifactStep[LevanterCheckpoint]
    unlock_diagnostics: tuple[RecoveryUnlockBranch, ...]
    layer_adapter_augment: ArtifactStep[LevanterCheckpoint]
    layer_adapter_diagnostics: tuple[RecoveryUnlockBranch, ...]


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

    def recovery_handle(
        stage: RecoveryStage,
        init_from: ArtifactStep[LevanterCheckpoint],
        *,
        branch_name: MergeBranchName,
        assignment_mode: AssignmentMode,
        prefit_applied: bool,
        local_cross_entropy_weight: float = 0.0,
        local_logit_kl_weight: float = 0.0,
        select_best_local_checkpoint: bool = False,
        initial_checkpoint_selection: RecoveryCheckpointSelection = RecoveryCheckpointSelection.LATEST,
    ) -> ArtifactStep[LevanterCheckpoint]:
        stage_label = "stage-a" if stage is RecoveryStage.LOCAL else "stage-b"
        recovery_name = f"grug/expert_merge/d512/{branch_name.value}/{stage_label}"
        recovery_version = resolve_version(recovery_name, version)
        training_tokens = 50_000_000 if stage is RecoveryStage.LOCAL else 200_000_000
        if stage is RecoveryStage.LOCAL:
            checkpoint_token_milestones = (
                (12_500_000, 25_000_000, 37_500_000, 50_000_000)
                if select_best_local_checkpoint
                else (25_000_000, 50_000_000)
            )
            cross_entropy_weight = local_cross_entropy_weight
            logit_kl_weight = local_logit_kl_weight
        else:
            checkpoint_token_milestones = (
                25_000_000,
                50_000_000,
                62_500_000,
                75_000_000,
                87_500_000,
                100_000_000,
                200_000_000,
            )
            cross_entropy_weight = 1.0
            logit_kl_weight = 0.1

        def recovery_config(ctx: StepContext) -> RecoveryJobConfig:
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
                trainable_scope=(
                    RecoveryTrainableScope.SHARED_BANK
                    if stage is RecoveryStage.LOCAL
                    else RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS
                ),
                initialization=(
                    RecoveryInitialization.CONVERTED_STEP_ZERO
                    if stage is RecoveryStage.LOCAL
                    else RecoveryInitialization.LOCAL_RECOVERY
                ),
                assignment_mode=assignment_mode,
                prefit_applied=prefit_applied,
                training_tokens=training_tokens,
                cross_entropy_weight=cross_entropy_weight,
                logit_kl_weight=logit_kl_weight,
                checkpoint_token_milestones=checkpoint_token_milestones,
                select_best_validation_checkpoint=(stage is RecoveryStage.LOCAL and select_best_local_checkpoint),
                initial_checkpoint_selection=initial_checkpoint_selection,
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

        stage_a = recovery_handle(
            RecoveryStage.LOCAL,
            converted,
            branch_name=branch_name,
            assignment_mode=assignment_mode,
            prefit_applied=prefit_applied,
        )
        stage_b = recovery_handle(
            RecoveryStage.PRESERVATION,
            stage_a,
            branch_name=branch_name,
            assignment_mode=assignment_mode,
            prefit_applied=prefit_applied,
        )
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

    native_converted = next(branch.converted for branch in branches if branch.name is MergeBranchName.NATIVE)
    for branch_name, cross_entropy_weight, logit_kl_weight in (
        (MergeBranchName.NATIVE_LOCAL_SELECTED, 0.0, 0.0),
        (MergeBranchName.NATIVE_LOCAL_CE, 0.05, 0.0),
        (MergeBranchName.NATIVE_LOCAL_KL, 0.0, 0.1),
        (MergeBranchName.NATIVE_LOCAL_CE_KL, 0.05, 0.1),
    ):
        stage_a = recovery_handle(
            RecoveryStage.LOCAL,
            native_converted,
            branch_name=branch_name,
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
            local_cross_entropy_weight=cross_entropy_weight,
            local_logit_kl_weight=logit_kl_weight,
            select_best_local_checkpoint=True,
        )
        stage_b = recovery_handle(
            RecoveryStage.PRESERVATION,
            stage_a,
            branch_name=branch_name,
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
            initial_checkpoint_selection=RecoveryCheckpointSelection.BEST_VALIDATION,
        )
        branches.append(
            MergeRecoveryBranch(
                name=branch_name,
                assignment_mode=AssignmentMode.NATIVE,
                prefit_applied=False,
                converted=native_converted,
                stage_a=stage_a,
                stage_b=stage_b,
            )
        )

    native_joint_name = "grug/expert_merge/d512/native_joint/converted"
    native_joint_version = resolve_version(native_joint_name, version)

    def native_joint_conversion_config(ctx: StepContext) -> ConversionJobConfig:
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
            prefit_path=None,
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-native-joint-convert-d512-l2-l3",
            assignment_mode=AssignmentMode.NATIVE,
        )

    native_joint_converted = ArtifactStep(
        name=user_namespaced_name(native_joint_name, native_joint_version),
        version=native_joint_version,
        artifact_type=LevanterCheckpoint,
        run=run_conversion,
        build_config=native_joint_conversion_config,
        deps=(teacher, calibration, matching),
        runtime_args={_RESOURCES_KEY: resources},
    )

    native_joint_stage_b_name = "grug/expert_merge/d512/native_joint/stage-b"
    native_joint_stage_b_version = resolve_version(native_joint_stage_b_name, version)

    def native_joint_stage_b_config(ctx: StepContext) -> RecoveryJobConfig:
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
            init_checkpoint_dir=_checkpoint_dir(ctx.artifact_path(native_joint_converted)),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-native-joint-stage-b-d512-l2-l3",
            stage=RecoveryStage.PRESERVATION,
            trainable_scope=RecoveryTrainableScope.SHARED_BANK_AND_ROUTERS,
            initialization=RecoveryInitialization.CONVERTED_STEP_ZERO,
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
            training_tokens=200_000_000,
            cross_entropy_weight=1.0,
            moe_loss_weight=1.0,
            logit_kl_weight=0.1,
            checkpoint_token_milestones=(25_000_000, 50_000_000, 100_000_000, 200_000_000),
        )

    native_joint_stage_b = ArtifactStep(
        name=user_namespaced_name(native_joint_stage_b_name, native_joint_stage_b_version),
        version=native_joint_stage_b_version,
        artifact_type=LevanterCheckpoint,
        run=run_recovery,
        build_config=native_joint_stage_b_config,
        deps=(teacher, matching, native_joint_converted, *data_deps),
        runtime_args={_RESOURCES_KEY: resources},
    )
    native_joint = DirectPreservationBranch(
        name=MergeBranchName.NATIVE_JOINT,
        assignment_mode=AssignmentMode.NATIVE,
        converted=native_joint_converted,
        stage_b=native_joint_stage_b,
    )

    selected_stage_a = next(branch.stage_a for branch in branches if branch.name is MergeBranchName.NATIVE_LOCAL_CE_KL)

    layer_adapter_augment_name = "grug/expert_merge/d512/native_local_ce_kl_adapter_r8/augment"
    layer_adapter_augment_version = resolve_version(layer_adapter_augment_name, version)

    def layer_adapter_augment_config(ctx: StepContext) -> LayerAdapterAugmentJobConfig:
        return LayerAdapterAugmentJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            init_checkpoint_dir=_checkpoint_dir(ctx.artifact_path(selected_stage_a)),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-native-local-ce-kl-adapter-r8-augment-d512-l2-l3",
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
            adapter_rank=8,
        )

    layer_adapter_augment = ArtifactStep(
        name=user_namespaced_name(layer_adapter_augment_name, layer_adapter_augment_version),
        version=layer_adapter_augment_version,
        artifact_type=LevanterCheckpoint,
        run=run_layer_adapter_augment,
        build_config=layer_adapter_augment_config,
        deps=(teacher, selected_stage_a),
        runtime_args={_RESOURCES_KEY: resources},
    )

    layer_adapter_diagnostics = []
    adapter_training_tokens = 50_069_504
    adapter_milestones = (12_582_912, 25_034_752, 37_617_664, adapter_training_tokens)
    for branch_name, trainable_scope, init_from, initialization, checkpoint_selection in (
        (
            MergeBranchName.NATIVE_LOCAL_CE_KL_ADAPTER_CONTROL,
            RecoveryTrainableScope.SHARED_BANK,
            selected_stage_a,
            RecoveryInitialization.LOCAL_RECOVERY,
            RecoveryCheckpointSelection.BEST_VALIDATION,
        ),
        (
            MergeBranchName.NATIVE_LOCAL_CE_KL_ADAPTER_R8,
            RecoveryTrainableScope.SHARED_BANK_AND_LAYER_ADAPTERS,
            layer_adapter_augment,
            RecoveryInitialization.LAYER_ADAPTER_AUGMENTED,
            RecoveryCheckpointSelection.LATEST,
        ),
    ):
        diagnostic_name = f"grug/expert_merge/d512/{branch_name.value}/stage-b"
        diagnostic_version = resolve_version(diagnostic_name, version)

        def adapter_diagnostic_config(
            ctx: StepContext,
            *,
            branch_name=branch_name,
            trainable_scope=trainable_scope,
            init_from=init_from,
            initialization=initialization,
            checkpoint_selection=checkpoint_selection,
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
                run_id=f"grug-xem-{branch_name.value}-stage-b-d512-l2-l3",
                stage=RecoveryStage.PRESERVATION,
                trainable_scope=trainable_scope,
                initialization=initialization,
                assignment_mode=AssignmentMode.NATIVE,
                prefit_applied=False,
                training_tokens=adapter_training_tokens,
                cross_entropy_weight=1.0,
                moe_loss_weight=1.0,
                logit_kl_weight=0.1,
                checkpoint_token_milestones=adapter_milestones,
                initial_checkpoint_selection=checkpoint_selection,
                layer_adapter_rank=8 if initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED else None,
                layer_adapter_source_checkpoint_dir=(
                    _checkpoint_dir(ctx.artifact_path(selected_stage_a))
                    if initialization is RecoveryInitialization.LAYER_ADAPTER_AUGMENTED
                    else None
                ),
            )

        adapter_source_deps = () if init_from is selected_stage_a else (selected_stage_a,)
        recovery = ArtifactStep(
            name=user_namespaced_name(diagnostic_name, diagnostic_version),
            version=diagnostic_version,
            artifact_type=LevanterCheckpoint,
            run=run_recovery,
            build_config=adapter_diagnostic_config,
            deps=(teacher, matching, init_from, *adapter_source_deps, *data_deps),
            runtime_args={_RESOURCES_KEY: resources},
        )
        layer_adapter_diagnostics.append(
            RecoveryUnlockBranch(
                name=branch_name,
                trainable_scope=trainable_scope,
                recovery=recovery,
            )
        )

    capacity_oracle_split_name = "grug/expert_merge/d512/native_local_ce_kl_capacity_oracle/split"
    capacity_oracle_split_version = resolve_version(capacity_oracle_split_name, version)

    def capacity_oracle_split_config(ctx: StepContext) -> CapacityOracleSplitJobConfig:
        return CapacityOracleSplitJobConfig(
            source=SourceCheckpointConfig(
                model=base_model,
                optimizer=base_optimizer,
                training_steps=source_steps,
                checkpoint_dir=_checkpoint_dir(ctx.artifact_path(teacher)),
                source_commit=teacher_commit,
            ),
            init_checkpoint_dir=_checkpoint_dir(ctx.artifact_path(selected_stage_a)),
            output_path=ctx.output_path,
            resources=ctx.runtime_arg(_RESOURCES_KEY),
            run_id="grug-xem-native-local-ce-kl-capacity-oracle-split-d512-l2-l3",
            assignment_mode=AssignmentMode.NATIVE,
            prefit_applied=False,
        )

    capacity_oracle_split = ArtifactStep(
        name=user_namespaced_name(capacity_oracle_split_name, capacity_oracle_split_version),
        version=capacity_oracle_split_version,
        artifact_type=LevanterCheckpoint,
        run=run_capacity_oracle_split,
        build_config=capacity_oracle_split_config,
        deps=(teacher, selected_stage_a),
        runtime_args={_RESOURCES_KEY: resources},
    )
    unlock_diagnostics = []
    for branch_name, trainable_scope, init_from, initialization, checkpoint_selection in (
        (
            MergeBranchName.NATIVE_LOCAL_CE_KL_BANK_ONLY,
            RecoveryTrainableScope.SHARED_BANK,
            selected_stage_a,
            RecoveryInitialization.LOCAL_RECOVERY,
            RecoveryCheckpointSelection.BEST_VALIDATION,
        ),
        (
            MergeBranchName.NATIVE_LOCAL_CE_KL_MLP_NORMS,
            RecoveryTrainableScope.SHARED_BANK_ROUTERS_AND_MLP_NORMS,
            selected_stage_a,
            RecoveryInitialization.LOCAL_RECOVERY,
            RecoveryCheckpointSelection.BEST_VALIDATION,
        ),
        (
            MergeBranchName.NATIVE_LOCAL_CE_KL_CAPACITY_ORACLE,
            RecoveryTrainableScope.AFFECTED_EXPERT_BANKS,
            capacity_oracle_split,
            RecoveryInitialization.CAPACITY_ORACLE_SPLIT,
            RecoveryCheckpointSelection.LATEST,
        ),
    ):
        diagnostic_name = f"grug/expert_merge/d512/{branch_name.value}/stage-b"
        diagnostic_version = resolve_version(diagnostic_name, version)

        def diagnostic_config(
            ctx: StepContext,
            *,
            branch_name=branch_name,
            trainable_scope=trainable_scope,
            init_from=init_from,
            initialization=initialization,
            checkpoint_selection=checkpoint_selection,
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
                run_id=f"grug-xem-{branch_name.value}-stage-b-d512-l2-l3",
                stage=RecoveryStage.PRESERVATION,
                trainable_scope=trainable_scope,
                initialization=initialization,
                assignment_mode=AssignmentMode.NATIVE,
                prefit_applied=False,
                training_tokens=50_000_000,
                cross_entropy_weight=1.0,
                moe_loss_weight=1.0,
                logit_kl_weight=0.1,
                checkpoint_token_milestones=(12_500_000, 25_000_000, 37_500_000, 50_000_000),
                initial_checkpoint_selection=checkpoint_selection,
            )

        recovery = ArtifactStep(
            name=user_namespaced_name(diagnostic_name, diagnostic_version),
            version=diagnostic_version,
            artifact_type=LevanterCheckpoint,
            run=run_recovery,
            build_config=diagnostic_config,
            deps=(teacher, matching, init_from, *data_deps),
            runtime_args={_RESOURCES_KEY: resources},
        )
        unlock_diagnostics.append(
            RecoveryUnlockBranch(
                name=branch_name,
                trainable_scope=trainable_scope,
                recovery=recovery,
            )
        )

    return MergeRecoveryPipeline(
        teacher=teacher,
        calibration=calibration,
        matching=matching,
        prefit=prefit,
        native_aggregate_prefit=native_aggregate_prefit,
        branches=tuple(branches),
        native_joint=native_joint,
        capacity_oracle_split=capacity_oracle_split,
        unlock_diagnostics=tuple(unlock_diagnostics),
        layer_adapter_augment=layer_adapter_augment,
        layer_adapter_diagnostics=tuple(layer_adapter_diagnostics),
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
@click.option("--stage", type=click.Choice([stage.value for stage in RecoveryStage]), default="preservation")
@build_options
def main(branch: str, stage: str) -> ArtifactStep[LevanterCheckpoint]:
    pipeline = pipeline_from_environment()
    if branch == MergeBranchName.NATIVE_JOINT.value:
        if stage != RecoveryStage.PRESERVATION.value:
            raise click.BadParameter("native_joint has no local-recovery stage", param_hint="--stage")
        return pipeline.native_joint.stage_b
    diagnostic = next((candidate for candidate in pipeline.unlock_diagnostics if candidate.name.value == branch), None)
    if diagnostic is None:
        diagnostic = next(
            (candidate for candidate in pipeline.layer_adapter_diagnostics if candidate.name.value == branch),
            None,
        )
    if diagnostic is not None:
        if stage != RecoveryStage.PRESERVATION.value:
            raise click.BadParameter("unlock diagnostics use the preservation objective", param_hint="--stage")
        return diagnostic.recovery
    selected = next(candidate for candidate in pipeline.branches if candidate.name.value == branch)
    return selected.stage_a if stage == RecoveryStage.LOCAL.value else selected.stage_b


if __name__ == "__main__":
    main()
