# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect fresh native BFCL counterparts and curate their complete masked pairs."""

from dataclasses import replace
from enum import StrEnum

import click
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.build_context import BuildContext, VersionCodex, build_context, resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import ArtifactDataSource, skyrl_step
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import COLLECTION_EXECUTION, MODELS, ModelSource, complement_data_step
from experiments.post_training.bfcl_rl.final_dpo import DPO_EXECUTION, INPUT_NAME, RunScale, final_dpo_spec
from experiments.post_training.bfcl_rl.final_smoke_data import FreshSmokeData, FreshSmokeDataConfig, run_fresh_smoke_data
from experiments.post_training.bfcl_rl.launch import recovered_model
from experiments.post_training.bfcl_rl.native_collection_snapshot import (
    CollectionSnapshotConfig,
    seal_completed_collection,
)
from experiments.post_training.bfcl_rl.offline_collect import TEACHER_MODEL, TEACHER_REVISION, offline_collection_step
from experiments.post_training.bfcl_rl.offline_curate import NativeCollectionInput, NativeCollectionScope
from experiments.post_training.bfcl_rl.offline_preferences import (
    NATIVE_PREFERENCE_RESOURCES,
    NativePreferenceConfig,
    run_native_preference_cache,
)
from experiments.post_training.bfcl_rl.offline_student_collect import native_student_collection_step
from experiments.post_training.bfcl_rl.recovery_data import RecoveryPreferenceCache

TEACHER_SEED = 511993168
TASK = "bfcl-irrelevance-0"
HARNESS = ("opencode",)
RECOVERY_VERSION = "2026.10.04.21"
EXPORT_VERSION = "2026.10.04.26"
CHECKPOINT_STEP = 57


class SmokeStage(StrEnum):
    COLLECT = "collect"
    PIPELINE = "pipeline"


def collection_snapshot_step(collection: ArtifactStep, data: ArtifactStep) -> ArtifactStep:
    name = collection.name.replace("rollouts/", "rollout-snapshots/")

    def build_config(ctx: StepContext) -> CollectionSnapshotConfig:
        return CollectionSnapshotConfig(
            str(StoragePath(ctx.artifact_path(collection)) / "terminal.json"),
            ctx.artifact_path(data),
            ctx.output_path,
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=Artifact,
        run=remote(seal_completed_collection, resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi")),
        build_config=build_config,
        deps=(collection, data),
        runtime_args={"execution": COLLECTION_EXECUTION},
    )


def collection_smoke_step(
    teacher_source: str, collection_version: str, student_seed: int
) -> ArtifactStep[RecoveryPreferenceCache]:
    with build_context(BuildContext(VersionCodex(collection_version))):
        teacher = offline_collection_step(teacher_source, TEACHER_SEED, TASK, 32, HARNESS)
        student = native_student_collection_step(
            RECOVERY_VERSION, EXPORT_VERSION, CHECKPOINT_STEP, student_seed, TASK, 32, HARNESS
        )
    policy = replace(recovered_model(RECOVERY_VERSION, EXPORT_VERSION), relative_path=f"hf/step-{CHECKPOINT_STEP}")
    data = complement_data_step()
    teacher_snapshot = collection_snapshot_step(teacher, data)
    student_snapshot = collection_snapshot_step(student, data)
    name = user_owned_name("data/bfcl-rl-final-native-collection-smoke")

    def build_config(ctx: StepContext) -> NativePreferenceConfig:
        original = MODELS["student"]
        return NativePreferenceConfig(
            teachers=(
                NativeCollectionInput(
                    str(StoragePath(ctx.artifact_path(teacher_snapshot)) / "snapshot.json"),
                    ModelSource(TEACHER_MODEL, TEACHER_REVISION, teacher_source, "pinned"),
                    TEACHER_SEED,
                    NativeCollectionScope.SEALED_BATCHES,
                ),
            ),
            student=NativeCollectionInput(
                str(StoragePath(ctx.artifact_path(student_snapshot)) / "snapshot.json"),
                ModelSource(original.model, original.revision, policy.resolve(ctx).uri, EXPORT_VERSION),
                student_seed,
                NativeCollectionScope.SEALED_BATCHES,
            ),
            data_root=ctx.artifact_path(data),
            student_tokenizer=f"{original.model}@{original.revision}",
            max_length=40960,
            output_path=ctx.output_path,
            max_workers=4,
            student_model_alias="bfcl-rl-recovered-policy",
        )

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=RecoveryPreferenceCache,
        run=remote(run_native_preference_cache, resources=NATIVE_PREFERENCE_RESOURCES),
        build_config=build_config,
        deps=(teacher_snapshot, student_snapshot, policy.step, data),
        runtime_args={"execution": COLLECTION_EXECUTION},
    )


def pipeline_smoke_step(
    teacher_source: str, input_version: str, collection_version: str, student_seed: int
) -> ArtifactStep:
    fresh = collection_smoke_step(teacher_source, collection_version, student_seed)
    name = user_owned_name(INPUT_NAME)
    frozen = ArtifactStep.adopt(name + "-input", input_version, f"{name}/{input_version}", kind=Artifact)
    complement = complement_data_step()
    smoke_name = user_owned_name("data/bfcl-rl-final-fresh-smoke-pairs")

    def build_config(ctx: StepContext) -> FreshSmokeDataConfig:
        return FreshSmokeDataConfig(
            ctx.artifact_path(fresh), ctx.artifact_path(frozen), ctx.artifact_path(complement), ctx.output_path
        )

    pairs = ArtifactStep(
        name=smoke_name,
        version=resolve_version(smoke_name, None),
        artifact_type=FreshSmokeData,
        run=remote(run_fresh_smoke_data, resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi")),
        build_config=build_config,
        deps=(fresh, frozen, complement),
        runtime_args={"execution": DPO_EXECUTION},
    )
    spec = final_dpo_spec(input_version, RunScale.SMOKE)
    spec = replace(
        spec,
        name=user_owned_name("models/bfcl-rl-final-native-dpo-pipeline-smoke"),
        train_data=(ArtifactDataSource(pairs, relative_path="smoke.parquet"),),
    )
    return skyrl_step(spec, DPO_EXECUTION, export_hf=True)


@click.command(help=__doc__)
@click.option("--teacher-source", required=True)
@click.option("--input-version", required=True)
@click.option(
    "--student-seed", type=int, required=True, help="Explicit seed for fresh student collection and provenance."
)
@click.option(
    "--collection-version", required=True, help="Immutable teacher/student collection version to build or reuse."
)
@click.option("--stage", type=click.Choice([value.value for value in SmokeStage]), required=True)
@rl_build_options
def main(
    teacher_source: str, input_version: str, collection_version: str, student_seed: int, stage: str
) -> ArtifactStep:
    if SmokeStage(stage) is SmokeStage.COLLECT:
        return collection_smoke_step(teacher_source, collection_version, student_seed)
    return pipeline_smoke_step(teacher_source, input_version, collection_version, student_seed)


if __name__ == "__main__":
    main()
