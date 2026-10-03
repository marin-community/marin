# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect paired BFCL rollouts and build a verifier-selected recovery preference cache."""

import click
from fray.types import ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.bfcl_rl.collect import SMOKE_TASKS, collection_step, complement_data_step
from experiments.post_training.bfcl_rl.recovery_data import (
    RecoveryCacheConfig,
    RecoveryPreferenceCache,
    build_recovery_cache,
    recovery_cache_value,
)


def dispatch_recovery_cache(config: RecoveryCacheConfig) -> RecoveryPreferenceCache:
    """Build the remote cache and return its persisted selection metadata."""
    remote(
        build_recovery_cache,
        resources=ResourceConfig.with_cpu(cpu=4, ram="32Gi", disk="64Gi"),
    )(config)
    return recovery_cache_value(config.output_path)


def recovery_cache_step(
    teacher: ArtifactStep[SkyRLRun], student: ArtifactStep[SkyRLRun], *, selection_name: str, max_length: int
) -> ArtifactStep[RecoveryPreferenceCache]:
    """Bind both collection artifacts and the audited source release into CPU cache construction."""
    data = complement_data_step()
    name = user_owned_name(f"data/bfcl-rl-recovery-preferences-{selection_name}")

    def build_config(ctx: StepContext) -> RecoveryCacheConfig:
        if ctx.is_fingerprint:
            teacher_uri = str(StoragePath(ctx.artifact_path(teacher)) / "terminal.json")
            student_uri = str(StoragePath(ctx.artifact_path(student)) / "terminal.json")
            data_root = ctx.artifact_path(data)
        else:
            teacher_uri = ctx.resolved(teacher).terminal_manifest_uri
            student_uri = ctx.resolved(student).terminal_manifest_uri
            data_root = ctx.resolved(data).path
        return RecoveryCacheConfig(teacher_uri, student_uri, data_root, max_length, ctx.output_path)

    return ArtifactStep(
        name=name,
        version=resolve_version(name, None),
        artifact_type=RecoveryPreferenceCache,
        run=dispatch_recovery_cache,
        build_config=build_config,
        deps=(teacher, student, data),
    )


@click.command(help=__doc__)
@click.option("--task", type=click.Choice(SMOKE_TASKS), default=None)
@click.option("--python-image", required=True)
@click.option("--java-image", required=True)
@click.option("--javascript-image", required=True)
@rl_build_options
def main(task: str | None, python_image: str, java_image: str, javascript_image: str) -> ArtifactStep:
    images = (python_image, java_image, javascript_image)
    teacher = collection_step("teacher", task, images)
    student = collection_step("student", task, images)
    return recovery_cache_step(teacher, student, selection_name=task or "full", max_length=40960)


if __name__ == "__main__":
    main()
