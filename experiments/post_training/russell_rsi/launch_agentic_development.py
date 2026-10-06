# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify and compare two frozen producers on eight adapted SWE-bench DEV tasks."""

import uuid
from collections.abc import Callable
from typing import TypeVar

import click
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import sanitize_job_name
from marin.experiment.cli import build_options
from marin.external_dependencies import MARIN_SKYRL
from marin.training.run_environment import dependency_groups_for_resources
from rigging.timing import Duration

from experiments.post_training.russell_rsi.agentic_development import (
    EVALUATION_JOB_TIMEOUT,
    HARBOR_COMMIT,
    LITELLM_VERSION,
    MINI_VERSION,
    NATIVE_MODEL_RETRY_ENV,
    QUALIFICATION_JOB_TIMEOUT,
    CheckpointEvaluationConfig,
    DevelopmentPlan,
    PairedReportConfig,
    QualificationConfig,
    load_plan,
    paired_report,
    run_checkpoint_evaluation,
    run_qualification,
)
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile

BRANCH_PYTHONPATH = (
    ":".join(
        f"/app/lib/{name}/src"
        for name in (
            "marin",
            "rigging",
            "shellbox",
            "fray",
            "iris",
            "levanter",
            "haliax",
            "rolloutengine",
            "taskcompendium",
        )
    )
    + ":/app"
)

NATIVE_REQUIREMENTS = [
    f"harbor @ git+https://github.com/marin-community/harbor.git@{HARBOR_COMMIT}",
    f"mini-swe-agent=={MINI_VERSION}",
    f"litellm=={LITELLM_VERSION}",
]


T = TypeVar("T")


def submit_worker(
    function: Callable[[T], None], config: T, resources: ResourceConfig, requirements: list[str], timeout: int
) -> None:
    request = JobRequest(
        name=sanitize_job_name(f"{function.__name__}-{uuid.uuid4().hex[:8]}"),
        entrypoint=Entrypoint.from_callable(lambda: function(config)),
        resources=resources,
        environment=create_environment(
            extras=dependency_groups_for_resources(resources, None),
            pip_packages=requirements,
            env_vars={
                "UV_PRERELEASE": "allow",
                "PYTHONPATH": BRANCH_PYTHONPATH,
                **NATIVE_MODEL_RETRY_ENV,
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            },
        ),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        timeout=Duration.from_seconds(timeout),
    )
    current_client().submit(request).wait(raise_on_failure=True)


def agentic_development_workflow(plan: DevelopmentPlan, version: str) -> ArtifactStep:
    qualification = ArtifactStep(
        name="evals/russell-rsi-native-development-qualification",
        version=version,
        artifact_type=Artifact,
        deps=(),
        build_config=lambda ctx: QualificationConfig(plan, ctx.output_path),
        run=lambda config: submit_worker(
            run_qualification,
            config,
            ResourceConfig(cpu=16, ram="64GB", disk="250GB", target_cluster=plan.worker_cluster),
            NATIVE_REQUIREMENTS,
            QUALIFICATION_JOB_TIMEOUT,
        ),
    )
    evaluations = []
    for index in range(2):
        evaluations.append(
            ArtifactStep(
                name=f"evals/russell-rsi-native-development-producer-{index}",
                version=version,
                artifact_type=Artifact,
                deps=(qualification, *evaluations),
                build_config=lambda ctx, role=index: CheckpointEvaluationConfig(
                    plan, role, ctx.artifact_path(qualification), ctx.output_path
                ),
                run=lambda config: submit_worker(
                    run_checkpoint_evaluation,
                    config,
                    ResourceConfig.with_gpu(
                        "H100", 8, cpu=32, ram="512GB", disk="2TB", target_cluster=plan.worker_cluster
                    ),
                    [MARIN_SKYRL.requirement(), *NATIVE_REQUIREMENTS],
                    EVALUATION_JOB_TIMEOUT,
                ),
            )
        )
    return ArtifactStep(
        name="documents/russell-rsi-native-development-comparison",
        version=version,
        artifact_type=Artifact,
        deps=tuple(evaluations),
        build_config=lambda ctx: PairedReportConfig(
            plan, (ctx.artifact_path(evaluations[0]), ctx.artifact_path(evaluations[1])), ctx.output_path
        ),
        run=paired_report,
    )


@click.command(help=__doc__)
@click.option("--config-uri", required=True)
@click.option("--config-sha256", required=True)
@build_options
def main(config_uri: str, config_sha256: str) -> list[ArtifactStep]:
    config = PinnedFile(config_uri, config_sha256).read_json()
    if resolve_version("russell-rsi-native-development", None) != config["evaluation_version"]:
        raise click.UsageError("DEVELOPMENT config and artifact version differ")
    return [agentic_development_workflow(load_plan(config), config["evaluation_version"])]


if __name__ == "__main__":
    main()
