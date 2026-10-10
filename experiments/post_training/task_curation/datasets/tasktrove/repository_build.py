# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deferred repository images and private workspace graders."""

from taskcompendium.convert.answers import unsupported
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    VerifierArtifact,
)
from taskcompendium.pipeline.models import ImportRejection, NormalizationChange, NormalizedTask
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import Spec, render_spec

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    DOCKERFILE,
    TEST_SH_REWARD,
    TaskFiles,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import (
    VERIFYIT_CONTEXT,
    verifyit_build_context,
)

WORKSPACE = "/testbed"
PUBLIC_CONTEXT = "taskcompendium-public"
PUBLIC_SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"
REPOSITORY_SETUP = "taskcompendium-repository-setup.sh"
VERIFIER_SPEC = "taskcompendium-verifier.toml"
VERIFYIT_SCRIPT = f"#!/bin/bash\nset -euo pipefail\nexec verifyit /tests/{VERIFIER_SPEC}\n".encode()


def repository_build_task(
    task: TaskSpec,
    files: TaskFiles,
    *,
    spec: Spec,
    dockerfile: str,
    verifier: tuple[TaskResource, ...],
    tags: tuple[str, ...],
    changes: tuple[NormalizationChange, ...] = (),
) -> NormalizedTask | ImportRejection:
    """Keep the legacy actor recipe and declare a fresh workspace grading environment."""
    original_dockerfile = files.text(DOCKERFILE)
    if any(
        resource.path.removeprefix("environment/").startswith((VERIFYIT_CONTEXT, PUBLIC_CONTEXT, REPOSITORY_SETUP))
        for resource in task.resources.oracle
        if resource.path.startswith("environment/")
    ):
        return unsupported("build_context_collision", "Source context occupies the bundled verifier path")
    actor_build = verifyit_build_context(dockerfile, task.resources.oracle, package=VERIFYIT_PACKAGE)
    dockerfile = resource_bytes(
        next(resource for resource in actor_build.files if resource.path == "Dockerfile")
    ).decode()
    # Canonical ScriptGrader uses a fresh machine. Its dependencies must be prepared without
    # changing the actor recipe that the legacy shared-environment exporter consumes.
    instruction = files.text("instruction.md")
    setup, end, _ = instruction.removeprefix(PUBLIC_SETUP_PREFIX).partition("\n```")
    grader_files = list(actor_build.files)
    if instruction.startswith(PUBLIC_SETUP_PREFIX) and end:
        grader_dockerfile = dockerfile
        if task.resources.worker:
            grader_dockerfile += f"COPY {PUBLIC_CONTEXT}/ /\n"
            grader_files.extend(
                resource.model_copy(update={"path": PUBLIC_CONTEXT + "/" + resource.path})
                for resource in task.resources.worker
            )
        grader_dockerfile += (
            f"COPY {REPOSITORY_SETUP} /opt/{REPOSITORY_SETUP}\n"
            f"RUN bash -e /opt/{REPOSITORY_SETUP} && rm -rf {WORKSPACE} && mkdir -p {WORKSPACE}\n"
        )
        grader_files = [resource for resource in grader_files if resource.path != "Dockerfile"]
        grader_files.extend(
            (
                inline_resource("Dockerfile", grader_dockerfile.encode()),
                inline_resource(REPOSITORY_SETUP, setup.encode()),
            )
        )
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=DockerBuildContext(files=tuple(grader_files))
    )
    verifier = (
        *verifier,
        inline_resource(VERIFIER_SPEC, render_spec(spec).encode()),
        inline_resource("test.sh", VERIFYIT_SCRIPT).model_copy(update={"mode": "0755"}),
    )
    original_tests = tuple(
        resource.model_copy(update={"path": "source_archive/tests/" + resource.path})
        for resource in task.resources.verifier
        if resource.path != "taskcompendium/archive-provenance.json"
    )
    grader = ScriptGrader(
        argv=("bash", "/tests/test.sh"),
        cwd="/",
        environment=environment,
        artifacts=(VerifierArtifact(source=WORKSPACE, target=WORKSPACE, kind=ArtifactKind.DIRECTORY),),
        answer_path=None,
        reward=TEST_SH_REWARD,
    )
    return NormalizedTask(
        task.model_copy(
            update={
                "environment_requirements": task.environment_requirements.model_copy(
                    update={"docker_build": actor_build}
                ),
                "answer_type": AnswerType.WORKSPACE_STATE,
                "grader": grader,
                "resources": ResourceGroups(
                    worker=task.resources.worker, verifier=verifier, oracle=(*task.resources.oracle, *original_tests)
                ),
                "tags": tags,
            }
        ),
        (
            NormalizationChange(
                field="environment/Dockerfile",
                reason="Carry repository test dependencies and the bundled verifier as an unresolved build recipe",
                original=original_dockerfile,
                replacement=dockerfile,
            ),
            *changes,
        ),
    )
