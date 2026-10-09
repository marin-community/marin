# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deferred repository images and private workspace graders."""

from functools import cache
from pathlib import Path

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.tasktrove import TEST_SH_REWARD, TaskFiles
from taskcompendium.models import (
    AnswerType,
    ArtifactKind,
    DockerBuildContext,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    TaskResource,
    TaskSpec,
    VerifierArtifact,
)
from taskcompendium.pipeline.models import ImportRejection, NormalizationChange, NormalizedTask
from taskcompendium.runtime.local import context_paths
from taskcompendium.runtime.resources import inline_resource
from verifyit.spec import Spec, render_spec

WORKSPACE = "/testbed"
VERIFYIT_PACKAGE = Path(__file__).resolve().parents[5] / "lib/verifyit"
VERIFYIT_CONTEXT = "taskcompendium-verifyit"
PUBLIC_SETUP_PREFIX = "## Environment Setup (complete these steps first)\n\n```bash\n"
REPOSITORY_SETUP = "taskcompendium-repository-setup.sh"
GRADER_SETUP = "taskcompendium-grader-setup.sh"
VERIFYIT_INSTALL = f"""
COPY {VERIFYIT_CONTEXT}/ /opt/taskcompendium-verifyit/
RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python 3.12 /opt/taskcompendium-verifyit
"""
VERIFIER_SPEC = "taskcompendium-verifier.toml"
VERIFYIT_SCRIPT = f"#!/bin/bash\nset -euo pipefail\nexec verifyit /tests/{VERIFIER_SPEC}\n".encode()


@cache
def verifyit_build_files() -> tuple[TaskResource, ...]:
    """Bundle the checked-out verifier package so the recipe names the code used in conversion."""
    package = VERIFYIT_PACKAGE
    paths = [package / "pyproject.toml", package / "README.md", *context_paths(package / "src/verifyit")]
    return tuple(
        inline_resource(f"{VERIFYIT_CONTEXT}/{path.relative_to(package).as_posix()}", path.read_bytes())
        for path in paths
    )


def repository_build_task(
    task: TaskSpec,
    files: TaskFiles,
    *,
    spec: Spec,
    dockerfile: str,
    verifier: tuple[TaskResource, ...],
    tags: tuple[str, ...],
    changes: tuple[NormalizationChange, ...] = (),
    grader_setup: str = "",
) -> NormalizedTask | ImportRejection:
    """Carry public setup into an unresolved image and grade a private copy of the workspace."""
    instruction = files.text("instruction.md")
    if not instruction.startswith(PUBLIC_SETUP_PREFIX):
        return unsupported("unsupported_repository_setup", "Expected the source's explicit Environment Setup bash block")
    setup, end, _ = instruction.removeprefix(PUBLIC_SETUP_PREFIX).partition("\n```")
    if not end:
        return unsupported("unsupported_repository_setup", "Environment Setup bash block is not closed")
    original_dockerfile = files.text("environment/Dockerfile")
    dockerfile += VERIFYIT_INSTALL
    if task.resources.worker:
        dockerfile += "COPY taskcompendium-public/ /\n"
    dockerfile += (
        f"COPY {REPOSITORY_SETUP} /opt/{REPOSITORY_SETUP}\n"
        f"RUN bash -e /opt/{REPOSITORY_SETUP} && rm -rf {WORKSPACE} && mkdir -p {WORKSPACE}\n"
    )
    if grader_setup:
        dockerfile += f"COPY {GRADER_SETUP} /opt/{GRADER_SETUP}\nRUN bash -e /opt/{GRADER_SETUP}\n"
    # Original environment bytes stay with provenance; the edited recipe is an environment input.
    context_files = tuple(
        resource.model_copy(
            update={
                "path": resource.path.removeprefix("environment/"),
                **(
                    {"source": inline_resource("Dockerfile", dockerfile.encode()).source}
                    if resource.path == "environment/Dockerfile"
                    else {}
                ),
            }
        )
        for resource in task.resources.oracle
        if resource.path.startswith("environment/")
    )
    if any(
        resource.path.startswith((VERIFYIT_CONTEXT, "taskcompendium-public", REPOSITORY_SETUP, GRADER_SETUP))
        for resource in context_files
    ):
        return unsupported("build_context_collision", "Source context occupies the bundled verifier path")
    public_context = tuple(
        resource.model_copy(update={"path": "taskcompendium-public/" + resource.path})
        for resource in task.resources.worker
    )
    environment = EnvironmentRequirements(
        docker_build=DockerBuildContext(
            files=(
                *context_files,
                *verifyit_build_files(),
                *public_context,
                inline_resource(REPOSITORY_SETUP, setup.encode()),
                *((inline_resource(GRADER_SETUP, grader_setup.encode()),) if grader_setup else ()),
            )
        )
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
                    update={"docker_build": environment.docker_build}
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
