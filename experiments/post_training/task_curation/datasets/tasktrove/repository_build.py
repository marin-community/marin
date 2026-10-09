# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deferred repository images and private workspace graders."""

import re
from functools import cache
from pathlib import Path

from taskcompendium.convert.answers import unsupported
from taskcompendium.convert.tasktrove import DOCKERFILE, TEST_SH_REWARD, UV_IMAGE, TaskFiles
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
VERIFYIT_INSTALL = (
    "# --- verifyit ---\n"
    "RUN command -v git >/dev/null || (apt-get update && apt-get install -y --no-install-recommends git"
    " && rm -rf /var/lib/apt/lists/*)\n"
    f"""COPY --from={UV_IMAGE} /uv /usr/local/bin/uv
COPY {VERIFYIT_CONTEXT}/ /opt/taskcompendium-verifyit/
RUN UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python ">=3.11" /opt/taskcompendium-verifyit
"""
)
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
) -> NormalizedTask | ImportRejection:
    """Keep the legacy actor recipe and declare a fresh workspace grading environment."""
    original_dockerfile = files.text(DOCKERFILE)
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in dockerfile.splitlines())).strip("\n")
    dockerfile = body + "\n\n" + VERIFYIT_INSTALL
    # Original environment bytes stay with provenance; the edited recipe is an environment input.
    context_files = tuple(
        resource.model_copy(
            update={
                "path": resource.path.removeprefix("environment/"),
                **(
                    {"source": inline_resource("Dockerfile", dockerfile.encode()).source}
                    if resource.path == DOCKERFILE
                    else {}
                ),
            }
        )
        for resource in task.resources.oracle
        if resource.path.startswith("environment/")
    )
    if any(
        resource.path.startswith((VERIFYIT_CONTEXT, "taskcompendium-public", REPOSITORY_SETUP))
        for resource in context_files
    ):
        return unsupported("build_context_collision", "Source context occupies the bundled verifier path")
    actor_build = DockerBuildContext(files=(*context_files, *verifyit_build_files()))
    # Canonical ScriptGrader uses a fresh machine. Its dependencies must be prepared without
    # changing the actor recipe that the legacy shared-environment exporter consumes.
    instruction = files.text("instruction.md")
    setup, end, _ = instruction.removeprefix(PUBLIC_SETUP_PREFIX).partition("\n```")
    grader_files = list(actor_build.files)
    if instruction.startswith(PUBLIC_SETUP_PREFIX) and end:
        grader_dockerfile = dockerfile
        if task.resources.worker:
            grader_dockerfile += "COPY taskcompendium-public/ /\n"
            grader_files.extend(
                resource.model_copy(update={"path": "taskcompendium-public/" + resource.path})
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
    environment = EnvironmentRequirements(docker_build=DockerBuildContext(files=tuple(grader_files)))
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
