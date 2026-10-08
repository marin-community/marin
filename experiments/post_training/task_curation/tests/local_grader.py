# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade tasks in a locally built image of the grader packages, which stands in for every grader environment.

A campaign grades most tasks in the worker, in a uv environment built from the grader lock; this image
holds the same packages. Build it from the repository root with::

    uv run python -c "from pathlib import Path; \\
    from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES; \\
    from experiments.post_training.task_curation.images.build import write_context; \\
    write_context(GRADER_PACKAGES, Path('/tmp/grader-context'))"
    docker build --platform linux/amd64 --build-context verifyit=lib/verifyit/src/verifyit \\
        -t local/task-curation-grader:test /tmp/grader-context
"""

import asyncio
import shutil
import subprocess
from dataclasses import dataclass, replace

import pytest
from shellbox.backends.docker.machine import DockerMachine, DockerMachineFactory
from shellbox.machine import Backend, DockerImage, HostImage, MachineSpec
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import ConversationTrace, EnvironmentRequirements, GradingAttempt, TaskResource, TaskSpec
from taskcompendium.pipeline.controls import FILE_SUBMISSION_MESSAGE
from taskcompendium.pipeline.models import Reply, WorkspaceFiles
from taskcompendium.runtime.task_grading import sandbox_grade

LOCAL_GRADER_IMAGE = "local/task-curation-grader:test"


@dataclass(frozen=True)
class LocalGraderFactory:
    """Starts the local grader image as a machine of the backend the environment declares."""

    backend: Backend

    async def create(self, spec: MachineSpec) -> DockerMachine:
        return await DockerMachineFactory().create(replace(spec, source=DockerImage(LOCAL_GRADER_IMAGE)))


class LocalGraderMachines:
    """Docker machines that run the local grader image whatever environment a task names."""

    def identity(self) -> dict:
        return {"backend": Backend.DOCKER.value, "image": LOCAL_GRADER_IMAGE}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[LocalGraderFactory, MachineSpec]:
        if Backend.LOCAL in environment.compatible_backends:
            return LocalGraderFactory(Backend.LOCAL), MachineSpec(HostImage(), memory_mb=memory_mb)
        assert environment.docker_image is not None
        return LocalGraderFactory(Backend.DOCKER), MachineSpec(
            DockerImage(environment.docker_image), memory_mb=memory_mb
        )


def local_grader_machines() -> LocalGraderMachines:
    """The local machines, skipping the calling test when Docker or the image is missing."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")
    inspected = subprocess.run(["docker", "image", "inspect", LOCAL_GRADER_IMAGE], capture_output=True, check=False)
    if inspected.returncode != 0:
        pytest.skip(f"{LOCAL_GRADER_IMAGE} is not built; see {__name__}")
    return LocalGraderMachines()


def grade(
    task: TaskSpec, submission: Reply | WorkspaceFiles, machines: LocalGraderMachines, memory_mb: int
) -> GradeResult:
    """Grade a reply, or the files an agent left in its workspace, as a campaign does."""
    if isinstance(submission, Reply):
        attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, submission.event)))
    else:
        trace = ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE))
        attempt = GradingAttempt(trace, dict(submission.files))
    environment = task.grader.environment
    assert environment is not None
    return asyncio.run(sandbox_grade(task, attempt, *machines.machine(environment, memory_mb)))


def with_verifier_file(task: TaskSpec, replacement: TaskResource) -> TaskSpec:
    """``task`` with the verifier resource at ``replacement``'s path swapped for it."""
    verifier = tuple(replacement if item.path == replacement.path else item for item in task.resources.verifier)
    assert replacement in verifier, f"{task.id} ships no {replacement.path}"
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": verifier})})
