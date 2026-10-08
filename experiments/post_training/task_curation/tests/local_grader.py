# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade tasks in the locally built grader image, which stands in for the grader image a task pins.

Build the image from the repository root with::

    docker build --platform linux/amd64 --build-context verifyit=lib/verifyit/src/verifyit \\
        -t local/task-curation-grader:test experiments/post_training/task_curation/images/grader
"""

import asyncio
import shutil
import subprocess
from dataclasses import replace

import pytest
from shellbox.backends.docker.machine import DockerMachine, DockerMachineFactory
from shellbox.machine import Backend, DockerImage, MachineSpec
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import ConversationTrace, EnvironmentRequirements, GradingAttempt, TaskResource, TaskSpec
from taskcompendium.pipeline.controls import FILE_SUBMISSION_MESSAGE
from taskcompendium.pipeline.models import Reply, WorkspaceFiles
from taskcompendium.runtime.task_grading import sandbox_grade

LOCAL_GRADER_IMAGE = "local/task-curation-grader:test"


class LocalGraderMachines:
    """Docker machines that run the local grader image whatever image a task names."""

    backend = Backend.DOCKER

    def identity(self) -> dict:
        return {"backend": self.backend.value, "image": LOCAL_GRADER_IMAGE}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple["LocalGraderMachines", MachineSpec]:
        assert environment.docker_image is not None
        return self, MachineSpec(DockerImage(environment.docker_image), memory_mb=memory_mb)

    async def create(self, spec: MachineSpec) -> DockerMachine:
        return await DockerMachineFactory().create(replace(spec, source=DockerImage(LOCAL_GRADER_IMAGE)))


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
