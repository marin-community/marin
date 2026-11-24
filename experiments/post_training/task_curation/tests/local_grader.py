# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade tasks in a locally built grader image, which stands in for every grader environment but one.

A campaign grades most tasks in the worker, in a uv environment built from the lock compiled from the
grader packages' requirements; this image holds the same packages and files. Build it from the repository
root with::

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
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import (
    Backend,
    DockerImage,
    ExecutionRequirements,
    HostImage,
    MachineSpec,
)
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import ConversationTrace, EnvironmentRequirements, GradingAttempt, TaskResource, TaskSpec
from taskcompendium.pipeline.controls import FILE_SUBMISSION_MESSAGE
from taskcompendium.pipeline.models import Reply, WorkspaceFiles
from taskcompendium.runtime.task_grading import sandbox_grade

LOCAL_GRADER_IMAGE = "local/task-curation-grader:test"
LEAN_KERNEL_GRADER_IMAGE = "local/task-curation-lean-kernel:test"
"""The Lean grader image for ``LEAN_KERNEL_GRADER_PACKAGES``: build it as above with that environment and this tag.
Its tests run in this image, since the shared local image lacks its toolchain."""


@dataclass(frozen=True)
class LocalGraderFactory:
    """Starts a local grader image as a machine of the backend the environment declares."""

    backend: Backend
    image: str = LOCAL_GRADER_IMAGE

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        if self.backend == Backend.LOCAL:
            check_local_environment(requirements, spec)
            return
        DockerMachineFactory().check_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> DockerMachine:
        return await DockerMachineFactory().create(replace(spec, source=DockerImage(self.image)))


@dataclass(frozen=True)
class LocalGraderMachines:
    """Docker machines that run one local grader image whatever environment a task names."""

    image: str = LOCAL_GRADER_IMAGE

    def identity(self) -> dict:
        return {"backend": Backend.DOCKER.value, "image": self.image}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[LocalGraderFactory, MachineSpec]:
        if environment.packages_lock is not None:
            return LocalGraderFactory(Backend.LOCAL, self.image), MachineSpec(HostImage(), memory_mb=memory_mb)
        assert environment.docker_image is not None
        return LocalGraderFactory(Backend.DOCKER, self.image), MachineSpec(
            DockerImage(environment.docker_image), memory_mb=memory_mb
        )


def local_grader_machines(image: str = LOCAL_GRADER_IMAGE) -> LocalGraderMachines:
    """The local machines for ``image``, skipping the calling test when Docker or the image is missing."""
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed")
    inspected = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False)
    if inspected.returncode != 0:
        pytest.skip(f"{image} is not built; see {__name__}")
    return LocalGraderMachines(image)


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
