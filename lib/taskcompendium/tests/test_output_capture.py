# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep recursive submissions inside their declared roots when staging a grader."""

import io
import json
import tarfile
from dataclasses import dataclass, field

import pytest
from shellbox.machine import (
    Backend,
    DockerImage,
    MachineSpec,
)
from verifyit.spec import StdioSpec

from taskcompendium.convert.tasks import workspace_task
from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    GradingAttempt,
    Source,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import inline_resource
from taskcompendium.runtime.shell import ShellFactory

from .test_runtime import EXITED, FileMachine, FileMachines, finished

SOLUTION_PATHS = ("/app/solution.py", "/app/solution.cpp")

IMAGE = "test@sha256:" + "a" * 64


@pytest.fixture
def executable_task():
    row = RawRow("program-1", Source(dataset="test/program", revision="1", row="1", importer_revision="1"), {})
    return workspace_task(
        row,
        instruction="Read two integers and print their sum in /app/solution.py.",
        spec=StdioSpec(command="python3 /app/solution.py"),
        environment=EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS),
        grader_environment=EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS),
        output_paths=SOLUTION_PATHS,
        verifier=(inline_resource("cases/input_1.txt", b"3 4\n"), inline_resource("cases/output_1.txt", b"7\n")),
        worker=(inline_resource("setup_files/readme.txt", b"Public setup"),),
    )


@dataclass
class GradingMachine(FileMachine):
    """A grader machine that unpacks the staged archive and writes a fixed verdict."""

    async def run(self, command):
        if command.argv[0] == "tar":
            with tarfile.open(fileobj=io.BytesIO(self.files[command.argv[2]])) as archive:
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    self.files["/" + member.name] = stream.read()
            return EXITED
        if command.argv[0] != "python3":
            return await super().run(command)
        self.files["/logs/verifier/verdict.json"] = json.dumps(
            {"status": "scored", "reward": 0.0, "detail": {}}
        ).encode()
        return EXITED


@dataclass
class GradingMachines:
    backend = Backend.DOCKER
    machines: list[GradingMachine] = field(default_factory=list)

    async def create(self, spec):
        machine = GradingMachine()
        self.machines.append(machine)
        return machine


@pytest.mark.asyncio
async def test_recursive_submission_stages_only_declared_files(executable_task):
    task = executable_task.model_copy(update={"output_paths": ("/app",)})
    machines = GradingMachines()
    files = {
        "/app/package/solution.py": b"print(7)",
        "/tests/reference.py": b"tampered",
        "/solution/secret.py": b"tampered",
        "/application/other.py": b"outside",
    }
    await grade_in_sandbox(task, GradingAttempt(finished(task), files), machines, MachineSpec(DockerImage(IMAGE)))
    uploaded = machines.machines[0].files
    assert uploaded["/app/package/solution.py"] == b"print(7)"
    assert all(path not in uploaded for path in ("/tests/reference.py", "/solution/secret.py", "/application/other.py"))


@pytest.mark.parametrize("root", ["/tests", "/logs", "/solution", "/"])
@pytest.mark.asyncio
async def test_private_capture_root_rejected_before_acquisition_and_staging(executable_task, root):
    task = executable_task.model_copy(update={"output_paths": (root,)})
    machines = FileMachines()
    with pytest.raises(ValueError, match="private"):
        await ShellFactory(machines, MachineSpec(DockerImage(IMAGE)), {}, 1, 1024).create(task)
    with pytest.raises(ValueError, match="private"):
        await grade_in_sandbox(
            task,
            GradingAttempt(finished(task), {root + "/file": b"tampered"}),
            machines,
            MachineSpec(DockerImage(IMAGE)),
        )
    assert not machines.machines


@pytest.mark.asyncio
async def test_submission_traversal_rejected_before_grader_acquisition(executable_task):
    task = executable_task.model_copy(update={"output_paths": ("/app",)})
    machines = GradingMachines()
    with pytest.raises(ValueError, match="normalized"):
        await grade_in_sandbox(
            task,
            GradingAttempt(finished(task), {"/app/../tests/reference.py": b"tampered"}),
            machines,
            MachineSpec(DockerImage(IMAGE)),
        )
    assert not machines.machines
