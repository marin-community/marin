# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep executable fixtures private and grade submissions independently."""

import base64
import io
import json
import tarfile
from dataclasses import dataclass, field

import pytest
from shellbox.machine import DockerImage, ExitReason, MachineSpec, Result
from verifyit.spec import StdioSpec, spec_to_table

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.datasets.executable_tasks import normalize
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.shell import ShellFactory
from taskcompendium.verifiers.executable import TaskTroveExecutableVerifier, grade_submission

from .test_runtime import FileMachine, FileMachines


@pytest.fixture
def executable_task():
    def encoded(value: bytes) -> str:
        return base64.b64encode(value).decode()

    row = RawRow(
        "program-1",
        Source(dataset="test/program", revision="1", row="1", importer_revision="1"),
        {
            "converted": {
                "instruction": "Read two integers and print their sum in /app/solution.py.",
                "grader_spec": spec_to_table(StdioSpec(command="python3 /app/solution.py")),
                "data_files": {
                    "setup_files/readme.txt": encoded(b"Public setup"),
                    "tests/cases/input_1.txt": encoded(b"3 4\n"),
                    "tests/cases/output_1.txt": encoded(b"7\n"),
                },
                "control_files": {"solution/solve.sh": encoded(b"printf 'oracle' > /app/solution.py\n")},
            }
        },
    )
    task = normalize(row, "sha256:test-image", 30.0, 256)
    assert isinstance(task, TaskSpec)
    return TaskSpec.model_validate_json(task.model_dump_json())


async def test_executable_agent_environment_excludes_tests_and_oracle(executable_task):
    machines = FileMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage("test")), {"backend": "test"}, 30.0, 1024)
    environment = await factory.create(executable_task)
    try:
        assert machines.machines[0].files == {"/setup_files/readme.txt": b"Public setup"}
        machines.machines[0].files["/app/solution.py"] = b"print(7)\n"
        evidence = await environment.evidence()
        assert evidence.files == {"/app/solution.py": b"print(7)\n"}
    finally:
        await environment.close()


@dataclass
class GradingMachine(FileMachine):
    async def run(self, command):
        if command.argv[0] == "tar":
            with tarfile.open(fileobj=io.BytesIO(self.files[command.argv[2]])) as archive:
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    self.files["/" + member.name] = stream.read()
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        if command.argv[0] != "python3":
            return await super().run(command)
        # External grading-service fake: the trusted verdict depends on the
        # submitted file, never an agent's supplied verdict/reward artifact.
        reward = float(self.files["/app/solution.py"] == b"print(7)\n")
        self.files["/logs/verifier/verdict.json"] = json.dumps(
            {"status": "scored", "reward": reward, "detail": {}}
        ).encode()
        return Result(0, b"", b"", False, False, ExitReason.EXITED)


@dataclass
class GradingMachines:
    machines: list[GradingMachine] = field(default_factory=list)

    async def create(self, spec):
        machine = GradingMachine()
        self.machines.append(machine)
        return machine


@pytest.mark.parametrize("program,reward", [(b"print(7)\n", 1.0), (b"print(0)\n", 0.0)])
async def test_captured_submission_cannot_supply_its_own_reward(executable_task, program, reward):
    verifier = TaskTroveExecutableVerifier.model_validate_json(executable_task.verifier.parameters_json)
    machines = GradingMachines()
    grade = await grade_submission(
        verifier,
        {
            "/app/solution.py": program,
            "/logs/verifier/verdict.json": b'{"status":"scored","reward":1,"detail":{}}',
            "/tests/cases/output_1.txt": b"0\n",
        },
        machines,
    )
    assert grade.reward == reward
    assert machines.machines[0].files["/tests/cases/output_1.txt"] == b"7\n"
    assert machines.machines[0].closed
