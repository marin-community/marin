# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep executable fixtures private and grade submissions independently."""

import base64
import io
import json
import tarfile
from dataclasses import dataclass, field
from functools import partial

import pytest
from shellbox.machine import Backend, DockerImage, ExitReason, MachineSpec, Result
from verifyit.spec import StdioSpec, spec_to_table

from taskcompendium.datasets.executable_tasks import SubmissionControl, executable_checks, grade_files, normalize
from taskcompendium.models import Source, TaskSpec, VerifyitGrader
from taskcompendium.pipeline.models import CheckStatus, RawRow, ReviewRubric, TaskPolicy
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.runtime.shell import ShellFactory

from .pipeline_stages import fixture_recipe
from .test_runtime import FileMachine, FileMachines


@pytest.fixture
def executable_row():
    def encoded(value: bytes) -> str:
        return base64.b64encode(value).decode()

    return RawRow(
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


@pytest.fixture
def executable_task(executable_row):
    task = normalize(executable_row, "test@sha256:" + "a" * 64, output_paths=("/app/solution.py", "/app/solution.cpp"))
    assert isinstance(task, TaskSpec)
    return TaskSpec.model_validate_json(task.model_dump_json())


def grading_spec(task: TaskSpec) -> MachineSpec:
    """The machine specification for the task's grading image."""
    assert isinstance(task.grader, VerifyitGrader) and task.grader.environment is not None
    image = task.grader.environment.docker_image
    assert image is not None
    return MachineSpec(DockerImage(image))


def test_linux_fixture_names_survive_normalization_and_traversal_cannot_produce_task(executable_row):
    recipe = fixture_recipe(
        TaskPolicy(
            normalize=partial(normalize, image="test@sha256:" + "a" * 64, output_paths=("/app/solution.py",)),
            rubric=ReviewRubric(id="fixture", version="1", criteria=()),
        )
    )
    executable_row.data["converted"]["data_files"]["setup_files/seeds/dir1/inner:file1.txt"] = "eA=="
    valid = normalize_row({"locator": "valid", "data": executable_row.data}, recipe)
    executable_row.data["converted"]["data_files"]["../escape.txt"] = "eA=="
    rejected = normalize_row({"locator": "invalid", "data": executable_row.data}, recipe)
    assert valid["audit"]["normalized"] is not None
    task = TaskSpec.model_validate(valid["audit"]["normalized"])
    assert "setup_files/seeds/dir1/inner:file1.txt" in {resource.path for resource in task.resources.worker}
    assert rejected["audit"]["normalized"] is None
    assert rejected["audit"]["normalization_rejection"]["reason"] == "invalid_task_spec"
    assert rejected["audit"]["raw"]["data"] == executable_row.data
    assert rejected["audit"]["normalization_rejection"]["kind"] == "converter_error"
    assert rejected["audit"]["decision"]["disposition"] == "defer"


async def test_executable_agent_environment_excludes_tests_and_oracle(executable_task):
    machines = FileMachines()
    image = executable_task.environment_requirements.docker_image
    assert image is not None
    factory = ShellFactory(machines, MachineSpec(DockerImage(image)), {}, 30.0, 1024)
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
    verdict_status: str = "scored"

    async def run(self, command):
        if command.argv[:2] == ("rm", "-f"):
            for path in command.argv[2:]:
                self.files.pop(path, None)
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
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
        reward = {b"print(7)\n": 1.0, b"partial": 0.25}.get(self.files.get("/app/solution.py", b""), 0.0)
        self.files["/logs/verifier/verdict.json"] = json.dumps(
            {"status": self.verdict_status, "reward": reward, "detail": {}}
        ).encode()
        return Result(0, b"", b"", False, False, ExitReason.EXITED)


@dataclass
class GradingMachines:
    backend = Backend.DOCKER
    machines: list[GradingMachine] = field(default_factory=list)
    verdict_status: str = "scored"

    async def create(self, spec):
        machine = GradingMachine(verdict_status=self.verdict_status)
        self.machines.append(machine)
        return machine


async def test_missing_executable_oracle_still_runs_negative_controls(executable_task):
    task = executable_task.model_copy(update={"resources": executable_task.resources.model_copy(update={"oracle": ()})})
    machines = GradingMachines()
    report = await executable_checks(
        task,
        factory=machines,
        machine_spec=grading_spec(task),
    )
    assert {check.check: check.status for check in report.checks} == {
        "missing_submission": CheckStatus.PASS,
        "empty_submission": CheckStatus.PASS,
        "wrong_submission": CheckStatus.PASS,
        "oracle": CheckStatus.SKIPPED,
    }
    assert all(machine.closed for machine in machines.machines)


@pytest.mark.parametrize(
    "verdict_status,expected",
    [("invalid_task", CheckStatus.FAIL), ("infra_error", CheckStatus.INFRA_ERROR)],
)
async def test_ungraded_zero_reward_cannot_pass_negative_controls(executable_task, verdict_status, expected):
    task = executable_task.model_copy(update={"resources": executable_task.resources.model_copy(update={"oracle": ()})})
    machines = GradingMachines(verdict_status=verdict_status)
    report = await executable_checks(
        task,
        factory=machines,
        machine_spec=grading_spec(task),
        controls=(SubmissionControl("wrong_submission", {"/app/solution.py": b"wrong"}, 0.0),),
    )
    assert {check.check: check.status for check in report.checks} == {
        "wrong_submission": expected,
        "oracle": CheckStatus.SKIPPED,
    }
    assert all(machine.closed for machine in machines.machines)


async def test_executable_controls_enforce_each_declared_reward(executable_task):
    task = executable_task.model_copy(update={"resources": executable_task.resources.model_copy(update={"oracle": ()})})
    machines = GradingMachines()
    report = await executable_checks(
        task,
        factory=machines,
        machine_spec=grading_spec(task),
        controls=(
            SubmissionControl("advertised_partial", {"/app/solution.py": b"partial"}, 0.25),
            SubmissionControl("strict_zero", {"/app/solution.py": b"partial"}, 0.0),
            SubmissionControl("full_credit", {"/app/solution.py": b"print(7)\n"}, 1.0),
        ),
    )
    assert {check.check: check.status for check in report.checks} == {
        "advertised_partial": CheckStatus.PASS,
        "strict_zero": CheckStatus.FAIL,
        "full_credit": CheckStatus.PASS,
        "oracle": CheckStatus.SKIPPED,
    }
    assert all(machine.closed for machine in machines.machines)


@pytest.mark.parametrize("program,reward", [(b"print(7)\n", 1.0), (b"print(0)\n", 0.0)])
async def test_captured_submission_cannot_supply_its_own_reward(executable_task, program, reward):
    machines = GradingMachines()
    files = {
        "/app/solution.py": program,
        "/logs/verifier/verdict.json": b'{"status":"scored","reward":1,"detail":{}}',
        "/tests/cases/output_1.txt": b"0\n",
    }
    grade = await grade_files(executable_task, files, machines, machine_spec=grading_spec(executable_task), timeout=10)
    assert grade.reward == reward
    assert machines.machines[0].files["/tests/cases/output_1.txt"] == b"7\n"
    assert machines.machines[0].closed


@pytest.mark.parametrize("role", ["worker", "grader"])
async def test_serialized_backend_contract_rejects_incompatible_runtime_before_start(executable_task, role):
    data = executable_task.model_dump(mode="json")
    requirements = data["environment_requirements"] if role == "worker" else data["grader"]["environment"]
    requirements["compatible_backends"] = ["gvisor"]
    task = TaskSpec.model_validate_json(json.dumps(data))
    machines = GradingMachines()
    with pytest.raises(ValueError, match="not declared compatible"):
        if role == "worker":
            await ShellFactory(machines, MachineSpec(DockerImage("test")), {}, 1, 1024).create(task)
        else:
            await executable_checks(task, factory=machines, machine_spec=grading_spec(task))
    assert not machines.machines
