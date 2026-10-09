# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell tasks graded on the agent's files keep hidden tests and oracles off the agent's machine."""

import base64
import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from shellbox.machine import Backend, Command, DockerImage, ExitReason, HostImage, MachineFactory, MachineSpec, Result
from verifyit.spec import StdioSpec

from taskcompendium.convert.executable import (
    SOLUTION_PATHS,
    python_delivery,
    solve_script,
    tasktrove_archive_task,
)
from taskcompendium.convert.tasktrove import INSTRUCTION, SOLUTION_DIR, TaskFiles
from taskcompendium.convert.tasktrove_converted_task import ConvertedTask
from taskcompendium.convert.tasktrove_stdio_cases import SOLUTION_COMMAND
from taskcompendium.models import (
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    Source,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import (
    CheckStatus,
    Controls,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
)
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import ShellFactory

from .pipeline_stages import fixture_recipe

IMAGE = "test@sha256:" + "a" * 64
ENVIRONMENT = EnvironmentRequirements(docker_image=IMAGE, compatible_backends=(Backend.DOCKER,))
EXITED = Result(0, b"", b"", False, False, ExitReason.EXITED)
MISSING_FILE = Result(44, b"", b"", False, False, ExitReason.EXITED)
CONTROLS = Controls(golden=solve_script)
ARCHIVE = {
    INSTRUCTION: b"Read two integers and print their sum. Write your program to /app/solution.py.",
    "setup_files/readme.txt": b"Public setup",
    "tests/cases/input_1.txt": b"3 4\n",
    "tests/cases/output_1.txt": b"7\n",
    "tests/setup_files/seed.txt": b"oracle seed",
    "solution/solve.sh": b"printf 'print(7)\\n' > /app/solution.py\n",
}


def convert_sum(task: TaskFiles) -> ConvertedTask:
    """Pass every archive file except the instruction and oracle through to the task."""
    return ConvertedTask(
        instruction=task.text(INSTRUCTION),
        spec=StdioSpec(command=SOLUTION_COMMAND),
        dockerfile="",
        tags=("code",),
        data_files={
            path: data for path, data in task.files.items() if path != INSTRUCTION and not path.startswith(SOLUTION_DIR)
        },
    )


def archive_row(files: dict[str, bytes]) -> dict:
    """A row as ``unpack_task_binary`` decodes it."""
    return {
        "instruction": files[INSTRUCTION].decode(),
        "files": {path: base64.b64encode(data).decode() for path, data in files.items()},
    }


def archive_task(row: RawRow) -> NormalizedTask | ImportRejection:
    return tasktrove_archive_task(
        row, convert=convert_sum, environment=ENVIRONMENT, grader_environment=ENVIRONMENT, output_paths=SOLUTION_PATHS
    )


def convert_archive(row: RawRow, _context: ConversionContext) -> NormalizedTask | ImportRejection:
    return archive_task(row)


@pytest.fixture
def executable_row():
    return RawRow(
        "program-1",
        Source(dataset="test/program", revision="1", row="1", importer_revision="1"),
        archive_row(ARCHIVE),
    )


@pytest.fixture
def executable_task(executable_row):
    result = archive_task(executable_row)
    assert isinstance(result, NormalizedTask)
    return TaskSpec.model_validate_json(result.task.model_dump_json())


@dataclass
class GradingMachine:
    """A machine that stores uploaded files and stands in for the external grading service."""

    verdict_status: str = "scored"
    files: dict[str, bytes] = field(default_factory=dict)
    closed: bool = False

    async def run(self, command: Command) -> Result:
        program = command.argv[0]
        if program == "mkdir":
            return EXITED
        if command.argv[:2] == ("rm", "-f"):
            for path in command.argv[2:]:
                self.files.pop(path, None)
            return EXITED
        if program == "tar":
            with tarfile.open(fileobj=io.BytesIO(self.files[command.argv[2]])) as archive:
                for member in archive.getmembers():
                    stream = archive.extractfile(member)
                    assert stream is not None
                    self.files["/" + member.name] = stream.read()
            return EXITED
        if program == "/bin/bash" and "/solution/solve.sh" in self.files:
            # The oracle command runs the fixture's solve.sh, which writes the reference program.
            self.files["/app/solution.py"] = b"print(7)\n"
            return EXITED
        if program == "python3":
            # The verdict depends on the submitted program, never on a verdict an agent supplied.
            reward = {b"print(7)\n": 1.0, b"partial": 0.25}.get(self.files.get("/app/solution.py", b""), 0.0)
            self.files["/logs/verifier/verdict.json"] = json.dumps(
                {"status": self.verdict_status, "reward": reward, "detail": {}}
            ).encode()
            return EXITED
        # A capture read: ``bash -c <script> capture <path> <limit>``.
        data = self.files.get(command.argv[4])
        if data is None:
            return MISSING_FILE
        limit = command.output_limit_bytes
        return Result(0, data[:limit], b"", len(data) > limit, False, ExitReason.EXITED)

    async def upload(self, source: Path, target: str) -> None:
        self.files[target] = source.read_bytes()

    async def download(self, source: str, target: Path) -> None:
        target.write_bytes(self.files[source])

    async def close(self) -> None:
        self.closed = True


@dataclass
class GradingMachines:
    backend: Backend = Backend.DOCKER
    machines: list[GradingMachine] = field(default_factory=list)
    verdict_status: str = "scored"

    async def create(self, spec: MachineSpec) -> GradingMachine:
        machine = GradingMachine(verdict_status=self.verdict_status)
        self.machines.append(machine)
        return machine


@dataclass
class ControlMachines:
    """The campaign's grading machines, all from one fake machine service."""

    factory: GradingMachines = field(default_factory=GradingMachines)

    def identity(self) -> dict:
        return {"backend": "fixture"}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        assert environment.docker_image is not None
        return self.factory, MachineSpec(DockerImage(environment.docker_image), memory_mb=memory_mb)


@dataclass
class RoutedMachines:
    """A campaign that grades local environments on the host and runs every image in a sandbox."""

    local: GradingMachines = field(default_factory=lambda: GradingMachines(backend=Backend.LOCAL))
    sandbox: GradingMachines = field(default_factory=GradingMachines)

    def identity(self) -> dict:
        return {"backend": "fixture"}

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        if Backend.LOCAL in environment.compatible_backends:
            return self.local, MachineSpec(HostImage())
        assert environment.docker_image is not None
        return self.sandbox, MachineSpec(DockerImage(environment.docker_image), memory_mb=memory_mb)


def grading_spec(task: TaskSpec) -> MachineSpec:
    """The machine specification for the task's grading image."""
    assert isinstance(task.grader, VerifyitGrader) and task.grader.environment is not None
    image = task.grader.environment.docker_image
    assert image is not None
    return MachineSpec(DockerImage(image))


def without_oracle(task: TaskSpec) -> TaskSpec:
    return task.model_copy(update={"resources": task.resources.model_copy(update={"oracle": ()})})


def checks(report) -> dict[str, CheckStatus]:
    return {check.check: check.status for check in report.checks}


def test_archive_files_take_their_resource_roles(executable_task):
    def paths(resources):
        return {resource.path: resource_bytes(resource) for resource in resources}

    assert paths(executable_task.resources.worker) == {"setup_files/readme.txt": b"Public setup"}
    assert {"cases/input_1.txt", "cases/output_1.txt"} <= set(paths(executable_task.resources.verifier))
    assert paths(executable_task.resources.oracle) == {
        "tests/setup_files/seed.txt": b"oracle seed",
        "solution/solve.sh": ARCHIVE["solution/solve.sh"],
    }


@pytest.mark.asyncio
async def test_agent_machine_receives_only_public_files(executable_task):
    machines = GradingMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage(IMAGE)), {}, 30.0, 1024)
    environment = await factory.create(executable_task)
    try:
        assert machines.machines[0].files == {"/setup_files/readme.txt": b"Public setup"}
        machines.machines[0].files["/app/solution.py"] = b"print(7)\n"
        assert (await environment.evidence()).files == {"/app/solution.py": b"print(7)\n"}
    finally:
        await environment.close()


def test_linux_file_names_survive_conversion_and_traversal_cannot_produce_task():
    recipe = fixture_recipe(convert_archive)
    valid = normalize_row(
        {"locator": "valid", "data": archive_row({**ARCHIVE, "setup_files/seeds/dir1/inner:file1.txt": b"x"})}, recipe
    )
    task = TaskSpec.model_validate(valid["audit"]["normalized"])
    assert "setup_files/seeds/dir1/inner:file1.txt" in {resource.path for resource in task.resources.worker}
    escaping = archive_row({**ARCHIVE, "../escape.txt": b"x"})
    rejected = normalize_row({"locator": "invalid", "data": escaping}, recipe)["audit"]
    assert rejected["normalized"] is None
    assert rejected["normalization_rejection"]["kind"] == "converter_error"
    assert rejected["normalization_rejection"]["reason"] == "invalid_task_spec"
    assert rejected["raw"]["data"] == escaping


def test_task_without_an_oracle_grades_an_empty_submission(executable_task):
    machines = ControlMachines()
    report = run_controls(without_oracle(executable_task), controls=CONTROLS, machines=machines)
    assert checks(report) == {"empty": CheckStatus.PASS}
    assert machines.factory.machines and all(machine.closed for machine in machines.factory.machines)


@pytest.mark.parametrize(
    "verdict_status, expected", [("invalid_task", CheckStatus.FAIL), ("infra_error", CheckStatus.INFRA_ERROR)]
)
def test_ungraded_zero_reward_cannot_pass_the_empty_control(executable_task, verdict_status, expected):
    machines = ControlMachines(GradingMachines(verdict_status=verdict_status))
    report = run_controls(without_oracle(executable_task), controls=CONTROLS, machines=machines)
    assert checks(report) == {"empty": expected}


@pytest.mark.parametrize("program, reward", [(b"print(7)\n", 1.0), (b"print(0)\n", 0.0)])
@pytest.mark.asyncio
async def test_captured_submission_cannot_supply_its_own_reward(executable_task, program, reward):
    machines = GradingMachines()
    files = {
        "/app/solution.py": program,
        "/logs/verifier/verdict.json": b'{"status":"scored","reward":1,"detail":{}}',
        "/tests/cases/output_1.txt": b"0\n",
    }
    attempt = GradingAttempt(
        ConversationTrace(events=(*executable_task.context.events, TextMessage(role="assistant", content="done"))),
        files,
    )
    grade = await grade_in_sandbox(executable_task, attempt, machines, grading_spec(executable_task), timeout=10)
    assert grade.reward == reward
    assert machines.machines[0].files["/tests/cases/output_1.txt"] == b"7\n"
    assert machines.machines[0].closed


def incompatible(task: TaskSpec, role: str) -> TaskSpec:
    """``task`` with the worker's or grader's environment declared for another backend."""
    data = task.model_dump(mode="json")
    requirements = data["environment_requirements"] if role == "worker" else data["grader"]["environment"]
    requirements["compatible_backends"] = ["gvisor"]
    return TaskSpec.model_validate_json(json.dumps(data))


@pytest.mark.asyncio
async def test_agent_machine_rejects_an_incompatible_backend_before_start(executable_task):
    machines = GradingMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage(IMAGE)), {}, 1, 1024)
    with pytest.raises(ValueError, match="not declared compatible"):
        await factory.create(incompatible(executable_task, "worker"))
    assert not machines.machines


def test_a_local_grader_grades_the_oracle_output_of_a_sandbox_of_the_agent_image(executable_task):
    data = executable_task.model_dump(mode="json")
    data["grader"]["environment"] = {"compatible_backends": ["local"], "packages_lock": "fixture/requirements.lock"}
    machines = RoutedMachines()
    report = run_controls(TaskSpec.model_validate_json(json.dumps(data)), controls=CONTROLS, machines=machines)
    assert checks(report) == {"golden": CheckStatus.PASS}
    (oracle,) = machines.sandbox.machines
    (grader,) = machines.local.machines
    assert "/solution/solve.sh" in oracle.files and "/solution/solve.sh" not in grader.files
    assert grader.files["/app/solution.py"] == b"print(7)\n"
    assert oracle.closed and grader.closed


def test_grader_controls_reject_an_incompatible_backend_before_start(executable_task):
    machines = ControlMachines()
    with pytest.raises(ValueError, match="not declared compatible"):
        run_controls(incompatible(executable_task, "grader"), controls=CONTROLS, machines=machines)
    assert not machines.factory.machines


@pytest.mark.parametrize(
    "instruction, tests, output_paths",
    [
        ("Write the parser to `/app/parser.py`.", {}, ("/app/parser.py",)),
        (
            "Build the package at /app/shapes with `area.py` and `volume.py`.",
            {},
            ("/app/shapes/area.py", "/app/shapes/volume.py"),
        ),
        ("Implement /app/lib.py; the grader runs test_lib.py.", {}, ("/app/lib.py",)),
        (
            "Implement `add(a, b)` returning the sum of two integers.",
            {"tests/test_add.py": b"import json\nfrom collections import Counter\nfrom calculator import add\n"},
            ("/app/calculator.py",),
        ),
    ],
)
def test_python_delivery_captures_the_files_the_task_names(instruction, tests, output_paths):
    delivery = python_delivery(instruction, tests)
    assert not isinstance(delivery, ImportRejection)
    assert delivery.output_paths == output_paths
    assert delivery.instruction.startswith(instruction)


def test_python_delivery_names_an_inferred_module_in_the_instruction():
    delivery = python_delivery("Implement `add(a, b)`.", {"tests/test_add.py": b"from calculator import add\n"})
    assert not isinstance(delivery, ImportRejection)
    assert delivery.instruction.endswith("Delivery: write the requested implementation to `/app/calculator.py`.\n")
    assert [change.field for change in delivery.changes] == ["instruction", "output_paths"]


@pytest.mark.parametrize(
    "tests",
    [
        {"tests/test_add.py": b"from calculator import add, subtract\n"},
        {"tests/test_add.py": b"from calculator import add\n", "tests/test_mul.py": b"from multiply import add\n"},
    ],
    ids=["unmentioned_name", "two_modules"],
)
def test_python_delivery_rejects_an_unstated_output_contract(tests):
    rejection = python_delivery("Implement `add(a, b)`.", tests)
    assert isinstance(rejection, ImportRejection)
    assert (rejection.kind, rejection.reason) == (ImportFailureKind.UNSUPPORTED, "unsupported_public_output_contract")
