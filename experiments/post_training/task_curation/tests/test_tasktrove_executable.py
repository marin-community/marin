# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shell tasks graded on the agent's files keep hidden tests and oracles off the agent's machine."""

import base64
import io
import json
import tarfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import pytest
from shellbox.machine import Backend, Command, DockerImage, ExitReason, MachineFactory, MachineSpec, Result
from taskcompendium.models import (
    CommandSemantics,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.controls import run_controls
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import (
    CheckStatus,
    Controls,
    ImportRejection,
    NormalizedTask,
    RawRow,
)
from taskcompendium.pipeline.transforms import normalize_row
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.runtime.shell import ShellFactory
from verifyit.spec import StdioSpec

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    INSTRUCTION,
    SOLUTION_DIR,
    TaskFiles,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.executable import (
    SOLUTION_PATHS,
    converted_workspace_task,
    solve_script,
    tasktrove_archive_task,
)
from experiments.post_training.task_curation.datasets.tasktrove.conversion.result import ConvertedTask
from experiments.post_training.task_curation.datasets.tasktrove.conversion.stdio_cases import SOLUTION_COMMAND
from experiments.post_training.task_curation.datasets.tasktrove.conversion.verifyit_build import verifyit_build_context
from experiments.post_training.task_curation.tasktrove.harbor_export import harbor_record
from lib.taskcompendium.tests.pipeline_stages import fixture_recipe

IMAGE = "test@sha256:" + "a" * 64
ENVIRONMENT = EnvironmentRequirements(docker_image=IMAGE, command_semantics=CommandSemantics.LINUX_PROCESS)
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
    result = cast(NormalizedTask, archive_task(executable_row))
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
        if program == "python3":
            # The verdict depends on the submitted program, never on a verdict an agent supplied.
            reward = {b"print(7)\n": 1.0, b"partial": 0.25}.get(self.files.get("/app/solution.py", b""), 0.0)
            self.files["/logs/verifier/verdict.json"] = json.dumps(
                {"status": self.verdict_status, "reward": reward, "detail": {}}
            ).encode()
            return EXITED
        # Read the verifier's result file through the machine boundary.
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
    grade = await grade_in_sandbox(executable_task, attempt, machines, MachineSpec(DockerImage(IMAGE)), timeout=10)
    assert grade.reward == reward
    assert machines.machines[0].files["/tests/cases/output_1.txt"] == b"7\n"
    assert machines.machines[0].closed


def simulator_task(task: TaskSpec) -> TaskSpec:
    """Require simulated shell behavior from a native process machine."""
    data = task.model_dump(mode="json")
    requirements = data["environment_requirements"]
    requirements["docker_image"] = None
    requirements["command_semantics"] = "shell_simulator"
    return TaskSpec.model_validate_json(json.dumps(data))


@pytest.mark.asyncio
async def test_agent_machine_rejects_incompatible_command_semantics_before_start(executable_task):
    machines = GradingMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage(IMAGE)), {}, 1, 1024)
    with pytest.raises(ValueError, match="semantics"):
        await factory.create(simulator_task(executable_task))
    assert not machines.machines


def test_grader_controls_reject_simulation_of_linux_processes_before_start(executable_task):
    machines = ControlMachines(factory=GradingMachines(backend=Backend.SHELLSIM))
    with pytest.raises(ValueError, match="native Linux processes"):
        run_controls(executable_task, controls=CONTROLS, machines=machines)
    assert not machines.factory.machines


@pytest.mark.parametrize("language,execution", [("python", "shared"), ("cpp", "separate")])
def test_converter_language_survives_harbor_export_without_changing_payload(executable_row, language, execution):
    converted = convert_sum(TaskFiles(ARCHIVE))
    environment = ENVIRONMENT
    if execution == "shared":
        # This explicit recipe has no archived Dockerfile to fall back to.
        build = verifyit_build_context("FROM python:3.12-slim\nWORKDIR /app\n", (), package=VERIFYIT_PACKAGE)
        environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=build)
    records = []
    for declared_language in ("", language):
        task = converted_workspace_task(
            executable_row,
            replace(converted, language=declared_language),
            instruction=converted.instruction,
            environment=environment,
            grader_environment=environment,
            output_paths=SOLUTION_PATHS,
        )
        records.append(
            harbor_record(
                {
                    "task_json": task.model_dump_json(),
                    "original_path": "program-1",
                    "source_row": "source/tasks.parquet:0",
                },
                grader_image=IMAGE,
                fallback_actor_image=IMAGE,
                family="stdio",
            )
        )
    baseline, result = records
    assert result.language == language
    assert replace(result, language="") == baseline
