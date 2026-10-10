# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control submissions check a grader through the grading path that rollouts use."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    Machine,
    MachineSpec,
    Result,
    ShellSimBuiltins,
    UnsupportedMachineSpec,
)
from verifyit.spec import SchemaFormat

from taskcompendium.convert.answers import (
    json_schema_task,
    numeric_answer_task,
)
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    DockerBuildContext,
    EnvironmentRequirements,
    NoGrader,
    ResourceGroups,
    SessionGrader,
    Source,
    TaskSpec,
)
from taskcompendium.pipeline.controls import answer_reply, control_suite, reference_reply
from taskcompendium.pipeline.models import CheckStatus, Controls, OracleCommand, RawRow, Reply, WorkspaceFiles
from taskcompendium.runtime.resources import inline_resource

from .pipeline_stages import GRADER_IMAGE, FixtureGradingMachines, ShellSimImages, UnavailableImages, script_graded

PASS, FAIL, DEFECT = CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.DEFECT
ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
REFERENCE_CONTROLS = Controls(golden=reference_reply)
AGENT_IMAGE = "agent@sha256:" + "a" * 64


@dataclass(eq=False)
class ImageMachine:
    """A ShellSim machine standing in for ``image``; it records the paths uploaded to it."""

    machine: Machine
    image: str
    uploads: list[str] = field(default_factory=list)

    async def run(self, command: Command) -> Result:
        return await self.machine.run(command)

    async def upload(self, source: Path, target: str) -> None:
        self.uploads.append(target)
        await self.machine.upload(source, target)

    async def download(self, source: str, target: Path) -> None:
        await self.machine.download(source, target)

    async def close(self) -> None:
        await self.machine.close()


@dataclass
class RecordingImages:
    """Fresh ShellSim machines, each recording the image it stands in for."""

    backend = Backend.DOCKER
    created: list[ImageMachine] = field(default_factory=list)

    async def create(self, spec: MachineSpec) -> ImageMachine:
        assert isinstance(spec.source, DockerImage)
        machine = ImageMachine(await ShellSimImages().create(spec), spec.source.reference)
        self.created.append(machine)
        return machine


def checks(task: TaskSpec, controls: Controls, machines: FixtureGradingMachines | None = None) -> dict[str, CheckStatus]:
    return {check.check: check.status for check in control_suite(controls, machines).run(task).checks}


def in_process_task(kind: str) -> TaskSpec:
    if kind == "numeric":
        task = numeric_answer_task(ROW, prompt="What is 3 + 4?", answer="7", tolerance_abs=0, tolerance_rel=0)
    else:
        task = json_schema_task(
            ROW, prompt="Return a JSON object.", schema=json.dumps({"type": "object"}), schema_format=SchemaFormat.JSON
        )
    assert isinstance(task, TaskSpec)
    return task


def answer_task() -> TaskSpec:
    """A conversation answer graded in its image by comparing /app/answer.txt with the public input."""
    task = in_process_task("numeric").model_copy(
        update={"resources": ResourceGroups(worker=(inline_resource("data/expected.txt", b"7"),))}
    )
    return script_graded(task, b'test "$(cat answer.txt)" = 7\n')


def file_task() -> TaskSpec:
    """A file answer at /app/solution.txt, with an oracle script that derives it from a worker file."""
    task = in_process_task("numeric").model_copy(
        update={
            "answer_type": AnswerType.FILE,
            "output_paths": ("/app/solution.txt",),
            "resources": ResourceGroups(
                worker=(inline_resource("data/expected.txt", b"7"),),
                oracle=(inline_resource("solution/solve.sh", b"cp /data/expected.txt solution.txt\n"),),
            ),
        }
    )
    return script_graded(task, b'test "$(cat solution.txt)" = 7\n', answer_path=None)


def wrong_reply(task: TaskSpec) -> Reply:
    return answer_reply(task, "__incorrect_answer__")


def test_reference_golden_passes_for_in_process_graders():
    assert checks(in_process_task("numeric"), REFERENCE_CONTROLS) == {"golden": PASS}


def test_golden_fails_when_the_grader_rejects_it():
    assert checks(in_process_task("numeric"), Controls(golden=wrong_reply)) == {"golden": FAIL}


@pytest.mark.parametrize("controls", [Controls(), REFERENCE_CONTROLS], ids=["none", "no_reference"])
def test_task_without_a_golden_is_checked_with_an_empty_submission(controls):
    # A JSON schema grader has no reference instance, so reference_reply also has no golden.
    assert checks(in_process_task("schema"), controls) == {"empty": PASS}


@pytest.mark.parametrize("grader", [NoGrader(reason="Source evaluator unavailable"), SessionGrader()])
def test_graders_without_offline_grading_have_unsupported_controls(grader):
    task = in_process_task("numeric").model_copy(update={"grader": grader})
    report = control_suite(REFERENCE_CONTROLS, FixtureGradingMachines()).run(task)
    assert [check.status for check in report.checks] == [CheckStatus.UNSUPPORTED]


@pytest.mark.parametrize("role", ["actor", "grader"])
def test_controls_reject_unresolved_build_before_machine_selection_or_oracle_fallback(role):
    task = file_task()
    build = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_build=DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM mutable:latest"),)),
    )
    if role == "actor":
        task = task.model_copy(update={"environment_requirements": build})
    else:
        task = task.model_copy(update={"grader": task.grader.model_copy(update={"environment": build})})
    # No machine provider is supplied: unresolved recipes must be rejected before
    # providers are required, and must never use the grader image for an oracle.
    suite = control_suite(Controls(golden=lambda _: OracleCommand("bash /solution/solve.sh")), None)
    with pytest.raises(UnsupportedMachineSpec):
        suite.run(task)


@pytest.mark.parametrize(
    "task,golden",
    [
        (answer_task(), lambda task: answer_reply(task, "7")),
        (answer_task(), lambda _: OracleCommand("cp /data/expected.txt answer.out", answer_file="answer.out")),
        (answer_task(), lambda _: OracleCommand("printf 7 > /tmp/answer.out", answer_file="/tmp/answer.out")),
        (file_task(), lambda _: WorkspaceFiles({"/app/solution.txt": b"7"})),
        (file_task(), lambda _: OracleCommand("bash /solution/solve.sh")),
    ],
    ids=["reply", "oracle_relative_answer_file", "oracle_absolute_answer_file", "workspace_files", "oracle_files"],
)
def test_sandbox_golden_is_graded_in_a_fresh_grader_machine(task, golden):
    assert checks(task, Controls(golden=golden), FixtureGradingMachines()) == {"golden": PASS}


@pytest.mark.parametrize(
    "task,oracle",
    [
        (file_task(), OracleCommand("exit 3")),
        (file_task(), OracleCommand("true")),
        (answer_task(), OracleCommand("true", answer_file="answer.out")),
    ],
    ids=["oracle_exits_nonzero", "oracle_writes_no_file", "oracle_writes_no_answer_file"],
)
def test_oracle_without_a_correct_submission_fails_the_golden_control(task, oracle):
    assert checks(task, Controls(golden=lambda _: oracle), FixtureGradingMachines()) == {"golden": FAIL}


@pytest.mark.parametrize(
    "script, status",
    [(b'test "$(cat answer.txt)" = 7\n', PASS), (b"test -f answer.txt && test ! -s answer.txt\n", DEFECT)],
    ids=["grader_rejects_empty_answer", "grader_accepts_only_empty_answer"],
)
def test_empty_control_runs_the_script_grader_on_an_empty_answer_file(script, status):
    # The second grader rewards only an empty /app/answer.txt, so its defect shows the grader ran on one.
    images = RecordingImages()
    task = script_graded(answer_task(), script)
    assert checks(task, Controls(), FixtureGradingMachines(images)) == {"empty": status}
    assert [machine.image for machine in images.created] == [GRADER_IMAGE]


def test_grader_that_rewards_an_empty_submission_marks_the_task_defective():
    task = script_graded(file_task(), b"true\n", answer_path=None)
    assert checks(task, Controls(), FixtureGradingMachines()) == {"empty": DEFECT}


@pytest.mark.parametrize(
    "golden,check",
    [
        (lambda _: WorkspaceFiles({"/app/solution.txt": b"7"}), "golden"),
        (lambda _: OracleCommand("bash /solution/solve.sh"), "golden"),
        (lambda _: None, "empty"),
    ],
    ids=["workspace_files", "oracle", "no_golden"],
)
def test_unavailable_grading_machines_are_infrastructure_errors(golden, check):
    controls = Controls(golden=golden)
    assert checks(file_task(), controls, FixtureGradingMachines(UnavailableImages())) == {check: CheckStatus.INFRA_ERROR}


@pytest.mark.parametrize(
    "agent_image, oracle_image",
    [(AGENT_IMAGE, AGENT_IMAGE), (None, GRADER_IMAGE)],
    ids=["agent_image", "no_agent_image"],
)
def test_oracle_runs_in_the_agent_image_and_otherwise_in_the_grader_image(agent_image, oracle_image):
    task = file_task()
    if agent_image is not None:
        requirements = EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=agent_image
        )
        task = TaskSpec.model_validate(task.model_copy(update={"environment_requirements": requirements}).model_dump())
    images = RecordingImages()
    controls = Controls(golden=lambda _: OracleCommand("bash /solution/solve.sh"))
    assert checks(task, controls, FixtureGradingMachines(images)) == {"golden": PASS}
    oracles = [machine for machine in images.created if "/solution/solve.sh" in machine.uploads]
    assert [machine.image for machine in oracles] == [oracle_image]
    assert {machine.image for machine in images.created if machine not in oracles} == {GRADER_IMAGE}


class OracleMachines(FixtureGradingMachines):
    """Provide image machines and a simulator, leaving lock preparation unsupported."""

    def machine(self, environment: EnvironmentRequirements, memory_mb: int):
        if environment.command_semantics == CommandSemantics.SHELL_SIMULATOR:
            return ShellSimMachineFactory(), MachineSpec(ShellSimBuiltins(), memory_mb=memory_mb)
        return self.factory, MachineSpec(DockerImage(environment.docker_image or GRADER_IMAGE), memory_mb=memory_mb)


def test_oracle_uses_simulator_directory_and_setup_before_native_grading(monkeypatch):
    monkeypatch.setenv("ORACLE_READY_NAME", "ready.txt")
    task = answer_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.SHELL_SIMULATOR,
                working_directory="/agent",
                setup_commands=('cp /data/expected.txt "$ANSWER_FILE"',),
                environment_variables={"ANSWER_FILE": "${ORACLE_READY_NAME}"},
            )
        }
    )
    controls = Controls(
        golden=lambda _: OracleCommand('test "$PWD" = /agent && cat ready.txt > answer.out', answer_file="answer.out")
    )
    assert checks(task, controls, OracleMachines()) == {"golden": PASS}


def test_lock_backed_oracle_rejects_image_only_provider_before_acquisition():
    task = file_task().model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS, packages_lock="unbuilt/requirements.lock"
            )
        }
    )
    images = RecordingImages()
    suite = control_suite(Controls(golden=lambda _: OracleCommand("bash /solution/solve.sh")), OracleMachines(images))
    with pytest.raises(UnsupportedMachineSpec):
        suite.run(task)
    assert images.created == []
