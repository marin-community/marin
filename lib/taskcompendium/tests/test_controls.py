# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Control submissions check a grader through the grading path that rollouts use."""

from dataclasses import dataclass

import pytest
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.context import MachineRoute
from shellbox.machine import (
    Backend,
    Command,
    DockerImage,
    ExitReason,
    Machine,
    MachineSpec,
    PackageSpec,
    Result,
    UnsupportedMachineSpec,
)

from taskcompendium.models import (
    CommandSemantics,
    EnvironmentRequirements,
    GraderMachine,
    NoGrader,
    ScriptGrader,
    TaskSpec,
)
from taskcompendium.pipeline.controls import answer_reply, control_suite
from taskcompendium.pipeline.models import CheckStatus, Controls, MachineRole, OracleCommand, WorkspaceFiles
from taskcompendium.runtime.local import LocalGraderMachines

from .control_machines import (
    AGENT_IMAGE,
    DEFECT,
    FAIL,
    PASS,
    REFERENCE_CONTROLS,
    RecordingImages,
    answer_task,
    checks,
    file_task,
    in_process_task,
    wrong_reply,
)
from .pipeline_stages import (
    GRADER_IMAGE,
    FixtureGradingMachines,
    UnavailableImages,
    script_graded,
)


def test_reference_golden_passes_for_in_process_graders():
    assert checks(in_process_task("numeric"), REFERENCE_CONTROLS) == {"golden": PASS}


def test_golden_fails_when_the_grader_rejects_it():
    assert checks(in_process_task("numeric"), Controls(golden=wrong_reply)) == {"golden": FAIL}


@pytest.mark.parametrize("controls", [Controls(), REFERENCE_CONTROLS], ids=["none", "no_reference"])
def test_task_without_a_golden_is_checked_with_an_empty_submission(controls):
    # A JSON schema grader has no reference instance, so reference_reply also has no golden.
    assert checks(in_process_task("schema"), controls) == {"empty": PASS}


def test_graders_without_offline_grading_have_unsupported_controls():
    task = in_process_task("numeric").model_copy(update={"grader": NoGrader(reason="Source evaluator unavailable")})
    report = control_suite(REFERENCE_CONTROLS, FixtureGradingMachines()).run(task)
    assert [check.status for check in report.checks] == [CheckStatus.UNSUPPORTED]


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
    grader = task.grader
    assert isinstance(grader, ScriptGrader)
    assert isinstance(grader.placement, GraderMachine)
    environment = grader.placement.environment.model_copy(
        update={"setup_commands": ("test -f /app/answer.txt && test ! -s /app/answer.txt",)}
    )
    task = task.model_copy(
        update={"grader": grader.model_copy(update={"placement": GraderMachine(environment=environment)})}
    )
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

    def routes(self, environment: EnvironmentRequirements, memory_mb: int):
        if environment.command_semantics == CommandSemantics.SHELL_SIMULATOR:
            return (MachineRoute(ShellSimMachineFactory(), MachineSpec(PackageSpec(), memory_mb=memory_mb)),)
        return (
            MachineRoute(
                self.factory,
                MachineSpec(DockerImage(environment.docker_image or GRADER_IMAGE), memory_mb=memory_mb),
            ),
        )


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


@dataclass(eq=False)
class FailingOracleMachine:
    """A machine that stages files but refuses every oracle command."""

    async def run(self, command: Command) -> Result:
        if command.argv[:1] == ("/bin/bash",):
            return Result(1, b"", b"the oracle is missing", False, False, ExitReason.EXITED)
        return Result(0, b"", b"", False, False, ExitReason.EXITED)

    async def upload(self, source, target) -> None:
        return

    async def download(self, source, target) -> None:
        raise RuntimeError("No files exist on this machine")

    async def close(self) -> None:
        return


@dataclass(eq=False)
class FailingOracleFactory:
    """A machine factory whose machines refuse every oracle command."""

    backend: Backend = Backend.DAYTONA

    def check_environment(self, requirements, spec) -> None:
        return

    async def create(self, spec: MachineSpec) -> FailingOracleMachine:
        return FailingOracleMachine()


class RefusingThenImageMachines(FixtureGradingMachines):
    """A task ladder whose cheapest backend refuses the requirements and whose next backend serves the oracle."""

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        refusing = MachineRoute(ShellSimMachineFactory(), MachineSpec(PackageSpec(), memory_mb=memory_mb))
        return (refusing, *super().routes(environment, memory_mb))


class FailingThenImageMachines(FixtureGradingMachines):
    """A task ladder whose cheapest backend fails the oracle and whose next backend serves it."""

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        routes = super().routes(environment, memory_mb)
        if environment.docker_image != AGENT_IMAGE:
            return routes
        failing = MachineRoute(
            FailingOracleFactory(backend=Backend.SHELLSIM), MachineSpec(PackageSpec(), memory_mb=memory_mb)
        )
        return (failing, *routes)


@pytest.mark.parametrize(
    "machines,refusal_status,verifier_refusal",
    [
        (RefusingThenImageMachines, "unsupported", ("verifier", "shellsim", "unsupported")),
        (FailingThenImageMachines, "fail", None),
    ],
    ids=["refusing_backend", "failing_backend"],
)
def test_oracle_fails_up_from_a_backend_that_cannot_serve_it(machines, refusal_status, verifier_refusal):
    task = file_task()
    requirements = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=AGENT_IMAGE)
    task = TaskSpec.model_validate(task.model_copy(update={"environment_requirements": requirements}).model_dump())
    images = RecordingImages()
    report = control_suite(Controls(golden=lambda _: OracleCommand("bash /solution/solve.sh")), machines(images)).run(
        task
    )
    assert {check.check: check.status for check in report.checks} == {"golden": PASS}
    # The task backend's verdict lands when its oracle's submission passes the control; the
    # grader machine's ladder serves where it can.
    assert [
        (verdict.role.value, verdict.backend.value, verdict.status.value)
        for check in report.checks
        for verdict in check.verdicts
    ] == [
        ("task", "shellsim", refusal_status),
        ("task", "docker", "skipped"),
        *((verifier_refusal,) if verifier_refusal is not None else ()),
        ("verifier", "docker", "pass"),
        ("task", "docker", "pass"),
    ]
    oracles = [machine for machine in images.created if "/solution/solve.sh" in machine.uploads]
    assert [machine.image for machine in oracles] == [AGENT_IMAGE]


@dataclass(eq=False)
class ScriptedAnswerMachine(FailingOracleMachine):
    """Canned command results with a fixed payload for ``cat``."""

    answer: bytes

    async def run(self, command: Command) -> Result:
        if command.argv[:1] == ("cat",):
            return Result(0, self.answer, b"", False, False, ExitReason.EXITED)
        return Result(0, b"", b"", False, False, ExitReason.EXITED)


@dataclass(eq=False)
class ScriptedAnswerFactory:
    """A machine factory whose machines answer the oracle with ``answer``."""

    answer: bytes
    backend: Backend = Backend.DAYTONA

    def check_environment(self, requirements, spec) -> None:
        return

    async def create(self, spec: MachineSpec) -> ScriptedAnswerMachine:
        return ScriptedAnswerMachine(self.answer)


def scripted_answer_route(answer: bytes, backend: Backend, memory_mb: int) -> MachineRoute:
    return MachineRoute(ScriptedAnswerFactory(answer, backend=backend), MachineSpec(PackageSpec(), memory_mb=memory_mb))


class WrongThenRightAnswerMachines(FixtureGradingMachines):
    """A task ladder whose cheapest backend's oracle submits a wrong answer and whose next backend passes."""

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        routes = super().routes(environment, memory_mb)
        if environment.docker_image != AGENT_IMAGE:
            return routes
        wrong = scripted_answer_route(b"wrong\n", Backend.SHELLSIM, memory_mb)
        right = scripted_answer_route(b"7\n", Backend.DAYTONA, memory_mb)
        return (wrong, right)


class PassingThenHigherMachines(FixtureGradingMachines):
    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        if environment.docker_image != AGENT_IMAGE:
            return super().routes(environment, memory_mb)
        return tuple(scripted_answer_route(b"7\n", backend, memory_mb) for backend in (Backend.LOCAL, Backend.DAYTONA))


def test_passing_oracle_retains_higher_candidates_for_other_deployments():
    requirements = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=AGENT_IMAGE)
    task = answer_task().model_copy(update={"environment_requirements": requirements})
    report = control_suite(
        Controls(golden=lambda _: OracleCommand("true", answer_file="answer.out")),
        PassingThenHigherMachines(RecordingImages()),
    ).run(task)
    assert {check.check: check.status for check in report.checks} == {"golden": PASS}
    assert [
        verdict.backend for check in report.checks for verdict in check.verdicts if verdict.role is MachineRole.TASK
    ] == [
        Backend.LOCAL,
        Backend.LOCAL,
    ]


def test_oracle_fails_up_when_its_submission_fails_grading():
    """An oracle that runs cleanly but submits a wrong answer fails up to the next task backend."""
    task = answer_task()
    requirements = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=AGENT_IMAGE)
    task = TaskSpec.model_validate(task.model_copy(update={"environment_requirements": requirements}).model_dump())
    report = control_suite(
        Controls(golden=lambda _: OracleCommand("true", answer_file="answer.out")),
        WrongThenRightAnswerMachines(RecordingImages()),
    ).run(task)
    assert {check.check: check.status for check in report.checks} == {"golden": PASS}
    assert [
        (verdict.role.value, verdict.backend.value, verdict.status.value)
        for check in report.checks
        for verdict in check.verdicts
    ] == [
        ("task", "shellsim", "skipped"),
        ("verifier", "docker", "fail"),
        ("task", "shellsim", "fail"),
        ("task", "daytona", "skipped"),
        ("verifier", "docker", "pass"),
        ("task", "daytona", "pass"),
    ]


@dataclass(eq=False)
class RefusingAtAcquisitionFactory:
    """An image factory that passes admission but refuses when acquiring the machine."""

    backend: Backend = Backend.DAYTONA

    def check_environment(self, requirements, spec) -> None:
        return

    async def create(self, spec: MachineSpec) -> Machine:
        raise UnsupportedMachineSpec("Only numeric root image users are supported")


class RefusingThenServingMachines(FixtureGradingMachines):
    """A verifier ladder whose cheapest backend refuses at acquisition and whose next backend serves."""

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        if environment.docker_image == AGENT_IMAGE:
            return (scripted_answer_route(b"7\n", Backend.DAYTONA, memory_mb),)
        refused = MachineRoute(
            RefusingAtAcquisitionFactory(backend=Backend.LOCAL),
            MachineSpec(DockerImage(environment.docker_image), memory_mb=memory_mb),
        )
        return (refused, *super().routes(environment, memory_mb))


def test_verifier_refusal_at_acquisition_fails_up_to_the_next_backend():
    """A backend that refuses only when acquiring the machine gets its own verdict, not an unsupported task."""
    task = answer_task()
    requirements = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=AGENT_IMAGE)
    task = TaskSpec.model_validate(task.model_copy(update={"environment_requirements": requirements}).model_dump())
    report = control_suite(
        Controls(golden=lambda _: OracleCommand("true", answer_file="answer.out")),
        RefusingThenServingMachines(RecordingImages()),
    ).run(task)
    assert {check.check: check.status for check in report.checks} == {"golden": PASS}
    assert [
        (verdict.role.value, verdict.backend.value, verdict.status.value)
        for check in report.checks
        for verdict in check.verdicts
    ] == [
        ("task", "daytona", "skipped"),
        ("verifier", "local", "unsupported"),
        ("verifier", "docker", "pass"),
        ("task", "daytona", "pass"),
    ]


def test_local_oracle_and_grader_setup_run_without_changing_host_user(local_machine_environment):
    environment, _factory = local_machine_environment
    task = answer_task()
    task = task.model_copy(
        update={
            "environment_requirements": environment.model_copy(
                update={"working_directory": "/agent", "setup_commands": ("cp /data/expected.txt ready.txt",)}
            ),
            "grader": task.grader.model_copy(
                update={
                    "placement": GraderMachine(
                        environment=environment.model_copy(
                            update={"working_directory": "/app", "setup_commands": ("test -f /data/expected.txt",)}
                        )
                    )
                }
            ),
        }
    )
    controls = Controls(golden=lambda _: OracleCommand("cat ready.txt > answer.out", answer_file="answer.out"))

    report = control_suite(controls, LocalGraderMachines()).run(task)

    assert {check.check: check.status for check in report.checks} == {"golden": PASS}


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
    report = suite.run(task)
    assert {check.check: check.status for check in report.checks} == {"golden": CheckStatus.UNSUPPORTED}
    assert [verdict.backend for check in report.checks for verdict in check.verdicts] == ["docker"]
    assert images.created == []
