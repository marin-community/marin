# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a source's control submissions through the same grading path rollouts use."""

import asyncio
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from shellbox.machine import Command, MachineFactory, MachineSpec
from verifyit.grade import positive_candidate
from verifyit.spec import ExactSpec, MathSpec, McqSpec, NumericSpec

from taskcompendium.grading_result import GradeResult
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    NoGrader,
    ScriptGrader,
    SessionGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grader_workspace,
    grades_in_process,
    require_resolved_environment,
    verifyit_spec,
)
from taskcompendium.pipeline.fingerprints import function_code_identity
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    Controls,
    ControlSubmission,
    OracleCommand,
    Reply,
    VerificationReport,
    WorkspaceFiles,
)
from taskcompendium.pipeline.verification import answer_event, control_result
from taskcompendium.runtime.environment import prepare_machine_spec, validate_machine_spec
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from taskcompendium.runtime.shell import ShellEnvironment, upload_resources
from taskcompendium.runtime.task_grading import grade_task, sandbox_grade

CONTROLS_REVISION = "6"
ORACLE_TIMEOUT = 600.0
ORACLE_OUTPUT_LIMIT_BYTES = 1_048_576
FILE_SUBMISSION_MESSAGE = TextMessage(role="assistant", content="The submission is in the workspace.")
"""The final message of a file submission; the grader reads the captured files."""


class GradingMachines(Protocol):
    """Fresh grading machines for a campaign's verification backends."""

    def identity(self) -> dict[str, Any]:
        """Backends, worker image and other settings that can change a control's outcome."""
        ...

    def machine(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineFactory, MachineSpec]:
        """A selected factory and specification for ``environment``, with network access denied."""
        ...


class OracleFailed(Exception):
    """The oracle command ran and did not produce a submission."""


def answer_reply(task: TaskSpec, answer: str) -> Reply:
    """A final assistant reply carrying ``answer`` in the task's answer format."""
    return Reply(answer_event(task, answer))


def reference_reply(task: TaskSpec) -> Reply | None:
    """The in-process grader's own reference answer as a reply, when its mode has one."""
    grader = task.grader
    if not isinstance(grader, VerifyitGrader):
        return None
    spec = verifyit_spec(grader)
    if isinstance(spec, MathSpec):
        return answer_reply(task, rf"\boxed{{{spec.expected}}}")
    if isinstance(spec, NumericSpec | McqSpec):
        return answer_reply(task, spec.expected)
    if isinstance(spec, ExactSpec):
        candidate = positive_candidate(spec)
        return answer_reply(task, candidate) if candidate is not None else None
    return None


def controls_identity(controls: Controls) -> dict[str, Any]:
    """The control code and machine size that can change a control's outcome."""
    return {
        "revision": CONTROLS_REVISION,
        "golden": function_code_identity(controls.golden) if controls.golden is not None else None,
        "memory_mb": controls.memory_mb,
    }


def control_suite(controls: Controls, machines: GradingMachines | None) -> CheckSuite:
    """Check each sampled task with its golden submission, or an empty one when it has no golden."""
    return CheckSuite(
        id="controls",
        revision=CONTROLS_REVISION,
        parameters={**(machines.identity() if machines is not None else {}), **controls_identity(controls)},
        run=partial(run_controls, controls=controls, machines=machines),
    )


def run_controls(task: TaskSpec, *, controls: Controls, machines: GradingMachines | None) -> VerificationReport:
    return asyncio.run(_control_checks(task, controls, machines))


def _empty_submission(task: TaskSpec) -> ControlSubmission:
    """The empty submission for an in-process grader, which scores it without a machine."""
    if task.answer_type in CONVERSATION_ANSWERS:
        return Reply(TextMessage(role="assistant", content=""))
    return WorkspaceFiles({})


async def _control_checks(task: TaskSpec, controls: Controls, machines: GradingMachines | None) -> VerificationReport:
    require_resolved_environment(task.environment_requirements)
    grader = task.grader
    if isinstance(grader, NoGrader | SessionGrader):
        return VerificationReport(
            [
                CheckResult(
                    check="grader", status=CheckStatus.UNSUPPORTED, detail=f"A {grader.kind} grader has no controls"
                )
            ]
        )
    sandbox = None
    if not grades_in_process(grader):
        environment = grader.environment
        assert environment is not None
        require_resolved_environment(environment)
        if machines is None:
            raise ValueError("Sandbox controls require grading machines")
        sandbox = _Sandbox(machines, controls.memory_mb, _selected_machine(machines, environment, controls.memory_mb))
    golden = controls.golden(task) if controls.golden is not None else None
    if golden is not None:
        check = await _control(task, "golden", golden, 1.0, sandbox)
    elif sandbox is not None:
        check = await _empty_sandbox_control(task, sandbox)
    else:
        check = await _control(task, "empty", _empty_submission(task), 0.0, None)
    return VerificationReport([check])


def _selected_machine(
    machines: GradingMachines, environment: EnvironmentRequirements, memory_mb: int
) -> tuple[MachineFactory, MachineSpec]:
    """Validate a selected machine without building its environment."""
    require_resolved_environment(environment)
    factory, spec = machines.machine(environment, memory_mb)
    validate_machine_spec(environment, factory, spec)
    return factory, spec


@dataclass(frozen=True)
class _Sandbox:
    """Where a sandbox-graded task's controls run."""

    machines: GradingMachines
    memory_mb: int
    grader: tuple[MachineFactory, MachineSpec]


def _oracle_environment(task: TaskSpec) -> EnvironmentRequirements:
    """Use the agent's requirements, falling back to the grader only for an empty agent environment."""
    if task.environment_requirements != EnvironmentRequirements():
        return task.environment_requirements
    grader = task.grader
    if not isinstance(grader, VerifyitGrader | ScriptGrader) or grader.environment is None:
        raise ValueError("An oracle requires a grader environment")
    return grader.environment


async def _control(
    task: TaskSpec,
    name: str,
    submission: ControlSubmission,
    expected: float,
    sandbox: _Sandbox | None,
) -> CheckResult:
    if isinstance(submission, OracleCommand):
        if sandbox is None:
            raise ValueError("An oracle command requires a sandbox grader")
        environment = _oracle_environment(task)
        factory, spec = _selected_machine(sandbox.machines, environment, sandbox.memory_mb)
        try:
            attempt = await _oracle_attempt(task, submission, factory, spec, environment)
        except OracleFailed as error:
            return CheckResult(check=name, status=CheckStatus.FAIL, detail=str(error))
        except (RuntimeError, OSError) as error:
            return CheckResult(check=name, status=CheckStatus.INFRA_ERROR, detail=str(error))
    elif isinstance(submission, Reply):
        attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, submission.event)))
    else:
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE)), dict(submission.files)
        )
    grade: GradeResult = (
        grade_task(task, attempt) if sandbox is None else await sandbox_grade(task, attempt, *sandbox.grader)
    )
    return control_result(grade, name, expected)


async def _empty_sandbox_control(task: TaskSpec, sandbox: _Sandbox) -> CheckResult:
    """Run the grader on an empty answer in a fresh grader machine; it must run cleanly.

    A zero reward or a rejected submission passes; a positive reward marks the task defective.
    """
    try:
        grade = await grade_empty_in_sandbox(task, *sandbox.grader)
    except (RuntimeError, OSError) as error:
        return CheckResult(check="empty", status=CheckStatus.INFRA_ERROR, detail=str(error))
    return control_result(grade, "empty", 0.0)


async def _oracle_attempt(
    task: TaskSpec,
    command: OracleCommand,
    factory: MachineFactory,
    spec: MachineSpec,
    environment: EnvironmentRequirements,
) -> GradingAttempt:
    """Run the oracle in a fresh machine from ``factory`` with the worker and oracle files mounted."""
    workspace = environment.working_directory or grader_workspace(task.grader)
    async with asyncio.timeout(spec.startup_timeout):
        prepared = await asyncio.to_thread(
            prepare_machine_spec,
            environment,
            factory,
            spec,
            dict(os.environ),
        )
        machine = await factory.create(prepared)
    try:
        await upload_resources(
            machine, (*task.resources.all, *task.resources.worker, *task.resources.oracle), ORACLE_TIMEOUT
        )
        commands = [Command(("mkdir", "-p", workspace), timeout=ORACLE_TIMEOUT)]
        commands.extend(
            Command(("sh", "-c", setup), cwd=workspace, timeout=ORACLE_TIMEOUT, user="0")
            for setup in environment.setup_commands
        )
        commands.append(
            Command(
                (
                    "/bin/bash",
                    "-lc",
                    command.command,
                ),
                cwd=workspace,
                timeout=ORACLE_TIMEOUT,
                output_limit_bytes=ORACLE_OUTPUT_LIMIT_BYTES,
            )
        )
        for step in commands:
            result = await machine.run(step)
            if result.exit_code != 0:
                raise OracleFailed(
                    f"Oracle command exited {result.exit_code}: {result.stderr.decode(errors='replace')[-2000:]}"
                )
        if command.answer_file is None:
            evidence = await ShellEnvironment(
                machine, task.output_paths, ORACLE_TIMEOUT, ORACLE_OUTPUT_LIMIT_BYTES, task.output_directories
            ).evidence()
            return GradingAttempt(
                ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE)), evidence.files
            )
        # Relative answers are read from the same workspace as the oracle command.
        answer = await machine.run(
            Command(
                ("cat", command.answer_file),
                cwd=workspace,
                timeout=ORACLE_TIMEOUT,
                output_limit_bytes=ORACLE_OUTPUT_LIMIT_BYTES,
            )
        )
        if answer.exit_code != 0 or answer.stdout_truncated:
            raise OracleFailed(f"Oracle wrote no readable answer file {command.answer_file}")
        reply = TextMessage(role="assistant", content=answer.stdout.decode(errors="replace"))
        return GradingAttempt(ConversationTrace(events=(*task.context.events, reply)))
    finally:
        await machine.close()
