# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a source's control submissions through its configured grading boundary."""

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

from shellbox.context import MachineRoute
from shellbox.machine import Backend, Command, MachineFactory, MachineSpec, UnsupportedMachineSpec
from verifyit.grade import positive_candidate
from verifyit.modes.grade_judge import JudgeConnection
from verifyit.spec import ExactSpec, MathSpec, McqSpec, NumericSpec, StructureClosureSpec

from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AgentMachine,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    NoGrader,
    TaskSpec,
    TextMessage,
    VerifyitGrader,
    grader_environment,
    grades_in_process,
    grades_with_judge,
    verifyit_spec,
)
from taskcompendium.pipeline.fingerprints import callable_identity
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    CheckSuite,
    Controls,
    ControlSubmission,
    OracleCommand,
    Reply,
    TierRole,
    TierVerdict,
    VerificationReport,
    WorkspaceFiles,
)
from taskcompendium.pipeline.verification import answer_event, control_result
from taskcompendium.runtime.environment import prepare_machine_spec, setup_user, validate_machine_spec
from taskcompendium.runtime.grading import grade_empty_in_sandbox
from taskcompendium.runtime.requirements import require_resolved_environment
from taskcompendium.runtime.shell import ShellEnvironment, upload_resources
from taskcompendium.runtime.task_grading import grade_task, sandbox_grade

CONTROLS_REVISION = "23"
ORACLE_TIMEOUT = 600.0
ORACLE_OUTPUT_LIMIT_BYTES = 1_048_576
FILE_SUBMISSION_MESSAGE = TextMessage(role="assistant", content="The submission is in the workspace.")
"""The final message of a file submission; the grader reads the captured files."""


class GradingMachines(Protocol):
    """Fresh grading machines for a campaign's verification backends."""

    def identity(self) -> dict[str, Any]:
        """Backends, worker image and other settings that can change a control's outcome."""
        ...

    def routes(self, environment: EnvironmentRequirements, memory_mb: int) -> tuple[MachineRoute, ...]:
        """The tier candidates for ``environment``, cheapest first; empty when no tier satisfies it."""
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
    if isinstance(spec, StructureClosureSpec):
        return answer_reply(task, spec.expected[len(spec.prefix) :])
    if isinstance(spec, ExactSpec):
        candidate = positive_candidate(spec)
        return answer_reply(task, candidate) if candidate is not None else None
    return None


def controls_identity(controls: Controls) -> dict[str, Any]:
    """The control code and machine size that can change a control's outcome."""
    return {
        "revision": CONTROLS_REVISION,
        "golden": callable_identity(controls.golden) if controls.golden is not None else None,
        "negative": callable_identity(controls.negative) if controls.negative is not None else None,
        "negative_reward": callable_identity(controls.negative_reward) if controls.negative_reward is not None else None,
        "memory_mb": controls.memory_mb,
    }


def control_suite(
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None = None,
) -> CheckSuite:
    """Check each sampled task with a declared positive or negative submission, falling back to empty."""
    return CheckSuite(
        id="controls",
        revision=CONTROLS_REVISION,
        parameters={
            **(machines.identity() if machines is not None else {}),
            **controls_identity(controls),
            **({"judge_base_url": judge_connection.base_url} if judge_connection is not None else {}),
        },
        run=partial(run_controls, controls=controls, machines=machines, judge_connection=judge_connection),
    )


def run_controls(
    task: TaskSpec,
    *,
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None = None,
) -> VerificationReport:
    return asyncio.run(_control_checks(task, controls, machines, judge_connection))


def _empty_submission(task: TaskSpec) -> ControlSubmission:
    """The empty submission for an in-process grader, which scores it without a machine."""
    if task.answer_type in CONVERSATION_ANSWERS:
        return Reply(TextMessage(role="assistant", content=""))
    return WorkspaceFiles({})


async def _control_checks(
    task: TaskSpec,
    controls: Controls,
    machines: GradingMachines | None,
    judge_connection: JudgeConnection | None,
) -> VerificationReport:
    grader = task.grader
    if isinstance(grader, NoGrader):
        return VerificationReport(
            [
                CheckResult(
                    check="grader", status=CheckStatus.UNSUPPORTED, detail=f"A {grader.kind} grader has no controls"
                )
            ]
        )
    if isinstance(grader.placement, AgentMachine):
        return VerificationReport(
            [
                CheckResult(
                    check="grader",
                    status=CheckStatus.UNSUPPORTED,
                    detail="Controls do not run graders placed in the agent's machine",
                )
            ]
        )
    judge_task = grades_with_judge(task.grader)
    if judge_task and judge_connection is None:
        raise ValueError("Judge controls require an explicit remote connection")
    sandbox = None
    if not grades_in_process(grader) and not (judge_task and isinstance(grader, VerifyitGrader)):
        environment = grader_environment(grader)
        assert environment is not None
        if machines is None:
            raise ValueError("Sandbox controls require grading machines")
        sandbox = _Sandbox(
            machines,
            controls.memory_mb,
            environment,
            _tier_candidates(machines, environment, controls.memory_mb),
        )
    golden = controls.golden(task) if controls.golden is not None else None
    if golden is not None:
        outcome = await _control(task, "golden", golden, 1.0, sandbox, judge_connection)
    elif controls.negative is not None and (negative := controls.negative(task)) is not None:
        expected = controls.negative_reward(task) if controls.negative_reward is not None else 0.0
        outcome = await _control(task, "negative", negative, expected, sandbox, judge_connection)
    elif sandbox is not None:
        outcome = await _empty_sandbox_control(task, sandbox, judge_connection)
    else:
        outcome = await _control(task, "empty", _empty_submission(task), 0.0, None, judge_connection)
    return VerificationReport([outcome.result], verdicts=outcome.verdicts, route_backends=outcome.route_backends)


def _tier_candidates(
    machines: GradingMachines, environment: EnvironmentRequirements, memory_mb: int
) -> tuple[MachineRoute, ...]:
    """Return configured candidates, rejecting tasks no factory supports."""
    require_resolved_environment(environment)
    routes = machines.routes(environment, memory_mb)
    if not routes:
        raise UnsupportedMachineSpec(
            f"No tier satisfies the declared requirements: {environment.model_dump_json(exclude_none=True)}"
        )
    return routes


def _verdict(role: TierRole, route: MachineRoute, status: CheckStatus, detail: str) -> TierVerdict:
    return TierVerdict(role=role, tier=route.tier, status=status, detail=detail)


def _exhausted(name: str, verdicts: list[TierVerdict]) -> CheckResult:
    """The final verdict after the ladder ends: the top tried tier's outcome."""
    last = verdicts[-1]
    return CheckResult(check=name, status=last.status, detail=last.detail)


@dataclass(frozen=True)
class _Sandbox:
    """Where a sandbox-graded task's controls run, with the verifier machine's tier candidates."""

    machines: GradingMachines
    memory_mb: int
    environment: EnvironmentRequirements
    """The grader environment every candidate serves."""
    grader_routes: tuple[MachineRoute, ...]


@dataclass(frozen=True)
class _OracleAttempt:
    """One task factory's submission and execution verdict."""

    attempt: GradingAttempt | None
    verdict: TierVerdict


@dataclass(frozen=True)
class _ControlOutcome:
    result: CheckResult
    verdicts: tuple[TierVerdict, ...]
    route_backends: tuple[Backend, ...]


async def _oracle_on_route(
    task: TaskSpec,
    command: OracleCommand,
    environment: EnvironmentRequirements,
    route: MachineRoute,
) -> _OracleAttempt:
    """Run the oracle on one task tier, recording what the tier did with it."""
    try:
        validate_machine_spec(environment, route.factory, route.spec)
        attempt = await _oracle_attempt(task, command, route.factory, route.spec, environment)
    except UnsupportedMachineSpec as error:
        return _OracleAttempt(None, _verdict(TierRole.TASK, route, CheckStatus.UNSUPPORTED, str(error)))
    except OracleFailed as error:
        return _OracleAttempt(None, _verdict(TierRole.TASK, route, CheckStatus.FAIL, str(error)))
    except (RuntimeError, OSError) as error:
        return _OracleAttempt(None, _verdict(TierRole.TASK, route, CheckStatus.INFRA_ERROR, str(error)))
    return _OracleAttempt(attempt, _verdict(TierRole.TASK, route, CheckStatus.SKIPPED, "Oracle produced a submission"))


async def _verifier_ladder(
    name: str,
    expected: float,
    environment: EnvironmentRequirements,
    routes: tuple[MachineRoute, ...],
    grade: Callable[[MachineRoute], Awaitable[GradeResult]],
) -> tuple[CheckResult, list[TierVerdict]]:
    """Grade across the verifier tier ladder, failing up on control failure.

    A pass ends the ladder; an infrastructure error stops it for the suite retry.
    Every other rejecting outcome — including a refusal the backend raises only at
    acquisition — tries the next tier, and the top tier's outcome is the final
    verdict.
    """
    verdicts: list[TierVerdict] = []
    for route in routes:
        try:
            validate_machine_spec(environment, route.factory, route.spec)
            result = control_result(await grade(route), name, expected)
        except UnsupportedMachineSpec as error:
            verdicts.append(_verdict(TierRole.VERIFIER, route, CheckStatus.UNSUPPORTED, str(error)))
            continue
        verdicts.append(_verdict(TierRole.VERIFIER, route, result.status, result.detail))
        if result.status in {CheckStatus.PASS, CheckStatus.INFRA_ERROR}:
            # An infrastructure error says nothing about the tier; the suite retry reruns the ladder.
            return result, verdicts
    return _exhausted(name, verdicts), verdicts


def _oracle_environment(task: TaskSpec) -> EnvironmentRequirements:
    """Use the agent's requirements, falling back to the grader only for an empty agent environment."""
    if task.environment_requirements != EnvironmentRequirements():
        return task.environment_requirements
    environment = grader_environment(task.grader)
    if environment is None:
        raise ValueError("An oracle requires a grader environment")
    return environment


def _submission_grader(
    task: TaskSpec, attempt: GradingAttempt, judge_connection: JudgeConnection | None
) -> Callable[[MachineRoute], Awaitable[GradeResult]]:
    def grade(route: MachineRoute) -> Awaitable[GradeResult]:
        return sandbox_grade(task, attempt, route.factory, route.spec, judge_connection=judge_connection)

    return grade


def _empty_submission_grader(
    task: TaskSpec, judge_connection: JudgeConnection | None
) -> Callable[[MachineRoute], Awaitable[GradeResult]]:
    async def grade(route: MachineRoute) -> GradeResult:
        try:
            return await grade_empty_in_sandbox(task, route.factory, route.spec, judge_connection=judge_connection)
        except (RuntimeError, OSError) as error:
            return GradeResult(Outcome.INFRA_ERROR, None, str(error))

    return grade


async def _oracle_control(
    task: TaskSpec,
    name: str,
    command: OracleCommand,
    expected: float,
    sandbox: _Sandbox,
    judge_connection: JudgeConnection | None,
) -> _ControlOutcome:
    """Retry oracle backends until a submission passes the verifier ladder."""
    environment = _oracle_environment(task)
    task_routes = _tier_candidates(sandbox.machines, environment, sandbox.memory_mb)
    verdicts: list[TierVerdict] = []
    for index, route in enumerate(task_routes):
        oracle = await _oracle_on_route(task, command, environment, route)
        verdicts.append(oracle.verdict)
        if oracle.verdict.status is CheckStatus.INFRA_ERROR:
            return _ControlOutcome(
                CheckResult(check=name, status=oracle.verdict.status, detail=oracle.verdict.detail),
                tuple(verdicts),
                (),
            )
        if oracle.attempt is None:
            continue
        result, verifier_verdicts = await _verifier_ladder(
            name,
            expected,
            sandbox.environment,
            sandbox.grader_routes,
            _submission_grader(task, oracle.attempt, judge_connection),
        )
        verdicts.extend(verifier_verdicts)
        if result.status is CheckStatus.PASS:
            passed = _verdict(TierRole.TASK, route, CheckStatus.PASS, "The oracle's submission passed the control")
            remaining = tuple(candidate.tier for candidate in task_routes[index:])
            return _ControlOutcome(result, (*verdicts, passed), remaining)
        if result.status is CheckStatus.INFRA_ERROR:
            return _ControlOutcome(result, tuple(verdicts), ())
        verdicts.append(_verdict(TierRole.TASK, route, CheckStatus.FAIL, "Grading failed on every verifier tier"))
    return _ControlOutcome(_exhausted(name, verdicts), tuple(verdicts), ())


async def _control(
    task: TaskSpec,
    name: str,
    submission: ControlSubmission,
    expected: float,
    sandbox: _Sandbox | None,
    judge_connection: JudgeConnection | None,
) -> _ControlOutcome:
    """Normalize and grade one control submission."""
    if isinstance(submission, OracleCommand):
        if sandbox is None:
            raise ValueError("An oracle command requires a sandbox grader")
        return await _oracle_control(task, name, submission, expected, sandbox, judge_connection)
    if isinstance(submission, Reply):
        attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, submission.event)))
    else:
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, FILE_SUBMISSION_MESSAGE)), dict(submission.files)
        )
    if sandbox is None:
        grade = grade_task(task, attempt, judge_connection=judge_connection)
        if (
            grades_with_judge(task.grader)
            and grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
            and "model" not in (grade.detail or {})
        ):
            return _ControlOutcome(
                CheckResult(
                    check=name,
                    status=CheckStatus.FAIL,
                    detail=f"Judge control was decided by a local gate without a remote verdict: {grade.detail}",
                ),
                (),
                (),
            )
        return _ControlOutcome(control_result(grade, name, expected), (), ())
    result, verdicts = await _verifier_ladder(
        name, expected, sandbox.environment, sandbox.grader_routes, _submission_grader(task, attempt, judge_connection)
    )
    return _ControlOutcome(result, tuple(verdicts), ())


async def _empty_sandbox_control(
    task: TaskSpec,
    sandbox: _Sandbox,
    judge_connection: JudgeConnection | None,
) -> _ControlOutcome:
    """Run the grader on an empty answer across the verifier ladder; it must run cleanly on one tier.

    A zero reward or a rejected submission passes; a positive reward marks the task defective.
    """
    result, verdicts = await _verifier_ladder(
        "empty", 0.0, sandbox.environment, sandbox.grader_routes, _empty_submission_grader(task, judge_connection)
    )
    return _ControlOutcome(result, tuple(verdicts), ())


async def _oracle_attempt(
    task: TaskSpec,
    command: OracleCommand,
    factory: MachineFactory,
    spec: MachineSpec,
    environment: EnvironmentRequirements,
) -> GradingAttempt:
    """Run the oracle in a fresh machine from ``factory`` with the worker and oracle files mounted."""
    async with asyncio.timeout(spec.startup_timeout):
        prepared = await asyncio.to_thread(
            prepare_machine_spec,
            environment,
            factory,
            spec,
            dict(os.environ),
        )
        machine = await factory.create(prepared)
    workspace = prepared.workdir
    try:
        await upload_resources(
            machine,
            (*task.resources.all, *task.resources.worker, *task.resources.oracle),
            ORACLE_TIMEOUT,
            user=setup_user(factory.backend),
        )
        commands = [Command(("mkdir", "-p", workspace), timeout=ORACLE_TIMEOUT)]
        commands.extend(
            Command(("sh", "-c", setup), cwd=workspace, timeout=ORACLE_TIMEOUT, user=setup_user(factory.backend))
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
                machine, task.output_paths, ORACLE_TIMEOUT, ORACLE_OUTPUT_LIMIT_BYTES
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
