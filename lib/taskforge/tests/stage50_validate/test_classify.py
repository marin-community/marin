# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Failures produced by a real RolloutEngine run map to the right Cause."""


import asyncio
from dataclasses import dataclass

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.shell import task_prompt
from rolloutengine.models import (
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GenerationLimitReached,
    GradedRollout,
    GradedRolloutInterrupted,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutLimits,
    RolloutOperation,
)
from shellbox.machine import Backend, UnsupportedMachineSpec
from taskcompendium.grader import GraderPackage
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import NoGrader, TaskSpec
from verifyit.spec import ExactSpec, PytestSpec

from taskforge.llm.client import ModelRequestRejected, ModelUnavailable
from taskforge.spec.draft import answer_grader, grader_environment, rollout_engine
from taskforge.stage50_validate.classify import candidate_code_error, classify, trial_outcome
from taskforge.stage50_validate.outcome import RETRYABLE, Cause, Graded, Ungraded
from tests.sandbox.fixture_images import SHELL_GRADER_ARGV, FixtureImageFactory

INVALID_REWARD_GRADER = """printf '%s' '{"reward": "nan?"}' > /logs/verifier/reward.json\n"""
FAILING_GRADER = "exit 3\n"
"""Exits before writing its verdict; a verdict-file grade reads the missing file, not the exit code."""
HANGING_GRADER = "# never finishes on HangingGraderFactory\n"
TASK_IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
COMMAND_TIMEOUT = 10
DEADLINE_SWALLOWING = "deadline-swallowing"


@dataclass(frozen=True)
class HangingGraderFactory:
    """Fixture image machines on which the grader program never finishes."""

    backend: Backend = Backend.DOCKER

    def check_environment(self, requirements, spec):
        FixtureImageFactory().check_environment(requirements, spec)

    async def create(self, spec):
        machine = await FixtureImageFactory().create(spec)

        class Hanging:
            async def run(self, command):
                if command.argv == SHELL_GRADER_ARGV:
                    await asyncio.Future()
                return await machine.run(command)

            async def upload(self, source, target):
                await machine.upload(source, target)

            async def download(self, source, target):
                await machine.download(source, target)

            async def close(self):
                await machine.close()

        return Hanging()


@dataclass(frozen=True)
class DeadlineSwallowingHarness:
    """Catches the cancellation of the attempt deadline during its first request and ends after the deadline."""

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        pass

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        try:
            await session.start(task_prompt(context.task, None, ()))
        except asyncio.CancelledError:
            pass


def engine(factory=None, verifier_factory=None) -> RolloutEngine:
    """Task machines from ``factory``; verifier machines from ``verifier_factory`` or the fixture image factory."""
    factories = {Backend.DOCKER.value: verifier_factory or FixtureImageFactory()}
    if factory is not None:
        factories[Backend.SHELLSIM.value] = factory
    shell = rollout_engine(factories, command_timeout=COMMAND_TIMEOUT)
    return RolloutEngine(factories, {**shell.harnesses, DEADLINE_SWALLOWING: DeadlineSwallowingHarness()})


def with_limits(lowered, **limits):
    return lowered.model_copy(update={"limits": lowered.limits.model_copy(update=limits)})


async def outcome_of(lowered, model, factory=None, verifier_factory=None):
    try:
        result = await engine(factory, verifier_factory).run_task(lowered, model)
    except Exception as error:
        return trial_outcome(error, lowered.task)
    return trial_outcome(result, lowered.task)


@pytest.mark.parametrize(
    "error,cause,retryable",
    [
        (lambda: RuntimeError("sandbox broker refused"), Cause.MACHINE_START, True),
        (lambda: UnsupportedMachineSpec("no gpus"), Cause.MACHINE_UNSUPPORTED, False),
    ],
)
async def test_machine_start_failures(file_task, fakes, error, cause, retryable):
    outcome = await outcome_of(file_task, fakes.script_model([]), fakes.flaky_factory(failures=1, error=error))

    assert isinstance(outcome, Ungraded)
    assert (outcome.cause, outcome.retryable) == (cause, retryable)


async def test_machine_start_past_startup_timeout_is_a_start_timeout(file_task, fakes):
    assert file_task.task_machine is not None
    task = file_task.model_copy(
        update={"task_machine": file_task.task_machine.model_copy(update={"startup_timeout": 0.01})}
    )
    outcome = await outcome_of(
        task, fakes.script_model([]), fakes.flaky_factory(failures=0, error=RuntimeError, delay=1)
    )

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.MACHINE_START_TIMEOUT


@pytest.mark.parametrize(
    "error,cause",
    [
        (lambda: ModelUnavailable("attempts exhausted", ()), Cause.MODEL_UNAVAILABLE),
        (lambda: ModelRequestRejected(422, "bad tool schema", ()), Cause.MODEL_REJECTED),
        (lambda: KeyError("choices"), Cause.UNCLASSIFIED),
    ],
)
async def test_model_errors(math_task, fakes, error, cause):
    outcome = await outcome_of(math_task, fakes.raising_model(error))

    assert isinstance(outcome, Ungraded) and outcome.cause is cause
    assert "Traceback" in outcome.detail


async def test_first_prompt_over_the_limit_is_a_generation_limit(math_task, fakes):
    outcome = await outcome_of(math_task, fakes.raising_model(lambda: GenerationLimitReached(())))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.GENERATION_LIMIT, False)


async def test_failing_task_setup_is_a_non_retryable_task_defect(file_task_with, fakes):
    task = file_task_with(setup=("exit 3",))

    outcome = await outcome_of(task, fakes.script_model([]), fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.TASK_SETUP, False)


async def test_turn_deadline_before_the_first_response_is_an_ungraded_agent_timeout(math_task, fakes):
    task = with_limits(math_task, total_turn_timeout=0.01)

    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")], hang_from=0))

    assert isinstance(outcome, Ungraded)
    assert (outcome.cause, outcome.retryable) == (Cause.AGENT_TIMEOUT, False)
    assert outcome.rollout is not None and outcome.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON


async def test_turn_deadline_after_work_keeps_the_engine_grade_and_records_the_timeout(file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")], hang_from=1)

    outcome = await outcome_of(with_limits(file_task, total_turn_timeout=1), model, fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Graded) and outcome.timed_out
    assert (outcome.reward, outcome.rollout.stop_reason) == (1.0, TOTAL_TURN_TIMEOUT_STOP_REASON)


async def test_attempt_timeout(math_task, fakes):
    task = with_limits(math_task, attempt_timeout=0.01)

    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")], hang_from=0))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.ATTEMPT_TIMEOUT, True)


async def test_a_harness_that_ends_after_the_attempt_deadline_is_an_ungraded_attempt_timeout(math_task, fakes):
    # The harness catches the deadline's cancellation and returns; RolloutEngine fails the attempt as
    # ``attempt`` with a TimeoutError and does not grade it.
    task = with_limits(math_task, attempt_timeout=0.01).model_copy(update={"harness": DEADLINE_SWALLOWING})

    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")], hang_from=0))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.ATTEMPT_TIMEOUT, True)
    assert "Attempt deadline expired" in outcome.detail
    assert outcome.grade is not None and outcome.grade.status is Outcome.UNAVAILABLE


async def test_a_model_call_past_its_deadline_is_a_retryable_model_timeout(math_task, fakes):
    task = with_limits(math_task, model_turn_timeout=0.01)

    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")], hang_from=0))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.MODEL_TIMEOUT, True)


async def test_a_machine_terminated_under_a_tool_turn_is_retryable(file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt"), fakes.text("Done.")])

    outcome = await outcome_of(file_task, model, fakes.faulty_factory(terminated=True))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.MACHINE_TERMINATED, True)


@pytest.mark.parametrize(
    "script,cause",
    [
        (INVALID_REWARD_GRADER, Cause.GRADER_INVALID_REWARD),
        (FAILING_GRADER, Cause.GRADER_MISSING_REWARD),
    ],
)
async def test_script_grader_failures_are_infrastructure_grades_that_keep_the_rollout(
    file_task_with, fakes, script, cause
):
    model = fakes.script_model([fakes.text("done")])

    outcome = await outcome_of(file_task_with(grader_script=script), model, fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (cause, False)
    assert outcome.rollout is not None and outcome.grade is not None and outcome.grade.status == Outcome.INFRA_ERROR


async def test_a_grader_past_the_verifier_deadline_is_a_retryable_grader_timeout(file_task_with, fakes):
    # The grader's own limit (1 s) outlasts the engine's verifier deadline (0.5 s), so the engine's expires first.
    task = with_limits(file_task_with(grader_script=HANGING_GRADER, grader_timeout=1), verifier_timeout=0.5)

    model = fakes.script_model([fakes.text("done")])

    outcome = await outcome_of(task, model, fakes.flaky_factory(0, RuntimeError), HangingGraderFactory())

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.GRADER_TIMEOUT, True)


async def test_a_task_without_a_grader_is_verifier_skipped(math_task, relower, fakes):
    task = relower(math_task.task.model_copy(update={"grader": NoGrader(reason="no reference answer")}))
    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")]))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.VERIFIER_SKIPPED, False)
    assert outcome.rollout is not None and outcome.grade is not None and outcome.grade.status == Outcome.UNAVAILABLE


@pytest.mark.parametrize(
    "grader,detail,unloaded",
    [
        ("pytest", {"reason": "collection_error", "category": "agent"}, True),
        ("pytest", {"reason": "startup_error", "category": "agent"}, True),
        ("pytest", {"reason": "tests", "passed": 0, "total": 3}, False),
        ("exact", {"reason": "collection_error"}, False),
    ],
)
def test_candidate_code_error_is_a_graded_failed_attempt(math_task, grader, detail, unloaded):
    package = (
        answer_grader(PytestSpec(paths=("tests/test_x.py",)), environment=grader_environment(TASK_IMAGE))
        if grader == "pytest"
        else answer_grader(ExactSpec(expected=("395",)))
    )
    task = math_task.task.model_copy(update={"grader": GraderPackage(package.grader).grader})
    grade = GradeResult(Outcome.GRADED, 0.0, detail=detail)

    outcome = trial_outcome(GradedRollout(Rollout("run", (), "stop"), grade), task)

    assert isinstance(outcome, Graded) and outcome.reward == 0.0
    assert outcome.candidate_code_error is unloaded
    assert not candidate_code_error(GradeResult(Outcome.GRADED, 1.0, detail=detail), task)


def test_a_reward_outside_the_graders_score_range_is_an_ungraded_grader_defect(math_task):
    """A grader scoring 0 to 100 is not a solve at 100; the trial carries the defect as its cause."""
    task = math_task.task
    result = GradedRollout(Rollout("run", (), "stop"), GradeResult(Outcome.GRADED, 100.0))

    outcome = trial_outcome(result, task)

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.GRADER_REWARD_RANGE
    assert outcome.detail == "reward 100.0 is outside the grader's score range [0.0, 1.0]"
    assert not outcome.retryable and (outcome.rollout, outcome.grade) == (result.rollout, result.grade)
    solved = GradedRollout(Rollout("run", (), "stop"), GradeResult(Outcome.GRADED, 1.0))
    assert isinstance(trial_outcome(solved, task), Graded)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Output capture exceeds its file or byte budget"),
        ValueError("Downloaded files fall outside the declared output paths"),
        RuntimeError("Download contains an invalid or oversized file: /workspace/out/big.bin"),
    ],
    ids=["budget", "outside-declared-paths", "oversized-file"],
)
def test_declared_outputs_the_engine_cannot_capture_count_against_the_task(math_task, error):
    """The agent wrote the files, so the capture failure is the task's, not a retryable grader fault."""
    task = math_task.task
    rollout, grade = Rollout("run", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None)
    interrupted = GradedRolloutInterrupted(rollout, RolloutOperation.GRADE, grade)
    interrupted.__cause__ = error

    outcome = trial_outcome(interrupted, task)

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.OUTPUT_CAPTURE
    assert not outcome.retryable and outcome.rollout is rollout and outcome.grade is grade
    raised = GradedRolloutInterrupted(rollout, RolloutOperation.GRADE, grade)
    raised.__cause__ = RuntimeError("verifier exploded")
    fallback = trial_outcome(raised, task)
    assert isinstance(fallback, Ungraded) and fallback.cause is Cause.GRADER_RAISED


@pytest.mark.parametrize("answer,status,reward", [("395", Outcome.GRADED, 1.0), ("", Outcome.SUBMISSION_FAILURE, 0.0)])
async def test_judged_submissions_are_graded(math_task, fakes, answer, status, reward):
    outcome = await outcome_of(math_task, fakes.script_model([fakes.text(answer)]))

    assert isinstance(outcome, Graded)
    assert (outcome.grade.status, outcome.reward) == (status, reward)


async def test_model_failure_after_a_turn_is_ungraded_even_though_the_engine_graded(file_task, fakes):
    script = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")])

    async def model(request: ModelRequest) -> ModelTurn:
        if request.messages[-1]["role"] == "tool":
            raise ModelUnavailable("pool drained", ())
        return await script(request)

    outcome = await outcome_of(file_task, model, fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.MODEL_UNAVAILABLE
    assert outcome.rollout is not None and outcome.grade is not None and outcome.grade.reward == 1.0


@pytest.mark.parametrize(
    ("error", "direct_cause", "wrapped_cause"),
    [
        (
            ConnectError(Code.DEADLINE_EXCEEDED, "TerminateJob timed out"),
            Cause.MACHINE_TERMINATED,
            Cause.MACHINE_TERMINATED,
        ),
        (
            ConnectError(Code.UNAVAILABLE, "controller connection refused"),
            Cause.MACHINE_TERMINATED,
            Cause.MACHINE_TERMINATED,
        ),
        (RuntimeError("Machine is closed"), Cause.MACHINE_TERMINATED, Cause.MACHINE_TERMINATED),
        (RuntimeError("ShellSim terminated: cpu_exhausted"), Cause.MACHINE_TERMINATED, Cause.MACHINE_TERMINATED),
        (ConnectError(Code.PERMISSION_DENIED, "token rejected"), Cause.UNCLASSIFIED, Cause.TOOL_EXECUTION),
    ],
    ids=["rpc-deadline", "rpc-unavailable", "machine-closed", "shellsim-terminated", "permission-denied"],
)
def test_a_machine_lost_under_a_trial_is_a_retryable_termination(math_task, error, direct_cause, wrapped_cause):
    """A host lost mid-trial, by RPC or by shellbox's bare RuntimeError, is retryable; a refused credential is not.

    Wrapped in an interruption, a refused credential falls to the interrupted operation's own cause.
    """
    ungraded = GradeResult(GradeStatus.UNAVAILABLE, None, "interrupted")
    wrapped = GradedRolloutInterrupted(Rollout("run", (), "stop"), RolloutOperation.ADVANCE, ungraded)
    wrapped.__cause__ = error

    assert classify(error, math_task.task) == direct_cause
    assert classify(wrapped, math_task.task) == wrapped_cause
    assert (direct_cause in RETRYABLE) == (direct_cause is Cause.MACHINE_TERMINATED)
