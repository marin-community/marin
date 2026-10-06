# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Failures produced by a real ShellboxRolloutEngine run map to the right Cause."""

import pytest
from rolloutengine.contracts import AGENT_TIMEOUT_STOP_REASON, GenerationLimitReached, ModelRequest, ModelTurn
from rolloutengine.engine import ShellboxRolloutEngine
from shellbox.machine import UnsupportedMachineSpec
from taskcompendium.environment import EnvironmentKind, HealthcheckSpec, StdoutReward
from taskcompendium.execution import TaskExecution
from taskcompendium.grading import skipped_verifier
from taskcompendium.grading_result import Outcome
from taskcompendium.submission import PlainText

from taskforge.llm.client import GlmRequestRejected, GlmUnavailable
from taskforge.spec.draft import file, shell_command, shell_verifier
from taskforge.validate.classify import trial_outcome
from taskforge.validate.outcome import Cause, Graded, Ungraded


def engine(model, factory=None) -> ShellboxRolloutEngine:
    return ShellboxRolloutEngine(
        model,
        {} if factory is None else {EnvironmentKind.SHELLSIM: factory},
        max_turns=4,
        command_timeout=10,
        cleanup_timeout=10,
        convention=PlainText(id="plain"),
    )


async def outcome_of(task, model, factory=None, execution=TaskExecution()):
    try:
        result = await engine(model, factory).run(task, execution=execution)
    except Exception as error:
        return trial_outcome(error)
    return trial_outcome(result)


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
    task = file_task.model_copy(
        update={"environment": file_task.environment.model_copy(update={"startup_timeout": 0.01})}
    )
    outcome = await outcome_of(
        task, fakes.script_model([]), fakes.flaky_factory(failures=0, error=RuntimeError, delay=1)
    )

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.MACHINE_START_TIMEOUT


@pytest.mark.parametrize(
    "error,cause",
    [
        (lambda: GlmUnavailable("attempts exhausted", ()), Cause.MODEL_UNAVAILABLE),
        (lambda: GlmRequestRejected(422, "bad tool schema"), Cause.MODEL_REJECTED),
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


async def test_changed_token_prefix_is_a_contract_failure(file_task, fakes):
    async def model(request: ModelRequest) -> ModelTurn:
        message = fakes.shell("ls") if not request.prefix_token_ids else fakes.text("done")
        return ModelTurn(message, (99,), (1,), None, "stop")

    outcome = await outcome_of(file_task, model, fakes.flaky_factory(failures=0, error=RuntimeError))

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.TOKEN_CONTRACT


@pytest.mark.parametrize(
    "setup,healthcheck",
    [
        ((shell_command("exit 3", timeout=10),), None),
        (
            (),
            HealthcheckSpec(
                command=shell_command("exit 1", 10), interval=0, start_period=0, start_interval=0, retries=2
            ),
        ),
    ],
)
async def test_failing_task_setup_is_a_non_retryable_task_defect(file_task, fakes, setup, healthcheck):
    environment = file_task.environment.model_copy(update={"setup": setup, "healthcheck": healthcheck})
    task = file_task.model_copy(update={"environment": environment})

    outcome = await outcome_of(task, fakes.script_model([]), fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.TASK_SETUP, False)


async def test_agent_timeout_before_the_first_response_is_an_ungraded_agent_timeout(math_task, fakes):
    outcome = await outcome_of(
        math_task, fakes.script_model([fakes.text("395")], hang_from=0), execution=TaskExecution(agent_timeout=0.01)
    )

    assert isinstance(outcome, Ungraded)
    assert (outcome.cause, outcome.retryable) == (Cause.AGENT_TIMEOUT, False)
    assert outcome.rollout is not None and outcome.rollout.stop_reason == AGENT_TIMEOUT_STOP_REASON


async def test_agent_timeout_after_work_keeps_the_engine_grade_and_records_the_timeout(file_task, fakes):
    model = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")], hang_from=1)

    outcome = await outcome_of(
        file_task, model, fakes.flaky_factory(0, RuntimeError), execution=TaskExecution(agent_timeout=1)
    )

    assert isinstance(outcome, Graded) and outcome.timed_out
    assert (outcome.reward, outcome.rollout.stop_reason) == (1.0, AGENT_TIMEOUT_STOP_REASON)


async def test_attempt_timeout(math_task, fakes):
    outcome = await outcome_of(
        math_task, fakes.script_model([fakes.text("395")], hang_from=0), execution=TaskExecution(attempt_timeout=0.01)
    )

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (Cause.ATTEMPT_TIMEOUT, True)


@pytest.mark.parametrize(
    "script,cause",
    [
        ("echo not-a-number\n", Cause.GRADER_INVALID_REWARD),
        ("exit 3\n", Cause.GRADER_EXECUTION),
    ],
)
async def test_grader_failures_keep_the_rollout(file_task, fakes, script, cause):
    task = file_task.model_copy(
        update={
            "verifier": shell_verifier(
                ("sh", "/grader/check.sh"), StdoutReward(), 10, files=(file("/grader/check.sh", script),)
            )
        }
    )
    outcome = await outcome_of(task, fakes.script_model([fakes.text("done")]), fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and (outcome.cause, outcome.retryable) == (cause, False)
    assert outcome.rollout is not None and outcome.rollout.grade.status == Outcome.INFRA_ERROR


async def test_skipped_verifier(math_task, fakes):
    task = math_task.model_copy(update={"verifier": skipped_verifier("no reference answer")})
    outcome = await outcome_of(task, fakes.script_model([fakes.text("395")]))

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.VERIFIER_SKIPPED


@pytest.mark.parametrize("answer,status,reward", [("395", Outcome.GRADED, 1.0), ("", Outcome.SUBMISSION_FAILURE, 0.0)])
async def test_judged_submissions_are_graded(math_task, fakes, answer, status, reward):
    outcome = await outcome_of(math_task, fakes.script_model([fakes.text(answer)]))

    assert isinstance(outcome, Graded)
    assert (outcome.grade.status, outcome.reward) == (status, reward)


async def test_model_failure_after_a_turn_is_ungraded_even_though_the_engine_graded(file_task, fakes):
    script = fakes.script_model([fakes.shell("echo 60 > /workspace/sum.txt")])

    async def model(request: ModelRequest) -> ModelTurn:
        if request.prefix_token_ids:
            raise GlmUnavailable("pool drained", ())
        return await script(request)

    outcome = await outcome_of(file_task, model, fakes.flaky_factory(0, RuntimeError))

    assert isinstance(outcome, Ungraded) and outcome.cause is Cause.MODEL_UNAVAILABLE
    assert outcome.rollout is not None and outcome.rollout.grade.reward == 1.0
