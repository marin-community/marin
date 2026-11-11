import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import GradedRollout, Prompt, TaskRolloutSpec
from rolloutengine.recorder import harness_rollout
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_limits
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import NoGrader

from skyrl_train.rollouts.agent_backend import AgentGenerationStopped
from skyrl_train.rollouts.harbor_tasks import harbor_grading_failure
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness
from skyrl_train.rollouts.task_projections import rollout_loss_eligible
from skyrl_train.rollouts.training_record import training_rollout
from skyrl_train.utils.harbor_errors import DEFAULT_ERROR_HANDLING_CONFIG


@pytest.mark.asyncio
@pytest.mark.parametrize("harness_name", ["harbor", "agent"])
@pytest.mark.parametrize("compact", [False, True])
async def test_real_pi_roundtrip_preserves_all_turns_and_compaction_starts_a_conversation(harness_name, compact):
    # Captured server/Pi assistant messages from Pi 0.87.0, including its JSON
    # reserialization and absent/null/empty optional-field differences.
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/pi-0.87-roundtrip.json").read_text())
    model = ReplayModel([copy.deepcopy(fixture["server_assistant"]) for _ in range(60)])

    async def run(complete):
        messages = ({"role": "user", "content": "Inspect the workspace."},)
        for index in range(60):
            await complete(Prompt(messages, {}))
            if compact and index == 29:
                messages = ({"role": "user", "content": "Intentional compacted summary."},)
            else:
                messages = (
                    *messages,
                    copy.deepcopy(fixture["pi_assistant"]),
                    {"role": "tool", "tool_call_id": fixture["pi_assistant"]["tool_calls"][0]["id"], "content": "ok"},
                )
        return TrialResult(
            task_name="swe",
            trial_name="pi-roundtrip",
            trial_uri="/tmp/pi-roundtrip",
            task_id=TaskConfig(path=Path("/tmp/task")).get_task_id(),
            task_checksum="fixture",
            agent_info=AgentInfo(name="pi", version="0.87.0"),
            config=TrialConfig(task=TaskConfig(path=Path("/tmp/task")), agent=AgentConfig(name="pi")),
            verifier_result=VerifierResult(rewards={"reward": 1.0}),
        )

    async def agent(context, complete):
        await run(complete)

    harness = HarborTrialHarness(run, "reward") if harness_name == "harbor" else AgentHarness(agent)
    result = await harness_rollout(harness, arithmetic_task(), model.complete, limits=rollout_limits(max_turns=60))
    assert len(result.conversations) == (2 if compact else 1)
    assert [len(conversation.steps) for conversation in result.conversations] == ([30, 30] if compact else [60])
    for conversation in result.conversations:
        generated = sum(len(step.turn.response_token_ids) for step in conversation.steps)
        assert sum(conversation.loss_mask) == generated
    assert sum(sum(conversation.loss_mask) for conversation in result.conversations) == 60
    assert sum(request.continuation is None for request in model.requests) == (2 if compact else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["max_turns", "permission", "network", "verifier", "missing_verdict"])
@pytest.mark.parametrize("verifier_reward", [0.0, 1.0])
async def test_real_pi_limit_keeps_verifier_reward_and_loss_but_infrastructure_remains_masked(failure, verifier_reward):
    fixtures = Path(__file__).parents[1] / "fixtures"
    roundtrip = json.loads((fixtures / "pi-0.87-roundtrip.json").read_text())
    transcript = json.loads((fixtures / "pi-0.87-max-turns.json").read_text())
    error_message = transcript["events"][-1]["message"]["errorMessage"]
    model = ReplayModel([copy.deepcopy(roundtrip["server_assistant"]) for _ in range(128)])
    task = arithmetic_task(grader=NoGrader(reason="The Harbor workspace verifier owns this score"))
    limits = rollout_limits(max_turns=128)

    async def run(complete):
        messages = ({"role": "user", "content": "Inspect the workspace."},)
        while True:
            try:
                await complete(Prompt(messages, {}))
            except AgentGenerationStopped as error:
                assert error.reason == "max_turns"
                break
            messages = (
                *messages,
                copy.deepcopy(roundtrip["pi_assistant"]),
                {"role": "tool", "tool_call_id": roundtrip["pi_assistant"]["tool_calls"][0]["id"], "content": "ok"},
            )
        exception_type = "NonZeroAgentExitCodeError"
        message = error_message
        if failure == "permission":
            message = "Permission denied while removing a staged runtime file"
        elif failure == "network":
            message = "httpx.ReadError while contacting serving"
        elif failure == "verifier":
            exception_type, message = "VerifierRuntimeError", "Task verifier crashed"
        return TrialResult(
            task_name="swe", trial_name="pi-limit", trial_uri="/tmp/pi-limit",
            task_id=TaskConfig(path=Path("/tmp/task")).get_task_id(), task_checksum="fixture",
            agent_info=AgentInfo(name="pi", version="0.87.0"),
            config=TrialConfig(task=TaskConfig(path=Path("/tmp/task")), agent=AgentConfig(name="pi")),
            verifier_result=(
                None if failure in {"verifier", "missing_verdict"} else VerifierResult(rewards={"reward": verifier_reward})
            ),
            exception_info=ExceptionInfo(
                exception_type=exception_type, exception_message=message,
                exception_traceback="", occurred_at=datetime.now(timezone.utc),
            ),
        )

    result = await harness_rollout(HarborTrialHarness(run, "reward"), task, model.complete, limits=limits)
    spec = TaskRolloutSpec(task=task, harness="harbor_trial", task_machine=None, verifier_machine=None, limits=limits)
    rollout = harbor_grading_failure(training_rollout(GradedRollout(result, GradeResult(Outcome.GRADED, 0)), spec))
    assert len(rollout.conversations) == 1
    assert len(rollout.steps) == 128
    assert sum(rollout.loss_mask) == 128
    eligible = rollout_loss_eligible(rollout, DEFAULT_ERROR_HANDLING_CONFIG, logprobs_required=True)
    assert eligible == (failure == "max_turns")
    if failure == "max_turns":
        assert rollout.failure is None
        assert rollout.grade.reward == verifier_reward
        assert rollout.grade.diagnostics["harbor_model_limit"] == "max_turns"
    else:
        assert rollout.failure is not None
