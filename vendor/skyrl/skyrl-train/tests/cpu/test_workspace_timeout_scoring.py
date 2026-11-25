"""A workspace timeout retains the final verifier verdict and exact model tokens."""

from dataclasses import replace
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from rolloutengine.models import (
    ConversationRecord, GradedRolloutInterrupted, ModelTurn, Prompt, Rollout, RolloutOperation, RolloutStep, Transition,
)
from skyrl_gym.verification import VerificationStatus
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.config.utils import get_default_config
from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, _failed_rollout
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness
from skyrl_train.rollouts.agent_backend import AgentGenerationStopped
from skyrl_train.trajectory_runners.types import TaskRequestBatch
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig, inherited_harbor_error_handling
from tests.cpu.task_specs import lowered_task, task_input, text_task
from omegaconf import OmegaConf


def test_model_server_error_retains_failure_instead_of_masking_it_with_attribute_error():
    error = ModelServerError(category="server_error", request_id="request", status_code=400)
    interruption = GradedRolloutInterrupted(
        Rollout("attempt", (), "error", {}, {}), RolloutOperation.MODEL,
        GradeResult(Outcome.INFRA_ERROR, None, "Model request failed"),
    )
    interruption.__cause__ = error
    result = _failed_rollout(interruption, lowered_task(text_task(), "agent"))
    assert result.failure is not None
    assert result.failure.diagnostics["error_category"] == "server_error"
    assert result.failure.diagnostics["message"] == str(error)
    assert result.failure.diagnostics["status_code"] == 400
    assert result.grade.reward is None


@pytest.mark.parametrize("reward", [0.0, 1.0, None])
def test_timeout_verdict_survives_worker_projection_and_evaluation(reward):
    turn = ModelTurn(
        message={"role": "assistant", "content": "Inspecting."}, prompt_token_ids=(1,),
        response_token_ids=(2, 3), logprobs=(-0.2, -0.3), stop_reason="tool_calls",
    )
    messages = ({"role": "user", "content": "Fix the workspace."}, turn.message)
    step = RolloutStep(turn, Transition(done=True, metrics={"advance_incomplete": 1.0}), 1, messages)
    conversation = ConversationRecord(messages, (1,), (2, 3), (1, 1), (-0.2, -0.3), (step,))
    grade = GradeResult(Outcome.GRADED, reward, passed=reward > 0) if reward is not None else GradeResult(
        Outcome.INFRA_ERROR, None, "Verifier unavailable",
    )
    interruption = GradedRolloutInterrupted(
        Rollout("attempt", (conversation,), "tool_calls", {}, {}), RolloutOperation.ADVANCE, grade,
    )
    interruption.__cause__ = TimeoutError("Agent budget exhausted")
    spec = lowered_task(text_task(), "agent")
    rollout = _failed_rollout(interruption, spec)
    config = get_default_config().generator
    projection = WholeTaskProjection(
        config, SimpleNamespace(eos_token_id=0),
        error_handling=ErrorHandlingConfig(enable_error_classification=True, default_error_treatment=ErrorTreatment.MASK),
    )
    batch = projection.project([rollout], TaskRequestBatch((task_input(spec),)))
    metrics = _calculate_eval_metrics(batch, ["task"], ["SWE"], 1)
    if reward is None:
        assert batch["exclude_from_baseline"] == [True]
        assert metrics["eval/SWE/num_scored"] == 0
        assert "eval/SWE/avg_score" not in metrics
    else:
        assert batch["response_ids"] == [[2, 3]]
        assert batch["loss_masks"] == [[1, 1]]
        assert batch["exclude_from_baseline"] == [False]
        assert batch["verification_results"][0].status is VerificationStatus.VERIFIED
        assert metrics["eval/SWE/num_scored"] == 1
        assert metrics["eval/SWE/avg_score"] == reward
        assert metrics["eval/SWE/avg_verifier_score"] == reward
        assert metrics["eval/SWE/pass_at_1"] == reward


@pytest.mark.asyncio
@pytest.mark.parametrize("reward,model_failure", [(0.0, False), (1.0, False), (None, False), (1.0, True)])
async def test_completed_harbor_trial_without_model_turn_keeps_actual_verdict(reward, model_failure):
    config = get_default_config().generator
    config.sampling_params.logprobs = 0
    projection = WholeTaskProjection(config, SimpleNamespace(eos_token_id=0))
    task = text_task().model_copy(update={"tags": ("harbor",)})
    spec = lowered_task(task, "harbor_trial")

    async def completed_trial(complete):
        # External trial boundary: the agent exhausted its budget, but Harbor
        # still ran the workspace verifier without a completed model response.
        if model_failure:
            with pytest.raises(AgentGenerationStopped):
                await complete(Prompt(({"role": "user", "content": "Inspect the workspace."},), {}))
        return TrialResult(
            task_name="workspace", trial_name="timeout", trial_uri="/tmp/timeout",
            task_id=TaskConfig(path=Path("/tmp/task")).get_task_id(), task_checksum="fixture",
            agent_info=AgentInfo(name="pi", version="0.87.0"),
            config=TrialConfig(task=TaskConfig(path=Path("/tmp/task")), agent=AgentConfig(name="pi")),
            verifier_result=None if reward is None else VerifierResult(rewards={"reward": reward}),
            exception_info=ExceptionInfo(
                exception_type="AgentTimeoutError", exception_message="Agent budget exhausted",
                exception_traceback="", occurred_at=datetime.now(timezone.utc),
            ),
        )

    async def shutdown():
        pass

    async def failed_generation(request):
        raise ModelServerError(category="server_error", request_id="request", status_code=503)

    worker = TaskRolloutWorker(
        config, projection, SimpleNamespace(generate=failed_generation), {}, shutdown_timeout=5,
        harbor=HarborTaskSettings.from_config(OmegaConf.create({})),
        installed_agents=SimpleNamespace(
            original_harness=lambda *_: HarborTrialHarness(completed_trial, "reward"), shutdown=shutdown,
        ),
    )
    request = TaskRequestBatch((task_input(spec),))
    try:
        rollouts = await worker.generate(request)
        batch = projection.project(rollouts, request)
        metrics = _calculate_eval_metrics(batch, ["task"], ["SWE"], 1)
        assert rollouts[0].response_token_ids == ()
        assert batch["loss_masks"] == [[0]]
        if model_failure:
            assert rollouts[0].failure.exception_type == "ModelServerError"
            assert metrics["eval/SWE/num_scored"] == 0
        elif reward is None:
            assert metrics["eval/SWE/num_scored"] == 0
            assert batch["verification_results"][0].score is None
        else:
            assert metrics["eval/SWE/num_scored"] == 1
            assert metrics["eval/SWE/avg_verifier_score"] == reward
            assert metrics["eval/SWE/pass_at_1"] == reward
    finally:
        await worker.shutdown()


def test_harbor_error_policy_inherits_only_undeclared_generator_fields():
    """Harbor configs keep declared classification fields and inherit the omitted ones."""
    generator = ErrorHandlingConfig(enable_error_classification=True, default_error_treatment=ErrorTreatment.MASK)
    undeclared = HarborTaskSettings.from_config(OmegaConf.create({}))
    assert undeclared.declared_error_handling_fields == frozenset()
    assert inherited_harbor_error_handling(
        undeclared.error_handling, undeclared.declared_error_handling_fields, generator
    ) == replace(
        undeclared.error_handling,
        enable_error_classification=True,
        default_error_treatment=ErrorTreatment.MASK,
    )
    declared = HarborTaskSettings.from_config(
        OmegaConf.create({"harbor": {"enable_error_classification": False, "default_error_treatment": "zero"}})
    )
    assert declared.declared_error_handling_fields == frozenset(
        {"enable_error_classification", "default_error_treatment"}
    )
    assert inherited_harbor_error_handling(
        declared.error_handling, declared.declared_error_handling_fields, generator
    ) == declared.error_handling


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", ["ContextLengthExceededError", "ValueError", "AgentTimeoutError"])
async def test_harbor_rows_score_under_the_inherited_generator_policy(exception_type):
    """A harbor-tagged CLE row keeps its scored verdict; unnamed errors stay masked."""
    config = get_default_config().generator
    config.sampling_params.logprobs = 0
    projection = WholeTaskProjection(config, SimpleNamespace(eos_token_id=0))
    task = text_task().model_copy(update={"tags": ("harbor",)})
    spec = lowered_task(task, "harbor_trial")

    async def completed_trial(complete):
        # External trial boundary: Harbor classified the failure but still ran
        # the workspace verifier, so the verdict is available either way.
        return TrialResult(
            task_name="workspace", trial_name="policy", trial_uri="/tmp/policy",
            task_id=TaskConfig(path=Path("/tmp/task")).get_task_id(), task_checksum="fixture",
            agent_info=AgentInfo(name="pi", version="0.87.0"),
            config=TrialConfig(task=TaskConfig(path=Path("/tmp/task")), agent=AgentConfig(name="pi")),
            verifier_result=VerifierResult(rewards={"reward": 1.0}),
            exception_info=ExceptionInfo(
                exception_type=exception_type, exception_message="Terminal agent failure",
                exception_traceback="", occurred_at=datetime.now(timezone.utc),
            ),
        )

    async def shutdown():
        pass

    async def failed_generation(request):
        raise ModelServerError(category="server_error", request_id="request", status_code=503)

    worker = TaskRolloutWorker(
        config, projection, SimpleNamespace(generate=failed_generation), {}, shutdown_timeout=5,
        harbor=HarborTaskSettings.from_config(OmegaConf.create({})),
        installed_agents=SimpleNamespace(
            original_harness=lambda *_: HarborTrialHarness(completed_trial, "reward"), shutdown=shutdown,
        ),
    )
    request = TaskRequestBatch((task_input(spec),))
    try:
        rollouts = await worker.generate(request)
        batch = worker.projection.project(rollouts, request)
        metrics = _calculate_eval_metrics(batch, ["task"], ["SWE"], 1)
        assert rollouts[0].failure.exception_type == exception_type
        if exception_type == "ValueError":
            # Unnamed errors keep the generator's masking default.
            assert batch["exclude_from_baseline"] == [True]
            assert metrics["eval/SWE/num_scored"] == 0
        else:
            # The verified verdict survives the failure and stays scored.
            assert batch["exclude_from_baseline"] == [False]
            assert batch["verification_results"][0].status is VerificationStatus.VERIFIED
            assert metrics["eval/SWE/num_scored"] == 1
            assert metrics["eval/SWE/avg_verifier_score"] == 1.0
            # Agent timeouts pass the optimization reward through; CLE zeroes it
            # while remaining in the baseline with its verdict.
            assert metrics["eval/SWE/avg_score"] == (1.0 if exception_type == "AgentTimeoutError" else 0.0)
    finally:
        await worker.shutdown()
