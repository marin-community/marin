"""A workspace timeout retains the final verifier verdict and exact model tokens."""

from types import SimpleNamespace

import pytest
from rolloutengine.models import (
    ConversationRecord, GradedRolloutInterrupted, ModelTurn, Rollout, RolloutOperation, RolloutStep, Transition,
)
from skyrl_gym.verification import VerificationStatus
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.config.utils import get_default_config
from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import _failed_rollout
from skyrl_train.trajectory_runners.types import TaskRequestBatch
from skyrl_train.utils.harbor_errors import ErrorHandlingConfig
from tests.cpu.task_specs import lowered_task, task_input, text_task


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
