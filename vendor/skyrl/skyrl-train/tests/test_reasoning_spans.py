# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from rolloutengine.models import ConversationRecord, ModelTurn, RolloutStep, Transition
from skyrl_gym.envs.nemotron_ultra.answer_extraction import ReasoningViolation
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.rollouts.training_record import (
    REASONING_VIOLATION_METADATA_KEY,
    REASONING_VIOLATION_TOKEN_METADATA_KEY,
    ReasoningSpanPolicy,
    ReasoningSpanSettings,
    TrainingRollout,
    TrainingStep,
    TrainingTransition,
    penalize_malformed_reasoning,
    penalize_malformed_reasoning_tokens,
    reject_malformed_reasoning,
)
from skyrl_train.trajectory_runners.model_clients import (
    REASONING_VIOLATION_TOKENS_KEY,
    REASONING_VIOLATIONS_KEY,
    DirectModelClient,
)
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.trajectory_reward_shaping import apply_reasoning_span_penalties

VOCABULARY = {1: "<|start_think|>", 2: "<|end_think|>", 3: "plan", 4: "call", 5: "<|eot_id|>"}
TOOL_CALL = {"id": "call-0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


class _Tokenizer:
    eos_token_id = 5

    def decode(self, token_ids, skip_special_tokens):
        return "".join(VOCABULARY[token] for token in token_ids)

    def encode(self, text, add_special_tokens):
        token_ids = [token for token, piece in VOCABULARY.items() if piece == text]
        return token_ids or [3, 3]


class _InferenceClient:
    """Serves one chat response whose server parser kept a tool call after the sampled tokens."""

    model_name = "policy"
    tokenizer = _Tokenizer()

    def __init__(self, response_ids, finish_reason):
        self.response_ids = response_ids
        self.finish_reason = finish_reason

    async def tokenize(self, request):
        return {"tokens": [3]}

    async def chat_completion(self, request):
        message = {"role": "assistant", "content": "call", "tool_calls": [TOOL_CALL]}
        return {"choices": [{"message": message, "token_ids": self.response_ids, "finish_reason": self.finish_reason}]}


def _generate(response_ids, finish_reason="tool_calls"):
    client = DirectModelClient(_InferenceClient(response_ids, finish_reason))
    request = InferenceEngineInput(
        prompts=[[{"role": "user", "content": "go"}]],
        chat_completion_params=[{}],
        sampling_params={},
    )
    return asyncio.run(client.generate(request))


def test_a_malformed_span_is_recorded_with_its_exact_tokens():
    output = _generate([1, 3, 2, 3, 2, 4, 5])
    assert output[REASONING_VIOLATIONS_KEY] == [ReasoningViolation.STRAY_END]
    assert output["response_ids"] == [[1, 3, 2, 3, 2, 4, 5]]


def test_a_well_formed_span_keeps_the_tool_call():
    output = _generate([1, 3, 2, 4, 5])
    assert output["assistant_messages"][0]["tool_calls"] == [TOOL_CALL]
    assert REASONING_VIOLATIONS_KEY not in output


def test_a_length_stop_inside_the_span_is_truncation():
    output = _generate([1, 3, 3], finish_reason="length")
    assert REASONING_VIOLATIONS_KEY not in output
    assert output["stop_reasons"] == ["length"]


def test_an_unclosed_span_that_stops_is_a_violation():
    output = _generate([1, 3, 5], finish_reason="stop")
    assert output[REASONING_VIOLATIONS_KEY] == [ReasoningViolation.UNCLOSED]


@pytest.mark.parametrize(
    ("response_ids", "token"),
    [
        ([1, 3, 2, 4, 1, 3, 5], 4),  # the second opening marker
        ([3, 2, 1, 3, 2, 5], 2),  # a second span after the one the template opened
        ([1, 3, 2, 3, 2, 4, 5], 4),  # the stray closing marker
        ([1, 3, 5], 2),  # the end-of-turn token sampled where the closing marker belonged
    ],
)
def test_the_violation_token_is_where_the_span_goes_wrong(response_ids, token):
    output = _generate(response_ids, finish_reason="stop")
    assert output[REASONING_VIOLATION_TOKENS_KEY] == [token]


def _step(violation, token=None):
    turn = ModelTurn(
        message={"role": "assistant", "content": "4"},
        prompt_token_ids=(3,),
        response_token_ids=(4,),
        logprobs=None,
        stop_reason="stop",
        metadata={REASONING_VIOLATION_METADATA_KEY: violation, REASONING_VIOLATION_TOKEN_METADATA_KEY: token},
    )
    return RolloutStep(turn, Transition(done=True), 0, ())


def _rollout(*violations, step_reward=None, token=None):
    conversations = tuple(
        ConversationRecord((), (3,), (4,), (1,), None, (_step(violation, token),)) for violation in violations
    )
    last = conversations[-1].steps[0]
    return TrainingRollout(
        task_id="task",
        messages=(),
        prompt_token_ids=(3,),
        response_token_ids=(4,),
        loss_mask=(1,),
        logprobs=None,
        grade=GradeResult(Outcome.GRADED, 1.0, passed=True),
        stop_reason="stop",
        steps=(TrainingStep(last.turn, TrainingTransition(done=True, reward=step_reward), 0, ()),),
        conversations=conversations,
    )


def test_an_earlier_malformed_turn_zeroes_a_correct_final_answer():
    rejected = reject_malformed_reasoning(_rollout(ReasoningViolation.SECOND_SPAN, None, step_reward=1.0))

    assert (rejected.grade.status, rejected.grade.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    assert rejected.grade.diagnostics["reasoning_violations"] == [
        {"conversation": 0, "turn": 0, "violation": ReasoningViolation.SECOND_SPAN, "token": None}
    ]
    assert rejected.steps[0].transition.reward is None


def test_a_penalized_rollout_keeps_its_task_grade():
    penalized = penalize_malformed_reasoning(_rollout(ReasoningViolation.STRAY_END), 0.2)

    assert (penalized.grade.status, penalized.grade.reward) == (Outcome.GRADED, 1.0)
    assert penalized.reasoning_span_penalty == 0.2


def test_the_penalty_comes_off_the_optimization_reward_but_not_the_outcome():
    output = {
        "response_ids": [[4], [4]],
        "loss_masks": [[1], [1]],
        "rewards": [1.0, 0.0],
        "unshaped_rewards": [1.0, 0.0],
        "reasoning_span_penalties": [0.2, 0.0],
    }

    apply_reasoning_span_penalties(output)

    assert output["rewards"] == [pytest.approx(0.8), 0.0]
    assert output["unshaped_rewards"] == [1.0, 0.0]
    assert "reasoning_span_penalties" not in output


def test_a_stepwise_trajectory_is_penalized_once_at_its_last_trainable_token():
    output = {
        "response_ids": [[4, 4], [4, 4, 4]],
        "loss_masks": [[1, 1], [1, 1, 0]],
        "rewards": [[0.0, 0.0], [0.0, 1.0, 0.0]],
        "is_last_step": [False, True],
        "reasoning_span_penalties": [0.2, 0.2],
    }

    apply_reasoning_span_penalties(output)

    assert output["rewards"] == [[0.0, 0.0], [0.0, pytest.approx(0.8), 0.0]]


def test_a_marker_penalty_charges_the_violation_token_and_keeps_the_reward():
    penalized = penalize_malformed_reasoning_tokens(
        _rollout(ReasoningViolation.STRAY_END, step_reward=1.0, token=0), 0.2
    )

    assert penalized.grade.reward == 1.0
    assert penalized.reasoning_span_penalty == 0.0
    assert penalized.steps[0].transition.reasoning_span_penalties == (0.2,)


def test_a_marker_penalty_without_a_token_position_charges_the_rollout():
    penalized = penalize_malformed_reasoning_tokens(_rollout(ReasoningViolation.STRAY_END), 0.2)

    assert penalized.reasoning_span_penalty == 0.2
    assert penalized.steps[0].transition.reasoning_span_penalties is None


def test_token_penalties_join_the_loop_credit_and_leave_the_reward():
    output = {
        "response_ids": [[4, 4], [4, 4]],
        "loss_masks": [[1, 1], [1, 1]],
        "rewards": [1.0, 0.0],
        "loop_advantages": [[0.0, -0.1], [0.0, 0.0]],
        "reasoning_span_token_penalties": [[0.2, 0.0], [0.0, 0.0]],
    }

    apply_reasoning_span_penalties(output)

    assert output["loop_advantages"] == [[-0.2, -0.1], [0.0, 0.0]]
    assert output["rewards"] == [1.0, 0.0]
    assert "reasoning_span_token_penalties" not in output


def test_a_batch_without_violations_concatenates_with_one_that_has_token_penalties():
    def batch(loop_advantages):
        output = {
            "prompt_token_ids": [[3]],
            "response_ids": [[4, 4]],
            "rewards": [1.0],
            "loss_masks": [[1, 1]],
            "stop_reasons": ["stop"],
            "rollout_logprobs": None,
        }
        if loop_advantages is not None:
            output["loop_advantages"] = loop_advantages
        return output

    merged = concatenate_trajectory_batches([batch(None), batch([[-0.2, 0.0]])], tis_lcs_alert_threshold=1.0)

    assert merged["loop_advantages"] == [[0.0, 0.0], [-0.2, 0.0]]
