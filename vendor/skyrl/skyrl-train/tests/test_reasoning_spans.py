# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import numpy as np
from rolloutengine.models import ConversationRecord, ModelTurn, RolloutStep, Transition
from taskcompendium.grading_result import GradeResult, Outcome
from verifyit.reasoning_markup import ReasoningViolation

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.rollouts.task_projections import _attach_reasoning_violations, _step_rollouts, _model_channel
from skyrl_train.rollouts.training_record import (
    TrainingRollout,
    TrainingStep,
    TrainingTransition,
    REASONING_VIOLATION_TOKEN_METADATA_KEY,
    reject_malformed_reasoning,
)
from skyrl_train.trajectory_runners.model_clients import (
    REASONING_VIOLATION_TOKENS_KEY,
    REASONING_VIOLATIONS_KEY,
    DirectModelClient,
)
from skyrl_train.trajectory_runners.trajectory_processing import concatenate_trajectory_batches
from skyrl_train.trajectory_runners.trajectory_reward_shaping import shape_trajectory_rewards

VOCABULARY = {1: "<|start_think|>", 2: "<|end_think|>", 3: "plan", 4: "call", 5: "<|eot_id|>", 6: "</think>"}
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


@pytest.mark.parametrize(
    ("response_ids", "finish_reason", "violation"),
    [
        ([1, 3, 2, 4, 5], "tool_calls", None),
        ([1, 3, 2, 3, 2, 4, 5], "tool_calls", ReasoningViolation.STRAY_END),
        ([1, 3, 3], "length", None),
        ([1, 3, 5], "stop", ReasoningViolation.UNCLOSED),
    ],
)
def test_the_chat_transport_records_each_reply_violation(response_ids, finish_reason, violation):
    output = _generate(response_ids, finish_reason)
    assert output.get(REASONING_VIOLATIONS_KEY, [None]) == [violation]


def _step(violation):
    turn = ModelTurn(
        message={"role": "assistant", "content": "4"},
        prompt_token_ids=(3,),
        response_token_ids=(4,),
        logprobs=None,
        stop_reason="stop",
        reasoning_violation=violation,
    )
    return RolloutStep(turn, Transition(done=True), 0, ())


def _rollout(*violations, step_reward=None):
    conversations = tuple(ConversationRecord((), (3,), (4,), (1,), None, (_step(violation),)) for violation in violations)
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
        {"conversation": 0, "turn": 0, "violation": ReasoningViolation.SECOND_SPAN}
    ]
    assert rejected.steps[0].transition.reward is None


@pytest.mark.parametrize(
    ("response_ids", "token"),
    [
        ([1, 3, 2, 4, 1, 3, 5], 4),  # the second opening marker
        ([3, 2, 1, 3, 2, 5], 2),  # a second span after the one the template opened
        ([1, 3, 2, 3, 2, 4, 5], 4),  # the stray closing marker
        ([1, 3, 6, 4, 2, 5], 2),  # a closing marker of another syntax
        ([1, 3, 5], 2),  # the end-of-turn token sampled where the closing marker belonged
    ],
)
def test_the_violation_token_is_where_the_span_goes_wrong(response_ids, token):
    output = _generate(response_ids, finish_reason="stop")
    assert output[REASONING_VIOLATION_TOKENS_KEY] == [token]


def _shape(output, target="rollout"):
    shape_trajectory_rewards(output, {"enabled": True, "reasoning_span": {"penalty": 0.2, "target": target}})
    return output


def test_the_penalty_comes_off_the_optimization_reward_but_not_the_outcome():
    output = _shape(
        {
            "response_ids": [[4], [4]],
            "loss_masks": [[1], [1]],
            "rewards": [1.0, 0.0],
            "reasoning_span_violations": [1, 0],
            "reasoning_span_violation_tokens": [[], []],
        }
    )

    assert output["rewards"] == [pytest.approx(0.8), 0.0]
    assert output["unshaped_rewards"] == [1.0, 0.0]
    assert output["rollout_metrics"]["generate/reward_shaping/reasoning_span_incidence"] == 0.5


def test_a_stepwise_trajectory_is_penalized_once_at_its_last_trainable_token():
    output = _shape(
        {
            "response_ids": [[4, 4], [4, 4, 4]],
            "loss_masks": [[1, 1], [1, 1, 0]],
            "rewards": [[0.0, 0.0], [0.0, 1.0, 0.0]],
            "is_last_step": [False, True],
            "reasoning_span_violations": [1, 0],
            "reasoning_span_violation_tokens": [[], []],
        }
    )

    assert output["rewards"] == [[0.0, 0.0], [0.0, pytest.approx(0.8), 0.0]]


@pytest.mark.parametrize(
    ("tokens", "loop_advantages", "reward"),
    [
        ([1], [0.0, -0.2], 1.0),  # charged at the marker token
        ([], [0.0, 0.0], 0.8),  # no marker token, so the reward is charged
    ],
)
def test_a_marker_penalty_charges_the_violation_token_when_it_has_one(tokens, loop_advantages, reward):
    output = _shape(
        {
            "response_ids": [[4, 4]],
            "loss_masks": [[1, 1]],
            "rewards": [1.0],
            "reasoning_span_violations": [1],
            "reasoning_span_violation_tokens": [tokens],
        },
        target="marker",
    )

    assert output["loop_advantages"] == [pytest.approx(loop_advantages)]
    assert output["rewards"] == [pytest.approx(reward)]
    metrics = output["rollout_metrics"]
    assert metrics["generate/reward_shaping/reasoning_span_penalty_mean"] == pytest.approx(-0.2)
    assert metrics["generate/reward_shaping/reasoning_span_incidence"] == 1.0
    assert metrics["generate/reward_shaping/loop_charged_tokens_mean"] == 0.0


@pytest.mark.parametrize(
    ("earlier_violation", "violations", "rewards", "loop_advantages"),
    [
        (None, [0, 0, 1], [0.0, 0.0, 1.0], [[0.0, 0.0], [0.0, 0.0], [0.0, -0.2]]),  # charged once, at the marker token
        # An earlier conversation remains trainable; its unlocated violation is charged once at the attempt end.
        (ReasoningViolation.STRAY_END, [1, 0, 1], [0.0, 0.0, 0.8], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    ],
)
def test_step_rows_count_only_their_own_turns(earlier_violation, violations, rewards, loop_advantages):
    def turn(violation, token=None):
        return ModelTurn(
            message={"role": "assistant", "content": "4"},
            prompt_token_ids=(3,),
            response_token_ids=(4, 4),
            logprobs=(-0.2, -0.3),
            stop_reason="stop",
            metadata={
                "response_policy_steps": [2, 2],
                "routed_experts": np.array([[[1, 3]], [[2, 0]]], dtype=np.int32),
                **({} if token is None else {REASONING_VIOLATION_TOKEN_METADATA_KEY: token}),
            },
            reasoning_violation=violation,
        )

    def conversation(*turns):
        steps = tuple(RolloutStep(t, Transition(done=False), 0, ()) for t in turns)
        return ConversationRecord((), (3,), (4,) * 2 * len(turns), (1,) * 2 * len(turns), None, steps)

    optimized = (turn(None), turn(ReasoningViolation.SECOND_SPAN, token=1))
    rollout = TrainingRollout(
        task_id="task",
        messages=(),
        prompt_token_ids=(3,),
        response_token_ids=(4, 4, 4, 4),
        loss_mask=(1, 1, 1, 1),
        logprobs=None,
        grade=GradeResult(Outcome.GRADED, 1.0, passed=True),
        stop_reason="stop",
        steps=tuple(
            TrainingStep(t, TrainingTransition(done=index == 1), 2 * index + 1, ()) for index, t in enumerate(optimized)
        ),
        conversations=(conversation(turn(earlier_violation)), conversation(*optimized)),
    )
    output = {
        "response_ids": [[4, 4], [4, 4], [4, 4]],
        "loss_masks": [[1, 1], [1, 1], [1, 1]],
        "rewards": [0.0, 0.0, 1.0],
        "is_last_step": [False, False, True],
    }

    projected = _step_rollouts(rollout)
    # The reset used to discard the first turn, including its behavior-policy and router evidence.
    assert len(projected) == 3
    for row in projected:
        assert row.loss_mask == (1, 1)
        assert row.prompt_token_ids == (3,)
        assert row.response_token_ids == (4, 4)
        assert row.logprobs == (-0.2, -0.3)
        np.testing.assert_array_equal(_model_channel(row, "response_policy_steps", 2, (), np.int32), [2, 2])
        np.testing.assert_array_equal(
            _model_channel(row, "routed_experts", 2, (1, 2), np.int32), [[[1, 3]], [[2, 0]]]
        )
    _attach_reasoning_violations(output, projected, ends_attempt=[False, False, True])
    _shape(output, target="marker")

    assert output["reasoning_span_violations"] == violations
    assert output["rewards"] == pytest.approx(rewards)
    assert output["loop_advantages"] == [pytest.approx(row) for row in loop_advantages]


def _batch(**fields):
    return {
        "prompt_token_ids": [[3]],
        "response_ids": [[4, 4]],
        "rewards": [1.0],
        "loss_masks": [[1, 1]],
        "stop_reasons": ["stop"],
        "rollout_logprobs": None,
        **fields,
    }


def test_loop_credit_survives_concatenation_with_a_batch_that_did_not_shape_rewards():
    merged = concatenate_trajectory_batches(
        [_batch(), _batch(loop_advantages=[[-0.2, 0.0]])], tis_lcs_alert_threshold=1.0
    )

    assert merged["loop_advantages"] == [[0.0, 0.0], [-0.2, 0.0]]


def test_reasoning_span_evidence_survives_concatenation_with_a_batch_that_has_none():
    malformed = _batch(reasoning_span_violations=[1], reasoning_span_violation_tokens=[[1]])

    merged = concatenate_trajectory_batches([malformed, _batch()], tis_lcs_alert_threshold=1.0)

    assert merged["reasoning_span_violations"] == [1, 0]
    assert merged["reasoning_span_violation_tokens"] == [[1], []]
