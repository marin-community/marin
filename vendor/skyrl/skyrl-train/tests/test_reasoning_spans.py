# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from rolloutengine.models import ConversationRecord, ModelTurn, RolloutStep, Transition
from taskcompendium.grading_result import GradeResult, Outcome
from verifyit.reasoning_markup import ReasoningViolation

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.rollouts.training_record import (
    TrainingRollout,
    TrainingStep,
    TrainingTransition,
    reject_malformed_reasoning,
)
from skyrl_train.trajectory_runners.model_clients import REASONING_VIOLATIONS_KEY, DirectModelClient

VOCABULARY = {1: "<|start_think|>", 2: "<|end_think|>", 3: "plan", 4: "call", 5: "<|eot_id|>"}
TOOL_CALL = {"id": "call-0", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}


class _Tokenizer:
    eos_token_id = 5

    def decode(self, token_ids, skip_special_tokens):
        return "".join(VOCABULARY[token] for token in token_ids)


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
