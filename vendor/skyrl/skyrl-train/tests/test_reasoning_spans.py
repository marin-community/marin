# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from taskcompendium.reasoning_spans import ReasoningViolation

from skyrl_train.inference_engines.base import InferenceEngineInput
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


def test_an_unclosed_span_that_stops_is_a_violation():
    output = _generate([1, 3, 5], finish_reason="stop")
    assert output[REASONING_VIOLATIONS_KEY] == [ReasoningViolation.UNCLOSED]
