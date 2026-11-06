# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

PLAN, CALL, END_OF_TURN, TOOL_CALL_END, END_OF_TEXT = 3, 4, 5, 6, 9
VOCABULARY = {
    PLAN: "plan",
    CALL: "call",
    END_OF_TURN: "<|eot_id|>",
    TOOL_CALL_END: "<|tool_call_end|>",
    END_OF_TEXT: "<|end_of_text|>",
}


class _LlamaStyleTokenizer:
    """Ends turns with a special token that is not ``eos_token_id``."""

    eos_token_id = END_OF_TEXT

    def decode(self, token_ids, skip_special_tokens):
        return "".join(VOCABULARY[token] for token in token_ids)

    def encode(self, text, add_special_tokens):
        token_ids = [token for token, piece in VOCABULARY.items() if piece == text]
        return token_ids or [PLAN, PLAN]


class _InferenceClient:
    model_name = "policy"
    tokenizer = _LlamaStyleTokenizer()

    def __init__(self, response_ids, finish_reason):
        self.response_ids = response_ids
        self.finish_reason = finish_reason

    async def tokenize(self, request):
        return {"tokens": [PLAN]}

    async def chat_completion(self, request):
        content = _LlamaStyleTokenizer().decode(self.response_ids, skip_special_tokens=False)
        message = {"role": "assistant", "content": content}
        return {"choices": [{"message": message, "token_ids": self.response_ids, "finish_reason": self.finish_reason}]}


def _generate(response_ids, finish_reason, chat_options=None):
    client = DirectModelClient(_InferenceClient(response_ids, finish_reason))
    request = InferenceEngineInput(
        prompts=[[{"role": "user", "content": "go"}]],
        chat_completion_params=[chat_options or {}],
        sampling_params={},
    )
    return asyncio.run(client.generate(request))


@pytest.mark.parametrize("terminator", [END_OF_TURN, END_OF_TEXT], ids=["end-of-turn", "eos"])
def test_the_special_token_a_turn_stopped_at_is_removed_from_the_reply(terminator):
    output = _generate([PLAN, terminator], "stop")

    assert output["assistant_messages"][0]["content"] == "plan"
    assert output["response_ids"][0] == [PLAN, terminator]


def test_a_length_stop_keeps_its_last_token():
    output = _generate([PLAN, END_OF_TURN], "length")

    assert output["assistant_messages"][0]["content"] == "plan<|eot_id|>"


def test_a_tool_call_that_ended_its_turn_at_the_budget_is_not_a_length_stop():
    output = _generate([CALL, END_OF_TURN], "tool_calls", {"max_completion_tokens": 2})

    assert output["stop_reasons"] == ["tool_calls"]


def test_a_tool_call_cut_off_at_the_budget_on_a_special_delimiter_is_a_length_stop():
    output = _generate([CALL, TOOL_CALL_END], "tool_calls", {"max_completion_tokens": 2})

    assert output["stop_reasons"] == ["length"]
