# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

PLAN, CALL, END_OF_TURN, END_OF_TEXT = 3, 4, 5, 9
VOCABULARY = {PLAN: "plan", CALL: "call", END_OF_TURN: "<|eot_id|>", END_OF_TEXT: "<|end_of_text|>"}


class _LlamaStyleTokenizer:
    """Ends turns with a special token that is not ``eos_token_id``."""

    eos_token_id = END_OF_TEXT
    added_tokens_decoder = {
        END_OF_TURN: SimpleNamespace(special=True),
        END_OF_TEXT: SimpleNamespace(special=True),
    }

    def decode(self, token_ids, skip_special_tokens):
        return "".join(VOCABULARY[token] for token in token_ids)


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


@pytest.mark.parametrize(
    ("response_ids", "finish_reason", "content"),
    [
        pytest.param([PLAN, END_OF_TURN], "stop", "plan", id="end-of-turn"),
        pytest.param([PLAN, END_OF_TEXT], "stop", "plan", id="eos"),
        pytest.param([PLAN, END_OF_TURN], "length", "plan<|eot_id|>", id="length-stop-keeps-token"),
    ],
)
def test_a_special_terminator_that_is_not_eos_is_removed_from_the_reply(response_ids, finish_reason, content):
    output = _generate(response_ids, finish_reason)

    assert output["assistant_messages"][0]["content"] == content
    assert output["response_ids"][0] == response_ids


def test_a_tool_call_that_ended_its_turn_at_the_budget_is_not_a_length_stop():
    output = _generate([CALL, END_OF_TURN], "tool_calls", {"max_completion_tokens": 2})

    assert output["stop_reasons"] == ["tool_calls"]
