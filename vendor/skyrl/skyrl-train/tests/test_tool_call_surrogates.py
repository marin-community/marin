# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

CALL, END_OF_TURN = 1, 2
# vLLM's Hermes parser decodes the model's `{"command": "echo \ud83d"}` and re-serializes it without ASCII escapes.
ARGUMENTS = json.dumps({"command": "echo \ud83d é"}, ensure_ascii=False)


class _Tokenizer:
    eos_token_id = END_OF_TURN

    def decode(self, token_ids, skip_special_tokens):
        return "".join({CALL: "call", END_OF_TURN: "<|eot_id|>"}[token] for token in token_ids)

    def encode(self, text, add_special_tokens):
        return [CALL]


class _InferenceClient:
    model_name = "policy"
    tokenizer = _Tokenizer()

    async def tokenize(self, request):
        return {"tokens": [CALL]}

    async def chat_completion(self, request):
        call = {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": ARGUMENTS}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        return {"choices": [{"message": message, "token_ids": [CALL, END_OF_TURN], "finish_reason": "tool_calls"}]}


def test_a_tool_call_with_a_lone_surrogate_keeps_its_arguments_and_encodes_for_the_next_turn():
    request = InferenceEngineInput(
        prompts=[[{"role": "user", "content": "go"}]], chat_completion_params=[{}], sampling_params={}
    )
    output = asyncio.run(DirectModelClient(_InferenceClient()).generate(request))

    arguments = output["assistant_messages"][0]["tool_calls"][0]["function"]["arguments"]
    arguments.encode()
    assert json.loads(arguments) == json.loads(ARGUMENTS)
    assert "é" in arguments
