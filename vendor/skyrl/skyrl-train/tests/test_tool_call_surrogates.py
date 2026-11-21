# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from copy import deepcopy

from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

CALL, END_OF_TURN = "call", "<|eot_id|>"
# vLLM's Hermes parser decodes the model's `{"command": "echo \ud83d é"}` and re-serializes it without ASCII escapes.
ARGUMENTS = json.dumps({"command": "echo \ud83d é"}, ensure_ascii=False)
TOKENIZER = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, CALL: 1, END_OF_TURN: 2}, unk_token="[UNK]")),
    unk_token="[UNK]",
    eos_token=END_OF_TURN,
    # The Snowball chat template writes decoded tool-call arguments with tojson, which keeps non-ASCII characters.
    chat_template=(
        "{% for message in messages %}{{ message.content or '' }}"
        "{% for call in message.tool_calls or [] %}{{ call.function.arguments | tojson }}{% endfor %}{% endfor %}"
    ),
)


class _InferenceClient:
    """Renders a prompt as vLLM does: it decodes tool-call arguments, then applies the chat template."""

    model_name = "policy"
    tokenizer = TOKENIZER

    async def tokenize(self, request):
        messages = deepcopy(request["json"]["messages"])
        for message in messages:
            for call in message.get("tool_calls") or []:
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        return {"tokens": TOKENIZER.apply_chat_template(messages, tokenize=True, return_dict=False)}

    async def chat_completion(self, request):
        call = {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": ARGUMENTS}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        token_ids = TOKENIZER.convert_tokens_to_ids([CALL, END_OF_TURN])
        return {"choices": [{"message": message, "token_ids": token_ids, "finish_reason": "tool_calls"}]}


def _generate(client, messages):
    request = InferenceEngineInput(prompts=[messages], chat_completion_params=[{}], sampling_params={})
    return asyncio.run(client.generate(request))


def test_the_turn_after_a_tool_call_with_a_lone_surrogate_renders():
    client = DirectModelClient(_InferenceClient())
    user = {"role": "user", "content": "go"}
    assistant = _generate(client, [user])["assistant_messages"][0]
    tool_reply = {"role": "tool", "tool_call_id": "call-1", "content": "done"}

    # The tokenizer raises TypeError for a lone surrogate, and the task worker stops the run on a TypeError.
    assert _generate(client, [user, assistant, tool_reply])["prompt_ids"][0]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"command": "echo \ufffd é"}
