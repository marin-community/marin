"""Keep each reply's raw decoded text, so a thinking-disabled check sees reasoning that parsing removes."""

from dataclasses import dataclass

import pytest
from skyrl_train.trajectory_runners.model_clients import DirectModelClient

TOKENS = {"<think>": 7, "</think>": 8, "42": 4}
WORDS = {token: word for word, token in TOKENS.items()}


class Tokenizer:
    eos_token_id = 99

    def decode(self, token_ids, skip_special_tokens):
        return "".join(WORDS[token] for token in token_ids)


@dataclass
class Engine:
    response_ids: list[int]
    tokenizer: Tokenizer = Tokenizer()
    model_name: str = "fixture"

    async def tokenize(self, request):
        return {"tokens": [1, 2]}

    async def chat_completion(self, request):
        text = self.tokenizer.decode(self.response_ids, skip_special_tokens=False)
        message = {"role": "assistant", "content": text}
        return {"choices": [{"message": message, "token_ids": self.response_ids, "finish_reason": "stop"}]}


@pytest.mark.asyncio
async def test_raw_responses_keep_reasoning_that_the_visible_reply_drops():
    client = DirectModelClient(Engine([7, 4, 8, 4]))
    output = await client.generate(
        {"prompts": [[{"role": "user", "content": "Reply 42."}]], "chat_completion_params": [{}], "sampling_params": {}}
    )
    assert (output["responses"], output["raw_responses"]) == (["42"], ["<think>42</think>42"])
