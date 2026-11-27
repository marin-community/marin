# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from skyrl_train.trajectory_runners.types import TokenProvenance


class WorkerTokenizer:
    eos_token_id = 99

    def decode(self, _token_ids, **_kwargs):
        return "Working"


def sampled_reply(content="Working", **message):
    return {
        "token_provenance": TokenProvenance.ENGINE,
        "assistant_messages": [{"role": "assistant", "content": content, **message}],
        "prompt_ids": [[1, 2]],
        "response_ids": [[3]],
        "response_logprobs": [[-0.1]],
        "stop_reasons": ["stop"],
    }


class ScriptedClient:
    """Answer each generation with the next reply, raising the replies that are exceptions."""

    def __init__(self, replies):
        self.replies = iter(replies)

    async def generate(self, _request):
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply
