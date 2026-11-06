# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``GlmBatchTransport`` under an OpenAI client, with a fake batch API that answers in its own order."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai
import pytest
from marin.inference.openai_batch import BatchOutput, BatchSubmission

from experiments.post_training.capability_driven_envs.glm_batch import GlmBatchTransport


def completion(content: str) -> dict[str, Any]:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "glm",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


@dataclass
class FakeBatchApi:
    """Echoes the last user message of each request, in reverse order, and leaves out the requests that say
    ``drop``."""

    batches: list[list[Mapping[str, Any]]] = field(default_factory=list)

    def submit(self, requests: Sequence[Mapping[str, Any]], filename: str) -> BatchSubmission:
        self.batches.append(list(requests))
        return BatchSubmission(file_id="f", batch_id=str(len(self.batches) - 1))

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]:
        return {"id": batch_id, "status": "completed"}

    def output(self, batch: Mapping[str, Any]) -> BatchOutput:
        rows = []
        for request in reversed(self.batches[int(batch["id"])]):
            content = request["body"]["messages"][-1]["content"]
            if content == "drop":
                continue
            response = {"status_code": 200, "body": completion(f"echo {content}")}
            rows.append(json.dumps({"custom_id": request["custom_id"], "response": response, "error": None}))
        return BatchOutput(output="\n".join(rows), errors=None)


async def ask(client: openai.AsyncOpenAI, content: str) -> str | None:
    response = await client.chat.completions.create(model="glm", messages=[{"role": "user", "content": content}])
    return response.choices[0].message.content


async def test_concurrent_requests_share_a_batch_and_get_their_own_answers():
    api = FakeBatchApi()
    transport = GlmBatchTransport(api, batch_size=64, flush=0.2, poll=0, concurrent_batches=1)
    client = openai.AsyncOpenAI(
        base_url="http://relay/v1", api_key="k", http_client=httpx.AsyncClient(transport=transport)
    )
    try:
        answers = await asyncio.gather(*(ask(client, str(n)) for n in range(5)))
    finally:
        await transport.aclose()

    assert answers == [f"echo {n}" for n in range(5)]
    assert len(api.batches) == 1


async def test_a_request_that_the_batch_does_not_answer_fails_after_the_client_retries():
    api = FakeBatchApi()
    transport = GlmBatchTransport(api, batch_size=64, flush=0.01, poll=0, concurrent_batches=1)
    client = openai.AsyncOpenAI(
        base_url="http://relay/v1", api_key="k", max_retries=1, http_client=httpx.AsyncClient(transport=transport)
    )
    try:
        with pytest.raises(openai.InternalServerError) as error:
            await ask(client, "drop")
    finally:
        await transport.aclose()

    assert error.value.status_code == 502
    # The OpenAI client sent the request again after the first 502.
    assert len(api.batches) == 2
