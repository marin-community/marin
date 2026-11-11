# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw completion wire contracts: exact text or pinned token IDs, output, and usage."""

import json

import httpx
import openai
import pytest

from rolloutengine.chat import ChatModel
from rolloutengine.completion import CompletionModel
from rolloutengine.models import ModelRequest, ModelResponseError, TokenUsage


@pytest.mark.asyncio
@pytest.mark.parametrize("token_ids", [None, (17, 29, 233)], ids=["exact-text", "pinned-token-ids"])
async def test_completion_transport_sends_exact_input_without_chat_template(token_ids):
    requests = []

    def serve(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "object": "text_completion",
                "created": 0,
                "model": "fixture",
                "choices": [{"index": 0, "finish_reason": "length", "text": " café \n"}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )

    options = {"temperature": 0, "top_p": 1, "max_tokens": 3, "stop": ["END"], "seed": 7, "extra_body": {"top_k": 11}}
    request = ModelRequest(({"role": "user", "content": "  café\n\nAnswer:"},), options, prompt_token_ids=token_ids)
    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://completion.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        turn = await CompletionModel(client, "fixture")(request)
    assert requests == [
        (
            "/v1/completions",
            {
                "model": "fixture",
                "prompt": list(token_ids) if token_ids is not None else "  café\n\nAnswer:",
                "stream": False,
                "temperature": 0,
                "top_p": 1,
                "max_tokens": 3,
                "stop": ["END"],
                "seed": 7,
                "top_k": 11,
            },
        )
    ]
    assert turn.message == {"role": "assistant", "content": " café \n"}
    assert turn.text == " café \n"
    assert turn.stop_reason == "length"
    assert turn.usage == TokenUsage(7, 3)
    assert turn.metadata == {"response_id": "response-1"}
    assert request.options == options


@pytest.mark.asyncio
async def test_completion_transport_rejects_a_response_with_unknown_token_usage():
    def serve(_request):
        return httpx.Response(
            200,
            json={
                "id": "missing-usage",
                "object": "text_completion",
                "created": 0,
                "model": "fixture",
                "choices": [{"index": 0, "finish_reason": "stop", "text": "retained output"}],
            },
        )

    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://completion.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        with pytest.raises(ModelResponseError, match="token usage") as failure:
            await CompletionModel(client, "fixture")(ModelRequest(({"role": "user", "content": "prompt"},), {}))
    assert failure.value.response["choices"][0]["text"] == "retained output"
    assert failure.value.response["id"] == "missing-usage"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [False, True], ids=["completion", "chat"])
@pytest.mark.parametrize("choice_count", [0, 2])
async def test_malformed_choice_lists_preserve_the_endpoint_response_without_retry(chat, choice_count):
    calls = []
    choice = {"index": 0, "finish_reason": "stop"}
    choice.update({"message": {"role": "assistant", "content": "answer"}} if chat else {"text": "answer"})
    body = {
        "id": "malformed-choices",
        "object": "chat.completion" if chat else "text_completion",
        "created": 0,
        "model": "fixture",
        "choices": [choice] * choice_count,
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }

    def serve(request):
        calls.append(request)
        return httpx.Response(200, json=body)

    async with openai.AsyncOpenAI(
        api_key="test",
        base_url="http://completion.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = ChatModel(client, "fixture", 3) if chat else CompletionModel(client, "fixture")
        with pytest.raises(ModelResponseError, match="one choice") as failure:
            await model(ModelRequest(({"role": "user", "content": "prompt"},), {}))
    assert len(calls) == 1
    assert failure.value.response["id"] == body["id"]
    assert len(failure.value.response["choices"]) == choice_count
    for recorded in failure.value.response["choices"]:
        assert recorded["finish_reason"] == "stop"
        assert recorded["index"] == 0
        assert recorded["message"]["content"] == "answer" if chat else recorded["text"] == "answer"
