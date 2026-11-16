# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace

import httpx
import openai
import pytest
from taskcompendium.request_budget import PromptBudget, RequestBudget, TokenizationIdentity, tool_free_prompt_sha256
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rolloutengine.models import ModelRequest, ModelResponseError
from rolloutengine.tool_free import ToolFreeChatModel

MESSAGES = ({"role": "system", "content": "coordinate task"}, {"role": "user", "content": "find atom"})


def native_model(tmp_path, *, response_fields=None, finish_reason="stop", context_tokens=100, status_code=200):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}assistant"
    identity = TokenizationIdentity(
        tokenizer="fixture/native",
        revision="a" * 40,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        template_arguments={},
    )
    count = len(
        tokenizer.apply_chat_template(list(MESSAGES), tokenize=True, return_dict=False, add_generation_prompt=True)
    )
    budget = PromptBudget(
        prompt_sha256=tool_free_prompt_sha256(MESSAGES),
        model="native-model",
        endpoint="https://native.test/v1",
        backend_identity_sha256="b" * 64,
        tokenization=identity,
        endpoint_limit_source="fixture native endpoint config",
        prompt_tokens=count,
        limits=RequestBudget(context_tokens, context_tokens - 1, 70),
    )
    requests = []
    payload = {
        "id": "completion",
        "object": "chat.completion",
        "created": 0,
        "model": "native-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "FINAL: C:T670:N",
                    **(response_fields or {}),
                },
            }
        ],
        "usage": {"prompt_tokens": count, "completion_tokens": 1, "total_tokens": count + 1},
    }

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(status_code, json=payload)

    client = openai.AsyncOpenAI(
        api_key="fixture",
        base_url="https://native.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    )
    return (
        ToolFreeChatModel(client, "native-model", "b" * 64, identity, tokenizer, budget, tmp_path / "traffic"),
        requests,
        payload,
    )


@pytest.mark.parametrize(
    "fields,finish_reason,truncated,refusal,tool_event",
    [
        ({}, "stop", False, False, False),
        ({}, "length", True, False, False),
        ({"refusal": "refused"}, "stop", False, True, False),
        (
            {
                "tool_calls": [
                    {"id": "call", "type": "function", "function": {"name": "shell", "arguments": "invalid JSON"}}
                ]
            },
            "tool_calls",
            False,
            False,
            True,
        ),
        ({"annotations": [{"type": "web_search_call"}]}, "stop", False, False, True),
    ],
)
async def test_native_single_completion_preserves_violation_evidence(
    tmp_path, fields, finish_reason, truncated, refusal, tool_event
):
    model, requests, payload = native_model(tmp_path, response_fields=fields, finish_reason=finish_reason)
    async with model.client:
        turn = await model(ModelRequest(MESSAGES, {"temperature": 0.7}))
    assert len(requests) == 1
    assert requests[0]["tools"] == [] and requests[0]["tool_choice"] == "none"
    assert requests[0]["plugins"] == [{"id": "web", "enabled": False}]
    assert requests[0]["messages"] == list(MESSAGES)
    assert turn.message == {"role": "assistant", "content": "FINAL: C:T670:N"}
    assert turn.reply_status is not None
    assert (turn.reply_status.truncated, turn.reply_status.provider_refusal, bool(turn.reply_status.tool_events)) == (
        truncated,
        refusal,
        tool_event,
    )
    assert json.loads((model.receipts / "raw_response.json").read_text()) == payload
    assert json.loads((model.receipts / "wire_request.json").read_text()) == requests[0]


@pytest.mark.parametrize("context_tokens", [100, 2])
async def test_native_request_uses_all_available_output_capacity(tmp_path, context_tokens):
    model, requests, _ = native_model(tmp_path, context_tokens=context_tokens)
    async with model.client:
        await model(ModelRequest(MESSAGES, {}))
    assert requests[0]["max_tokens"] == min(70, context_tokens - model.budget.prompt_tokens)


@pytest.mark.parametrize("mismatch", ["prompt", "backend", "count", "template", "endpoint", "input_cap"])
async def test_budget_mismatch_never_sends_inference(tmp_path, mismatch):
    model, requests, _ = native_model(tmp_path)
    messages = MESSAGES
    if mismatch == "prompt":
        messages = (MESSAGES[0], {"role": "user", "content": "changed coordinates"})
    elif mismatch == "backend":
        model = replace(model, backend_identity_sha256="c" * 64)
    elif mismatch == "count":
        model = replace(model, budget=model.budget.model_copy(update={"prompt_tokens": model.budget.prompt_tokens + 1}))
    elif mismatch == "template":
        model.tokenizer.chat_template += " altered"
    elif mismatch == "endpoint":
        model = replace(model, budget=model.budget.model_copy(update={"endpoint": "https://another.test/v1"}))
    else:
        model = replace(model, budget=model.budget.model_copy(update={"limits": RequestBudget(100, 50, 70)}))
    async with model.client:
        with pytest.raises(ValueError):
            await model(ModelRequest(messages, {}))
    assert requests == []


async def test_endpoint_usage_mismatch_is_a_technical_failure_with_raw_evidence(tmp_path):
    model, requests, payload = native_model(tmp_path)
    payload["usage"]["prompt_tokens"] += 1
    async with model.client:
        with pytest.raises(ModelResponseError, match="exact native count"):
            await model(ModelRequest(MESSAGES, {}))
    assert len(requests) == 1
    assert json.loads((model.receipts / "raw_response.json").read_text()) == payload


@pytest.mark.parametrize(
    "options",
    [
        {"max_tokens": 1},
        {"tools": [{"type": "function"}]},
        {"extra_body": {"plugins": [{"id": "web", "enabled": True}]}},
    ],
)
async def test_sampling_cannot_enable_tools_or_shrink_native_output(tmp_path, options):
    model, requests, _ = native_model(tmp_path)
    async with model.client:
        with pytest.raises(ValueError, match="override"):
            await model(ModelRequest(MESSAGES, options))
    assert requests == []


async def test_rejected_request_retains_raw_http_body_and_status(tmp_path):
    model, requests, payload = native_model(tmp_path, status_code=400)
    async with model.client:
        with pytest.raises(openai.BadRequestError):
            await model(ModelRequest(MESSAGES, {}))
    assert len(requests) == 1
    assert json.loads((model.receipts / "raw_response.json").read_text()) == payload
    assert json.loads((model.receipts / "wire_request.json").read_text()) == requests[0]
    assert json.loads((model.receipts / "response_status.json").read_text()) == {"status": 400}
