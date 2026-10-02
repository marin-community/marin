# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
from pathlib import Path

import httpx
import litellm
import pytest
from marin.evaluation.harbor.mini_swe_request import ContextBudgetExhausted, ContextLimits, context_max_tokens
from minisweagent.exceptions import LimitsExceeded
from minisweagent.models import get_model


@pytest.mark.parametrize(
    ("prompt_tokens", "requested", "server_context", "expected"),
    ((16000, 16384, 65536, 16384), (32000, 16384, 65536, 768), (32000, 128, 65536, 128), (16000, 16384, 16384, 384)),
)
def test_reply_budget_uses_rendered_prompt_and_retains_smaller_request_limits(
    prompt_tokens, requested, server_context, expected
):
    messages = [{"role": "user", "content": "Fix the issue"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]

    def tokenizer(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://model/tokenize"
        assert json.loads(request.content) == {
            "model": "served-model",
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        return httpx.Response(200, json={"count": prompt_tokens, "max_model_len": server_context})

    with httpx.Client(transport=httpx.MockTransport(tokenizer)) as client:
        budget = context_max_tokens(
            client,
            "http://model/v1",
            "served-model",
            messages,
            tools,
            {"max_tokens": requested, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            ContextLimits(32768, 32768, 16384),
        )
    assert budget == expected


@pytest.mark.parametrize("prompt_tokens", (32768, 40000))
def test_context_exhaustion_stops_before_requesting_a_reply(prompt_tokens):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"count": prompt_tokens, "max_model_len": 65536})
        )
    ) as client:
        with pytest.raises(ContextBudgetExhausted):
            context_max_tokens(
                client,
                "http://model/v1",
                "served-model",
                [{"role": "user", "content": "Long conversation"}],
                [],
                {},
                ContextLimits(32768, 32768, 16384),
            )


def test_native_harness_caps_requests_and_exits_cleanly_at_the_context_boundary(monkeypatch):
    model_directory = Path(__file__).parents[2] / "lib/marin/src/marin/evaluation/harbor"
    monkeypatch.syspath_prepend(str(model_directory))
    importlib.import_module("mini_swe_model")
    requests = []
    counts = iter((32000, 32768))
    client_type = httpx.Client
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json={"count": next(counts), "max_model_len": 65536})
    )
    monkeypatch.setattr("mini_swe_model.httpx.Client", lambda **kwargs: client_type(transport=transport, **kwargs))

    def completion(**kwargs):
        requests.append(kwargs)
        return litellm.ModelResponse(
            model="served-model",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "bash", "arguments": '{"command": "echo ready"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 32000, "completion_tokens": 768, "total_tokens": 32768},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    model = get_model(
        config={
            "model_class": "mini_swe_model.ContextLimitedModel",
            "model_name": "hosted_vllm/served-model",
            "api_base": "http://model/v1",
            "max_context_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 16384,
            "cost_tracking": "ignore_errors",
            "model_kwargs": {"temperature": 0.25},
        }
    )
    messages = [{"role": "user", "content": "Fix the issue"}]
    reply = model.query(messages)
    assert reply["extra"]["actions"][0]["command"] == "echo ready"
    assert requests[0]["max_tokens"] == 768
    assert requests[0]["temperature"] == 0.25
    assert model.serialize()["info"]["config"]["model"]["max_context_tokens"] == 32768
    with pytest.raises(LimitsExceeded) as exhausted:
        model.query(messages)
    assert exhausted.value.messages[-1]["extra"]["exit_status"] == "ContextLimit"
    assert len(requests) == 1
