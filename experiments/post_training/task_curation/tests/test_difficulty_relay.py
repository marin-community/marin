# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise dynamic admission and retained evidence through the real HTTP relay."""

import json
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from taskcompendium.request_budget import RequestBudget

from experiments.post_training.task_curation import difficulty_relay
from experiments.post_training.task_curation.difficulty_output_budget import DynamicOutputBudget
from experiments.post_training.task_curation.difficulty_provider import ProviderLimits

HTTP_CLIENT = httpx.Client


@contextmanager
def relay(monkeypatch, tmp_path, response, *, dynamic=True):
    requests = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return list(range(int(messages[0]["content"])))

    def serve(request):
        assert request.url == "https://model.invalid/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json=response(body))

    monkeypatch.setattr(difficulty_relay.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(
        difficulty_relay.httpx, "Client", lambda **kwargs: HTTP_CLIENT(transport=httpx.MockTransport(serve), **kwargs)
    )
    with difficulty_relay.budget_relay(
        endpoint="https://model.invalid/v1",
        model="frozen-model",
        api_key="fixture-key",
        tokenizer_model="frozen-model",
        tokenizer_revision="a" * 40,
        parameters={"temperature": 0.7, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        budget=RequestBudget(65536, 49152, 16384),
        provider_limits=ProviderLimits(max_in_flight=1, attempts=1, request_timeout=10),
        receipts=tmp_path,
        dynamic_output=DynamicOutputBudget() if dynamic else None,
    ) as endpoint:
        with HTTP_CLIENT(base_url=endpoint) as client:
            yield client, requests


def request(prompt_tokens, max_tokens=7):
    return {
        "model": "guest-alias",
        "messages": [{"role": "user", "content": str(prompt_tokens)}],
        "max_tokens": max_tokens,
    }


@pytest.mark.parametrize("kind", ["answer", "early_stop", "full_length"])
def test_dynamic_relay_grants_the_entire_allowance_and_retains_original_request(monkeypatch, tmp_path, kind):
    cases = [(100, 65372), (12000, 53472), (49088, 16384)]
    expected = dict(cases)

    def response(body):
        prompt = int(body["messages"][0]["content"])
        return {
            "choices": [
                {
                    "finish_reason": "length" if kind == "full_length" else "stop",
                    "message": {"content": None if kind != "answer" else "answer"},
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": expected[prompt] if kind == "full_length" else 0 if kind == "early_stop" else 4,
            },
        }

    with relay(monkeypatch, tmp_path, response) as (client, upstream):
        for prompt, allowance in cases:
            original = request(prompt)
            got = client.post("/task/v1/chat/completions", json=original)
            assert got.status_code == 200
            assert got.json() == response(original)
            assert upstream[-1]["max_tokens"] == allowance
            assert upstream[-1]["messages"] == original["messages"]
            assert upstream[-1]["temperature"] == 0.7
            assert upstream[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    archived = sorted(tmp_path.glob("*/*/request.json"))
    assert len(archived) == 3
    for path, (prompt, allowance) in zip(archived, cases, strict=True):
        evidence = json.loads(path.read_text())
        assert evidence["requested_body"] == request(prompt)
        assert evidence["body"]["max_tokens"] == allowance
        assert evidence["prompt_tokens"] == prompt
        assert evidence["dynamic_output"]["protocol_id"] == "large-context-minus-prompt-20261127-v2"


def test_dynamic_relay_rejects_prompts_below_floor_without_dispatch_or_truncation(monkeypatch, tmp_path):
    def unexpected_response(body):
        raise AssertionError("An unserved prompt reached the model")

    with relay(monkeypatch, tmp_path, unexpected_response) as (client, upstream):
        for prompt in [49089, 65000]:
            got = client.post("/task/v1/chat/completions", json=request(prompt))
            assert got.status_code == 400
            assert got.json()["error"]["code"] == "input_budget_unserved"
        assert not upstream
    archived = sorted(tmp_path.glob("*/*/request.json"))
    assert [json.loads(path.read_text())["requested_body"] for path in archived] == [request(49089), request(65000)]
    assert not list(tmp_path.glob("*/*/upstream-response.body"))


@pytest.mark.parametrize(
    "violation", ["prompt_count", "output_count", "short_length", "missing_usage", "missing_choices"]
)
def test_dynamic_relay_refuses_server_usage_that_cannot_verify_the_allowance(monkeypatch, tmp_path, violation):
    def response(body):
        result: dict[str, Any] = {
            "choices": [{"finish_reason": "length" if violation == "short_length" else "stop"}],
            "usage": {
                "prompt_tokens": 101 if violation == "prompt_count" else 100,
                "completion_tokens": 65373 if violation == "output_count" else 4,
            },
        }
        if violation == "missing_usage":
            del result["usage"]
        if violation == "missing_choices":
            result["choices"] = None
        return result

    with relay(monkeypatch, tmp_path, response) as (client, upstream):
        got = client.post("/task/v1/chat/completions", json=request(100))
        assert got.status_code == 502
        assert got.json()["error"]["type"] == "ProtocolUsageViolation"
        assert len(upstream) == 1
    original = json.loads(next(tmp_path.glob("*/*/upstream-response.body")).read_bytes())
    assert original == response({})
    assert json.loads(next(tmp_path.glob("*/*/response.json")).read_text())["status"] == 502


def test_fixed_protocol_keeps_the_guest_cap_and_original_receipt_format(monkeypatch, tmp_path):
    def response(body):
        return {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 100, "completion_tokens": 4}}

    with relay(monkeypatch, tmp_path, response, dynamic=False) as (client, upstream):
        assert client.post("/task/v1/chat/completions", json=request(100)).status_code == 200
        assert upstream[0]["max_tokens"] == 7
    evidence = json.loads(next(tmp_path.glob("*/*/request.json")).read_text())
    assert "dynamic_output" not in evidence
