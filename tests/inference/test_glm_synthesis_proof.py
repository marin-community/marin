# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
import pytest

from experiments.glm_synthesis_serving import prove_replicas


@pytest.mark.parametrize(
    "status,content,finish,done,expected",
    [
        (200, '{"actions": []}', "stop", True, True),
        (401, "", "stop", True, False),
        (200, '{"actions": []}', "stop", False, False),
        (200, '{"actions": []}', "length", True, False),
        (200, "", "stop", True, False),
        (200, "invalid json", "stop", True, False),
        (200, None, "stop", True, False),
    ],
)
def test_real_request_proof_requires_authorized_complete_json(monkeypatch, status, content, finish, done, expected):
    bodies = [{"model": "glm-5.3", "messages": [{"role": "user", "content": "fixture"}]}]
    observed = []

    def handle(request):
        assert request.headers["Authorization"] == "Bearer fixture-endpoint-token"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "glm-5.3"}]})
        observed.append(json.loads(request.content))
        event = {
            "choices": [{"delta": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        if content is None:
            event = {"error": {"message": "fixture engine fault"}}
        stream = "data: " + json.dumps(event) + "\n\n"
        if done:
            stream += "data: [DONE]\n\n"
        return httpx.Response(status, text=stream)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        prove_replicas.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    target = prove_replicas.Target("/serve/fixture", "http://fixture/v1", "fixture-endpoint-token", 0)
    outcomes = asyncio.run(prove_replicas.prove([target], bodies, concurrency=1))
    assert observed == bodies
    assert outcomes[0].ok is expected


def test_sustained_load_stops_after_engine_error(monkeypatch):
    def handle(request):
        return httpx.Response(200, text='data: {"error": {"message": "fixture engine fault"}}\n\n')

    original = httpx.AsyncClient
    monkeypatch.setattr(
        prove_replicas.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs),
    )
    target = prove_replicas.Target("/serve/fixture", "http://fixture/v1", "fixture-endpoint-token", 0)
    outcomes = asyncio.run(
        prove_replicas.sustained_load([target], [{"cache_salt": "fixture"}], concurrency_per_replica=1, duration=900)
    )
    assert len(outcomes) == 1
    assert outcomes[0].state is prove_replicas.RequestState.STREAM_ERROR
