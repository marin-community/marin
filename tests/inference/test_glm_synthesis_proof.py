# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import httpx
import pytest
from rigging.filesystem.storage_path import StoragePath

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
            "choices": [{"delta": {"content": content, "reasoning": "fixture thought"}, "finish_reason": finish}],
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
    if status == 200 and content is not None:
        assert outcomes[0].stream_done is done
        assert outcomes[0].content_characters == len(content)
        assert outcomes[0].reasoning_characters == len("fixture thought")
        assert outcomes[0].request_index == 0


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


def test_published_manifest_keeps_credentials_in_references_and_exposes_capacity(tmp_path):
    (tmp_path / "credentials").mkdir()
    target = prove_replicas.Target("/serve/fixture", "http://fixture/v1", "fixture-endpoint-token", 12345)
    prove_replicas.publish_manifest(StoragePath(str(tmp_path)), [target], concurrency_per_replica=24)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest == [{"base_url": "http://fixture/v1", "api_key_ref": str(tmp_path / "credentials/fixture.jwt")}]
    assert (tmp_path / "credentials/fixture.jwt").read_text() == "fixture-endpoint-token"
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["healthy_entry_count"] * status["per_engine_concurrency"] == status["total_concurrency"] == 24
