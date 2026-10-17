# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import requests
from marin.inference.types import OpenAIEndpoint
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.collect_teacher import collect_one


@pytest.mark.parametrize("status", [200, 503])
def test_collection_preserves_native_response_and_never_resamples_existing_record(tmp_path, monkeypatch, status):
    prompt = {
        "custom_id": "input-0",
        "source_id": "sample_0",
        "task": "input",
        "b_request": {"messages": [{"role": "user", "content": "Find an input"}]},
    }
    payload = {"choices": [{"message": {"content": "reasoning<|end_think|>answer"}, "token_ids": [1, 128003, 2]}]}
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()

    def post(url, *, json, headers, timeout):
        assert json["messages"] == prompt["b_request"]["messages"]
        assert json["skip_special_tokens"] is False
        assert json["return_token_ids"] is True
        return response

    monkeypatch.setattr(requests, "post", post)
    endpoint = OpenAIEndpoint("http://unused/v1", "teacher")
    destination = StoragePath(str(tmp_path))
    (destination / "responses").mkdirs()
    if status == 200:
        collect_one(endpoint, destination, (prompt, 0))
    else:
        with pytest.raises(requests.HTTPError):
            collect_one(endpoint, destination, (prompt, 0))
    saved = json.loads(next((tmp_path / "responses").glob("*.json")).read_text())
    assert saved["status_code"] == status
    assert json.loads(saved["response_text"]) == payload
    if status == 200:
        assert saved["response"] == payload

    def forbidden_post(*args, **kwargs):
        pytest.fail("A recorded sample must not be silently replaced")

    monkeypatch.setattr(requests, "post", forbidden_post)
    if status == 200:
        collect_one(endpoint, destination, (prompt, 0))
    else:
        with pytest.raises(ValueError):
            collect_one(endpoint, destination, (prompt, 0))
