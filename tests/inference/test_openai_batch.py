# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import urllib.request

import pytest
from marin.inference.openai_batch import OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Answer(_StrictModel):
    value: int
    evidence: str = Field(min_length=1)


def _tool_response(name: str, arguments: object) -> dict:
    return {"choices": [{"message": {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def test_structured_tool_schema_and_parse_share_the_pydantic_contract() -> None:
    tool = StructuredTool("submit_answer", "Submit the answer.", _Answer)

    parameters = tool.definition()["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["additionalProperties"] is False
    assert set(parameters["required"]) == {"value", "evidence"}
    assert parameters["properties"]["value"]["type"] == "integer"
    assert parameters["properties"]["evidence"]["minLength"] == 1
    assert tool.parse(_tool_response("submit_answer", {"value": 7, "evidence": "fact-1"})) == _Answer(
        value=7, evidence="fact-1"
    )
    with pytest.raises(ValidationError):
        tool.parse(_tool_response("submit_answer", {"value": 7, "evidence": "", "extra": True}))


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.body


def test_openai_batch_client_round_trips_batch_wire_protocol(monkeypatch) -> None:
    requests: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, *, timeout: float):
        assert timeout == 12
        requests.append(request)
        url = request.full_url
        if "/files?" in url:
            return _Response(b'{"id":"file-1"}')
        if url.endswith("/batches"):
            return _Response(b'{"id":"batch-1"}')
        if url.endswith("/batches/batch-1"):
            return _Response(b'{"status":"completed","output_file_id":"output-1","error_file_id":null}')
        if url.endswith("/files/output-1/content"):
            return _Response(b'{"custom_id":"request-1"}\n')
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", timeout=12)
    assert "secret" not in repr(client)
    submission = client.submit(
        [{"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {}}],
        "requests.jsonl",
    )
    batch = client.wait(submission.batch_id, poll_seconds=0)
    output = client.output(batch)

    assert submission.file_id == "file-1"
    assert submission.batch_id == "batch-1"
    assert output.output == '{"custom_id":"request-1"}\n'
    assert output.errors is None
    create_body = json.loads(requests[1].data)
    assert create_body == {
        "input_file_id": "file-1",
        "endpoint": "/v1/chat/completions",
        "priority": "bulk",
    }
