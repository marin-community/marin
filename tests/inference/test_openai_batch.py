# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import Message
from email.utils import formatdate

import pytest
from marin.inference.openai_batch import BatchArtifactMissingError, BatchRequestError, OpenAIBatchClient
from marin.inference.structured_output import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from zephyr.counters import current_stage
from zephyr.runners import _InProcessWorkerContext
from zephyr.worker_context import _worker_ctx_var


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Answer(_StrictModel):
    value: int
    evidence: str = Field(min_length=1)


def _tool_response(name: str, arguments: object) -> dict:
    return {"choices": [{"message": {"tool_calls": [{"function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def test_structured_tool_schema_and_parse_share_the_pydantic_contract() -> None:
    tool = StructuredTool("submit_answer", "Submit the answer.", _Answer)

    function = tool.definition()["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
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


@pytest.fixture
def batch_counters():
    worker = _InProcessWorkerContext(chunk_prefix="", execution_id="", stage_name="review")
    token = _worker_ctx_var.set(worker)
    try:
        yield current_stage()
    finally:
        _worker_ctx_var.reset(token)


@pytest.mark.parametrize("failed_operation", ["upload", "create", "poll", "download"])
def test_openai_batch_retries_transient_errors_without_repeating_inference(
    monkeypatch, failed_operation, batch_counters
):
    failures = 2
    created_batches = []
    delays = []

    def urlopen(request, *, timeout):
        nonlocal failures
        url = request.full_url
        if "/files?" in url:
            operation, response = "upload", b'{"id":"file-1"}'
        elif url.endswith("/batches"):
            operation, response = "create", b'{"id":"batch-1"}'
        elif url.endswith("/batches/batch-1"):
            operation, response = "poll", b'{"status":"completed","output_file_id":"output-1"}'
        else:
            operation, response = "download", b'{"custom_id":"request-1"}\n'
        if operation == failed_operation and failures:
            failures -= 1
            status = 429 if operation == "create" else 503
            raise urllib.error.HTTPError(url, status, "Temporarily unavailable", Message(), None)
        if operation == "create":
            created_batches.append(json.loads(request.data))
        return _Response(response)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", delays.append)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", request_attempts=3)
    submission = client.submit([{"custom_id": "request-1"}], "requests.jsonl")
    output = client.output(client.wait(submission.batch_id, poll_seconds=0))

    assert output.output == '{"custom_id":"request-1"}\n'
    assert len(created_batches) == 1
    assert len(delays) == 2
    observed = batch_counters.get_counters()
    assert observed == {
        "batch_api/request_attempts": 6,
        "batch_api/retries": 2,
        "batch_api/upload/request_attempts": 3 if failed_operation == "upload" else 1,
        "batch_api/create/request_attempts": 3 if failed_operation == "create" else 1,
        "batch_api/poll/request_attempts": 3 if failed_operation == "poll" else 1,
        "batch_api/download/request_attempts": 3 if failed_operation == "download" else 1,
        f"batch_api/{failed_operation}/retries": 2,
    }


@pytest.mark.parametrize("status,expected_attempts", [(503, 3), (401, 1)])
def test_openai_batch_stops_after_retry_budget_or_permanent_failure(monkeypatch, status, expected_attempts):
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        raise urllib.error.HTTPError(request.full_url, status, "provider failure", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", request_attempts=3)
    error_type = BatchRequestError if status == 503 else urllib.error.HTTPError
    with pytest.raises(error_type):
        client.wait("batch-1", poll_seconds=0)
    assert len(requests) == expected_attempts


def test_openai_batch_does_not_repeat_ambiguous_batch_creation(monkeypatch):
    created_batches = []

    def urlopen(request, *, timeout):
        if "/files?" in request.full_url:
            return _Response(b'{"id":"file-1"}')
        created_batches.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 503, "response lost", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret")
    with pytest.raises(BatchRequestError):
        client.submit([{"custom_id": "request-1"}], "requests.jsonl")
    assert len(created_batches) == 1


def test_failed_upload_exhausts_http_budget_without_creating_inference(monkeypatch):
    calls = []
    fail_upload = True

    def urlopen(request, *, timeout):
        calls.append(request.full_url)
        if "/files?" in request.full_url:
            if fail_upload:
                raise TimeoutError("upload timed out")
            return _Response(b'{"id":"file-recovered"}')
        assert request.full_url.endswith("/batches")
        assert json.loads(request.data)["input_file_id"] == "file-recovered"
        return _Response(b'{"id":"batch-recovered"}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", lambda _delay: None)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", request_attempts=3)
    with pytest.raises(BatchRequestError):
        client.upload([{"custom_id": "request-1"}], "requests.jsonl")
    assert len(calls) == 3 and all("/files?" in url for url in calls)
    fail_upload = False
    submission = client.create(client.upload([{"custom_id": "request-1"}], "requests.jsonl"))
    assert submission.batch_id == "batch-recovered"
    assert sum(url.endswith("/batches") for url in calls) == 1


@pytest.mark.parametrize(
    "operation,path", [("poll", "batches/batch-missing"), ("download", "files/file-missing/content")]
)
def test_missing_provider_artifact_is_typed_without_resubmission(monkeypatch, operation, path):
    requests = []
    token = "private-provider-token"

    def urlopen(request, *, timeout):
        requests.append(request)
        body = f"batch disappeared; credential={token}; ".encode() + b"x" * 3000 + b"OMITTED_TAIL"
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", token)
    with pytest.raises(BatchArtifactMissingError) as failure:
        if operation == "poll":
            client.wait("batch-missing", poll_seconds=0)
        else:
            client.output({"status": "completed", "output_file_id": "file-missing"})

    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert failure.value.status_code == 404
    assert failure.value.filename == path
    evidence = str(failure.value)
    assert path in evidence
    assert "batch disappeared" in evidence
    assert "[REDACTED]" in evidence
    assert token not in evidence
    assert "OMITTED_TAIL" not in evidence
    assert len(evidence) < 2600


@pytest.mark.parametrize("missing_endpoint", ["files", "batches"])
def test_missing_submission_endpoint_remains_a_permanent_error(monkeypatch, missing_endpoint):
    requests = []

    def urlopen(request, *, timeout):
        requests.append(request)
        if "/files?" in request.full_url and missing_endpoint != "files":
            return _Response(b'{"id":"file-1"}')
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret")
    with pytest.raises(urllib.error.HTTPError):
        client.submit([{"custom_id": "request-1"}], "requests.jsonl")
    assert len(requests) == (1 if missing_endpoint == "files" else 2)


@pytest.mark.parametrize("retry_after", ["7", formatdate(1_700_000_007, usegmt=True)])
def test_openai_batch_honors_retry_after_and_redacts_recovery_evidence(monkeypatch, caplog, retry_after):
    requests = []
    delays = []
    token = "secret-provider-token"

    def urlopen(request, *, timeout):
        requests.append(request)
        if len(requests) == 1:
            headers = Message()
            headers["Retry-After"] = retry_after
            body = f"upstream storage busy; credential={token}; ".encode() + b"x" * 3000 + b"OMITTED_TAIL"
            raise urllib.error.HTTPError(request.full_url, 503, "temporarily unavailable", headers, io.BytesIO(body))
        return _Response(b'{"status":"completed"}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    monkeypatch.setattr(time, "sleep", delays.append)
    client = OpenAIBatchClient("https://inference.example/v1", token, request_attempts=2)

    assert client.wait("batch-1", poll_seconds=0)["status"] == "completed"
    assert len(requests) == 2
    assert delays == [7]
    assert "upstream storage busy" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert token not in caplog.text
    assert "OMITTED_TAIL" not in caplog.text


@pytest.mark.parametrize("retry_after,expected_attempts", [("1", 3), ("invalid", 3), ("120", 1)])
def test_openai_batch_retains_bounded_failure_evidence_without_extra_attempts(
    monkeypatch, caplog, retry_after, expected_attempts
):
    requests = []
    delays = []
    token = "secret-provider-token"

    def urlopen(request, *, timeout):
        requests.append(request)
        headers = Message()
        headers["Retry-After"] = retry_after
        body = b"upstream queue full; " + b"x" * 2018 + token.encode() + b"z" * 4000 + b"OMITTED_TAIL"
        raise urllib.error.HTTPError(request.full_url, 503, f"failure echo {token}", headers, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", delays.append)
    client = OpenAIBatchClient("https://inference.example/v1", token, request_attempts=3)
    with pytest.raises(BatchRequestError) as failure:
        client.wait("batch-1", poll_seconds=0)

    evidence = str(failure.value) + caplog.text
    assert len(requests) == expected_attempts
    assert len(delays) == expected_attempts - 1
    assert "upstream queue full" in evidence
    assert token not in evidence
    assert "secret-pro" not in evidence
    assert "OMITTED_TAIL" not in evidence
    assert len(str(failure.value)) < 2600


@dataclass
class _Clock:
    now: float = 0
    sleeps: list[float] = field(default_factory=list)

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


def test_openai_batch_wait_deadline_stops_stalled_finalization_and_clamps_http_timeout(monkeypatch, clock):
    timeouts = []

    def urlopen(request, *, timeout):
        timeouts.append(timeout)
        clock.now += 3
        return _Response(
            b'{"status":"in_progress","request_counts":{"total":64,"completed":64,"failed":0},"output_file_id":null}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", timeout=60, batch_wait_timeout=10)
    with pytest.raises(TimeoutError, match="deadline"):
        client.wait("batch-1", poll_seconds=4)

    assert clock.now == 10
    assert timeouts == [10, 3]
    assert clock.sleeps == [4]


@pytest.mark.parametrize("failure", ["http", "socket_timeout"])
def test_openai_batch_wait_deadline_bounds_http_retries_and_retry_after(monkeypatch, clock, failure):
    timeouts = []

    def urlopen(request, *, timeout):
        timeouts.append(timeout)
        if failure == "socket_timeout":
            clock.now += timeout
            raise TimeoutError("Socket timeout")
        clock.now += 2
        headers = Message()
        headers["Retry-After"] = "20"
        raise urllib.error.HTTPError(request.full_url, 503, "busy", headers, io.BytesIO(b"queue busy"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret", timeout=60, batch_wait_timeout=5)
    with pytest.raises(TimeoutError, match="deadline"):
        client.wait("batch-1", poll_seconds=1)

    assert clock.now == 5
    assert timeouts == [5]
    assert clock.sleeps == ([3] if failure == "http" else [])


def test_missing_error_file_preserves_downloaded_completions(monkeypatch):
    output = '{"custom_id":"kept","response":{"status_code":200}}\n'

    def urlopen(request, *, timeout):
        if request.full_url.endswith("/files/output/content"):
            return _Response(output.encode())
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", Message(), io.BytesIO(b"file not found"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = OpenAIBatchClient("https://inference.example/v1", "secret")
    result = client.output({"status": "completed", "output_file_id": "output", "error_file_id": "lost"})
    assert result.output == output
    assert json.loads(result.errors)["transport_error"]["code"] == "missing_error_file"
