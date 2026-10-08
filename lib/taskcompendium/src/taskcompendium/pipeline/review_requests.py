# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit model batches and retain the provider request and response evidence."""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue
from zephyr import counters
from zephyr.writers import write_jsonl_file

from taskcompendium.chat import assistant_message
from taskcompendium.models import AssistantToolCalls
from taskcompendium.pipeline.models import ReviewStatus

DEFAULT_MAX_BATCH_REQUESTS = 64
DEFAULT_MAX_BATCH_BYTES = 4 * 1024 * 1024
RAW_OUTPUT_FILENAME = "raw-output.jsonl"
REQUEST_EVIDENCE_FILENAME = "requests.jsonl"
SUBMISSION_EVIDENCE_FILENAME = "batch-submission.json"


@dataclass(frozen=True)
class BatchToolArguments:
    task_id: str
    status: ReviewStatus
    arguments: dict[str, JsonValue] | None
    detail: str


class TypedBatchValue(Protocol):
    @property
    def task_id(self) -> str: ...


@dataclass(frozen=True)
class TypedBatchRecord[T: TypedBatchValue]:
    task_id: str
    status: ReviewStatus
    value: T | None
    detail: str


def typed_batch_records[T: TypedBatchValue](
    output: str,
    task_ids: Sequence[str],
    *,
    tool_name: str,
    validate: Callable[[str], T],
    identity_error: str,
) -> list[TypedBatchRecord[T]]:
    """Validate structured values and identities while retaining protocol failures."""
    records = []
    for response in batch_tool_arguments(output, task_ids, tool_name=tool_name):
        task_id = response.task_id
        if response.status != ReviewStatus.REVIEWED:
            records.append(TypedBatchRecord(task_id, response.status, None, response.detail))
            continue
        try:
            assert response.arguments is not None
            value = validate(json.dumps(response.arguments))
            if value.task_id != task_id:
                raise ValueError(identity_error)
        except (KeyError, TypeError, ValueError) as error:
            records.append(TypedBatchRecord(task_id, ReviewStatus.INVALID, None, str(error)))
            continue
        records.append(TypedBatchRecord(task_id, ReviewStatus.REVIEWED, value, ""))
    return records


def batch_tool_arguments(output: str, task_ids: Sequence[str], *, tool_name: str) -> list[BatchToolArguments]:
    """Parse one complete tool call per request, retaining request failures."""
    expected = set(task_ids)
    responses: dict[str, list[dict[str, Any]]] = {}
    for line in output.split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        custom_id = row["custom_id"]
        if custom_id not in expected:
            raise ValueError(f"Unexpected batch response ID: {custom_id}")
        responses.setdefault(custom_id, []).append(row)

    records = []
    for task_id in task_ids:
        rows = responses.get(task_id, [])
        if not rows:
            records.append(BatchToolArguments(task_id, ReviewStatus.UNAVAILABLE, None, "Missing batch response"))
            continue
        try:
            if len(rows) != 1:
                raise ValueError("Duplicate batch response ID")
            response = rows[0].get("response")
            if response is None or response.get("status_code") != 200:
                detail = json.dumps(rows[0].get("error") or response or "Provider request failed", ensure_ascii=False)
                records.append(BatchToolArguments(task_id, ReviewStatus.UNAVAILABLE, None, detail))
                continue
            choices = response["body"]["choices"]
            # GLM's relay reports completed tool calls with finish_reason=stop.
            if len(choices) != 1 or choices[0]["finish_reason"] not in {"tool_calls", "stop"}:
                raise ValueError("Incomplete or truncated structured response")
            message = assistant_message(choices[0]["message"])
            if (
                not isinstance(message, AssistantToolCalls)
                or len(message.calls) != 1
                or message.calls[0].name != tool_name
            ):
                raise ValueError(f"Expected exactly one {tool_name} call")
        except (KeyError, TypeError, ValueError) as error:
            records.append(BatchToolArguments(task_id, ReviewStatus.INVALID, None, str(error)))
            continue
        records.append(BatchToolArguments(task_id, ReviewStatus.REVIEWED, message.calls[0].arguments, ""))
    return records


class BatchSubmission(Protocol):
    @property
    def file_id(self) -> str: ...

    @property
    def batch_id(self) -> str: ...


class BatchOutput(Protocol):
    @property
    def output(self) -> str: ...

    @property
    def errors(self) -> str | None: ...


class BatchClient(Protocol):
    """Provider operations; wait/output raise FileNotFoundError only for confirmed missing artifacts."""

    def upload(self, requests: Sequence[Mapping[str, Any]], filename: str) -> str: ...

    def create(self, file_id: str) -> BatchSubmission: ...

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]: ...

    def output(self, batch: Mapping[str, Any]) -> BatchOutput: ...


def request_batches[T: Mapping[str, Any]](
    requests: Sequence[T],
    *,
    max_requests: int,
    max_bytes: int,
    oversized: Callable[[T, int], None],
) -> list[list[T]]:
    """Pack indivisible UTF-8 JSONL requests, reporting oversized requests separately."""
    batches = []
    pending = []
    pending_bytes = 0
    for request in requests:
        request_bytes = len(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1
        if request_bytes > max_bytes:
            oversized(request, request_bytes)
            continue
        if pending and (len(pending) >= max_requests or pending_bytes + request_bytes > max_bytes):
            batches.append(pending)
            pending, pending_bytes = [], 0
        pending.append(request)
        pending_bytes += request_bytes
    if pending:
        batches.append(pending)
    return batches


def batch_output(
    client: BatchClient,
    requests: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    filename: str,
    poll_seconds: float,
    max_batch_requests: int = DEFAULT_MAX_BATCH_REQUESTS,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> str:
    """Split inference uploads by request count and UTF-8 JSONL byte size.

    A request is indivisible: one exceeding the upload budget is recorded as
    unavailable, while the other tasks continue. The reviewer owns retries.
    """
    if max_batch_requests < 1 or max_batch_bytes < 1:
        raise ValueError("Inference batch budgets must be positive")
    metrics = counters.current_stage()
    outputs = []

    def oversized(request: Mapping[str, Any], request_bytes: int) -> None:
        metrics.update_counter("review/requests/oversized_requests", 1)
        outputs.append(
            _unavailable_output(
                [request],
                "batch_request_too_large",
                f"Request requires {request_bytes} bytes; batch budget is {max_batch_bytes}",
            )
        )

    parts = request_batches(
        requests,
        max_requests=max_batch_requests,
        max_bytes=max_batch_bytes,
        oversized=oversized,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl_file(requests, str(output_path / REQUEST_EVIDENCE_FILENAME))
    for index, part in enumerate(parts):
        directory = output_path if len(parts) == 1 else output_path / f"part-{index:05d}"
        started = time.monotonic()
        try:
            outputs.append(
                _submitted_batch_output(
                    client,
                    part,
                    directory,
                    filename=filename,
                    poll_seconds=poll_seconds,
                )
            )
        except (ConnectionError, TimeoutError, FileNotFoundError) as error:
            metrics.update_counter("review/requests/unavailable_requests", len(part))
            failure = _unavailable_output(part, "batch_request_failed", str(error))
            (directory / RAW_OUTPUT_FILENAME).write_text(failure)
            outputs.append(failure)
        finally:
            metrics.update_counter("review/requests/provider_seconds", time.monotonic() - started)
    output = "\n".join(part.rstrip("\n") for part in outputs if part)
    (output_path / RAW_OUTPUT_FILENAME).write_text(output)
    return output


def _unavailable_output(requests: Sequence[Mapping[str, Any]], code: str, message: str) -> str:
    return "\n".join(
        json.dumps({"custom_id": request["custom_id"], "error": {"code": code, "message": message}})
        for request in requests
    )


def _submitted_batch_output(
    client: BatchClient,
    requests: Sequence[Mapping[str, Any]],
    output_path: Path,
    *,
    filename: str,
    poll_seconds: float,
) -> str:
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl_file(requests, str(output_path / REQUEST_EVIDENCE_FILENAME))
    file_id = client.upload(requests, filename)
    submission = client.create(file_id)
    (output_path / SUBMISSION_EVIDENCE_FILENAME).write_text(
        json.dumps({"file_id": submission.file_id, "batch_id": submission.batch_id}, indent=2)
    )
    metrics = counters.current_stage()
    metrics.update_counter("review/requests/submitted_batches", 1)
    metrics.update_counter("review/requests/submitted_requests", len(requests))
    metrics.update_counter(
        "review/requests/submitted_bytes",
        sum(len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode()) + 1 for row in requests),
    )
    batch = client.wait(submission.batch_id, poll_seconds)
    (output_path / "batch-result.json").write_text(json.dumps(batch, indent=2))
    result = client.output(batch)
    if result.errors is not None:
        (output_path / "raw-errors.jsonl").write_text(result.errors)
    (output_path / RAW_OUTPUT_FILENAME).write_text(result.output)
    return result.output
