# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit model batches and retain the provider request and response evidence."""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue
from zephyr.writers import write_jsonl_file

from taskcompendium.harbor.protocol import assistant_message
from taskcompendium.models import AssistantToolCalls
from taskcompendium.pipeline.models import ReviewStatus


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
    """Parse one complete tool call per request, retaining transport failures."""
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
                records.append(BatchToolArguments(task_id, ReviewStatus.UNAVAILABLE, None, "Provider request failed"))
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
    def submit(self, requests: Sequence[Mapping[str, Any]], filename: str) -> BatchSubmission: ...

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]: ...

    def output(self, batch: Mapping[str, Any]) -> BatchOutput: ...


def batch_output(
    client: BatchClient, requests: Sequence[Mapping[str, Any]], output_path: Path, *, filename: str, poll_seconds: float
) -> str:
    """Submit one model batch and save its request and response evidence."""
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl_file(requests, str(output_path / "requests.jsonl"))
    submission = client.submit(requests, filename)
    (output_path / "batch-submission.json").write_text(
        json.dumps({"file_id": submission.file_id, "batch_id": submission.batch_id})
    )
    batch = client.wait(submission.batch_id, poll_seconds)
    result = client.output(batch)
    if result.errors is not None:
        (output_path / "raw-errors.jsonl").write_text(result.errors)
    (output_path / "batch-result.json").write_text(json.dumps(batch, indent=2))
    (output_path / "raw-output.jsonl").write_text(result.output)
    return result.output
