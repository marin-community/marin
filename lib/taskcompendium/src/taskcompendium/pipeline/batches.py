# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist requests, acknowledged submissions, and raw outputs for resumable batches."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


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
    """Resume the exact request batch after an acknowledged submission."""
    output_path.mkdir(parents=True, exist_ok=True)
    request_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests)
    requests_path = output_path / "requests.jsonl"
    if requests_path.exists() and requests_path.read_text() != request_text:
        raise ValueError("Saved batch requests differ from this run")
    requests_path.write_text(request_text)
    state_path = output_path / "batch-state.json"
    if state_path.exists():
        batch_id = json.loads(state_path.read_text())["batch_id"]
    else:
        submission = client.submit(requests, filename)
        state_path.write_text(json.dumps({"file_id": submission.file_id, "batch_id": submission.batch_id}))
        batch_id = submission.batch_id
    raw_path = output_path / "raw-output.jsonl"
    if raw_path.exists():
        raw_output = raw_path.read_text()
    else:
        batch = client.wait(batch_id, poll_seconds)
        result = client.output(batch)
        raw_output = result.output
        if result.errors is not None:
            (output_path / "raw-errors.jsonl").write_text(result.errors)
        (output_path / "batch-result.json").write_text(json.dumps(batch, indent=2))
        raw_path.write_text(raw_output)
    return raw_output
