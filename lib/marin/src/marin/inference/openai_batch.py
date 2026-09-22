# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal client for the OpenAI-compatible file and batch APIs."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL_BATCH_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


def jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize JSON objects as compact newline-delimited text."""

    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


@dataclass(frozen=True)
class BatchSubmission:
    file_id: str
    batch_id: str


@dataclass(frozen=True)
class BatchOutput:
    output: str
    errors: str | None


@dataclass(frozen=True)
class OpenAIBatchClient:
    """OpenAI-compatible batch transport with no provider-specific discovery or credentials."""

    base_url: str
    token: str = field(repr=False)
    priority: str = "bulk"
    timeout: float = 600

    def _request(self, path: str, *, data: bytes | None = None, content_type: str | None = None) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}", "x-priority": self.priority}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()

    def _request_json(self, path: str, *, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        raw = self._request(path, data=data, content_type="application/json" if data is not None else None)
        return json.loads(raw)

    def submit(
        self,
        requests: Sequence[Mapping[str, Any]],
        filename: str,
        *,
        endpoint: str = "/v1/chat/completions",
    ) -> BatchSubmission:
        """Upload request JSONL and create one batch."""

        query = urllib.parse.urlencode({"purpose": "batch", "filename": filename})
        file_response = json.loads(
            self._request(
                f"files?{query}",
                data=jsonl_text(requests).encode(),
                content_type="application/jsonl",
            )
        )
        batch = self._request_json(
            "batches",
            body={"input_file_id": file_response["id"], "endpoint": endpoint, "priority": self.priority},
        )
        return BatchSubmission(file_id=file_response["id"], batch_id=batch["id"])

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]:
        """Wait for a batch to reach an OpenAI terminal state."""

        last_counts = None
        while True:
            batch = self._request_json(f"batches/{batch_id}")
            counts = batch.get("request_counts")
            if counts != last_counts:
                logger.info("batch %s status=%s counts=%s", batch_id, batch.get("status"), counts)
                last_counts = counts
            if batch.get("status") in TERMINAL_BATCH_STATES:
                return batch
            time.sleep(poll_seconds)

    def output(self, batch: Mapping[str, Any]) -> BatchOutput:
        """Download a batch's output and optional error files."""

        output_id = batch.get("output_file_id")
        output = "" if not output_id else self._request(f"files/{output_id}/content").decode()
        error_id = batch.get("error_file_id")
        errors = None if not error_id else self._request(f"files/{error_id}/content").decode()
        return BatchOutput(output=output, errors=errors)
