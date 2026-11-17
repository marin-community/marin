# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal client for the OpenAI-compatible file and batch APIs."""

from __future__ import annotations

import errno
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

from rigging.timing import Deadline, ExponentialBackoff
from zephyr.counters import current_stage

logger = logging.getLogger(__name__)

TERMINAL_BATCH_STATES = frozenset({"completed", "failed", "expired", "cancelled"})
CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_ERROR_BODY_BYTES = 2048
MAX_RETRY_AFTER = 60
DEFAULT_BATCH_WAIT_TIMEOUT = 3600


class BatchOperation(StrEnum):
    UPLOAD = "upload"
    CREATE = "create"
    POLL = "poll"
    DOWNLOAD = "download"


def _retry_after_delay(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = int(value)
        return float(seconds) if seconds >= 0 else None
    except (ValueError, OverflowError):
        try:
            requested = parsedate_to_datetime(value)
            if requested.tzinfo is None:
                return None
            return max(0.0, requested.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


class BatchRequestError(ConnectionError):
    """A provider failure left a batch operation without a usable result."""


class BatchArtifactMissingError(FileNotFoundError):
    """A batch or output-file GET returned a confirmed HTTP 404.

    ``filename`` identifies the provider resource; the message retains bounded,
    redacted response evidence. This never describes an ambiguous creation POST.
    """

    status_code = 404


def _missing_batch_artifact(path: str, request_method: str, status_code: int) -> bool:
    if status_code != 404 or request_method != "GET":
        return False
    parts = path.split("/")
    return (len(parts) == 2 and parts[0] == "batches" and bool(parts[1])) or (
        len(parts) == 3 and parts[0] == "files" and bool(parts[1]) and parts[2] == "content"
    )


def _http_error_details(error: urllib.error.HTTPError, token: str) -> tuple[str, float | None]:
    """Retain bounded provider evidence after redacting credentials."""
    try:
        body = error.read(MAX_ERROR_BODY_BYTES + len(token.encode())).decode("utf-8", errors="replace")
        if token:
            body = body.replace(token, "[REDACTED]")
        body = body.encode("utf-8")[:MAX_ERROR_BODY_BYTES].decode("utf-8", errors="ignore")
        header = error.headers.get("Retry-After")
        detail = f"{error}; Retry-After={header[:256] if header is not None else None!r}; body={body!r}"
        return detail, _retry_after_delay(header)
    finally:
        error.close()


def jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
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
    request_attempts: int = 8
    batch_wait_timeout: float = DEFAULT_BATCH_WAIT_TIMEOUT

    def __post_init__(self) -> None:
        if self.request_attempts < 1:
            raise ValueError("At least one batch API request attempt is required")
        if self.timeout <= 0 or self.batch_wait_timeout <= 0:
            raise ValueError("Batch API timeouts must be positive")

    def _request(
        self,
        path: str,
        *,
        operation: BatchOperation,
        data: bytes | None = None,
        content_type: str | None = None,
        deadline: Deadline | None = None,
    ) -> bytes:
        headers = {"Authorization": f"Bearer {self.token}", "x-priority": self.priority}
        if content_type is not None:
            headers["Content-Type"] = content_type
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        backoff = ExponentialBackoff(initial=1, maximum=30, factor=2)
        stage = current_stage()
        for attempt in range(self.request_attempts):
            timeout = self.timeout
            if deadline is not None:
                deadline.raise_if_expired(f"Batch API deadline exceeded while requesting {path}")
                timeout = min(timeout, deadline.remaining_seconds())
            stage.update_counter("batch_api/request_attempts", 1)
            stage.update_counter(f"batch_api/{operation}/request_attempts", 1)
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result = response.read()
                if deadline is not None:
                    deadline.raise_if_expired(f"Batch API deadline exceeded while requesting {path}")
                return result
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if deadline is not None and deadline.expired():
                    raise TimeoutError(f"Batch API deadline exceeded while requesting {path}") from None
                # Re-uploading a file cannot repeat inference. A batch-creation POST
                # may have succeeded despite a lost response, so only retry its 429s.
                safe_to_repeat = data is None or path.startswith("files?")
                detail = str(error)
                retry_after = None
                missing_artifact = False
                if isinstance(error, urllib.error.HTTPError):
                    missing_artifact = _missing_batch_artifact(path, request.method, error.code)
                    if error.code not in RETRYABLE_HTTP_STATUSES and not missing_artifact:
                        raise
                    if missing_artifact:
                        # Preserve the submitted batch ID for recovery; polling a
                        # vanished batch or file cannot restore provider state.
                        safe_to_repeat = False
                    safe_to_repeat |= error.code == 429
                    detail, retry_after = _http_error_details(error, self.token)
                if self.token:
                    detail = detail.replace(self.token, "[REDACTED]")
                if missing_artifact:
                    stage.update_counter("batch_api/missing_artifacts", 1)
                    stage.update_counter(f"batch_api/{operation}/missing_artifacts", 1)
                    raise BatchArtifactMissingError(errno.ENOENT, detail, path) from None
                if retry_after is not None and retry_after > MAX_RETRY_AFTER:
                    safe_to_repeat = False
                    detail += f"; provider delay exceeds {MAX_RETRY_AFTER}s retry wait budget"
                if not safe_to_repeat or attempt + 1 == self.request_attempts:
                    stage.update_counter("batch_api/unavailable", 1)
                    stage.update_counter(f"batch_api/{operation}/unavailable", 1)
                    raise BatchRequestError(f"Batch API {request.method} {path} failed: {detail}") from None
                delay = backoff.next_interval()
                if retry_after is not None:
                    delay = max(delay, retry_after)
                if deadline is not None:
                    delay = min(delay, deadline.remaining_seconds())
                stage.update_counter("batch_api/retries", 1)
                stage.update_counter(f"batch_api/{operation}/retries", 1)
                logger.warning("Batch API %s %s failed (%s); retrying in %.1fs", request.method, path, detail, delay)
                time.sleep(delay)
        raise AssertionError("Batch API retry loop did not return or raise")

    def _request_json(
        self,
        path: str,
        *,
        operation: BatchOperation,
        body: Mapping[str, Any] | None = None,
        deadline: Deadline | None = None,
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
        raw = self._request(
            path,
            operation=operation,
            data=data,
            content_type="application/json" if data is not None else None,
            deadline=deadline,
        )
        return json.loads(raw)

    def upload(self, requests: Sequence[Mapping[str, Any]], filename: str) -> str:
        """Upload JSONL without starting inference, returning its provider file ID."""
        query = urllib.parse.urlencode({"purpose": "batch", "filename": filename})
        file_response = json.loads(
            self._request(
                f"files?{query}",
                operation=BatchOperation.UPLOAD,
                data=jsonl_text(requests).encode(),
                content_type="application/jsonl",
            )
        )
        return file_response["id"]

    def create(self, file_id: str, *, endpoint: str = CHAT_COMPLETIONS_ENDPOINT) -> BatchSubmission:
        """Start inference on an uploaded file; a lost response is ambiguous."""
        batch = self._request_json(
            "batches",
            operation=BatchOperation.CREATE,
            body={"input_file_id": file_id, "endpoint": endpoint, "priority": self.priority},
        )
        return BatchSubmission(file_id=file_id, batch_id=batch["id"])

    def submit(
        self,
        requests: Sequence[Mapping[str, Any]],
        filename: str,
        *,
        endpoint: str = CHAT_COMPLETIONS_ENDPOINT,
    ) -> BatchSubmission:
        """Upload and create one batch for callers without durable reservations."""
        return self.create(self.upload(requests, filename), endpoint=endpoint)

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]:
        """Poll for a terminal state within one deadline, including HTTP retries."""

        deadline = Deadline.from_seconds(self.batch_wait_timeout)
        last_counts = None
        while True:
            batch = self._request_json(f"batches/{batch_id}", operation=BatchOperation.POLL, deadline=deadline)
            counts = batch.get("request_counts")
            if counts != last_counts:
                logger.info("batch %s status=%s counts=%s", batch_id, batch.get("status"), counts)
                last_counts = counts
            if batch.get("status") in TERMINAL_BATCH_STATES:
                return batch
            time.sleep(min(poll_seconds, deadline.remaining_seconds()))

    def output(self, batch: Mapping[str, Any]) -> BatchOutput:
        """Download a batch's output and optional error files."""

        output_id = batch.get("output_file_id")
        output = (
            ""
            if not output_id
            else self._request(f"files/{output_id}/content", operation=BatchOperation.DOWNLOAD).decode()
        )
        error_id = batch.get("error_file_id")
        errors = None
        if error_id:
            try:
                errors = self._request(f"files/{error_id}/content", operation=BatchOperation.DOWNLOAD).decode()
            except BatchArtifactMissingError as error:
                if not output:
                    raise
                # Keep successful rows even when the provider loses its error
                # artifact. Missing rows remain unavailable to the caller.
                errors = json.dumps(
                    {
                        "transport_error": {
                            "code": "missing_error_file",
                            "resource": error.filename,
                            "status_code": error.status_code,
                            "detail": str(error),
                        }
                    }
                )
        return BatchOutput(output=output, errors=errors)
