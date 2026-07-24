# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared inference serving types."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

# Keep logs readable when a poll returns many request IDs.
_REQUEST_ID_LOG_LIMIT = 4


def format_request_ids(ids: list[str]) -> str:
    if not ids:
        return "-"
    if len(ids) <= _REQUEST_ID_LOG_LIMIT:
        return ",".join(ids)
    return f"{','.join(ids[:_REQUEST_ID_LOG_LIMIT])},...(+{len(ids) - _REQUEST_ID_LOG_LIMIT})"


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str
    model: str
    api_key: str | None = None

    def url(self, path: str) -> str:
        """Return an endpoint URL under the API root."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True)
class RunningModel:
    """A model that is already being served by a launcher-owned backend."""

    endpoint: OpenAIEndpoint
    tokenizer: str | None = None


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    method: str
    path: str
    payload: bytes
    query_string: str = ""
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    status_code: int
    payload: bytes
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LeasedInferenceRequest:
    lease_id: str
    request: InferenceRequest


@dataclass(frozen=True)
class LeasedInferenceResponse:
    lease_id: str
    response: InferenceResponse


@dataclass(frozen=True)
class InferenceWorkerMetadata:
    tensor_parallel_size: int
    backend_name: str


# Brokers route payloads only; callers own broker and worker lifecycle.
class InferenceRequestProvider(Protocol):
    def fetch_requests(self, *, max_items: int) -> list[LeasedInferenceRequest]: ...

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None: ...

    def register_worker(self, worker_id: str, metadata: InferenceWorkerMetadata) -> None: ...

    def worker_metadata(self) -> dict[str, InferenceWorkerMetadata]: ...


class InferenceResponseProvider(Protocol):
    def submit_request(self, request: InferenceRequest) -> None: ...

    def fetch_responses(self, *, max_items: int) -> list[InferenceResponse]: ...
