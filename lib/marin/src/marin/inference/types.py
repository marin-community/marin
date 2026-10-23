# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared inference serving types."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

# Keep logs readable when a poll returns many request IDs.
_REQUEST_ID_LOG_LIMIT = 4
STREAM_BUFFER_BYTES = 4 * 1024 * 1024
METADATA_REQUEST_SLOTS = 4
WORKER_HEARTBEAT_SECONDS = 5.0
WORKER_LEASE_SECONDS = 3 * WORKER_HEARTBEAT_SECONDS


class InferenceBackendState(StrEnum):
    READY = "ready"
    RECOVERING = "recovering"
    FINISHED = "finished"


class InferenceRequestKind(StrEnum):
    METADATA = "metadata"
    GENERATION = "generation"


def format_request_ids(ids: list[str]) -> str:
    if not ids:
        return "-"
    if len(ids) <= _REQUEST_ID_LOG_LIMIT:
        return ",".join(ids)
    return f"{','.join(ids[:_REQUEST_ID_LOG_LIMIT])},...(+{len(ids) - _REQUEST_ID_LOG_LIMIT})"


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str = field(repr=False)
    model: str
    api_key: str | None = field(default=None, repr=False)

    def url(self, path: str) -> str:
        """Return an endpoint URL under the API root."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def server_url(self, path: str) -> str:
        """Return an endpoint URL relative to the inference server root."""
        server_root = self.base_url.rstrip("/").removesuffix("/v1")
        return f"{server_root}/{path.lstrip('/')}"


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
    streaming: bool = False

    @property
    def kind(self) -> InferenceRequestKind:
        match self.method, self.path:
            case ("GET", "/v1/models" | "/metrics") | ("POST", "/tokenize" | "/detokenize"):
                return InferenceRequestKind.METADATA
            case _:
                return InferenceRequestKind.GENERATION


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
class StreamStart:
    request_id: str
    lease_id: str
    sequence: int
    status_code: int
    headers: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class StreamChunk:
    request_id: str
    lease_id: str
    sequence: int
    payload: bytes


@dataclass(frozen=True)
class StreamEnd:
    request_id: str
    lease_id: str
    sequence: int


@dataclass(frozen=True)
class StreamError:
    request_id: str
    lease_id: str
    sequence: int
    message: str


type StreamEvent = StreamStart | StreamChunk | StreamEnd | StreamError
type BrokerResponse = InferenceResponse | StreamEvent


@dataclass(frozen=True)
class InferenceWorkerMetadata:
    tensor_parallel_size: int
    backend_name: str


@dataclass(frozen=True)
class BrokerStatus:
    queued_requests: int
    leased_requests: int
    buffered_responses: int
    submitted_requests: int
    dispatched_requests: int
    completed_requests: int
    failed_responses: int
    dropped_responses: int
    mean_queue_wait_seconds: float
    max_queue_wait_seconds: float
    workers: dict[str, InferenceWorkerMetadata]
    readiness: InferenceBackendState
    recovery_count: int


# Brokers route payloads only; callers own broker and worker lifecycle.
class InferenceRequestProvider(Protocol):
    def fetch_requests(
        self, *, max_items: int, kind: InferenceRequestKind | None = None
    ) -> list[LeasedInferenceRequest]: ...

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None: ...

    def register_worker(self, worker_id: str, metadata: InferenceWorkerMetadata) -> None: ...

    def worker_metadata(self) -> dict[str, InferenceWorkerMetadata]: ...

    def heartbeat(self, worker_id: str) -> None:
        """Renew liveness using the identity supplied to register_worker."""
        ...

    def submit_stream_event(self, event: StreamEvent) -> bool:
        """Submit an event for the current request lease; stop forwarding on False.

        Start at sequence zero with StreamStart, then increment the sequence for
        each chunk and terminal event. Before StreamStart is accepted, report
        errors through submit_responses; afterward, send StreamError.
        """
        ...

    def finish(self) -> None: ...


class InferenceResponseProvider(Protocol):
    def submit_request(self, request: InferenceRequest) -> None: ...

    def fetch_responses(self, *, max_items: int) -> list[BrokerResponse]: ...

    def cancel_request(self, request_id: str) -> None: ...

    def status(self) -> BrokerStatus: ...
