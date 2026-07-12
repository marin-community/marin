# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared inference serving types and payload helpers."""

import json
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

import zstandard

# Keep logs readable when a poll returns many request IDs.
_REQUEST_ID_LOG_LIMIT = 4


def pack_json_payload(payload: Mapping[str, Any]) -> bytes:
    return zstandard.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def unpack_json_payload(payload: bytes) -> dict[str, Any]:
    return json.loads(zstandard.decompress(payload).decode("utf-8"))


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


@dataclass(frozen=True)
class InferenceResponse:
    request_id: str
    status_code: int
    payload: bytes


@dataclass(frozen=True)
class LeasedInferenceRequest:
    lease_id: str
    request: InferenceRequest


@dataclass(frozen=True)
class LeasedInferenceResponse:
    lease_id: str
    response: InferenceResponse


@dataclass(frozen=True)
class ModelDeployment:
    """Configuration needed by a launcher to serve a model artifact."""

    model_name: str
    model_path: str
    tokenizer: str | None = None
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)


class ModelLauncher(Protocol):
    """Launch a model and own its serving lifecycle."""

    def launch(self, deployment: ModelDeployment) -> AbstractContextManager[RunningModel]:
        """Return a context manager that yields a running served model."""
        ...


# Brokers route payloads only; callers own broker and worker lifecycle.
class InferenceRequestProvider(Protocol):
    def fetch_requests(self, *, max_items: int) -> list[LeasedInferenceRequest]: ...

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None: ...


class InferenceResponseProvider(Protocol):
    def submit_request(self, request: InferenceRequest) -> None: ...

    def fetch_responses(self, *, max_items: int) -> list[InferenceResponse]: ...
