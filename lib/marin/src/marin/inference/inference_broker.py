# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Shared broker types for inference request/response routing.

Lifecycle contract:
The client task owns broker and worker lifetimes. In Iris, broker/workers should
be spawned as children of the client task; Iris tears them down when the client
exits. The broker only moves request/response payloads and does not own run
shutdown.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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


class InferenceBroker(Protocol):
    def submit_request(self, request: InferenceRequest) -> None: ...

    def fetch_requests(self, *, max_items: int) -> list[LeasedInferenceRequest]: ...

    def submit_responses(self, responses: Iterable[LeasedInferenceResponse]) -> None: ...

    def fetch_responses(self, *, max_items: int) -> list[InferenceResponse]: ...
