# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IPC protocol for packed vLLM rollout child processes."""

from __future__ import annotations

import dataclasses
import socket
import struct
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
from marin.rl.environments.inference_ctx.base import InferenceRequestKind

_HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class PackedReplicaStatus:
    """Current state of one packed inference replica."""

    worker_index: int
    active_weight_id: int
    pending_weight_id: int | None
    busy: bool
    error: str | None = None
    total_weight_fetches: int = 0
    total_weight_activations: int = 0
    total_generate_requests: int = 0
    total_train_generate_requests: int = 0
    total_eval_generate_requests: int = 0
    total_micro_eval_generate_requests: int = 0
    last_generation_seconds: float | None = None
    transfer_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PackedChildInitRequest:
    """Initialize one packed replica."""

    inference_config: object
    weight_transfer_config: object
    coordinator_handle: object | None
    worker_index: int


@dataclass(frozen=True)
class PackedChildStatusRequest:
    """Fetch the latest child status."""


@dataclass(frozen=True)
class PackedChildActivateRequest:
    """Activate a previously fetched pending weight."""

    expected_weight_id: int


@dataclass(frozen=True)
class PackedChildGenerateRequest:
    """Run one batched generate call on a child replica."""

    request_id: str
    prompts: list[str] | list[list[dict[str, Any]]]
    request_kind: InferenceRequestKind
    temperature: float
    n: int
    max_tokens: int | None
    top_k: int | None
    stop: list[str] | None
    system_prompt: str | None
    expected_weight_id: int


@dataclass(frozen=True)
class PackedChildShutdownRequest:
    """Shutdown a child replica."""


@dataclass(frozen=True)
class PackedChildAckResponse:
    """Simple acknowledgement from a child replica."""

    status: PackedReplicaStatus


@dataclass(frozen=True)
class PackedChildStatusResponse:
    """Status response from a child replica."""

    status: PackedReplicaStatus


@dataclass(frozen=True)
class PackedChildActivateResponse:
    """Activation response from a child replica."""

    applied: bool
    status: PackedReplicaStatus


@dataclass(frozen=True)
class PackedChildGenerateResponse:
    """Generate response from a child replica."""

    completions: list[object]
    status: PackedReplicaStatus


@dataclass(frozen=True)
class PackedChildErrorResponse:
    """Structured child failure response."""

    message: str
    status: PackedReplicaStatus | None = None
    traceback_text: str | None = None


def send_packed_message(sock: socket.socket, payload: object) -> None:
    """Send one framed cloudpickle payload."""
    data = cloudpickle.dumps(payload)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def receive_packed_message(sock: socket.socket) -> object:
    """Receive one framed cloudpickle payload."""
    header = _read_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    return cloudpickle.loads(_read_exact(sock, size))


def status_to_metrics(prefix: str, status: PackedReplicaStatus) -> dict[str, float]:
    """Flatten replica status into tracker metrics."""
    metrics: dict[str, float] = {
        f"{prefix}/active_weight_id": float(status.active_weight_id),
        f"{prefix}/busy": float(int(status.busy)),
        f"{prefix}/total_weight_fetches": float(status.total_weight_fetches),
        f"{prefix}/total_weight_activations": float(status.total_weight_activations),
        f"{prefix}/total_generate_requests": float(status.total_generate_requests),
        f"{prefix}/total_train_generate_requests": float(status.total_train_generate_requests),
        f"{prefix}/total_eval_generate_requests": float(status.total_eval_generate_requests),
        f"{prefix}/total_micro_eval_generate_requests": float(status.total_micro_eval_generate_requests),
    }
    if status.pending_weight_id is not None:
        metrics[f"{prefix}/pending_weight_id"] = float(status.pending_weight_id)
    if status.last_generation_seconds is not None:
        metrics[f"{prefix}/last_generation_seconds"] = float(status.last_generation_seconds)
    for key, value in status.transfer_metrics.items():
        metrics[f"{prefix}/transfer/{key}"] = float(value)
    return metrics


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Packed vLLM control socket closed")
        chunks.extend(chunk)
    return bytes(chunks)


def replace_status_error(status: PackedReplicaStatus | None, message: str) -> PackedReplicaStatus | None:
    """Return a copy of the status with the error field replaced."""
    if status is None:
        return None
    return dataclasses.replace(status, error=message)
