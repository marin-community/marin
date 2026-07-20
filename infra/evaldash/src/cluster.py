# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live Iris job status and finelog logs for evaldash.

Cloud Run reaches the Iris controller and finelog hub through Direct VPC egress. Their internal IPs
are resolved from GCE instance filters and cached briefly. RPCs use the generated Connect clients and
protobuf messages; successful responses are serialized with protobuf's canonical JSON conversion.

Outside the VPC, discovery or RPC failures become reachable=False payloads so the dashboard can show
recorded fallback data instead of failing the whole run-detail request.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from connectrpc.errors import ConnectError
from discovery import resolve_internal_ip
from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from iris.rpc import controller_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)

PROJECT = "hai-gcp-models"
ZONE = "us-central1-a"

CONTROLLER_FILTER = "labels.iris-marin-controller=true AND status=RUNNING"
CONTROLLER_PORT = 10000
FINELOG_FILTER = "name = finelog-marin"
FINELOG_PORT = 10001

IP_CACHE_TTL = 300.0
RPC_TIMEOUT = 4.0

ResponseT = TypeVar("ResponseT")


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"[:400]


def _message_dict(message: Message) -> dict:
    return MessageToDict(
        message,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )


@dataclass
class _CachedIp:
    ip: str
    expires_at: float


class ClusterGateway:
    """Resolve internal service addresses and query them through generated Connect clients."""

    def __init__(self, *, timeout: float = RPC_TIMEOUT, ip_ttl: float = IP_CACHE_TTL) -> None:
        self._timeout_ms = int(timeout * 1000)
        self._ip_ttl = ip_ttl
        self._lock = threading.Lock()
        self._ips: dict[str, _CachedIp] = {}

    def _resolve(self, instance_filter: str, port: int) -> str:
        now = time.monotonic()
        with self._lock:
            cached = self._ips.get(instance_filter)
            if cached is not None and cached.expires_at > now:
                return f"http://{cached.ip}:{port}"
        ip = resolve_internal_ip(PROJECT, ZONE, instance_filter, timeout=self._timeout_ms / 1000)
        with self._lock:
            self._ips[instance_filter] = _CachedIp(ip, now + self._ip_ttl)
        logger.info("resolved %s to %s", instance_filter, ip)
        return f"http://{ip}:{port}"

    def _invalidate(self, instance_filter: str) -> None:
        with self._lock:
            self._ips.pop(instance_filter, None)

    def _call(self, instance_filter: str, port: int, rpc: Callable[[str], ResponseT]) -> ResponseT:
        """Run one RPC, re-resolving the service once after a Connect transport failure."""
        for attempt in range(2):
            address = self._resolve(instance_filter, port)
            try:
                return rpc(address)
            except ConnectError:
                self._invalidate(instance_filter)
                if attempt == 1:
                    raise
        raise AssertionError("unreachable")

    def job_status(self, job_path: str) -> dict:
        """Return canonical protobuf JSON for one Iris job and its tasks."""

        def fetch(address: str):
            client = ControllerServiceClientSync(address=address, timeout_ms=self._timeout_ms)
            try:
                status = client.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_path))
                tasks = client.list_tasks(controller_pb2.Controller.ListTasksRequest(job_id=job_path))
                return status.job, tasks.tasks
            finally:
                client.close()

        try:
            job, tasks = self._call(CONTROLLER_FILTER, CONTROLLER_PORT, fetch)
        except Exception as exc:
            logger.info("iris controller unreachable for %s: %s", job_path, exc)
            return {
                "reachable": False,
                "error": f"iris controller unreachable — {_describe(exc)}",
                "job": None,
                "tasks": [],
            }
        return {
            "reachable": True,
            "error": None,
            "job": _message_dict(job),
            "tasks": [_message_dict(task) for task in tasks],
        }

    def fetch_logs(self, job_path: str, *, max_lines: int, substring: str | None) -> dict:
        """Return canonical protobuf JSON for the latest finelog entries under one Iris job."""
        source = f"{job_path.rstrip('/')}/"
        request = logging_pb2.FetchLogsRequest(
            source=source,
            match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
            max_lines=max_lines,
            tail=True,
            substring=substring or "",
        )

        def fetch(address: str):
            client = LogServiceClientSync(address=address, timeout_ms=self._timeout_ms)
            try:
                return client.fetch_logs(request)
            finally:
                client.close()

        try:
            response = self._call(FINELOG_FILTER, FINELOG_PORT, fetch)
        except Exception as exc:
            logger.info("finelog unreachable for %s: %s", source, exc)
            return {
                "reachable": False,
                "error": f"finelog unreachable — {_describe(exc)}",
                "source": source,
                "entries": [],
            }
        return {
            "reachable": True,
            "error": None,
            "source": source,
            "entries": [_message_dict(entry) for entry in response.entries],
        }
