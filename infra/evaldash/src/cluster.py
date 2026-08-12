# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live Iris job status and finelog logs for evaldash.

Cloud Run reaches the Iris controller and finelog hub through Direct VPC egress. Their internal IPs
are resolved from GCE instance filters and cached briefly. Iris status reads use its typed resource
client; finelog still uses its generated Connect client and canonical protobuf JSON conversion.

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
from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from iris.resources.attempt import AttemptSummary
from iris.resources.job import JobDetail, JobQuery
from iris.resources.task import TaskDetail, TaskQuery
from iris.rpc import job_pb2
from iris.rpc.resource_client import ResourceRpcClient
from rigging.timing import Timestamp

from .discovery import resolve_internal_ip

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
    """Resolve internal service addresses and query Iris and finelog."""

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
        """Return the dashboard's live status shape for one Iris Job resource."""

        def fetch(address: str):
            client = ResourceRpcClient(controller_address=address, timeout_ms=self._timeout_ms)
            try:
                job = _find_job(client, job_path)
                tasks = _job_tasks(client, job)
                return job, tasks
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
            "job": _job_dict(job),
            "tasks": [_task_dict(task) for task in tasks],
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


def _find_job(client: ResourceRpcClient, job_path: str) -> JobDetail:
    page_token = None
    while True:
        page = client.list_jobs(JobQuery(job_id_prefix=job_path, page_size=500, page_token=page_token))
        for summary in page.items:
            if summary.identity.key.resource_id == job_path:
                return client.describe_job(summary.identity.key)
        if page.next_page_token is None:
            raise LookupError(f"Iris job not found: {job_path}")
        page_token = page.next_page_token


def _job_tasks(client: ResourceRpcClient, job: JobDetail) -> list[TaskDetail]:
    tasks: list[TaskDetail] = []
    page_token = None
    while True:
        page = client.list_tasks(TaskQuery(job=job.summary.identity.key, page_size=500, page_token=page_token))
        if page.items:
            tasks.extend(client.describe_tasks(tuple(summary.identity.key for summary in page.items)))
        if page.next_page_token is None:
            return tasks
        page_token = page.next_page_token


def _add_timestamps(result: dict, *, started_at: Timestamp | None, finished_at: Timestamp | None) -> dict:
    if started_at is not None:
        result["started_at"] = {"epoch_ms": started_at.epoch_ms()}
    if finished_at is not None:
        result["finished_at"] = {"epoch_ms": finished_at.epoch_ms()}
    return result


def _attempt_dict(attempt: AttemptSummary) -> dict:
    return _add_timestamps(
        {
            "attempt_id": attempt.identity.attempt_number,
            "state": job_pb2.TaskState.Name(attempt.state),
            "worker_id": attempt.node.key.resource_id if attempt.node is not None else "",
            "exit_code": attempt.exit_code or 0,
            "error": attempt.error_message,
            "is_worker_failure": attempt.state == job_pb2.TASK_STATE_WORKER_FAILED,
            "attempt_uid": attempt.identity.attempt_uid,
        },
        started_at=attempt.started_at,
        finished_at=attempt.finished_at,
    )


def _task_dict(task: TaskDetail) -> dict:
    current_number = task.summary.current_attempt.attempt_number if task.summary.current_attempt is not None else -1
    latest = next(
        (attempt for attempt in task.attempts if attempt.identity.attempt_number == current_number),
        task.attempts[-1] if task.attempts else None,
    )
    return _add_timestamps(
        {
            "task_id": task.summary.identity.key.resource_id,
            "state": job_pb2.TaskState.Name(task.summary.state),
            "worker_id": task.summary.current_node.key.resource_id if task.summary.current_node is not None else "",
            "exit_code": (latest.exit_code or 0) if latest is not None else 0,
            "error": task.summary.error_message,
            "current_attempt_id": current_number,
            "attempts": [_attempt_dict(attempt) for attempt in task.attempts],
        },
        started_at=task.summary.started_at,
        finished_at=task.summary.finished_at,
    )


def _job_dict(job: JobDetail) -> dict:
    return _add_timestamps(
        {
            "state": job_pb2.JobState.Name(job.summary.state),
            "error": job.summary.error_message,
            "exit_code": job.summary.exit_code or 0,
            "name": job.spec.name,
            "status_message": job.summary.pending_reason,
        },
        started_at=job.summary.started_at,
        finished_at=job.summary.finished_at,
    )
