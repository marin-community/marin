# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP server for Iris and Zephyr job babysitting."""

import argparse
import base64
import os
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.cluster.log_keys import build_log_source
from iris.cluster.runtime.profile import SYSTEM_PROCESS_TARGET
from iris.cluster.types import JobName, TaskAttempt
from iris.resources.attempt import AttemptSummary
from iris.resources.endpoint import (
    CpuProfileConfiguration,
    CpuProfileFormat,
    MemoryProfileConfiguration,
    MemoryProfileFormat,
    ProfileConfiguration,
    ThreadsProfileConfiguration,
)
from iris.resources.execution import CpuDevice, GpuDevice, ResourceSpec, TpuDevice
from iris.resources.identity import ResourceKey, ResourceKind
from iris.resources.job import JobDetail, JobQuery, JobSummary
from iris.resources.log import LogMatchScope
from iris.resources.node import NodeHealth, NodeQuery, NodeSummary
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.rpc import job_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.proto_display import job_state_friendly, task_state_friendly
from iris.rpc.resource_client import ResourceRpcClient
from mcp.server.fastmcp import FastMCP
from rigging.auth import BearerTokenInjector, StaticTokenProvider, TokenProvider
from rigging.credential_store import cluster_name_from_url
from rigging.credentials import MARIN_CLUSTER_TOKEN_ENV
from rigging.timing import Duration, Timestamp

DEFAULT_LOG_LINES = 200
DEFAULT_ZEPHYR_LOG_LINES = 5_000
DEFAULT_PROFILE_SECONDS = 1
MAX_LIST_JOBS_PAGE_SIZE = 500
DEFAULT_LIST_JOBS_LIMIT = 100

_ZEPHYR_PROGRESS_RE = re.compile(
    r"\[(?P<stage>[^\]]+)\]\s+"
    r"(?P<completed>\d+)/(?P<total>\d+)\s+complete,\s+"
    r"(?P<in_flight>\d+)\s+in-flight,\s+"
    r"(?P<queued>\d+)\s+queued,\s+"
    r"(?P<workers_alive>\d+)/(?P<workers_total>\d+)\s+workers alive,\s+"
    r"(?P<workers_dead>\d+)\s+dead"
)
_PULL_LOG_NOISE = ("pull_task", "Started operation", "report_result", "registered", "tasks completed")
_OOM_RE = re.compile(r"\b(oom|oomkilled|exit\s+137|killed\s+137)\b", re.IGNORECASE)
_QUOTA_RE = re.compile(r"\b(quota|resource_exhausted|backoff|insufficient capacity|capacity exhausted)\b", re.IGNORECASE)
_TPU_XLA_RE = re.compile(r"\b(tpu|xla|hlo).*\b(bad|fault|hardware|unavailable|failed)\b", re.IGNORECASE)
_DEAD_WORKER_RE = re.compile(r"\b(heartbeat timeout|dead worker|worker.*lost|worker.*crashed)\b", re.IGNORECASE)
_TERMINATED_BY_USER_RE = re.compile(r"terminated by user", re.IGNORECASE)
_ZEPHYR_COORDINATOR_LOOP_FRAME = "_coordinator_loop"
_ZEPHYR_WAIT_FOR_STAGE_FRAME = "_wait_for_stage"
_ZEPHYR_WORKER_POOL_FRAME = "_worker"


@dataclass(frozen=True)
class IrisConnectionConfig:
    """Connection settings for a resident MCP server."""

    controller_url: str
    cluster: str = "default"
    timeout_ms: int = 30_000


def _now_ms() -> int:
    return Timestamp.now().epoch_ms()


def _timestamp_ms(timestamp) -> int | None:
    if timestamp is None or not timestamp.epoch_ms:
        return None
    return int(timestamp.epoch_ms)


def _duration_ms(start, end) -> int | None:
    start_ms = _timestamp_ms(start)
    end_ms = _timestamp_ms(end)
    if start_ms is None:
        return None
    if end_ms is None:
        end_ms = _now_ms()
    return max(0, end_ms - start_ms)


def _resource_spec_to_json(resources: ResourceSpec) -> dict[str, Any]:
    return {
        "cpu_millicores": resources.cpu_millicores,
        "memory_bytes": resources.memory,
        "disk_bytes": resources.disk,
        "device": _device_to_json(resources.device),
    }


def _device_to_json(device: CpuDevice | GpuDevice | TpuDevice | None) -> dict[str, Any]:
    if isinstance(device, GpuDevice):
        return {
            "type": "gpu",
            "variant": device.variant,
            "count": device.count,
        }
    if isinstance(device, TpuDevice):
        return {
            "type": "tpu",
            "variant": device.variant,
            "topology": device.topology,
            "count": device.count,
        }
    if isinstance(device, CpuDevice):
        return {"type": "cpu", "variant": device.variant}
    return _cpu_device_json()


def _cpu_device_json() -> dict[str, str]:
    return {"type": "cpu", "variant": ""}


def _resource_timestamp_ms(timestamp: Timestamp | None) -> int | None:
    return timestamp.epoch_ms() if timestamp is not None else None


def _resource_duration_ms(start: Timestamp | None, end: Timestamp | None) -> int | None:
    if start is None:
        return None
    return max(0, (end or Timestamp.now()).epoch_ms() - start.epoch_ms())


def _attempt_to_json(attempt: AttemptSummary) -> dict[str, Any]:
    return {
        "attempt_id": attempt.identity.attempt_number,
        "attempt_uid": attempt.identity.attempt_uid,
        "worker_id": attempt.node.key.resource_id if attempt.node is not None else "",
        "state": task_state_friendly(attempt.state),
        "exit_code": attempt.exit_code,
        "error": attempt.error_message,
        "terminal_reason": attempt.terminal_reason,
        "started_at_ms": _resource_timestamp_ms(attempt.started_at),
        "finished_at_ms": _resource_timestamp_ms(attempt.finished_at),
        "duration_ms": _resource_duration_ms(attempt.started_at, attempt.finished_at),
    }


def task_status_to_json(task: TaskDetail) -> dict[str, Any]:
    """Serialize a typed Task detail into the MCP's stable JSON envelope."""
    summary = task.summary
    return {
        "task_id": summary.identity.key.resource_id,
        "task_uid": summary.identity.task_uid,
        "state": task_state_friendly(summary.state),
        "worker_id": summary.current_node.key.resource_id if summary.current_node is not None else "",
        "error": summary.error_message,
        "status_message": summary.status_message,
        "started_at_ms": _resource_timestamp_ms(summary.started_at),
        "finished_at_ms": _resource_timestamp_ms(summary.finished_at),
        "duration_ms": _resource_duration_ms(summary.started_at, summary.finished_at),
        "current_attempt_id": summary.current_attempt.attempt_number if summary.current_attempt is not None else None,
        "failure_count": summary.failure_count,
        "preemption_count": summary.preemption_count,
        "attempts": [_attempt_to_json(attempt) for attempt in task.attempts],
        "root_cause_highlights": list(task.root_cause_highlights),
    }


def job_status_to_json(job: JobSummary) -> dict[str, Any]:
    """Serialize a typed Job summary into stable JSON."""
    return {
        "job_id": job.identity.key.resource_id,
        "job_uid": job.identity.job_uid,
        "state": job_state_friendly(job.state),
        "error": job.error_message,
        "submitted_at_ms": _resource_timestamp_ms(job.submitted_at),
        "started_at_ms": _resource_timestamp_ms(job.started_at),
        "finished_at_ms": _resource_timestamp_ms(job.finished_at),
        "duration_ms": _resource_duration_ms(job.started_at, job.finished_at),
        "pending_reason": job.pending_reason,
        "task_count": job.num_tasks,
        "owner_id": job.owner_id,
        "execution_cluster_id": job.execution_cluster_id,
        "backend_id": job.backend_id,
        "parent_job_id": job.parent.key.resource_id if job.parent is not None else "",
    }


def worker_status_to_json(node: NodeSummary) -> dict[str, Any]:
    """Serialize a public Node summary for the legacy MCP response field."""
    return {
        "worker_id": node.identity.key.resource_id,
        "node_uid": node.identity.node_uid,
        "backend_id": node.identity.backend_id,
        "healthy": node.health is NodeHealth.READY,
        "status_message": node.health.value,
        "last_heartbeat_ms": node.observed_at.epoch_ms(),
        "running_task_count": node.running_task_count,
        "scaling_group_id": node.scaling_group_id,
    }


def log_entry_to_json(entry: logging_pb2.LogEntry) -> dict[str, Any]:
    """Serialize an Iris log entry into stable JSON."""
    return {
        "timestamp_ms": _timestamp_ms(entry.timestamp),
        "source": entry.source,
        "data": entry.data,
        "attempt_id": int(entry.attempt_id),
        "level": logging_pb2.LogLevel.Name(entry.level).removeprefix("LOG_LEVEL_").lower(),
        "key": entry.key,
        "task_id": _task_id_from_log_key(entry.key),
    }


def _task_id_from_log_key(key: str) -> str:
    if not key:
        return ""
    return key.split(":", 1)[0]


def _response(data: Any, *, warnings: list[str] | None, cluster: str, auth_ok: bool = True) -> dict[str, Any]:
    return {
        "data": data,
        "warnings": warnings or [],
        "auth_ok": auth_ok,
        "cluster": cluster,
        "fetched_at_ms": _now_ms(),
    }


def parse_zephyr_progress(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse Zephyr coordinator progress logs, keeping the latest snapshot per stage."""
    snapshots_by_stage: dict[str, dict[str, Any]] = {}
    for line in lines:
        if any(noise in line for noise in _PULL_LOG_NOISE):
            continue
        match = _ZEPHYR_PROGRESS_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        stage = groups["stage"]
        snapshots_by_stage[stage] = {
            "stage": stage,
            "completed": int(groups["completed"]),
            "total": int(groups["total"]),
            "in_flight": int(groups["in_flight"]),
            "queued": int(groups["queued"]),
            "workers_alive": int(groups["workers_alive"]),
            "workers_total": int(groups["workers_total"]),
            "workers_dead": int(groups["workers_dead"]),
        }
    return list(snapshots_by_stage.values())


def parse_zephyr_thread_state(thread_dump: str) -> dict[str, Any]:
    """Classify a Zephyr coordinator thread dump into a compact liveness state."""
    if not thread_dump:
        return {"state": "unknown", "evidence": ["empty thread dump"]}

    evidence: list[str] = []
    has_wait_for_stage = _ZEPHYR_WAIT_FOR_STAGE_FRAME in thread_dump
    has_coordinator_loop = _ZEPHYR_COORDINATOR_LOOP_FRAME in thread_dump
    has_worker_pool = _ZEPHYR_WORKER_POOL_FRAME in thread_dump

    if has_wait_for_stage:
        evidence.append("waiting for stage completion")
    if has_coordinator_loop:
        evidence.append("coordinator loop thread present")
    if has_wait_for_stage or has_coordinator_loop:
        return {"state": "active", "evidence": evidence}
    if has_worker_pool:
        return {"state": "zombie_suspected", "evidence": ["worker pool frames without coordinator loop"]}
    return {"state": "unknown", "evidence": ["no Zephyr coordinator frames found"]}


def classify_diagnosis(
    *,
    job: dict[str, Any],
    logs: Iterable[dict[str, Any]],
    workers: Iterable[dict[str, Any]],
    thread_dump: str,
) -> list[dict[str, Any]]:
    """Classify common Iris/Zephyr babysitting failure signals."""
    signals: list[dict[str, Any]] = []
    log_text = "\n".join(str(entry.get("data", "")) for entry in logs)
    tasks = list(job.get("tasks", []))

    def add(signal: str, severity: str, evidence: list[str], escalation_hint: str) -> None:
        signals.append(
            {
                "signal": signal,
                "severity": severity,
                "evidence": evidence,
                "escalation_hint": escalation_hint,
            }
        )

    pending_reason = str(job.get("pending_reason", ""))
    pending_tasks = [task for task in tasks if task.get("state") in ("pending", "assigned")]
    if job.get("state") == "pending" or pending_reason:
        add(
            "pending",
            "warning",
            [pending_reason or f"{len(pending_tasks)} pending/assigned task(s)"],
            "Check scheduler constraints, quota, and autoscaler state.",
        )

    stuck_assigned = [task for task in pending_tasks if task.get("state") == "assigned"]
    if stuck_assigned:
        add(
            "stuck_assigned",
            "warning",
            [task.get("task_id", "") for task in stuck_assigned[:5]],
            "Inspect worker status and task attempt logs.",
        )

    retry_tasks = [task for task in tasks if len(task.get("attempts", [])) > 1]
    if int(job.get("failure_count", 0) or 0) > 0 or retry_tasks:
        add(
            "repeated_retries",
            "error",
            [f"failure_count={job.get('failure_count', 0)}", *[task.get("task_id", "") for task in retry_tasks[:4]]],
            "Compare failed attempts and look for a repeated terminal error.",
        )

    oom_tasks = [
        task
        for task in tasks
        if int(task.get("exit_code", 0) or 0) == 137
        or _OOM_RE.search(str(task.get("error", "")))
        or _OOM_RE.search(log_text)
    ]
    if oom_tasks:
        add(
            "oom_or_exit_137",
            "error",
            [task.get("task_id", "") for task in oom_tasks[:5]],
            "Increase memory or inspect per-task memory peaks before retrying.",
        )

    if _TPU_XLA_RE.search(log_text):
        add(
            "tpu_xla_bad_node",
            "error",
            ["TPU/XLA bad-node pattern in recent logs"],
            "Collect worker/process status and escalate to infrastructure triage.",
        )

    if _QUOTA_RE.search(pending_reason) or _QUOTA_RE.search(log_text):
        add(
            "quota_or_backoff",
            "warning",
            [pending_reason or "quota/backoff pattern in recent logs"],
            "Check capacity, quota, and autoscaler backoff state.",
        )

    unhealthy_workers = [
        worker
        for worker in workers
        if not worker.get("healthy", True) or _DEAD_WORKER_RE.search(str(worker.get("status_message", "")))
    ]
    if unhealthy_workers or _DEAD_WORKER_RE.search(log_text):
        add(
            "dead_worker",
            "error",
            [worker.get("worker_id", "") for worker in unhealthy_workers[:5]] or ["worker death pattern in logs"],
            "Inspect involved workers and recent process logs.",
        )

    thread_state = parse_zephyr_thread_state(thread_dump)
    if thread_state["state"] == "zombie_suspected":
        add(
            "zombie_coordinator",
            "error",
            thread_state["evidence"],
            "Restart only after confirming with the user.",
        )

    if _TERMINATED_BY_USER_RE.search(str(job.get("error", ""))) and signals:
        add(
            "misleading_terminated_by_user",
            "warning",
            [str(job.get("error", ""))],
            "Treat the termination message as a symptom; use the other signals as root-cause candidates.",
        )

    return signals


class IrisBabysitter:
    """Resident Iris client wrapper exposed through MCP tools."""

    def __init__(self, config: IrisConnectionConfig):
        self.config = config
        self.token_provider = _token_provider()
        interceptors = [BearerTokenInjector(self.token_provider, "authorization")] if self.token_provider else []
        self.controller = ControllerServiceClientSync(
            config.controller_url,
            timeout_ms=config.timeout_ms,
            interceptors=interceptors,
            accept_compression=IRIS_RPC_COMPRESSIONS,
            send_compression=IRIS_RPC_COMPRESSIONS[0],
        )
        self.logs = LogServiceClientSync(
            config.controller_url,
            timeout_ms=config.timeout_ms,
            interceptors=interceptors,
        )
        self.resources = ResourceRpcClient(
            config.controller_url,
            timeout_ms=config.timeout_ms,
            interceptors=interceptors,
        )

    def close(self) -> None:
        self.resources.close()
        self.logs.close()
        self.controller.close()

    def envelope(self, data: Any, *, warnings: list[str] | None = None, auth_ok: bool = True) -> dict[str, Any]:
        return _response(data, warnings=warnings, cluster=self.config.cluster, auth_ok=auth_ok)

    def list_jobs(
        self,
        *,
        prefix: str = "",
        state: str = "",
        name_filter: str = "",
        limit: int = DEFAULT_LIST_JOBS_LIMIT,
    ) -> dict[str, Any]:
        normalized_state = _normalize_state_filter(state)
        states = (
            frozenset({job_pb2.JobState.Value(f"JOB_STATE_{normalized_state.upper()}")})
            if normalized_state
            else frozenset()
        )
        jobs: list[dict[str, Any]] = []
        page_token: str | None = None
        capped_limit = max(1, limit)
        while len(jobs) < capped_limit:
            page = self.resources.list_jobs(
                JobQuery(
                    states=states,
                    job_id_prefix=prefix or None,
                    page_size=min(MAX_LIST_JOBS_PAGE_SIZE, capped_limit - len(jobs)),
                    page_token=page_token,
                )
            )
            for job in page.items:
                if name_filter and name_filter not in job.identity.key.resource_id:
                    continue
                jobs.append(job_status_to_json(job))
                if len(jobs) >= capped_limit:
                    break
            page_token = page.next_page_token
            if page_token is None:
                break
        return self.envelope({"jobs": jobs, "count": len(jobs)})

    def job_summary(self, job_id: str) -> dict[str, Any]:
        detail = self._job_detail(job_id)
        tasks = self._tasks_for_job(detail.summary.identity.key)
        return self.envelope(_job_summary_payload(detail, tasks))

    def job_tree(self, job_id: str) -> dict[str, Any]:
        root = JobName.from_wire(job_id)
        child_jobs = self._jobs_with_prefix(job_id)
        nodes = {job.identity.key.resource_id: {**job_status_to_json(job), "children": []} for job in child_jobs}

        for node_id in nodes:
            parent = JobName.from_wire(node_id).parent
            if parent is not None and root.is_ancestor_of(JobName.from_wire(node_id), include_self=False):
                parent_id = parent.to_wire()
                if parent_id in nodes:
                    nodes[parent_id]["children"].append(node_id)

        return self.envelope({"root": job_id, "nodes": nodes})

    def task_summary(self, task_id: str) -> dict[str, Any]:
        detail = self._task_detail(task_id)
        payload = task_status_to_json(detail)
        job = self.resources.describe_job(detail.summary.job.key)
        payload["job_resources"] = _resource_spec_to_json(job.spec.resources)
        return self.envelope(payload)

    def _job_detail(self, job_id: str) -> JobDetail:
        canonical_id = JobName.from_wire(job_id).to_wire()
        return self.resources.describe_job(ResourceKey(self.config.cluster, ResourceKind.JOB, canonical_id))

    def _task_detail(self, task_id: str) -> TaskDetail:
        canonical_id = JobName.from_wire(task_id).to_wire()
        return self.resources.describe_task(ResourceKey(self.config.cluster, ResourceKind.TASK, canonical_id))

    def _tasks_for_job(self, job: ResourceKey) -> list[TaskSummary]:
        tasks: list[TaskSummary] = []
        page_token: str | None = None
        while True:
            page = self.resources.list_tasks(
                TaskQuery(job=job, page_size=MAX_LIST_JOBS_PAGE_SIZE, page_token=page_token)
            )
            tasks.extend(page.items)
            page_token = page.next_page_token
            if page_token is None:
                return tasks

    def tail_logs(
        self,
        *,
        target: str,
        since_ms: int = 0,
        cursor: int = 0,
        max_lines: int = DEFAULT_LOG_LINES,
        substring: str = "",
        min_level: str = "",
        attempt_id: int = -1,
        tail: bool = True,
    ) -> dict[str, Any]:
        source, match_scope = _log_source(target, attempt_id)
        response = self.logs.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=source,
                match_scope=int(match_scope),
                since_ms=since_ms,
                cursor=cursor,
                max_lines=max_lines,
                substring=substring,
                min_level=min_level,
                tail=tail,
            )
        )
        return self.envelope(
            {
                "entries": [log_entry_to_json(entry) for entry in response.entries],
                "cursor": int(response.cursor),
                "source": source,
            }
        )

    def worker_status(self, job_id: str = "") -> dict[str, Any]:
        task_nodes: set[str] | None = None
        if job_id:
            job = self._job_detail(job_id)
            task_nodes = {
                task.current_node.key.resource_id
                for task in self._tasks_for_job(job.summary.identity.key)
                if task.current_node is not None
            }
        workers: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            page = self.resources.list_nodes(NodeQuery(page_size=500, page_token=page_token))
            workers.extend(
                worker_status_to_json(node)
                for node in page.items
                if task_nodes is None or node.identity.key.resource_id in task_nodes
            )
            page_token = page.next_page_token
            if page_token is None:
                break
        return self.envelope({"workers": workers, "count": len(workers)})

    def process_status(
        self,
        *,
        target: str = "",
        max_log_lines: int = 0,
        log_substring: str = "",
        min_log_level: str = "",
    ) -> dict[str, Any]:
        response = self.controller.get_process_status(
            job_pb2.GetProcessStatusRequest(
                target=target,
                max_log_lines=max_log_lines,
                log_substring=log_substring,
                min_log_level=min_log_level,
            )
        )
        info = response.process_info
        return self.envelope(
            {
                "process": {
                    "hostname": info.hostname,
                    "pid": int(info.pid),
                    "python_version": info.python_version,
                    "uptime_ms": int(info.uptime_ms),
                    "memory_rss_bytes": int(info.memory_rss_bytes),
                    "memory_vms_bytes": int(info.memory_vms_bytes),
                    "memory_total_bytes": int(info.memory_total_bytes),
                    "cpu_count": int(info.cpu_count),
                    "cpu_millicores": int(info.cpu_millicores),
                    "thread_count": int(info.thread_count),
                    "open_fd_count": int(info.open_fd_count),
                    "git_hash": info.provenance.tree_hash,
                },
                "logs": [log_entry_to_json(entry) for entry in response.log_entries],
            }
        )

    def profile_task(
        self,
        *,
        target: str = SYSTEM_PROCESS_TARGET,
        profile_type: str = "threads",
        duration_seconds: int = DEFAULT_PROFILE_SECONDS,
        include_locals: bool = False,
    ) -> dict[str, Any]:
        profile = _profile_type(profile_type, include_locals=include_locals)
        if target.startswith("/system/"):
            response = self.controller.profile_task(
                job_pb2.ProfileTaskRequest(
                    target=target,
                    duration_seconds=duration_seconds,
                    profile_type=profile_configuration_to_proto(profile),
                )
            )
            profile_data = response.profile_data
            error = response.error
        else:
            requested = TaskAttempt.from_wire(target)
            requested.task_id.require_task()
            task = self._task_detail(requested.task_id.to_wire())
            attempt = task.summary.current_attempt
            if requested.attempt_id is not None:
                attempt = next(
                    (item.identity for item in task.attempts if item.identity.attempt_number == requested.attempt_id),
                    None,
                )
            if attempt is None:
                error = f"Task {requested.task_id} has no matching Attempt to profile"
                profile_data = b""
            else:
                result = self.resources.profile_attempt(
                    attempt,
                    profile=profile,
                    duration=Duration.from_seconds(duration_seconds),
                )
                profile_data = result.profile_data
                error = result.error_message
        if error:
            return self.envelope({"error": error}, warnings=[error], auth_ok=True)
        if profile_type == "threads":
            data = {"text": profile_data.decode("utf-8", errors="replace"), "encoding": "utf-8"}
        else:
            data = {
                "data_base64": base64.b64encode(profile_data).decode("ascii"),
                "encoding": "base64",
                "profile_type": profile_type,
            }
        return self.envelope(data)

    def zephyr_stage_progress(self, *, coord_job_id: str, max_lines: int = DEFAULT_ZEPHYR_LOG_LINES) -> dict[str, Any]:
        log_payload = self.tail_logs(target=coord_job_id, max_lines=max_lines, tail=True)["data"]
        lines = [entry["data"] for entry in log_payload["entries"]]
        return self.envelope({"progress": parse_zephyr_progress(lines), "cursor": log_payload["cursor"]})

    def zephyr_coordinator_status(self, *, coord_job_id: str) -> dict[str, Any]:
        summary = self.job_summary(coord_job_id)["data"]
        progress_payload = self.zephyr_stage_progress(coord_job_id=coord_job_id)["data"]
        thread_target = f"{coord_job_id}/0"
        thread_profile = self.profile_task(
            target=thread_target,
            profile_type="threads",
            duration_seconds=DEFAULT_PROFILE_SECONDS,
        )
        thread_dump = str(thread_profile["data"].get("text", ""))
        thread_state = parse_zephyr_thread_state(thread_dump)
        thread_warnings = list(thread_profile["warnings"])
        if thread_warnings:
            thread_state = {
                "state": "unavailable",
                "evidence": thread_warnings,
            }
        diagnosis = classify_diagnosis(job=summary, logs=[], workers=[], thread_dump=thread_dump)
        return self.envelope(
            {
                "summary": summary,
                "progress": progress_payload["progress"],
                "cursor": progress_payload["cursor"],
                "thread_liveness": {
                    "target": thread_target,
                    **thread_state,
                },
                "diagnosis": diagnosis,
            },
            warnings=thread_warnings,
        )

    def diagnose(self, *, job_id: str, log_lines: int = DEFAULT_LOG_LINES) -> dict[str, Any]:
        summary = self.job_summary(job_id)["data"]
        logs = self.tail_logs(target=job_id, max_lines=log_lines, tail=True)["data"]["entries"]
        workers = self.worker_status(job_id)["data"]["workers"]
        signals = classify_diagnosis(job=summary, logs=logs, workers=workers, thread_dump="")
        return self.envelope({"signals": signals, "job_id": job_id})

    def _jobs_with_prefix(self, prefix: str) -> list[JobSummary]:
        jobs: list[JobSummary] = []
        page_token: str | None = None
        while True:
            page = self.resources.list_jobs(
                JobQuery(
                    job_id_prefix=prefix,
                    page_size=MAX_LIST_JOBS_PAGE_SIZE,
                    page_token=page_token,
                )
            )
            jobs.extend(page.items)
            page_token = page.next_page_token
            if page_token is None:
                return jobs


def _job_summary_payload(job: JobDetail, tasks: list[TaskSummary]) -> dict[str, Any]:
    state_counts = Counter(task_state_friendly(task.state) for task in tasks)
    task_payloads = []
    for task in sorted(tasks, key=lambda item: item.task_index):
        task_payloads.append(
            {
                "task_id": task.identity.key.resource_id,
                "task_uid": task.identity.task_uid,
                "index": str(task.task_index),
                "state": task_state_friendly(task.state),
                "worker_id": task.current_node.key.resource_id if task.current_node is not None else "",
                "status_message": task.status_message,
                "error": task.error_message,
                "failure_count": task.failure_count,
                "preemption_count": task.preemption_count,
                "started_at_ms": _resource_timestamp_ms(task.started_at),
                "finished_at_ms": _resource_timestamp_ms(task.finished_at),
                "duration_ms": _resource_duration_ms(task.started_at, task.finished_at),
            }
        )
    summary = job.summary
    return {
        **job_status_to_json(summary),
        "name": job.spec.name,
        "failure_count": sum(task.failure_count for task in tasks),
        "preemption_count": sum(task.preemption_count for task in tasks),
        "completed_count": sum(count for state, count in state_counts.items() if state in {"succeeded", "killed"}),
        "task_state_counts": dict(state_counts),
        "resource_requests": _resource_spec_to_json(job.spec.resources),
        "ports": dict.fromkeys(job.spec.ports, 0),
        "tasks": task_payloads,
    }


def _token_provider() -> TokenProvider | None:
    """Explicit Authorization bearer for CI / headless runs, else None.

    The controller mints no user token, so nothing is cached to attach; a caller
    may inject one (e.g. a worker JWT) via ``$MARIN_CLUSTER_TOKEN``. Otherwise the
    babysitter relies on transport trust (SSH tunnel / loopback), like any other
    tokenless client.
    """
    override = os.environ.get(MARIN_CLUSTER_TOKEN_ENV)
    return StaticTokenProvider(override) if override else None


def _normalize_state_filter(state: str) -> str:
    normalized = state.strip().lower()
    if normalized.startswith("job_state_"):
        return normalized.removeprefix("job_state_")
    return normalized


def _log_source(target: str, attempt_id: int) -> tuple[str, LogMatchScope]:
    if target.startswith("/system/"):
        return target, LogMatchScope.EXACT
    return build_log_source(JobName.from_wire(target), attempt_id)


def _profile_type(profile_type: str, *, include_locals: bool) -> ProfileConfiguration:
    if profile_type == "threads":
        return ThreadsProfileConfiguration(include_locals=include_locals)
    if profile_type == "cpu":
        return CpuProfileConfiguration(format=CpuProfileFormat.SPEEDSCOPE, rate_hz=0, native=None)
    if profile_type == "mem":
        return MemoryProfileConfiguration(format=MemoryProfileFormat.FLAMEGRAPH, leaks=False)
    raise ValueError(f"Unknown profile_type: {profile_type}")


def build_server(service: IrisBabysitter, *, host: str = "127.0.0.1", port: int = 8000) -> FastMCP:
    """Build the FastMCP server for a resident Iris connection."""
    server = FastMCP(
        "marin-mcp-babysitter",
        instructions="Structured Iris and Zephyr job babysitting tools.",
        host=host,
        port=port,
    )

    @server.tool()
    def iris_list_jobs(
        prefix: str = "",
        state: str = "",
        name_filter: str = "",
        limit: int = DEFAULT_LIST_JOBS_LIMIT,
    ) -> dict[str, Any]:
        return service.list_jobs(prefix=prefix, state=state, name_filter=name_filter, limit=limit)

    @server.tool()
    def iris_job_summary(job_id: str) -> dict[str, Any]:
        return service.job_summary(job_id)

    @server.tool()
    def iris_job_tree(job_id: str) -> dict[str, Any]:
        return service.job_tree(job_id)

    @server.tool()
    def iris_task_summary(task_id: str) -> dict[str, Any]:
        return service.task_summary(task_id)

    @server.tool()
    def iris_tail_logs(
        target: str,
        since_ms: int = 0,
        cursor: int = 0,
        max_lines: int = DEFAULT_LOG_LINES,
        substring: str = "",
        min_level: str = "",
        attempt_id: int = -1,
        tail: bool = True,
    ) -> dict[str, Any]:
        return service.tail_logs(
            target=target,
            since_ms=since_ms,
            cursor=cursor,
            max_lines=max_lines,
            substring=substring,
            min_level=min_level,
            attempt_id=attempt_id,
            tail=tail,
        )

    @server.tool()
    def iris_worker_status(job_id: str = "") -> dict[str, Any]:
        return service.worker_status(job_id)

    @server.tool()
    def iris_process_status(
        target: str = "",
        max_log_lines: int = 0,
        log_substring: str = "",
        min_log_level: str = "",
    ) -> dict[str, Any]:
        return service.process_status(
            target=target,
            max_log_lines=max_log_lines,
            log_substring=log_substring,
            min_log_level=min_log_level,
        )

    @server.tool()
    def iris_profile_task(
        target: str = SYSTEM_PROCESS_TARGET,
        profile_type: str = "threads",
        duration_seconds: int = DEFAULT_PROFILE_SECONDS,
        include_locals: bool = False,
    ) -> dict[str, Any]:
        return service.profile_task(
            target=target,
            profile_type=profile_type,
            duration_seconds=duration_seconds,
            include_locals=include_locals,
        )

    @server.tool()
    def zephyr_stage_progress(coord_job_id: str, max_lines: int = DEFAULT_ZEPHYR_LOG_LINES) -> dict[str, Any]:
        return service.zephyr_stage_progress(coord_job_id=coord_job_id, max_lines=max_lines)

    @server.tool()
    def zephyr_coordinator_status(coord_job_id: str) -> dict[str, Any]:
        return service.zephyr_coordinator_status(coord_job_id=coord_job_id)

    @server.tool()
    def diagnose(job_id: str, log_lines: int = DEFAULT_LOG_LINES) -> dict[str, Any]:
        return service.diagnose(job_id=job_id, log_lines=log_lines)

    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the Marin Iris/Zephyr babysitting MCP server.")
    parser.add_argument("--controller-url", required=True, help="Iris controller URL.")
    parser.add_argument("--cluster", default=None, help="Cluster label and Iris token-store key.")
    parser.add_argument("--timeout-ms", type=int, default=30_000, help="Controller RPC timeout in milliseconds.")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host for SSE/streamable-http transports.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port for SSE/streamable-http transports.")
    args = parser.parse_args(argv)
    cluster = args.cluster or cluster_name_from_url(args.controller_url)

    service = IrisBabysitter(
        IrisConnectionConfig(
            controller_url=args.controller_url,
            cluster=cluster,
            timeout_ms=args.timeout_ms,
        )
    )
    try:
        build_server(service, host=args.host, port=args.port).run(transport=args.transport)
    finally:
        service.close()


if __name__ == "__main__":
    main()
