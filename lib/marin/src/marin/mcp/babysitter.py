# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MCP server for Iris and Zephyr job babysitting."""

import argparse
import base64
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finelog.rpc import logging_pb2
from finelog.rpc.logging_connect import LogServiceClientSync
from google.protobuf import json_format
from iris.cli.bug_report import gather_bug_report
from iris.cli.job import build_job_summary
from iris.cli.token_store import cluster_name_from_url, load_any_token, load_token
from iris.cluster.log_store_helpers import build_log_source
from iris.cluster.runtime.profile import SYSTEM_PROCESS_TARGET
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthTokenInjector, StaticTokenProvider, TokenProvider
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.proto_utils import job_state_friendly, task_state_friendly
from mcp.server.fastmcp import FastMCP
from rigging.timing import Timestamp

DEFAULT_LOG_LINES = 200
DEFAULT_ZEPHYR_LOG_LINES = 5_000
DEFAULT_PROFILE_SECONDS = 1
MAX_LIST_JOBS_PAGE_SIZE = 500
DEFAULT_LIST_JOBS_LIMIT = 100
_PROTO_TO_DICT_OPTIONS = dict(preserving_proto_field_name=True)

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


def _resource_usage_to_json(usage: job_pb2.ResourceUsage) -> dict[str, int]:
    return {
        "memory_mb": int(usage.memory_mb),
        "memory_peak_mb": int(usage.memory_peak_mb),
        "cpu_millicores": int(usage.cpu_millicores),
        "disk_mb": int(usage.disk_mb),
        "process_count": int(usage.process_count),
    }


def _resource_spec_to_json(resources: job_pb2.ResourceSpecProto) -> dict[str, Any]:
    return {
        "cpu_millicores": int(resources.cpu_millicores),
        "memory_bytes": int(resources.memory_bytes),
        "disk_bytes": int(resources.disk_bytes),
        "device": _device_config_to_json(resources.device) if resources.HasField("device") else _cpu_device_json(),
    }


def _device_config_to_json(device_config: job_pb2.DeviceConfig) -> dict[str, Any]:
    kind = device_config.WhichOneof("device")
    if kind == "gpu":
        return {
            "type": "gpu",
            "variant": device_config.gpu.variant,
            "count": int(device_config.gpu.count),
        }
    if kind == "tpu":
        return {
            "type": "tpu",
            "variant": device_config.tpu.variant,
            "topology": device_config.tpu.topology,
            "count": int(device_config.tpu.count),
        }
    if kind == "cpu":
        return {"type": "cpu", "variant": device_config.cpu.variant}
    return _cpu_device_json()


def _cpu_device_json() -> dict[str, str]:
    return {"type": "cpu", "variant": ""}


def _attempt_to_json(attempt: job_pb2.TaskAttempt) -> dict[str, Any]:
    return {
        "attempt_id": int(attempt.attempt_id),
        "worker_id": attempt.worker_id,
        "state": task_state_friendly(attempt.state),
        "exit_code": int(attempt.exit_code),
        "error": attempt.error,
        "started_at_ms": _timestamp_ms(attempt.started_at),
        "finished_at_ms": _timestamp_ms(attempt.finished_at),
        "duration_ms": _duration_ms(attempt.started_at, attempt.finished_at),
        "is_worker_failure": bool(attempt.is_worker_failure),
    }


def task_status_to_json(task: job_pb2.TaskStatus) -> dict[str, Any]:
    """Serialize Iris task status into stable JSON."""
    return {
        "task_id": task.task_id,
        "state": task_state_friendly(task.state),
        "worker_id": task.worker_id,
        "worker_address": task.worker_address,
        "exit_code": int(task.exit_code),
        "error": task.error,
        "started_at_ms": _timestamp_ms(task.started_at),
        "finished_at_ms": _timestamp_ms(task.finished_at),
        "duration_ms": _duration_ms(task.started_at, task.finished_at),
        "current_attempt_id": int(task.current_attempt_id),
        "pending_reason": task.pending_reason,
        "can_be_scheduled": bool(task.can_be_scheduled),
        "container_id": task.container_id,
        "ports": dict(task.ports),
        "resource_usage": _resource_usage_to_json(task.resource_usage),
        "attempts": [_attempt_to_json(attempt) for attempt in task.attempts],
    }


def job_status_to_json(job: job_pb2.JobStatus, tasks: Iterable[job_pb2.TaskStatus] = ()) -> dict[str, Any]:
    """Serialize Iris job status into stable JSON."""
    task_payloads = [task_status_to_json(task) for task in tasks]
    return {
        "job_id": job.job_id,
        "name": job.name,
        "state": job_state_friendly(job.state),
        "exit_code": int(job.exit_code),
        "error": job.error,
        "submitted_at_ms": _timestamp_ms(job.submitted_at),
        "started_at_ms": _timestamp_ms(job.started_at),
        "finished_at_ms": _timestamp_ms(job.finished_at),
        "duration_ms": _duration_ms(job.started_at, job.finished_at),
        "status_message": job.status_message,
        "pending_reason": job.pending_reason,
        "failure_count": int(job.failure_count),
        "preemption_count": int(job.preemption_count),
        "task_count": int(job.task_count),
        "completed_count": int(job.completed_count),
        "task_state_counts": dict(job.task_state_counts),
        "has_children": bool(job.has_children),
        "resource_usage": _resource_usage_to_json(job.resource_usage),
        "resource_requests": _resource_spec_to_json(job.resources),
        "ports": dict(job.ports),
        "tasks": task_payloads,
    }


def _worker_metadata_to_json(metadata: job_pb2.WorkerMetadata) -> dict[str, Any]:
    return json_format.MessageToDict(metadata, **_PROTO_TO_DICT_OPTIONS)


def worker_status_to_json(worker: controller_pb2.Controller.WorkerHealthStatus) -> dict[str, Any]:
    """Serialize Iris worker health into stable JSON."""
    return {
        "worker_id": worker.worker_id,
        "healthy": bool(worker.healthy),
        "consecutive_failures": int(worker.consecutive_failures),
        "last_heartbeat_ms": _timestamp_ms(worker.last_heartbeat),
        "running_job_ids": list(worker.running_job_ids),
        "address": worker.address,
        "status_message": worker.status_message,
        "metadata": _worker_metadata_to_json(worker.metadata),
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
        self.token_provider = _token_provider(config.cluster)
        interceptors = [AuthTokenInjector(self.token_provider)] if self.token_provider else []
        self.controller = ControllerServiceClientSync(
            config.controller_url,
            timeout_ms=config.timeout_ms,
            interceptors=interceptors,
        )
        self.logs = LogServiceClientSync(
            config.controller_url,
            timeout_ms=config.timeout_ms,
            interceptors=interceptors,
        )

    def close(self) -> None:
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
        state_filter = _normalize_state_filter(state)
        jobs: list[dict[str, Any]] = []
        offset = 0
        capped_limit = max(1, limit)
        prefix_job = JobName.from_wire(prefix) if prefix else None
        while len(jobs) < capped_limit:
            query = controller_pb2.Controller.JobQuery(
                state_filter=state_filter,
                name_filter=name_filter,
                sort_field=controller_pb2.Controller.JOB_SORT_FIELD_DATE,
                sort_direction=controller_pb2.Controller.SORT_DIRECTION_DESC,
                offset=offset,
                limit=MAX_LIST_JOBS_PAGE_SIZE,
            )
            response = self.controller.list_jobs(controller_pb2.Controller.ListJobsRequest(query=query))
            for job in response.jobs:
                if prefix_job is not None and not _job_matches_prefix(job.job_id, prefix_job):
                    continue
                jobs.append(job_status_to_json(job))
                if len(jobs) >= capped_limit:
                    break
            if not response.has_more:
                break
            offset += len(response.jobs)
        return self.envelope({"jobs": jobs, "count": len(jobs)})

    def job_summary(self, job_id: str) -> dict[str, Any]:
        job_response = self.controller.get_job_status(controller_pb2.Controller.GetJobStatusRequest(job_id=job_id))
        tasks_response = self.controller.list_tasks(controller_pb2.Controller.ListTasksRequest(job_id=job_id))
        return self.envelope(_job_summary_payload(job_response.job, list(tasks_response.tasks)))

    def job_tree(self, job_id: str) -> dict[str, Any]:
        root = JobName.from_wire(job_id)
        child_jobs = self._jobs_with_prefix(job_id)
        nodes: dict[str, dict[str, Any]] = {}
        for job in child_jobs:
            nodes[job.job_id] = job_status_to_json(job)
            nodes[job.job_id]["children"] = []

        for node_id in nodes:
            parent = JobName.from_wire(node_id).parent
            if parent is not None and root.is_ancestor_of(JobName.from_wire(node_id), include_self=False):
                parent_id = parent.to_wire()
                if parent_id in nodes:
                    nodes[parent_id]["children"].append(node_id)

        return self.envelope({"root": job_id, "nodes": nodes})

    def task_summary(self, task_id: str) -> dict[str, Any]:
        response = self.controller.get_task_status(controller_pb2.Controller.GetTaskStatusRequest(task_id=task_id))
        payload = task_status_to_json(response.task)
        payload["job_resources"] = _resource_spec_to_json(response.job_resources)
        return self.envelope(payload)

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
        source = _log_source(target, attempt_id)
        response = self.logs.fetch_logs(
            logging_pb2.FetchLogsRequest(
                source=source,
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
        response = self.controller.list_workers(controller_pb2.Controller.ListWorkersRequest())
        workers = [worker_status_to_json(worker) for worker in response.workers]
        if job_id:
            task_workers = {
                task.worker_id
                for task in self.controller.list_tasks(controller_pb2.Controller.ListTasksRequest(job_id=job_id)).tasks
                if task.worker_id
            }
            workers = [
                worker
                for worker in workers
                if job_id in worker["running_job_ids"] or worker["worker_id"] in task_workers
            ]
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
                    "git_hash": info.git_hash,
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
        request = job_pb2.ProfileTaskRequest(
            target=target,
            duration_seconds=duration_seconds,
            profile_type=_profile_type(profile_type, include_locals=include_locals),
        )
        response = self.controller.profile_task(request)
        if response.error:
            return self.envelope({"error": response.error}, warnings=[response.error], auth_ok=True)
        if profile_type == "threads":
            data = {"text": response.profile_data.decode("utf-8", errors="replace"), "encoding": "utf-8"}
        else:
            data = {
                "data_base64": base64.b64encode(response.profile_data).decode("ascii"),
                "encoding": "base64",
                "profile_type": profile_type,
            }
        return self.envelope(data)

    def bug_report(self, *, job_id: str, tail: int = 50) -> dict[str, Any]:
        report = gather_bug_report(
            self.config.controller_url,
            JobName.from_wire(job_id),
            tail=tail,
            token_provider=self.token_provider,
        )
        return self.envelope(asdict(report))

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

    def _jobs_with_prefix(self, prefix: str) -> list[job_pb2.JobStatus]:
        jobs: list[job_pb2.JobStatus] = []
        offset = 0
        root = JobName.from_wire(prefix)
        while True:
            query = controller_pb2.Controller.JobQuery(
                sort_field=controller_pb2.Controller.JOB_SORT_FIELD_DATE,
                sort_direction=controller_pb2.Controller.SORT_DIRECTION_DESC,
                offset=offset,
                limit=MAX_LIST_JOBS_PAGE_SIZE,
            )
            response = self.controller.list_jobs(controller_pb2.Controller.ListJobsRequest(query=query))
            jobs.extend(job for job in response.jobs if _job_matches_prefix(job.job_id, root))
            if not response.has_more:
                return jobs
            offset += len(response.jobs)


def _job_summary_payload(job: job_pb2.JobStatus, tasks: list[job_pb2.TaskStatus]) -> dict[str, Any]:
    summary = build_job_summary(job, tasks)
    for key, value in job_status_to_json(job).items():
        summary.setdefault(key, value)
    return summary


def _job_matches_prefix(job_id: str, prefix: JobName) -> bool:
    return prefix.is_ancestor_of(JobName.from_wire(job_id), include_self=True)


def _token_provider(cluster: str, *, store_path: Path | None = None) -> TokenProvider | None:
    credential = load_token(cluster, store_path=store_path)
    if credential is None:
        credential = load_any_token(store_path=store_path)
    if credential is None:
        return None
    return StaticTokenProvider(credential.token)


def _normalize_state_filter(state: str) -> str:
    normalized = state.strip().lower()
    if normalized.startswith("job_state_"):
        return normalized.removeprefix("job_state_")
    return normalized


def _log_source(target: str, attempt_id: int) -> str:
    if target.startswith("/system/"):
        return target
    return build_log_source(JobName.from_wire(target), attempt_id)


def _profile_type(profile_type: str, *, include_locals: bool) -> job_pb2.ProfileType:
    if profile_type == "threads":
        return job_pb2.ProfileType(threads=job_pb2.ThreadsProfile(locals=include_locals))
    if profile_type == "cpu":
        return job_pb2.ProfileType(cpu=job_pb2.CpuProfile(format=job_pb2.CpuProfile.SPEEDSCOPE))
    if profile_type == "mem":
        return job_pb2.ProfileType(memory=job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.FLAMEGRAPH))
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
    def iris_bug_report(job_id: str, tail: int = 50) -> dict[str, Any]:
        return service.bug_report(job_id=job_id, tail=tail)

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
