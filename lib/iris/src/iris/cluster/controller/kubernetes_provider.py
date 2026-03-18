# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KubernetesProvider: DirectTaskProvider that executes tasks as Kubernetes Pods.

No worker daemon, no synthetic worker row. The controller talks directly to the
k8s API via kubectl, launching one Pod per task attempt.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from iris.cluster.controller.direct_provider import ClusterCapacity, DirectProviderSyncResult, SchedulingEvent
from iris.cluster.controller.transitions import DirectProviderBatch, RunningTaskEntry, TaskUpdate
from iris.cluster.k8s.constants import CW_INTERRUPTABLE_TOLERATION
from iris.cluster.k8s.kubectl import Kubectl, KubectlLogLine
from iris.cluster.types import JobName, get_gpu_count
from iris.rpc import cluster_pb2, logging_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

# Label key prefix for iris-managed pod identification.
_LABEL_MANAGED = "iris.managed"
_LABEL_RUNTIME = "iris.runtime"
_LABEL_TASK_ID = "iris.task_id"
_LABEL_ATTEMPT_ID = "iris.attempt_id"
# Collision-resistant hash of the full (unsanitized) task_id; 16 hex chars (64 bits).
_LABEL_TASK_HASH = "iris.task_hash"
_LABEL_JOB_ID = "iris.job_id"

# Runtime identifier for pods created by KubernetesProvider.
_RUNTIME_LABEL_VALUE = "iris-kubernetes"

# Max pod name length is 253 chars in k8s. We stay well under it.
_MAX_POD_NAME_LEN = 63

# CoreWeave nodes are labeled with {label_prefix}.{attribute_key} by the NodePool.
# Map well-known Iris constraint keys to their k8s node label keys.
# The "iris." prefix matches platform.label_prefix in coreweave.yaml.
_CONSTRAINT_KEY_TO_NODE_LABEL: dict[str, str] = {
    "pool": "iris.pool",
    "region": "iris.region",
}

# Kubernetes label values: max 63 chars, alphanumeric plus [-_.], must start/end alphanumeric.
_K8S_LABEL_MAX_LEN = 63


def _constraints_to_node_selector(
    constraints: Sequence[cluster_pb2.Constraint],
) -> dict[str, str]:
    """Map Iris constraints to k8s nodeSelector entries.

    Only EQ constraints with known label keys are mapped. Other ops/keys
    are silently skipped (affinity rules for complex constraints not yet implemented).
    """
    node_selector: dict[str, str] = {}
    for c in constraints:
        label_key = _CONSTRAINT_KEY_TO_NODE_LABEL.get(c.key)
        if label_key is None:
            continue
        if c.op == cluster_pb2.CONSTRAINT_OP_EQ and c.HasField("value"):
            node_selector[label_key] = c.value.string_value
        else:
            logger.warning("Unsupported constraint op=%s for key=%s; skipping nodeSelector", c.op, c.key)
    return node_selector


def _task_hash(task_id: str) -> str:
    """Return a 16-hex-char SHA-256 hash of task_id, safe as a k8s label value."""
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def _sanitize_label_value(value: str) -> str:
    """Sanitize a string for use as a Kubernetes label value."""
    sanitized = []
    for ch in value:
        if ch.isalnum() or ch in "-_.":
            sanitized.append(ch)
        else:
            sanitized.append(".")
    result = "".join(sanitized)
    result = result.strip("-_.")
    if len(result) > _K8S_LABEL_MAX_LEN:
        result = result[:_K8S_LABEL_MAX_LEN].rstrip("-_.")
    return result or "unknown"


def _job_id_from_task(task_id: JobName) -> str:
    """Extract job path from task wire ID.

    Task IDs are of the form '/job-name/task-N'. The job_id is the parent
    path without the task suffix, sanitized for use as a k8s label value.
    """
    wire = task_id.to_wire()
    parent = wire.rsplit("/", 1)[0] if "/" in wire else wire
    return _sanitize_label_value(parent) if parent else "unknown"


def _pod_name(task_id: JobName, attempt_id: int) -> str:
    """Build a DNS-label-safe pod name from task_id and attempt_id.

    k8s pod names must match [a-z0-9][a-z0-9-]* and be at most 253 chars.
    We lowercase and replace non-alphanumeric chars with hyphens, then truncate.

    Both a 8-char task hash and the attempt_id are reserved before truncating
    the readable prefix, so:
    - Different task IDs with the same long prefix cannot share a pod name
      (the task hash distinguishes them).
    - Different retry attempts of the same task cannot share a pod name
      (the attempt_id distinguishes them).
    """
    task_id_wire = task_id.to_wire()
    # 8-char hash ensures different task IDs produce different pod names
    # even after prefix truncation.
    hash8 = hashlib.sha256(task_id_wire.encode()).hexdigest()[:8]
    suffix = f"-{hash8}-{attempt_id}"
    prefix_raw = f"iris-{task_id_wire}"
    prefix = re.sub(r"[^a-z0-9-]", "-", prefix_raw.lower())
    prefix = re.sub(r"-{2,}", "-", prefix).strip("-")
    max_prefix_len = _MAX_POD_NAME_LEN - len(suffix)
    if len(prefix) > max_prefix_len:
        prefix = prefix[:max_prefix_len].rstrip("-")
    return (prefix + suffix) if prefix else f"iris-task{suffix}"


def _build_pod_manifest(
    run_req: cluster_pb2.Worker.RunTaskRequest,
    namespace: str,
    default_image: str,
    colocation_topology_key: str = "",
) -> dict:
    """Build a Pod manifest dict from a RunTaskRequest."""
    task_id = JobName.from_wire(run_req.task_id)
    attempt_id = run_req.attempt_id
    pod_name = _pod_name(task_id, attempt_id)

    env_list = [{"name": k, "value": v} for k, v in run_req.environment.env_vars.items()]

    container: dict = {
        "name": "task",
        "image": default_image,
        "imagePullPolicy": "IfNotPresent",
        "env": env_list,
    }

    if run_req.entrypoint.run_command.argv:
        container["command"] = list(run_req.entrypoint.run_command.argv)

    resources: dict = {}
    gpu_count = 0
    if run_req.HasField("resources"):
        res = run_req.resources
        limits: dict[str, str] = {}
        if res.cpu_millicores:
            limits["cpu"] = f"{res.cpu_millicores}m"
        if res.memory_bytes:
            limits["memory"] = str(res.memory_bytes)
        if res.HasField("device"):
            gpu_count = get_gpu_count(res.device)
            if gpu_count > 0:
                limits["nvidia.com/gpu"] = str(gpu_count)
        if limits:
            resources["limits"] = limits
        if res.disk_bytes:
            disk_gi = max(1, res.disk_bytes // (1024**3))
            resources.setdefault("requests", {})["ephemeral-storage"] = f"{disk_gi}Gi"
            resources.setdefault("limits", {})["ephemeral-storage"] = f"{disk_gi}Gi"
    if resources:
        container["resources"] = resources

    job_id = _job_id_from_task(task_id)
    metadata = {
        "name": pod_name,
        "namespace": namespace,
        "labels": {
            _LABEL_MANAGED: "true",
            _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
            _LABEL_TASK_ID: _sanitize_label_value(run_req.task_id),
            _LABEL_ATTEMPT_ID: str(attempt_id),
            _LABEL_TASK_HASH: _task_hash(run_req.task_id),
            _LABEL_JOB_ID: job_id,
        },
    }

    spec: dict = {
        "restartPolicy": "Never",
        "containers": [container],
    }

    node_selector = _constraints_to_node_selector(run_req.constraints)
    if node_selector:
        spec["nodeSelector"] = node_selector

    if gpu_count > 0:
        spec.setdefault("tolerations", []).append(CW_INTERRUPTABLE_TOLERATION)

    # Prefer co-locating sibling task pods on the same network spine for IB connectivity.
    if run_req.num_tasks > 1 and colocation_topology_key:
        spec["affinity"] = {
            "podAffinity": {
                "preferredDuringSchedulingIgnoredDuringExecution": [
                    {
                        "weight": 100,
                        "podAffinityTerm": {
                            "labelSelector": {
                                "matchLabels": {
                                    _LABEL_JOB_ID: job_id,
                                },
                            },
                            "topologyKey": colocation_topology_key,
                        },
                    }
                ],
            }
        }

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": spec,
    }


def _kubectl_log_line_to_log_entry(kll: KubectlLogLine, attempt_id: int) -> logging_pb2.LogEntry:
    entry = logging_pb2.LogEntry(source=kll.stream, data=kll.data, attempt_id=attempt_id)
    entry.timestamp.CopyFrom(Timestamp.from_seconds(kll.timestamp.timestamp()).to_proto())
    return entry


def _task_update_from_pod(entry: RunningTaskEntry, pod: dict) -> TaskUpdate:
    """Build a TaskUpdate from a Kubernetes Pod dict."""
    phase = pod.get("status", {}).get("phase", "Unknown")
    task_id = entry.task_id
    attempt_id = entry.attempt_id

    if phase in ("Pending", "Running"):
        return TaskUpdate(
            task_id=task_id,
            attempt_id=attempt_id,
            new_state=cluster_pb2.TASK_STATE_RUNNING,
        )

    if phase == "Succeeded":
        return TaskUpdate(
            task_id=task_id,
            attempt_id=attempt_id,
            new_state=cluster_pb2.TASK_STATE_SUCCEEDED,
        )

    # Failed or Unknown
    exit_code = _extract_exit_code(pod)
    return TaskUpdate(
        task_id=task_id,
        attempt_id=attempt_id,
        new_state=cluster_pb2.TASK_STATE_FAILED,
        exit_code=exit_code,
        error=_extract_error(pod),
    )


def _extract_exit_code(pod: dict) -> int | None:
    """Extract exit code from the first container's terminated state."""
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if statuses:
        terminated = statuses[0].get("state", {}).get("terminated", {})
        code = terminated.get("exitCode")
        if isinstance(code, int):
            return code
    return None


def _extract_error(pod: dict) -> str | None:
    """Extract error reason/message from pod container statuses."""
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if not statuses:
        return pod.get("status", {}).get("reason") or None
    terminated = statuses[0].get("state", {}).get("terminated", {})
    reason = terminated.get("reason", "")
    message = terminated.get("message", "")
    if reason == "Completed":
        return message or None
    return message or reason or None


def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    if n >= 2**30:
        return f"{n / 2**30:.1f} GiB"
    if n >= 2**20:
        return f"{n / 2**20:.1f} MiB"
    if n >= 2**10:
        return f"{n / 2**10:.1f} KiB"
    return f"{n} B"


def _parse_k8s_quantity(value: str) -> int:
    """Parse a Kubernetes resource quantity string to an integer.

    Examples: "2" -> 2, "500m" -> 500 (millicores), "4Gi" -> 4294967296 (bytes).
    Quantities ending in 'm' are millicores for CPU, or use standard SI/binary suffixes.
    """
    if not value:
        return 0
    binary_suffixes = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}
    si_suffixes = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}
    for suffix, mult in binary_suffixes.items():
        if value.endswith(suffix):
            return int(value[: -len(suffix)]) * mult
    for suffix, mult in si_suffixes.items():
        if value.endswith(suffix) and not value.endswith("i"):
            return int(value[: -len(suffix)]) * mult
    if value.endswith("m"):
        return int(value[:-1])
    return int(value)


@dataclass
class KubernetesProvider:
    """DirectTaskProvider that executes tasks as Kubernetes Pods.

    No worker daemon. Capacity is derived from node allocatable resources
    minus running pod resource requests, queried via kubectl each sync cycle.

    Pod naming: iris-{task_id_sanitized}-{attempt_id}
    """

    kubectl: Kubectl
    namespace: str
    default_image: str
    colocation_topology_key: str = "coreweave.cloud/spine"

    @property
    def is_direct_provider(self) -> bool:
        return True

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        """Sync task state: apply new pods, delete killed pods, poll running pods."""
        try:
            for run_req in batch.tasks_to_run:
                self._apply_pod(run_req)
            for task_id in batch.tasks_to_kill:
                self._delete_pods_by_task_id(task_id)
            updates = self._poll_pods(batch.running_tasks)
            capacity = self._query_capacity()
            scheduling_events = self._fetch_scheduling_events()
            return DirectProviderSyncResult(updates=updates, scheduling_events=scheduling_events, capacity=capacity)
        except Exception as e:
            logger.warning("KubernetesProvider sync failed: %s", e)
            return DirectProviderSyncResult()

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs from a task pod.

        Cursor semantics:
        - Primary path (running pod): cursor is a byte offset into the stdout
          stream, returned as the next cursor.
        - Fallback path (terminated pod): all logs are replayed from the start
          regardless of cursor, because byte offsets cannot map to line indices.
          Returns len(lines) as the next cursor so subsequent calls are no-ops.
        """
        pod_name = _pod_name(JobName.from_wire(task_id), attempt_id)
        result = self.kubectl.stream_logs(pod_name, container="task", byte_offset=cursor)
        entries = [_kubectl_log_line_to_log_entry(kll, attempt_id) for kll in result.lines]

        if not entries:
            # Pod may have terminated; try fetching the complete logs.
            # The primary path cursor is a byte offset that cannot be reliably
            # mapped to a line index, so we always replay all terminated-pod
            # logs from the start. The returned cursor equals len(lines) so
            # subsequent calls return nothing.
            raw = self.kubectl.logs(pod_name, container="task", previous=True, tail=-1)
            if raw:
                lines = raw.splitlines()
                sliced = lines[:max_lines] if max_lines > 0 else lines
                now_ts = Timestamp.now()
                fallback_entries: list[logging_pb2.LogEntry] = []
                for line in sliced:
                    entry = logging_pb2.LogEntry(source="stdout", data=line, attempt_id=attempt_id)
                    entry.timestamp.CopyFrom(now_ts.to_proto())
                    fallback_entries.append(entry)
                return fallback_entries, len(lines)

        if max_lines > 0:
            entries = entries[:max_lines]

        return entries, result.byte_offset

    def close(self) -> None:
        """No persistent resources to release."""

    def get_cluster_status(self) -> cluster_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Query Kubernetes for cluster-level status: node counts, capacity, and recent pod statuses."""
        nodes: list[dict] = []
        try:
            nodes = self.kubectl.list_json("nodes", cluster_scoped=True)
        except Exception as e:
            logger.warning("Failed to query nodes: %s", e)

        total_nodes = len(nodes)
        schedulable_nodes = 0
        total_cpu_mc = 0
        total_memory_bytes = 0
        for node in nodes:
            spec = node.get("spec", {})
            taints = spec.get("taints", [])
            if any(t.get("effect") in ("NoSchedule", "NoExecute") for t in taints):
                continue
            schedulable_nodes += 1
            allocatable = node.get("status", {}).get("allocatable", {})
            cpu_str = allocatable.get("cpu", "0")
            cpu_val = _parse_k8s_quantity(cpu_str)
            if not cpu_str.endswith("m"):
                cpu_val *= 1000
            total_cpu_mc += cpu_val
            total_memory_bytes += _parse_k8s_quantity(allocatable.get("memory", "0"))

        allocatable_cpu = f"{total_cpu_mc / 1000:.1f} cores" if total_cpu_mc else "0 cores"
        allocatable_memory = _format_bytes(total_memory_bytes)

        pod_statuses: list[cluster_pb2.Controller.KubernetesPodStatus] = []
        try:
            pods = self.kubectl.list_json(
                "pods",
                labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE},
            )
            for pod in pods:
                meta = pod.get("metadata", {})
                pod_name = meta.get("name", "")
                labels = meta.get("labels", {})
                task_id = labels.get(_LABEL_TASK_ID, "")
                phase = pod.get("status", {}).get("phase", "Unknown")
                reason = ""
                message = ""
                last_ts = Timestamp.now()

                container_statuses = pod.get("status", {}).get("containerStatuses", [])
                if container_statuses:
                    state = container_statuses[0].get("state", {})
                    for state_name in ("waiting", "terminated"):
                        if state_name in state:
                            reason = state[state_name].get("reason", "")
                            message = state[state_name].get("message", "")
                            break
                if not reason:
                    conditions = pod.get("status", {}).get("conditions", [])
                    for cond in conditions:
                        if cond.get("status") == "False":
                            reason = cond.get("reason", "")
                            message = cond.get("message", "")
                            last_transition_str = cond.get("lastTransitionTime", "")
                            if last_transition_str:
                                try:
                                    dt = datetime.fromisoformat(last_transition_str.replace("Z", "+00:00"))
                                    last_ts = Timestamp.from_seconds(dt.timestamp())
                                except (ValueError, AttributeError):
                                    pass
                            break

                ps = cluster_pb2.Controller.KubernetesPodStatus(
                    pod_name=pod_name,
                    task_id=task_id,
                    phase=phase,
                    reason=reason,
                    message=message,
                )
                ps.last_transition.CopyFrom(last_ts.to_proto())
                pod_statuses.append(ps)
        except Exception as e:
            logger.warning("Failed to query pod statuses: %s", e)

        return cluster_pb2.Controller.GetKubernetesClusterStatusResponse(
            namespace=self.namespace,
            total_nodes=total_nodes,
            schedulable_nodes=schedulable_nodes,
            allocatable_cpu=allocatable_cpu,
            allocatable_memory=allocatable_memory,
            pod_statuses=pod_statuses,
            provider_version="iris-kubernetes/v1",
        )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _apply_pod(self, run_req: cluster_pb2.Worker.RunTaskRequest) -> None:
        """Create or update the Pod for a task attempt."""
        manifest = _build_pod_manifest(run_req, self.namespace, self.default_image, self.colocation_topology_key)
        self.kubectl.apply_json(manifest)
        task_id = run_req.task_id
        logger.info(
            "Applied pod %s for task %s attempt %d",
            manifest["metadata"]["name"],
            task_id,
            run_req.attempt_id,
        )

    def _delete_pods_by_task_id(self, task_id: str) -> None:
        """Delete all pods for a given task_id (any attempt).

        Uses the SHA-256 task hash label for collision-resistant pod lookup,
        avoiding false matches that _sanitize_label_value's lossy truncation
        could cause when distinct task IDs share the same sanitized prefix.
        """
        task_hash = _task_hash(task_id)
        pods = self.kubectl.list_json(
            "pods",
            labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, _LABEL_TASK_HASH: task_hash},
        )
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            if pod_name:
                self.kubectl.delete("pod", pod_name)
                logger.info("Deleted pod %s for task %s", pod_name, task_id)

    def _poll_pods(self, running: list[RunningTaskEntry]) -> list[TaskUpdate]:
        """Poll pod phases for all running tasks and build TaskUpdates."""
        if not running:
            return []

        all_pods = self.kubectl.list_json(
            "pods",
            labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE},
        )
        pods_by_name: dict[str, dict] = {pod.get("metadata", {}).get("name", ""): pod for pod in all_pods}

        updates: list[TaskUpdate] = []
        for entry in running:
            pod_name = _pod_name(entry.task_id, entry.attempt_id)
            pod = pods_by_name.get(pod_name)
            if pod is None:
                updates.append(
                    TaskUpdate(
                        task_id=entry.task_id,
                        attempt_id=entry.attempt_id,
                        new_state=cluster_pb2.TASK_STATE_WORKER_FAILED,
                        error="Pod not found",
                    )
                )
                continue

            update = _task_update_from_pod(entry, pod)

            # Eagerly fetch logs on terminal pod to prevent log loss on controller crash.
            phase = pod.get("status", {}).get("phase", "")
            if phase in ("Succeeded", "Failed"):
                log_entries = self._fetch_completed_pod_logs(pod_name, entry.attempt_id)
                update = TaskUpdate(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    new_state=update.new_state,
                    error=update.error,
                    exit_code=update.exit_code,
                    resource_usage=update.resource_usage,
                    log_entries=log_entries,
                )

            updates.append(update)

        return updates

    def _fetch_completed_pod_logs(self, pod_name: str, attempt_id: int) -> list[logging_pb2.LogEntry]:
        """Fetch all logs from a completed pod."""
        try:
            result = self.kubectl.stream_logs(pod_name, container="task", byte_offset=0)
            return [_kubectl_log_line_to_log_entry(kll, attempt_id) for kll in result.lines]
        except Exception as e:
            logger.warning("Failed to fetch logs for completed pod %s: %s", pod_name, e)
            return []

    def _query_capacity(self) -> ClusterCapacity | None:
        """Compute cluster capacity from node allocatable minus running pod requests."""
        try:
            nodes = self.kubectl.list_json("nodes", cluster_scoped=True)
        except Exception as e:
            logger.warning("Failed to query node resources: %s", e)
            return None

        total_cpu_mc = 0
        total_memory_bytes = 0
        schedulable_count = 0
        for node in nodes:
            spec = node.get("spec", {})
            taints = spec.get("taints", [])
            if any(t.get("effect") in ("NoSchedule", "NoExecute") for t in taints):
                continue
            allocatable = node.get("status", {}).get("allocatable", {})
            cpu_str = allocatable.get("cpu", "0")
            # Node allocatable CPU is in cores (e.g. "4"), convert to millicores.
            cpu_val = _parse_k8s_quantity(cpu_str)
            if not cpu_str.endswith("m"):
                cpu_val *= 1000
            total_cpu_mc += cpu_val
            total_memory_bytes += _parse_k8s_quantity(allocatable.get("memory", "0"))
            schedulable_count += 1

        if schedulable_count == 0:
            return None

        # Compute used resources from running pod requests.
        used_cpu_mc = 0
        used_memory_bytes = 0
        try:
            pods = self.kubectl.list_json(
                "pods",
                labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE},
            )
            for pod in pods:
                phase = pod.get("status", {}).get("phase", "")
                if phase not in ("Pending", "Running"):
                    continue
                for container in pod.get("spec", {}).get("containers", []):
                    requests = container.get("resources", {}).get("requests", {})
                    limits = container.get("resources", {}).get("limits", {})
                    # Use requests if available, else limits.
                    cpu_req = requests.get("cpu") or limits.get("cpu", "0")
                    mem_req = requests.get("memory") or limits.get("memory", "0")
                    cpu_v = _parse_k8s_quantity(cpu_req)
                    if not cpu_req.endswith("m"):
                        cpu_v *= 1000
                    used_cpu_mc += cpu_v
                    used_memory_bytes += _parse_k8s_quantity(mem_req)
        except Exception as e:
            logger.warning("Failed to query pod resources for capacity: %s", e)

        return ClusterCapacity(
            schedulable_nodes=schedulable_count,
            total_cpu_millicores=total_cpu_mc,
            available_cpu_millicores=total_cpu_mc - used_cpu_mc,
            total_memory_bytes=total_memory_bytes,
            available_memory_bytes=total_memory_bytes - used_memory_bytes,
        )

    def _fetch_scheduling_events(self) -> list[SchedulingEvent]:
        """Fetch recent k8s events for iris-managed pods."""
        try:
            events = self.kubectl.list_json(
                "events",
                labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE},
            )
        except Exception as e:
            logger.warning("Failed to fetch scheduling events: %s", e)
            return []

        result: list[SchedulingEvent] = []
        for ev in events:
            metadata = ev.get("metadata", {})
            labels = metadata.get("labels", {})
            task_id = labels.get(_LABEL_TASK_ID, "")
            attempt_str = labels.get(_LABEL_ATTEMPT_ID, "0")
            try:
                attempt_id = int(attempt_str)
            except (ValueError, TypeError):
                attempt_id = 0

            ts = Timestamp.now()

            result.append(
                SchedulingEvent(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    event_type=ev.get("type", "Normal"),
                    reason=ev.get("reason", ""),
                    message=ev.get("message", ""),
                    timestamp=ts,
                )
            )
        return result
