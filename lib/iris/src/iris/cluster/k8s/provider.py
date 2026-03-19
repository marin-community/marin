# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KubernetesProvider: executes tasks as Kubernetes Pods.

No worker daemon, no synthetic worker row. The controller talks directly to the
k8s API via kubectl, launching one Pod per task attempt.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from iris.cluster.controller.transitions import ClusterCapacity, DirectProviderSyncResult, SchedulingEvent
from iris.cluster.controller.transitions import DirectProviderBatch, RunningTaskEntry, TaskUpdate
from iris.cluster.k8s.constants import CW_INTERRUPTABLE_TOLERATION, NVIDIA_GPU_TOLERATION
from iris.cluster.k8s.kubectl import Kubectl, KubectlLogLine
from iris.cluster.runtime.env import build_common_iris_env, normalize_workdir_relative_path
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

# Number of consecutive sync cycles where a pod is missing from the k8s API
# before declaring FAILED. Avoids false positives from transient API misses.
_POD_NOT_FOUND_GRACE_CYCLES = 3

# Kubernetes terminated reasons that indicate infrastructure failure (not application error).
# OOMKilled: container exceeded memory limits or node ran out of memory.
# Evicted: kubelet evicted the pod due to resource pressure.
# DeadlineExceeded: pod's activeDeadlineSeconds expired.
# Preempting: scheduler preempted the pod for a higher-priority workload.
_INFRASTRUCTURE_FAILURE_REASONS = frozenset({"OOMKilled", "Evicted", "DeadlineExceeded", "Preempting"})


def _constraints_to_node_selector(
    constraints: Sequence[cluster_pb2.Constraint],
) -> dict[str, str]:
    """Map Iris constraints to k8s nodeSelector entries.

    Only EQ constraints with known label keys are mapped. Unknown keys are
    silently skipped. Known keys with non-EQ ops raise ValueError.
    """
    node_selector: dict[str, str] = {}
    for c in constraints:
        label_key = _CONSTRAINT_KEY_TO_NODE_LABEL.get(c.key)
        if label_key is None:
            continue
        if c.op == cluster_pb2.CONSTRAINT_OP_EQ and c.HasField("value"):
            node_selector[label_key] = c.value.string_value
        else:
            raise ValueError(
                f"Unsupported constraint op={c.op} for key={c.key!r}: "
                f"only CONSTRAINT_OP_EQ is supported for nodeSelector mapping"
            )
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


_STANDARD_MOUNTS = [
    # (volume_name, container_path, kind)
    ("workdir", "/app", "workdir"),
    ("tmpfs", "/tmp", "tmpfs"),
    ("uv-cache", "/uv/cache", "cache"),
    ("cargo-registry", "/root/.cargo/registry", "cache"),
    ("cargo-target", "/root/.cargo/target", "cache"),
]


def _build_volumes_and_mounts(
    cache_dir: str,
    has_accelerator: bool,
) -> tuple[list[dict], list[dict]]:
    """Build standard pod volumes and container volume mounts.

    Workdir and tmpfs use emptyDir; cache mounts use hostPath so they persist
    across pod restarts on the same node. /dev/shm is memory-backed with a
    generous limit for GPU/TPU multi-process communication.

    NOTE: On CoreWeave bare-metal GPU nodes the root filesystem is a 15GB
    ramdisk. Set cache_dir to a path on the NVMe (e.g. /mnt/local/iris-cache)
    to avoid running out of space installing torch+CUDA.
    """
    volumes: list[dict] = []
    mounts: list[dict] = []
    for name, path, kind in _STANDARD_MOUNTS:
        if kind in ("workdir", "tmpfs"):
            volumes.append({"name": name, "emptyDir": {}})
        else:
            volumes.append(
                {
                    "name": name,
                    "hostPath": {
                        "path": f"{cache_dir}/{path.strip('/').replace('/', '-')}",
                        "type": "DirectoryOrCreate",
                    },
                }
            )
        mounts.append({"name": name, "mountPath": path})

    shm_spec: dict = {"medium": "Memory"}
    if has_accelerator:
        shm_spec["sizeLimit"] = "100Gi"
    volumes.append({"name": "dshm", "emptyDir": shm_spec})
    mounts.append({"name": "dshm", "mountPath": "/dev/shm"})

    return volumes, mounts


@dataclass(frozen=True)
class PodConfig:
    """Non-request parameters for pod manifest construction.

    Bundles the cluster-level settings that _build_pod_manifest needs beyond
    the RunTaskRequest itself, avoiding a long positional parameter list.
    """

    namespace: str
    default_image: str
    colocation_topology_key: str = ""
    cache_dir: str = "/cache"
    service_account: str = ""
    host_network: bool = False
    controller_address: str | None = None
    managed_label: str = ""


def _build_task_script(run_req: cluster_pb2.Worker.RunTaskRequest) -> str:
    """Build a shell script that runs setup_commands then the run_command."""
    lines = ["set -e", "ulimit -c 0", "mkdir -p /app", "cd /app"]
    for cmd in run_req.entrypoint.setup_commands:
        lines.append(cmd)
    if run_req.entrypoint.run_command.argv:
        lines.append("exec " + shlex.join(run_req.entrypoint.run_command.argv))
    return "\n".join(lines)


def _build_init_container_spec(
    run_req: cluster_pb2.Worker.RunTaskRequest,
    pod_name: str,
    default_image: str,
    controller_address: str | None,
) -> tuple[list[dict], list[dict], str | None]:
    """Build init containers for bundle fetch and workdir file staging.

    Returns (init_containers, extra_volumes, configmap_name_or_None).
    The init container runs a standalone Python script that downloads the
    bundle zip from the controller and copies workdir files from a ConfigMap.
    """
    has_bundle = bool(run_req.bundle_id) and bool(controller_address)
    workdir_files = dict(run_req.entrypoint.workdir_files)
    if not has_bundle and not workdir_files:
        return [], [], None

    script_path = Path(__file__).with_name("bundle_fetch.py")
    bundle_script = script_path.read_text()

    init_env: list[dict] = [{"name": "IRIS_WORKDIR", "value": "/app"}]
    init_mounts: list[dict] = [{"name": "workdir", "mountPath": "/app"}]
    extra_volumes: list[dict] = []
    configmap_name: str | None = None

    if has_bundle:
        init_env.extend(
            [
                {"name": "IRIS_BUNDLE_ID", "value": run_req.bundle_id},
                {"name": "IRIS_CONTROLLER_URL", "value": controller_address},
            ]
        )

    if workdir_files:
        configmap_name = f"{pod_name}-wf"
        extra_volumes.append(
            {
                "name": "workdir-files",
                "configMap": {
                    "name": configmap_name,
                    "items": [
                        {"key": f"f{i:04d}", "path": normalize_workdir_relative_path(name)}
                        for i, name in enumerate(workdir_files)
                    ],
                },
            }
        )
        init_mounts.append(
            {
                "name": "workdir-files",
                "mountPath": "/iris/staged-workdir-files",
                "readOnly": True,
            }
        )
        init_env.append({"name": "IRIS_WORKDIR_FILES_SRC", "value": "/iris/staged-workdir-files"})

    init_containers = [
        {
            "name": "stage-workdir",
            "image": default_image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["python", "-c", bundle_script],
            "env": init_env,
            "volumeMounts": init_mounts,
        }
    ]

    return init_containers, extra_volumes, configmap_name


def _build_pod_manifest(
    run_req: cluster_pb2.Worker.RunTaskRequest,
    config: PodConfig,
) -> dict:
    """Build a Pod manifest dict from a RunTaskRequest and cluster config."""
    task_id = JobName.from_wire(run_req.task_id)
    attempt_id = run_req.attempt_id
    pod_name = _pod_name(task_id, attempt_id)

    namespace = config.namespace
    default_image = config.default_image
    colocation_topology_key = config.colocation_topology_key
    cache_dir = config.cache_dir
    service_account = config.service_account
    host_network = config.host_network
    managed_label = config.managed_label

    # User env vars as base, then iris system env vars override.
    iris_env = build_common_iris_env(
        task_id=run_req.task_id,
        attempt_id=run_req.attempt_id,
        num_tasks=run_req.num_tasks,
        bundle_id=run_req.bundle_id,
        controller_address=config.controller_address,
        environment=run_req.environment,
        constraints=run_req.constraints,
        ports=run_req.ports,
        resources=run_req.resources if run_req.HasField("resources") else None,
        service_account=run_req.service_account or None,
    )
    combined = {**dict(run_req.environment.env_vars), **iris_env}
    env_list: list[dict] = [{"name": k, "value": v} for k, v in combined.items()]
    # Pod IP via downward API — not expressible as a static value.
    env_list.append(
        {
            "name": "IRIS_ADVERTISE_HOST",
            "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
        }
    )

    # Parse resources first so device info is known before building volumes.
    resources: dict = {}
    gpu_count = 0
    has_tpu = False
    if run_req.HasField("resources"):
        res = run_req.resources
        limits: dict[str, str] = {}
        if res.cpu_millicores:
            limits["cpu"] = f"{res.cpu_millicores}m"
        if res.memory_bytes:
            limits["memory"] = str(res.memory_bytes)
        if res.HasField("device"):
            gpu_count = get_gpu_count(res.device)
            has_tpu = res.device.HasField("tpu")
            if gpu_count > 0:
                limits["nvidia.com/gpu"] = str(gpu_count)
                if host_network:
                    # Request RDMA/IB devices for multi-host NCCL over InfiniBand.
                    limits["rdma/ib"] = str(gpu_count)
        if limits:
            resources["limits"] = limits
        if res.disk_bytes:
            disk_gi = max(1, res.disk_bytes // (1024**3))
            resources.setdefault("requests", {})["ephemeral-storage"] = f"{disk_gi}Gi"
            resources.setdefault("limits", {})["ephemeral-storage"] = f"{disk_gi}Gi"

    has_accelerator = gpu_count > 0 or has_tpu
    volumes, vol_mounts = _build_volumes_and_mounts(cache_dir, has_accelerator=has_accelerator)

    container: dict = {
        "name": "task",
        "image": default_image,
        "imagePullPolicy": "IfNotPresent",
        "env": env_list,
        "workingDir": "/app",
        "volumeMounts": vol_mounts,
        "command": ["bash", "-lc", _build_task_script(run_req)],
    }

    # SYS_PTRACE for profiling; SYS_RESOURCE for TPU memlock ulimits.
    capabilities = ["SYS_PTRACE"]
    if has_tpu:
        capabilities.append("SYS_RESOURCE")
    container["securityContext"] = {"capabilities": {"add": capabilities}}

    if resources:
        container["resources"] = resources

    job_id = _job_id_from_task(task_id)
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_ID: _sanitize_label_value(run_req.task_id),
        _LABEL_ATTEMPT_ID: str(attempt_id),
        _LABEL_TASK_HASH: _task_hash(run_req.task_id),
        _LABEL_JOB_ID: job_id,
    }
    if managed_label:
        labels[managed_label] = "true"
    metadata = {
        "name": pod_name,
        "namespace": namespace,
        "labels": labels,
    }

    spec: dict = {
        "restartPolicy": "Never",
        "containers": [container],
        "volumes": volumes,
    }

    node_selector = _constraints_to_node_selector(run_req.constraints)
    if managed_label:
        node_selector[managed_label] = "true"
    if node_selector:
        spec["nodeSelector"] = node_selector

    if gpu_count > 0:
        spec.setdefault("tolerations", []).extend(
            [
                CW_INTERRUPTABLE_TOLERATION,
                NVIDIA_GPU_TOLERATION,
            ]
        )

    if service_account:
        spec["serviceAccountName"] = service_account
    if host_network:
        spec["hostNetwork"] = True
        spec["dnsPolicy"] = "ClusterFirstWithHostNet"

    if run_req.HasField("timeout") and run_req.timeout.milliseconds > 0:
        spec["activeDeadlineSeconds"] = max(1, run_req.timeout.milliseconds // 1000)

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


def _is_infrastructure_failure(pod: dict) -> bool:
    """Check if the pod failure was caused by infrastructure (OOM, eviction, etc.).

    Returns True when the terminated reason indicates the failure was NOT caused
    by the application itself, so it should be classified as a worker/preemption
    failure rather than an application failure.
    """
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if not statuses:
        # Pod-level eviction: the pod status reason indicates infrastructure.
        pod_reason = pod.get("status", {}).get("reason", "")
        return pod_reason in _INFRASTRUCTURE_FAILURE_REASONS
    terminated = statuses[0].get("state", {}).get("terminated", {})
    return terminated.get("reason", "") in _INFRASTRUCTURE_FAILURE_REASONS


def _task_update_from_pod(entry: RunningTaskEntry, pod: dict) -> TaskUpdate:
    """Build a TaskUpdate from a Kubernetes Pod dict.

    Infrastructure failures (OOMKilled, eviction) are reported as WORKER_FAILED
    so they count against max_retries_preemption (default: 100).
    Application failures (non-zero exit code) are reported as FAILED so they
    count against max_retries_failure (default: 0, no retries).
    """
    phase = pod.get("status", {}).get("phase", "Unknown")
    task_id = entry.task_id
    attempt_id = entry.attempt_id

    if phase == "Pending":
        return TaskUpdate(
            task_id=task_id,
            attempt_id=attempt_id,
            new_state=cluster_pb2.TASK_STATE_BUILDING,
        )

    if phase == "Running":
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

    # Failed or Unknown — distinguish infrastructure vs application failure.
    exit_code = _extract_exit_code(pod)
    if _is_infrastructure_failure(pod):
        new_state = cluster_pb2.TASK_STATE_WORKER_FAILED
    else:
        new_state = cluster_pb2.TASK_STATE_FAILED
    return TaskUpdate(
        task_id=task_id,
        attempt_id=attempt_id,
        new_state=new_state,
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
            return int(float(value[: -len(suffix)]) * mult)
    for suffix, mult in si_suffixes.items():
        if value.endswith(suffix) and not value.endswith("i"):
            return int(float(value[: -len(suffix)]) * mult)
    if value.endswith("m"):
        return int(value[:-1])
    return int(float(value))


@dataclass
class KubernetesProvider:
    """Executes tasks as Kubernetes Pods without worker daemons.

    No worker daemon. Capacity is derived from node allocatable resources
    minus running pod resource requests, queried via kubectl each sync cycle.

    Pod naming: iris-{task_id_sanitized}-{attempt_id}
    """

    kubectl: Kubectl
    namespace: str
    default_image: str
    colocation_topology_key: str = "coreweave.cloud/spine"
    cache_dir: str = "/cache"
    service_account: str = ""
    host_network: bool = False
    controller_address: str | None = None
    managed_label: str = ""
    _log_cursors: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pod_not_found_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        """Sync task state: apply new pods, delete killed pods, poll running pods."""
        for run_req in batch.tasks_to_run:
            self._apply_pod(run_req)
        for task_id in batch.tasks_to_kill:
            self._delete_pods_by_task_id(task_id)
        updates = self._poll_pods(batch.running_tasks)
        capacity = self._query_capacity()
        scheduling_events = self._fetch_scheduling_events()
        return DirectProviderSyncResult(updates=updates, scheduling_events=scheduling_events, capacity=capacity)

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

    def profile_task(
        self,
        task_id: str,
        attempt_id: int,
        request: cluster_pb2.ProfileTaskRequest,
    ) -> cluster_pb2.ProfileTaskResponse:
        """Profile a running task pod via kubectl exec."""
        pod_name = _pod_name(JobName.from_wire(task_id), attempt_id)
        duration = request.duration_seconds or 10
        profile_type = request.profile_type

        try:
            if profile_type.HasField("threads"):
                return self._profile_threads(pod_name, profile_type.threads)
            elif profile_type.HasField("cpu"):
                return self._profile_cpu(pod_name, profile_type.cpu, duration)
            elif profile_type.HasField("memory"):
                return self._profile_memory(pod_name, profile_type.memory, duration)
            else:
                return cluster_pb2.ProfileTaskResponse(error="Unknown profile type")
        except Exception as e:
            return cluster_pb2.ProfileTaskResponse(error=str(e))

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
                node_name = pod.get("spec", {}).get("nodeName", "")
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
                    node_name=node_name,
                )
                ps.last_transition.CopyFrom(last_ts.to_proto())
                pod_statuses.append(ps)
        except Exception as e:
            logger.warning("Failed to query pod statuses: %s", e)

        node_pools: list[cluster_pb2.Controller.NodePoolStatus] = []
        try:
            np_labels = {self.managed_label: "true"} if self.managed_label else None
            pools = self.kubectl.list_json("nodepools", labels=np_labels, cluster_scoped=True)
            for pool in pools:
                meta = pool.get("metadata", {})
                pool_labels = meta.get("labels", {})
                spec = pool.get("spec", {})
                status = pool.get("status", {})
                scale_group = ""
                for lk, lv in pool_labels.items():
                    if "scale-group" in lk:
                        scale_group = lv
                        break
                node_pools.append(
                    cluster_pb2.Controller.NodePoolStatus(
                        name=meta.get("name", ""),
                        instance_type=spec.get("instanceType", ""),
                        scale_group=scale_group,
                        target_nodes=spec.get("targetNodes", 0),
                        current_nodes=status.get("currentNodes", 0),
                        queued_nodes=status.get("queuedNodes", 0),
                        in_progress_nodes=status.get("inProgressNodes", 0),
                        autoscaling=spec.get("autoscaling", False),
                        min_nodes=spec.get("minNodes", 0),
                        max_nodes=spec.get("maxNodes", 0),
                        capacity=status.get("capacity", ""),
                        quota=status.get("quota", ""),
                    )
                )
        except Exception as e:
            logger.warning("Failed to query nodepools: %s", e)

        return cluster_pb2.Controller.GetKubernetesClusterStatusResponse(
            namespace=self.namespace,
            total_nodes=total_nodes,
            schedulable_nodes=schedulable_nodes,
            allocatable_cpu=allocatable_cpu,
            allocatable_memory=allocatable_memory,
            pod_statuses=pod_statuses,
            provider_version="iris-kubernetes/v1",
            node_pools=node_pools,
        )

    # -------------------------------------------------------------------------
    # Profiling helpers
    # -------------------------------------------------------------------------

    def _kubectl_exec_shell(self, pod_name: str, cmd: str, timeout: float | None = None) -> str:
        """Execute a shell command in a task pod with venv activation.

        Returns stdout. Raises RuntimeError on non-zero exit.
        """
        shell_cmd = ["bash", "-lc", f"source /app/.venv/bin/activate 2>/dev/null; {cmd}"]
        result = self.kubectl.exec(pod_name, shell_cmd, container="task", timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"kubectl exec failed (exit {result.returncode}): {result.stderr}")
        return result.stdout

    def _profile_threads(
        self, pod_name: str, threads_config: cluster_pb2.ThreadsProfile
    ) -> cluster_pb2.ProfileTaskResponse:
        """Get thread stacks via py-spy dump."""
        from iris.cluster.runtime.profile import build_pyspy_dump_cmd

        cmd = shlex.join(build_pyspy_dump_cmd("1", include_locals=threads_config.locals))
        stdout = self._kubectl_exec_shell(pod_name, cmd, timeout=30)
        return cluster_pb2.ProfileTaskResponse(profile_data=stdout.encode("utf-8"))

    def _profile_cpu(
        self, pod_name: str, cpu_config: cluster_pb2.CpuProfile, duration: int
    ) -> cluster_pb2.ProfileTaskResponse:
        """Record CPU profile via py-spy."""
        from iris.cluster.runtime.profile import build_pyspy_cmd, resolve_cpu_spec

        spec = resolve_cpu_spec(cpu_config, duration, pid="1")
        output_path = f"/tmp/iris-profile.{spec.ext}"
        cmd = shlex.join(build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path=output_path))
        self._kubectl_exec_shell(pod_name, cmd, timeout=duration + 30)
        data = self.kubectl.read_file(pod_name, output_path, container="task")
        self.kubectl.rm_files(pod_name, [output_path], container="task")
        return cluster_pb2.ProfileTaskResponse(profile_data=data)

    def _profile_memory(
        self, pod_name: str, memory_config: cluster_pb2.MemoryProfile, duration: int
    ) -> cluster_pb2.ProfileTaskResponse:
        """Record memory profile via memray."""
        from iris.cluster.runtime.profile import (
            build_memray_attach_cmd,
            build_memray_transform_cmd,
            resolve_memory_spec,
        )

        spec = resolve_memory_spec(memory_config, duration, pid="1")
        trace_path = "/tmp/iris-memray.bin"
        output_path = f"/tmp/iris-memray.{spec.ext}"

        attach_cmd = shlex.join(build_memray_attach_cmd(spec, memray_bin="memray", trace_path=trace_path))
        self._kubectl_exec_shell(pod_name, attach_cmd, timeout=duration + 30)

        transform_cmd = shlex.join(
            build_memray_transform_cmd(spec, memray_bin="memray", trace_path=trace_path, output_path=output_path)
        )
        transform_stdout = self._kubectl_exec_shell(pod_name, transform_cmd, timeout=30)

        if spec.output_is_file:
            data = self.kubectl.read_file(pod_name, output_path, container="task")
        else:
            data = transform_stdout.encode("utf-8")

        self.kubectl.rm_files(pod_name, [trace_path, output_path], container="task")
        return cluster_pb2.ProfileTaskResponse(profile_data=data)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @property
    def pod_config(self) -> PodConfig:
        """Build PodConfig from provider fields."""
        return PodConfig(
            namespace=self.namespace,
            default_image=self.default_image,
            colocation_topology_key=self.colocation_topology_key,
            cache_dir=self.cache_dir,
            service_account=self.service_account,
            host_network=self.host_network,
            controller_address=self.controller_address,
            managed_label=self.managed_label,
        )

    def _apply_pod(self, run_req: cluster_pb2.Worker.RunTaskRequest) -> None:
        """Create or update the Pod for a task attempt."""
        manifest = _build_pod_manifest(run_req, self.pod_config)

        task_id_name = JobName.from_wire(run_req.task_id)
        pod_name = _pod_name(task_id_name, run_req.attempt_id)

        init_containers, extra_volumes, configmap_name = _build_init_container_spec(
            run_req,
            pod_name,
            self.default_image,
            self.controller_address,
        )

        if configmap_name:
            workdir_files = dict(run_req.entrypoint.workdir_files)
            cm = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": configmap_name,
                    "namespace": self.namespace,
                    "labels": {
                        _LABEL_MANAGED: "true",
                        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
                        _LABEL_TASK_HASH: _task_hash(run_req.task_id),
                        **(({self.managed_label: "true"}) if self.managed_label else {}),
                    },
                },
                "binaryData": {
                    f"f{i:04d}": base64.b64encode(data).decode() for i, (_name, data) in enumerate(workdir_files.items())
                },
            }
            self.kubectl.apply_json(cm)

        if init_containers:
            manifest["spec"]["initContainers"] = init_containers
        if extra_volumes:
            manifest["spec"]["volumes"].extend(extra_volumes)

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

        # Clean up associated ConfigMaps (workdir files).
        configmaps = self.kubectl.list_json(
            "configmaps",
            labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, _LABEL_TASK_HASH: task_hash},
        )
        for cm in configmaps:
            cm_name = cm.get("metadata", {}).get("name", "")
            if cm_name:
                self.kubectl.delete("configmap", cm_name)
                logger.info("Deleted configmap %s for task %s", cm_name, task_id)

    def _fetch_incremental_logs(
        self,
        pod_name: str,
        attempt_id: int,
        cursor_key: str,
    ) -> list[logging_pb2.LogEntry]:
        """Fetch logs from a pod starting at the last-seen byte offset."""
        byte_offset = self._log_cursors.get(cursor_key, 0)
        try:
            result = self.kubectl.stream_logs(pod_name, container="task", byte_offset=byte_offset)
            self._log_cursors[cursor_key] = result.byte_offset
            return [_kubectl_log_line_to_log_entry(kll, attempt_id) for kll in result.lines]
        except Exception as e:
            logger.warning("Failed to fetch logs for pod %s: %s", pod_name, e)
            return []

    def _poll_pods(self, running: list[RunningTaskEntry]) -> list[TaskUpdate]:
        """Poll pod phases for all running tasks and build TaskUpdates.

        Fetches incremental logs every sync cycle (running + terminal pods).
        On terminal pods, a final full-log fetch ensures nothing is missed,
        then the cursor is cleaned up.
        """
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
            cursor_key = f"{entry.task_id.to_wire()}:{entry.attempt_id}"
            pod = pods_by_name.get(pod_name)

            if pod is None:
                count = self._pod_not_found_counts.get(cursor_key, 0) + 1
                self._pod_not_found_counts[cursor_key] = count
                if count < _POD_NOT_FOUND_GRACE_CYCLES:
                    updates.append(
                        TaskUpdate(
                            task_id=entry.task_id,
                            attempt_id=entry.attempt_id,
                            new_state=cluster_pb2.TASK_STATE_RUNNING,
                        )
                    )
                    continue
                # Grace exhausted — pod is truly gone. Treat as application
                # failure (FAILED) rather than infrastructure (WORKER_FAILED)
                # because we cannot determine the cause. Using FAILED ensures
                # max_retries_failure (default: 0) applies instead of
                # max_retries_preemption (default: 100), preventing runaway retries.
                self._pod_not_found_counts.pop(cursor_key, None)
                self._log_cursors.pop(cursor_key, None)
                updates.append(
                    TaskUpdate(
                        task_id=entry.task_id,
                        attempt_id=entry.attempt_id,
                        new_state=cluster_pb2.TASK_STATE_FAILED,
                        error="Pod not found",
                    )
                )
                continue

            self._pod_not_found_counts.pop(cursor_key, None)
            log_entries = self._fetch_incremental_logs(pod_name, entry.attempt_id, cursor_key)

            update = _task_update_from_pod(entry, pod)
            phase = pod.get("status", {}).get("phase", "")

            resource_usage = None
            if phase == "Running":
                try:
                    top_result = self.kubectl.top_pod(pod_name)
                    if top_result is not None:
                        cpu_mc, mem_bytes = top_result
                        resource_usage = cluster_pb2.ResourceUsage(
                            cpu_millicores=cpu_mc,
                            memory_mb=mem_bytes // (1024 * 1024),
                        )
                except Exception as e:
                    logger.debug("Failed to fetch resource stats for pod %s: %s", pod_name, e)

            if phase in ("Succeeded", "Failed"):
                final_logs = self._fetch_completed_pod_logs(pod_name, entry.attempt_id)
                if len(final_logs) > len(log_entries):
                    log_entries = final_logs
                self._log_cursors.pop(cursor_key, None)

            updates.append(
                TaskUpdate(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    new_state=update.new_state,
                    error=update.error,
                    exit_code=update.exit_code,
                    resource_usage=resource_usage or update.resource_usage,
                    log_entries=log_entries,
                )
            )

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
        """Fetch recent k8s events for iris-managed pods.

        K8s Events don't carry pod labels, so we query all events in the
        namespace and filter client-side by pod name prefix.
        """
        try:
            all_pods = self.kubectl.list_json(
                "pods",
                labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE},
            )
            pod_names = {pod.get("metadata", {}).get("name", "") for pod in all_pods}
            pod_labels = {
                pod.get("metadata", {}).get("name", ""): pod.get("metadata", {}).get("labels", {}) for pod in all_pods
            }
        except Exception as e:
            logger.warning("Failed to query pods for scheduling events: %s", e)
            return []

        if not pod_names:
            return []

        try:
            events = self.kubectl.list_json("events")
        except Exception as e:
            logger.warning("Failed to fetch scheduling events: %s", e)
            return []

        result: list[SchedulingEvent] = []
        for ev in events:
            involved = ev.get("involvedObject", {})
            if involved.get("kind") != "Pod":
                continue
            involved_name = involved.get("name", "")
            if involved_name not in pod_names:
                continue

            labels = pod_labels.get(involved_name, {})
            task_id = labels.get(_LABEL_TASK_ID, "")
            attempt_str = labels.get(_LABEL_ATTEMPT_ID, "0")
            try:
                attempt_id = int(attempt_str)
            except (ValueError, TypeError):
                attempt_id = 0

            result.append(
                SchedulingEvent(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    event_type=ev.get("type", "Normal"),
                    reason=ev.get("reason", ""),
                    message=ev.get("message", ""),
                    timestamp=Timestamp.now(),
                )
            )
        return result
