# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sTaskProvider: executes tasks as Kubernetes Pods.

No worker daemon, no synthetic worker row. The controller talks directly to the
k8s API via kubectl, launching one Pod per task attempt.
"""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import logging
import re
import shlex
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.controller.transitions import ClusterCapacity, DirectProviderSyncResult, SchedulingEvent
from iris.cluster.controller.transitions import DirectProviderBatch, RunningTaskEntry, TaskUpdate
from iris.cluster.log_store._types import LogPusherProtocol, TaskAttempt, task_log_key
from iris.cluster.providers.k8s.constants import NVIDIA_GPU_TOLERATION
from iris.cluster.providers.k8s.service import K8sService
from iris.cluster.providers.k8s.types import K8sResource, KubectlError, KubectlLogLine, parse_k8s_quantity
from iris.cluster.runtime.env import build_common_iris_env, normalize_workdir_relative_path
from iris.cluster.types import JobName, get_gpu_count
from iris.logging import str_to_log_level
from rigging.log_setup import parse_log_level
from iris.rpc import logging_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.rpc import worker_pb2
from iris.time_proto import timestamp_to_proto
from rigging.timing import Timestamp

logger = logging.getLogger(__name__)

# Label key prefix for iris-managed pod identification.
_LABEL_MANAGED = "iris.managed"
_LABEL_RUNTIME = "iris.runtime"
_LABEL_TASK_ID = "iris.task_id"
_LABEL_ATTEMPT_ID = "iris.attempt_id"
# Collision-resistant hash of the full (unsanitized) task_id; 16 hex chars (64 bits).
_LABEL_TASK_HASH = "iris.task_hash"
_LABEL_JOB_ID = "iris.job_id"

# Runtime identifier for pods created by K8sTaskProvider.
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
# Evicted: kubelet evicted the pod due to resource pressure.
# DeadlineExceeded: pod's activeDeadlineSeconds expired.
# Preempting: scheduler preempted the pod for a higher-priority workload.
# NOTE: OOMKilled is intentionally excluded — it indicates a misconfigured job
# (requesting too little memory), not transient infrastructure failure.
_INFRASTRUCTURE_FAILURE_REASONS = frozenset({"Evicted", "DeadlineExceeded", "Preempting"})


def _constraints_to_node_selector(
    constraints: Sequence[job_pb2.Constraint],
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
        if c.op == job_pb2.CONSTRAINT_OP_EQ and c.HasField("value"):
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
    task_env: dict[str, str] = field(default_factory=dict)


def _build_task_script(run_req: job_pb2.RunTaskRequest) -> str:
    """Build a shell script that runs setup_commands then the run_command."""
    lines = ["set -e", "ulimit -c 0", "mkdir -p /app", "cd /app"]
    for cmd in run_req.entrypoint.setup_commands:
        lines.append(cmd)
    if run_req.entrypoint.run_command.argv:
        lines.append("exec " + shlex.join(run_req.entrypoint.run_command.argv))
    return "\n".join(lines)


def _build_init_container_spec(
    run_req: job_pb2.RunTaskRequest,
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

    script_path = Path(__file__).parent / "bundle_fetch.py"
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


def _is_coordinator_task(run_req: job_pb2.RunTaskRequest) -> bool:
    """Heuristic: single-task job with no accelerators is a coordinator/orchestrator.

    Coordinator pods (e.g. zephyr *-coord jobs) are single-replica, CPU-only
    processes whose loss kills the entire pipeline. Returns True so the caller
    can create a PodDisruptionBudget to prevent voluntary eviction.
    """
    if run_req.num_tasks > 1:
        return False
    if run_req.HasField("resources") and run_req.resources.HasField("device"):
        device = run_req.resources.device
        if device.HasField("gpu") or device.HasField("tpu"):
            return False
    return True


def _pdb_name(pod_name: str) -> str:
    """Derive a PDB name from a pod name."""
    return f"{pod_name}-pdb"


def _build_pdb_manifest(
    pod_name: str,
    namespace: str,
    task_hash: str,
    managed_label: str = "",
) -> dict:
    """Build a PodDisruptionBudget manifest for a coordinator task pod."""
    labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
        _LABEL_TASK_HASH: task_hash,
    }
    if managed_label:
        labels[managed_label] = "true"
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {
            "name": _pdb_name(pod_name),
            "namespace": namespace,
            "labels": labels,
        },
        "spec": {
            "minAvailable": 1,
            "selector": {"matchLabels": {_LABEL_TASK_HASH: task_hash}},
        },
    }


def _build_pod_manifest(
    run_req: job_pb2.RunTaskRequest,
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
    )
    combined = {**config.task_env, **dict(run_req.environment.env_vars), **iris_env}
    env_list: list[dict] = [{"name": k, "value": v} for k, v in combined.items()]
    # Pod IP via downward API -- not expressible as a static value.
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
            # Set requests = limits for CPU/memory so K8s schedules based on
            # actual resource needs. GPU/RDMA are excluded (K8s treats GPU
            # limits as implicit requests).
            requests = {k: v for k, v in limits.items() if k in ("cpu", "memory")}
            if requests:
                resources.setdefault("requests", {}).update(requests)
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
        spec.setdefault("tolerations", []).append(NVIDIA_GPU_TOLERATION)

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
    level_name = parse_log_level(kll.data)
    level = str_to_log_level(level_name) if level_name else 0
    entry = logging_pb2.LogEntry(source=kll.stream, data=kll.data, attempt_id=attempt_id, level=level)
    entry.timestamp.CopyFrom(timestamp_to_proto(Timestamp.from_seconds(kll.timestamp.timestamp())))
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

    Infrastructure failures (eviction, preemption) are reported as WORKER_FAILED
    so they count against max_retries_preemption.
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
            new_state=job_pb2.TASK_STATE_BUILDING,
        )

    if phase == "Running":
        return TaskUpdate(
            task_id=task_id,
            attempt_id=attempt_id,
            new_state=job_pb2.TASK_STATE_RUNNING,
        )

    if phase == "Succeeded":
        return TaskUpdate(
            task_id=task_id,
            attempt_id=attempt_id,
            new_state=job_pb2.TASK_STATE_SUCCEEDED,
        )

    # Failed or Unknown -- distinguish infrastructure vs application failure.
    exit_code = _extract_exit_code(pod)
    if _is_infrastructure_failure(pod):
        new_state = job_pb2.TASK_STATE_WORKER_FAILED
    else:
        new_state = job_pb2.TASK_STATE_FAILED
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


# Field selector to exclude completed pods from list calls. Reduces API server
# response payload when many tasks have finished.
_ACTIVE_PODS_FIELD_SELECTOR = "status.phase!=Succeeded,status.phase!=Failed"

# Standard label filter for iris-managed pods.
_MANAGED_POD_LABELS = {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE}

# Garbage collection: how often to run the terminal-pod cleanup pass (seconds).
_GC_INTERVAL_SECONDS = 300  # 5 minutes

# Garbage collection: delete terminal pods and orphaned configmaps/PDBs older than this (seconds).
_GC_MAX_AGE_SECONDS = 3600  # 1 hour


def _build_pod_statuses(pods: list[dict]) -> list[controller_pb2.Controller.KubernetesPodStatus]:
    """Build pod status protos from raw kubectl pod objects."""
    statuses = []
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

        ps = controller_pb2.Controller.KubernetesPodStatus(
            pod_name=pod_name,
            task_id=task_id,
            phase=phase,
            reason=reason,
            message=message,
            node_name=node_name,
        )
        ps.last_transition.CopyFrom(timestamp_to_proto(last_ts))
        statuses.append(ps)
    return statuses


def _fetch_node_pools(kubectl: K8sService, managed_label: str) -> list[controller_pb2.Controller.NodePoolStatus]:
    """Fetch node pool statuses from the cluster."""
    try:
        np_labels = {managed_label: "true"} if managed_label else None
        pools = kubectl.list_json(K8sResource.NODE_POOLS, labels=np_labels)
    except Exception as e:
        logger.warning("Failed to query nodepools: %s", e)
        return []

    result = []
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
        result.append(
            controller_pb2.Controller.NodePoolStatus(
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
    return result


class ClusterState:
    """Live cluster state maintained by the sync thread.

    update() is called once per sync cycle with the freshly-fetched raw
    kubectl data. capacity() and to_status_response() may be called from
    any thread (e.g. the dashboard RPC handler) without holding any external
    lock — the internal lock is acquired only for the brief copy.

    Pods are kept sorted by name so that pagination is stable across
    consecutive dashboard polls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pods: list[dict] = []
        self._nodes: list[dict] = []
        self._node_pools: list[controller_pb2.Controller.NodePoolStatus] = []

    def update(
        self,
        pods: list[dict],
        nodes: list[dict],
        node_pools: list[controller_pb2.Controller.NodePoolStatus],
    ) -> None:
        """Atomically replace all cluster state from a completed sync cycle."""
        new_pods = sorted(pods, key=lambda p: p.get("metadata", {}).get("name", ""))
        new_nodes = sorted(nodes, key=lambda n: n.get("metadata", {}).get("name", ""))
        with self._lock:
            self._pods = new_pods
            self._nodes = new_nodes
            self._node_pools = list(node_pools)

    def capacity(self) -> ClusterCapacity | None:
        """Compute scheduling capacity: node allocatable minus running pod requests."""
        with self._lock:
            pods = self._pods[:]
            nodes = self._nodes[:]

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
            cpu_val = parse_k8s_quantity(cpu_str)
            if not cpu_str.endswith("m"):
                cpu_val *= 1000
            total_cpu_mc += cpu_val
            total_memory_bytes += parse_k8s_quantity(allocatable.get("memory", "0"))
            schedulable_count += 1

        if schedulable_count == 0:
            return None

        used_cpu_mc = 0
        used_memory_bytes = 0
        for pod in pods:
            if pod.get("status", {}).get("phase", "") not in ("Pending", "Running"):
                continue
            for container in pod.get("spec", {}).get("containers", []):
                requests = container.get("resources", {}).get("requests", {})
                limits = container.get("resources", {}).get("limits", {})
                cpu_req = requests.get("cpu") or limits.get("cpu", "0")
                mem_req = requests.get("memory") or limits.get("memory", "0")
                cpu_v = parse_k8s_quantity(cpu_req)
                if not cpu_req.endswith("m"):
                    cpu_v *= 1000
                used_cpu_mc += cpu_v
                used_memory_bytes += parse_k8s_quantity(mem_req)

        return ClusterCapacity(
            schedulable_nodes=schedulable_count,
            total_cpu_millicores=total_cpu_mc,
            available_cpu_millicores=total_cpu_mc - used_cpu_mc,
            total_memory_bytes=total_memory_bytes,
            available_memory_bytes=total_memory_bytes - used_memory_bytes,
        )

    def to_status_response(self, namespace: str) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Build the dashboard RPC response from current state. No kubectl calls."""
        with self._lock:
            pods = self._pods[:]
            nodes = self._nodes[:]
            node_pools = self._node_pools[:]

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
            cpu_val = parse_k8s_quantity(cpu_str)
            if not cpu_str.endswith("m"):
                cpu_val *= 1000
            total_cpu_mc += cpu_val
            total_memory_bytes += parse_k8s_quantity(allocatable.get("memory", "0"))

        return controller_pb2.Controller.GetKubernetesClusterStatusResponse(
            namespace=namespace,
            total_nodes=total_nodes,
            schedulable_nodes=schedulable_nodes,
            allocatable_cpu=f"{total_cpu_mc / 1000:.1f} cores" if total_cpu_mc else "0 cores",
            allocatable_memory=_format_bytes(total_memory_bytes),
            pod_statuses=_build_pod_statuses(pods),
            provider_version="iris-kubernetes/v1",
            node_pools=node_pools,
        )


@dataclass
class _LogPod:
    """A pod tracked by LogCollector for incremental log fetching."""

    pod_name: str
    task_id: JobName
    attempt_id: int
    last_timestamp: datetime | None = None
    consecutive_failures: int = 0


class LogCollector:
    """Background log fetcher that pushes entries to the LogService.

    Runs on its own daemon thread with a bounded ThreadPoolExecutor.
    The sync loop calls set_pods() once per cycle with the authoritative
    set of pods to track. The collector diffs against its internal state,
    does a final fetch for removed pods, and starts tracking new ones.
    This avoids drift between what the sync loop thinks is tracked and
    what the collector is actually polling.
    """

    _DEFAULT_LIMIT_BYTES: int = 100_000

    def __init__(
        self,
        kubectl: K8sService,
        log_pusher: LogPusherProtocol,
        concurrency: int = 8,
        poll_interval: float = 15.0,
        limit_bytes: int | None = _DEFAULT_LIMIT_BYTES,
    ):
        self._kubectl = kubectl
        self._log_pusher = log_pusher
        self._poll_interval = poll_interval
        self._limit_bytes = limit_bytes
        self._pods: dict[str, _LogPod] = {}
        self._lock = threading.Lock()
        self._pod_locks: dict[str, threading.Lock] = {}
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="log-collect")
        self._thread = threading.Thread(target=self._run, daemon=True, name="log-collector")
        self._thread.start()

    def set_pods(self, pods: dict[str, _LogPod]) -> None:
        """Declare the authoritative set of pods to collect logs for.

        New keys are added. Keys absent from `pods` are removed after a
        synchronous final log fetch. Existing keys are preserved (keeping
        their cursor state).
        """
        with self._lock:
            removed_keys = self._pods.keys() - pods.keys()
            removed = [(key, self._pods[key], self._pod_locks.get(key)) for key in removed_keys]
            for key in removed_keys:
                del self._pods[key]
                self._pod_locks.pop(key, None)
            for key, pod in pods.items():
                if key not in self._pods:
                    self._pods[key] = pod
                    self._pod_locks[key] = threading.Lock()

        # Final fetch for removed pods (outside lock to avoid holding it during I/O).
        for _key, pod, pod_lock in removed:
            if pod_lock is not None:
                with pod_lock:
                    self._fetch_and_store(pod)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                snapshot = list(self._pods.items())
            for key, pod in snapshot:
                with self._lock:
                    pod_lock = self._pod_locks.get(key)
                if pod_lock is not None:
                    self._executor.submit(self._guarded_fetch, key, pod, pod_lock)
            self._stop.wait(timeout=self._poll_interval)

    def _guarded_fetch(self, key: str, pod: _LogPod, pod_lock: threading.Lock) -> None:
        if not pod_lock.acquire(blocking=False):
            return
        try:
            self._fetch_and_store(pod)
        finally:
            pod_lock.release()

    def _fetch_and_store(self, pod: _LogPod) -> bool:
        """Fetch logs since last timestamp and advance. Must be called under pod lock."""
        try:
            result = self._kubectl.stream_logs(
                pod.pod_name, container="task", since_time=pod.last_timestamp, limit_bytes=self._limit_bytes
            )
            if result.lines:
                entries = [_kubectl_log_line_to_log_entry(kll, pod.attempt_id) for kll in result.lines]
                key = task_log_key(TaskAttempt(task_id=pod.task_id, attempt_id=pod.attempt_id))
                self._log_pusher.push(key, entries)
            pod.last_timestamp = result.last_timestamp
            pod.consecutive_failures = 0
            return True
        except Exception as e:
            pod.consecutive_failures += 1
            if pod.consecutive_failures <= 1:
                logger.warning("LogCollector: fetch failed for pod %s: %s", pod.pod_name, e)
            return False

    def close(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False)
        self._thread.join(timeout=5)


class ResourceCollector:
    """Background resource usage collector.

    Same set_pods() pattern as LogCollector: the sync loop declares the
    authoritative set of running pods once per cycle, and the collector
    diffs internally. Pods removed from the set have their cached results
    cleaned up immediately.
    """

    def __init__(self, kubectl: K8sService, concurrency: int = 8):
        self._kubectl = kubectl
        self._pods: dict[str, str] = {}  # cursor_key -> pod_name
        self._results: dict[str, job_pb2.ResourceUsage] = {}  # cursor_key -> latest reading
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="resource-collect")
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-collector")
        self._thread.start()

    def set_pods(self, pods: dict[str, str]) -> None:
        """Declare the authoritative set of pods to collect resources for.

        `pods` maps cursor_key -> pod_name. Keys absent from `pods` are
        removed along with their cached results.
        """
        with self._lock:
            removed_keys = self._pods.keys() - pods.keys()
            for key in removed_keys:
                self._results.pop(key, None)
            self._pods = dict(pods)

    def get(self, task_id: JobName, attempt_id: int) -> job_pb2.ResourceUsage | None:
        """Return the latest resource reading for a pod (non-blocking)."""
        key = f"{task_id.to_wire()}:{attempt_id}"
        with self._lock:
            return self._results.get(key)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                snapshot = list(self._pods.items())
            if snapshot:
                futures = {self._executor.submit(self._fetch_one, key, pod_name): key for key, pod_name in snapshot}
                for f in concurrent.futures.as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass
            self._stop.wait(timeout=5.0)

    def _fetch_one(self, key: str, pod_name: str) -> None:
        try:
            top = self._kubectl.top_pod(pod_name)
        except Exception as e:
            logger.debug("ResourceCollector: top_pod raised for pod %s: %s", pod_name, e)
            top = None

        if top is not None:
            usage = job_pb2.ResourceUsage(
                cpu_millicores=top.cpu_millicores,
                memory_mb=top.memory_bytes // (1024 * 1024),
            )
            with self._lock:
                # Only store if the key is still tracked (may have been removed by set_pods).
                if key in self._pods:
                    self._results[key] = usage

    def close(self) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False)
        self._thread.join(timeout=5)


@dataclass
class K8sTaskProvider:
    """Executes tasks as Kubernetes Pods without worker daemons.

    Implements the "direct provider" interface used by the controller when no
    separate worker daemon is involved. The controller calls sync() with a
    DirectProviderBatch and receives back a DirectProviderSyncResult — not the
    standard TaskProvider/DispatchBatch/HeartbeatApplyRequest protocol used by
    worker-based providers (e.g. GCP). This is intentional: K8s pods are
    launched and monitored directly via kubectl rather than through a worker
    gRPC daemon.

    Capacity is derived from node allocatable resources minus running pod
    resource requests, queried via kubectl each sync cycle.

    Pod naming: iris-{task_id_sanitized}-{attempt_id}
    """

    kubectl: K8sService
    namespace: str
    default_image: str
    colocation_topology_key: str = "coreweave.cloud/spine"
    cache_dir: str = "/cache"
    service_account: str = ""
    host_network: bool = False
    controller_address: str | None = None
    managed_label: str = ""
    task_env: dict[str, str] = field(default_factory=dict)
    log_pusher: LogPusherProtocol | None = None
    poll_concurrency: int = 32
    log_poll_interval: float = 15.0
    _pod_not_found_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _log_collector: LogCollector | None = field(default=None, init=False, repr=False)
    _resource_collector: ResourceCollector | None = field(default=None, init=False, repr=False)
    _cluster_state: ClusterState = field(default_factory=ClusterState, init=False, repr=False)
    _last_gc_time: float = field(default=0.0, init=False, repr=False)
    _pending_gc_hashes: set[str] = field(default_factory=set, init=False, repr=False)

    def _ensure_resource_collector(self) -> ResourceCollector:
        if self._resource_collector is None:
            self._resource_collector = ResourceCollector(self.kubectl, concurrency=self.poll_concurrency)
        return self._resource_collector

    def _ensure_log_collector(self) -> LogCollector | None:
        if self._log_collector is None and self.log_pusher is not None:
            self._log_collector = LogCollector(
                self.kubectl, self.log_pusher, concurrency=self.poll_concurrency, poll_interval=self.log_poll_interval
            )
        return self._log_collector

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        """Sync task state: apply new pods, delete killed pods, poll running pods."""
        apply_failures: list[TaskUpdate] = []
        for run_req in batch.tasks_to_run:
            try:
                self._apply_pod(run_req)
            except KubectlError as exc:
                logger.error("Failed to apply pod for task %s: %s", run_req.task_id, exc)
                apply_failures.append(
                    TaskUpdate(
                        task_id=JobName.from_wire(run_req.task_id),
                        attempt_id=run_req.attempt_id,
                        new_state=job_pb2.TASK_STATE_FAILED,
                        error=str(exc),
                    )
                )

        # Single pod list for the entire cycle — excludes terminal pods via field selector.
        managed_pods = self.kubectl.list_json(
            K8sResource.PODS,
            labels=_MANAGED_POD_LABELS,
            field_selector=_ACTIVE_PODS_FIELD_SELECTOR,
        )

        self._bulk_delete_task_pods(batch.tasks_to_kill, managed_pods)
        updates = apply_failures + self._poll_pods(batch.running_tasks, managed_pods)
        scheduling_events = self._fetch_scheduling_events(managed_pods)

        try:
            nodes = self.kubectl.list_json(K8sResource.NODES)
        except Exception as e:
            logger.warning("Failed to query node resources: %s", e)
            nodes = []

        node_pools = _fetch_node_pools(self.kubectl, self.managed_label)
        self._cluster_state.update(managed_pods, nodes, node_pools)
        capacity = self._cluster_state.capacity()

        self._maybe_gc_terminal_resources(managed_pods)

        return DirectProviderSyncResult(updates=updates, scheduling_events=scheduling_events, capacity=capacity)

    def profile_task(
        self,
        task_id: str,
        attempt_id: int,
        request: job_pb2.ProfileTaskRequest,
    ) -> job_pb2.ProfileTaskResponse:
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
                return job_pb2.ProfileTaskResponse(error="Unknown profile type")
        except Exception as e:
            return job_pb2.ProfileTaskResponse(error=str(e))

    def exec_in_container(
        self,
        task_id: str,
        attempt_id: int,
        command: list[str],
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task pod via kubectl exec."""
        pod_name = _pod_name(JobName.from_wire(task_id), attempt_id)
        effective_timeout: float | None = timeout_seconds if timeout_seconds >= 0 else None
        try:
            result = self.kubectl.exec(pod_name, command, container="task", timeout=effective_timeout)
            return worker_pb2.Worker.ExecInContainerResponse(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except Exception as e:
            return worker_pb2.Worker.ExecInContainerResponse(error=str(e))

    def close(self) -> None:
        if self._log_collector is not None:
            self._log_collector.close()
        if self._resource_collector is not None:
            self._resource_collector.close()

    def get_cluster_status(self) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Return cluster status from the latest sync() snapshot. No kubectl calls."""
        return self._cluster_state.to_status_response(self.namespace)

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

    def _profile_threads(self, pod_name: str, threads_config: job_pb2.ThreadsProfile) -> job_pb2.ProfileTaskResponse:
        """Get thread stacks via py-spy dump."""
        from iris.cluster.runtime.profile import build_pyspy_dump_cmd

        cmd = shlex.join(build_pyspy_dump_cmd("1", include_locals=threads_config.locals))
        stdout = self._kubectl_exec_shell(pod_name, cmd, timeout=30)
        return job_pb2.ProfileTaskResponse(profile_data=stdout.encode("utf-8"))

    def _profile_cpu(self, pod_name: str, cpu_config: job_pb2.CpuProfile, duration: int) -> job_pb2.ProfileTaskResponse:
        """Record CPU profile via py-spy."""
        from iris.cluster.runtime.profile import build_pyspy_cmd, resolve_cpu_spec

        spec = resolve_cpu_spec(cpu_config, duration, pid="1")
        output_path = f"/tmp/iris-profile.{spec.ext}"
        cmd = shlex.join(build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path=output_path))
        self._kubectl_exec_shell(pod_name, cmd, timeout=duration + 30)
        data = self.kubectl.read_file(pod_name, output_path, container="task")
        self.kubectl.rm_files(pod_name, [output_path], container="task")
        return job_pb2.ProfileTaskResponse(profile_data=data)

    def _profile_memory(
        self, pod_name: str, memory_config: job_pb2.MemoryProfile, duration: int
    ) -> job_pb2.ProfileTaskResponse:
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

        if spec.is_raw:
            data = self.kubectl.read_file(pod_name, trace_path, container="task")
            self.kubectl.rm_files(pod_name, [trace_path], container="task")
            return job_pb2.ProfileTaskResponse(profile_data=data)

        transform_cmd = shlex.join(
            build_memray_transform_cmd(spec, memray_bin="memray", trace_path=trace_path, output_path=output_path)
        )
        transform_stdout = self._kubectl_exec_shell(pod_name, transform_cmd, timeout=30)

        if spec.output_is_file:
            data = self.kubectl.read_file(pod_name, output_path, container="task")
        else:
            data = transform_stdout.encode("utf-8")

        self.kubectl.rm_files(pod_name, [trace_path, output_path], container="task")
        return job_pb2.ProfileTaskResponse(profile_data=data)

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
            task_env=self.task_env,
        )

    def _apply_pod(self, run_req: job_pb2.RunTaskRequest) -> None:
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

        if _is_coordinator_task(run_req):
            pdb = _build_pdb_manifest(
                pod_name,
                self.namespace,
                _task_hash(run_req.task_id),
                managed_label=self.managed_label,
            )
            self.kubectl.apply_json(pdb)
            logger.info("Applied PDB %s for coordinator task %s", pdb["metadata"]["name"], task_id)

    def _bulk_delete_task_pods(self, task_ids: list[str], cached_pods: list[dict]) -> None:
        """Delete pods for killed tasks. ConfigMaps and PDBs are cleaned up by the
        periodic GC pass (_gc_terminal_resources) to avoid listing all configmaps/PDBs
        on every sync cycle — which was an O(total_resources) scan on the hot path.
        """
        if not task_ids:
            return
        task_hashes = {_task_hash(tid) for tid in task_ids}

        pod_names = [
            p["metadata"]["name"]
            for p in cached_pods
            if p.get("metadata", {}).get("labels", {}).get(_LABEL_TASK_HASH) in task_hashes
        ]

        if pod_names:
            self.kubectl.delete_many(K8sResource.PODS, pod_names, wait=False)

        # Enqueue task hashes for deferred configmap/PDB cleanup by the GC pass.
        self._pending_gc_hashes.update(task_hashes)

        logger.info("Deleted %d pods for %d tasks (CM/PDB cleanup deferred to GC)", len(pod_names), len(task_ids))

    def _maybe_gc_terminal_resources(self, active_pods: list[dict]) -> None:
        """Periodically delete terminal (Succeeded/Failed) pods and their associated
        configmaps/PDBs that are older than _GC_MAX_AGE_SECONDS.

        Without this, completed pods and their configmaps accumulate in etcd indefinitely
        since the sync loop's field selector excludes terminal pods from its queries.

        active_pods is the list of Pending/Running pods from the current sync cycle,
        used to protect configmaps/PDBs for tasks that have active retry attempts.
        """
        now = time.monotonic()
        if now - self._last_gc_time < _GC_INTERVAL_SECONDS:
            return
        self._last_gc_time = now

        try:
            self._gc_terminal_resources(active_pods)
        except Exception:
            logger.exception("GC pass failed; will retry next interval")

    def _gc_terminal_resources(self, active_pods: list[dict]) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - _GC_MAX_AGE_SECONDS

        # Collect task hashes that still have active (Pending/Running) pods.
        # These must NOT have their configmaps/PDBs deleted, even if an older
        # attempt of the same task is terminal — task_hash is shared across attempts.
        active_hashes: set[str] = set()
        for pod in active_pods:
            h = pod.get("metadata", {}).get("labels", {}).get(_LABEL_TASK_HASH)
            if h:
                active_hashes.add(h)

        # 1. Targeted cleanup: delete configmaps/PDBs for tasks that were killed
        #    since last GC. Uses label-selector deletes (one kubectl call per hash)
        #    instead of listing all resources and filtering client-side.
        #    Only remove hashes we actually clean up; skipped hashes (still active)
        #    stay in the set for the next GC cycle.
        safe_pending = self._pending_gc_hashes - active_hashes
        self._pending_gc_hashes -= safe_pending
        for task_hash in safe_pending:
            labels = {**_MANAGED_POD_LABELS, _LABEL_TASK_HASH: task_hash}
            self.kubectl.delete_by_labels(K8sResource.CONFIGMAPS, labels, wait=False)
            self.kubectl.delete_by_labels(K8sResource.PDBS, labels, wait=False)
        if safe_pending:
            logger.info("GC: cleaned up CMs/PDBs for %d killed task hashes", len(safe_pending))

        # 2. Age-based sweep: delete terminal pods older than the cutoff, and
        #    their associated configmaps/PDBs (by task_hash label-selector delete).
        #    Skip hashes that still have active pods to avoid deleting live resources.
        old_pod_names: list[str] = []
        old_task_hashes: set[str] = set()
        for phase in ("Succeeded", "Failed"):
            pods = self.kubectl.list_json(
                K8sResource.PODS,
                labels=_MANAGED_POD_LABELS,
                field_selector=f"status.phase={phase}",
            )
            for pod in pods:
                meta = pod.get("metadata", {})
                created = meta.get("creationTimestamp", "")
                if not created:
                    continue
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                if ts < cutoff:
                    old_pod_names.append(meta["name"])
                    task_hash = meta.get("labels", {}).get(_LABEL_TASK_HASH)
                    if task_hash:
                        old_task_hashes.add(task_hash)

        if old_pod_names:
            self.kubectl.delete_many(K8sResource.PODS, old_pod_names, wait=False)
        safe_hashes = old_task_hashes - active_hashes
        for task_hash in safe_hashes:
            labels = {**_MANAGED_POD_LABELS, _LABEL_TASK_HASH: task_hash}
            self.kubectl.delete_by_labels(K8sResource.CONFIGMAPS, labels, wait=False)
            self.kubectl.delete_by_labels(K8sResource.PDBS, labels, wait=False)

        if old_pod_names:
            logger.info(
                "GC: deleted %d terminal pods + CMs/PDBs for %d task hashes (age > %ds, %d skipped with active pods)",
                len(old_pod_names),
                len(safe_hashes),
                _GC_MAX_AGE_SECONDS,
                len(old_task_hashes - safe_hashes),
            )

    def _delete_pods_by_task_id(self, task_id: str) -> None:
        """Delete all pods, configmaps, and PDBs for a given task_id (any attempt).

        Uses the SHA-256 task hash label for collision-resistant pod lookup,
        avoiding false matches that _sanitize_label_value's lossy truncation
        could cause when distinct task IDs share the same sanitized prefix.
        """
        task_hash = _task_hash(task_id)
        managed_labels = {_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE, _LABEL_TASK_HASH: task_hash}
        pods = self.kubectl.list_json(K8sResource.PODS, labels=managed_labels)
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            if pod_name:
                self.kubectl.delete(K8sResource.PODS, pod_name)
                logger.info("Deleted pod %s for task %s", pod_name, task_id)

        # Clean up associated ConfigMaps (workdir files).
        configmaps = self.kubectl.list_json(K8sResource.CONFIGMAPS, labels=managed_labels)
        for cm in configmaps:
            cm_name = cm.get("metadata", {}).get("name", "")
            if cm_name:
                self.kubectl.delete(K8sResource.CONFIGMAPS, cm_name)
                logger.info("Deleted configmap %s for task %s", cm_name, task_id)

        # Clean up associated PDBs (coordinator tasks).
        pdbs = self.kubectl.list_json(K8sResource.PDBS, labels=managed_labels)
        for pdb in pdbs:
            pdb_name = pdb.get("metadata", {}).get("name", "")
            if pdb_name:
                self.kubectl.delete(K8sResource.PDBS, pdb_name)
                logger.info("Deleted PDB %s for task %s", pdb_name, task_id)

    def _poll_pods(self, running: list[RunningTaskEntry], cached_pods: list[dict]) -> list[TaskUpdate]:
        """Poll pod phases for all running tasks.

        Uses the pre-fetched pod list (active pods only, terminal pods excluded
        by field selector). For entries missing from the cached list, does a
        targeted get_json to check if the pod completed — this avoids the grace-
        period-to-FAILED path for legitimately Succeeded pods.

        Log fetching and resource usage collection are handled by background
        LogCollector and ResourceCollector threads. After building updates,
        this method calls set_pods() on each collector with the authoritative
        set of non-terminal pods, so the collectors can never drift.
        """
        if not running:
            # No running tasks — clear all collectors.
            log_collector = self._ensure_log_collector()
            if log_collector is not None:
                log_collector.set_pods({})
            if self._resource_collector is not None:
                self._resource_collector.set_pods({})
            return []

        pods_by_name: dict[str, dict] = {pod.get("metadata", {}).get("name", ""): pod for pod in cached_pods}
        updates: list[TaskUpdate] = []

        # Build up the authoritative pod sets for collectors.
        log_pods: dict[str, _LogPod] = {}
        resource_pods: dict[str, str] = {}  # cursor_key -> pod_name
        terminal_log_pods: dict[str, _LogPod] = {}  # pods that completed this cycle

        for entry in running:
            pod_name = _pod_name(entry.task_id, entry.attempt_id)
            cursor_key = f"{entry.task_id.to_wire()}:{entry.attempt_id}"
            pod = pods_by_name.get(pod_name)

            if pod is None:
                # Pod not in active list — may have completed or truly vanished.
                pod = self.kubectl.get_json(K8sResource.PODS, pod_name)

            if pod is None:
                count = self._pod_not_found_counts.get(cursor_key, 0) + 1
                self._pod_not_found_counts[cursor_key] = count
                if count < _POD_NOT_FOUND_GRACE_CYCLES:
                    updates.append(
                        TaskUpdate(
                            task_id=entry.task_id,
                            attempt_id=entry.attempt_id,
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    )
                    continue
                # Grace exhausted — pod is truly gone.
                self._pod_not_found_counts.pop(cursor_key, None)
                updates.append(
                    TaskUpdate(
                        task_id=entry.task_id,
                        attempt_id=entry.attempt_id,
                        new_state=job_pb2.TASK_STATE_FAILED,
                        error="Pod not found",
                    )
                )
                continue

            self._pod_not_found_counts.pop(cursor_key, None)
            update = _task_update_from_pod(entry, pod)
            phase = pod.get("status", {}).get("phase", "")

            if phase not in ("Succeeded", "Failed"):
                log_pods[cursor_key] = _LogPod(pod_name=pod_name, task_id=entry.task_id, attempt_id=entry.attempt_id)
                if phase == "Running":
                    resource_pods[cursor_key] = pod_name
            else:
                terminal_log_pods[cursor_key] = _LogPod(
                    pod_name=pod_name, task_id=entry.task_id, attempt_id=entry.attempt_id
                )

            # Read latest cached resource usage (non-blocking).
            resource_usage = None
            if self._resource_collector is not None:
                resource_usage = self._resource_collector.get(entry.task_id, entry.attempt_id)

            updates.append(
                TaskUpdate(
                    task_id=update.task_id,
                    attempt_id=update.attempt_id,
                    new_state=update.new_state,
                    error=update.error,
                    exit_code=update.exit_code,
                    resource_usage=resource_usage or update.resource_usage,
                )
            )

        # Sync collectors with the authoritative pod sets.
        # set_pods() does a final log fetch for pods that drop out of the set.
        # For pods that completed this cycle, we include them first so they're
        # added (if not already tracked), then call set_pods again without them
        # to trigger the final fetch on removal.
        log_collector = self._ensure_log_collector()
        if log_collector is not None:
            if terminal_log_pods:
                log_collector.set_pods({**log_pods, **terminal_log_pods})
            log_collector.set_pods(log_pods)
        self._ensure_resource_collector().set_pods(resource_pods)

        return updates

    def _fetch_scheduling_events(self, cached_pods: list[dict]) -> list[SchedulingEvent]:
        """Fetch recent k8s events for iris-managed pods.

        K8s Events don't carry pod labels, so we query all events in the
        namespace and filter client-side by pod name prefix.
        """
        pod_names = {pod.get("metadata", {}).get("name", "") for pod in cached_pods}
        pod_labels = {
            pod.get("metadata", {}).get("name", ""): pod.get("metadata", {}).get("labels", {}) for pod in cached_pods
        }

        if not pod_names:
            return []

        try:
            # Filter server-side to pod warning events only; fetching all namespace
            # events is expensive and causes OOM on busy clusters.
            events = self.kubectl.list_json(
                K8sResource.EVENTS,
                field_selector="involvedObject.kind=Pod,type=Warning",
            )
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
