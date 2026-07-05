# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sTaskProvider: executes tasks as Kubernetes Pods.

No worker daemon, no synthetic worker row. The controller talks directly to the
k8s API via kubectl, launching one Pod per task attempt.
"""

import base64
import hashlib
import json
import logging
import re
import shlex
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from finelog.client.log_client import Table
from rigging.timing import Timestamp

from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    ProviderUnsupportedError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
    user_admitted,
)
from iris.cluster.controller.ops.task import apply_dispatch_updates
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.platforms.k8s.constants import COREWEAVE_INTERRUPTABLE_TOLERATION, NVIDIA_GPU_TOLERATION
from iris.cluster.platforms.k8s.coreweave_topology import CW_LABEL_LEAFGROUP, CW_LABEL_NVLINK_DOMAIN
from iris.cluster.platforms.k8s.service import K8sService
from iris.cluster.platforms.k8s.types import (
    IRIS_PRIORITY_CLASS_BATCH,
    IRIS_PRIORITY_CLASS_INTERACTIVE,
    IRIS_PRIORITY_CLASS_PRODUCTION,
    K8sResource,
    KubectlError,
    parse_k8s_quantity,
    parse_k8s_timestamp,
)
from iris.cluster.runtime.env import (
    VENV_PATH,
    build_common_iris_env,
    normalize_workdir_relative_path,
    render_setup_steps,
)
from iris.cluster.runtime.profile import (
    PROFILER_WATCHDOG_GRACE_SECONDS,
    ExecResult,
    build_profile_row,
    capture_cpu,
    capture_memory_attach,
    capture_threads,
    sigcont_sweep_argv,
    wrap_with_kill_watchdog,
)
from iris.cluster.types import JobName, WorkerId, get_gpu_count
from iris.cluster.worker.stats import IrisTaskStat, build_task_stat
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2
from iris.rpc.proto_display import resolve_container_profile
from iris.time_proto import timestamp_to_proto

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

# Extended resource name for NVIDIA GPUs in pod requests/limits.
_GPU_RESOURCE = "nvidia.com/gpu"

# Name of the task container in the pod. Exit-code/error extraction matches the
# task status by this name rather than by position in containerStatuses.
_TASK_CONTAINER_NAME = "task"

# Native log-shipping sidecar (initContainer + restartPolicy: Always). It reads
# the task container's CRI log file from the node and pushes to finelog, so the
# controller never pulls pod logs through the apiserver.
_LOGSHIP_CONTAINER_NAME = "log-shipper"
_LOGSHIP_VOLUME_NAME = "varlogpods"
_NODE_POD_LOG_DIR = "/var/log/pods"

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

# ---------------------------------------------------------------------------
# Kueue gang admission (coscheduled jobs only)
# ---------------------------------------------------------------------------
# Iris expresses gang scheduling to Kueue's plain-pod-group integration. The
# mutating webhook injects a scheduling gate on any pod carrying the
# queue-name label and removes it only once the whole group's Workload is
# admitted, giving all-or-nothing startup. Kueue never recreates pods — Iris
# remains the only pod controller (the _terminate/_requeue sibling cascade in
# transitions.py owns retries). See .agents/projects/20260529_iris_k8s_gang_admission.md.
_KUEUE_POD_GROUP_NAME = "kueue.x-k8s.io/pod-group-name"
_KUEUE_POD_GROUP_TOTAL = "kueue.x-k8s.io/pod-group-total-count"
_KUEUE_QUEUE_NAME = "kueue.x-k8s.io/queue-name"
_KUEUE_PRIORITY_CLASS = "kueue.x-k8s.io/priority-class"
_KUEUE_REQUIRED_TOPOLOGY = "kueue.x-k8s.io/podset-required-topology"
_KUEUE_PREFERRED_TOPOLOGY = "kueue.x-k8s.io/podset-preferred-topology"
# Per-pod ordinal within the gang. Kueue's TAS plain-pod-group path uses it to
# assign each pod a topology domain rank; basic gang admission does not need it,
# but stamping it is harmless and required once a podset-topology annotation is
# present. Sourced from the task ordinal (JobName.task_index).
_KUEUE_POD_GROUP_POD_INDEX = "kueue.x-k8s.io/pod-group-pod-index"
# Pod finalizer Kueue's webhook stamps on admitted gang pods. Kueue only
# strips it for pods it considers accounted for; on teardown Iris removes it
# itself so the pod objects actually disappear instead of pinning the
# pod-group Workload (Kueue rebuilds it from surviving labeled pods).
_KUEUE_MANAGED_FINALIZER = "kueue.x-k8s.io/managed"

# CoreWeave-convention fallback for KueueConfig.topologies: group_by -> (node
# label, required?). Used only when the cluster config leaves topologies unset.
# group_by names the ACTUAL topology level the gang runs against (a convention,
# not a portable abstraction — CoreWeave names leak by design). The keys are the
# levels in CoreWeave's Kueue Topology CRs (see scripts/install_kueue.py):
#   leafgroup     soft (preferred): multi-node IB colocation on one leaf group
#                 (H100 InfiniBand deployments).
#   nvlink.domain hard (required): one GB200 NVLink domain (H100 has no
#                 nvlink.domain label, so this only binds on GB200 capacity).
# A cluster whose Topology uses different levels overrides this via
# kubernetes_provider.kueue.topologies. Priority classes have NO default: Iris
# never invents WorkloadPriorityClass names (a missing one is rejected by
# Kueue), so a band is stamped only when the config maps it explicitly.
_CW_DEFAULT_TOPOLOGIES: dict[str, tuple[str, bool]] = {
    "leafgroup": (CW_LABEL_LEAFGROUP, False),
    "nvlink.domain": (CW_LABEL_NVLINK_DOMAIN, True),
}

_DEFAULT_PRIORITY_CLASS_NAMES: dict[int, str] = {
    job_pb2.PRIORITY_BAND_PRODUCTION: IRIS_PRIORITY_CLASS_PRODUCTION,
    job_pb2.PRIORITY_BAND_INTERACTIVE: IRIS_PRIORITY_CLASS_INTERACTIVE,
    job_pb2.PRIORITY_BAND_BATCH: IRIS_PRIORITY_CLASS_BATCH,
}


def _job_path(task_id: JobName) -> str:
    """Return the raw (unsanitized) parent job path of a task wire ID.

    Siblings of a coscheduled job share this path, so hashing it yields one
    pod-group identity for the whole gang. Distinct from _job_id_from_task,
    which sanitizes the result for use as a label *value*; here we want a
    collision-resistant input to _task_hash.
    """
    wire = task_id.to_wire()
    return wire.rsplit("/", 1)[0] if "/" in wire else wire


def _pod_group_name(task_id: JobName, attempt_id: int) -> str:
    """Kueue pod-group-name shared by every sibling of a coscheduled gang.

    Keyed by the job (parent path) so all siblings join one group, and by
    attempt_id as the generation key. A full-gang requeue bumps every
    sibling's attempt in lockstep (drain_for_dispatch promotes the gang
    all-or-none), so the retry gets a fresh pod-group-name and a fresh atomic
    admission; Kueue never resurrects the prior generation's Workload.
    """
    return f"iris-pg-{_task_hash(_job_path(task_id))}-{attempt_id}"


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
    """Job path of a task, sanitized for use as a k8s label value.

    Shares the parent-path extraction with :func:`_job_path` (which returns the
    raw path for hashing); here we sanitize it. ``_sanitize_label_value`` falls
    back to "unknown" on an empty result.
    """
    return _sanitize_label_value(_job_path(task_id))


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
    # Image for the log-shipper sidecar. The task default_image is a bare runtime
    # that only gains the iris package after the task's own `uv sync`, so the
    # sidecar instead runs the iris controller image (iris + finelog installed),
    # which can launch `python -m iris.cluster.backends.k8s.logship` directly.
    logship_image: str = ""
    cache_dir: str = "/cache"
    service_account: str = ""
    host_network: bool = False
    controller_address: str | None = None
    managed_label: str = ""
    task_env: dict[str, str] = field(default_factory=dict)
    # Name of a Secret whose keys are projected into every task container via
    # envFrom (operator-injected env, defaults.inject_env). Empty disables it.
    env_secret_name: str = ""
    # Kueue LocalQueue for coscheduled gang admission. Coscheduled jobs REQUIRE
    # this: dispatching one with no LocalQueue configured raises (Kueue or
    # nothing — there is no non-Kueue colocation fallback).
    local_queue: str = ""
    # PriorityBand -> WorkloadPriorityClass name. A band with no entry is not
    # stamped (Kueue uses its default priority); Iris never invents class names.
    kueue_priority_classes: dict[int, str] = field(default_factory=dict)
    # coscheduling group_by -> (node label, required?). Defaults to CoreWeave
    # conventions; a group_by with no entry carries no topology annotation.
    kueue_topologies: dict[str, tuple[str, bool]] = field(default_factory=lambda: dict(_CW_DEFAULT_TOPOLOGIES))
    # PriorityBand -> Kubernetes PriorityClass name. Sets spec.priorityClassName.
    # UNSPECIFIED is treated as INTERACTIVE. Defaults to the iris-{band} classes
    # Iris creates at startup; override via kubernetes_provider.priority_classes.
    priority_class_names: dict[int, str] = field(default_factory=lambda: dict(_DEFAULT_PRIORITY_CLASS_NAMES))


def _build_task_script(run_req: job_pb2.RunTaskRequest) -> str:
    """Build a shell script that runs the setup steps then the run_command."""
    lines = ["set -e", "ulimit -c 0", "mkdir -p /app", "cd /app"]
    lines.extend(render_setup_steps(run_req.entrypoint.setup_commands))
    # Activate the venv the setup script populated. Conditional on it existing so
    # a custom or no-setup script that brings its own environment runs as-is.
    lines.append('[ -f "$IRIS_VENV/bin/activate" ] && source "$IRIS_VENV/bin/activate"')
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
    workdir_file_refs = dict(run_req.entrypoint.workdir_file_refs)
    has_blob_refs = bool(workdir_file_refs) and bool(controller_address)
    if not has_bundle and not workdir_files and not has_blob_refs:
        return [], [], None

    script_path = Path(__file__).parent / "bundle_fetch.py"
    bundle_script = script_path.read_text()

    init_env: list[dict] = [{"name": "IRIS_WORKDIR", "value": "/app"}]
    init_mounts: list[dict] = [{"name": "workdir", "mountPath": "/app"}]
    extra_volumes: list[dict] = []
    configmap_name: str | None = None

    if has_bundle or has_blob_refs:
        init_env.append({"name": "IRIS_CONTROLLER_URL", "value": controller_address})

    if has_bundle:
        init_env.append({"name": "IRIS_BUNDLE_ID", "value": run_req.bundle_id})

    if has_blob_refs:
        init_env.append({"name": "IRIS_WORKDIR_BLOB_REFS", "value": json.dumps(workdir_file_refs)})

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


def _build_logship_sidecar(
    task_id_wire: str,
    controller_address: str | None,
    logship_image: str,
) -> dict:
    """Build the native log-shipping sidecar container spec.

    A native sidecar (initContainer with ``restartPolicy: Always``) so it starts
    before the task container and is excluded from the pod-phase computation —
    the pod still reaches Succeeded/Failed when only the task container exits,
    and the kubelet terminates the sidecar after it. The sidecar tails the task
    container's CRI log file from the node (mounted read-only via the
    ``varlogpods`` hostPath) and pushes lines to finelog. It resolves the log
    server via the controller and pushes unauthenticated — the finelog log
    service performs no auth, matching the controller's own writes.

    Runs ``logship_image`` (the iris controller image) rather than the task
    image, which lacks the iris package until the task's own dependency sync.
    """
    env: list[dict] = [
        {"name": "IRIS_TASK_ID", "value": task_id_wire},
        {"name": "IRIS_POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
        {"name": "IRIS_POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
    ]
    if controller_address:
        env.append({"name": "IRIS_CONTROLLER_ADDRESS", "value": controller_address})
    return {
        "name": _LOGSHIP_CONTAINER_NAME,
        "image": logship_image,
        "imagePullPolicy": "IfNotPresent",
        "restartPolicy": "Always",
        # iris is installed in the image's .venv (resolved relative to the image
        # WORKDIR), so launch the same interpreter the controller container does.
        "command": [".venv/bin/python", "-m", "iris.cluster.backends.k8s.logship"],
        "env": env,
        "volumeMounts": [{"name": _LOGSHIP_VOLUME_NAME, "mountPath": _NODE_POD_LOG_DIR, "readOnly": True}],
        "resources": {"requests": {"cpu": "50m", "memory": "64Mi"}},
    }


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


def _security_context(profile: int, has_tpu: bool) -> dict:
    """Build the container ``securityContext`` for a container security profile.

    DOCKER_ACCESS is rejected: k8s nodes run containerd, so there is no host
    docker socket to mount, and a weaker context would fake isolation the pod
    does not have.
    """
    resolved = resolve_container_profile(profile)

    if resolved == job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS:
        raise ValueError(
            "container profile DOCKER_ACCESS is not supported on the Kubernetes backend "
            "(nodes run containerd, not dockerd, so there is no host docker socket); use "
            "the docker worker backend, or PRIVILEGED with an in-pod runtime"
        )

    if resolved == job_pb2.CONTAINER_PROFILE_RESTRICTED:
        return {
            "capabilities": {"drop": ["ALL"], "add": []},
            "allowPrivilegeEscalation": False,
            "seccompProfile": {"type": "RuntimeDefault"},
        }

    # DEFAULT and PRIVILEGED keep the profiling cap; TPU adds the memlock cap.
    capabilities = ["SYS_PTRACE"]
    if has_tpu:
        capabilities.append("SYS_RESOURCE")
    ctx: dict = {"capabilities": {"add": capabilities}}
    if resolved == job_pb2.CONTAINER_PROFILE_PRIVILEGED:
        ctx["privileged"] = True
        ctx["allowPrivilegeEscalation"] = True
    return ctx


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
    # Per-task image override (RunTaskRequest.task_image). The init container
    # keeps default_image since it runs iris's own bundle_fetch tooling.
    task_image = run_req.task_image or default_image
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
        requests: dict[str, str] = {}
        if res.cpu_millicores:
            # CPU as a request only (no limits.cpu) so containers can burst onto
            # idle node CPU. The scheduler still places by cpu_millicores, and
            # under contention CFS shares CPU proportionally to requests. This
            # matches the soft-cap behavior the docker runtime uses for
            # CAPACITY_TYPE_ON_DEMAND workers.
            requests["cpu"] = f"{res.cpu_millicores}m"
        if res.memory_bytes:
            # Memory stays a hard cap — overshoot is fatal, not just slow.
            limits["memory"] = str(res.memory_bytes)
            requests["memory"] = str(res.memory_bytes)
        if res.HasField("device"):
            gpu_count = get_gpu_count(res.device)
            has_tpu = res.device.HasField("tpu")
            if gpu_count > 0:
                # K8s treats accelerator limits as implicit requests.
                limits[_GPU_RESOURCE] = str(gpu_count)
                if host_network:
                    # Request RDMA/IB devices for multi-host NCCL over InfiniBand.
                    limits["rdma/ib"] = str(gpu_count)
        if limits:
            resources["limits"] = limits
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
        "image": task_image,
        "imagePullPolicy": "IfNotPresent",
        "env": env_list,
        "workingDir": "/app",
        "volumeMounts": vol_mounts,
        "command": ["bash", "-lc", _build_task_script(run_req)],
    }
    # Operator-injected env (defaults.inject_env). envFrom is the lowest
    # precedence in K8s, so explicit env entries above (user -e, iris vars) win.
    if config.env_secret_name:
        container["envFrom"] = [{"secretRef": {"name": config.env_secret_name, "optional": True}}]

    # Raises for DOCKER_ACCESS, which this backend rejects (see _security_context).
    container["securityContext"] = _security_context(run_req.container_profile, has_tpu)

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
    metadata: dict = {
        "name": pod_name,
        "namespace": namespace,
        "labels": labels,
    }

    # Kueue gang admission for coscheduled jobs. Coscheduling requires Kueue:
    # there is no non-Kueue colocation fallback, so a coscheduled job dispatched
    # to a cluster where Kueue is not configured is a misconfiguration.
    kueue_enabled = bool(run_req.coscheduling.group_by)
    if kueue_enabled and not config.local_queue:
        raise ValueError(
            f"Coscheduled task {run_req.task_id!r} (group_by={run_req.coscheduling.group_by!r}) "
            "requires Kueue gang admission, but Kueue is not configured. Install Kueue "
            "(lib/iris/scripts/install_kueue.py) and set kubernetes_provider.kueue.cluster_queue."
        )
    if kueue_enabled:
        group_by = run_req.coscheduling.group_by
        # group_by must name a topology level this cluster provisioned. An
        # unmapped value is a misconfiguration: it would gang atomically but
        # land unconstrained, which is exactly the silent-placement bug the
        # topology annotation exists to prevent. Fail fast before stamping.
        topo = config.kueue_topologies.get(group_by)
        if topo is None:
            raise ValueError(
                f"Coscheduled task {run_req.task_id!r} has group_by={group_by!r}, which has no "
                f"topology mapping on this cluster (known: {sorted(config.kueue_topologies)}). "
                "group_by must name a topology level the cluster provisioned; configure "
                "kubernetes_provider.kueue.topologies or use a known level."
            )
        labels[_KUEUE_POD_GROUP_NAME] = _pod_group_name(task_id, attempt_id)
        labels[_KUEUE_QUEUE_NAME] = config.local_queue
        # Per-pod ordinal within the gang (0..total-1) for Kueue TAS rank assignment.
        labels[_KUEUE_POD_GROUP_POD_INDEX] = str(task_id.task_index)
        # Stamp the WorkloadPriorityClass only when the cluster maps this band:
        # an unmapped band gets Kueue's default priority, never an invented name.
        wpc = config.kueue_priority_classes.get(run_req.priority)
        if wpc:
            labels[_KUEUE_PRIORITY_CLASS] = wpc
        node_label, required = topo
        anno_key = _KUEUE_REQUIRED_TOPOLOGY if required else _KUEUE_PREFERRED_TOPOLOGY
        metadata["annotations"] = {
            _KUEUE_POD_GROUP_TOTAL: str(run_req.num_tasks),
            anno_key: node_label,
        }

    # Native log-shipping sidecar: ships the task container's node-side CRI log
    # file to finelog. As an initContainer with restartPolicy: Always it is
    # excluded from pod-phase computation, so completion detection (which keys on
    # pod.status.phase) is unaffected. The hostPath volume gives it read-only
    # access to the node's pod log directory.
    logship = _build_logship_sidecar(
        iris_env["IRIS_TASK_ID"],
        config.controller_address,
        config.logship_image,
    )
    volumes.append(
        {
            "name": _LOGSHIP_VOLUME_NAME,
            "hostPath": {"path": _NODE_POD_LOG_DIR, "type": "Directory"},
        }
    )

    spec: dict = {
        "restartPolicy": "Never",
        "containers": [container],
        "initContainers": [logship],
        "volumes": volumes,
    }

    node_selector = _constraints_to_node_selector(run_req.constraints)
    if managed_label:
        node_selector[managed_label] = "true"
    if node_selector:
        spec["nodeSelector"] = node_selector

    if gpu_count > 0:
        spec.setdefault("tolerations", []).append(NVIDIA_GPU_TOLERATION)
        # GPU pools are normally on-demand, but a pool may come up on CoreWeave
        # interruptable capacity (qos.coreweave.cloud/interruptable:NoExecute) when
        # on-demand is exhausted. Tolerate it so pods can land there and Kueue TAS can
        # place the gang (TAS excludes nodes whose NoExecute taints the pod doesn't
        # tolerate). Iris tasks are retryable, so interruptable capacity is acceptable.
        spec.setdefault("tolerations", []).append(COREWEAVE_INTERRUPTABLE_TOLERATION)

    if service_account:
        spec["serviceAccountName"] = service_account
    if host_network:
        spec["hostNetwork"] = True
        spec["dnsPolicy"] = "ClusterFirstWithHostNet"

    # Skip activeDeadlineSeconds for Kueue-gated pods: k8s counts it from pod
    # creation, including time spent SchedulingGated waiting for admission, so
    # a gang that waits on the autoscaler could hit DeadlineExceeded before it
    # ever runs. The controller's own timeout accounting governs these.
    if run_req.HasField("timeout") and run_req.timeout.milliseconds > 0 and not kueue_enabled:
        spec["activeDeadlineSeconds"] = max(1, run_req.timeout.milliseconds // 1000)

    # Stamp the native k8s PriorityClass so the scheduler knows how to
    # preempt/queue this pod relative to others. UNSPECIFIED defaults to
    # INTERACTIVE (the normal user work band). A band with no configured
    # class name leaves priorityClassName unset (cluster default applies).
    effective_band = run_req.priority or job_pb2.PRIORITY_BAND_INTERACTIVE
    priority_class_name = config.priority_class_names.get(effective_band)
    if priority_class_name:
        spec["priorityClassName"] = priority_class_name

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata,
        "spec": spec,
    }


def _task_container_status(pod: dict) -> dict | None:
    """Return the task container's status, matched by name.

    Returns None when the pod has no container statuses yet. Matching by name
    rather than by position keeps exit-code extraction pinned to the task
    container; falls back to the first status if none is named ``task``.
    """
    statuses = pod.get("status", {}).get("containerStatuses", [])
    if not statuses:
        return None
    for status in statuses:
        if status.get("name") == _TASK_CONTAINER_NAME:
            return status
    return statuses[0]


def _is_infrastructure_failure(pod: dict) -> bool:
    """Check if the pod failure was caused by infrastructure (OOM, eviction, etc.).

    Returns True when the terminated reason indicates the failure was NOT caused
    by the application itself, so it should be classified as a worker/preemption
    failure rather than an application failure.
    """
    status = _task_container_status(pod)
    if status is None:
        # Pod-level eviction: the pod status reason indicates infrastructure.
        pod_reason = pod.get("status", {}).get("reason", "")
        return pod_reason in _INFRASTRUCTURE_FAILURE_REASONS
    terminated = status.get("state", {}).get("terminated", {})
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
    """Extract exit code from the task container's terminated state."""
    status = _task_container_status(pod)
    if status is not None:
        terminated = status.get("state", {}).get("terminated", {})
        code = terminated.get("exitCode")
        if isinstance(code, int):
            return code
    return None


def _extract_error(pod: dict) -> str | None:
    """Extract error reason/message from the task container's status."""
    status = _task_container_status(pod)
    if status is None:
        return pod.get("status", {}).get("reason") or None
    terminated = status.get("state", {}).get("terminated", {})
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
# 1 minute bounds how long a wedged gang can pin idle GPU nodes (the pass is two
# field-selector list calls, so the cadence is cheap).
_GC_INTERVAL_SECONDS = 60

# Garbage collection: delete terminal pods and orphaned configmaps/PDBs older than this (seconds).
_GC_MAX_AGE_SECONDS = 3600  # 1 hour

# Garbage collection: shorter retention for terminal gang (Kueue pod-group)
# pods. Gang pods pin Kueue quota and the TAS topology reservation for as long
# as they exist — Kueue rebuilds the pod-group Workload from surviving labeled
# pods — so they cannot get the 1h debugging window plain pods do: every held
# slot is an idle GPU node. 1 minute is still far above pod-status poll
# latency (the reconcile loop runs every few seconds), so prompt deletion
# cannot race exit-status collection.
_GANG_GC_MAX_AGE_SECONDS = 60

# Blocker eviction: minimum interval between reconcile-driven eviction sweeps
# of preempt_namespaces. Gang pods can stay SchedulingGated for many cycles
# while Kueue retries admission; without this floor every reconcile would
# re-list the foreign namespaces and re-issue deletes for pods already
# terminating.
_PREEMPT_INTERVAL_SECONDS = 30


def _has_gated_gang_pods(pods: list[dict]) -> bool:
    """True when any Kueue gang pod is still held by a scheduling gate.

    Kueue's webhook gates every pod carrying the queue-name label and removes
    the gate only on Workload admission, so a surviving gate means the gang is
    still waiting for capacity.
    """
    for pod in pods:
        if _KUEUE_POD_GROUP_NAME not in pod.get("metadata", {}).get("labels", {}):
            continue
        if pod.get("spec", {}).get("schedulingGates"):
            return True
    return False


def _pod_gpu_request(pod: dict) -> int:
    """Total GPUs the pod requests, counting limits as implicit requests."""
    total = 0
    for container in pod.get("spec", {}).get("containers", []):
        resources = container.get("resources", {})
        value = resources.get("requests", {}).get(_GPU_RESOURCE) or resources.get("limits", {}).get(_GPU_RESOURCE)
        if value:
            total += parse_k8s_quantity(str(value))
    return total


def _is_preemptible_blocker(pod: dict) -> bool:
    """Whether a foreign-namespace pod is safe for Iris to evict.

    Hard guards, independent of configuration: the pod must declare a negative
    priority (its priority class marks it scheduler-preemptible by design) AND
    request GPUs (it actually holds capacity Kueue TAS counts against gangs).
    Terminal and already-terminating pods are skipped.
    """
    meta = pod.get("metadata", {})
    if meta.get("deletionTimestamp"):
        return False
    if pod.get("status", {}).get("phase") in ("Succeeded", "Failed"):
        return False
    priority = pod.get("spec", {}).get("priority")
    if not isinstance(priority, int) or priority >= 0:
        return False
    return _pod_gpu_request(pod) > 0


def _kueue_workloads_by_name(workloads: list[dict]) -> dict[str, dict]:
    result = {}
    for workload in workloads:
        name = workload.get("metadata", {}).get("name", "")
        if name:
            result[name] = workload
    return result


def _format_kueue_condition(cond: dict) -> str:
    """Render one Kueue Workload condition as a compact diagnostic."""
    condition_type = cond.get("type", "Condition")
    status = cond.get("status", "")
    reason = cond.get("reason", "")
    message = cond.get("message", "")

    prefix = condition_type
    if status and status != "False":
        prefix = f"{prefix}={status}"
    if reason:
        prefix = f"{prefix} ({reason})"
    return f"{prefix}: {message}" if message else prefix


def _format_kueue_workload_status(pod: dict, workload: dict | None) -> str:
    """Return Kueue admission context for a gated pod."""
    pod_group = pod.get("metadata", {}).get("labels", {}).get(_KUEUE_POD_GROUP_NAME, "")
    if workload is None:
        return f"Kueue workload {pod_group!r} not found yet; waiting for Kueue to create/admit the pod group"

    spec = workload.get("spec", {})
    status = workload.get("status", {})
    details = []

    cluster_queue = status.get("admission", {}).get("clusterQueue", "")
    queue_name = spec.get("queueName", "")
    if queue_name and cluster_queue:
        details.append(f"queue={queue_name}, clusterQueue={cluster_queue}")
    elif queue_name:
        details.append(f"queue={queue_name}")
    elif cluster_queue:
        details.append(f"clusterQueue={cluster_queue}")

    conditions = status.get("conditions", [])
    waiting_conditions = [
        _format_kueue_condition(cond)
        for cond in conditions
        if cond.get("status") != "True" and (cond.get("reason") or cond.get("message"))
    ]
    if waiting_conditions:
        details.extend(waiting_conditions)
    elif status.get("admission"):
        details.append("admitted by Kueue; waiting for scheduler gate removal")
    else:
        details.append("waiting for Kueue admission")

    workload_name = workload.get("metadata", {}).get("name", pod_group)
    return f"Kueue workload {workload_name}: " + "; ".join(details)


def _build_pod_statuses(
    pods: list[dict], workloads: list[dict] | None = None
) -> list[controller_pb2.Controller.KubernetesPodStatus]:
    """Build pod status protos from raw kubectl pod objects."""
    statuses = []
    workloads_by_name = _kueue_workloads_by_name(workloads or [])
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
                            dt = parse_k8s_timestamp(last_transition_str)
                            last_ts = Timestamp.from_seconds(dt.timestamp())
                        except (ValueError, AttributeError):
                            pass
                    break
        if reason == "SchedulingGated" and labels.get(_KUEUE_POD_GROUP_NAME):
            pod_group = labels[_KUEUE_POD_GROUP_NAME]
            kueue_status = _format_kueue_workload_status(pod, workloads_by_name.get(pod_group))
            message = f"{message}; {kueue_status}" if message else kueue_status

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
    kubectl data. to_status_response() may be called from any thread (e.g.
    the dashboard RPC handler) without holding any external lock — the
    internal lock is acquired only for the brief copy.

    Pods are kept sorted by name so that pagination is stable across
    consecutive dashboard polls.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pods: list[dict] = []
        self._nodes: list[dict] = []
        self._workloads: list[dict] = []
        self._node_pools: list[controller_pb2.Controller.NodePoolStatus] = []

    def update(
        self,
        pods: list[dict],
        nodes: list[dict],
        workloads: list[dict],
        node_pools: list[controller_pb2.Controller.NodePoolStatus],
    ) -> None:
        """Atomically replace all cluster state from a completed sync cycle."""
        new_pods = sorted(pods, key=lambda p: p.get("metadata", {}).get("name", ""))
        new_nodes = sorted(nodes, key=lambda n: n.get("metadata", {}).get("name", ""))
        new_workloads = sorted(workloads, key=lambda w: w.get("metadata", {}).get("name", ""))
        with self._lock:
            self._pods = new_pods
            self._nodes = new_nodes
            self._workloads = new_workloads
            self._node_pools = list(node_pools)

    def to_status_response(self, namespace: str) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Build the dashboard RPC response from current state. No kubectl calls."""
        with self._lock:
            pods = self._pods[:]
            nodes = self._nodes[:]
            workloads = self._workloads[:]
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
            pod_statuses=_build_pod_statuses(pods, workloads),
            provider_version="iris-kubernetes/v1",
            node_pools=node_pools,
        )


class ResourceCollector:
    """Background thread that samples running pods' CPU/memory usage.

    The reconcile loop declares the authoritative set of running pods via
    ``set_pods()`` once per cycle. Each ``poll_interval`` the collector samples
    those pods via one bulk metrics query and appends an ``IrisTaskStat`` row
    per pod to the ``iris.task`` table — the same table the worker daemon writes
    to on the GCE/TPU path, so the dashboard's ``iris.task`` queries cover both
    runtimes uniformly.

    ``poll_interval`` defaults to the metrics-server scrape resolution (15s);
    polling faster only re-reads the same sample.
    """

    def __init__(
        self,
        kubectl: K8sService,
        task_stats_table: Table,
        *,
        labels: dict[str, str] | None = None,
        poll_interval: float = 15.0,
    ):
        self._kubectl = kubectl
        self._table = task_stats_table
        self._labels = labels
        self._poll_interval = poll_interval
        # (task_id_wire, attempt_id) -> pod_name. Tuple keys carry the
        # identity needed to build IrisTaskStat without parsing strings.
        self._pods: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="resource-collector")
        self._thread.start()

    def set_pods(self, pods: dict[tuple[str, int], str]) -> None:
        """Declare the authoritative set of pods to collect resources for."""
        with self._lock:
            self._pods = dict(pods)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.collect_once()
            self._stop.wait(timeout=self._poll_interval)

    def collect_once(self) -> None:
        """Sample every tracked pod once and append a stat row per pod with usage.

        Runs each ``poll_interval`` on the background thread; also the unit of
        collection tests drive directly.
        """
        with self._lock:
            snapshot = list(self._pods.items())
        if not snapshot:
            return
        try:
            usage_by_pod = self._kubectl.top_pods(labels=self._labels)
        except Exception as e:
            logger.debug("ResourceCollector: top_pods raised: %s", e)
            return

        stats: list[IrisTaskStat] = []
        for (task_id_wire, attempt_id), pod_name in snapshot:
            top = usage_by_pod.get(pod_name)
            if top is None:
                continue
            stats.append(
                build_task_stat(
                    task_id=task_id_wire,
                    attempt_id=attempt_id,
                    # Pod name is the per-attempt platform identity on k8s,
                    # mirroring worker_id on the GCE/TPU path.
                    worker_id=pod_name,
                    usage=job_pb2.ResourceUsage(
                        cpu_millicores=top.cpu_millicores,
                        memory_mb=top.memory_bytes // (1024 * 1024),
                    ),
                )
            )
        if not stats:
            return
        try:
            self._table.write(stats)
        except Exception:
            logger.debug("ResourceCollector: write to iris.task failed", exc_info=True)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def _get_pod_node_name(kubectl: K8sService, pod_name: str) -> str:
    """Return the pod's spec.nodeName, or empty string if unschedulable / not yet bound."""
    try:
        pod = kubectl.get_json(K8sResource.PODS, pod_name)
    except KubectlError:
        return ""
    if pod is None:
        return ""
    return pod.get("spec", {}).get("nodeName", "") or ""


@dataclass(frozen=True)
class _K8sProfileDispatch:
    """``ProfileDispatch`` backed by ``kubectl exec`` into a task pod.

    Profiler and transform commands run with the task venv activated. Profilers
    run in the pod's isolated PID namespace, so ``exec_profiler`` applies the
    kill-watchdog and SIGCONT recovery (see ``runtime.profile``).
    """

    kubectl: K8sService
    pod_name: str
    pyspy_bin: str = "py-spy"
    memray_bin: str = "memray"

    @contextmanager
    def scratch(self, *suffixes: str) -> Iterator[tuple[str, ...]]:
        paths = tuple(f"/tmp/iris-profile.{suffix}" for suffix in suffixes)
        try:
            yield paths
        finally:
            self.kubectl.rm_files(self.pod_name, list(paths), container="task")

    def exec_profiler(self, cmd: list[str], *, sample_timeout: int) -> ExecResult:
        watchdog_cmd = wrap_with_kill_watchdog(cmd, sample_timeout)
        try:
            return self._venv_exec(watchdog_cmd, timeout=sample_timeout + PROFILER_WATCHDOG_GRACE_SECONDS)
        finally:
            self._sigcont_sweep()

    def exec(self, cmd: list[str], *, timeout: int) -> ExecResult:
        return self._venv_exec(cmd, timeout=timeout)

    def read_file(self, path: str) -> bytes:
        return self.kubectl.read_file(self.pod_name, path, container="task")

    def _venv_exec(self, cmd: list[str], *, timeout: int) -> ExecResult:
        shell_cmd = ["bash", "-lc", f"source {VENV_PATH}/bin/activate 2>/dev/null; {shlex.join(cmd)}"]
        result = self.kubectl.exec(self.pod_name, shell_cmd, container="task", timeout=timeout)
        return ExecResult(result.returncode, (result.stdout or "").encode("utf-8"), result.stderr or "")

    def _sigcont_sweep(self) -> None:
        try:
            self.kubectl.exec(self.pod_name, sigcont_sweep_argv(), container="task", timeout=10)
        except Exception as e:
            logger.warning("SIGCONT sweep failed for pod %s: %s", self.pod_name, e)


@dataclass
class K8sTaskProvider:
    """Executes tasks as Kubernetes Pods without worker daemons.

    A cluster :class:`~iris.cluster.controller.backend.TaskBackend`: Kueue owns
    placement, so ``schedule`` and ``autoscale`` are no-ops; ``reconcile``
    consumes the dispatch drain (``tasks_to_run`` + ``running_tasks``) carried on
    the :class:`ReconcileRequest` and returns neutral task ``updates``. K8s pods
    are launched and monitored directly via kubectl rather than through a worker
    gRPC daemon.

    Capacity is derived from node allocatable resources minus running pod
    resource requests, queried via kubectl each sync cycle.

    Pod naming: iris-{task_id_sanitized}-{attempt_id}
    """

    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.CLUSTER_VIEW})

    kubectl: K8sService
    namespace: str
    default_image: str
    # Iris controller image, used for the log-shipper sidecar (see PodConfig).
    logship_image: str = ""
    cache_dir: str = "/cache"
    service_account: str = ""
    host_network: bool = False
    controller_address: str | None = None
    managed_label: str = ""
    task_env: dict[str, str] = field(default_factory=dict)
    env_secret_name: str = ""
    local_queue: str = ""
    kueue_priority_classes: dict[int, str] = field(default_factory=dict)
    kueue_topologies: dict[str, tuple[str, bool]] = field(default_factory=lambda: dict(_CW_DEFAULT_TOPOLOGIES))
    priority_class_names: dict[int, str] = field(default_factory=lambda: dict(_DEFAULT_PRIORITY_CLASS_NAMES))
    # Namespaces whose preemptible (negative-priority) GPU pods Iris evicts
    # when it has gang work for Kueue. Empty disables the feature; see
    # _evict_preemptible_blockers for the safety guards.
    preempt_namespaces: list[str] = field(default_factory=list)
    # Pre-resolved iris.task Table handle, built from the controller's log client
    # and passed in by the composer; when None — e.g. tests without finelog — the
    # resource collector is disabled. K8s pods ship their own logs via the
    # log-shipper sidecar, so the backend needs only the tables, not the client.
    task_stats_table: Table | None = None
    # Pre-resolved iris.profile Table handle, passed alongside task_stats_table.
    # None in test mode.
    profile_table: Table | None = None
    # Resource-usage poll cadence. Defaults to the metrics-server scrape
    # resolution (15s) — sampling faster only re-reads the same value. One bulk
    # metrics list per tick covers every managed pod (see ResourceCollector).
    resource_poll_interval: float = 15.0
    # Cluster-wide kubectl scans (pod list, stray-pod GC, pod poll, node refresh)
    # are coarse-grained: the controller ticks reconcile at poll_interval (1s),
    # but these LISTs run at most once per cluster_scan_interval to bound kubectl
    # load. New-pod application (dispatch) is NOT gated — it runs every tick.
    # Tests set this to 0.0 so every reconcile scans.
    cluster_scan_interval: float = 5.0
    name: str = "kubernetes"
    # Routing metadata the meta-scheduler reads, set by the composer via configure_routing.
    advertised: dict[str, set[str]] = field(default_factory=dict)
    allowed_users: frozenset[str] = frozenset({"*"})
    # K8s provisions its own capacity (cluster autoscaler + Kueue); no Iris autoscaler.
    autoscaler: Autoscaler | None = field(default=None, init=False, repr=False)
    # A cluster backend tracks no Iris worker liveness; the controller's union read
    # skips a None tracker, and no worker registers into a k8s scale group.
    health: WorkerHealthTracker | None = field(default=None, init=False, repr=False)
    # The controller-DB read surface this backend authors its dispatch effects
    # from, passed by the composer at construction (a cluster backend has no
    # worker store, so it reads its dispatch drain through this).
    transition_reader: TransitionReader | None = field(default=None, repr=False)
    _pod_not_found_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _resource_collector: ResourceCollector | None = field(default=None, init=False, repr=False)
    _cluster_state: ClusterState = field(default_factory=ClusterState, init=False, repr=False)
    _last_gc_time: float = field(default=0.0, init=False, repr=False)
    _last_cluster_scan: float = field(default=0.0, init=False, repr=False)
    _last_preempt_time: float = field(default=0.0, init=False, repr=False)
    _pending_gc_hashes: set[str] = field(default_factory=set, init=False, repr=False)

    def _ensure_resource_collector(self) -> ResourceCollector | None:
        if self.task_stats_table is None:
            return None
        if self._resource_collector is None:
            self._resource_collector = ResourceCollector(
                self.kubectl,
                self.task_stats_table,
                labels=_MANAGED_POD_LABELS,
                poll_interval=self.resource_poll_interval,
            )
        return self._resource_collector

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def admits(self, user: str) -> bool:
        return user_admitted(self.allowed_users, user)

    def configure_routing(self, advertised: dict[str, set[str]], allowed_users: frozenset[str]) -> None:
        self.advertised = advertised
        self.allowed_users = allowed_users

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """No-op: Kueue owns placement, so Iris makes no scheduling decisions."""
        return ScheduleResult()

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """No-op: the cluster autoscaler + Kueue provision nodes; K8s has no
        Iris-managed slices to tear down."""
        return AutoscaleResult()

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        """No-op: a cluster backend tracks no Iris workers, so it builds no worker source."""

    def seed_liveness(self) -> None:
        """No-op: a cluster backend tracks no Iris worker liveness to seed."""

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        """Author the pod projection: sync task state, then resolve it into effects.

        ``sync`` converges the cluster (apply new pods, delete strays, poll running
        pods) and returns the neutral task updates it observed; this resolves those
        into committable task ``effects`` against the backend's own read snapshot.
        A cluster backend tracks no Iris workers, so it folds no liveness.
        """
        assert self.transition_reader is not None, "K8sTaskProvider.reconcile called before transition reader attached"
        updates = self.sync(request)
        effects = apply_dispatch_updates(self.transition_reader, updates, now=Timestamp.now())
        return ReconcileResult(effects=effects)

    def run_teardown(self) -> None:
        """No-op: a cluster backend tracks no Iris workers to reap."""

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        """No-op: a cluster backend tracks no Iris workers to reap."""

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        """No-op: a cluster backend tracks no Iris workers to garbage-collect."""
        return 0

    def sync(self, request: ReconcileRequest) -> list[TaskUpdate]:
        """Sync task state: apply new pods, delete strays, poll running pods.

        Kill targets are derived here, not buffered in the controller: any
        managed pod whose ``(task_hash, attempt_id)`` is not in the desired
        set (``tasks_to_run`` union ``running_tasks``) is deleted on this tick.
        Producing transitions only need to update ``tasks.state``; the next
        sync sees the diff.

        New-pod application runs every tick so dispatch stays responsive; the
        cluster-wide kubectl scans (pod list, stray-pod GC, pod poll, node
        refresh, terminal GC) run at most once per ``cluster_scan_interval``,
        and continue to run on an idle cluster (the controller never gates a
        cluster backend's reconcile on having work) so orphaned pods are reaped.
        """
        # Free GPU capacity for incoming gangs before their pods are created:
        # Kueue TAS computes node capacity at admission, so blockers must be
        # gone (or terminating) by the time it evaluates the new Workload.
        if self.preempt_namespaces and any(r.coscheduling.group_by for r in request.tasks_to_run):
            self._evict_preemptible_blockers(reason="coscheduled gang submission", force=True)

        apply_failures: list[TaskUpdate] = []
        for run_req in request.tasks_to_run:
            try:
                self._apply_pod(run_req)
            except KubectlError as exc:
                logger.error("Failed to apply pod for task %s: %s", run_req.task_id, exc)
                # The pod was never created, so there is no k8s verdict to track
                # and nothing ran. Treat any apply failure as worker loss so the
                # task retries (ASSIGNED -> WORKER_FAILED rolls back to PENDING
                # without charging the preemption budget) and the next sync
                # re-applies. The raw k8s error is logged above.
                apply_failures.append(
                    TaskUpdate(
                        task_id=JobName.from_wire(run_req.task_id),
                        attempt_id=run_req.attempt_id,
                        new_state=job_pb2.TASK_STATE_WORKER_FAILED,
                        error=str(exc),
                    )
                )

        now = time.time()
        if now - self._last_cluster_scan < self.cluster_scan_interval:
            return apply_failures
        self._last_cluster_scan = now

        # Single pod list for the entire cycle — excludes terminal pods via field selector.
        managed_pods = self.kubectl.list_json(
            K8sResource.PODS,
            labels=_MANAGED_POD_LABELS,
            field_selector=_ACTIVE_PODS_FIELD_SELECTOR,
        )

        # Blockers can also appear AFTER submission (health checks target any
        # idle GPU node), so keep evicting while a gang waits for admission.
        if self.preempt_namespaces and _has_gated_gang_pods(managed_pods):
            self._evict_preemptible_blockers(reason="gang pods held SchedulingGated awaiting Kueue admission")

        desired_keys: set[tuple[str, int]] = set()
        for run_req in request.tasks_to_run:
            desired_keys.add((_task_hash(run_req.task_id), int(run_req.attempt_id)))
        for entry in request.running_tasks:
            desired_keys.add((_task_hash(entry.task_id.to_wire()), int(entry.attempt_id)))
        self._delete_stray_pods(managed_pods, desired_keys)
        updates = apply_failures + self._poll_pods(request.running_tasks, managed_pods)

        try:
            nodes = self.kubectl.list_json(K8sResource.NODES)
        except Exception as e:
            logger.warning("Failed to query node resources: %s", e)
            nodes = []

        if self.local_queue:
            try:
                workloads = self.kubectl.list_json(K8sResource.WORKLOADS)
            except Exception as e:
                logger.warning("Failed to query Kueue workloads: %s", e)
                workloads = []
        else:
            workloads = []

        node_pools = _fetch_node_pools(self.kubectl, self.managed_label)
        self._cluster_state.update(managed_pods, nodes, workloads, node_pools)

        self._maybe_gc_terminal_resources(managed_pods)

        return updates

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        """Profile a running task pod via kubectl exec.

        On success, writes one IrisProfile row to the finelog profile_table
        (when not None). On failure, returns ProfileTaskResponse(error=...) and
        skips the write. ``timeout_ms`` is unused — kubectl exec is bounded by
        the profile duration itself.
        """
        attempt_id = target.attempt_id
        pod_name = _pod_name(JobName.from_wire(target.task_id), attempt_id)
        duration = request.duration_seconds or 10
        profile_type = request.profile_type
        dispatch = _K8sProfileDispatch(self.kubectl, pod_name)

        try:
            if profile_type.HasField("threads"):
                data = capture_threads(dispatch, pid="1", include_locals=profile_type.threads.locals)
            elif profile_type.HasField("cpu"):
                data = capture_cpu(dispatch, profile_type.cpu, duration, pid="1")
            elif profile_type.HasField("memory"):
                data = capture_memory_attach(dispatch, profile_type.memory, duration, pid="1")
            else:
                return job_pb2.ProfileTaskResponse(error="Unknown profile type")
        except Exception as e:
            return job_pb2.ProfileTaskResponse(error=str(e))

        resp = job_pb2.ProfileTaskResponse(profile_data=data)

        if self.profile_table is not None and resp.profile_data:
            pod_node_name = _get_pod_node_name(self.kubectl, pod_name)
            row = build_profile_row(
                source=request.target,
                attempt_id=attempt_id,
                vm_id=f"k8s/{pod_node_name or pod_name}",
                duration_seconds=duration,
                profile_type=profile_type,
                profile_data=resp.profile_data,
            )
            self.profile_table.write([row])

        return resp

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        """Execute a command in a running task pod via kubectl exec."""
        command = list(request.command)
        pod_name = _pod_name(JobName.from_wire(target.task_id), target.attempt_id)
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

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("K8s backend does not support per-process status")

    def close(self) -> None:
        if self._resource_collector is not None:
            self._resource_collector.close()

    def get_cluster_status(self) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        """Return cluster status from the latest sync() snapshot. No kubectl calls."""
        return self._cluster_state.to_status_response(self.namespace)

    def status(self) -> controller_pb2.Controller.BackendStatus:
        """Author the ``kubernetes`` status variant from the cluster-state snapshot."""
        return controller_pb2.Controller.BackendStatus(kubernetes=self.get_cluster_status())

    def autoscaler_status(self) -> vm_pb2.AutoscalerStatus:
        """Empty: K8s provisions its own capacity and runs no Iris autoscaler."""
        return vm_pb2.AutoscalerStatus()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    @property
    def pod_config(self) -> PodConfig:
        """Build PodConfig from provider fields."""
        return PodConfig(
            namespace=self.namespace,
            default_image=self.default_image,
            logship_image=self.logship_image,
            cache_dir=self.cache_dir,
            service_account=self.service_account,
            host_network=self.host_network,
            controller_address=self.controller_address,
            managed_label=self.managed_label,
            task_env=self.task_env,
            env_secret_name=self.env_secret_name,
            local_queue=self.local_queue,
            kueue_priority_classes=self.kueue_priority_classes,
            kueue_topologies=self.kueue_topologies,
            priority_class_names=self.priority_class_names,
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

        # Prepend the workdir-staging init containers before the log-shipper
        # native sidecar already on the manifest: staging must run to completion
        # first; the native sidecar starts before the task container regardless
        # of its position in the list.
        if init_containers:
            manifest["spec"]["initContainers"] = init_containers + manifest["spec"]["initContainers"]
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

    def _evict_preemptible_blockers(self, *, reason: str, force: bool = False) -> None:
        """Delete preemptible GPU pods from preempt_namespaces to unblock gang admission.

        Kueue TAS counts every non-Kueue pod's GPU requests as fixed node usage
        and its preemption only targets Kueue Workloads, while gang pods never
        reach the kube-scheduler until admitted — so pods the kube-scheduler
        would displace (negative priority, PreemptLowerPriority) instead starve
        gangs indefinitely. Iris performs that eviction itself, in the layer it
        owns.

        Safety guards regardless of configuration: only pods that pass
        _is_preemptible_blocker (negative priority AND GPU request, not already
        terminating) are deleted, and Iris's own namespace is never touched.
        ``force`` bypasses the debounce for discrete events (gang submission);
        the reconcile-driven path is rate-limited to _PREEMPT_INTERVAL_SECONDS.

        Eviction is best-effort: per-namespace list/delete failures are logged
        and skipped so they can never block task dispatch in reconcile().
        """
        now = time.monotonic()
        if not force and now - self._last_preempt_time < _PREEMPT_INTERVAL_SECONDS:
            return
        self._last_preempt_time = now
        for ns in self.preempt_namespaces:
            if ns == self.namespace:
                logger.warning("preempt_namespaces includes iris's own namespace %r; refusing to evict there", ns)
                continue
            try:
                pods = self.kubectl.list_pods_in_namespace(ns)
            except KubectlError as e:
                logger.warning("Failed to list pods in preempt namespace %s: %s", ns, e)
                continue
            for pod in pods:
                if not _is_preemptible_blocker(pod):
                    continue
                name = pod.get("metadata", {}).get("name", "")
                if not name:
                    continue
                logger.info(
                    "Evicting preemptible blocker pod %s/%s (priority=%s, gpus=%d): %s",
                    ns,
                    name,
                    pod.get("spec", {}).get("priority"),
                    _pod_gpu_request(pod),
                    reason,
                )
                try:
                    self.kubectl.delete_pod_in_namespace(ns, name)
                except KubectlError as e:
                    logger.warning("Failed to evict blocker pod %s/%s: %s", ns, name, e)

    def _delete_stray_pods(self, cached_pods: list[dict], desired_keys: set[tuple[str, int]]) -> None:
        """Delete pods that aren't in the desired ``(task_hash, attempt_id)`` set.

        Stray = the controller no longer wants this attempt running (task is
        terminal in ``tasks``, or the attempt has rolled to a newer one). The
        producing transition has already updated ``tasks.state``; we observe
        the absence here and tear the pod down.

        ConfigMaps and PDBs are cleaned up by the periodic GC pass
        (_gc_terminal_resources) to avoid listing all configmaps/PDBs on
        every sync cycle — which was an O(total_resources) scan on the hot
        path.
        """
        stray_pod_names: list[str] = []
        stray_hashes: set[str] = set()
        stray_pod_groups: set[str] = set()
        stray_gang_pod_names: list[str] = []
        for pod in cached_pods:
            labels = pod.get("metadata", {}).get("labels", {})
            task_hash = labels.get(_LABEL_TASK_HASH)
            attempt_str = labels.get(_LABEL_ATTEMPT_ID)
            if not task_hash or attempt_str is None:
                continue
            try:
                attempt_id = int(attempt_str)
            except (ValueError, TypeError):
                continue
            if (task_hash, attempt_id) in desired_keys:
                continue
            pod_name = pod.get("metadata", {}).get("name")
            if pod_name:
                stray_pod_names.append(pod_name)
                stray_hashes.add(task_hash)
                pod_group = labels.get(_KUEUE_POD_GROUP_NAME)
                if pod_group:
                    stray_pod_groups.add(pod_group)
                    stray_gang_pod_names.append(pod_name)

        if not stray_pod_names:
            return

        self.kubectl.delete_many(K8sResource.PODS, stray_pod_names, wait=False)
        # The GC pass re-drives any gang pods that survive this teardown.
        self._release_gang_reservations(stray_gang_pod_names, stray_pod_groups)
        # Enqueue task hashes for deferred configmap/PDB cleanup by the GC pass.
        self._pending_gc_hashes.update(stray_hashes)

        logger.info(
            "Deleted %d stray pods for %d task hashes (%d Kueue workloads released, CM/PDB cleanup deferred to GC)",
            len(stray_pod_names),
            len(stray_hashes),
            len(stray_pod_groups),
        )

    def _release_gang_reservations(self, gang_pod_names: list[str], pod_groups: set[str]) -> None:
        """Release the Kueue gang reservation for torn-down pod-group generations.

        Kueue parks a coscheduled Workload in WaitingForReplacementPods when its
        pods are deleted, holding the quota until the Workload itself is removed;
        a gang requeue (which bumps to a new pod-group generation) would deadlock
        behind the old generation's still-reserved quota. Stripping Kueue's pod
        finalizer is what guarantees the labeled pods actually disappear —
        otherwise Kueue rebuilds the Workload from the surviving pods and
        re-holds the quota/TAS slots.
        """
        for pod_name in gang_pod_names:
            self.kubectl.remove_finalizer(K8sResource.PODS, pod_name, _KUEUE_MANAGED_FINALIZER)
        self._delete_kueue_workloads(pod_groups)

    def _delete_kueue_workloads(self, pod_group_names: set[str]) -> None:
        """Delete the Kueue Workload backing each coscheduled pod-group generation.

        Kueue names the Workload after the pod-group-name, so the name Iris
        stamped on the pods is the Workload name. Deletion is idempotent
        (NotFound is ignored), so it is safe for non-Kueue clusters and for
        groups whose Workload Kueue already finished on its own.
        """
        for name in pod_group_names:
            self.kubectl.delete(K8sResource.WORKLOADS, name, wait=False)

    def _maybe_gc_terminal_resources(self, active_pods: list[dict]) -> None:
        """Periodically delete terminal (Succeeded/Failed) pods and their associated
        configmaps/PDBs that are older than _GC_MAX_AGE_SECONDS, and sweep terminal
        gang pods (with their Kueue Workloads) on the shorter gang retention.

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
        """One GC pass: deferred CM/PDB cleanup, the 1h terminal-pod sweep, and a
        short-retention sweep of terminal gang pods that strips the Kueue pod
        finalizer and deletes the pod-group Workloads they would otherwise pin.
        """
        now = datetime.now(UTC).timestamp()
        cutoff = now - _GC_MAX_AGE_SECONDS
        gang_cutoff = now - _GANG_GC_MAX_AGE_SECONDS

        # Collect task hashes that still have active (Pending/Running) pods.
        # These must NOT have their configmaps/PDBs deleted, even if an older
        # attempt of the same task is terminal — task_hash is shared across attempts.
        active_hashes: set[str] = set()
        # Pod-groups with live (Pending/Running) members share one Kueue
        # Workload across the gang; releasing it would evict the running
        # siblings, so the gang sweep must skip those groups entirely.
        active_gang_groups: set[str] = set()
        for pod in active_pods:
            labels = pod.get("metadata", {}).get("labels", {})
            h = labels.get(_LABEL_TASK_HASH)
            if h:
                active_hashes.add(h)
            g = labels.get(_KUEUE_POD_GROUP_NAME)
            if g:
                active_gang_groups.add(g)

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
        gang_pod_names: list[str] = []
        gang_pod_groups: set[str] = set()
        gang_task_hashes: set[str] = set()
        for pod in self._list_terminal_pods():
            meta = pod.get("metadata", {})
            created = meta.get("creationTimestamp", "")
            if not created:
                continue
            ts = parse_k8s_timestamp(created).timestamp()
            task_hash = meta.get("labels", {}).get(_LABEL_TASK_HASH)
            pod_group = meta.get("labels", {}).get(_KUEUE_POD_GROUP_NAME)
            # Gang sweep: a deletionTimestamp means a prior delete is
            # wedged on the Kueue finalizer; otherwise the shorter gang
            # retention applies. Handled pods are excluded from the 1h
            # sweep below. Pods whose group still has live members are
            # deferred wholesale (not even age-swept): a partial delete
            # would wedge on the finalizer, and releasing the shared
            # Workload would evict the running siblings.
            if pod_group and pod_group in active_gang_groups:
                continue
            if pod_group and (meta.get("deletionTimestamp") or ts < gang_cutoff):
                gang_pod_names.append(meta["name"])
                gang_pod_groups.add(pod_group)
                if task_hash:
                    gang_task_hashes.add(task_hash)
                continue
            if ts < cutoff:
                old_pod_names.append(meta["name"])
                if task_hash:
                    old_task_hashes.add(task_hash)

        if gang_pod_names:
            # force (gracePeriodSeconds=0): these pods are already terminal, so
            # there is nothing to terminate gracefully, and force unsticks
            # deletion when the node's kubelet is gone (node failure).
            self.kubectl.delete_many(K8sResource.PODS, gang_pod_names, force=True, wait=False)
            self._release_gang_reservations(gang_pod_names, gang_pod_groups)
            # CM/PDB cleanup follows the deferred path so active retry
            # attempts sharing the task hash keep their resources.
            self._pending_gc_hashes.update(gang_task_hashes)
            logger.info(
                "GC: swept %d terminal gang pods, released %d Kueue workloads",
                len(gang_pod_names),
                len(gang_pod_groups),
            )

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

    def _list_terminal_pods(self) -> list[dict]:
        """Bulk-list managed pods in a terminal phase (Succeeded or Failed)."""
        pods: list[dict] = []
        # Field selectors AND their comma-separated terms, so a single
        # status.phase==Succeeded,status.phase==Failed matches nothing (a pod is
        # never both); list each terminal phase separately.
        for phase in ("Succeeded", "Failed"):
            pods.extend(
                self.kubectl.list_json(
                    K8sResource.PODS,
                    labels=_MANAGED_POD_LABELS,
                    field_selector=f"status.phase={phase}",
                )
            )
        return pods

    def _poll_pods(self, running: list[RunningTaskEntry], cached_pods: list[dict]) -> list[TaskUpdate]:
        """Poll pod phases for all running tasks.

        Uses the pre-fetched active-pods list (terminal pods excluded by field
        selector). Running tasks whose pod has left that list have either
        completed (phase moved to Succeeded/Failed) or vanished; they are
        resolved with a single bulk terminal-pods list rather than a per-pod
        get_json each, so the reconcile thread issues only bulk LISTs even when a
        whole gang finishes in one cycle. A pod absent from both lists falls to
        the grace-period path below.

        Task logs are shipped by the per-pod log-shipper sidecar, not pulled
        here. This method drives task state and registers running pods with the
        ResourceCollector, calling set_pods() once with the authoritative set of
        running pods so the collector can never drift.
        """
        if not running:
            if self._resource_collector is not None:
                self._resource_collector.set_pods({})
            return []

        pods_by_name: dict[str, dict] = {pod.get("metadata", {}).get("name", ""): pod for pod in cached_pods}
        updates: list[TaskUpdate] = []

        # Resolve running tasks whose pod has left the active list (completed or
        # vanished) with one bulk terminal-pods list instead of a per-pod GET
        # each. Lazy: only fetched on cycles where at least one pod is missing,
        # so steady-state cycles add no call. setdefault keeps the active entry
        # if a name somehow appears in both.
        if any(_pod_name(entry.task_id, entry.attempt_id) not in pods_by_name for entry in running):
            for pod in self._list_terminal_pods():
                pods_by_name.setdefault(pod.get("metadata", {}).get("name", ""), pod)

        # (task_id_wire, attempt_id) -> pod_name. Resource samples are
        # appended directly to iris.task by the collector; the controller no
        # longer multiplexes them through TaskUpdate.
        resource_pods: dict[tuple[str, int], str] = {}

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
                            new_state=job_pb2.TASK_STATE_RUNNING,
                        )
                    )
                    continue
                # Grace exhausted — pod is truly gone. For a coscheduled task
                # this is almost always a Kueue gang preemption (Kueue deletes
                # every pod in a preempted group, leaving no terminal status to
                # read), so bill it to the preemption budget (WORKER_FAILED)
                # rather than the application budget (FAILED).
                self._pod_not_found_counts.pop(cursor_key, None)
                gone_state = job_pb2.TASK_STATE_WORKER_FAILED if entry.coscheduled else job_pb2.TASK_STATE_FAILED
                updates.append(
                    TaskUpdate(
                        task_id=entry.task_id,
                        attempt_id=entry.attempt_id,
                        new_state=gone_state,
                        error="Pod not found",
                    )
                )
                continue

            self._pod_not_found_counts.pop(cursor_key, None)
            update = _task_update_from_pod(entry, pod)
            phase = pod.get("status", {}).get("phase", "")
            if phase == "Running":
                resource_pods[(entry.task_id.to_wire(), entry.attempt_id)] = pod_name

            updates.append(update)

        resource_collector = self._ensure_resource_collector()
        if resource_collector is not None:
            resource_collector.set_pods(resource_pods)

        return updates
