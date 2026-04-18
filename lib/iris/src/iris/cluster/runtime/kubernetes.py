# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kubernetes-native runtime for task execution.

This runtime maps the ContainerRuntime protocol to Kubernetes Pods. Each Iris
task attempt is represented as one Pod. GPU and RDMA resources must be
explicitly requested in the Pod spec (the kubelet device plugin allocates them).
When using this runtime on CoreWeave, the **worker Pod must not request
GPU/RDMA resources** — those are claimed by the task Pods instead.
"""

from __future__ import annotations

import base64
import logging
import os
import shlex
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from iris.cluster.bundle import BundleStore
from iris.cluster.runtime.env import normalize_workdir_relative_path
from iris.cluster.k8s.kubectl import Kubectl, KubectlLogLine
from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    build_pyspy_dump_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.cluster.types import get_gpu_count
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerPhase,
    ContainerStats,
    ContainerStatus,
    MountKind,
)
from iris.cluster.worker.worker_types import LogLine
from iris.rpc import cluster_pb2
from iris.time_utils import Deadline, Duration

logger = logging.getLogger(__name__)

# Kubernetes label values must match: (([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?
# and be at most 63 characters. Task IDs like "/smoke-job/0" contain slashes
# which are not permitted.
_K8S_LABEL_MAX_LEN = 63
_POD_NOT_FOUND_RETRY_COUNT = 3
_POD_NOT_FOUND_RETRY_WINDOW = Duration.from_seconds(15.0)


def _sanitize_label_value(value: str) -> str:
    """Sanitize a string for use as a Kubernetes label value.

    Replaces invalid characters with dots, strips leading/trailing non-alphanumeric
    characters, and truncates to 63 characters.
    """
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


def _build_task_script(config: ContainerConfig) -> str:
    """Build a shell script that executes setup_commands and run_command."""
    lines = [
        "set -e",
        "ulimit -c 0",
        'echo "iris-task starting (git_hash=${IRIS_GIT_HASH:-unknown})"',
        f"mkdir -p {shlex.quote(config.workdir)}",
        f"cd {shlex.quote(config.workdir)}",
    ]

    lines.extend(config.entrypoint.setup_commands)
    run_cmd = " ".join(shlex.quote(arg) for arg in config.entrypoint.run_command.argv)
    lines.append(f"exec {run_cmd}")
    return "\n".join(lines)


def _build_stage_init_script() -> str:
    """Build init-container script that stages bundle ID and workdir files.

    Reads the script from kubernetes_bundle_fetch.py so it can be tested
    and syntax-checked independently.
    """
    script_path = Path(__file__).with_name("kubernetes_bundle_fetch.py")
    script_content = script_path.read_text()
    return f"python - <<'IRIS_STAGE_FILES'\n{script_content}IRIS_STAGE_FILES"


def _build_gpu_resources(config: ContainerConfig) -> dict[str, str]:
    """Extract GPU resource limits from the task's requested resources."""
    limits: dict[str, str] = {}
    if config.resources and config.resources.HasField("device"):
        gpu_count = get_gpu_count(config.resources.device)
        if gpu_count > 0:
            limits["nvidia.com/gpu"] = str(gpu_count)
    return limits


def _build_tolerations(config: ContainerConfig) -> list[dict]:
    """Build tolerations for GPU node taints.

    GPU nodes may carry ``nvidia.com/gpu:NoSchedule`` and CoreWeave nodes may
    additionally carry ``qos.coreweave.cloud/interruptable:NoExecute``. The CW
    toleration is harmless on non-CoreWeave clusters.
    """
    tolerations: list[dict] = []
    if config.resources and config.resources.HasField("device") and get_gpu_count(config.resources.device) > 0:
        tolerations.append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})
        tolerations.append({"key": "qos.coreweave.cloud/interruptable", "operator": "Exists", "effect": "NoExecute"})
    return tolerations


def _kubectl_log_line_to_log_line(kll: KubectlLogLine) -> LogLine:
    return LogLine(timestamp=kll.timestamp, source=kll.stream, data=kll.data)


class KubernetesLogReader:
    """Incremental log reader for a Kubernetes pod container using byte-offset deduplication."""

    def __init__(self, kubectl: Kubectl, pod_name: str, container: str) -> None:
        self._kubectl = kubectl
        self._pod_name = pod_name
        self._container = container
        self._byte_offset: int = 0

    def read(self) -> list[LogLine]:
        result = self._kubectl.stream_logs(self._pod_name, container=self._container, byte_offset=self._byte_offset)
        self._byte_offset = result.byte_offset
        return [_kubectl_log_line_to_log_line(kll) for kll in result.lines]

    def read_all(self) -> list[LogLine]:
        result = self._kubectl.stream_logs(self._pod_name, container=self._container, byte_offset=0)
        return [_kubectl_log_line_to_log_line(kll) for kll in result.lines]


@dataclass
class KubernetesContainerHandle:
    """ContainerHandle backed by a single Kubernetes Pod."""

    config: ContainerConfig
    kubectl: Kubectl
    cache_dir: Path | None = None
    service_account_name: str = ""
    s3_secret_name: str = ""
    s3_endpoint_url: str = ""
    fsspec_s3_conf: str = ""
    owner_pod_name: str = ""
    owner_pod_uid: str = ""
    _pod_name: str = field(default="", repr=False)
    _workdir_configmap_name: str | None = field(default=None, repr=False)
    _started: bool = field(default=False, repr=False)
    _pod_not_found_count: int = field(default=0, repr=False)
    _pod_not_found_deadline: Deadline | None = field(default=None, repr=False)

    @property
    def container_id(self) -> str | None:
        return self._pod_name or None

    def build(self) -> list[LogLine]:
        """No-op build phase.

        For K8s Pods, setup commands run in the same container command script.
        """
        return []

    def run(self) -> None:
        """Create and start the task Pod."""
        if self._started:
            return

        self._pod_name = f"iris-task-{uuid.uuid4().hex[:12]}"
        self._workdir_configmap_name = None
        task_script = _build_task_script(self.config)

        env_list: list[dict] = [
            {"name": k, "value": v} for k, v in self.config.env.items() if k != "IRIS_ADVERTISE_HOST"
        ]

        # The worker sets IRIS_ADVERTISE_HOST to its own host IP, but in the
        # kubernetes runtime task pods can be scheduled on any node. Use the
        # downward API to inject the actual node IP where the pod is running.
        env_list.append({"name": "IRIS_ADVERTISE_HOST", "valueFrom": {"fieldRef": {"fieldPath": "status.hostIP"}}})

        # Pull S3 credentials from the K8s Secret so task containers can
        # access S3 object storage (e.g. for training data).
        if self.s3_secret_name:
            for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
                env_list.append(
                    {
                        "name": key,
                        "valueFrom": {"secretKeyRef": {"name": self.s3_secret_name, "key": key}},
                    }
                )
        if self.s3_endpoint_url and not any(e.get("name") == "AWS_ENDPOINT_URL" for e in env_list):
            env_list.append({"name": "AWS_ENDPOINT_URL", "value": self.s3_endpoint_url})
        if self.fsspec_s3_conf and not any(e.get("name") == "FSSPEC_S3" for e in env_list):
            env_list.append({"name": "FSSPEC_S3", "value": self.fsspec_s3_conf})

        container: dict[str, object] = {
            "name": "task",
            "image": self.config.image,
            "imagePullPolicy": "Always",
            "command": ["bash", "-lc", task_script],
            "workingDir": self.config.workdir,
            "env": env_list,
        }

        mounts = []
        volumes = []
        for i, mount in enumerate(self.config.mounts):
            volume_name = f"mount-{i}"
            if mount.kind in (MountKind.WORKDIR, MountKind.TMPFS):
                empty_dir_spec: dict[str, str] = {}
                if mount.size_bytes > 0:
                    empty_dir_spec["sizeLimit"] = str(mount.size_bytes)
                mounts.append(
                    {
                        "name": volume_name,
                        "mountPath": mount.container_path,
                        "readOnly": mount.read_only,
                    }
                )
                volumes.append(
                    {
                        "name": volume_name,
                        "emptyDir": empty_dir_spec,
                    }
                )
            elif mount.kind == MountKind.CACHE:
                if self.cache_dir is not None:
                    host_path = str(self.cache_dir / mount.container_path.strip("/").replace("/", "-"))
                else:
                    host_path = mount.container_path
                mounts.append(
                    {
                        "name": volume_name,
                        "mountPath": mount.container_path,
                        "readOnly": mount.read_only,
                    }
                )
                volumes.append(
                    {
                        "name": volume_name,
                        "hostPath": {"path": host_path, "type": "DirectoryOrCreate"},
                    }
                )

        workdir_files = dict(self.config.entrypoint.workdir_files)
        if workdir_files:
            self._workdir_configmap_name = f"{self._pod_name}-workdir-files"
            binary_data: dict[str, str] = {}
            config_items: list[dict[str, str]] = []
            for i, (path, data) in enumerate(workdir_files.items()):
                normalized = normalize_workdir_relative_path(path)
                key = f"f{i:04d}"
                binary_data[key] = base64.b64encode(data).decode("ascii")
                config_items.append({"key": key, "path": normalized})

            config_map_manifest = {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": self._workdir_configmap_name,
                    "namespace": self.kubectl.namespace,
                    "labels": {
                        "iris.managed": "true",
                        "iris.runtime": "kubernetes",
                        "iris.task_id": _sanitize_label_value(self.config.task_id or "unknown"),
                    },
                },
                "binaryData": binary_data,
            }
            self.kubectl.apply_json(config_map_manifest)

            volumes.append(
                {
                    "name": "workdir-files",
                    "configMap": {"name": self._workdir_configmap_name, "items": config_items},
                }
            )

        container["volumeMounts"] = mounts

        # Find the workdir volume name for the init container
        workdir_volume_name = None
        for i, mount in enumerate(self.config.mounts):
            if mount.kind == MountKind.WORKDIR:
                workdir_volume_name = f"mount-{i}"
                break

        init_containers: list[dict[str, object]] = []
        bundle_id = self.config.env.get("IRIS_BUNDLE_ID", "")
        if bundle_id or workdir_files:
            stage_mounts = [{"name": workdir_volume_name, "mountPath": self.config.workdir, "readOnly": False}]
            stage_env = [
                {"name": "IRIS_WORKDIR", "value": self.config.workdir},
                {"name": "IRIS_BUNDLE_ID", "value": bundle_id},
                {"name": "IRIS_CONTROLLER_URL", "value": self.config.env.get("IRIS_CONTROLLER_URL", "")},
            ]
            if workdir_files:
                stage_mounts.append(
                    {"name": "workdir-files", "mountPath": "/iris/staged-workdir-files", "readOnly": True}
                )
                stage_env.append({"name": "IRIS_WORKDIR_FILES_SRC", "value": "/iris/staged-workdir-files"})
            init_containers.append(
                {
                    "name": "stage-workdir",
                    "image": self.config.image,
                    "imagePullPolicy": "Always",
                    "command": ["bash", "-lc", _build_stage_init_script()],
                    "env": stage_env,
                    "volumeMounts": stage_mounts,
                }
            )

        resources: dict[str, dict[str, str]] = {}
        cpu = self.config.get_cpu_millicores()
        if cpu:
            resources.setdefault("limits", {})["cpu"] = f"{cpu}m"
        memory_mb = self.config.get_memory_mb()
        if memory_mb:
            resources.setdefault("limits", {})["memory"] = f"{memory_mb}Mi"
        disk_bytes = self.config.get_disk_bytes()
        if disk_bytes:
            disk_gi = max(1, disk_bytes // (1024 * 1024 * 1024))
            resources.setdefault("requests", {})["ephemeral-storage"] = f"{disk_gi}Gi"
            resources.setdefault("limits", {})["ephemeral-storage"] = f"{disk_gi}Gi"
        gpu_limits = _build_gpu_resources(self.config)
        if gpu_limits:
            resources.setdefault("limits", {}).update(gpu_limits)
        if resources:
            container["resources"] = resources

        container["securityContext"] = {"capabilities": {"add": ["SYS_PTRACE"]}}

        spec: dict[str, object] = {
            "restartPolicy": "Never",
            "containers": [container],
            "volumes": volumes,
        }
        if init_containers:
            spec["initContainers"] = init_containers

        if self.config.network_mode == "host":
            spec["hostNetwork"] = True
            spec["dnsPolicy"] = "ClusterFirstWithHostNet"

        if self.service_account_name:
            spec["serviceAccountName"] = self.service_account_name

        tolerations = _build_tolerations(self.config)
        if tolerations:
            spec["tolerations"] = tolerations

        metadata: dict[str, object] = {
            "name": self._pod_name,
            "namespace": self.kubectl.namespace,
            "labels": {
                "iris.managed": "true",
                "iris.runtime": "kubernetes",
                "iris.task_id": _sanitize_label_value(self.config.task_id or "unknown"),
            },
        }
        if self.owner_pod_name and self.owner_pod_uid:
            metadata["ownerReferences"] = [
                {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "name": self.owner_pod_name,
                    "uid": self.owner_pod_uid,
                    "blockOwnerDeletion": False,
                }
            ]

        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": metadata,
            "spec": spec,
        }

        try:
            self.kubectl.apply_json(manifest)
        except Exception:
            if self._workdir_configmap_name:
                self.kubectl.delete("configmap", self._workdir_configmap_name)
                self._workdir_configmap_name = None
            raise
        self._started = True
        workdir_mount = next((m for m in self.config.mounts if m.kind == MountKind.WORKDIR), None)
        workdir_size = workdir_mount.size_bytes if workdir_mount else 0
        logger.info(
            "Started Kubernetes task pod %s (task_id=%s, gpus=%s, disk=%s, hostNetwork=%s)",
            self._pod_name,
            self.config.task_id,
            gpu_limits.get("nvidia.com/gpu", "0"),
            f"{workdir_size // (1024 * 1024 * 1024)}Gi" if workdir_size else "default",
            self.config.network_mode == "host",
        )

    def stop(self, force: bool = False) -> None:
        if not self._pod_name:
            return
        self.kubectl.delete("pod", self._pod_name, force=force)
        if self._workdir_configmap_name is not None:
            self.kubectl.delete("configmap", self._workdir_configmap_name)
            self._workdir_configmap_name = None

    def status(self) -> ContainerStatus:
        if not self._pod_name:
            return ContainerStatus(phase=ContainerPhase.STOPPED, error="Pod not started")

        pod = self.kubectl.get_json("pod", self._pod_name)
        if pod is None:
            self._pod_not_found_count += 1
            if self._pod_not_found_deadline is None:
                self._pod_not_found_deadline = Deadline.from_now(_POD_NOT_FOUND_RETRY_WINDOW)

            within_retry_window = self._pod_not_found_deadline is not None and not self._pod_not_found_deadline.expired()
            if self._pod_not_found_count < _POD_NOT_FOUND_RETRY_COUNT and within_retry_window:
                logger.warning(
                    "Pod lookup miss for %s (%d/%d) within retry window; treating as transient",
                    self._pod_name,
                    self._pod_not_found_count,
                    _POD_NOT_FOUND_RETRY_COUNT,
                )
                return ContainerStatus(phase=ContainerPhase.PENDING)

            attempt = self.config.attempt_id if self.config.attempt_id is not None else -1
            return ContainerStatus(
                phase=ContainerPhase.STOPPED,
                error=(
                    "Task pod not found after retry window: "
                    f"name={self._pod_name}, namespace={self.kubectl.namespace}, "
                    f"task_id={self.config.task_id or 'unknown'}, attempt_id={attempt}, "
                    f"observations={self._pod_not_found_count}"
                ),
                error_kind=ContainerErrorKind.INFRA_NOT_FOUND,
            )

        # Pod is visible again; clear transient miss tracking.
        self._pod_not_found_count = 0
        self._pod_not_found_deadline = None

        phase = pod.get("status", {}).get("phase", "")
        if phase == "Pending":
            return ContainerStatus(phase=ContainerPhase.PENDING)
        if phase == "Running":
            return ContainerStatus(phase=ContainerPhase.RUNNING)

        # Check init containers first — if staging (bundle fetch, workdir
        # setup) failed, the main container never started and the real error
        # lives in initContainerStatuses.
        init_statuses = pod.get("status", {}).get("initContainerStatuses", [])
        for init_st in init_statuses:
            init_terminated = init_st.get("state", {}).get("terminated", {})
            init_exit = init_terminated.get("exitCode")
            if init_exit is not None and init_exit != 0:
                init_reason = init_terminated.get("reason", "")
                init_message = init_terminated.get("message", "")
                init_name = init_st.get("name", "init")
                error_detail = init_message or init_reason or f"init container {init_name} failed"
                return ContainerStatus(
                    phase=ContainerPhase.STOPPED,
                    exit_code=init_exit,
                    error=f"Init container '{init_name}' failed: {error_detail}",
                    oom_killed=init_reason == "OOMKilled",
                )

        statuses = pod.get("status", {}).get("containerStatuses", [])
        terminated = {}
        if statuses:
            terminated = statuses[0].get("state", {}).get("terminated", {})

        exit_code = terminated.get("exitCode")
        reason = terminated.get("reason", "")
        message = terminated.get("message", "")
        # "Completed" is K8s's normal termination reason for containers that
        # exit cleanly. It is not an error.
        if reason == "Completed":
            error = message or None
        else:
            error = message or reason or None
        oom_killed = reason == "OOMKilled"
        return ContainerStatus(
            phase=ContainerPhase.STOPPED,
            exit_code=exit_code if isinstance(exit_code, int) else 1,
            error=error,
            oom_killed=oom_killed,
        )

    def log_reader(self) -> KubernetesLogReader:
        return KubernetesLogReader(self.kubectl, self._pod_name, "task")

    def stats(self) -> ContainerStats:
        if not self._pod_name:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)
        metrics = self.kubectl.top_pod(self._pod_name)
        if metrics is None:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)
        cpu_mc, mem_bytes = metrics
        return ContainerStats(
            memory_mb=mem_bytes // (1024 * 1024),
            cpu_percent=cpu_mc // 10,
            process_count=1,
            available=True,
        )

    def disk_usage_mb(self) -> int:
        """K8s workdir lives inside the pod; disk usage isn't observable from the worker."""
        return 0

    def profile(self, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes:
        """Profile the running process using py-spy (CPU), memray (memory), or thread dump."""
        if not self._pod_name:
            raise RuntimeError("Cannot profile: no running pod")

        profile_id = uuid.uuid4().hex[:8]

        if profile_type.HasField("threads"):
            return self._profile_threads(include_locals=profile_type.threads.locals)
        elif profile_type.HasField("cpu"):
            return self._profile_cpu(duration_seconds, profile_type.cpu, profile_id)
        elif profile_type.HasField("memory"):
            return self._profile_memory(duration_seconds, profile_type.memory, profile_id)
        else:
            raise RuntimeError("ProfileType must specify cpu, memory, or threads profiler")

    def _profile_threads(self, *, include_locals: bool = False) -> bytes:
        """Collect thread stacks from the pod using py-spy dump."""
        cmd = build_pyspy_dump_cmd(pid="1", py_spy_bin="py-spy", include_locals=include_locals)
        result = self.kubectl.exec(self._pod_name, self._wrap_in_venv_shell(cmd), container="task", timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"py-spy dump failed: {result.stderr}")
        return result.stdout.encode("utf-8")

    def _wrap_in_venv_shell(self, cmd: list[str]) -> list[str]:
        """Wrap a command to run inside the task venv via a login shell.

        kubectl exec resolves the binary at the OCI layer before entering the
        container mount namespace. With --link-mode symlink, venv binaries
        (py-spy, memray) are symlinks into the uv cache which may not resolve
        at the OCI stat() call. Running through bash with an explicit activate
        lets the shell resolve the binary via PATH instead.
        """
        escaped = " ".join(shlex.quote(arg) for arg in cmd)
        return ["bash", "-lc", f"source /app/.venv/bin/activate && {escaped}"]

    def _profile_cpu(self, duration_seconds: int, cpu_config: cluster_pb2.CpuProfile, profile_id: str) -> bytes:
        spec = resolve_cpu_spec(cpu_config, duration_seconds, pid="1")
        output_path = f"/tmp/profile-cpu-{profile_id}.{spec.ext}"
        cmd = build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path=output_path)

        logger.info(
            "CPU profiling pod %s for %ds (format=%s, rate=%dHz)",
            self._pod_name,
            duration_seconds,
            spec.py_spy_format,
            spec.rate_hz,
        )
        try:
            result = self.kubectl.exec(
                self._pod_name, self._wrap_in_venv_shell(cmd), container="task", timeout=duration_seconds + 30
            )
            if result.returncode != 0:
                raise RuntimeError(f"py-spy failed: {result.stderr}")
            return self.kubectl.read_file(self._pod_name, output_path, container="task")
        finally:
            self.kubectl.rm_files(self._pod_name, [output_path], container="task")

    def _profile_memory(self, duration_seconds: int, memory_config: cluster_pb2.MemoryProfile, profile_id: str) -> bytes:
        spec = resolve_memory_spec(memory_config, duration_seconds, pid="1")
        trace_path = f"/tmp/memray-trace-{profile_id}.bin"
        output_path = f"/tmp/memray-output-{profile_id}.{spec.ext}"

        attach_cmd = build_memray_attach_cmd(spec, "memray", trace_path)
        transform_cmd = build_memray_transform_cmd(spec, "memray", trace_path, output_path)

        logger.info(
            "Memory profiling pod %s for %ds (format=%s, leaks=%s)",
            self._pod_name,
            duration_seconds,
            spec.reporter,
            spec.leaks,
        )
        try:
            result = self.kubectl.exec(
                self._pod_name, self._wrap_in_venv_shell(attach_cmd), container="task", timeout=duration_seconds + 10
            )
            if result.returncode != 0:
                raise RuntimeError(f"memray attach failed: {result.stderr}")

            result = self.kubectl.exec(
                self._pod_name, self._wrap_in_venv_shell(transform_cmd), container="task", timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(f"memray {spec.reporter} failed: {result.stderr}")

            if spec.output_is_file:
                return self.kubectl.read_file(self._pod_name, output_path, container="task")
            else:
                return result.stdout.encode("utf-8")
        finally:
            self.kubectl.rm_files(self._pod_name, [trace_path, output_path], container="task")

    def cleanup(self) -> None:
        self.stop(force=True)


class KubernetesRuntime:
    """ContainerRuntime implementation backed by Kubernetes Pods.

    When running on CoreWeave, GPU and RDMA resources are claimed by task Pods
    (not by the worker Pod). The worker Pod should set ``runtime: kubernetes``
    in bootstrap config so the platform skips GPU resource requests on it.
    """

    def __init__(
        self,
        *,
        namespace: str | None = None,
        service_account_name: str | None = None,
        s3_secret_name: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        resolved_namespace = namespace or os.environ.get("IRIS_POD_NAMESPACE") or "iris"
        self._service_account_name = service_account_name or os.environ.get("IRIS_SERVICE_ACCOUNT_NAME") or ""
        self._s3_secret_name = s3_secret_name or os.environ.get("IRIS_S3_SECRET_NAME") or ""
        self._s3_endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "")
        self._fsspec_s3_conf = os.environ.get("FSSPEC_S3", "")
        self._owner_pod_name = os.environ.get("IRIS_POD_NAME", "")
        self._owner_pod_uid = os.environ.get("IRIS_POD_UID", "")
        # TODO(marin): ownerReferences only trigger GC when the worker Pod object
        # is deleted. If worker containers crash-loop in-place, task Pods remain.
        # Consider restartPolicy=Never for worker Pods or explicit stale-task cleanup.
        self._kubectl = Kubectl(namespace=resolved_namespace)
        self._cache_dir = cache_dir
        self._handles: list[KubernetesContainerHandle] = []

    def create_container(self, config: ContainerConfig) -> KubernetesContainerHandle:
        handle = KubernetesContainerHandle(
            config=config,
            kubectl=self._kubectl,
            cache_dir=self._cache_dir,
            service_account_name=self._service_account_name,
            s3_secret_name=self._s3_secret_name,
            s3_endpoint_url=self._s3_endpoint_url,
            fsspec_s3_conf=self._fsspec_s3_conf,
            owner_pod_name=self._owner_pod_name,
            owner_pod_uid=self._owner_pod_uid,
        )
        self._handles.append(handle)
        return handle

    def prepare_workdir(self, workdir: Path, disk_bytes: int) -> None:
        pass

    def stage_bundle(
        self,
        *,
        bundle_id: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        bundle_store: BundleStore,
    ) -> None:
        """No-op: Kubernetes task Pods materialize bundle/workdir in-pod."""
        del bundle_id, workdir, workdir_files, bundle_store

    def list_containers(self) -> list[KubernetesContainerHandle]:
        return list(self._handles)

    def remove_all_iris_containers(self) -> int:
        pods = self._kubectl.list_json("pods", labels={"iris.managed": "true"})
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            if pod_name:
                self._kubectl.delete("pod", pod_name, force=True)
        return len(pods)

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()
