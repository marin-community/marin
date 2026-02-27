# Copyright 2025 The Marin Authors
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from iris.cluster.k8s.kubectl import Kubectl, KubectlLogLine
from iris.cluster.types import get_gpu_count
from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.cluster.runtime.types import (
    ContainerConfig,
    ContainerErrorKind,
    ContainerPhase,
    ContainerStats,
    ContainerStatus,
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
    """Build a shell script that prepares workdir, then runs the task."""
    lines = [
        "set -e",
        'echo "iris-task starting (git_hash=${IRIS_GIT_HASH:-unknown})"',
        f"mkdir -p {shlex.quote(config.workdir)}",
        f"cd {shlex.quote(config.workdir)}",
    ]

    # Kubernetes runtime uses an emptyDir for /app and materializes bundle in-Pod.
    # S3 config (endpoint, addressing style) is passed via the FSSPEC_S3 env var
    # injected by the platform; credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
    # fsspec reads FSSPEC_S3 automatically at import time via fsspec.config.set_conf_env().
    lines.extend(
        [
            'if [ -n "${IRIS_BUNDLE_GCS_PATH:-}" ]; then',
            "python - <<'IRIS_BUNDLE_EOF'",
            "import os, tempfile, zipfile, fsspec",
            "bundle = os.environ['IRIS_BUNDLE_GCS_PATH']",
            "with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:",
            "    tmp_path = tmp.name",
            "try:",
            "    with fsspec.open(bundle, 'rb') as src, open(tmp_path, 'wb') as dst:",
            "        dst.write(src.read())",
            "    with zipfile.ZipFile(tmp_path) as zf:",
            "        zf.extractall(os.getcwd())",
            "finally:",
            "    os.unlink(tmp_path)",
            "IRIS_BUNDLE_EOF",
            "fi",
        ]
    )

    for i, (name, data) in enumerate(config.entrypoint.workdir_files.items()):
        encoded = base64.b64encode(data).decode("ascii")
        quoted_name = shlex.quote(name)
        lines.append(f"mkdir -p $(dirname {quoted_name})")
        lines.append(f"base64 -d > {quoted_name} <<'IRIS_WORKDIR_FILE_{i}'")
        lines.append(encoded)
        lines.append(f"IRIS_WORKDIR_FILE_{i}")

    lines.extend(config.entrypoint.setup_commands)
    run_cmd = " ".join(shlex.quote(arg) for arg in config.entrypoint.run_command.argv)
    lines.append(f"exec {run_cmd}")
    return "\n".join(lines)


def _build_gpu_resources(config: ContainerConfig) -> dict[str, str]:
    """Extract GPU resource limits from the task's requested resources."""
    limits: dict[str, str] = {}
    if config.resources and config.resources.HasField("device"):
        gpu_count = get_gpu_count(config.resources.device)
        if gpu_count > 0:
            limits["nvidia.com/gpu"] = str(gpu_count)
    return limits


def _build_tolerations(config: ContainerConfig) -> list[dict]:
    """Build tolerations for GPU/RDMA node taints.

    CoreWeave GPU nodes carry ``nvidia.com/gpu`` taints. Tolerations ensure
    task Pods are eligible for those nodes.
    """
    tolerations: list[dict] = []
    if config.resources and config.resources.HasField("device") and get_gpu_count(config.resources.device) > 0:
        tolerations.append({"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"})
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
    service_account_name: str = ""
    s3_secret_name: str = ""
    s3_endpoint_url: str = ""
    fsspec_s3_conf: str = ""
    owner_pod_name: str = ""
    owner_pod_uid: str = ""
    _pod_name: str = field(default="", repr=False)
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
        for i, (host_path, container_path, mode) in enumerate(self.config.mounts):
            # /app is pod-local emptyDir so tasks can run on any node.
            if container_path == self.config.workdir:
                continue

            volume_name = f"mount-{i}"
            mounts.append(
                {
                    "name": volume_name,
                    "mountPath": container_path,
                    "readOnly": "ro" in mode,
                }
            )
            volumes.append(
                {
                    "name": volume_name,
                    "hostPath": {"path": host_path, "type": "DirectoryOrCreate"},
                }
            )

        mounts.append({"name": "workdir", "mountPath": self.config.workdir, "readOnly": False})
        volumes.append({"name": "workdir", "emptyDir": {}})

        container["volumeMounts"] = mounts

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

        self.kubectl.apply_json(manifest)
        self._started = True
        logger.info(
            "Started Kubernetes task pod %s (task_id=%s, gpus=%s, disk=%s, hostNetwork=%s)",
            self._pod_name,
            self.config.task_id,
            gpu_limits.get("nvidia.com/gpu", "0"),
            f"{disk_gi}Gi" if disk_bytes else "default",
            self.config.network_mode == "host",
        )

    def stop(self, force: bool = False) -> None:
        if not self._pod_name:
            return
        self.kubectl.delete("pod", self._pod_name, force=force)

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

    def profile(self, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes:
        """Profile the running process using py-spy (CPU) or memray (memory)."""
        if not self._pod_name:
            raise RuntimeError("Cannot profile: no running pod")

        profile_id = uuid.uuid4().hex[:8]

        if profile_type.HasField("cpu"):
            return self._profile_cpu(duration_seconds, profile_type.cpu, profile_id)
        elif profile_type.HasField("memory"):
            return self._profile_memory(duration_seconds, profile_type.memory, profile_id)
        else:
            raise RuntimeError("ProfileType must specify either cpu or memory profiler")

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
                self._pod_name, self._wrap_in_venv_shell(cmd), container="task", timeout=duration_seconds + 5
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
        self._handles: list[KubernetesContainerHandle] = []

    def create_container(self, config: ContainerConfig) -> KubernetesContainerHandle:
        handle = KubernetesContainerHandle(
            config=config,
            kubectl=self._kubectl,
            service_account_name=self._service_account_name,
            s3_secret_name=self._s3_secret_name,
            s3_endpoint_url=self._s3_endpoint_url,
            fsspec_s3_conf=self._fsspec_s3_conf,
            owner_pod_name=self._owner_pod_name,
            owner_pod_uid=self._owner_pod_uid,
        )
        self._handles.append(handle)
        return handle

    def stage_bundle(
        self,
        *,
        bundle_gcs_path: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        fetch_bundle: Callable[[str], Path],
    ) -> None:
        """No-op: Kubernetes task Pods materialize bundle/workdir in-pod."""
        del bundle_gcs_path, workdir, workdir_files, fetch_bundle

    def list_containers(self) -> list[KubernetesContainerHandle]:
        return list(self._handles)

    def remove_all_iris_containers(self) -> int:
        pods = self._kubectl.list_json("pods", labels={"iris.managed": "true"})
        for pod in pods:
            pod_name = pod.get("metadata", {}).get("name", "")
            if pod_name:
                self._kubectl.delete("pod", pod_name, force=True)
        return len(pods)

    def remove(self, container_id: str) -> None:
        """No-op: pod lifecycle is managed by ContainerHandle.stop() / delete()."""

    def cleanup(self) -> None:
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()
