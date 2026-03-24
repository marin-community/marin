# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sService protocol and CloudK8sService (kubectl-backed) implementation."""

from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from iris.cluster.providers.k8s.types import ExecResult, KubectlError, KubectlLogLine, KubectlLogResult
from iris.cluster.providers.k8s.types import parse_k8s_cpu as _parse_k8s_cpu
from iris.cluster.providers.k8s.types import parse_k8s_memory as _parse_k8s_memory
from iris.cluster.providers.types import find_free_port
from iris.time_utils import Deadline, ExponentialBackoff

logger = logging.getLogger(__name__)

# Default timeout for kubectl commands (seconds)
DEFAULT_TIMEOUT: float = 60.0


@runtime_checkable
class K8sService(Protocol):
    """Protocol for Kubernetes operations.

    Consumers that only need high-level Kubernetes operations should depend on
    this protocol rather than the concrete CloudK8sService class, enabling test
    doubles that don't shell out to kubectl.
    """

    @property
    def namespace(self) -> str: ...

    def apply_json(self, manifest: dict) -> None: ...

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None: ...

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]: ...

    def delete(
        self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True
    ) -> None: ...

    def delete_by_labels(
        self,
        resource: str,
        labels: dict[str, str],
        *,
        cluster_scoped: bool = False,
        wait: bool = False,
    ) -> int:
        """Delete all resources matching label selector. Returns count deleted."""
        ...

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str: ...

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        byte_offset: int = 0,
    ) -> KubectlLogResult: ...

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult: ...

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None: ...

    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None: ...

    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None: ...

    def get_events(
        self,
        field_selector: str | None = None,
    ) -> list[dict]: ...

    def top_pod(self, pod_name: str) -> tuple[int, int] | None: ...

    def read_file(
        self,
        pod_name: str,
        path: str,
        *,
        container: str | None = None,
    ) -> bytes: ...

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None: ...

    def port_forward(
        self,
        service_name: str,
        remote_port: int,
        local_port: int | None = None,
        timeout: float = 90.0,
    ) -> AbstractContextManager[str]:
        """Port-forward to a K8s Service, yielding the local URL.

        The returned context manager keeps the tunnel alive for its duration.
        Implementations may reconnect transparently on transient failures.

        In DRY_RUN/LOCAL mode, returns a fake URL without spawning subprocesses.
        In CLOUD mode, runs ``kubectl port-forward`` with reconnection.
        """
        ...


# ---------------------------------------------------------------------------
# CloudK8sService — kubectl-backed implementation
# ---------------------------------------------------------------------------


@dataclass
class CloudK8sService:
    """K8sService backed by the kubectl CLI. Used in CLOUD mode.

    Encapsulates command prefix construction (including optional --kubeconfig),
    namespace injection, JSON parsing, and error handling. All operations use
    subprocess with a configurable timeout.
    """

    namespace: str
    kubeconfig_path: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _prefix: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kubeconfig_path:
            self.kubeconfig_path = os.path.expanduser(self.kubeconfig_path)
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        self._prefix = cmd

    def _run(
        self,
        args: list[str],
        *,
        namespaced: bool = False,
        timeout: float | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a kubectl command with consistent timeout and error capture."""
        effective_timeout = timeout if timeout is not None else self.timeout
        cmd = list(self._prefix)
        if namespaced:
            cmd.extend(["-n", self.namespace])
        cmd.extend(args)
        if stdin:
            logger.info("kubectl: %s\n  stdin=%s", " ".join(cmd), stdin[:2000])
        else:
            logger.info("kubectl: %s", " ".join(cmd))
        t0 = time.monotonic()
        result = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=effective_timeout)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if result.returncode != 0:
            logger.info("kubectl exit %d: %dms stderr=%s", result.returncode, elapsed_ms, result.stderr.strip()[:500])
        elif elapsed_ms > 2000:
            logger.warning("kubectl slow: %dms cmd=%s", elapsed_ms, " ".join(args))
        return result

    def apply_json(self, manifest: dict) -> None:
        """Apply a Kubernetes manifest dict via ``kubectl apply -f -``."""
        result = self._run(["apply", "-f", "-"], stdin=json.dumps(manifest))
        if result.returncode != 0:
            raise KubectlError(f"kubectl apply failed: {result.stderr.strip()}")

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None:
        """Get a Kubernetes resource as a parsed dict. Returns None if not found."""
        result = self._run(
            ["get", resource, name, "-o", "json"],
            namespaced=not cluster_scoped,
        )
        if result.returncode != 0:
            if "not found" in result.stderr.lower() or "NotFound" in result.stderr:
                return None
            raise KubectlError(f"kubectl get {resource}/{name} failed: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]:
        """List Kubernetes resources, optionally filtered by labels."""
        args = ["get", resource, "-o", "json"]
        if labels:
            selector = ",".join(f"{k}={v}" for k, v in labels.items())
            args.extend(["-l", selector])
        result = self._run(args, namespaced=not cluster_scoped)
        if result.returncode != 0:
            raise KubectlError(f"kubectl get {resource} failed: {result.stderr.strip()}")
        data = json.loads(result.stdout)
        return data.get("items", [])

    def delete(
        self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True
    ) -> None:
        """Delete a Kubernetes resource, ignoring NotFound errors."""
        args = ["delete", resource, name, "--ignore-not-found"]
        if force:
            args.extend(["--grace-period=0", "--force"])
        if not wait:
            args.append("--wait=false")
        result = self._run(
            args,
            namespaced=not cluster_scoped,
        )
        if result.returncode != 0:
            raise KubectlError(f"kubectl delete {resource}/{name} failed: {result.stderr.strip()}")

    def delete_by_labels(
        self,
        resource: str,
        labels: dict[str, str],
        *,
        cluster_scoped: bool = False,
        wait: bool = False,
    ) -> int:
        """Delete all resources matching label selector in a single kubectl call."""
        selector = ",".join(f"{k}={v}" for k, v in labels.items())
        args = ["delete", resource, "-l", selector, "--ignore-not-found"]
        if not wait:
            args.append("--wait=false")
        result = self._run(args, namespaced=not cluster_scoped)
        if result.returncode != 0:
            raise KubectlError(f"kubectl delete {resource} -l {selector} failed: {result.stderr.strip()}")
        # kubectl prints one "deleted" line per resource
        return result.stdout.strip().count("deleted")

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None:
        """Set the container image on a resource via ``kubectl set image``."""
        args = ["set", "image", f"{resource}/{name}", f"{container}={image}"]
        result = self._run(args, namespaced=namespaced)
        if result.returncode != 0:
            raise KubectlError(f"kubectl set image failed: {result.stderr.strip()}")

    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None:
        """Restart a rollout via ``kubectl rollout restart``."""
        args = ["rollout", "restart", f"{resource}/{name}"]
        result = self._run(args, namespaced=namespaced)
        if result.returncode != 0:
            raise KubectlError(f"kubectl rollout restart failed: {result.stderr.strip()}")

    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None:
        """Wait for a rollout to complete via ``kubectl rollout status``."""
        args = ["rollout", "status", f"{resource}/{name}", f"--timeout={int(timeout)}s"]
        result = self._run(args, namespaced=namespaced, timeout=timeout + 30)
        if result.returncode != 0:
            raise KubectlError(f"kubectl rollout status failed: {result.stderr.strip()}")

    def get_events(
        self,
        field_selector: str | None = None,
    ) -> list[dict]:
        """Get Kubernetes events, optionally filtered by field selector."""
        args = ["get", "events", "-o", "json"]
        if field_selector:
            args.extend(["--field-selector", field_selector])
        result = self._run(args, namespaced=True)
        if result.returncode != 0:
            logger.warning("kubectl get events failed: %s", result.stderr.strip()[:200])
            return []
        data = json.loads(result.stdout)
        return data.get("items", [])

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        """Fetch logs from a Pod container."""
        args = ["logs", pod_name]
        if container:
            args.extend(["-c", container])
        if previous:
            args.append("--previous")
        args.extend([f"--tail={tail}"])
        result = self._run(args, namespaced=True)
        if result.returncode != 0:
            return ""
        return result.stdout

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run a command inside a Pod container via ``kubectl exec``."""
        args = ["exec", pod_name]
        if container:
            args.extend(["-c", container])
        args.extend(["--", *cmd])
        result = self._run(args, namespaced=True, timeout=timeout)
        return ExecResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)

    def read_file(
        self,
        pod_name: str,
        path: str,
        *,
        container: str | None = None,
    ) -> bytes:
        """Read a file from inside a Pod container."""
        result = self.exec(pod_name, ["cat", path], container=container, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr}")
        return result.stdout.encode("utf-8")

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None:
        """Remove files inside a Pod container. Ignores missing files."""
        self.exec(pod_name, ["rm", "-f", *paths], container=container, timeout=10)

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        byte_offset: int = 0,
    ) -> KubectlLogResult:
        """Fetch new log lines from a pod using byte-offset deduplication."""
        args = ["logs", pod_name, "--timestamps=true"]
        if container:
            args.extend(["-c", container])
        result = self._run(args, namespaced=True)
        if result.returncode != 0:
            return KubectlLogResult(lines=[], byte_offset=byte_offset)

        raw_bytes = result.stdout.encode("utf-8")
        if len(raw_bytes) <= byte_offset:
            return KubectlLogResult(lines=[], byte_offset=byte_offset)

        new_content = raw_bytes[byte_offset:].decode("utf-8", errors="replace")
        new_offset = len(raw_bytes)

        lines: list[KubectlLogLine] = []
        for line in new_content.splitlines():
            if not line.strip():
                continue
            lines.append(_parse_kubectl_log_line(line))

        return KubectlLogResult(lines=lines, byte_offset=new_offset)

    def top_pod(self, pod_name: str) -> tuple[int, int] | None:
        """Get (cpu_millicores, memory_bytes) for a pod."""
        result = self._run(
            ["top", "pod", pod_name, "--no-headers", "--containers"],
            namespaced=True,
            timeout=10.0,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                return (_parse_k8s_cpu(parts[2]), _parse_k8s_memory(parts[3]))
        return None

    def _popen(
        self,
        args: list[str],
        *,
        namespaced: bool = False,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a kubectl subprocess without waiting for completion."""
        cmd = list(self._prefix)
        if namespaced:
            cmd.extend(["-n", self.namespace])
        cmd.extend(args)
        return subprocess.Popen(cmd, **kwargs)

    @contextmanager
    def port_forward(
        self,
        service_name: str,
        remote_port: int,
        local_port: int | None = None,
        timeout: float = 90.0,
    ) -> Iterator[str]:
        """Port-forward to a K8s Service, yielding the local URL.

        Uses exponential backoff to handle freshly provisioned nodes whose
        konnectivity agent may not be ready when the pod first passes its
        readiness probe. If the kubectl process exits, it is relaunched.
        """
        if local_port is None:
            local_port = find_free_port(start=10000)

        proc: subprocess.Popen | None = None

        def _stop() -> None:
            nonlocal proc
            if proc is None:
                return
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc = None

        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)

        while not deadline.expired():
            if proc is None:
                proc = self._popen(
                    ["port-forward", f"svc/{service_name}", f"{local_port}:{remote_port}"],
                    namespaced=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )

            if proc.poll() is not None:
                stderr = proc.stderr.read() if proc.stderr else ""
                logger.warning("Port-forward exited (retrying): %s", stderr.strip())
                proc = None
                time.sleep(min(backoff.next_interval(), max(0, deadline.remaining_seconds())))
                continue

            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            _stop()
            # Capture konnectivity-agent state for diagnostics.
            try:
                diag = self._popen(
                    ["get", "pods", "-n", "kube-system", "-o", "wide"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, _ = diag.communicate(timeout=10)
                if diag.returncode == 0:
                    logger.warning("kube-system pods at tunnel failure:\n%s", stdout.strip())
            except (subprocess.TimeoutExpired, OSError):
                pass
            raise RuntimeError(f"kubectl port-forward to {service_name}:{remote_port} failed after {timeout}s")

        logger.info("Tunnel ready: 127.0.0.1:%d -> %s:%d", local_port, service_name, remote_port)
        try:
            yield f"http://127.0.0.1:{local_port}"
        finally:
            _stop()


def _parse_kubectl_log_line(line: str) -> KubectlLogLine:
    """Parse a ``kubectl logs --timestamps`` line.

    Format: ``<RFC3339-timestamp> <message>``
    e.g. ``2026-02-20T21:19:05.826882951Z Built haliax @ file:///app/lib/haliax``
    """
    parts = line.split(" ", 1)
    if len(parts) == 2:
        ts_str, payload = parts
        try:
            # Truncate nanoseconds -> microseconds for fromisoformat
            if len(ts_str) > 27 and ts_str.endswith("Z"):
                ts_str = ts_str[:26] + "Z"
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return KubectlLogLine(timestamp=ts, stream="stdout", data=payload)
        except ValueError:
            logger.warning("Failed to parse timestamp from kubectl log line: %r", line[:120])
    else:
        logger.warning("Unexpected kubectl log line format (no space-separated timestamp): %r", line[:120])
    return KubectlLogLine(timestamp=datetime.now(timezone.utc), stream="stdout", data=line)
