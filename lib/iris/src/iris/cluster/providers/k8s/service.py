# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sService protocol and CloudK8sService (kubernetes-client-backed) implementation."""

from __future__ import annotations

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

import kubernetes
import kubernetes.client
import kubernetes.config
import kubernetes.stream
from kubernetes.client.exceptions import ApiException

from iris.cluster.providers.k8s.types import ExecResult, K8sResource, KubectlError, KubectlLogLine, KubectlLogResult
from iris.cluster.providers.k8s.types import parse_k8s_cpu as _parse_k8s_cpu
from iris.cluster.providers.k8s.types import parse_k8s_memory as _parse_k8s_memory
from iris.cluster.providers.types import find_free_port
from rigging.timing import Deadline, ExponentialBackoff

logger = logging.getLogger(__name__)

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT: float = 60.0

# Slow-operation warning threshold (milliseconds)
_SLOW_THRESHOLD_MS: float = 2000.0


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

    def get_json(self, resource: K8sResource, name: str) -> dict | None: ...

    def list_json(
        self,
        resource: K8sResource,
        *,
        labels: dict[str, str] | None = None,
        field_selector: str | None = None,
    ) -> list[dict]: ...

    def delete(self, resource: K8sResource, name: str, *, force: bool = False, wait: bool = True) -> None: ...

    def delete_many(self, resource: K8sResource, names: list[str], *, wait: bool = False) -> None:
        """Delete multiple resources by name in a single kubectl call."""
        ...

    def delete_by_labels(self, resource: K8sResource, labels: dict[str, str], *, wait: bool = False) -> None:
        """Delete all resources matching the given label selector."""
        ...

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str: ...

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        since_time: datetime | None = None,
    ) -> KubectlLogResult: ...

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult: ...

    def set_image(self, resource: K8sResource, name: str, container: str, image: str) -> None: ...

    def rollout_restart(self, resource: K8sResource, name: str) -> None: ...

    def rollout_status(self, resource: K8sResource, name: str, *, timeout: float = 600.0) -> None: ...

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


def _label_selector(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in labels.items())


def _log_operation(operation: str, t0: float) -> None:
    """Log elapsed time, warning if slow."""
    elapsed_ms = (time.monotonic() - t0) * 1000
    if elapsed_ms > _SLOW_THRESHOLD_MS:
        logger.warning("k8s slow: %dms op=%s", elapsed_ms, operation)


# ---------------------------------------------------------------------------
# CloudK8sService — kubernetes-client-backed implementation
# ---------------------------------------------------------------------------


@dataclass
class CloudK8sService:
    """K8sService backed by the kubernetes Python client. Used in CLOUD mode.

    Encapsulates API client construction (including optional kubeconfig),
    namespace injection, serialization, and error handling.
    """

    namespace: str
    kubeconfig_path: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _api_client: kubernetes.client.ApiClient = field(init=False, repr=False)
    _core_v1: kubernetes.client.CoreV1Api = field(init=False, repr=False)
    _apps_v1: kubernetes.client.AppsV1Api = field(init=False, repr=False)
    _policy_v1: kubernetes.client.PolicyV1Api = field(init=False, repr=False)
    _custom: kubernetes.client.CustomObjectsApi = field(init=False, repr=False)
    _kubectl_prefix: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kubeconfig_path:
            self.kubeconfig_path = os.path.expanduser(self.kubeconfig_path)
            self._api_client = kubernetes.config.new_client_from_config(
                config_file=self.kubeconfig_path,
            )
        else:
            try:
                kubernetes.config.load_incluster_config()
                self._api_client = kubernetes.client.ApiClient()
            except kubernetes.config.ConfigException:
                self._api_client = kubernetes.config.new_client_from_config()

        self._core_v1 = kubernetes.client.CoreV1Api(self._api_client)
        self._apps_v1 = kubernetes.client.AppsV1Api(self._api_client)
        self._policy_v1 = kubernetes.client.PolicyV1Api(self._api_client)
        self._custom = kubernetes.client.CustomObjectsApi(self._api_client)

        # Keep kubectl prefix for port-forward subprocess only.
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        self._kubectl_prefix = cmd

    def _serialize(self, obj: object) -> dict:
        """Convert a kubernetes client object to a camelCase dict."""
        return self._api_client.sanitize_for_serialization(obj)

    def _call_api(
        self,
        path: str,
        method: str,
        *,
        body: dict | None = None,
        query_params: list[tuple[str, str]] | None = None,
        content_type: str = "application/json",
        timeout: float | None = None,
    ) -> dict:
        """Central API dispatch. Injects auth and timeout for every call."""
        resp = self._api_client.call_api(
            path,
            method,
            body=body,
            query_params=query_params or [],
            header_params={"Content-Type": content_type, "Accept": "application/json"},
            response_type="object",
            auth_settings=["BearerToken"],
            _request_timeout=timeout or self.timeout,
        )
        return resp[0]  # call_api returns (data, status, headers)

    # -- apply ---------------------------------------------------------------

    def apply_json(self, manifest: dict) -> None:
        """Apply a manifest using GET-then-create-or-patch, matching kubectl apply."""
        kind = manifest.get("kind", "?")
        name = manifest["metadata"]["name"]
        res = K8sResource.from_kind(manifest["kind"])
        ns = manifest["metadata"].get("namespace", self.namespace)
        item_path = res.item_path(name, ns)
        collection_path = res.collection_path(ns)
        logger.info("k8s: apply %s/%s", kind, name)
        t0 = time.monotonic()

        exists = False
        try:
            self._call_api(item_path, "GET")
            exists = True
        except ApiException as e:
            if e.status != 404:
                raise KubectlError(
                    f"apply get {kind}/{name} failed ({e.status}): {e.reason} {(e.body or '')[:500]}"
                ) from e

        try:
            if exists:
                self._call_api(item_path, "PATCH", body=manifest, content_type="application/strategic-merge-patch+json")
            else:
                self._call_api(collection_path, "POST", body=manifest)
        except ApiException as e:
            op = "patch" if exists else "create"
            raise KubectlError(f"apply {op} {kind}/{name} failed ({e.status}): {e.reason} {(e.body or '')[:500]}") from e
        _log_operation(f"apply {kind}/{name}", t0)

    # -- get -----------------------------------------------------------------

    def get_json(self, resource: K8sResource, name: str) -> dict | None:
        """Get a Kubernetes resource as a parsed dict. Returns None if not found."""
        path = resource.item_path(name, self.namespace)
        logger.info("k8s: GET %s", path)
        t0 = time.monotonic()
        try:
            result = self._call_api(path, "GET")
            _log_operation(f"get {resource.plural}/{name}", t0)
            return result
        except ApiException as e:
            if e.status == 404:
                return None
            raise KubectlError(f"get {resource.plural}/{name} failed ({e.status}): {e.reason}") from e

    # -- list ----------------------------------------------------------------

    def list_json(
        self,
        resource: K8sResource,
        *,
        labels: dict[str, str] | None = None,
        field_selector: str | None = None,
    ) -> list[dict]:
        """List Kubernetes resources, optionally filtered by labels and/or field selectors."""
        path = resource.collection_path(self.namespace)
        query_params: list[tuple[str, str]] = []
        if labels:
            query_params.append(("labelSelector", _label_selector(labels)))
        if field_selector:
            query_params.append(("fieldSelector", field_selector))

        logger.info("k8s: LIST %s labels=%s field_selector=%s", path, labels, field_selector)
        t0 = time.monotonic()
        try:
            data = self._call_api(path, "GET", query_params=query_params)
            _log_operation(f"list {resource.plural}", t0)
            return data.get("items", [])
        except ApiException as e:
            raise KubectlError(f"list {resource.plural} failed ({e.status}): {e.reason}") from e

    # -- delete --------------------------------------------------------------

    def delete(self, resource: K8sResource, name: str, *, force: bool = False, wait: bool = True) -> None:
        """Delete a Kubernetes resource, ignoring NotFound errors."""
        path = resource.item_path(name, self.namespace)
        query_params: list[tuple[str, str]] = []
        if force:
            query_params.append(("gracePeriodSeconds", "0"))
        if not wait:
            query_params.append(("propagationPolicy", "Background"))

        logger.info("k8s: DELETE %s force=%s wait=%s", path, force, wait)
        t0 = time.monotonic()
        try:
            self._call_api(path, "DELETE", query_params=query_params)
        except ApiException as e:
            if e.status == 404:
                return
            raise KubectlError(f"delete {resource.plural}/{name} failed ({e.status}): {e.reason}") from e
        _log_operation(f"delete {resource.plural}/{name}", t0)

    def delete_many(self, resource: K8sResource, names: list[str], *, wait: bool = False) -> None:
        """Delete multiple resources by name via individual API calls."""
        if not names:
            return
        logger.info("k8s: DELETE_MANY %s count=%d", resource.plural, len(names))
        t0 = time.monotonic()
        for name in names:
            self.delete(resource, name, wait=wait)
        _log_operation(f"delete_many {resource.plural} ({len(names)})", t0)

    def delete_by_labels(self, resource: K8sResource, labels: dict[str, str], *, wait: bool = False) -> None:
        """Delete all resources matching the given label selector."""
        if not labels:
            return
        path = resource.collection_path(self.namespace)
        selector = _label_selector(labels)
        query_params: list[tuple[str, str]] = [("labelSelector", selector)]
        if not wait:
            query_params.append(("propagationPolicy", "Background"))

        logger.info("k8s: DELETE_COLLECTION %s labels=%s", path, labels)
        t0 = time.monotonic()
        try:
            self._call_api(path, "DELETE", query_params=query_params)
        except ApiException as e:
            if e.status == 404:
                return
            raise KubectlError(
                f"delete_by_labels {resource.plural} -l {selector} failed ({e.status}): {e.reason}"
            ) from e
        _log_operation(f"delete_by_labels {resource.plural} -l {selector}", t0)

    # -- set_image -----------------------------------------------------------

    def set_image(self, resource: K8sResource, name: str, container: str, image: str) -> None:
        """Set the container image on a deployment/statefulset."""
        patch_body = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{"name": container, "image": image}],
                    }
                }
            }
        }
        path = resource.item_path(name, self.namespace)
        logger.info("k8s: PATCH set_image %s/%s container=%s image=%s", resource.plural, name, container, image)
        t0 = time.monotonic()
        try:
            self._call_api(path, "PATCH", body=patch_body, content_type="application/strategic-merge-patch+json")
        except ApiException as e:
            raise KubectlError(f"set_image {resource.plural}/{name} failed ({e.status}): {e.reason}") from e
        _log_operation(f"set_image {resource.plural}/{name}", t0)

    # -- rollout_restart -----------------------------------------------------

    def rollout_restart(self, resource: K8sResource, name: str) -> None:
        """Restart a rollout by patching the restart annotation."""
        now = datetime.now(timezone.utc).isoformat()
        patch_body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": now,
                        }
                    }
                }
            }
        }
        path = resource.item_path(name, self.namespace)
        logger.info("k8s: PATCH rollout_restart %s/%s", resource.plural, name)
        t0 = time.monotonic()
        try:
            self._call_api(path, "PATCH", body=patch_body, content_type="application/strategic-merge-patch+json")
        except ApiException as e:
            raise KubectlError(f"rollout_restart {resource.plural}/{name} failed ({e.status}): {e.reason}") from e
        _log_operation(f"rollout_restart {resource.plural}/{name}", t0)

    # -- rollout_status ------------------------------------------------------

    def rollout_status(self, resource: K8sResource, name: str, *, timeout: float = 600.0) -> None:
        """Wait for a rollout to complete by polling deployment conditions."""
        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)
        path = resource.item_path(name, self.namespace)
        logger.info("k8s: rollout_status %s/%s timeout=%.0fs", resource.plural, name, timeout)

        while not deadline.expired():
            try:
                obj = self._call_api(path, "GET")
            except ApiException as e:
                raise KubectlError(f"rollout_status {resource.plural}/{name} failed ({e.status}): {e.reason}") from e

            status = obj.get("status", {})
            spec = obj.get("spec", {})
            desired = spec.get("replicas", 1)
            updated = status.get("updatedReplicas", 0)
            ready = status.get("readyReplicas", 0)
            available = status.get("availableReplicas", 0)

            if updated >= desired and ready >= desired and available >= desired:
                # Check that observedGeneration matches metadata.generation
                observed = status.get("observedGeneration", 0)
                generation = obj.get("metadata", {}).get("generation", 0)
                if observed >= generation:
                    logger.info("k8s: rollout_status %s/%s complete", resource.plural, name)
                    return

            time.sleep(min(backoff.next_interval(), max(0, deadline.remaining_seconds())))

        raise KubectlError(f"rollout_status {resource.plural}/{name} timed out after {timeout}s")

    # -- events --------------------------------------------------------------

    def get_events(
        self,
        field_selector: str | None = None,
    ) -> list[dict]:
        """Get Kubernetes events, optionally filtered by field selector."""
        logger.info("k8s: list events field_selector=%s", field_selector)
        t0 = time.monotonic()
        try:
            kwargs: dict = {"namespace": self.namespace, "_request_timeout": self.timeout}
            if field_selector:
                kwargs["field_selector"] = field_selector
            result = self._core_v1.list_namespaced_event(**kwargs)
            _log_operation("get_events", t0)
            return [self._serialize(item) for item in result.items]
        except ApiException:
            logger.warning("get_events failed", exc_info=True)
            return []

    # -- logs ----------------------------------------------------------------

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        """Fetch logs from a Pod container."""
        logger.info("k8s: logs %s container=%s tail=%d previous=%s", pod_name, container, tail, previous)
        t0 = time.monotonic()
        try:
            kwargs: dict = {
                "name": pod_name,
                "namespace": self.namespace,
                "tail_lines": tail,
                "previous": previous,
                "_request_timeout": self.timeout,
            }
            if container:
                kwargs["container"] = container
            result = self._core_v1.read_namespaced_pod_log(**kwargs)
            _log_operation(f"logs {pod_name}", t0)
            return result
        except ApiException:
            return ""

    # -- stream_logs ---------------------------------------------------------

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        since_time: datetime | None = None,
    ) -> KubectlLogResult:
        """Fetch new log lines from a pod, bounded by since_time."""
        logger.info("k8s: stream_logs %s since=%s", pod_name, since_time)
        t0 = time.monotonic()
        try:
            kwargs: dict = {
                "name": pod_name,
                "namespace": self.namespace,
                "timestamps": True,
                "_request_timeout": 15.0,
            }
            if container:
                kwargs["container"] = container
            if since_time is not None:
                # Compute since_seconds from the delta. Add 1s margin to ensure
                # we don't miss boundary lines (we filter duplicates below).
                delta = datetime.now(timezone.utc) - since_time
                since_sec = max(1, int(delta.total_seconds()) + 1)
                kwargs["since_seconds"] = since_sec

            raw = self._core_v1.read_namespaced_pod_log(**kwargs)
            _log_operation(f"stream_logs {pod_name}", t0)
        except ApiException:
            return KubectlLogResult(lines=[], last_timestamp=since_time)

        lines: list[KubectlLogLine] = []
        for line_str in raw.splitlines():
            if not line_str.strip():
                continue
            parsed = _parse_kubectl_log_line(line_str)
            # since_seconds is inclusive; skip already-seen boundary lines
            if since_time is not None and parsed.timestamp <= since_time:
                continue
            lines.append(parsed)

        last_ts = lines[-1].timestamp if lines else since_time
        return KubectlLogResult(lines=lines, last_timestamp=last_ts)

    # -- exec ----------------------------------------------------------------

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        """Run a command inside a Pod container."""
        effective_timeout = timeout if timeout is not None else self.timeout
        logger.info("k8s: exec %s cmd=%s container=%s", pod_name, cmd, container)
        t0 = time.monotonic()
        try:
            kwargs: dict = {
                "name": pod_name,
                "namespace": self.namespace,
                "command": cmd,
                "stdout": True,
                "stderr": True,
                "stdin": False,
                "tty": False,
                "_request_timeout": effective_timeout,
            }
            if container:
                kwargs["container"] = container

            resp = kubernetes.stream.stream(
                self._core_v1.connect_get_namespaced_pod_exec,
                **kwargs,
            )
            _log_operation(f"exec {pod_name}", t0)
            # stream() returns the combined stdout when not using a websocket client.
            # For separate streams we'd need _preload_content=False, but the simple
            # interface returns stdout as a string. stderr comes via the error channel.
            return ExecResult(returncode=0, stdout=resp, stderr="")
        except ApiException as e:
            _log_operation(f"exec {pod_name} (error)", t0)
            return ExecResult(returncode=1, stdout="", stderr=str(e))

    # -- read_file / rm_files ------------------------------------------------

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

    # -- top_pod -------------------------------------------------------------

    def top_pod(self, pod_name: str) -> tuple[int, int] | None:
        """Get (cpu_millicores, memory_bytes) for a pod via metrics.k8s.io API."""
        logger.info("k8s: top_pod %s", pod_name)
        t0 = time.monotonic()
        try:
            result = self._custom.get_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=self.namespace,
                plural="pods",
                name=pod_name,
            )
            _log_operation(f"top_pod {pod_name}", t0)
        except ApiException:
            return None

        containers = result.get("containers", [])
        if not containers:
            return None

        total_cpu = 0
        total_mem = 0
        for c in containers:
            usage = c.get("usage", {})
            if "cpu" in usage:
                total_cpu += _parse_k8s_cpu(usage["cpu"])
            if "memory" in usage:
                total_mem += _parse_k8s_memory(usage["memory"])
        return (total_cpu, total_mem)

    # -- port_forward (subprocess-based) -------------------------------------

    def _popen(
        self,
        args: list[str],
        *,
        namespaced: bool = False,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a kubectl subprocess without waiting for completion."""
        cmd = list(self._kubectl_prefix)
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
