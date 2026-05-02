# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""K8sService protocol and CloudK8sService (kubernetes DynamicClient) implementation."""

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

try:
    import kubernetes
    import kubernetes.client
    import kubernetes.config
    import kubernetes.stream
    from kubernetes.client.exceptions import ApiException
    from kubernetes.dynamic import DynamicClient
    from kubernetes.dynamic.exceptions import NotFoundError
except ImportError:
    kubernetes = None  # type: ignore[assignment]
    ApiException = Exception  # type: ignore[assignment,misc]
    NotFoundError = Exception  # type: ignore[assignment,misc]
    DynamicClient = None  # type: ignore[assignment,misc]


from rigging.log_setup import slow_log
from rigging.timing import Deadline, ExponentialBackoff

from iris.cluster.providers.k8s.types import (
    ExecResult,
    K8sResource,
    KubectlError,
    KubectlLogLine,
    KubectlLogResult,
    PodResourceUsage,
    parse_k8s_cpu,
    parse_k8s_quantity,
)
from iris.cluster.providers.types import find_free_port

logger = logging.getLogger(__name__)

# Default timeout for API calls (seconds)
DEFAULT_TIMEOUT: float = 60.0

# Threshold for slow-operation warnings (milliseconds)
_SLOW_THRESHOLD_MS: int = 2000


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
        """Delete multiple resources by name."""
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
        limit_bytes: int | None = None,
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

    def top_pod(self, pod_name: str) -> PodResourceUsage | None: ...

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
        """Port-forward to a K8s Service, yielding the local URL."""
        ...


def _label_selector(labels: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in labels.items())


# ---------------------------------------------------------------------------
# CloudK8sService — DynamicClient-backed implementation
# ---------------------------------------------------------------------------


@dataclass
class CloudK8sService:
    """K8sService backed by the kubernetes DynamicClient.

    Uses DynamicClient for CRUD operations (handles auth, serialization,
    content types, and URL construction automatically). Falls back to
    typed CoreV1Api for subresource operations (logs, exec) and
    subprocess for port-forward.
    """

    namespace: str
    kubeconfig_path: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _api_client: kubernetes.client.ApiClient = field(init=False, repr=False)
    _dyn: DynamicClient = field(init=False, repr=False)
    _core_v1: kubernetes.client.CoreV1Api = field(init=False, repr=False)
    _custom: kubernetes.client.CustomObjectsApi = field(init=False, repr=False)
    _kubectl_prefix: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if kubernetes is None:
            raise ImportError("Install iris[controller] to use CloudK8sService")
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

        assert DynamicClient is not None
        self._dyn = DynamicClient(self._api_client)
        self._core_v1 = kubernetes.client.CoreV1Api(self._api_client)
        self._custom = kubernetes.client.CustomObjectsApi(self._api_client)

        # kubectl prefix for port-forward subprocess only
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        self._kubectl_prefix = cmd

    def _resource_api(self, resource: K8sResource):
        """Get the DynamicClient resource handle for a K8sResource enum member."""
        api_version = f"{resource.api_group}/{resource.api_version}" if resource.api_group else resource.api_version
        return self._dyn.resources.get(api_version=api_version, kind=resource.kind)

    def _ns_kwargs(self, resource: K8sResource) -> dict:
        """Return namespace kwarg if the resource is namespaced."""
        if resource.is_namespaced:
            return {"namespace": self.namespace}
        return {}

    def _request_timeout_kwargs(self, timeout: float | None = None) -> dict:
        """Return the default client request timeout kwargs for K8s API calls."""
        return {"_request_timeout": timeout if timeout is not None else self.timeout}

    # -- apply ---------------------------------------------------------------

    def apply_json(self, manifest: dict) -> None:
        """Apply a manifest via server-side apply, matching kubectl apply semantics.

        Uses SSA (application/apply-patch+yaml) which correctly reconciles
        field deletions, unlike merge-patch which only adds/overwrites.
        Pods are immutable so they get delete-then-create instead.
        """
        kind = manifest.get("kind", "?")
        name = manifest["metadata"]["name"]
        res = K8sResource.from_kind(manifest["kind"])
        ns = manifest["metadata"].get("namespace", self.namespace) if res.is_namespaced else None

        logger.info("k8s: apply %s/%s", kind, name)
        with slow_log(logger, f"apply {kind}/{name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                if res is K8sResource.PODS:
                    self._apply_pod(res, name, ns, manifest)
                else:
                    self._dyn.server_side_apply(
                        resource=self._resource_api(res),
                        body=manifest,
                        name=name,
                        field_manager="iris",
                        force_conflicts=True,
                        **self._request_timeout_kwargs(),
                        **({"namespace": ns} if ns else {}),
                    )
            except ApiException as e:
                raise KubectlError(f"apply {kind}/{name} failed ({e.status}): {e.reason} {(e.body or '')[:500]}") from e

    def _apply_pod(self, res: K8sResource, name: str, ns: str | None, manifest: dict) -> None:
        """Apply a Pod manifest. Pods are mostly immutable, so delete-then-create."""
        api = self._resource_api(res)
        ns_kw = {"namespace": ns} if ns else {}
        timeout_kw = self._request_timeout_kwargs()
        try:
            api.delete(name=name, **timeout_kw, **ns_kw)
        except (NotFoundError, ApiException):
            pass
        api.create(body=manifest, **timeout_kw, **ns_kw)

    # -- get -----------------------------------------------------------------

    def get_json(self, resource: K8sResource, name: str) -> dict | None:
        """Get a Kubernetes resource as a parsed dict. Returns None if not found."""
        logger.info("k8s: GET %s/%s", resource.plural, name)
        with slow_log(logger, f"get {resource.plural}/{name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                result = self._resource_api(resource).get(
                    name=name,
                    **self._request_timeout_kwargs(),
                    **self._ns_kwargs(resource),
                )
                return result.to_dict()
            except NotFoundError:
                return None
            except ApiException as e:
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
        logger.info("k8s: LIST %s labels=%s field_selector=%s", resource.plural, labels, field_selector)
        kwargs = self._ns_kwargs(resource)
        if labels:
            kwargs["label_selector"] = _label_selector(labels)
        if field_selector:
            kwargs["field_selector"] = field_selector
        kwargs.update(self._request_timeout_kwargs())
        with slow_log(logger, f"list {resource.plural}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                result = self._resource_api(resource).get(**kwargs)
                return [item.to_dict() for item in result.items]
            except ApiException as e:
                raise KubectlError(f"list {resource.plural} failed ({e.status}): {e.reason}") from e

    # -- delete --------------------------------------------------------------

    def delete(self, resource: K8sResource, name: str, *, force: bool = False, wait: bool = True) -> None:
        """Delete a Kubernetes resource, ignoring NotFound errors."""
        logger.info("k8s: DELETE %s/%s force=%s wait=%s", resource.plural, name, force, wait)
        kwargs = self._ns_kwargs(resource)
        kwargs.update(self._request_timeout_kwargs())
        body: dict = {}
        if force:
            body["gracePeriodSeconds"] = 0
        if not wait:
            body["propagationPolicy"] = "Background"
        with slow_log(logger, f"delete {resource.plural}/{name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                self._resource_api(resource).delete(name=name, body=body, **kwargs)
            except NotFoundError:
                return
            except ApiException as e:
                raise KubectlError(f"delete {resource.plural}/{name} failed ({e.status}): {e.reason}") from e

    def delete_many(self, resource: K8sResource, names: list[str], *, wait: bool = False) -> None:
        """Delete multiple resources by name."""
        if not names:
            return
        logger.info("k8s: DELETE_MANY %s count=%d", resource.plural, len(names))
        with slow_log(logger, f"delete_many {resource.plural} ({len(names)})", threshold_ms=_SLOW_THRESHOLD_MS):
            for name in names:
                self.delete(resource, name, wait=wait)

    def delete_by_labels(self, resource: K8sResource, labels: dict[str, str], *, wait: bool = False) -> None:
        """Delete all resources matching the given label selector."""
        if not labels:
            return
        selector = _label_selector(labels)
        logger.info("k8s: DELETE_COLLECTION %s labels=%s", resource.plural, labels)
        kwargs = self._ns_kwargs(resource)
        kwargs["label_selector"] = selector
        if not wait:
            kwargs["propagation_policy"] = "Background"
        kwargs.update(self._request_timeout_kwargs())
        with slow_log(logger, f"delete_by_labels {resource.plural} -l {selector}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                api = self._resource_api(resource)
                items = api.get(**{k: v for k, v in kwargs.items() if k != "propagation_policy"}).items
                for item in items:
                    self.delete(resource, item.metadata.name, wait=wait)
            except NotFoundError:
                return
            except ApiException as e:
                raise KubectlError(
                    f"delete_by_labels {resource.plural} -l {selector} failed ({e.status}): {e.reason}"
                ) from e

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
        logger.info("k8s: PATCH set_image %s/%s container=%s image=%s", resource.plural, name, container, image)
        with slow_log(logger, f"set_image {resource.plural}/{name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                self._resource_api(resource).patch(
                    body=patch_body,
                    name=name,
                    content_type="application/strategic-merge-patch+json",
                    **self._request_timeout_kwargs(),
                    **self._ns_kwargs(resource),
                )
            except ApiException as e:
                raise KubectlError(f"set_image {resource.plural}/{name} failed ({e.status}): {e.reason}") from e

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
        logger.info("k8s: PATCH rollout_restart %s/%s", resource.plural, name)
        with slow_log(logger, f"rollout_restart {resource.plural}/{name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                self._resource_api(resource).patch(
                    body=patch_body,
                    name=name,
                    content_type="application/strategic-merge-patch+json",
                    **self._request_timeout_kwargs(),
                    **self._ns_kwargs(resource),
                )
            except ApiException as e:
                raise KubectlError(f"rollout_restart {resource.plural}/{name} failed ({e.status}): {e.reason}") from e

    # -- rollout_status ------------------------------------------------------

    def rollout_status(self, resource: K8sResource, name: str, *, timeout: float = 600.0) -> None:
        """Wait for a rollout to complete by polling deployment conditions."""
        deadline = Deadline.from_seconds(timeout)
        backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)
        logger.info("k8s: rollout_status %s/%s timeout=%.0fs", resource.plural, name, timeout)

        while not deadline.expired():
            obj = self.get_json(resource, name)
            if obj is None:
                raise KubectlError(f"rollout_status {resource.plural}/{name} not found")

            status = obj.get("status", {})
            spec = obj.get("spec", {})
            desired = spec.get("replicas", 1)
            updated = status.get("updatedReplicas", 0)
            ready = status.get("readyReplicas", 0)
            available = status.get("availableReplicas", 0)

            if updated >= desired and ready >= desired and available >= desired:
                observed = status.get("observedGeneration", 0)
                generation = obj.get("metadata", {}).get("generation", 0)
                if observed >= generation:
                    logger.info("k8s: rollout_status %s/%s complete", resource.plural, name)
                    return

            time.sleep(min(backoff.next_interval(), max(0, deadline.remaining_seconds())))

        raise KubectlError(f"rollout_status {resource.plural}/{name} timed out after {timeout}s")

    # -- events --------------------------------------------------------------

    def get_events(self, field_selector: str | None = None) -> list[dict]:
        """Get Kubernetes events, optionally filtered by field selector."""
        logger.info("k8s: list events field_selector=%s", field_selector)
        with slow_log(logger, "get_events", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                kwargs: dict = {"namespace": self.namespace, "_request_timeout": self.timeout}
                if field_selector:
                    kwargs["field_selector"] = field_selector
                result = self._core_v1.list_namespaced_event(**kwargs)
                return self._api_client.sanitize_for_serialization(result)["items"]
            except ApiException as e:
                if e.status == 404:
                    return []
                raise

    # -- logs ----------------------------------------------------------------

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        """Fetch logs from a Pod container."""
        logger.info("k8s: logs %s container=%s tail=%d previous=%s", pod_name, container, tail, previous)
        with slow_log(logger, f"logs {pod_name}", threshold_ms=_SLOW_THRESHOLD_MS):
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
                return self._core_v1.read_namespaced_pod_log(**kwargs)
            except ApiException as e:
                if e.status == 404:
                    return ""
                raise

    # -- stream_logs ---------------------------------------------------------

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        since_time: datetime | None = None,
        limit_bytes: int | None = None,
    ) -> KubectlLogResult:
        """Fetch new log lines from a pod, bounded by since_time.

        When limit_bytes is set the K8s API caps the response size server-side.
        The response may be cut mid-line, so we drop the trailing partial line
        before parsing.
        """
        logger.info("k8s: stream_logs %s since=%s limit_bytes=%s", pod_name, since_time, limit_bytes)
        with slow_log(logger, f"stream_logs {pod_name}", threshold_ms=_SLOW_THRESHOLD_MS):
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
                    delta = datetime.now(timezone.utc) - since_time
                    since_sec = max(1, int(delta.total_seconds()) + 1)
                    kwargs["since_seconds"] = since_sec
                if limit_bytes is not None:
                    kwargs["limit_bytes"] = limit_bytes

                raw = self._core_v1.read_namespaced_pod_log(**kwargs)
            except ApiException as e:
                if e.status == 404:
                    return KubectlLogResult(lines=[], last_timestamp=since_time)
                raise

        # When limit_bytes is active the response may be truncated mid-line.
        # Drop the trailing partial line by trimming to the last newline.
        if limit_bytes is not None and raw and not raw.endswith("\n"):
            last_nl = raw.rfind("\n")
            raw = raw[: last_nl + 1] if last_nl >= 0 else ""

        lines: list[KubectlLogLine] = []
        for line_str in raw.splitlines():
            if not line_str.strip():
                continue
            parsed = _parse_kubectl_log_line(line_str)
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
        with slow_log(logger, f"exec {pod_name}", threshold_ms=_SLOW_THRESHOLD_MS):
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
                return ExecResult(returncode=0, stdout=resp, stderr="")
            except ApiException as e:
                return ExecResult(returncode=1, stdout="", stderr=str(e))

    # -- read_file / rm_files ------------------------------------------------

    def read_file(self, pod_name: str, path: str, *, container: str | None = None) -> bytes:
        """Read a file from inside a Pod container."""
        result = self.exec(pod_name, ["cat", path], container=container, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr}")
        return result.stdout.encode("utf-8")

    def rm_files(self, pod_name: str, paths: list[str], *, container: str | None = None) -> None:
        """Remove files inside a Pod container. Ignores missing files."""
        self.exec(pod_name, ["rm", "-f", *paths], container=container, timeout=10)

    # -- top_pod -------------------------------------------------------------

    def top_pod(self, pod_name: str) -> PodResourceUsage | None:
        """Get CPU/memory usage for a pod via metrics.k8s.io API."""
        logger.info("k8s: top_pod %s", pod_name)
        with slow_log(logger, f"top_pod {pod_name}", threshold_ms=_SLOW_THRESHOLD_MS):
            try:
                result = self._custom.get_namespaced_custom_object(
                    group="metrics.k8s.io",
                    version="v1beta1",
                    namespace=self.namespace,
                    plural="pods",
                    name=pod_name,
                    **self._request_timeout_kwargs(),
                )
            except ApiException as e:
                if e.status == 404:
                    return None
                raise

        containers = result.get("containers", [])
        if not containers:
            return None

        total_cpu = 0
        total_mem = 0
        for c in containers:
            usage = c.get("usage", {})
            if "cpu" in usage:
                total_cpu += parse_k8s_cpu(usage["cpu"])
            if "memory" in usage:
                total_mem += parse_k8s_quantity(usage["memory"])
        return PodResourceUsage(cpu_millicores=total_cpu, memory_bytes=total_mem)

    # -- port_forward (subprocess-based) -------------------------------------

    def _popen(self, args: list[str], *, namespaced: bool = False, **kwargs) -> subprocess.Popen:
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
        """Port-forward to a K8s Service, yielding the local URL."""
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
    """Parse a timestamped log line (format: ``<RFC3339> <message>``)."""
    parts = line.split(" ", 1)
    if len(parts) == 2:
        ts_str, payload = parts
        try:
            if len(ts_str) > 27 and ts_str.endswith("Z"):
                ts_str = ts_str[:26] + "Z"
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return KubectlLogLine(timestamp=ts, stream="stdout", data=payload)
        except ValueError:
            logger.warning("Failed to parse timestamp from log line: %r", line[:120])
    else:
        logger.warning("Unexpected log line format (no space-separated timestamp): %r", line[:120])
    return KubectlLogLine(timestamp=datetime.now(timezone.utc), stream="stdout", data=line)
