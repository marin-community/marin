# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory K8sService implementation for DRY_RUN/LOCAL testing.

Validates manifests, tracks state in memory, and supports failure injection
for testing K8s-dependent code without shelling out to kubectl.
"""

from __future__ import annotations

import subprocess
from enum import StrEnum

from iris.cluster.k8s.kubectl import KubectlError, KubectlLogResult


class ServiceMode(StrEnum):
    DRY_RUN = "dry_run"
    LOCAL = "local"
    CLOUD = "cloud"


# Resource types that K8s recognizes in container resource requests/limits.
VALID_RESOURCE_TYPES = frozenset(
    {
        "cpu",
        "memory",
        "nvidia.com/gpu",
        "google.com/tpu",
        "ephemeral-storage",
    }
)


class K8sServiceImpl:
    """In-memory K8sService for DRY_RUN/LOCAL testing.

    Validates manifests, tracks state, supports failure injection.
    In LOCAL mode, could spawn local processes for pods (future work).
    """

    def __init__(
        self,
        namespace: str = "iris",
        mode: ServiceMode = ServiceMode.DRY_RUN,
        available_node_pools: list[str] | None = None,
    ):
        self._namespace = namespace
        self._mode = mode
        self._available_node_pools: set[str] | None = (
            set(available_node_pools) if available_node_pools is not None else None
        )
        self._resources: dict[tuple[str, str], dict] = {}  # (kind, name) -> manifest
        self._injected_failures: dict[str, Exception] = {}
        self._logs: dict[str, str] = {}  # pod_name -> log text
        self._events: list[dict] = []

    @property
    def namespace(self) -> str:
        return self._namespace

    # -- Validation --

    def _validate_manifest(self, manifest: dict) -> None:
        """Validate manifest structure, node selectors, and resource request types."""
        kind = manifest.get("kind", "")
        if not kind:
            raise KubectlError("Manifest missing 'kind'")

        name = manifest.get("metadata", {}).get("name", "")
        if not name:
            raise KubectlError("Manifest missing 'metadata.name'")

        # For pod-like resources, validate spec contents
        spec = _pod_spec(manifest)
        if spec is None:
            return

        node_selector = spec.get("nodeSelector", {})
        pool = node_selector.get("cloud.google.com/gke-nodepool")
        if pool and self._available_node_pools is not None and pool not in self._available_node_pools:
            raise KubectlError(f"Node pool {pool!r} not found")

        for container in spec.get("containers", []):
            resources = container.get("resources", {})
            for section in ("requests", "limits"):
                for key in resources.get(section, {}):
                    if key not in VALID_RESOURCE_TYPES:
                        raise KubectlError(f"Unknown resource type: {key!r}")

    # -- Failure injection --

    def inject_failure(self, operation: str, error: Exception) -> None:
        """Inject a one-shot failure for the next call to *operation*."""
        self._injected_failures[operation] = error

    def clear_failure(self, operation: str) -> None:
        self._injected_failures.pop(operation, None)

    def remove_node_pool(self, pool_name: str) -> None:
        """Simulate a node pool disappearing."""
        if self._available_node_pools is not None:
            self._available_node_pools.discard(pool_name)

    def add_node_pool(self, pool_name: str) -> None:
        if self._available_node_pools is None:
            self._available_node_pools = set()
        self._available_node_pools.add(pool_name)

    # -- Test helpers --

    def set_logs(self, pod_name: str, text: str) -> None:
        """Pre-populate logs for a pod."""
        self._logs[pod_name] = text

    def add_event(self, event: dict) -> None:
        self._events.append(event)

    # -- Protocol methods --

    def _check_failure(self, operation: str) -> None:
        if err := self._injected_failures.pop(operation, None):
            raise err

    def apply_json(self, manifest: dict) -> None:
        self._check_failure("apply_json")
        self._validate_manifest(manifest)
        kind = manifest["kind"].lower()
        name = manifest["metadata"]["name"]
        self._resources[(kind, name)] = manifest

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None:
        self._check_failure("get_json")
        return self._resources.get((resource.lower(), name))

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]:
        self._check_failure("list_json")
        results = []
        resource_lower = resource.lower()
        for (kind, _), manifest in self._resources.items():
            if kind != resource_lower:
                continue
            if labels:
                res_labels = manifest.get("metadata", {}).get("labels", {})
                if not all(res_labels.get(k) == v for k, v in labels.items()):
                    continue
            results.append(manifest)
        return results

    def delete(
        self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True
    ) -> None:
        self._check_failure("delete")
        self._resources.pop((resource.lower(), name), None)

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        self._check_failure("logs")
        text = self._logs.get(pod_name, "")
        if tail and text:
            lines = text.splitlines()
            return "\n".join(lines[-tail:])
        return text

    def stream_logs(
        self,
        pod_name: str,
        *,
        container: str | None = None,
        byte_offset: int = 0,
    ) -> KubectlLogResult:
        self._check_failure("stream_logs")
        text = self._logs.get(pod_name, "")
        raw = text.encode("utf-8")
        if len(raw) <= byte_offset:
            return KubectlLogResult(lines=[], byte_offset=byte_offset)
        # Return empty lines list; callers needing parsed lines can use logs()
        return KubectlLogResult(lines=[], byte_offset=len(raw))

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._check_failure("exec")
        if (pod_name.lower(), pod_name) not in self._resources and not any(
            name == pod_name for (_, name) in self._resources
        ):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=f"pod {pod_name!r} not found")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None:
        self._check_failure("set_image")
        manifest = self._resources.get((resource.lower(), name))
        if manifest is None:
            raise KubectlError(f"{resource}/{name} not found")

    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None:
        self._check_failure("rollout_restart")

    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None:
        self._check_failure("rollout_status")

    def get_events(self, field_selector: str | None = None) -> list[dict]:
        self._check_failure("get_events")
        if field_selector is None:
            return list(self._events)
        # Simple field_selector matching: "involvedObject.name=foo"
        results = []
        for event in self._events:
            if _matches_field_selector(event, field_selector):
                results.append(event)
        return results

    def top_pod(self, pod_name: str) -> tuple[int, int] | None:
        self._check_failure("top_pod")
        # Return synthetic metrics if pod exists
        if any(name == pod_name for (_, name) in self._resources):
            return (100, 256 * 1024 * 1024)  # 100m CPU, 256Mi memory
        return None

    def read_file(
        self,
        pod_name: str,
        path: str,
        *,
        container: str | None = None,
    ) -> bytes:
        self._check_failure("read_file")
        raise RuntimeError(f"read_file not supported in {self._mode} mode")

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None:
        self._check_failure("rm_files")


def _pod_spec(manifest: dict) -> dict | None:
    """Extract the pod spec from a manifest, handling Pods, Deployments, Jobs, etc."""
    kind = manifest.get("kind", "").lower()
    spec = manifest.get("spec", {})

    if kind == "pod":
        return spec
    if kind in ("deployment", "replicaset", "statefulset", "daemonset"):
        return spec.get("template", {}).get("spec", {})
    if kind == "job":
        return spec.get("template", {}).get("spec", {})
    return None


def _matches_field_selector(event: dict, selector: str) -> bool:
    """Minimal field_selector matching for testing (supports key=value pairs)."""
    for part in selector.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        segments = key.strip().split(".")
        obj = event
        for seg in segments:
            if not isinstance(obj, dict):
                return False
            obj = obj.get(seg)
            if obj is None:
                return False
        if str(obj) != value.strip():
            return False
    return True
