# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory K8sService implementation for DRY_RUN/LOCAL testing.

Validates manifests, tracks state in memory, and supports failure injection
for testing K8s-dependent code without shelling out to kubectl.

Includes a simplified K8s scheduler: node model with labels/taints/resources,
constraint-based pod scheduling, and resource commitment tracking.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime

from iris.cluster.k8s.k8s_types import KubectlError, KubectlLogLine, KubectlLogResult
from iris.cluster.service_mode import ServiceMode

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

# kubectl accepts plural resource names (e.g. "pods") but manifests use
# singular kind (e.g. "Pod"). Normalize for consistent internal lookup.
_PLURAL_TO_SINGULAR = {
    "pods": "pod",
    "nodes": "node",
    "configmaps": "configmap",
    "events": "event",
    "deployments": "deployment",
    "replicasets": "replicaset",
    "statefulsets": "statefulset",
    "daemonsets": "daemonset",
    "jobs": "job",
    "services": "service",
    "nodepools": "nodepool",
}


# ---------------------------------------------------------------------------
# Node & resource dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FakeNodeResources:
    """Resources a node has available (allocatable in K8s terms)."""

    cpu_millicores: int = 4000
    memory_bytes: int = 16 * 1024**3
    gpu_count: int = 0
    ephemeral_storage_bytes: int = 100 * 1024**3

    def to_allocatable(self) -> dict[str, str]:
        d: dict[str, str] = {
            "cpu": f"{self.cpu_millicores}m",
            "memory": str(self.memory_bytes),
            "ephemeral-storage": str(self.ephemeral_storage_bytes),
        }
        if self.gpu_count > 0:
            d["nvidia.com/gpu"] = str(self.gpu_count)
        return d


@dataclass
class FakeNode:
    """A schedulable node in the fake cluster."""

    name: str
    labels: dict[str, str]
    taints: list[dict[str, str]]
    allocatable: FakeNodeResources
    committed: FakeNodeResources = field(
        default_factory=lambda: FakeNodeResources(
            cpu_millicores=0, memory_bytes=0, gpu_count=0, ephemeral_storage_bytes=0
        )
    )

    def available(self) -> FakeNodeResources:
        return FakeNodeResources(
            cpu_millicores=self.allocatable.cpu_millicores - self.committed.cpu_millicores,
            memory_bytes=self.allocatable.memory_bytes - self.committed.memory_bytes,
            gpu_count=self.allocatable.gpu_count - self.committed.gpu_count,
            ephemeral_storage_bytes=self.allocatable.ephemeral_storage_bytes - self.committed.ephemeral_storage_bytes,
        )

    def to_k8s_dict(self) -> dict:
        result: dict = {
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {"name": self.name, "labels": dict(self.labels)},
            "status": {"allocatable": self.allocatable.to_allocatable()},
        }
        if self.taints:
            result["spec"] = {"taints": list(self.taints)}
        else:
            result["spec"] = {}
        return result


@dataclass
class NodePoolConfig:
    """Configuration for a pool of identical nodes."""

    name: str
    instance_type: str
    node_count: int
    labels: dict[str, str] = field(default_factory=dict)
    taints: list[dict[str, str]] = field(default_factory=list)
    per_node_resources: FakeNodeResources = field(default_factory=FakeNodeResources)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _normalize_resource(resource: str) -> str:
    """Normalize a resource type string to its singular lowercase form."""
    lower = resource.lower()
    return _PLURAL_TO_SINGULAR.get(lower, lower)


def _parse_k8s_quantity(val: str) -> int:
    """Parse K8s resource quantity strings like '4000m', '16Gi', '8'.

    Handles binary suffixes (Ki, Mi, Gi, Ti), SI suffixes (K, M, G, T),
    millicore 'm' suffix, and plain integers.
    """
    if not val:
        return 0
    binary_suffixes = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "Pi": 2**50}
    si_suffixes = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12, "P": 10**15}
    for suffix, mult in binary_suffixes.items():
        if val.endswith(suffix):
            return int(float(val[: -len(suffix)]) * mult)
    for suffix, mult in si_suffixes.items():
        if val.endswith(suffix) and not val.endswith("i"):
            return int(float(val[: -len(suffix)]) * mult)
    if val.endswith("m"):
        return int(val[:-1])
    return int(float(val))


def _extract_resource_requests(spec: dict) -> dict[str, int]:
    """Sum resource requests across all containers in a pod spec."""
    totals: dict[str, int] = {}
    for container in spec.get("containers", []):
        reqs = container.get("resources", {}).get("requests", {})
        limits = container.get("resources", {}).get("limits", {})
        for key in ("cpu", "memory", "nvidia.com/gpu", "ephemeral-storage"):
            val = reqs.get(key) or limits.get(key)
            if val:
                totals[key] = totals.get(key, 0) + _parse_k8s_quantity(val)
    return totals


def _node_selector_matches(node: FakeNode, selector: dict[str, str]) -> bool:
    return all(node.labels.get(k) == v for k, v in selector.items())


def _tolerations_satisfy_taints(tolerations: list[dict], taints: list[dict]) -> bool:
    """Every NoSchedule taint must be tolerated. Mirrors k8s TaintToleration plugin."""
    for taint in taints:
        if taint.get("effect") != "NoSchedule":
            continue
        if not any(_toleration_matches(t, taint) for t in tolerations):
            return False
    return True


def _toleration_matches(toleration: dict, taint: dict) -> bool:
    if toleration.get("operator") == "Exists":
        return toleration.get("key", "") == "" or toleration.get("key") == taint.get("key")
    return (
        toleration.get("key") == taint.get("key")
        and toleration.get("value", "") == taint.get("value", "")
        and toleration.get("effect", "") == taint.get("effect", "")
    )


def _resources_fit(node: FakeNode, requests: dict[str, int]) -> bool:
    avail = node.available()
    if requests.get("cpu", 0) > avail.cpu_millicores:
        return False
    if requests.get("memory", 0) > avail.memory_bytes:
        return False
    if requests.get("nvidia.com/gpu", 0) > avail.gpu_count:
        return False
    if requests.get("ephemeral-storage", 0) > avail.ephemeral_storage_bytes:
        return False
    return True


def _commit_resources(node: FakeNode, requests: dict[str, int]) -> None:
    """Add requested resources to the node's committed totals."""
    node.committed.cpu_millicores += requests.get("cpu", 0)
    node.committed.memory_bytes += requests.get("memory", 0)
    node.committed.gpu_count += requests.get("nvidia.com/gpu", 0)
    node.committed.ephemeral_storage_bytes += requests.get("ephemeral-storage", 0)


def _release_resources(node: FakeNode, requests: dict[str, int]) -> None:
    """Remove requested resources from the node's committed totals."""
    node.committed.cpu_millicores = max(0, node.committed.cpu_millicores - requests.get("cpu", 0))
    node.committed.memory_bytes = max(0, node.committed.memory_bytes - requests.get("memory", 0))
    node.committed.gpu_count = max(0, node.committed.gpu_count - requests.get("nvidia.com/gpu", 0))
    node.committed.ephemeral_storage_bytes = max(
        0, node.committed.ephemeral_storage_bytes - requests.get("ephemeral-storage", 0)
    )


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


# ---------------------------------------------------------------------------
# K8sServiceImpl
# ---------------------------------------------------------------------------


class K8sServiceImpl:
    """In-memory K8sService for DRY_RUN/LOCAL testing.

    Validates manifests, tracks state, supports failure injection.
    Includes a simplified scheduler with node/resource/taint matching.
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
        self._exec_responses: dict[str, list[subprocess.CompletedProcess[str]]] = {}
        self._file_contents: dict[tuple[str, str], bytes] = {}  # (pod_name, path) -> data
        self._rm_files_calls: list[tuple[str, list[str]]] = []
        self._top_pod_overrides: dict[str, tuple[int, int] | None] = {}

        # Node model
        self._nodes: dict[str, FakeNode] = {}
        self._node_pools: dict[str, NodePoolConfig] = {}
        # Track which node each pod was scheduled on + its resource requests
        self._pod_node_assignments: dict[str, str] = {}  # pod_name -> node_name
        self._pod_resource_requests: dict[str, dict[str, int]] = {}  # pod_name -> requests

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

    # -- Scheduling --

    def _schedule_pod(self, manifest: dict) -> None:
        """Run simplified scheduling: match labels, tolerations, resources."""
        spec = _pod_spec(manifest)
        if spec is None:
            return

        # If no nodes exist, skip scheduling (backward compat: tests that don't
        # set up nodes get manifests stored as-is without status changes).
        if not self._nodes:
            return

        node_selector = spec.get("nodeSelector", {})
        tolerations = spec.get("tolerations", [])
        requests = _extract_resource_requests(spec)
        pod_name = manifest["metadata"]["name"]

        for node in self._nodes.values():
            if not _node_selector_matches(node, node_selector):
                continue
            if not _tolerations_satisfy_taints(tolerations, node.taints):
                continue
            if not _resources_fit(node, requests):
                continue

            # Schedule on this node
            _commit_resources(node, requests)
            spec["nodeName"] = node.name
            manifest["status"] = {
                "phase": "Running",
                "containerStatuses": [{"name": "task", "state": {"running": {}}}],
            }
            self._pod_node_assignments[pod_name] = node.name
            self._pod_resource_requests[pod_name] = requests
            return

        # No node fits → Pending
        manifest["status"] = {
            "phase": "Pending",
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": "No node matched constraints",
                }
            ],
        }
        self._auto_event(manifest, "FailedScheduling", "No node matched constraints")

    def _auto_event(self, manifest: dict, reason: str, message: str) -> None:
        """Create and store a K8s event dict."""
        name = manifest.get("metadata", {}).get("name", "")
        self._events.append(
            {
                "involvedObject": {
                    "kind": manifest.get("kind", ""),
                    "name": name,
                },
                "reason": reason,
                "message": message,
                "type": "Warning",
                "firstTimestamp": datetime.now(UTC).isoformat(),
            }
        )

    def _release_pod_resources(self, pod_name: str) -> None:
        """Release committed resources when a pod is deleted."""
        node_name = self._pod_node_assignments.pop(pod_name, None)
        requests = self._pod_resource_requests.pop(pod_name, None)
        if node_name and requests and node_name in self._nodes:
            _release_resources(self._nodes[node_name], requests)

    # -- Failure injection --

    def inject_failure(self, operation: str, error: Exception) -> None:
        """Inject a one-shot failure for the next call to *operation*."""
        self._injected_failures[operation] = error

    def clear_failure(self, operation: str) -> None:
        self._injected_failures.pop(operation, None)

    # -- Node pool management --

    def remove_node_pool(self, pool_name: str) -> None:
        """Remove a node pool and all its nodes."""
        if self._available_node_pools is not None:
            self._available_node_pools.discard(pool_name)
        # Remove nodes belonging to this pool
        if pool_name in self._node_pools:
            pool = self._node_pools.pop(pool_name)
            for i in range(pool.node_count):
                node_name = f"{pool_name}-{i}"
                self._nodes.pop(node_name, None)

    def add_node_pool(
        self,
        pool_name: str,
        *,
        node_count: int = 1,
        labels: dict[str, str] | None = None,
        taints: list[dict[str, str]] | None = None,
        resources: FakeNodeResources | None = None,
        instance_type: str = "n1-standard-4",
    ) -> None:
        """Add a node pool with actual FakeNode objects.

        Creates `node_count` nodes named `{pool_name}-{i}`, each with the given
        labels, taints, and resources. Default resources: 4 CPU cores, 16GB RAM, 0 GPUs.
        """
        if self._available_node_pools is None:
            self._available_node_pools = set()
        self._available_node_pools.add(pool_name)

        pool_labels = dict(labels or {})
        pool_labels["cloud.google.com/gke-nodepool"] = pool_name
        pool_taints = list(taints or [])
        per_node = resources or FakeNodeResources()

        config = NodePoolConfig(
            name=pool_name,
            instance_type=instance_type,
            node_count=node_count,
            labels=pool_labels,
            taints=pool_taints,
            per_node_resources=per_node,
        )
        self._node_pools[pool_name] = config

        for i in range(node_count):
            node_name = f"{pool_name}-{i}"
            self._nodes[node_name] = FakeNode(
                name=node_name,
                labels=dict(pool_labels),
                taints=list(pool_taints),
                allocatable=FakeNodeResources(
                    cpu_millicores=per_node.cpu_millicores,
                    memory_bytes=per_node.memory_bytes,
                    gpu_count=per_node.gpu_count,
                    ephemeral_storage_bytes=per_node.ephemeral_storage_bytes,
                ),
            )

    def set_node_count(self, pool_name: str, count: int) -> None:
        """Adjust node count for an existing pool."""
        if pool_name not in self._node_pools:
            raise KubectlError(f"Node pool {pool_name!r} not found")

        config = self._node_pools[pool_name]
        current = config.node_count

        if count > current:
            # Add new nodes
            for i in range(current, count):
                node_name = f"{pool_name}-{i}"
                self._nodes[node_name] = FakeNode(
                    name=node_name,
                    labels=dict(config.labels),
                    taints=list(config.taints),
                    allocatable=FakeNodeResources(
                        cpu_millicores=config.per_node_resources.cpu_millicores,
                        memory_bytes=config.per_node_resources.memory_bytes,
                        gpu_count=config.per_node_resources.gpu_count,
                        ephemeral_storage_bytes=config.per_node_resources.ephemeral_storage_bytes,
                    ),
                )
        elif count < current:
            # Remove excess nodes (from the end)
            for i in range(count, current):
                node_name = f"{pool_name}-{i}"
                self._nodes.pop(node_name, None)

        config.node_count = count

    # -- Pod lifecycle helpers --

    def transition_pod(self, name: str, phase: str, *, exit_code: int | None = None, reason: str | None = None) -> None:
        """Manually override pod status phase and optional container status."""
        key = ("pod", name)
        manifest = self._resources.get(key)
        if manifest is None:
            raise KubectlError(f"Pod {name!r} not found")

        status: dict = {"phase": phase}
        if exit_code is not None or reason is not None:
            container_state: dict = {}
            if phase == "Failed" or phase == "Succeeded":
                terminated: dict = {}
                if exit_code is not None:
                    terminated["exitCode"] = exit_code
                if reason is not None:
                    terminated["reason"] = reason
                container_state["terminated"] = terminated
            status["containerStatuses"] = [{"name": "task", "state": container_state}]

        manifest["status"] = status

    # -- Test helpers --

    def set_logs(self, pod_name: str, text: str) -> None:
        """Pre-populate logs for a pod."""
        self._logs[pod_name] = text

    def add_event(self, event: dict) -> None:
        self._events.append(event)

    def set_exec_response(self, pod_name: str, response: subprocess.CompletedProcess[str]) -> None:
        """Queue an exec response for a pod. Multiple calls queue FIFO responses."""
        self._exec_responses.setdefault(pod_name, []).append(response)

    def set_file_content(self, pod_name: str, path: str, data: bytes) -> None:
        """Pre-populate file content readable via read_file."""
        self._file_contents[(pod_name, path)] = data

    def set_top_pod(self, pod_name: str, result: tuple[int, int] | None) -> None:
        """Configure a specific top_pod result for a pod."""
        self._top_pod_overrides[pod_name] = result

    # -- Protocol methods --

    def _check_failure(self, operation: str) -> None:
        if err := self._injected_failures.pop(operation, None):
            raise err

    def apply_json(self, manifest: dict) -> None:
        self._check_failure("apply_json")
        self._validate_manifest(manifest)
        kind = _normalize_resource(manifest["kind"])
        name = manifest["metadata"]["name"]
        self._resources[(kind, name)] = manifest

        # Run scheduling for pod-bearing manifests
        if kind == "pod":
            self._schedule_pod(manifest)

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None:
        self._check_failure("get_json")
        return self._resources.get((_normalize_resource(resource), name))

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]:
        self._check_failure("list_json")
        normalized = _normalize_resource(resource)

        # Nodes: merge FakeNode objects and any raw node manifests in _resources
        if normalized == "node":
            results = []
            seen_names: set[str] = set()
            for node in self._nodes.values():
                if labels:
                    if not all(node.labels.get(k) == v for k, v in labels.items()):
                        continue
                results.append(node.to_k8s_dict())
                seen_names.add(node.name)
            # Also include raw node dicts stored via direct _resources manipulation
            for (kind, name), manifest in self._resources.items():
                if kind != "node" or name in seen_names:
                    continue
                if labels:
                    res_labels = manifest.get("metadata", {}).get("labels", {})
                    if not all(res_labels.get(k) == v for k, v in labels.items()):
                        continue
                results.append(manifest)
            return results

        results = []
        for (kind, _), manifest in self._resources.items():
            if kind != normalized:
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
        normalized = _normalize_resource(resource)
        self._resources.pop((normalized, name), None)

        # Release resources when deleting a pod
        if normalized == "pod":
            self._release_pod_resources(name)

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        self._check_failure("logs")
        text = self._logs.get(pod_name, "")
        # tail <= 0 means "all lines"; tail > 0 means "last N lines"
        if tail > 0 and text:
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
        remaining = raw[byte_offset:].decode("utf-8")
        lines = [
            KubectlLogLine(timestamp=datetime.now(UTC), stream="stdout", data=line)
            for line in remaining.splitlines()
            if line
        ]
        return KubectlLogResult(lines=lines, byte_offset=len(raw))

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self._check_failure("exec")
        # Return queued response if available
        if self._exec_responses.get(pod_name):
            return self._exec_responses[pod_name].pop(0)
        if not any(name == pod_name for (_, name) in self._resources):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr=f"pod {pod_name!r} not found")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None:
        self._check_failure("set_image")
        manifest = self._resources.get((_normalize_resource(resource), name))
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
        if pod_name in self._top_pod_overrides:
            return self._top_pod_overrides[pod_name]
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
        key = (pod_name, path)
        if key in self._file_contents:
            return self._file_contents[key]
        raise RuntimeError(f"read_file: no content for {pod_name}:{path}")

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None:
        self._check_failure("rm_files")
        self._rm_files_calls.append((pod_name, paths))
