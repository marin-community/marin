# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-memory K8sService for DRY_RUN/LOCAL testing.

Validates manifests, tracks state, supports failure injection.
Includes a simplified scheduler with node/resource/taint matching.

In LOCAL mode, pod commands are actually executed via subprocess in
background threads, with stdout/stderr captured as pod logs and pod
status updated automatically on completion.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from iris.cluster.service_mode import ServiceMode

from iris.cluster.providers.k8s.types import (
    ExecResult,
    K8sResource,
    KubectlError,
    KubectlLogLine,
    KubectlLogResult,
    PodResourceUsage,
    parse_k8s_quantity,
)

# Resource types that K8s recognizes in container resource requests/limits.
logger = logging.getLogger(__name__)

VALID_RESOURCE_TYPES = frozenset(
    {
        "cpu",
        "memory",
        "nvidia.com/gpu",
        "google.com/tpu",
        "ephemeral-storage",
        "rdma/ib",
    }
)


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


def _extract_resource_requests(spec: dict) -> dict[str, int]:
    """Sum resource requests across all containers in a pod spec."""
    totals: dict[str, int] = {}
    for container in spec.get("containers", []):
        reqs = container.get("resources", {}).get("requests", {})
        limits = container.get("resources", {}).get("limits", {})
        for key in ("cpu", "memory", "nvidia.com/gpu", "ephemeral-storage"):
            val = reqs.get(key) or limits.get(key)
            if val:
                totals[key] = totals.get(key, 0) + parse_k8s_quantity(val)
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


def _matches_field_selector(obj: dict, selector: str) -> bool:
    """Minimal field_selector matching for testing (supports = and != operators)."""
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "!=" in part:
            key, value = part.split("!=", 1)
            negate = True
        elif "=" in part:
            key, value = part.split("=", 1)
            negate = False
        else:
            continue
        segments = key.strip().split(".")
        cursor = obj
        for seg in segments:
            if not isinstance(cursor, dict):
                if negate:
                    break
                return False
            cursor = cursor.get(seg)
            if cursor is None:
                if negate:
                    break
                return False
        else:
            matched = str(cursor) == value.strip()
            if negate and matched:
                return False
            if not negate and not matched:
                return False
    return True


# ---------------------------------------------------------------------------
# InMemoryK8sService
# ---------------------------------------------------------------------------


class InMemoryK8sService:
    """In-memory K8sService for DRY_RUN/LOCAL testing.

    Validates manifests, tracks state, supports failure injection.
    Includes a simplified scheduler with node/resource/taint matching.

    In LOCAL mode, pod commands are actually executed via subprocess in
    background threads, with stdout/stderr captured as pod logs and pod
    status updated automatically on completion.
    """

    def __init__(
        self,
        namespace: str = "iris",
        available_node_pools: list[str] | None = None,
        mode: ServiceMode = ServiceMode.DRY_RUN,
    ):
        self._namespace = namespace
        self._mode = mode
        self._executor: ThreadPoolExecutor | None = None
        if mode == ServiceMode.LOCAL:
            self._executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="k8s-local-pod")
        self._available_node_pools: set[str] | None = (
            set(available_node_pools) if available_node_pools is not None else None
        )
        self._resources: dict[tuple[str, str], dict] = {}  # (kind, name) -> manifest
        self._injected_failures: dict[str, Exception] = {}
        self._logs: dict[str, str] = {}  # pod_name -> log text
        self._events: list[dict] = []
        self._exec_responses: dict[str, list[ExecResult]] = {}
        self._file_contents: dict[tuple[str, str], bytes] = {}  # (pod_name, path) -> data
        self._rm_files_calls: list[tuple[str, list[str]]] = []
        self._top_pod_overrides: dict[str, PodResourceUsage | None] = {}
        self._log_watermarks: dict[str, int] = {}  # pod_name -> bytes consumed

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

        try:
            K8sResource.from_kind(kind)
        except ValueError as e:
            raise KubectlError(f"Unknown manifest kind: {kind!r}") from e

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
        """Run simplified scheduling: match labels, tolerations, resources, affinity."""
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

        # NOTE: pod affinity/anti-affinity rules are NOT evaluated here.
        # Use Kind-based integration tests for scheduling correctness (see #3940).

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

        # No node fits -> Pending
        message = "No node matched constraints"
        manifest["status"] = {
            "phase": "Pending",
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "Unschedulable",
                    "message": message,
                }
            ],
        }
        self._auto_event(manifest, "FailedScheduling", message)

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

    # -- LOCAL mode execution --

    def _run_pod_locally(self, pod_name: str, manifest: dict) -> None:
        """In LOCAL mode, execute a pod's command in a background thread."""
        assert self._executor is not None
        spec = _pod_spec(manifest)
        if spec is None:
            return
        containers = spec.get("containers", [])
        if not containers:
            return
        self._executor.submit(self._execute_pod_command, pod_name, manifest)

    def _execute_pod_command(self, pod_name: str, manifest: dict) -> None:
        """Background thread: run a pod's command and update status."""
        spec = _pod_spec(manifest)
        assert spec is not None, "manifest must be a pod"
        container = spec["containers"][0]
        command = container.get("command", []) + container.get("args", [])
        if not command:
            self.transition_pod(pod_name, "Succeeded")
            return

        env = dict(os.environ)
        for e in container.get("env", []):
            if "value" in e:
                env[e["name"]] = e["value"]

        self.transition_pod(pod_name, "Running")

        try:
            proc = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.set_logs(pod_name, proc.stdout + proc.stderr)
            if proc.returncode == 0:
                self.transition_pod(pod_name, "Succeeded")
            else:
                self.transition_pod(pod_name, "Failed", exit_code=proc.returncode, reason="Error")
        except subprocess.TimeoutExpired:
            self.set_logs(pod_name, "Pod execution timed out (300s)")
            self.transition_pod(pod_name, "Failed", exit_code=137, reason="DeadlineExceeded")
        except Exception:
            logger.exception("Failed to execute pod %s locally", pod_name)
            self.transition_pod(pod_name, "Failed", exit_code=1, reason="Error")

    def close(self) -> None:
        """Shut down the LOCAL mode executor."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

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
            for i in range(count, current):
                node_name = f"{pool_name}-{i}"
                self._nodes.pop(node_name, None)

        config.node_count = count

    # -- Pod lifecycle helpers --

    def transition_pod(self, name: str, phase: str, *, exit_code: int | None = None, reason: str | None = None) -> None:
        """Manually override pod status phase and optional container status."""
        key = (K8sResource.PODS.plural, name)
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

    def set_exec_response(self, pod_name: str, response: ExecResult) -> None:
        """Queue an exec response for a pod. Multiple calls queue FIFO responses."""
        self._exec_responses.setdefault(pod_name, []).append(response)

    def set_file_content(self, pod_name: str, path: str, data: bytes) -> None:
        """Pre-populate file content readable via read_file."""
        self._file_contents[(pod_name, path)] = data

    def set_top_pod(self, pod_name: str, result: PodResourceUsage | None) -> None:
        """Configure a specific top_pod result for a pod."""
        self._top_pod_overrides[pod_name] = result

    def seed_resource(self, resource: K8sResource, name: str, manifest: dict) -> None:
        """Directly insert a resource into the in-memory store for test setup.

        Use this to pre-populate pods, nodes, etc. without going through
        apply_json validation and scheduling.
        """
        self._resources[(resource.plural, name)] = manifest

    # -- Protocol methods --

    def _check_failure(self, operation: str) -> None:
        if err := self._injected_failures.pop(operation, None):
            raise err

    def apply_json(self, manifest: dict) -> None:
        self._check_failure("apply_json")
        self._validate_manifest(manifest)
        resource = K8sResource.from_kind(manifest["kind"])
        name = manifest["metadata"]["name"]
        self._resources[(resource.plural, name)] = manifest

        # Run scheduling for pod-bearing manifests
        if resource is K8sResource.PODS:
            self._schedule_pod(manifest)

        if self._mode == ServiceMode.LOCAL and resource is K8sResource.PODS:
            self._run_pod_locally(name, manifest)

    def get_json(self, resource: K8sResource, name: str) -> dict | None:
        self._check_failure("get_json")
        return self._resources.get((resource.plural, name))

    def list_json(
        self,
        resource: K8sResource,
        *,
        labels: dict[str, str] | None = None,
        field_selector: str | None = None,
    ) -> list[dict]:
        self._check_failure("list_json")
        plural = resource.plural
        self._check_failure(f"list_json:{plural}")

        # Nodes: merge FakeNode objects and any raw node manifests in _resources
        if resource is K8sResource.NODES:
            results = []
            seen_names: set[str] = set()
            for node in self._nodes.values():
                if labels:
                    if not all(node.labels.get(k) == v for k, v in labels.items()):
                        continue
                results.append(node.to_k8s_dict())
                seen_names.add(node.name)
            # Also include raw node dicts stored via direct _resources manipulation
            for (stored_plural, name), manifest in self._resources.items():
                if stored_plural != plural or name in seen_names:
                    continue
                if labels:
                    res_labels = manifest.get("metadata", {}).get("labels", {})
                    if not all(res_labels.get(k) == v for k, v in labels.items()):
                        continue
                results.append(manifest)
            return results

        results = []
        for (stored_plural, _), manifest in self._resources.items():
            if stored_plural != plural:
                continue
            if labels:
                res_labels = manifest.get("metadata", {}).get("labels", {})
                if not all(res_labels.get(k) == v for k, v in labels.items()):
                    continue
            if field_selector and not _matches_field_selector(manifest, field_selector):
                continue
            results.append(manifest)
        return results

    def delete(self, resource: K8sResource, name: str, *, force: bool = False, wait: bool = True) -> None:
        self._check_failure("delete")
        self._resources.pop((resource.plural, name), None)

        # Release resources when deleting a pod
        if resource is K8sResource.PODS:
            self._release_pod_resources(name)

    def delete_many(self, resource: K8sResource, names: list[str], *, wait: bool = False) -> None:
        """Delete multiple resources by name."""
        for name in names:
            self.delete(resource, name)

    def delete_by_labels(self, resource: K8sResource, labels: dict[str, str], *, wait: bool = False) -> None:
        """Delete all resources matching the given label selector."""
        self._check_failure("delete_by_labels")
        if not labels:
            return
        plural = resource.plural
        to_delete = []
        for (stored_plural, name), manifest in self._resources.items():
            if stored_plural != plural:
                continue
            res_labels = manifest.get("metadata", {}).get("labels", {})
            if all(res_labels.get(k) == v for k, v in labels.items()):
                to_delete.append(name)
        for name in to_delete:
            self.delete(resource, name)

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
        since_time: datetime | None = None,
        limit_bytes: int | None = None,
    ) -> KubectlLogResult:
        self._check_failure("stream_logs")
        pod_exists = any(name == pod_name for (_, name) in self._resources)
        has_logs = pod_name in self._logs
        if not pod_exists and not has_logs:
            raise KubectlError(f"pod {pod_name!r} not found")
        text = self._logs.get(pod_name, "")
        raw = text.encode("utf-8")
        watermark = self._log_watermarks.get(pod_name, 0) if since_time is not None else 0
        if len(raw) <= watermark:
            return KubectlLogResult(lines=[], last_timestamp=since_time)
        remaining = raw[watermark:].decode("utf-8")
        lines = [
            KubectlLogLine(timestamp=datetime.now(UTC), stream="stdout", data=line)
            for line in remaining.splitlines()
            if line
        ]
        self._log_watermarks[pod_name] = len(raw)
        last_ts = lines[-1].timestamp if lines else since_time
        return KubectlLogResult(lines=lines, last_timestamp=last_ts)

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> ExecResult:
        self._check_failure("exec")
        # Return queued response if available
        if self._exec_responses.get(pod_name):
            return self._exec_responses[pod_name].pop(0)
        if not any(name == pod_name for (_, name) in self._resources):
            return ExecResult(returncode=1, stdout="", stderr=f"pod {pod_name!r} not found")
        return ExecResult(returncode=0, stdout="", stderr="")

    def set_image(self, resource: K8sResource, name: str, container: str, image: str) -> None:
        self._check_failure("set_image")
        manifest = self._resources.get((resource.plural, name))
        if manifest is None:
            raise KubectlError(f"{resource.plural}/{name} not found")

    def rollout_restart(self, resource: K8sResource, name: str) -> None:
        self._check_failure("rollout_restart")

    def rollout_status(self, resource: K8sResource, name: str, *, timeout: float = 600.0) -> None:
        self._check_failure("rollout_status")

    def get_events(self, field_selector: str | None = None) -> list[dict]:
        self._check_failure("get_events")
        if field_selector is None:
            return list(self._events)
        results = []
        for event in self._events:
            if _matches_field_selector(event, field_selector):
                results.append(event)
        return results

    def top_pod(self, pod_name: str) -> PodResourceUsage | None:
        self._check_failure("top_pod")
        if pod_name in self._top_pod_overrides:
            return self._top_pod_overrides[pod_name]
        if any(name == pod_name for (_, name) in self._resources):
            return PodResourceUsage(cpu_millicores=100, memory_bytes=256 * 1024 * 1024)
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
        raise KubectlError(f"read_file: no content for {pod_name}:{path}")

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None:
        self._check_failure("rm_files")
        self._rm_files_calls.append((pod_name, paths))

    @contextmanager
    def port_forward(
        self,
        service_name: str,
        remote_port: int,
        local_port: int | None = None,
        timeout: float = 90.0,
    ) -> Iterator[str]:
        self._check_failure("port_forward")
        port = local_port or 19999
        yield f"http://127.0.0.1:{port}"
