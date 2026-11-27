#!/usr/bin/env python3
"""Resolve Iris GPU gang resources from live node and pod admission data."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml


AUTOMATIC_RESOURCE_REQUEST = "auto"
MEMORY_RESOURCE = "memory"
DISK_RESOURCE = "ephemeral-storage"
# Leave the remainder of live allocatable RAM and disk to kubelet, daemonsets,
# and filesystem overhead.
NODE_RESOURCE_FRACTION = 0.80


def _parse_quantity_to_gib(quantity: str) -> float:
    """Parse a Kubernetes or Iris byte quantity to GiB."""
    quantity = quantity.strip()
    for suf, mult in (("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("Pi", 2**50)):
        if quantity.endswith(suf):
            return float(quantity[: -len(suf)]) * mult / 2**30
    for suf, mult in (
        ("KB", 1e3),
        ("MB", 1e6),
        ("GB", 1e9),
        ("TB", 1e12),
        ("PB", 1e15),
        ("k", 1e3),
        ("M", 1e6),
        ("G", 1e9),
        ("T", 1e12),
        ("P", 1e15),
    ):
        if quantity.endswith(suf):
            return float(quantity[: -len(suf)]) * mult / 2**30
    return float(quantity) / 2**30  # plain bytes


class KubernetesMetadata(TypedDict, total=False):
    name: str
    labels: dict[str, str]


class KubernetesResources(TypedDict, total=False):
    requests: dict[str, str]


class KubernetesContainer(TypedDict, total=False):
    resources: KubernetesResources


class KubernetesSpec(TypedDict, total=False):
    nodeName: str
    unschedulable: bool
    containers: list[KubernetesContainer]
    initContainers: list[KubernetesContainer]
    overhead: dict[str, str]


class KubernetesCondition(TypedDict, total=False):
    type: str
    status: str


class KubernetesStatus(TypedDict, total=False):
    phase: str
    conditions: list[KubernetesCondition]
    allocatable: dict[str, str]


class KubernetesRecord(TypedDict, total=False):
    kind: str
    metadata: KubernetesMetadata
    spec: KubernetesSpec
    status: KubernetesStatus


@dataclass(frozen=True)
class ResourceQuantities:
    """Per-node memory and disk quantities in GiB."""

    memory_gib: float
    disk_gib: float


@dataclass(frozen=True)
class NodeResourceBudget:
    """Live headroom and policy-capped automatic requests for one node."""

    headroom: ResourceQuantities
    automatic: ResourceQuantities


@dataclass(frozen=True)
class ClusterResourceSnapshot:
    """Available matching nodes observed in one Kubernetes query."""

    context: str
    gpu_variant: str
    gpus_per_node: int
    nodes: tuple[NodeResourceBudget, ...]


@dataclass(frozen=True)
class ResolvedResourceRequests:
    """Per-node memory and disk requests ready for Iris submission."""

    memory: str
    disk: str


def _pod_resource_request_gib(pod: KubernetesRecord, resource: str) -> float:
    """Return the scheduler-effective request for one pod resource."""
    spec = pod.get("spec", {})

    def request(container: KubernetesContainer) -> float:
        quantity = container.get("resources", {}).get("requests", {}).get(resource)
        return _parse_quantity_to_gib(quantity) if isinstance(quantity, str) else 0.0

    application_request = sum(request(container) for container in spec.get("containers", []))
    init_request = max((request(container) for container in spec.get("initContainers", [])), default=0.0)
    overhead = spec.get("overhead", {}).get(resource)
    overhead_request = _parse_quantity_to_gib(overhead) if isinstance(overhead, str) else 0.0
    return max(application_request, init_request) + overhead_request


def _node_is_ready(node: KubernetesRecord) -> bool:
    conditions = node.get("status", {}).get("conditions", [])
    return any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions)


def _node_resource_budgets(
    items: list[KubernetesRecord],
    *,
    gpu_variant: str,
    gpus_per_node: int,
) -> list[NodeResourceBudget]:
    """Return live headroom and automatic request budgets for matching Ready nodes."""
    pod_requests: dict[str, ResourceQuantities] = {}
    for item in items:
        if item.get("kind") != "Pod" or item.get("status", {}).get("phase") in {"Succeeded", "Failed"}:
            continue
        node_name = item.get("spec", {}).get("nodeName")
        if not isinstance(node_name, str):
            continue
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        pod_requests[node_name] = ResourceQuantities(
            memory_gib=requested.memory_gib + _pod_resource_request_gib(item, MEMORY_RESOURCE),
            disk_gib=requested.disk_gib + _pod_resource_request_gib(item, DISK_RESOURCE),
        )

    available = []
    for item in items:
        if item.get("kind") != "Node" or item.get("spec", {}).get("unschedulable") or not _node_is_ready(item):
            continue
        metadata = item.get("metadata", {})
        allocatable = item.get("status", {}).get("allocatable", {})
        product = metadata.get("labels", {}).get("nvidia.com/gpu.product", "")
        if allocatable.get("nvidia.com/gpu") != str(gpus_per_node) or gpu_variant.lower() not in product.lower():
            continue
        node_name = metadata.get("name")
        if not isinstance(node_name, str):
            continue
        memory_allocatable = _parse_quantity_to_gib(allocatable[MEMORY_RESOURCE])
        disk_allocatable = _parse_quantity_to_gib(allocatable[DISK_RESOURCE])
        requested = pod_requests.get(node_name, ResourceQuantities(memory_gib=0.0, disk_gib=0.0))
        headroom = ResourceQuantities(
            memory_gib=memory_allocatable - requested.memory_gib,
            disk_gib=disk_allocatable - requested.disk_gib,
        )
        available.append(
            NodeResourceBudget(
                headroom=headroom,
                automatic=ResourceQuantities(
                    memory_gib=min(memory_allocatable * NODE_RESOURCE_FRACTION, headroom.memory_gib),
                    disk_gib=min(disk_allocatable * NODE_RESOURCE_FRACTION, headroom.disk_gib),
                ),
            )
        )
    return available


def _inspect_cluster_resources(
    cluster_config_path: str, *, gpu_variant: str, gpus_per_node: int
) -> ClusterResourceSnapshot:
    """Read one live node-and-pod resource snapshot from the selected cluster."""
    cluster_config = _load_cluster_config(cluster_config_path)
    platform = cluster_config.get("platform")
    coreweave = platform.get("coreweave") if isinstance(platform, dict) else None
    kubeconfig = coreweave.get("kubeconfig_path") if isinstance(coreweave, dict) else None
    context = coreweave.get("kube_context") if isinstance(coreweave, dict) else None
    if not isinstance(kubeconfig, str) or not isinstance(context, str):
        raise SystemExit(
            "The selected cluster config needs platform.coreweave.kubeconfig_path and kube_context "
            "to derive automatic memory and disk requests; pass both explicitly."
        )

    try:
        out = subprocess.run(
            [
                "kubectl",
                "--kubeconfig",
                str(Path(kubeconfig).expanduser()),
                "--context",
                context,
                "get",
                "nodes,pods",
                "--all-namespaces",
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - convert cluster/tool failures into launch guidance
        raise SystemExit(
            f"Could not inspect {gpus_per_node}x{gpu_variant} nodes in kube context {context!r}: {exc}. "
            "Pass explicit --memory and --disk values."
        ) from exc
    items = json.loads(out)["items"]
    if not isinstance(items, list):
        raise ValueError("kubectl node-and-pod snapshot has a non-list items field")
    nodes = _node_resource_budgets(cast(list[KubernetesRecord], items), gpu_variant=gpu_variant, gpus_per_node=gpus_per_node)
    return ClusterResourceSnapshot(
        context=context,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
        nodes=tuple(nodes),
    )


def _resource_request_is_automatic(request: str) -> bool:
    return request.strip().lower() == AUTOMATIC_RESOURCE_REQUEST


def _resolve_gang_resource_requests(
    snapshot: ClusterResourceSnapshot,
    *,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Select one resource pair that fits the requested gang in a live snapshot."""
    automatic_memory = _resource_request_is_automatic(memory_request)
    automatic_disk = _resource_request_is_automatic(disk_request)
    requested_memory_gib = None if automatic_memory else _parse_quantity_to_gib(memory_request)
    requested_disk_gib = None if automatic_disk else _parse_quantity_to_gib(disk_request)
    candidates = [
        resources
        for resources in snapshot.nodes
        if (requested_memory_gib is None or resources.headroom.memory_gib >= requested_memory_gib)
        and (requested_disk_gib is None or resources.headroom.disk_gib >= requested_disk_gib)
    ]
    if len(candidates) < num_nodes:
        raise SystemExit(
            f"Automatic resources need {num_nodes} Ready, schedulable "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes, "
            f"but only {len(candidates)} of {len(snapshot.nodes)} matching nodes in kube context "
            f"{snapshot.context!r} fit the constraints memory={memory_request}, disk={disk_request}; "
            "adjust the resource requests."
        )

    if automatic_memory:
        selected = sorted(candidates, key=lambda resources: resources.automatic.memory_gib, reverse=True)[:num_nodes]
    else:
        selected = sorted(candidates, key=lambda resources: resources.automatic.disk_gib, reverse=True)[:num_nodes]
    memory_gib = int(min(resources.automatic.memory_gib for resources in selected))
    disk_gib = int(min(resources.automatic.disk_gib for resources in selected))
    if (automatic_memory and memory_gib < 1) or (automatic_disk and disk_gib < 1):
        automatic_resources = " and ".join(
            resource for resource, automatic in (("memory", automatic_memory), ("disk", automatic_disk)) if automatic
        )
        raise SystemExit(
            f"No positive automatic {automatic_resources} request fits {num_nodes} "
            f"{snapshot.gpus_per_node}x{snapshot.gpu_variant} nodes "
            f"in kube context {snapshot.context!r}; pass explicit --memory and --disk values."
        )
    resolved_memory = f"{memory_gib}Gi" if automatic_memory else memory_request
    resolved_disk = f"{disk_gib}Gi" if automatic_disk else disk_request
    return ResolvedResourceRequests(memory=resolved_memory, disk=resolved_disk)


def resolve_node_resource_requests(
    cluster_config_path: str,
    *,
    gpu_variant: str,
    gpus_per_node: int,
    num_nodes: int,
    memory_request: str,
    disk_request: str,
) -> ResolvedResourceRequests:
    """Return live admission-aware resource requests for a GPU gang."""
    snapshot = _inspect_cluster_resources(
        cluster_config_path,
        gpu_variant=gpu_variant,
        gpus_per_node=gpus_per_node,
    )
    resolved = _resolve_gang_resource_requests(
        snapshot,
        num_nodes=num_nodes,
        memory_request=memory_request,
        disk_request=disk_request,
    )
    print(
        f"[rl-iris] automatic node resources: largest requests with live headroom on {num_nodes} of "
        f"{len(snapshot.nodes)} matching nodes, capped at {NODE_RESOURCE_FRACTION:.0%} allocatable = "
        f"memory {resolved.memory}, disk {resolved.disk}",
        flush=True,
    )
    return resolved


def _load_cluster_config(cluster_config: str) -> dict[str, Any]:
    """Load the selected Iris cluster configuration for launch-time defaults."""
    try:
        with open(cluster_config) as f:
            loaded = yaml.safe_load(f)
    except OSError as exc:
        raise SystemExit(
            f"Could not load --cluster-config {cluster_config!r} to resolve RL launch defaults: {exc}. "
            "Pass an existing cluster config or explicit --cpu and --rendezvous-dir values."
        ) from exc
    if not isinstance(loaded, dict):
        raise SystemExit(f"--cluster-config {cluster_config!r} must contain a YAML mapping.")
    return loaded
