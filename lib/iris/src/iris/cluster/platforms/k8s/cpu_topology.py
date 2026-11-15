# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preview or repair live CPU topology labels without replacing NodePools."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import click

from iris.cluster.config import IrisClusterConfig, load_config
from iris.cluster.platforms.k8s.nodepool_manifests import CPU_TOPOLOGY_NODE_LABELS, nodepool_name
from iris.cluster.platforms.k8s.service import CloudK8sService, K8sService
from iris.cluster.platforms.k8s.types import K8sResource, parse_k8s_quantity
from iris.cluster.platforms.types import Labels
from iris.cluster.types import AcceleratorType


@dataclass(frozen=True)
class CpuTopologyRepair:
    """A label-only patch fenced to the inspected Node revision."""

    name: str
    uid: str
    resource_version: str
    labels: dict[str, str]


def plan_cpu_topology_repairs(
    config: IrisClusterConfig, nodes: Sequence[dict], nodepools: Sequence[dict]
) -> list[CpuTopologyRepair]:
    """Compare managed CPU Nodes with their declared synthetic topology.

    Reject mismatched pool declarations or accelerator-bearing Nodes before
    producing any repairs. Only the four synthetic topology labels may change.
    """
    labels = Labels(config.platform.label_prefix)
    pools = {pool["metadata"]["name"]: pool for pool in nodepools}
    desired = dict(CPU_TOPOLOGY_NODE_LABELS)
    repairs = []
    for group_name, group in config.scale_groups.items():
        if group.resources is None or group.resources.device_type != AcceleratorType.CPU:
            continue
        template = group.slice_template
        if template is None or template.coreweave is None:
            continue
        pool_name = nodepool_name(config.platform.label_prefix, group_name)
        pool = pools[pool_name]
        declared = pool["spec"]["nodeLabels"]
        if any(declared.get(key) != value for key, value in desired.items()):
            raise ValueError(f"NodePool {pool_name} does not declare the synthetic CPU topology; repair IaC first")
        for node in nodes:
            metadata = node["metadata"]
            actual = metadata.get("labels", {})
            if actual.get(labels.iris_managed) != "true" or actual.get(labels.iris_scale_group) != group_name:
                continue
            allocatable = node.get("status", {}).get("allocatable", {})
            if actual.get("node.kubernetes.io/instance-type") != template.coreweave.instance_type or any(
                parse_k8s_quantity(allocatable.get(key, "0")) > 0 for key in ("nvidia.com/gpu", "rdma/ib")
            ):
                raise ValueError(f"Node {metadata['name']} does not match CPU NodePool {pool_name}")
            changes = {key: value for key, value in desired.items() if actual.get(key) != value}
            if changes:
                repairs.append(
                    CpuTopologyRepair(metadata["name"], metadata["uid"], metadata["resourceVersion"], changes)
                )
    return repairs


def apply_cpu_topology_repairs(k8s: K8sService, repairs: Sequence[CpuTopologyRepair]) -> None:
    """Apply inspected patches, failing on a concurrent Node change or failed readback."""
    for repair in repairs:
        k8s.patch_node(
            repair.name,
            {
                "metadata": {
                    "uid": repair.uid,
                    "resourceVersion": repair.resource_version,
                    "labels": repair.labels,
                }
            },
        )
        node = k8s.get_json(K8sResource.NODES, repair.name)
        if node is None or any(node["metadata"]["labels"].get(k) != v for k, v in repair.labels.items()):
            raise RuntimeError(f"CPU topology label readback failed for Node {repair.name}")


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--apply", is_flag=True, help="Patch live Nodes; otherwise only print the proposed repairs.")
def main(config_path: Path, apply: bool) -> None:
    """Reconcile existing CoreWeave CPU Nodes with the NodePool topology declaration."""
    config = load_config(config_path)
    platform = config.platform.coreweave
    if platform is None or not platform.kube_context or not platform.kubeconfig_path:
        raise click.ClickException("The config must name an explicit CoreWeave kubeconfig and context")
    k8s = CloudK8sService(
        namespace=platform.namespace, kubeconfig_path=platform.kubeconfig_path, context=platform.kube_context
    )
    nodes = k8s.list_json(K8sResource.NODES, labels={Labels(config.platform.label_prefix).iris_managed: "true"})
    if not nodes:
        raise click.ClickException("No managed Nodes found; check the cluster config and Kubernetes context")
    repairs = plan_cpu_topology_repairs(config, nodes, k8s.list_json(K8sResource.NODE_POOLS))
    click.echo(json.dumps({"context": platform.kube_context, "repairs": [asdict(r) for r in repairs]}, indent=2))
    if apply:
        apply_cpu_topology_repairs(k8s, repairs)
        click.echo(f"Repaired and verified {len(repairs)} CPU Nodes")


if __name__ == "__main__":
    main()
