# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project Iris `scale_groups` onto CoreWeave NodePool specs.

Naming, labels, and rack-based derivation come from the shared
`iris.cluster.platforms.k8s.nodepool_manifests` builders, so IaC and any imperative
caller (e.g. the GPU gang smoke harness) render identically.
"""

from dataclasses import dataclass

from iris.cluster.config import IrisClusterConfig
from iris.cluster.platforms.k8s.nodepool_manifests import (
    compute_target_racks,
    nodepool_name,
    nodepool_node_labels,
)


@dataclass(frozen=True)
class NodePoolSpec:
    """One CoreWeave NodePool projected from an Iris scale group.

    A rack-based (NVL72) pool sets `target_racks` and `autoscaling=False`; a node-based pool
    leaves `target_racks=None` and autoscales within `[min_nodes, max_nodes]`.
    """

    name: str
    instance_type: str
    min_nodes: int
    max_nodes: int
    node_labels: dict[str, str]
    autoscaling: bool = True
    target_racks: int | None = None


def derive_nodepools(config: IrisClusterConfig) -> list[NodePoolSpec]:
    """Return one NodePoolSpec per scale group that defines a CoreWeave slice_template.

    Scale groups without a CoreWeave `instance_type` are skipped, not errored.
    """
    label_prefix = config.platform.label_prefix
    specs: list[NodePoolSpec] = []
    for name, scale_group in config.scale_groups.items():
        template = scale_group.slice_template
        coreweave = template.coreweave if template is not None else None
        if coreweave is None or not coreweave.instance_type:
            continue
        num_vms = max(1, template.num_vms)
        min_nodes = scale_group.buffer_slices * num_vms
        max_nodes = scale_group.max_slices * num_vms
        target_racks = compute_target_racks(coreweave.instance_type, max_nodes, name)
        specs.append(
            NodePoolSpec(
                name=nodepool_name(label_prefix, name),
                instance_type=coreweave.instance_type,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                node_labels=nodepool_node_labels(label_prefix, name, min_nodes=min_nodes),
                autoscaling=target_racks is None,
                target_racks=target_racks,
            )
        )
    return specs
