# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project Iris `scale_groups` onto CoreWeave NodePool specs.

Byte-compatible with the manifest Iris builds today in
`iris.cluster.platforms.k8s.controller` (`ensure_nodepools` / `_ensure_one_nodepool`):
one NodePool per scale group that has a CoreWeave `slice_template`, names normalized to
RFC 1123, node counts multiplied by `num_vms`, and the iris managed/scale-group labels
reproduced so Iris scheduling and `cluster status` keep working after the cede.
"""

from dataclasses import dataclass

from iris.cluster.config import IrisClusterConfig
from iris.cluster.platforms.k8s.coreweave_topology import RACK_SIZE, is_rack_based
from iris.cluster.platforms.types import Labels

# NodePool spec.nodeLabels key that pins system pods (Konnectivity, monitoring) to
# always-on nodes so GPU pools can scale to zero. Applied only when min_nodes > 0.
SYSTEM_CRITICAL_LABEL = "cks.coreweave.cloud/system-critical"


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


def _nodepool_name(label_prefix: str, scale_group: str) -> str:
    # metadata.name must be a valid RFC 1123 subdomain (lowercase, '-', '.').
    return f"{label_prefix}-{scale_group}".replace("_", "-").lower()


def derive_nodepools(config: IrisClusterConfig) -> list[NodePoolSpec]:
    """Return one NodePoolSpec per scale group that defines a CoreWeave slice_template.

    Scale groups without a CoreWeave `instance_type` are skipped (mirroring
    `ensure_nodepools`), not errored.
    """
    label_prefix = config.platform.label_prefix
    labels = Labels(label_prefix)
    specs: list[NodePoolSpec] = []
    for name, scale_group in config.scale_groups.items():
        template = scale_group.slice_template
        coreweave = template.coreweave if template is not None else None
        if coreweave is None or not coreweave.instance_type:
            continue
        num_vms = max(1, template.num_vms)
        min_nodes = scale_group.buffer_slices * num_vms
        max_nodes = scale_group.max_slices * num_vms
        node_labels = {
            labels.iris_managed: "true",
            labels.iris_scale_group: name,
        }
        if min_nodes > 0:
            node_labels[SYSTEM_CRITICAL_LABEL] = "true"

        rack_based = is_rack_based(coreweave.instance_type)
        target_racks: int | None = None
        if rack_based:
            if max_nodes % RACK_SIZE != 0:
                raise ValueError(
                    f"scale group {name!r} is a rack-based ({coreweave.instance_type}) NVL72 pool, "
                    f"so its node count must be a whole number of {RACK_SIZE}-node racks; got {max_nodes}"
                )
            target_racks = max_nodes // RACK_SIZE
        specs.append(
            NodePoolSpec(
                name=_nodepool_name(label_prefix, name),
                instance_type=coreweave.instance_type,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                node_labels=node_labels,
                autoscaling=not rack_based,
                target_racks=target_racks,
            )
        )
    return specs
