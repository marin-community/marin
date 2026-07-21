# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure CoreWeave NodePool manifest builders and their constants.

One source of truth for NodePool naming/labels/shape: the IaC component
(``iac.nodepools`` / ``iac.coreweave.cluster``) and any imperative caller (e.g. the GPU
gang smoke harness, ``tests/e2e/gpu_gang_smoke.py``) import these so they render
byte-identical manifests. Everything here is pure — the functions return plain dicts
and do no I/O; callers own reconciling the live ``targetNodes``/``targetRacks`` value
(a persistent IaC-managed cluster and a one-shot ephemeral test fixture want different
reconcile policies for that one field — see ``build_nodepool_manifest``).
"""

from iris.cluster.platforms.k8s.coreweave_topology import RACK_SIZE, is_rack_based
from iris.cluster.platforms.types import Labels

# NodePool spec.nodeLabels key that pins system pods (Konnectivity, monitoring) to
# always-on nodes so GPU pools can scale to zero. Applied only when min_nodes > 0.
SYSTEM_CRITICAL_LABEL = "cks.coreweave.cloud/system-critical"


def nodepool_name(label_prefix: str, scale_group: str) -> str:
    """NodePool metadata.name must be a valid RFC 1123 subdomain (lowercase, '-', '.')."""
    return f"{label_prefix}-{scale_group}".replace("_", "-").lower()


def nodepool_node_labels(label_prefix: str, scale_group: str, *, min_nodes: int) -> dict[str, str]:
    """The managed + scale-group labels a NodePool's spec.nodeLabels carries."""
    labels = Labels(label_prefix)
    node_labels = {labels.iris_managed: "true", labels.iris_scale_group: scale_group}
    if min_nodes > 0:
        node_labels[SYSTEM_CRITICAL_LABEL] = "true"
    return node_labels


def compute_target_racks(instance_type: str, max_nodes: int, scale_group_name: str) -> int | None:
    """None for a node-based instance type; else max_nodes // RACK_SIZE, validated divisible.

    GB200/GB300 NVL72 instances deploy in whole racks: CoreWeave rejects the autoscaler
    and a partial rack on rack-based instance types.
    """
    if not is_rack_based(instance_type):
        return None
    if max_nodes % RACK_SIZE != 0:
        raise ValueError(
            f"scale group {scale_group_name!r} is a rack-based ({instance_type}) NVL72 pool, "
            f"so its node count must be a whole number of {RACK_SIZE}-node racks; got {max_nodes}"
        )
    return max_nodes // RACK_SIZE


def build_nodepool_manifest(
    pool_name: str,
    instance_type: str,
    *,
    node_labels: dict[str, str],
    min_nodes: int = 0,
    max_nodes: int = 0,
    target_nodes: int = 0,
    target_racks: int | None = None,
    autoscaling: bool = True,
) -> dict:
    """Build one CoreWeave NodePool manifest.

    Rack-based (NVL72) pools declare only spec.targetRacks (no autoscaler envelope) —
    pass ``target_racks`` (from ``compute_target_racks``) and ``target_nodes``/
    ``min_nodes``/``max_nodes``/``autoscaling`` are ignored. Node-based pools declare
    the [minNodes, maxNodes] autoscaler envelope plus spec.targetNodes, which the CRD
    requires at create; callers compute ``target_nodes`` themselves (a persistent
    cluster seeds it once and lets ``ignore_changes``/CoreWeave's autoscaler own it
    thereafter; a one-shot ephemeral pool can just seed 0 and patch it after create).
    """
    metadata_labels = {k: v for k, v in node_labels.items() if k != SYSTEM_CRITICAL_LABEL}
    spec: dict = {"computeClass": "default", "instanceType": instance_type, "nodeLabels": node_labels}
    if target_racks is not None:
        spec["targetRacks"] = target_racks
    else:
        spec["autoscaling"] = autoscaling
        spec["minNodes"] = min_nodes
        spec["maxNodes"] = max_nodes
        spec["targetNodes"] = target_nodes
    return {
        "apiVersion": "compute.coreweave.com/v1alpha1",
        "kind": "NodePool",
        "metadata": {"name": pool_name, "labels": metadata_labels},
        "spec": spec,
    }
