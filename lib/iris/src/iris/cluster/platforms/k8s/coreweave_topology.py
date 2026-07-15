# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical CoreWeave node-label keys for Kueue topology-aware scheduling.

These keys are the levels in CoreWeave's Kueue Topology CRs and the node
selectors that three sites must agree on: the K8s task provider stamps
``podset-{required,preferred}-topology`` annotations naming them
(``providers/k8s/tasks.py``), the install script declares them as Topology
levels + ResourceFlavor selectors (``scripts/install_kueue.py``), and the kind
smoke stamps them onto synthetic nodes so TAS resolves the same layout it would
on a real CKS cluster (``tests/e2e/gpu_gang_smoke.py``). Declared once here so
those three sites cannot drift.

Names leak CoreWeave conventions by design: ``group_by`` reflects the actual
topology the gang runs against, it is not a portable abstraction.
"""

# InfiniBand fabric hierarchy, coarse -> fine. A leafgroup is one IB
# leaf-switch group (soft/preferred multi-node colocation); superpod and fabric
# are the wider domains above it.
CW_LABEL_FABRIC = "backend.coreweave.cloud/fabric"
CW_LABEL_SUPERPOD = "backend.coreweave.cloud/superpod"
CW_LABEL_LEAFGROUP = "backend.coreweave.cloud/leafgroup"

# Per-flavor capacity selector. Every IB-fabric node carries
# ``backend.coreweave.cloud/flavor=infiniband``; the cw-ib ResourceFlavor
# selects on it and the kind smoke stamps it so the flavor matches.
CW_LABEL_FLAVOR = "backend.coreweave.cloud/flavor"
CW_FLAVOR_INFINIBAND = "infiniband"

# GB200 NVLink domain (hard/required single-domain colocation). H100 nodes do
# NOT carry this label, so an H100 IB deployment has no nvlink.domain level.
CW_LABEL_NVLINK_DOMAIN = "ds.coreweave.com/nvlink.domain"

# NVL72 (GB200/GB300) instances deploy in whole racks of 18 nodes. Such NodePools are
# declared by rack count (spec.targetRacks) and do NOT autoscale — CoreWeave rejects both a
# partial rack and the autoscaler on rack-based instances. Every other instance type is
# node-based (spec.targetNodes + autoscaling). See docs.coreweave.com nvl72 instance docs.
RACK_SIZE = 18
NVL72_INSTANCE_PREFIXES = ("gb200", "gb300")


def is_rack_based(instance_type: str) -> bool:
    """True if `instance_type` is an NVL72 SKU that deploys in whole racks (spec.targetRacks)."""
    return instance_type.lower().startswith(NVL72_INSTANCE_PREFIXES)
