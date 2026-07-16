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
# One NVL72 rack is also exactly one NVLink domain (RACK_SIZE nodes share the rack's NVLink
# switch), so RACK_SIZE doubles as the hard upper bound on a single-nvlink.domain gang.
RACK_SIZE = 18
NVL72_INSTANCE_PREFIXES = ("gb200", "gb300")


def is_rack_based(instance_type: str) -> bool:
    """True if `instance_type` is an NVL72 SKU that deploys in whole racks (spec.targetRacks)."""
    return instance_type.lower().startswith(NVL72_INSTANCE_PREFIXES)


# Kueue coscheduling group_by keys: the topology *levels* a multi-node GPU gang can bind to.
# They name levels in the infiniband / multinode-nvlink-ib Topology CRs (kueue_manifests.py)
# and are the keys of the K8s task provider's topology map (_CW_DEFAULT_TOPOLOGIES). Declared
# here so the CLI (which picks a level per job), the provider (which maps a level to its node
# label + hard/soft mode), and the installer all share one vocabulary.
COSCHEDULE_LEAFGROUP = "leafgroup"
COSCHEDULE_NVLINK_DOMAIN = "nvlink.domain"
# Soft variant that binds the SAME nvlink.domain label as a preference rather than a hard
# requirement, for a GB200 gang too large to fit one rack. Kueue packs the replicas into as
# few whole NVLink domains (racks) as possible, so GPUs within a rack keep NVLink while the
# gang spills across racks over InfiniBand.
COSCHEDULE_NVLINK_DOMAIN_PREFERRED = "nvlink.domain.preferred"


def gpu_gang_coscheduling_level(gpu_variant: str, replicas: int) -> str:
    """The Kueue topology level a multi-node GPU gang of ``replicas`` nodes should bind to.

    NVL72 (GB200/GB300) nodes carry ``ds.coreweave.com/nvlink.domain`` and one rack is a
    single NVLink domain of ``RACK_SIZE`` nodes. A gang that fits inside one rack binds HARD
    to ``nvlink.domain`` (``podset-required-topology``) so every replica shares the rack's
    NVLink fabric — the reason NVL72 exists. A gang larger than one rack cannot fit a single
    NVLink domain (NVLink does not cross racks), so it binds SOFT to the same level
    (``nvlink.domain.preferred`` -> ``podset-preferred-topology``): Kueue packs the replicas
    into as few whole NVLink domains as possible, keeping NVLink within each rack while the
    gang spills across racks over InfiniBand.

    H100 and every non-NVL72 GPU carry no ``nvlink.domain`` label, so they always coschedule
    on ``leafgroup`` (soft IB colocation), which is the behavior this preserves for them.
    """
    if is_rack_based(gpu_variant):
        if replicas <= RACK_SIZE:
            return COSCHEDULE_NVLINK_DOMAIN
        return COSCHEDULE_NVLINK_DOMAIN_PREFERRED
    return COSCHEDULE_LEAFGROUP
