# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical CoreWeave node-label keys for Kueue topology-aware scheduling.

These keys are the levels in CoreWeave's Kueue Topology CRs and the node
selectors that three sites must agree on: the K8s task provider stamps
``podset-{required,preferred,slice-required}-topology`` annotations naming them
(``providers/k8s/tasks.py``), the install script declares them as Topology
levels + ResourceFlavor selectors (``scripts/install_kueue.py``), and the kind
smoke stamps them onto synthetic nodes so TAS resolves the same layout it would
on a real CKS cluster (``tests/e2e/gpu_gang_smoke.py``). Declared once here so
those three sites cannot drift.

Names leak CoreWeave conventions by design: ``group_by`` reflects the actual
topology the gang runs against, it is not a portable abstraction.
"""

from dataclasses import dataclass
from enum import StrEnum

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
# One NVL72 rack is also exactly one NVLink domain: RACK_SIZE nodes share the rack's NVLink
# switch, so RACK_SIZE is the physical size of a single nvlink.domain and the unit racks are
# provisioned in.
RACK_SIZE = 18
NVL72_INSTANCE_PREFIXES = ("gb200", "gb300")

# GPUs on one NVL72 compute tray (gb200-4x / gb300-4x). A gang whose pods each request this
# many GPUs is node-saturating: one pod per node, so pod count == node count and the rack-slice
# capacity arithmetic (a 16-pod slice fills 16 whole nodes) holds. A sub-node NVL72 pod would
# let multiple slices share a rack, silently voiding one-slice-per-rack; the sliced level
# rejects it.
NVL72_GPUS_PER_NODE = 4

# CoreWeave keeps only this many of a rack's RACK_SIZE nodes schedulable at once; the rest
# absorb host failures and maintenance. A hard single-nvlink.domain gang can therefore only be
# guaranteed placement up to this size. A larger gang would need every node in one rack healthy
# at the same moment, so binding it hard could leave it unschedulable indefinitely — above this
# size a gang binds the nvlink.domain label softly instead. This is the hard upper bound on a
# single-nvlink.domain gang, distinct from the physical RACK_SIZE.
SCHEDULABLE_RACK_NODES = 16


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
# gang spills across racks over InfiniBand. Reachable only via explicit config/group_by; the
# CLI routes multi-rack GB200 to the sliced level below instead.
COSCHEDULE_NVLINK_DOMAIN_PREFERRED = "nvlink.domain.preferred"
# Multi-rack GB200: partition the gang into per-rack slices, each slice hard-bound to one
# nvlink.domain (Kueue's PodSet-slice feature). The slice size is computed per gang to spread it
# evenly over the fewest racks (see balanced_rack_slice_size); because two slices exceed a rack,
# each lands on its own nvlink.domain, giving an exact balanced layout instead of the unbalanced
# fill preferred would give.
COSCHEDULE_NVLINK_DOMAIN_SLICED = "nvlink.domain.sliced"


class TopologyMode(StrEnum):
    """How a coscheduling level's node label constrains Kueue placement.

    PREFERRED/REQUIRED bind the whole PodSet to one domain of the label as a soft hint or a
    hard requirement. SLICE_REQUIRED instead partitions the PodSet into balanced per-rack slices
    and hard-binds each slice to one domain of the label (Kueue's PodSet-slice feature).
    """

    PREFERRED = "preferred"
    REQUIRED = "required"
    SLICE_REQUIRED = "slice"


@dataclass(frozen=True)
class KueueTopologyBinding:
    """The Kueue topology request a coscheduling ``group_by`` level maps to.

    ``node_label`` is the Topology-CR level the constraint binds. ``coarse_preferred_label``
    optionally adds a soft whole-PodSet preference at a coarser level (used by SLICE_REQUIRED so
    the per-rack slices also cluster near each other on the IB fabric). The SLICE_REQUIRED slice
    size is not stored here — it is computed per gang from its node count (balanced_rack_slice_size).
    """

    node_label: str
    mode: TopologyMode
    coarse_preferred_label: str | None = None


def balanced_rack_slice_size(num_tasks: int) -> int:
    """Nodes per rack slice for a multi-rack NVL72 gang, balanced across the fewest racks.

    Places the gang on ``ceil(num_tasks / SCHEDULABLE_RACK_NODES)`` NVLink domains — the fewest
    that hold at most ``SCHEDULABLE_RACK_NODES`` nodes each — and splits it evenly, so every rack
    runs the same node count. Returns that per-rack size (also Kueue's ``podset-slice-size``).
    Raises ``ValueError`` when the gang cannot split into equal slices that each exceed half a
    rack, the condition under which two slices could share one rack and the one-slice-per-rack
    guarantee would break.
    """
    if num_tasks <= 0:
        raise ValueError(f"gang size must be positive, got {num_tasks}")
    num_racks = -(-num_tasks // SCHEDULABLE_RACK_NODES)  # ceil
    min_rack_slice = RACK_SIZE // 2 + 1  # more than half a rack, so two slices can't share one
    if num_tasks % num_racks:
        raise ValueError(
            f"{num_tasks} nodes do not divide evenly across {num_racks} racks "
            f"(<= {SCHEDULABLE_RACK_NODES} nodes each); a multi-rack NVL72 gang must split into equal "
            f"{min_rack_slice}-{SCHEDULABLE_RACK_NODES} node rack slices (e.g. 20, 24, 32, 48)"
        )
    slice_size = num_tasks // num_racks
    if slice_size < min_rack_slice:
        raise ValueError(
            f"{num_tasks} nodes would place {num_racks} racks of {slice_size}, but a rack slice must "
            f"exceed half a rack ({min_rack_slice}+ of {RACK_SIZE} nodes) to keep one slice per rack; "
            f"use a larger gang (e.g. 20, 24, 32, 48)"
        )
    return slice_size


def gpu_gang_coscheduling_level(gpu_variant: str, replicas: int) -> str:
    """The Kueue topology level a multi-node GPU gang of ``replicas`` nodes should bind to.

    NVL72 (GB200/GB300) nodes carry ``ds.coreweave.com/nvlink.domain`` and one rack is a
    single NVLink domain of ``RACK_SIZE`` nodes, of which CoreWeave keeps only
    ``SCHEDULABLE_RACK_NODES`` schedulable at once. A gang that fits the guaranteed-schedulable
    slice of one rack binds HARD to ``nvlink.domain`` (``podset-required-topology``) so every
    replica shares the rack's NVLink fabric — the reason NVL72 exists. Binding a rack-sized gang
    hard would demand a fully healthy rack and could leave it unschedulable whenever a rack is
    down a node, so that is the largest hard single-domain gang.

    A larger gang binds to ``nvlink.domain.sliced``: the gang is partitioned into
    ``SCHEDULABLE_RACK_NODES``-node slices, each hard-bound to its own nvlink.domain, so it lands
    as an exact N racks x SCHEDULABLE_RACK_NODES balanced layout (see ``COSCHEDULE_NVLINK_DOMAIN_SLICED``).

    H100 and every non-NVL72 GPU carry no ``nvlink.domain`` label, so they always coschedule
    on ``leafgroup`` (soft IB colocation), which is the behavior this preserves for them.
    """
    if is_rack_based(gpu_variant):
        if replicas <= SCHEDULABLE_RACK_NODES:
            return COSCHEDULE_NVLINK_DOMAIN
        return COSCHEDULE_NVLINK_DOMAIN_SLICED
    return COSCHEDULE_LEAFGROUP
