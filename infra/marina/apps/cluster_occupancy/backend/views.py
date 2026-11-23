# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shapes and ordering shared by the live and history views.

Both sources produce the same node/holder model so the frontend renders "now"
and history identically, and both order it the same way.
"""

from dataclasses import dataclass

# CoreWeave stamps the rack (NVLink domain) into each attempt's topology JSON.
RACK_JSON_KEY = "ds.coreweave.com/nvlink.domain"


@dataclass(frozen=True)
class Holder:
    """What one root job holds on one node."""

    user_id: str | None
    job_id: str | None
    root_job_id: str | None
    root_name: str | None
    priority_band: str | None
    accelerator_type: str | None
    gpus: int
    started_ms: int | None
    finished_ms: int | None


@dataclass(frozen=True)
class NodeView:
    """One node: its rack, GPU kind, capacity, and current holders."""

    node_name: str
    rack: str
    gpu_type: str
    capacity_gpus: int
    holders: list[Holder]


def holder_sort_key(holder: Holder) -> tuple[int, str, str]:
    """Largest holding first, then by job identity."""
    return (-holder.gpus, holder.root_name or "", holder.job_id or "")


def node_order(rack: str, node_name: str) -> tuple[bool, str, str]:
    """Racked nodes first (sorted by rack then name), rackless nodes last."""
    return (rack == "", rack, node_name)


def node_sort_key(node: NodeView) -> tuple[bool, str, str]:
    return node_order(node.rack, node.node_name)
