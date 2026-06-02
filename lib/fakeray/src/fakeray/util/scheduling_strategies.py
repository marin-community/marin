# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``ray.util.scheduling_strategies`` shim.

The strategies are accepted and reduced to the information the actor scheduler
can actually use: a placement-group strategy carries the target bundle's
resources (so ``.options(scheduling_strategy=...)`` can size the actor); a node-
affinity strategy is accepted but best-effort (Iris picks the node).
"""

from __future__ import annotations

from typing import Any


class PlacementGroupSchedulingStrategy:
    """Carries a bundle's resources to ``.options()`` (best-effort placement)."""

    def __init__(
        self,
        placement_group: Any,
        placement_group_bundle_index: int = -1,
        placement_group_capture_child_tasks: bool | None = None,
        **_ignored: Any,
    ):
        self.placement_group = placement_group
        self.placement_group_bundle_index = placement_group_bundle_index
        # `_bundle` is what _resources_from_options() reads.
        idx = placement_group_bundle_index if placement_group_bundle_index >= 0 else 0
        self._bundle = placement_group.bundle_resources(idx) if hasattr(placement_group, "bundle_resources") else {}


class NodeAffinitySchedulingStrategy:
    """Accepted for Ray-compat; node pinning is best-effort (Iris decides)."""

    def __init__(self, node_id: str | None = None, soft: bool = True, **_ignored: Any):
        self.node_id = node_id
        self.soft = soft
        self._bundle = None


__all__ = ["NodeAffinitySchedulingStrategy", "PlacementGroupSchedulingStrategy"]
