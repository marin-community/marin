# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort ``ray.util.placement_group`` shim.

Ray reserves *bundles* of resources atomically (gang) and lets actors pin to a
bundle index, with ``PACK``/``SPREAD`` locality and fractional-GPU colocation.
Iris has no matching primitive, so v1 **captures** the bundle list + strategy
without enforcing colocation or atomicity: each actor scheduled "into" the group
is sized from its bundle and scheduled independently by Iris.

This makes the API present and gives the right number/size of GPU actors. It does
NOT provide PACK/SPREAD locality, atomic gang bring-up, or fractional-GPU
colocation — see the design note. The discrepancy is logged loudly.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any

from fakeray._object_ref import ObjectRef

logger = logging.getLogger(__name__)


class PlacementGroup:
    """Lightweight stand-in for a Ray PlacementGroup.

    Holds the requested bundles + strategy. ``ready()`` resolves immediately
    (no real reservation happens).
    """

    def __init__(self, bundles: list[dict], strategy: str = "PACK"):
        self.bundles = list(bundles)
        self.strategy = strategy
        logger.warning(
            "fakeray: placement_group(strategy=%s, %d bundles) is ADVISORY — Iris schedules "
            "bundles independently (no PACK/SPREAD locality, atomic gang, or fractional-GPU "
            "colocation). See the fray-ray design note.",
            strategy,
            len(bundles),
        )

    def ready(self) -> ObjectRef:
        fut: Future = Future()
        fut.set_result(self)
        import uuid

        return ObjectRef(id=uuid.uuid4().hex, future=fut)

    def bundle_resources(self, index: int) -> dict:
        return self.bundles[index] if 0 <= index < len(self.bundles) else {}

    @property
    def bundle_specs(self) -> list[dict]:
        """Ray-compat alias used by e.g. SkyRL's get_ray_pg_ready_with_timeout."""
        return self.bundles


def placement_group(bundles: list[dict], strategy: str = "PACK", **_ignored: Any) -> PlacementGroup:
    return PlacementGroup(bundles, strategy=strategy)


def placement_group_table(pg: PlacementGroup) -> dict:
    """Ray-compat ``placement_group_table`` (best-effort).

    Real Ray returns per-bundle node/resource placement. We have no bundle->node
    reservation (bundles are scheduled independently), so every bundle reports a
    single synthetic node id. Callers that only need bundle count + a stable
    node mapping (e.g. SkyRL's ``_probe_bundle_placement``) work; callers that
    depend on real cross-node bundle ordering do not (documented limitation).
    """
    n = len(pg.bundles)
    return {
        "bundles": dict(enumerate(pg.bundles)),
        "bundles_to_node_id": {i: "fakeray-node-0" for i in range(n)},
        "strategy": pg.strategy,
        "state": "CREATED",
    }


def remove_placement_group(pg: PlacementGroup) -> None:
    """No-op: nothing was reserved."""
    return None
