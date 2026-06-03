# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``ray.util.placement_group`` — re-export the shim so ``from
ray.util.placement_group import placement_group`` resolves (SkyRL's import)."""

from fakeray._placement import (
    PlacementGroup,
    placement_group,
    placement_group_table,
    remove_placement_group,
)

__all__ = ["PlacementGroup", "placement_group", "placement_group_table", "remove_placement_group"]
