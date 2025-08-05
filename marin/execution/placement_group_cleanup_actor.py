import time

import ray
from ray._private.utils import hex_to_binary
from ray._raylet import PlacementGroupID
from ray.util import state  # noqa: F401
from ray.util.placement_group import (
    PlacementGroup,
    placement_group_table,
    remove_placement_group,
)


@ray.remote(num_cpus=0)
class PlacementGroupCleanupActor:
    """Periodically cleans failed placement groups from the cluster."""

    def __init__(self, interval_seconds: int = 600):
        self.interval_seconds = interval_seconds

    def run(self) -> None:
        """Continuously run cleanup in a loop."""
        while True:
            self._cleanup_once()
            time.sleep(self.interval_seconds)

    def _cleanup_once(self) -> None:
        for placement_group_info in placement_group_table().values():
            pg = PlacementGroup(PlacementGroupID(hex_to_binary(placement_group_info["placement_group_id"])))
            too_many_tries = placement_group_info["stats"]["scheduling_attempt"] > 500
            failure_mode = placement_group_info["stats"]["scheduling_state"] == "NO_RESOURCES"
            if too_many_tries and failure_mode:
                remove_placement_group(pg)
