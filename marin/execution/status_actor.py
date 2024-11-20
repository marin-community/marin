from collections import OrderedDict
from datetime import datetime

import ray
from ray import ObjectRef

from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    ExecutorStepEvent,
    append_status_event,
    get_status_path,
    is_failure,
    read_events,
)


@ray.remote
class StatusActor:

    def __init__(self, cache_size: int = 10_000):
        self.value_to_reference_status: dict[str, tuple[str, ObjectRef]] = {}
        self.lru_cache: OrderedDict[str, None] = OrderedDict()  # lru_cache to keep dict size to cache_size
        self.cache_size = cache_size

    def _add_status_and_reference(
        self, output_path: str, executor_step_event: ExecutorStepEvent | None, reference: list[ObjectRef] | None
    ):
        """
        Main function to update the status and reference of a output path. Reference is passed by reference as a list.
        If either one of status or reference is None, we use the previous value.
        """
        if reference is None and executor_step_event is None:
            return
        elif reference is None:
            # Status is being updated, We need to write this to GCP too
            reference = self.value_to_reference_status.get(output_path, (None, None))[1]
            append_status_event(output_path, executor_step_event)
            status = executor_step_event.status

        elif executor_step_event is None:
            status = self.get_status(output_path)
            reference = reference[0]

        else:
            status = executor_step_event.status
            append_status_event(output_path, executor_step_event)
            reference = reference[0]

        self.value_to_reference_status[output_path] = (status, reference)

        # Manage LRU cache for statuses that are SUCCESS or FAILED
        if status in {STATUS_SUCCESS, STATUS_FAILED}:
            self.lru_cache[output_path] = None
            self.lru_cache.move_to_end(output_path)  # Mark as recently used

            if len(self.lru_cache) > self.cache_size:
                # Evict the least recently used item
                oldest = self.lru_cache.popitem(last=False)
                del self.value_to_reference_status[oldest[0]]

    def add_update_status(
        self, output_path: str, status: str, message: str | None = None, ray_task_id: str | None = None
    ):
        """
        Update the status of a output path. We also write the output to GCP.
        """
        date = datetime.now().isoformat()
        event = ExecutorStepEvent(date=date, status=status, message=message, ray_task_id=ray_task_id)
        self._add_status_and_reference(output_path, event, None)

    def add_update_reference(self, output_path: str, reference: list[ObjectRef]):
        """
        We update the reference for a output path. We need to pass reference as a list to ensure we pass by reference
        """
        self._add_status_and_reference(output_path, None, reference)

    def get_status_and_reference(self, output_path: str) -> tuple[str, ObjectRef]:
        return self.value_to_reference_status[output_path]

    def get_status(self, output_path: str) -> str | None:
        """If a key is present in the ACTOR then we return the status else we go to GCP and check,
        if it's SUCCESS or FAILED, it's true status, else it's a stale status and we don't consider it
        and return None"""

        if output_path in self.value_to_reference_status:
            return self.value_to_reference_status[output_path][0]
        else:
            status_path = get_status_path(output_path)
            events = read_events(status_path)
            if len(events) > 0:
                if is_failure(events[-1].status) or events[-1].status == STATUS_SUCCESS:
                    self.value_to_reference_status[output_path] = (events[-1].status, None)
                    return events[-1].status
            else:  # No status file, so it's a new step
                self.value_to_reference_status[output_path] = (None, None)
                return None

    def get_reference(self, output_path: str) -> ObjectRef | None:
        return self.value_to_reference_status[output_path][1]

    def get_all_status(self) -> dict[str, tuple[str, ObjectRef]]:
        return self.value_to_reference_status.copy()
