from datetime import datetime
import json
from dataclasses import asdict
from typing import List

import fsspec
import ray
from ray import ObjectRef

from marin.execution.executor_step_status import get_status_path, ExecutorStepEvent, read_events, is_failure, \
    STATUS_SUCCESS, append_status_event


@ray.remote
class StatusActor:

    def __init__(self):
        self.value_to_reference_status: dict[str, tuple[str,ObjectRef]] = {}

    def _add_status_and_reference(self, output_path: str, executor_step_event: ExecutorStepEvent | None,
                                  reference: ObjectRef | None):

        if reference is None and executor_step_event is None:
            return
        elif reference is not None:
            # Status is being updated, We need to write this to GCP too
            reference = self.value_to_reference_status.get(output_path, (None, None))[1]
            append_status_event(output_path, executor_step_event)
            status = executor_step_event.status

        elif executor_step_event is None:
            status = self.get_status(output_path)

        else:
            status = executor_step_event.status
            append_status_event(output_path, executor_step_event)

        self.value_to_reference_status[output_path] = (status, reference)


    def add_update_status(self, output_path: str, status: str, message: str | None = None, ray_task_id: str | None = None):
        # First update this status on GCS, then update
        date = datetime.now().isoformat()
        event = ExecutorStepEvent(date=date, status=status, message=message, ray_task_id=ray_task_id)
        self._add_status_and_reference(output_path, event, None)

    def add_update_reference(self, output_path: str, reference: ObjectRef):
        self._add_status_and_reference(output_path, None, reference)

    def get_status_and_reference(self, output_path: str) -> tuple[str,ObjectRef]:
        return self.value_to_reference_status[output_path]

    def get_status(self, output_path: str) -> str | None:
        # If a key is present in the ACTOR then we return the status else we go to GCP and check,
        # if it’s SUCCESS or FAILED, it’s true status, else it’s a stale status and we don’t consider it and return None

        if output_path in self.value_to_reference_status:
            return self.value_to_reference_status[output_path][0]
        else:
            status_path = get_status_path(output_path)
            events = read_events(status_path)
            if len(events) > 0:
                if is_failure(events[-1].status) or events[-1].status == STATUS_SUCCESS:
                    self.value_to_reference_status[output_path] = (events[-1], None)
                    return events[-1].status
            else: # No status file, so it's a new step
                self.value_to_reference_status[output_path] = (None, None)
                return None