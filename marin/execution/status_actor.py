import datetime
import json
from dataclasses import asdict
from typing import List

import fsspec
import ray
from ray import ObjectRef

from marin.execution.executor_step_status import get_status_path, ExecutorStepEvent, read_events, is_failure, \
    STATUS_SUCCESS


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



            status = executor_step_event.status

        elif executor_step_event is None:
            status = self.get_status(output_path)

        else:
            status = executor_step_event.status
            path = get_status_path(output_path)
            events = read_events(path)
            events.append(executor_step_event)
            # Note: gcs files are immutable so can't append, so have to read everything.
            with fsspec.open(path, "w") as f:
                for event in events:
                    print(json.dumps(asdict(event)), file=f)


        self.value_to_reference_status[output_path] = (status, reference)


    def add_update_status(self, output_path: str, status: str, message: str | None = None, ray_task_id: str | None = None):
        # First update this status on GCS, then update
        date = datetime.now().isoformat()
        event = ExecutorStepEvent(date=date, status=status, message=message, ray_task_id=ray_task_id)
        self._add_status_and_reference(output_path, event, None)
        # self.value_to_reference_status[output_path] = (status, self.value_to_reference_status[output_path][1])

    def add_update_reference(self, output_path: str, reference: ObjectRef):
        self.value_to_reference_status[output_path] = (self.value_to_reference_status.get(output_path, (None, None))[0],
                                                                                          reference)

    def get_status_and_reference(self, output_path: str) -> tuple[str,ObjectRef]:
        return self.value_to_reference_status[output_path]

    def get_status(self, output_path: str) -> str | None:
        # (If a key is present in the ACTOR then we return the status else we go to GCP and check.
        # If it’s SUCCESS or FAILED, it’s true status, else it’s a stale status and we don’t consider it.
        if output_path in self.value_to_reference_status:
            return self.value_to_reference_status[output_path][0]
        else:
            status_path = get_status_path(output_path)
            events = read_events(status_path)
            if len(events) > 0:
                if is_failure(events[-1].status) or events[-1].status == STATUS_SUCCESS:
                    return events[-1].status

        return None