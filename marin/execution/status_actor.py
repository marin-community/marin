from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime

import ray
from ray import ObjectRef
from ray.util import state  # noqa

from marin.execution.executor_step_status import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    STATUS_WAITING,
    ExecutorStepEvent,
    append_status_event,
    get_status_path,
    is_failure,
    read_events,
)


@dataclass
class RayObjectRef:
    """This class wrap a ray object reference to pass it by reference. If we just pass the object reference, it will be
    passed by value and ray would try to resolve it before passing it to the function.


        @ray.remote
        class Actor:
            def add(self, ref):
                return ref

        @ray.remote
        def f():
            return "Hello"

        actor = Actor.remote()
        print(ray.get(actor.add.remote(f.remote())))

        The above code will print "Hello"

    """

    ref: ObjectRef


@ray.remote
class StatusActor:
    """
    This class is used to keep track of the status and reference of each output path across various experiments.
    This enables steps from one experiment to depend on steps from another experiment.
    Actor class is backed by GCP, where we write any status updates of the output path. Incase we have ray cluster
    failure, we use the status file to recover the status of the output path.
    We map the previous status (before cluster failure) to new status (after cluster failure) according to:
            old_status = [STATUS_SUCCESS, STATUS_FAILED, STATUS_WAITING, STATUS_RUNNING, STATUS_DEP_FAILED]
            new_status = [STATUS_SUCCESS, STATUS_FAILED, None, None, STATUS_DEP_FAILED]
    """

    def __init__(self, cache_size: int = 10_000):
        self.value_to_status_reference: dict[str, tuple[str | None, ObjectRef | None]] = {}
        # TODO(abhi): Make the values of the dict a dataclass
        self.lru_cache: OrderedDict[str, None] = OrderedDict()  # lru_cache to keep dict size to cache_size
        self.cache_size = cache_size
        self.lock_output_path_to_task_id: dict[str, str] = {}
        print("StatusActor initialized")

    def _add_status_and_reference(
        self, output_path: str, executor_step_event: ExecutorStepEvent | None, reference: RayObjectRef | None
    ):
        """
        Main function to update the status and reference of an output path.
        reference is a RayObjectRef object that wraps the reference to pass it by reference.
        executor_step_event is the event updating the status of a step. We update our dict and also write to GCP.
        If either one of status or reference is None, we use the previous value.
        """

        reference = reference and reference.ref
        if reference is None and executor_step_event is None:
            return
        elif reference is None:
            # Status is being updated, We need to write this to GCP too
            reference = self.value_to_status_reference.get(output_path, (None, None))[1]
            append_status_event(output_path, executor_step_event)
            status = executor_step_event.status

        elif executor_step_event is None:
            status = self.get_status(output_path)

        else:
            status = executor_step_event.status
            append_status_event(output_path, executor_step_event)

        self.value_to_status_reference[output_path] = (status, reference)

        # Manage LRU cache for statuses that are SUCCESS or FAILED
        if status in {STATUS_SUCCESS, STATUS_FAILED}:
            self.lru_cache[output_path] = None
            self.lru_cache.move_to_end(output_path)  # Mark as recently used

            if len(self.lru_cache) > self.cache_size:
                # Evict the least recently used item
                oldest = self.lru_cache.popitem(last=False)
                del self.value_to_status_reference[oldest[0]]

    def update_status(self, output_path: str, status: str, message: str | None = None, ray_task_id: str | None = None):
        """
        Update the status of an output path. We also write the output to GCP.
        """
        date = datetime.now().isoformat()
        event = ExecutorStepEvent(date=date, status=status, message=message, ray_task_id=ray_task_id)
        self._add_status_and_reference(output_path, event, None)

    def update_reference(self, output_path: str, reference: RayObjectRef):
        """
        We update the reference for an output path. We need to pass reference as a list to ensure we pass by reference
        """
        self._add_status_and_reference(output_path, None, reference)

    def get_status(self, output_path: str) -> str | None:
        """Returns the step's status, if known.
        If this actor knows about it (e.g. it's currently running or recently failed), we return that.
        Otherwise, we check against the .executor_status file to see. If it has a "final" status (SUCCESS or FAILED),
        then we return that. Otherwise, return None."""

        if output_path in self.value_to_status_reference:
            status = self.value_to_status_reference[output_path][0]
            if status == STATUS_RUNNING or status == STATUS_WAITING:
                # Verify if this is still running and was not stopped by ray job API or any other way
                # There must be a task_id with lock
                task_id = self.lock_output_path_to_task_id[output_path]
                task_state = ray.util.state.get_task(task_id, timeout=60)

                if task_state is None:  # We try for 60 seconds. If we don't get the task state, we assume it's running
                    return status

                if type(task_state) is list:  # Due to retires in ray, task_state can be a list of states
                    task_state = task_state[-1]

                if task_state.state == "FAILED":
                    self.update_status(
                        output_path, STATUS_FAILED, message="Task was stopped by ray API", ray_task_id=task_id
                    )
                    self.release_lock(output_path)
                    return STATUS_FAILED
            return status

        else:
            status_path = get_status_path(output_path)
            events = read_events(status_path)
            if len(events) > 0:
                if is_failure(events[-1].status) or events[-1].status == STATUS_SUCCESS:
                    self.value_to_status_reference[output_path] = (events[-1].status, None)
                    return events[-1].status
                else:
                    return None
            else:  # No status file, so it's a new step
                self.value_to_status_reference[output_path] = (None, None)
                return None

    def get_reference(self, output_path: str) -> ObjectRef | None:
        return self.value_to_status_reference[output_path][1]

    def get_all_status(self) -> dict[str, tuple[str, ObjectRef]]:
        return self.value_to_status_reference.copy()

    def get_lock(self, output_path: str, ray_task_id: str) -> str:
        """Returns the lock for the given output path. If some other task has already locked the output path, then
        return the task ID of the task that has locked it."""
        if output_path not in self.lock_output_path_to_task_id:
            self.lock_output_path_to_task_id[output_path] = ray_task_id
        return self.lock_output_path_to_task_id[output_path]

    def release_lock(self, output_path: str):
        """Release the lock for the given output path."""
        del self.lock_output_path_to_task_id[output_path]

    def get_statuses(self, output_paths: list[str]) -> list[str | None]:
        return [self.get_status(output_path) for output_path in output_paths]
