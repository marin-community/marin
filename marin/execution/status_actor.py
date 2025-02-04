import ray
from ray.util import state  # noqa


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
        self.task_id_locks: dict[str, str] = {}
        print("StatusActor initialized")

    def get_task_id_with_lock(self, output_path: str) -> str:
        return self.task_id_locks.get(output_path, None)

    def get_lock_by_replacing_task_id(self, output_path: str, task_id: str, current_owner_task_id: str | None) -> bool:
        if self.task_id_locks.get(output_path, None) == current_owner_task_id:
            self.task_id_locks[output_path] = task_id
            return True
        return False

    def get_task_id_locks(self):
        return self.task_id_locks.copy()
