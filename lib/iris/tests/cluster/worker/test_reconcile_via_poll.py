# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side reconcile-via-poll behavior.

End-to-end correctness is covered by ``tests/e2e/test_smoke.py`` and
``tests/e2e/test_iris_run.py`` which submit jobs and watch them run through
the real fetch + submit path. The cases here pin down the local-only
behaviors the e2e suite would not catch quickly on regression:

  - Rapid duplicate polls do not enqueue twice (the placeholder is already
    in ``self._tasks``).
  - On fetch failure, the attempt transitions to WORKER_FAILED via its
    normal lifecycle and pushes a status update to the controller.

The placeholder-enqueue path itself is covered by
``test_worker.py::test_poll_tasks_without_adoption_omits_unknown_expected``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from iris.cluster.types import JobName
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import job_pb2, worker_pb2
from iris.test_util import wait_for_condition
from rigging.timing import Duration

from tests.cluster.worker.conftest import create_run_task_request

pytestmark = pytest.mark.timeout(10)


@pytest.fixture
def worker(mock_bundle_store, mock_runtime, tmp_path) -> Worker:
    config = WorkerConfig(
        port=0,
        port_range=(50100, 50200),
        poll_interval=Duration.from_seconds(0.1),
        cache_dir=tmp_path / "cache",
        default_task_image="mock-image",
    )
    return Worker(config, bundle_store=mock_bundle_store, container_runtime=mock_runtime)


def _expected(task_id: str, attempt_id: int = 0) -> job_pb2.WorkerTaskStatus:
    return job_pb2.WorkerTaskStatus(task_id=task_id, attempt_id=attempt_id)


def test_duplicate_polls_do_not_double_enqueue(worker):
    """Rapid duplicate polls only install one TaskAttempt."""
    task_id = JobName.root("test-user", "dup").task(0).to_wire()
    canned = create_run_task_request(task_id=task_id, attempt_id=0)
    rpc_spy = Mock(return_value=canned)
    worker._controller_client = SimpleNamespace(get_task_attempt_info=rpc_spy, update_task_status=Mock())

    poll = worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected(task_id, 0)])
    worker.handle_poll_tasks(poll)
    worker.handle_poll_tasks(poll)
    worker.handle_poll_tasks(poll)

    # One placeholder, one attempt key.
    matching = [k for k in worker._tasks if k[0] == task_id]
    assert matching == [(task_id, 0)]


def test_fetch_failure_transitions_attempt_to_worker_failed(worker):
    """Fetch raising → attempt's lifecycle handler maps it to WORKER_FAILED
    and the state-change callback pushes UpdateTaskStatus to the controller."""
    task_id = JobName.root("test-user", "fetch-fail").task(0).to_wire()
    rpc_spy = Mock(side_effect=RuntimeError("simulated fetch failure"))
    update_spy = Mock()
    worker._controller_client = SimpleNamespace(
        get_task_attempt_info=rpc_spy,
        update_task_status=update_spy,
    )
    worker._worker_id = "w-1"

    worker.handle_poll_tasks(worker_pb2.Worker.PollTasksRequest(expected_tasks=[_expected(task_id, 0)]))

    task = worker.get_task(task_id, attempt_id=0)
    assert task is not None
    wait_for_condition(lambda: task.status == job_pb2.TASK_STATE_WORKER_FAILED)
    assert "simulated fetch failure" in (task.error or "")

    # The lifecycle's on_state_change fired UpdateTaskStatus for WORKER_FAILED.
    wait_for_condition(lambda: update_spy.call_count >= 1)
    final_states = [u.state for call in update_spy.call_args_list for u in call.args[0].updates]
    assert job_pb2.TASK_STATE_WORKER_FAILED in final_states
