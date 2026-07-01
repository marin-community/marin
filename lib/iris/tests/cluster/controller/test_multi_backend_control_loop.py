# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end control-loop tests for a controller driving two in-process backends.

A single live ``Controller`` holds two worker-daemon backends ("a" and "b"),
each owning its own scale group and workers. These tests drive the real
``_control_tick`` schedule phase and assert that the meta-scheduler routes each
job to the right backend, the per-backend partitions keep work isolated (a job
pinned to "a" never lands on "b"'s worker), and a job that matches no backend is
finalized UNSCHEDULABLE.
"""

import pytest
from iris.cluster.config import BackendConfig
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.timing import Timestamp
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    FakeProvider,
    make_job_request,
    make_scale_group_config,
    make_worker_metadata,
    query_job,
    query_task,
    query_tasks_for_job,
    reconcile_once,
    register_worker_into_backend,
    schedule_once,
    submit_job,
)
from tests.cluster.controller.transition_driver import WorkerTaskUpdates, apply_task_observations

pytestmark = pytest.mark.timeout(15)

BACKEND_CONFIGS = {
    "a": BackendConfig(kind="worker_daemon", scale_groups={"sg-a": make_scale_group_config()}),
    "b": BackendConfig(kind="worker_daemon", scale_groups={"sg-b": make_scale_group_config()}),
}


def _backend_constraint(backend_id: str) -> job_pb2.Constraint:
    return Constraint.create(key="backend", op=ConstraintOp.EQ, value=backend_id).to_proto()


@pytest.fixture
def two_backend_controller(make_controller):
    controller = make_controller(
        backends={"a": FakeProvider(), "b": FakeProvider()},
        backend_configs=BACKEND_CONFIGS,
    )
    return controller


@pytest.fixture
def state(two_backend_controller) -> ControllerTestState:
    # Each backend owns its own tracker and attrs projection, so workers register
    # through ``register_worker_into_backend`` (routed by scale group); this state's
    # own ``_health``/``_worker_attrs`` are unused for these multi-backend cases.
    controller = two_backend_controller
    return ControllerTestState(
        controller._db,
        endpoints=controller._endpoints,
        run_template_cache=controller._run_template_cache,
    )


def _submit_pinned(state: ControllerTestState, name: str, backend_id: str) -> JobName:
    # submit_job auto-injects device constraints and merges with the user
    # constraints, preserving the `backend` directive in the stored task.
    req = make_job_request(name=name, cpu=1, replicas=1)
    req.constraints.append(_backend_constraint(backend_id))
    submit_job(state, name, req)
    return JobName.root("test-user", name)


def _assigned(task) -> tuple[int, str | None, str]:
    return task.state, task.current_worker_id, task.backend_id


def test_jobs_route_to_their_pinned_backend(two_backend_controller, state):
    controller = two_backend_controller
    register_worker_into_backend(controller, "wa", "wa:8080", make_worker_metadata(), scale_group="sg-a")
    register_worker_into_backend(controller, "wb", "wb:8080", make_worker_metadata(), scale_group="sg-b")

    job_a = _submit_pinned(state, "job-a", "a")
    job_b = _submit_pinned(state, "job-b", "b")

    schedule_once(controller)

    (state_a, worker_a, backend_a) = _assigned(query_tasks_for_job(state, job_a)[0])
    (state_b, worker_b, backend_b) = _assigned(query_tasks_for_job(state, job_b)[0])

    assert (state_a, worker_a, backend_a) == (job_pb2.TASK_STATE_ASSIGNED, "wa", "a")
    assert (state_b, worker_b, backend_b) == (job_pb2.TASK_STATE_ASSIGNED, "wb", "b")


def test_job_never_leaks_onto_the_other_backends_worker(two_backend_controller, state):
    # Only backend "b" has a worker. A job pinned to "a" must NOT be placed on
    # "b"'s worker — it stays pending for lack of capacity in its own backend.
    controller = two_backend_controller
    register_worker_into_backend(controller, "wb", "wb:8080", make_worker_metadata(), scale_group="sg-b")

    job_a = _submit_pinned(state, "job-a", "a")

    schedule_once(controller)

    task = query_tasks_for_job(state, job_a)[0]
    assert task.state == job_pb2.TASK_STATE_PENDING
    assert task.current_worker_id is None
    # The job is still pinned to its backend for the next tick.
    assert task.backend_id == "a"


def test_unroutable_job_is_unschedulable(two_backend_controller, state):
    controller = two_backend_controller
    register_worker_into_backend(controller, "wa", "wa:8080", make_worker_metadata(), scale_group="sg-a")

    job_c = _submit_pinned(state, "job-c", "does-not-exist")

    schedule_once(controller)

    task = query_tasks_for_job(state, job_c)[0]
    assert task.state == job_pb2.TASK_STATE_UNSCHEDULABLE


def _observe_running(state: ControllerTestState, controller, backend_id: str, task_id: JobName, worker_id) -> None:
    attempt_id = query_task(state, task_id).current_attempt_id
    with controller._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[TaskUpdate(task_id=task_id, attempt_id=attempt_id, new_state=job_pb2.TASK_STATE_RUNNING)],
                )
            ],
            health=controller.backends[backend_id].health,
            endpoints=controller._endpoints,
            now=Timestamp.now(),
        )


def test_cross_backend_cascade_kills_descendant_on_other_backend(two_backend_controller, state):
    """A parent job's completion on backend "a" cascades to kill its child job's
    task on backend "b" in the same fold+commit, not a later tick.

    The controller folds the job-DAG recompute over the union of every
    backend's direct results in one pass, so a cascade triggered by backend
    "a" alone still reaches rows owned by backend "b".
    """
    controller = two_backend_controller
    wa = register_worker_into_backend(controller, "wa", "wa:8080", make_worker_metadata(), scale_group="sg-a")
    wb = register_worker_into_backend(controller, "wb", "wb:8080", make_worker_metadata(), scale_group="sg-b")

    parent_job = _submit_pinned(state, "parent", "a")

    child_req = make_job_request("child")
    child_req.constraints.append(_backend_constraint("b"))
    submit_job(state, "/test-user/parent/child", child_req)
    child_job = JobName.from_string("/test-user/parent/child")

    schedule_once(controller)

    parent_task = query_tasks_for_job(state, parent_job)[0]
    child_task = query_tasks_for_job(state, child_job)[0]
    assert (parent_task.state, parent_task.backend_id) == (job_pb2.TASK_STATE_ASSIGNED, "a")
    assert (child_task.state, child_task.backend_id) == (job_pb2.TASK_STATE_ASSIGNED, "b")

    _observe_running(state, controller, "a", parent_task.task_id, wa)
    _observe_running(state, controller, "b", child_task.task_id, wb)
    assert query_task(state, child_task.task_id).state == job_pb2.TASK_STATE_RUNNING

    # Only backend "a" reports a transition this tick.
    with controller._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=wa,
                    updates=[
                        TaskUpdate(
                            task_id=parent_task.task_id,
                            attempt_id=query_task(state, parent_task.task_id).current_attempt_id,
                            new_state=job_pb2.TASK_STATE_SUCCEEDED,
                        )
                    ],
                )
            ],
            health=controller.backends["a"].health,
            endpoints=controller._endpoints,
            now=Timestamp.now(),
        )

    assert query_job(state, parent_job).state == job_pb2.JOB_STATE_SUCCEEDED
    assert query_task(state, child_task.task_id).state == job_pb2.TASK_STATE_KILLED
    assert query_job(state, child_job).state == job_pb2.JOB_STATE_KILLED

    # Backend "b"'s next reconcile sees the cascade-killed task and doesn't
    # resurrect it or error.
    reconcile_once(controller)
    assert query_task(state, child_task.task_id).state == job_pb2.TASK_STATE_KILLED
