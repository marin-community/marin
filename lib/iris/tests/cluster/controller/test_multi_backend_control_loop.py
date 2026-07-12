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
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    FakeProvider,
    make_job_request,
    make_scale_group_config,
    make_worker_metadata,
    query_tasks_for_job,
    register_worker_into_backend,
    schedule_once,
    submit_job,
)

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
    # Each backend owns its own tracker; workers register through
    # ``register_worker_into_backend`` (routed by scale group). The global
    # WorkerAttrsProjection is shared via db.caches across all backends.
    controller = two_backend_controller
    return ControllerTestState(
        controller._db,
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
