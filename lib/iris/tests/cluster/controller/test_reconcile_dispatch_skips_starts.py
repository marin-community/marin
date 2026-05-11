# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch-loop tests for the reconcile-via-poll protocol mode.

When a worker advertises ``protocol-mode=reconcile_via_poll`` via its
WorkerMetadata attributes, the controller's ``_reconcile_worker_batch``
must skip StartTasks materialization for that worker; its expected_tasks
still flow through PollTasks.
"""

import pytest
from iris.cluster.constraints import ProtocolMode, WellKnownAttribute
from iris.cluster.controller.transitions import Assignment
from iris.cluster.controller.worker_provider import WorkerReconcilePlan, WorkerReconcileResult
from iris.rpc import worker_pb2

from .conftest import (
    FakeProvider,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)

pytestmark = pytest.mark.timeout(15)


class _RecordingProvider(FakeProvider):
    """Provider that records every ``WorkerReconcilePlan`` it sees."""

    def __init__(self) -> None:
        self.plans: list[WorkerReconcilePlan] = []

    def reconcile_workers(self, plans):
        # Snapshot the plans so the test can assert on start_tasks /
        # expected_tasks per worker after the dispatch tick.
        self.plans.extend(plans)
        return [
            WorkerReconcileResult(
                worker_id=plan.worker_id,
                start_response=worker_pb2.Worker.StartTasksResponse() if plan.start_tasks else None,
                start_error=None,
                poll_updates=[],
                poll_error=None,
            )
            for plan in plans
        ]


def _set_protocol_mode(metadata, mode: ProtocolMode):
    metadata.attributes[WellKnownAttribute.PROTOCOL_MODE].string_value = mode
    return metadata


def test_reconcile_dispatch_skips_start_tasks_for_reconcile_via_poll(make_controller):
    """Two workers, one legacy + one reconcile-via-poll, both assigned a task.

    The dispatch loop must:
      * emit a StartTasks payload only for the legacy worker
      * include both tasks in the PollTasks ``expected_tasks`` so each
        worker reconciles its own copy
    """
    provider = _RecordingProvider()
    controller = make_controller(provider=provider)
    state = controller.state

    legacy_wid = register_worker(
        state,
        "legacy-w",
        "legacy:8080",
        _set_protocol_mode(make_worker_metadata(), ProtocolMode.START_STOP),
    )
    poll_wid = register_worker(
        state,
        "poll-w",
        "poll:8080",
        _set_protocol_mode(make_worker_metadata(), ProtocolMode.RECONCILE_VIA_POLL),
    )

    # One task per worker. Submitting two jobs avoids any single-replica
    # contention; each task lands on its dedicated worker via the explicit
    # assignment below.
    legacy_tasks = submit_job(state, "legacy-job", make_job_request(name="legacy-job", cpu=1, replicas=1))
    poll_tasks = submit_job(state, "poll-job", make_job_request(name="poll-job", cpu=1, replicas=1))
    legacy_task_id = legacy_tasks[0].task_id
    poll_task_id = poll_tasks[0].task_id

    with state._store.transaction() as cur:
        state.queue_assignments(
            cur,
            [
                Assignment(task_id=legacy_task_id, worker_id=legacy_wid),
                Assignment(task_id=poll_task_id, worker_id=poll_wid),
            ],
        )

    controller._reconcile_worker_batch()

    plans_by_worker = {plan.worker_id: plan for plan in provider.plans}
    assert set(plans_by_worker) == {legacy_wid, poll_wid}

    legacy_plan = plans_by_worker[legacy_wid]
    poll_plan = plans_by_worker[poll_wid]

    # Legacy worker: receives a StartTasks payload for its assigned task.
    assert [req.task_id for req in legacy_plan.start_tasks] == [legacy_task_id.to_wire()]
    assert [(e.task_id, e.attempt_id) for e in legacy_plan.expected_tasks] == [
        (legacy_task_id, 0),
    ]

    # Reconcile-via-poll worker: NO StartTasks; the task still appears in
    # expected_tasks so the worker fetches the spec via GetTaskAttemptInfo.
    assert poll_plan.start_tasks == []
    assert [(e.task_id, e.attempt_id) for e in poll_plan.expected_tasks] == [
        (poll_task_id, 0),
    ]


def test_reconcile_dispatch_defaults_legacy_when_attribute_absent(make_controller):
    """A worker registered without protocol-mode is treated as legacy."""
    provider = _RecordingProvider()
    controller = make_controller(provider=provider)
    state = controller.state

    # Build metadata without the protocol-mode attribute, simulating
    # rollout of new controller against pre-A.2 workers.
    metadata = make_worker_metadata()
    assert WellKnownAttribute.PROTOCOL_MODE not in metadata.attributes
    wid = register_worker(state, "legacy-default", "legacy:8080", metadata)

    tasks = submit_job(state, "legacy-default", make_job_request(name="legacy-default", cpu=1, replicas=1))
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=tasks[0].task_id, worker_id=wid)])

    controller._reconcile_worker_batch()

    plans_by_worker = {plan.worker_id: plan for plan in provider.plans}
    assert wid in plans_by_worker
    assert [req.task_id for req in plans_by_worker[wid].start_tasks] == [tasks[0].task_id.to_wire()]
