# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An attempt keeps the region of the worker it was assigned to."""

from iris.cluster.controller import writes
from iris.cluster.controller.schema import task_attempts_table
from iris.cluster.types import JobName, WellKnownAttribute, WorkerId
from iris.testing.controller import (
    assign_task,
    make_job_request,
    make_worker_metadata,
    register_worker,
    submit_job,
)
from iris.testing.controller_state import ControllerTestState
from sqlalchemy import select


def _assign_task_to_new_worker(state: ControllerTestState, region: str | None) -> tuple[JobName, WorkerId]:
    metadata = make_worker_metadata()
    if region is not None:
        metadata.attributes[WellKnownAttribute.REGION].string_value = region
    worker_id = register_worker(state, "worker-1", "worker-1:8080", metadata)
    task = submit_job(state, "job", make_job_request())[0]
    assign_task(state, task, worker_id)
    return task.task_id, worker_id


def _attempt_regions(state: ControllerTestState, task_id: JobName) -> list[str | None]:
    with state._db.read_snapshot() as tx:
        rows = tx.execute(select(task_attempts_table.c.region).where(task_attempts_table.c.task_id == task_id)).all()
    return [row.region for row in rows]


def test_attempt_region_survives_the_worker_being_removed(state: ControllerTestState) -> None:
    task_id, worker_id = _assign_task_to_new_worker(state, "us-central2")

    with state._db.transaction() as cur:
        writes.remove_worker(cur, worker_id, health=state._health)

    assert _attempt_regions(state, task_id) == ["us-central2"]


def test_attempt_region_is_null_when_the_worker_advertises_none(state: ControllerTestState) -> None:
    task_id, _ = _assign_task_to_new_worker(state, None)

    assert _attempt_regions(state, task_id) == [None]
