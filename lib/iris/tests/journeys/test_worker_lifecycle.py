# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.rpc import job_pb2

from lib.iris.tests.journeys.worker import WorkerJourney


@pytest.fixture
def worker_journey(tmp_path, monkeypatch):
    world = WorkerJourney(tmp_path, monkeypatch)
    try:
        yield world
    finally:
        world.close()


def test_running_worker_disappearance_releases_capacity_for_replacement_attempt(worker_journey):
    journey = worker_journey
    old = journey.add_worker("worker-a", "worker-a:8080")
    job = journey.submit("worker-loss")
    journey.run_until_task_state(job, job_pb2.TASK_STATE_RUNNING)

    old.reachable = False
    journey.advance(1)
    journey.run_until_worker_releases_task("worker-a", job)

    released = journey.task(job)
    assert released.state == job_pb2.TASK_STATE_PENDING
    assert released.worker_id == ""
    assert "worker-a" not in journey.worker_ids()

    journey.add_worker("worker-b", "worker-b:8080")
    journey.run_until_task_state(job, job_pb2.TASK_STATE_RUNNING)

    replacement = journey.task(job)
    assert replacement.worker_id == "worker-b"
    assert [(attempt.attempt_id, attempt.state) for attempt in replacement.attempts] == [
        (0, job_pb2.TASK_STATE_WORKER_FAILED),
        (1, job_pb2.TASK_STATE_RUNNING),
    ]


def test_recycled_worker_address_fences_prior_incarnation_and_stale_observation(worker_journey):
    journey = worker_journey
    address = "recycled:8080"
    journey.add_worker("worker-a", address)
    job = journey.submit("recycled-address")
    journey.run_until_task_state(job, job_pb2.TASK_STATE_RUNNING)
    old_attempt_uid = journey.task(job).attempts[0].attempt_uid

    replacement = journey.replace_daemon(address, "worker-b")
    journey.advance(1)
    journey.run_until_worker_releases_task("worker-a", job)

    assert journey.task(job).state == job_pb2.TASK_STATE_PENDING
    assert "worker-a" not in journey.worker_ids()

    replacement.queue_observation(old_attempt_uid, job_pb2.TASK_STATE_SUCCEEDED)
    journey.add_worker("worker-b", address)
    journey.run_until_task_state(job, job_pb2.TASK_STATE_RUNNING)

    task = journey.task(job)
    assert old_attempt_uid in replacement.delivered_observation_uids
    assert task.worker_id == "worker-b"
    assert [(attempt.attempt_id, attempt.state) for attempt in task.attempts] == [
        (0, job_pb2.TASK_STATE_WORKER_FAILED),
        (1, job_pb2.TASK_STATE_RUNNING),
    ]


def test_degraded_worker_is_not_scheduled_but_recovers_through_reconcile(worker_journey):
    journey = worker_journey
    degraded = journey.add_worker("worker-a", "worker-a:8080")
    journey.add_worker("worker-b", "worker-b:8080")

    degraded.fail_next_reconciles(1)
    journey.step()
    assert journey.worker("worker-a").consecutive_failures == 1

    job = journey.submit("avoid-degraded")
    journey.run_until_task_state(job, job_pb2.TASK_STATE_RUNNING)

    assert journey.task(job).worker_id == "worker-b"
    assert journey.worker("worker-a").healthy
    assert journey.worker("worker-a").consecutive_failures == 0


def test_preempted_attempt_holds_capacity_until_worker_reports_exact_terminal_state(worker_journey):
    journey = worker_journey
    daemon = journey.add_worker("worker-a", "worker-a:8080")
    low = journey.submit("batch", priority_band=job_pb2.PRIORITY_BAND_BATCH)
    journey.run_until_task_state(low, job_pb2.TASK_STATE_RUNNING)

    daemon.acknowledge_stops = False
    journey.preempt(low)
    journey.step()
    journey.run_until_task_state(low, job_pb2.TASK_STATE_PENDING)
    high = journey.submit("interactive", priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE)
    journey.step()

    low_before_ack = journey.task(low)
    assert not low_before_ack.attempts[0].HasField("finished_at")
    assert journey.task(high).state == job_pb2.TASK_STATE_PENDING

    daemon.acknowledge_stops = True
    journey.step()
    assert journey.task(high).state == job_pb2.TASK_STATE_PENDING
    assert journey.task(low).attempts[0].HasField("finished_at")

    journey.run_until_task_state(high, job_pb2.TASK_STATE_RUNNING)
    assert journey.task(high).worker_id == "worker-a"
