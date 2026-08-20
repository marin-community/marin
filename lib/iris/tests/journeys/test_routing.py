# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whole-Job routing across multiple placement-owning backends."""

from iris.rpc import job_pb2


def test_jobs_route_to_one_matching_backend_without_cross_backend_leakage(multi_backend_journey):
    east = multi_backend_journey.submit("east", required_attributes={"region": "us-east1"})
    west = multi_backend_journey.submit("west", required_attributes={"region": "us-west1"})

    multi_backend_journey.settle()

    assert multi_backend_journey.task(east[0]).summary.backend_id == "east"
    assert multi_backend_journey.task(west[0]).summary.backend_id == "west"
    assert [event.task_id for event in multi_backend_journey.backend_events(kind="launched", backend_id="east")] == [
        east[0].wire_id
    ]
    assert [event.task_id for event in multi_backend_journey.backend_events(kind="launched", backend_id="west")] == [
        west[0].wire_id
    ]


def test_job_matching_no_backend_becomes_unschedulable(multi_backend_journey):
    job = multi_backend_journey.submit("nowhere", required_attributes={"region": "moon-1"})

    multi_backend_journey.settle()

    assert multi_backend_journey.task(job[0]).summary.state == job_pb2.TASK_STATE_UNSCHEDULABLE
    assert multi_backend_journey.backend_events(kind="launched") == []


def test_unplaceable_job_does_not_starve_a_later_placeable_job(mixed_capacity_journey):
    blocked = mixed_capacity_journey.submit("blocked", required_attributes={"region": "blocked"})
    ready = mixed_capacity_journey.submit("ready", required_attributes={"region": "ready"})

    mixed_capacity_journey.settle()

    assert mixed_capacity_journey.task(blocked[0]).summary.state == job_pb2.TASK_STATE_PENDING
    assert mixed_capacity_journey.task(ready[0]).summary.state == job_pb2.TASK_STATE_RUNNING
    assert [event.task_id for event in mixed_capacity_journey.backend_events(kind="launched", backend_id="ready")] == [
        ready[0].wire_id
    ]
