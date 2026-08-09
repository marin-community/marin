# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.resources.source import SourceState


def test_global_task_view_spans_jobs_and_backends_with_exact_current_attempts(multi_backend_journey):
    journey = multi_backend_journey
    east = journey.submit("east", tasks=2, required_attributes={"region": "us-east1"})
    west = journey.submit("west", tasks=3, required_attributes={"region": "us-west1"})
    journey.settle()

    page = journey.task_page()
    by_id = {task.identity.key.resource_id: task for task in page.items}

    assert set(by_id) == {east[index].wire_id for index in range(2)} | {west[index].wire_id for index in range(3)}
    assert {by_id[east[index].wire_id].backend_id for index in range(2)} == {"east"}
    assert {by_id[west[index].wire_id].backend_id for index in range(3)} == {"west"}
    assert len({task.identity.task_uid for task in page.items}) == 5
    assert len({task.current_attempt.attempt_uid for task in page.items if task.current_attempt is not None}) == 5

    east_page = journey.task_page(backend_id="east")
    assert {task.identity.key.resource_id for task in east_page.items} == {east[index].wire_id for index in range(2)}


def test_global_task_view_keeps_rows_when_one_backend_source_is_unavailable(multi_backend_journey):
    journey = multi_backend_journey
    east = journey.submit("east", required_attributes={"region": "us-east1"})
    west = journey.submit("west", required_attributes={"region": "us-west1"})
    journey.settle()
    journey.resource_source_outage("west")

    page = journey.task_page()
    statuses = {status.backend_id: status for status in page.source_statuses if status.backend_id}

    assert {task.identity.key.resource_id for task in page.items} == {east[0].wire_id, west[0].wire_id}
    assert statuses["east"].state is SourceState.AVAILABLE
    assert statuses["west"].state is SourceState.UNAVAILABLE
    assert statuses["west"].error_code == "backend_unavailable"
