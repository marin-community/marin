# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from iris.cluster.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.cluster.resources.activity import ActivityEntry
from iris.cluster.resources.attempt import AttemptSummary
from iris.cluster.resources.endpoint import EndpointAccess, EndpointSummary
from iris.cluster.resources.errors import InvalidResourceKey
from iris.cluster.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.cluster.resources.node import NodeAttribute, NodeAttributeKind
from iris.cluster.resources.source import Freshness, ResourceSourceStatus, SourceState
from iris.cluster.resources.task import TaskDetail, TaskSummary
from iris.rpc import job_pb2
from rigging.timing import Timestamp

NOW = Timestamp.from_ms(1)


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(lambda: ResourceKey(" ", ResourceKind.JOB, "/owner/job"), id="blank-cluster"),
        pytest.param(lambda: ResourceKey("cluster", ResourceKind.ATTEMPT, "/owner/job:0:not-a-number"), id="attempt"),
        pytest.param(
            lambda: JobIdentity(ResourceKey("cluster", ResourceKind.TASK, "/owner/job:0"), "job-uid"),
            id="wrong-kind",
        ),
        pytest.param(
            lambda: AttemptLocator(ResourceKey("cluster", ResourceKind.TASK, "/owner/job:0"), -1),
            id="negative-attempt",
        ),
    ],
)
def test_identity_with_malformed_coordinates_raises_public_key_error(construct: Callable[[], object]) -> None:
    with pytest.raises(InvalidResourceKey):
        construct()


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(
            lambda: ResourceSourceStatus(
                source_id="backend:gpu",
                backend_id="gpu",
                state=SourceState.AVAILABLE,
                freshness=Freshness.CURRENT,
                observed_at=None,
                error_code="",
                error_message="",
            ),
            id="available-without-observation",
        ),
        pytest.param(
            lambda: ResourceSourceStatus(
                source_id="controller:cluster",
                backend_id="",
                state=SourceState.UNAVAILABLE,
                freshness=Freshness.UNKNOWN,
                observed_at=None,
                error_code="",
                error_message="offline",
            ),
            id="unavailable-without-code",
        ),
        pytest.param(
            lambda: ResourceSourceStatus(
                source_id="backend:gpu",
                backend_id="other",
                state=SourceState.AVAILABLE,
                freshness=Freshness.CURRENT,
                observed_at=NOW,
                error_code="",
                error_message="",
            ),
            id="backend-coordinate-mismatch",
        ),
        pytest.param(
            lambda: ResourceSourceStatus(
                source_id="backend:gpu",
                backend_id="gpu",
                state=SourceState.UNSUPPORTED,
                freshness=Freshness.STALE,
                observed_at=None,
                error_code="unsupported",
                error_message="",
            ),
            id="unsupported-with-stale-freshness",
        ),
        pytest.param(
            lambda: ResourceSourceStatus(
                source_id="backend:gpu",
                backend_id="gpu",
                state=SourceState.UNAVAILABLE,
                freshness=Freshness.UNKNOWN,
                observed_at=None,
                error_code="e" * 65,
                error_message="offline",
            ),
            id="oversized-code",
        ),
    ],
)
def test_source_status_with_contradictory_or_unbounded_data_is_rejected(construct: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        construct()


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(
            lambda: NodeAttribute(key="region", kind=NodeAttributeKind.STRING),
            id="selected-value-missing",
        ),
        pytest.param(
            lambda: NodeAttribute(
                key="region",
                kind=NodeAttributeKind.STRING,
                string_value="us-east1",
                integer_value=1,
            ),
            id="multiple-values",
        ),
        pytest.param(
            lambda: NodeAttribute(key="cores", kind=NodeAttributeKind.INTEGER, string_value="eight"),
            id="wrong-value-kind",
        ),
    ],
)
def test_node_attribute_requires_exactly_the_selected_oneof_value(construct: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        construct()


def _activity_entry(
    *,
    entry_id: str = "finelog:iris.task_event:1",
    message: str = "running",
    attributes: dict[str, str] | None = None,
) -> ActivityEntry:
    return ActivityEntry(
        entry_id=entry_id,
        occurred_at=NOW,
        source="finelog:cluster",
        severity="info",
        kind="task_state",
        message=message,
        target=ResourceKey("cluster", ResourceKind.TASK, "/owner/job:0"),
        attempt_uid=None,
        correlation_id=None,
        attributes=attributes or {},
    )


@pytest.mark.parametrize(
    "construct",
    [
        pytest.param(lambda: _activity_entry(entry_id="finelog:missing-sequence"), id="entry-id"),
        pytest.param(lambda: _activity_entry(message="x" * 2_049), id="message"),
        pytest.param(
            lambda: _activity_entry(attributes={str(index): "v" for index in range(33)}),
            id="attribute-count",
        ),
        pytest.param(lambda: _activity_entry(attributes={"k" * 65: "v"}), id="attribute-key"),
        pytest.param(lambda: _activity_entry(attributes={"k": "v" * 513}), id="attribute-value"),
    ],
)
def test_activity_entry_rejects_malformed_identity_or_unbounded_payload(construct: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        construct()


def _failed_task_detail(root_cause_highlights: tuple[str, ...]) -> TaskDetail:
    job = JobIdentity(ResourceKey("cluster", ResourceKind.JOB, "/owner/job"), "job-uid")
    task_key = ResourceKey("cluster", ResourceKind.TASK, "/owner/job:0")
    task = TaskIdentity(task_key, "task-uid")
    attempt = AttemptIdentity(task_key, 0, "attempt-uid")
    summary = TaskSummary(
        identity=task,
        job=job,
        task_index=0,
        state=job_pb2.TASK_STATE_FAILED,
        execution_cluster_id="cluster",
        backend_id="backend",
        current_attempt=attempt,
        current_node=None,
        failure_count=1,
        preemption_count=0,
        submitted_at=NOW,
        started_at=NOW,
        finished_at=NOW,
        status_message="",
        error_message="failed",
    )
    attempt_summary = AttemptSummary(
        identity=attempt,
        state=job_pb2.TASK_STATE_FAILED,
        execution_cluster_id="cluster",
        backend_id="backend",
        node=None,
        created_at=NOW,
        started_at=NOW,
        finished_at=NOW,
        exit_code=1,
        error_message="failed",
        terminal_reason="application",
    )
    return TaskDetail(
        summary=summary,
        attempts=(attempt_summary,),
        source_statuses=(),
        root_cause_highlights=root_cause_highlights,
    )


@pytest.mark.parametrize(
    "highlights",
    [
        pytest.param(tuple("line" for _ in range(21)), id="count"),
        pytest.param(("x" * 513,), id="line-length"),
    ],
)
def test_task_detail_rejects_unbounded_root_cause_highlights(highlights: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        _failed_task_detail(highlights)


def test_action_receipt_rejects_a_target_kind_that_cannot_receive_the_action() -> None:
    with pytest.raises(InvalidResourceKey):
        ActionReceipt(
            action_id="action-1",
            kind=ActionKind.CANCEL_JOB,
            target=ResourceKey("cluster", ResourceKind.TASK, "/owner/job:0"),
            expected_target_uid="task-uid",
            expected_attempt_uid=None,
            state=ActionState.ACCEPTED,
            result_code=ActionResult.NONE,
            result_message="",
            created_at=NOW,
            updated_at=NOW,
            completed_at=None,
        )


def test_endpoint_summary_rejects_divergent_key_and_registration_identity() -> None:
    with pytest.raises(InvalidResourceKey):
        EndpointSummary(
            key=ResourceKey("cluster", ResourceKind.ENDPOINT, "endpoint-a"),
            endpoint_id="endpoint-b",
            name="dashboard",
            task=None,
            execution_cluster_id="cluster",
            access=EndpointAccess.PRIVATE,
            lease_deadline=None,
        )
