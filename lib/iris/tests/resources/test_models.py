# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.endpoint import EndpointAccess, EndpointSummary
from iris.resources.errors import InvalidResourceKey
from iris.resources.identity import (
    AttemptLocator,
    JobIdentity,
    ResourceKey,
    ResourceKind,
)
from iris.resources.node import NodeAttribute, NodeAttributeKind
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
