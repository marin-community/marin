# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the shared native-resource wire codecs."""

import pytest
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.endpoint import EndpointAccess
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeIdentity,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceIdentity,
    SliceLocator,
    TaskIdentity,
)
from iris.resources.source import Freshness, ResourceSourceStatus, SourceState
from iris.rpc import resource_pb2
from iris.rpc.resource_codec import (
    action_receipt_from_proto,
    action_receipt_to_proto,
    attempt_identity_from_proto,
    attempt_identity_to_proto,
    attempt_locator_from_proto,
    attempt_locator_to_proto,
    endpoint_access_from_proto,
    endpoint_access_to_proto,
    job_identity_from_proto,
    job_identity_to_proto,
    job_spec_from_proto,
    node_identity_from_proto,
    node_identity_to_proto,
    node_locator_from_proto,
    node_locator_to_proto,
    redacted_job_spec_to_proto,
    resource_key_from_proto,
    resource_key_to_proto,
    resource_source_status_from_proto,
    resource_source_status_to_proto,
    slice_identity_from_proto,
    slice_identity_to_proto,
    slice_locator_from_proto,
    slice_locator_to_proto,
    task_identity_from_proto,
    task_identity_to_proto,
)
from rigging.timing import Timestamp


@pytest.mark.parametrize(
    "key",
    [
        ResourceKey("cluster", ResourceKind.JOB, "/owner/job"),
        ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0"),
        ResourceKey("cluster", ResourceKind.ATTEMPT, "/owner/job/0:2"),
        ResourceKey("cluster", ResourceKind.ENDPOINT, "endpoint-id"),
        ResourceKey("cluster", ResourceKind.NODE, "node-id"),
        ResourceKey("cluster", ResourceKind.SLICE, "slice-id"),
    ],
)
def test_resource_key_codec_round_trips_every_kind(key: ResourceKey) -> None:
    assert resource_key_from_proto(resource_key_to_proto(key)) == key


def test_exact_identity_codecs_round_trip_resource_incarnations() -> None:
    job = JobIdentity(ResourceKey("cluster", ResourceKind.JOB, "/owner/job"), "job-uid")
    task = TaskIdentity(ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0"), "task-uid")
    attempt = AttemptIdentity(task.key, 0, "attempt-uid")
    node = NodeIdentity(ResourceKey("cluster", ResourceKind.NODE, "node"), "backend", "node-uid")
    slice_identity = SliceIdentity(ResourceKey("cluster", ResourceKind.SLICE, "slice"), "backend", "slice-uid")

    assert job_identity_from_proto(job_identity_to_proto(job)) == job
    assert task_identity_from_proto(task_identity_to_proto(task)) == task
    assert attempt_identity_from_proto(attempt_identity_to_proto(attempt)) == attempt
    assert node_identity_from_proto(node_identity_to_proto(node)) == node
    assert slice_identity_from_proto(slice_identity_to_proto(slice_identity)) == slice_identity


@pytest.mark.parametrize(
    ("locator", "to_proto", "from_proto"),
    [
        (
            AttemptLocator(ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0"), None),
            attempt_locator_to_proto,
            attempt_locator_from_proto,
        ),
        (
            AttemptLocator(ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0"), 0),
            attempt_locator_to_proto,
            attempt_locator_from_proto,
        ),
        (
            NodeLocator(ResourceKey("cluster", ResourceKind.NODE, "node"), "backend"),
            node_locator_to_proto,
            node_locator_from_proto,
        ),
        (
            NodeLocator(ResourceKey("cluster", ResourceKind.NODE, "node"), "backend", "node-uid"),
            node_locator_to_proto,
            node_locator_from_proto,
        ),
        (
            SliceLocator(ResourceKey("cluster", ResourceKind.SLICE, "slice"), "backend"),
            slice_locator_to_proto,
            slice_locator_from_proto,
        ),
        (
            SliceLocator(ResourceKey("cluster", ResourceKind.SLICE, "slice"), "backend", "slice-uid"),
            slice_locator_to_proto,
            slice_locator_from_proto,
        ),
    ],
)
def test_locator_codecs_preserve_optional_exact_identity(locator, to_proto, from_proto) -> None:
    assert from_proto(to_proto(locator)) == locator


def test_status_and_action_codecs_preserve_presence_and_zero_valued_enums() -> None:
    observed_at = Timestamp.from_ms(1_000)
    source = ResourceSourceStatus(
        source_id="controller:cluster",
        backend_id="",
        state=SourceState.AVAILABLE,
        freshness=Freshness.CURRENT,
        observed_at=observed_at,
        error_code="",
        error_message="",
    )
    receipt = ActionReceipt(
        action_id="action-id",
        kind=ActionKind.RETRY_TASK,
        target=ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0"),
        expected_target_uid="task-uid",
        expected_attempt_uid="attempt-uid",
        expected_attempt_number=0,
        state=ActionState.SUCCEEDED,
        result_code=ActionResult.SATISFIED,
        result_message="",
        created_at=observed_at,
        updated_at=observed_at,
        completed_at=observed_at,
    )

    assert endpoint_access_to_proto(EndpointAccess.PRIVATE) == resource_pb2.ENDPOINT_ACCESS_PRIVATE
    assert endpoint_access_from_proto(resource_pb2.ENDPOINT_ACCESS_PRIVATE) is EndpointAccess.PRIVATE
    assert resource_source_status_from_proto(resource_source_status_to_proto(source)) == source
    assert action_receipt_from_proto(action_receipt_to_proto(receipt)) == receipt


def test_shared_decoders_reject_unspecified_resource_and_action_enums() -> None:
    with pytest.raises(ValueError, match="resource kind wire value"):
        resource_key_from_proto(resource_pb2.ResourceKey(cluster_id="cluster", resource_id="node"))
    with pytest.raises(ValueError, match="action kind wire value"):
        action_receipt_from_proto(resource_pb2.ActionReceipt())


def test_redacted_job_spec_omits_secrets_and_workdir_payloads() -> None:
    spec = job_spec_from_proto(
        resource_pb2.JobSpec(
            name="/owner/job",
            environment=resource_pb2.EnvironmentConfig(
                env_vars={"WANDB_API_KEY": "secret", "SAFE": "visible"},
            ),
            entrypoint=resource_pb2.RuntimeEntrypoint(
                workdir_files={"secret.txt": b"payload"},
                workdir_file_refs={"model": "gs://private/model"},
            ),
        )
    )

    redacted = redacted_job_spec_to_proto(spec)

    assert redacted.environment.env_vars == {"WANDB_API_KEY": "[REDACTED]", "SAFE": "visible"}
    assert redacted.entrypoint.workdir_files == {}
    assert redacted.entrypoint.workdir_file_refs == {}
