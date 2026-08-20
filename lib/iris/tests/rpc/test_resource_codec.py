# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.resources.endpoint import EndpointAccess, ThreadsProfileConfiguration
from iris.resources.identity import (
    AttemptLocator,
    NodeLocator,
    ResourceKey,
    ResourceKind,
    SliceLocator,
)
from iris.rpc import (
    job_pb2,
    resource_action_pb2,
    resource_command_pb2,
    resource_endpoint_pb2,
    resource_identity_pb2,
    resource_job_pb2,
)
from iris.rpc.profile_codec import (
    profile_configuration_from_proto as legacy_profile_configuration_from_proto,
)
from iris.rpc.profile_codec import (
    profile_configuration_to_proto as legacy_profile_configuration_to_proto,
)
from iris.rpc.resource_codec import (
    action_receipt_from_proto,
    attempt_locator_from_proto,
    endpoint_access_from_proto,
    endpoint_access_to_proto,
    job_spec_from_proto,
    node_locator_from_proto,
    profile_configuration_from_proto,
    profile_configuration_to_proto,
    redacted_job_spec_to_proto,
    resource_key_from_proto,
    resource_key_to_proto,
    slice_locator_from_proto,
)


def _wire_key(kind: int, resource_id: str) -> resource_identity_pb2.ResourceKey:
    return resource_identity_pb2.ResourceKey(cluster_id="cluster", kind=kind, resource_id=resource_id)


@pytest.mark.parametrize(
    ("kind", "wire_kind", "resource_id"),
    [
        (ResourceKind.JOB, resource_identity_pb2.RESOURCE_KIND_JOB, "/owner/job"),
        (ResourceKind.TASK, resource_identity_pb2.RESOURCE_KIND_TASK, "/owner/job/0"),
        (ResourceKind.ATTEMPT, resource_identity_pb2.RESOURCE_KIND_ATTEMPT, "/owner/job/0:2"),
        (ResourceKind.ENDPOINT, resource_identity_pb2.RESOURCE_KIND_ENDPOINT, "endpoint-id"),
        (ResourceKind.NODE, resource_identity_pb2.RESOURCE_KIND_NODE, "node-id"),
        (ResourceKind.SLICE, resource_identity_pb2.RESOURCE_KIND_SLICE, "slice-id"),
    ],
)
def test_resource_key_codec_uses_stable_wire_kinds(kind: ResourceKind, wire_kind: int, resource_id: str) -> None:
    key = ResourceKey("cluster", kind, resource_id)
    wire = _wire_key(wire_kind, resource_id)

    assert resource_key_to_proto(key) == wire
    assert resource_key_from_proto(wire) == key


def test_locator_decoders_preserve_optional_exact_identity() -> None:
    task_key = ResourceKey("cluster", ResourceKind.TASK, "/owner/job/0")
    wire_task = _wire_key(resource_identity_pb2.RESOURCE_KIND_TASK, task_key.resource_id)
    assert attempt_locator_from_proto(resource_identity_pb2.AttemptLocator(task=wire_task)) == AttemptLocator(
        task_key, None
    )
    assert attempt_locator_from_proto(
        resource_identity_pb2.AttemptLocator(task=wire_task, attempt_number=0)
    ) == AttemptLocator(task_key, 0)

    node_key = ResourceKey("cluster", ResourceKind.NODE, "node")
    wire_node = _wire_key(resource_identity_pb2.RESOURCE_KIND_NODE, node_key.resource_id)
    assert node_locator_from_proto(
        resource_identity_pb2.NodeLocator(key=wire_node, backend_id="backend")
    ) == NodeLocator(node_key, "backend")
    assert node_locator_from_proto(
        resource_identity_pb2.NodeLocator(key=wire_node, backend_id="backend", node_uid="node-uid")
    ) == NodeLocator(node_key, "backend", "node-uid")

    slice_key = ResourceKey("cluster", ResourceKind.SLICE, "slice")
    wire_slice = _wire_key(resource_identity_pb2.RESOURCE_KIND_SLICE, slice_key.resource_id)
    assert slice_locator_from_proto(
        resource_identity_pb2.SliceLocator(key=wire_slice, backend_id="backend")
    ) == SliceLocator(slice_key, "backend")
    assert slice_locator_from_proto(
        resource_identity_pb2.SliceLocator(key=wire_slice, backend_id="backend", slice_uid="slice-uid")
    ) == SliceLocator(slice_key, "backend", "slice-uid")


def test_endpoint_access_preserves_private_zero_wire_value() -> None:
    assert endpoint_access_to_proto(EndpointAccess.PRIVATE) == resource_endpoint_pb2.ENDPOINT_ACCESS_PRIVATE
    assert endpoint_access_from_proto(resource_endpoint_pb2.ENDPOINT_ACCESS_PRIVATE) is EndpointAccess.PRIVATE


def test_profile_codecs_preserve_thread_detail_options() -> None:
    profile = ThreadsProfileConfiguration(include_locals=True, include_native=True)
    resource_wire = resource_command_pb2.ProfileType(
        threads=resource_command_pb2.ThreadsProfile(locals=True, native=True)
    )
    legacy_wire = job_pb2.ProfileType(threads=job_pb2.ThreadsProfile(locals=True, native=True))

    assert profile_configuration_to_proto(profile) == resource_wire
    assert profile_configuration_from_proto(resource_wire) == profile
    assert legacy_profile_configuration_to_proto(profile) == legacy_wire
    assert legacy_profile_configuration_from_proto(legacy_wire) == profile


def test_shared_decoders_reject_unspecified_resource_and_action_enums() -> None:
    with pytest.raises(ValueError, match="resource kind wire value"):
        resource_key_from_proto(resource_identity_pb2.ResourceKey(cluster_id="cluster", resource_id="node"))
    with pytest.raises(ValueError, match="action kind wire value"):
        action_receipt_from_proto(resource_action_pb2.ActionReceipt())


def test_redacted_job_spec_omits_secrets_and_workdir_payloads() -> None:
    spec = job_spec_from_proto(
        resource_job_pb2.JobSpec(
            name="/owner/job",
            environment=resource_job_pb2.EnvironmentConfig(
                env_vars={"WANDB_API_KEY": "secret", "SAFE": "visible"},
            ),
            entrypoint=resource_job_pb2.RuntimeEntrypoint(
                workdir_files={"secret.txt": b"payload"},
                workdir_file_refs={"model": "gs://private/model"},
            ),
        )
    )

    redacted = redacted_job_spec_to_proto(spec)

    assert redacted.environment.env_vars == {"WANDB_API_KEY": "[REDACTED]", "SAFE": "visible"}
    assert redacted.entrypoint.workdir_files == {}
    assert redacted.entrypoint.workdir_file_refs == {}
