# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codecs between native public resources and the ResourceService wire."""

from collections.abc import Mapping

from rigging.redaction import redact_value

from iris.cluster.constraints import AttributeValue, Constraint, ConstraintMode, ConstraintOp
from iris.resources.action import ActionKind, ActionReceipt, ActionResult, ActionState
from iris.resources.endpoint import (
    CpuProfileConfiguration,
    CpuProfileFormat,
    EndpointAccess,
    MemoryProfileConfiguration,
    MemoryProfileFormat,
    ProfileConfiguration,
    ThreadsProfileConfiguration,
)
from iris.resources.execution import (
    CommandEntrypoint,
    CpuDevice,
    Environment,
    GpuDevice,
    ResourceSpec,
    RuntimeEntrypoint,
    TpuDevice,
)
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
from iris.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    PriorityBand,
)
from iris.resources.node import NodeHealth
from iris.resources.slice import MembershipState, SliceCapacityState, SliceLifecycle
from iris.resources.source import Freshness, ResourceSourceStatus, SourceState
from iris.rpc import (
    resource_action_pb2,
    resource_command_pb2,
    resource_endpoint_pb2,
    resource_fleet_pb2,
    resource_identity_pb2,
    resource_job_pb2,
    resource_pb2,
)
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto

_CPU_FORMAT_TO_PROTO = {
    CpuProfileFormat.UNSPECIFIED: resource_command_pb2.CpuProfile.FORMAT_UNSPECIFIED,
    CpuProfileFormat.FLAMEGRAPH: resource_command_pb2.CpuProfile.FLAMEGRAPH,
    CpuProfileFormat.SPEEDSCOPE: resource_command_pb2.CpuProfile.SPEEDSCOPE,
    CpuProfileFormat.RAW: resource_command_pb2.CpuProfile.RAW,
}
_CPU_FORMAT_FROM_PROTO = {value: key for key, value in _CPU_FORMAT_TO_PROTO.items()}
_MEMORY_FORMAT_TO_PROTO = {
    MemoryProfileFormat.UNSPECIFIED: resource_command_pb2.MemoryProfile.FORMAT_UNSPECIFIED,
    MemoryProfileFormat.FLAMEGRAPH: resource_command_pb2.MemoryProfile.FLAMEGRAPH,
    MemoryProfileFormat.TABLE: resource_command_pb2.MemoryProfile.TABLE,
    MemoryProfileFormat.STATS: resource_command_pb2.MemoryProfile.STATS,
    MemoryProfileFormat.RAW: resource_command_pb2.MemoryProfile.RAW,
}
_MEMORY_FORMAT_FROM_PROTO = {value: key for key, value in _MEMORY_FORMAT_TO_PROTO.items()}
_RESOURCE_KIND_TO_PROTO = {
    ResourceKind.JOB: resource_identity_pb2.RESOURCE_KIND_JOB,
    ResourceKind.TASK: resource_identity_pb2.RESOURCE_KIND_TASK,
    ResourceKind.ATTEMPT: resource_identity_pb2.RESOURCE_KIND_ATTEMPT,
    ResourceKind.ENDPOINT: resource_identity_pb2.RESOURCE_KIND_ENDPOINT,
    ResourceKind.NODE: resource_identity_pb2.RESOURCE_KIND_NODE,
    ResourceKind.SLICE: resource_identity_pb2.RESOURCE_KIND_SLICE,
}
_RESOURCE_KIND_FROM_PROTO = {value: key for key, value in _RESOURCE_KIND_TO_PROTO.items()}
_ACTION_KIND_TO_PROTO = {
    ActionKind.CANCEL_JOB: resource_action_pb2.ACTION_KIND_CANCEL_JOB,
    ActionKind.RETRY_TASK: resource_action_pb2.ACTION_KIND_RETRY_TASK,
    ActionKind.TERMINATE_ATTEMPT: resource_action_pb2.ACTION_KIND_TERMINATE_ATTEMPT,
    ActionKind.FAIL_ATTEMPT: resource_action_pb2.ACTION_KIND_FAIL_ATTEMPT,
}
_ACTION_KIND_FROM_PROTO = {value: key for key, value in _ACTION_KIND_TO_PROTO.items()}
_ACTION_STATE_TO_PROTO = {
    ActionState.ACCEPTED: resource_action_pb2.ACTION_STATE_ACCEPTED,
    ActionState.VERIFYING: resource_action_pb2.ACTION_STATE_VERIFYING,
    ActionState.SUCCEEDED: resource_action_pb2.ACTION_STATE_SUCCEEDED,
    ActionState.FAILED: resource_action_pb2.ACTION_STATE_FAILED,
}
_ACTION_STATE_FROM_PROTO = {value: key for key, value in _ACTION_STATE_TO_PROTO.items()}
_ACTION_RESULT_TO_PROTO = {
    ActionResult.NONE: resource_action_pb2.ACTION_RESULT_NONE,
    ActionResult.SATISFIED: resource_action_pb2.ACTION_RESULT_SATISFIED,
    ActionResult.TARGET_ABSENT: resource_action_pb2.ACTION_RESULT_TARGET_ABSENT,
    ActionResult.PROVIDER_REJECTED: resource_action_pb2.ACTION_RESULT_PROVIDER_REJECTED,
    ActionResult.INTERNAL_ERROR: resource_action_pb2.ACTION_RESULT_INTERNAL_ERROR,
}
_ACTION_RESULT_FROM_PROTO = {value: key for key, value in _ACTION_RESULT_TO_PROTO.items()}
_SOURCE_STATE_TO_PROTO = {
    SourceState.AVAILABLE: resource_pb2.SOURCE_STATE_AVAILABLE,
    SourceState.UNAVAILABLE: resource_pb2.SOURCE_STATE_UNAVAILABLE,
    SourceState.UNSUPPORTED: resource_pb2.SOURCE_STATE_UNSUPPORTED,
}
_SOURCE_STATE_FROM_PROTO = {value: key for key, value in _SOURCE_STATE_TO_PROTO.items()}
_FRESHNESS_TO_PROTO = {
    Freshness.CURRENT: resource_pb2.FRESHNESS_CURRENT,
    Freshness.STALE: resource_pb2.FRESHNESS_STALE,
    Freshness.UNKNOWN: resource_pb2.FRESHNESS_UNKNOWN,
}
_FRESHNESS_FROM_PROTO = {value: key for key, value in _FRESHNESS_TO_PROTO.items()}
_NODE_HEALTH_TO_PROTO = {
    NodeHealth.READY: resource_fleet_pb2.NODE_HEALTH_READY,
    NodeHealth.DEGRADED: resource_fleet_pb2.NODE_HEALTH_DEGRADED,
    NodeHealth.UNAVAILABLE: resource_fleet_pb2.NODE_HEALTH_UNAVAILABLE,
    NodeHealth.RETIRED: resource_fleet_pb2.NODE_HEALTH_RETIRED,
}
_NODE_HEALTH_FROM_PROTO = {value: key for key, value in _NODE_HEALTH_TO_PROTO.items()}
_SLICE_LIFECYCLE_TO_PROTO = {
    SliceLifecycle.CREATING: resource_fleet_pb2.SLICE_LIFECYCLE_CREATING,
    SliceLifecycle.READY: resource_fleet_pb2.SLICE_LIFECYCLE_READY,
    SliceLifecycle.DELETING: resource_fleet_pb2.SLICE_LIFECYCLE_DELETING,
    SliceLifecycle.FAILED: resource_fleet_pb2.SLICE_LIFECYCLE_FAILED,
}
_SLICE_LIFECYCLE_FROM_PROTO = {value: key for key, value in _SLICE_LIFECYCLE_TO_PROTO.items()}
_MEMBERSHIP_STATE_TO_PROTO = {
    MembershipState.UNKNOWN: resource_fleet_pb2.MEMBERSHIP_STATE_UNKNOWN,
    MembershipState.OBSERVED: resource_fleet_pb2.MEMBERSHIP_STATE_OBSERVED,
}
_MEMBERSHIP_STATE_FROM_PROTO = {value: key for key, value in _MEMBERSHIP_STATE_TO_PROTO.items()}
_SLICE_CAPACITY_STATE_TO_PROTO = {
    SliceCapacityState.UNKNOWN: resource_fleet_pb2.SLICE_CAPACITY_STATE_UNKNOWN,
    SliceCapacityState.AVAILABLE: resource_fleet_pb2.SLICE_CAPACITY_STATE_AVAILABLE,
    SliceCapacityState.IN_USE: resource_fleet_pb2.SLICE_CAPACITY_STATE_IN_USE,
    SliceCapacityState.IDLE: resource_fleet_pb2.SLICE_CAPACITY_STATE_IDLE,
    SliceCapacityState.DEGRADED: resource_fleet_pb2.SLICE_CAPACITY_STATE_DEGRADED,
}
_SLICE_CAPACITY_STATE_FROM_PROTO = {value: key for key, value in _SLICE_CAPACITY_STATE_TO_PROTO.items()}
_ENDPOINT_ACCESS_TO_PROTO = {
    EndpointAccess.PRIVATE: resource_endpoint_pb2.ENDPOINT_ACCESS_PRIVATE,
    EndpointAccess.LINK: resource_endpoint_pb2.ENDPOINT_ACCESS_LINK,
}
_ENDPOINT_ACCESS_FROM_PROTO = {value: key for key, value in _ENDPOINT_ACCESS_TO_PROTO.items()}


def _enum_from_proto[T](mapping: Mapping[int, T], value: int, field_name: str) -> T:
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"{field_name} wire value is unspecified") from exc


def resource_key_to_proto(value: ResourceKey) -> resource_identity_pb2.ResourceKey:
    return resource_identity_pb2.ResourceKey(
        cluster_id=value.cluster_id,
        kind=_RESOURCE_KIND_TO_PROTO[value.kind],
        resource_id=value.resource_id,
    )


def resource_key_from_proto(value: resource_identity_pb2.ResourceKey) -> ResourceKey:
    kind = _enum_from_proto(_RESOURCE_KIND_FROM_PROTO, value.kind, "resource kind")
    return ResourceKey(value.cluster_id, kind, value.resource_id)


def job_identity_to_proto(value: JobIdentity) -> resource_identity_pb2.JobIdentity:
    return resource_identity_pb2.JobIdentity(key=resource_key_to_proto(value.key), job_uid=value.job_uid)


def job_identity_from_proto(value: resource_identity_pb2.JobIdentity) -> JobIdentity:
    return JobIdentity(resource_key_from_proto(value.key), value.job_uid)


def task_identity_to_proto(value: TaskIdentity) -> resource_identity_pb2.TaskIdentity:
    return resource_identity_pb2.TaskIdentity(key=resource_key_to_proto(value.key), task_uid=value.task_uid)


def task_identity_from_proto(value: resource_identity_pb2.TaskIdentity) -> TaskIdentity:
    return TaskIdentity(resource_key_from_proto(value.key), value.task_uid)


def attempt_identity_to_proto(value: AttemptIdentity) -> resource_identity_pb2.AttemptIdentity:
    return resource_identity_pb2.AttemptIdentity(
        task=resource_key_to_proto(value.task),
        attempt_number=value.attempt_number,
        attempt_uid=value.attempt_uid,
    )


def attempt_identity_from_proto(value: resource_identity_pb2.AttemptIdentity) -> AttemptIdentity:
    return AttemptIdentity(resource_key_from_proto(value.task), value.attempt_number, value.attempt_uid)


def attempt_locator_to_proto(value: AttemptLocator) -> resource_identity_pb2.AttemptLocator:
    result = resource_identity_pb2.AttemptLocator(task=resource_key_to_proto(value.task))
    if value.attempt_number is not None:
        result.attempt_number = value.attempt_number
    return result


def attempt_locator_from_proto(value: resource_identity_pb2.AttemptLocator) -> AttemptLocator:
    return AttemptLocator(
        resource_key_from_proto(value.task),
        value.attempt_number if value.HasField("attempt_number") else None,
    )


def node_identity_to_proto(value: NodeIdentity) -> resource_identity_pb2.NodeIdentity:
    return resource_identity_pb2.NodeIdentity(
        key=resource_key_to_proto(value.key),
        backend_id=value.backend_id,
        node_uid=value.node_uid,
    )


def node_identity_from_proto(value: resource_identity_pb2.NodeIdentity) -> NodeIdentity:
    return NodeIdentity(resource_key_from_proto(value.key), value.backend_id, value.node_uid)


def node_locator_to_proto(value: NodeLocator) -> resource_identity_pb2.NodeLocator:
    result = resource_identity_pb2.NodeLocator(key=resource_key_to_proto(value.key), backend_id=value.backend_id)
    if value.node_uid is not None:
        result.node_uid = value.node_uid
    return result


def node_locator_from_proto(value: resource_identity_pb2.NodeLocator) -> NodeLocator:
    return NodeLocator(
        key=resource_key_from_proto(value.key),
        backend_id=value.backend_id,
        node_uid=value.node_uid or None,
    )


def slice_identity_to_proto(value: SliceIdentity) -> resource_identity_pb2.SliceIdentity:
    return resource_identity_pb2.SliceIdentity(
        key=resource_key_to_proto(value.key),
        backend_id=value.backend_id,
        slice_uid=value.slice_uid,
    )


def slice_identity_from_proto(value: resource_identity_pb2.SliceIdentity) -> SliceIdentity:
    return SliceIdentity(resource_key_from_proto(value.key), value.backend_id, value.slice_uid)


def slice_locator_to_proto(value: SliceLocator) -> resource_identity_pb2.SliceLocator:
    result = resource_identity_pb2.SliceLocator(key=resource_key_to_proto(value.key), backend_id=value.backend_id)
    if value.slice_uid is not None:
        result.slice_uid = value.slice_uid
    return result


def slice_locator_from_proto(value: resource_identity_pb2.SliceLocator) -> SliceLocator:
    return SliceLocator(
        key=resource_key_from_proto(value.key),
        backend_id=value.backend_id,
        slice_uid=value.slice_uid or None,
    )


def source_state_to_proto(value: SourceState) -> int:
    return _SOURCE_STATE_TO_PROTO[value]


def source_state_from_proto(value: int) -> SourceState:
    return _enum_from_proto(_SOURCE_STATE_FROM_PROTO, value, "source state")


def freshness_to_proto(value: Freshness) -> int:
    return _FRESHNESS_TO_PROTO[value]


def freshness_from_proto(value: int) -> Freshness:
    return _enum_from_proto(_FRESHNESS_FROM_PROTO, value, "freshness")


def node_health_to_proto(value: NodeHealth) -> int:
    return _NODE_HEALTH_TO_PROTO[value]


def node_health_from_proto(value: int) -> NodeHealth:
    return _enum_from_proto(_NODE_HEALTH_FROM_PROTO, value, "node health")


def slice_lifecycle_to_proto(value: SliceLifecycle) -> int:
    return _SLICE_LIFECYCLE_TO_PROTO[value]


def slice_lifecycle_from_proto(value: int) -> SliceLifecycle:
    return _enum_from_proto(_SLICE_LIFECYCLE_FROM_PROTO, value, "slice lifecycle")


def membership_state_to_proto(value: MembershipState) -> int:
    return _MEMBERSHIP_STATE_TO_PROTO[value]


def membership_state_from_proto(value: int) -> MembershipState:
    return _enum_from_proto(_MEMBERSHIP_STATE_FROM_PROTO, value, "membership state")


def slice_capacity_state_to_proto(value: SliceCapacityState) -> int:
    return _SLICE_CAPACITY_STATE_TO_PROTO[value]


def slice_capacity_state_from_proto(value: int) -> SliceCapacityState:
    return _enum_from_proto(_SLICE_CAPACITY_STATE_FROM_PROTO, value, "slice capacity state")


def endpoint_access_to_proto(value: EndpointAccess) -> int:
    return _ENDPOINT_ACCESS_TO_PROTO[value]


def endpoint_access_from_proto(value: int) -> EndpointAccess:
    return _enum_from_proto(_ENDPOINT_ACCESS_FROM_PROTO, value, "endpoint access")


def resource_source_status_to_proto(value: ResourceSourceStatus) -> resource_pb2.ResourceSourceStatus:
    result = resource_pb2.ResourceSourceStatus(
        source_id=value.source_id,
        backend_id=value.backend_id,
        state=source_state_to_proto(value.state),
        freshness=freshness_to_proto(value.freshness),
        error_code=value.error_code,
        error_message=value.error_message,
    )
    if value.observed_at is not None:
        result.observed_at.CopyFrom(timestamp_to_proto(value.observed_at))
    return result


def resource_source_status_from_proto(value: resource_pb2.ResourceSourceStatus) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=value.source_id,
        backend_id=value.backend_id,
        state=source_state_from_proto(value.state),
        freshness=freshness_from_proto(value.freshness),
        observed_at=timestamp_from_proto(value.observed_at) if value.HasField("observed_at") else None,
        error_code=value.error_code,
        error_message=value.error_message,
    )


def action_receipt_to_proto(value: ActionReceipt) -> resource_action_pb2.ActionReceipt:
    result = resource_action_pb2.ActionReceipt(
        action_id=value.action_id,
        kind=_ACTION_KIND_TO_PROTO[value.kind],
        target=resource_key_to_proto(value.target),
        expected_target_uid=value.expected_target_uid,
        expected_attempt_uid=value.expected_attempt_uid or "",
        state=_ACTION_STATE_TO_PROTO[value.state],
        result_code=_ACTION_RESULT_TO_PROTO[value.result_code],
        result_message=value.result_message,
        created_at=timestamp_to_proto(value.created_at),
        updated_at=timestamp_to_proto(value.updated_at),
    )
    if value.completed_at is not None:
        result.completed_at.CopyFrom(timestamp_to_proto(value.completed_at))
    if value.expected_attempt_number is not None:
        result.expected_attempt_number = value.expected_attempt_number
    return result


def action_receipt_from_proto(value: resource_action_pb2.ActionReceipt) -> ActionReceipt:
    kind = _enum_from_proto(_ACTION_KIND_FROM_PROTO, value.kind, "action kind")
    state = _enum_from_proto(_ACTION_STATE_FROM_PROTO, value.state, "action state")
    result = _enum_from_proto(_ACTION_RESULT_FROM_PROTO, value.result_code, "action result")
    return ActionReceipt(
        action_id=value.action_id,
        kind=kind,
        target=resource_key_from_proto(value.target),
        expected_target_uid=value.expected_target_uid,
        expected_attempt_uid=value.expected_attempt_uid or None,
        state=state,
        result_code=result,
        result_message=value.result_message,
        created_at=timestamp_from_proto(value.created_at),
        updated_at=timestamp_from_proto(value.updated_at),
        completed_at=timestamp_from_proto(value.completed_at) if value.HasField("completed_at") else None,
        expected_attempt_number=value.expected_attempt_number if value.HasField("expected_attempt_number") else None,
    )


def device_to_proto(value: CpuDevice | GpuDevice | TpuDevice) -> resource_job_pb2.DeviceConfig:
    if isinstance(value, CpuDevice):
        return resource_job_pb2.DeviceConfig(cpu=resource_job_pb2.CpuDevice(variant=value.variant))
    if isinstance(value, GpuDevice):
        return resource_job_pb2.DeviceConfig(gpu=resource_job_pb2.GpuDevice(variant=value.variant, count=value.count))
    return resource_job_pb2.DeviceConfig(
        tpu=resource_job_pb2.TpuDevice(variant=value.variant, topology=value.topology, count=value.count)
    )


def device_from_proto(value: resource_job_pb2.DeviceConfig) -> CpuDevice | GpuDevice | TpuDevice | None:
    """Decode a device, treating an empty message as no device."""
    match value.WhichOneof("device"):
        case "cpu":
            return CpuDevice(variant=value.cpu.variant)
        case "gpu":
            return GpuDevice(variant=value.gpu.variant, count=value.gpu.count or 1)
        case "tpu":
            return TpuDevice(variant=value.tpu.variant, topology=value.tpu.topology, count=value.tpu.count)
        case _:
            if value.ByteSize() == 0:
                return None
            raise ValueError("device has no selected kind")


def resource_spec_to_proto(value: ResourceSpec) -> resource_job_pb2.ResourceSpecProto:
    result = resource_job_pb2.ResourceSpecProto(
        cpu_millicores=value.cpu_millicores,
        memory_bytes=value.memory,
        disk_bytes=value.disk,
    )
    if value.device is not None:
        result.device.CopyFrom(device_to_proto(value.device))
    return result


def resource_spec_from_proto(value: resource_job_pb2.ResourceSpecProto) -> ResourceSpec:
    return ResourceSpec(
        cpu=value.cpu_millicores / 1_000,
        memory=value.memory_bytes,
        disk=value.disk_bytes,
        device=device_from_proto(value.device),
    )


def environment_to_proto(value: Environment) -> resource_job_pb2.EnvironmentConfig:
    return resource_job_pb2.EnvironmentConfig(env_vars=dict(value.env_vars), setup_scripts=value.setup_scripts)


def environment_from_proto(value: resource_job_pb2.EnvironmentConfig) -> Environment:
    return Environment(env_vars=dict(value.env_vars), setup_scripts=tuple(value.setup_scripts))


def runtime_entrypoint_to_proto(value: RuntimeEntrypoint) -> resource_job_pb2.RuntimeEntrypoint:
    return resource_job_pb2.RuntimeEntrypoint(
        setup_commands=value.setup_commands,
        run_command=resource_job_pb2.CommandEntrypoint(argv=value.run_command.argv),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def runtime_entrypoint_from_proto(value: resource_job_pb2.RuntimeEntrypoint) -> RuntimeEntrypoint:
    return RuntimeEntrypoint(
        setup_commands=tuple(value.setup_commands),
        run_command=CommandEntrypoint(tuple(value.run_command.argv)),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def _attribute_value_to_proto(value: AttributeValue) -> resource_job_pb2.AttributeValue:
    result = resource_job_pb2.AttributeValue()
    if isinstance(value.value, str):
        result.string_value = value.value
    elif isinstance(value.value, int):
        result.int_value = value.value
    else:
        result.float_value = value.value
    return result


def _attribute_value_from_proto(value: resource_job_pb2.AttributeValue) -> AttributeValue:
    match value.WhichOneof("value"):
        case "string_value":
            return AttributeValue(value.string_value)
        case "int_value":
            return AttributeValue(value.int_value)
        case "float_value":
            return AttributeValue(value.float_value)
        case _:
            return AttributeValue("")


def constraint_to_proto(value: Constraint) -> resource_job_pb2.Constraint:
    result = resource_job_pb2.Constraint(key=value.key, op=int(value.op), mode=int(value.mode))
    if value.op is ConstraintOp.IN:
        result.values.extend(_attribute_value_to_proto(item) for item in value.values)
    elif value.values:
        result.value.CopyFrom(_attribute_value_to_proto(value.values[0]))
    return result


def constraint_from_proto(value: resource_job_pb2.Constraint) -> Constraint:
    op = ConstraintOp(value.op)
    if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
        values = ()
    elif op is ConstraintOp.IN:
        values = tuple(_attribute_value_from_proto(item) for item in value.values)
    else:
        values = (_attribute_value_from_proto(value.value),)
    return Constraint(key=value.key, op=op, values=values, mode=ConstraintMode(value.mode))


def job_spec_to_proto(value: JobSpec) -> resource_job_pb2.JobSpec:
    result = resource_job_pb2.JobSpec(
        version=value.version,
        name=value.name,
        entrypoint=runtime_entrypoint_to_proto(value.entrypoint),
        resources=resource_spec_to_proto(value.resources),
        environment=environment_to_proto(value.environment),
        bundle_id=value.bundle_id,
        ports=value.ports,
        max_task_failures=value.max_task_failures,
        max_retries_failure=value.max_retries_failure,
        max_retries_preemption=value.max_retries_preemption,
        constraints=[constraint_to_proto(item) for item in value.constraints],
        replicas=value.replicas,
        fail_if_exists=value.fail_if_exists,
        preemption_policy=int(value.preemption_policy),
        existing_job_policy=int(value.existing_job_policy),
        priority_band=int(value.priority_band),
        task_image=value.task_image,
        submit_argv=value.submit_argv,
        client_revision_date=value.client_revision_date,
        container_profile=int(value.container_profile),
    )
    if value.scheduling_timeout is not None:
        result.scheduling_timeout.CopyFrom(duration_to_proto(value.scheduling_timeout))
    if value.coscheduling is not None:
        result.coscheduling.group_by = value.coscheduling.group_by
    if value.timeout is not None:
        result.timeout.CopyFrom(duration_to_proto(value.timeout))
    return result


def redacted_job_spec_to_proto(value: JobSpec) -> resource_job_pb2.JobSpec:
    """Encode a Job spec without exposing submitted secrets or workdir payloads."""
    result = job_spec_to_proto(value)
    redacted_env = redact_value(dict(result.environment.env_vars))
    assert isinstance(redacted_env, dict)
    result.environment.env_vars.clear()
    result.environment.env_vars.update(redacted_env)
    result.entrypoint.workdir_files.clear()
    result.entrypoint.workdir_file_refs.clear()
    return result


def job_spec_from_proto(value: resource_job_pb2.JobSpec) -> JobSpec:
    return JobSpec(
        version=value.version,
        name=value.name,
        entrypoint=runtime_entrypoint_from_proto(value.entrypoint),
        resources=resource_spec_from_proto(value.resources),
        environment=environment_from_proto(value.environment),
        bundle_id=value.bundle_id,
        scheduling_timeout=(
            duration_from_proto(value.scheduling_timeout) if value.HasField("scheduling_timeout") else None
        ),
        ports=tuple(value.ports),
        max_task_failures=value.max_task_failures,
        max_retries_failure=value.max_retries_failure,
        max_retries_preemption=value.max_retries_preemption,
        constraints=tuple(constraint_from_proto(item) for item in value.constraints),
        coscheduling=(
            CoschedulingConfig(group_by=value.coscheduling.group_by) if value.HasField("coscheduling") else None
        ),
        replicas=value.replicas,
        timeout=duration_from_proto(value.timeout) if value.HasField("timeout") else None,
        fail_if_exists=value.fail_if_exists,
        preemption_policy=JobPreemptionPolicy(value.preemption_policy),
        existing_job_policy=ExistingJobPolicy(value.existing_job_policy),
        priority_band=PriorityBand(value.priority_band),
        task_image=value.task_image,
        submit_argv=tuple(value.submit_argv),
        client_revision_date=value.client_revision_date,
        container_profile=ContainerProfile(value.container_profile),
    )


def profile_configuration_to_proto(value: ProfileConfiguration | None) -> resource_command_pb2.ProfileType:
    if isinstance(value, CpuProfileConfiguration):
        cpu = resource_command_pb2.CpuProfile(format=_CPU_FORMAT_TO_PROTO[value.format], rate_hz=value.rate_hz)
        if value.native is not None:
            cpu.native = value.native
        return resource_command_pb2.ProfileType(cpu=cpu)
    if isinstance(value, MemoryProfileConfiguration):
        return resource_command_pb2.ProfileType(
            memory=resource_command_pb2.MemoryProfile(format=_MEMORY_FORMAT_TO_PROTO[value.format], leaks=value.leaks)
        )
    if isinstance(value, ThreadsProfileConfiguration):
        return resource_command_pb2.ProfileType(threads=resource_command_pb2.ThreadsProfile(locals=value.include_locals))
    return resource_command_pb2.ProfileType()


def profile_configuration_from_proto(value: resource_command_pb2.ProfileType) -> ProfileConfiguration | None:
    match value.WhichOneof("profiler"):
        case "cpu":
            return CpuProfileConfiguration(
                format=_CPU_FORMAT_FROM_PROTO.get(value.cpu.format, CpuProfileFormat.UNSPECIFIED),
                rate_hz=value.cpu.rate_hz,
                native=value.cpu.native if value.cpu.HasField("native") else None,
            )
        case "memory":
            return MemoryProfileConfiguration(
                format=_MEMORY_FORMAT_FROM_PROTO.get(value.memory.format, MemoryProfileFormat.UNSPECIFIED),
                leaks=value.memory.leaks,
            )
        case "threads":
            return ThreadsProfileConfiguration(include_locals=value.threads.locals)
        case _:
            return None
