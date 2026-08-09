# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codecs between native public resources and the ResourceService wire."""

from iris.cluster.constraints import AttributeValue, Constraint, ConstraintMode, ConstraintOp
from iris.cluster.resources.endpoint import (
    CpuProfileConfiguration,
    CpuProfileFormat,
    MemoryProfileConfiguration,
    MemoryProfileFormat,
    ProfileConfiguration,
    ThreadsProfileConfiguration,
)
from iris.cluster.resources.execution import (
    CommandEntrypoint,
    CpuDevice,
    Environment,
    GpuDevice,
    ResourceSpec,
    RuntimeEntrypoint,
    TpuDevice,
)
from iris.cluster.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    PriorityBand,
)
from iris.rpc import resource_pb2
from iris.time_proto import duration_from_proto, duration_to_proto

_CPU_FORMAT_TO_PROTO = {
    CpuProfileFormat.UNSPECIFIED: resource_pb2.CpuProfile.FORMAT_UNSPECIFIED,
    CpuProfileFormat.FLAMEGRAPH: resource_pb2.CpuProfile.FLAMEGRAPH,
    CpuProfileFormat.SPEEDSCOPE: resource_pb2.CpuProfile.SPEEDSCOPE,
    CpuProfileFormat.RAW: resource_pb2.CpuProfile.RAW,
}
_CPU_FORMAT_FROM_PROTO = {value: key for key, value in _CPU_FORMAT_TO_PROTO.items()}
_MEMORY_FORMAT_TO_PROTO = {
    MemoryProfileFormat.UNSPECIFIED: resource_pb2.MemoryProfile.FORMAT_UNSPECIFIED,
    MemoryProfileFormat.FLAMEGRAPH: resource_pb2.MemoryProfile.FLAMEGRAPH,
    MemoryProfileFormat.TABLE: resource_pb2.MemoryProfile.TABLE,
    MemoryProfileFormat.STATS: resource_pb2.MemoryProfile.STATS,
    MemoryProfileFormat.RAW: resource_pb2.MemoryProfile.RAW,
}
_MEMORY_FORMAT_FROM_PROTO = {value: key for key, value in _MEMORY_FORMAT_TO_PROTO.items()}


def device_to_proto(value: CpuDevice | GpuDevice | TpuDevice) -> resource_pb2.DeviceConfig:
    if isinstance(value, CpuDevice):
        return resource_pb2.DeviceConfig(cpu=resource_pb2.CpuDevice(variant=value.variant))
    if isinstance(value, GpuDevice):
        return resource_pb2.DeviceConfig(gpu=resource_pb2.GpuDevice(variant=value.variant, count=value.count))
    return resource_pb2.DeviceConfig(
        tpu=resource_pb2.TpuDevice(variant=value.variant, topology=value.topology, count=value.count)
    )


def device_from_proto(value: resource_pb2.DeviceConfig) -> CpuDevice | GpuDevice | TpuDevice:
    match value.WhichOneof("device"):
        case "cpu":
            return CpuDevice(variant=value.cpu.variant)
        case "gpu":
            return GpuDevice(variant=value.gpu.variant, count=value.gpu.count or 1)
        case "tpu":
            return TpuDevice(variant=value.tpu.variant, topology=value.tpu.topology, count=value.tpu.count)
        case _:
            raise ValueError("device has no selected kind")


def resource_spec_to_proto(value: ResourceSpec) -> resource_pb2.ResourceSpecProto:
    result = resource_pb2.ResourceSpecProto(
        cpu_millicores=value.cpu_millicores,
        memory_bytes=value.memory,
        disk_bytes=value.disk,
    )
    if value.device is not None:
        result.device.CopyFrom(device_to_proto(value.device))
    return result


def resource_spec_from_proto(value: resource_pb2.ResourceSpecProto) -> ResourceSpec:
    return ResourceSpec(
        cpu=value.cpu_millicores / 1_000,
        memory=value.memory_bytes,
        disk=value.disk_bytes,
        device=device_from_proto(value.device) if value.HasField("device") else None,
    )


def environment_to_proto(value: Environment) -> resource_pb2.EnvironmentConfig:
    return resource_pb2.EnvironmentConfig(env_vars=dict(value.env_vars), setup_scripts=value.setup_scripts)


def environment_from_proto(value: resource_pb2.EnvironmentConfig) -> Environment:
    return Environment(env_vars=dict(value.env_vars), setup_scripts=tuple(value.setup_scripts))


def runtime_entrypoint_to_proto(value: RuntimeEntrypoint) -> resource_pb2.RuntimeEntrypoint:
    return resource_pb2.RuntimeEntrypoint(
        setup_commands=value.setup_commands,
        run_command=resource_pb2.CommandEntrypoint(argv=value.run_command.argv),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def runtime_entrypoint_from_proto(value: resource_pb2.RuntimeEntrypoint) -> RuntimeEntrypoint:
    return RuntimeEntrypoint(
        setup_commands=tuple(value.setup_commands),
        run_command=CommandEntrypoint(tuple(value.run_command.argv)),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def _attribute_value_to_proto(value: AttributeValue) -> resource_pb2.AttributeValue:
    result = resource_pb2.AttributeValue()
    if isinstance(value.value, str):
        result.string_value = value.value
    elif isinstance(value.value, int):
        result.int_value = value.value
    else:
        result.float_value = value.value
    return result


def _attribute_value_from_proto(value: resource_pb2.AttributeValue) -> AttributeValue:
    match value.WhichOneof("value"):
        case "string_value":
            return AttributeValue(value.string_value)
        case "int_value":
            return AttributeValue(value.int_value)
        case "float_value":
            return AttributeValue(value.float_value)
        case _:
            return AttributeValue("")


def constraint_to_proto(value: Constraint) -> resource_pb2.Constraint:
    result = resource_pb2.Constraint(key=value.key, op=int(value.op), mode=int(value.mode))
    if value.op is ConstraintOp.IN:
        result.values.extend(_attribute_value_to_proto(item) for item in value.values)
    elif value.values:
        result.value.CopyFrom(_attribute_value_to_proto(value.values[0]))
    return result


def constraint_from_proto(value: resource_pb2.Constraint) -> Constraint:
    op = ConstraintOp(value.op)
    if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
        values = ()
    elif op is ConstraintOp.IN:
        values = tuple(_attribute_value_from_proto(item) for item in value.values)
    else:
        values = (_attribute_value_from_proto(value.value),)
    return Constraint(key=value.key, op=op, values=values, mode=ConstraintMode(value.mode))


def job_spec_to_proto(value: JobSpec) -> resource_pb2.JobSpec:
    result = resource_pb2.JobSpec(
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


def job_spec_from_proto(value: resource_pb2.JobSpec) -> JobSpec:
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


def profile_configuration_to_proto(value: ProfileConfiguration | None) -> resource_pb2.ProfileType:
    if isinstance(value, CpuProfileConfiguration):
        cpu = resource_pb2.CpuProfile(format=_CPU_FORMAT_TO_PROTO[value.format], rate_hz=value.rate_hz)
        if value.native is not None:
            cpu.native = value.native
        return resource_pb2.ProfileType(cpu=cpu)
    if isinstance(value, MemoryProfileConfiguration):
        return resource_pb2.ProfileType(
            memory=resource_pb2.MemoryProfile(format=_MEMORY_FORMAT_TO_PROTO[value.format], leaks=value.leaks)
        )
    if isinstance(value, ThreadsProfileConfiguration):
        return resource_pb2.ProfileType(threads=resource_pb2.ThreadsProfile(locals=value.include_locals))
    return resource_pb2.ProfileType()


def profile_configuration_from_proto(value: resource_pb2.ProfileType) -> ProfileConfiguration | None:
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
