# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Codecs shared by task-launch and the legacy job-oriented RPC wire."""

from iris.cluster.constraints import AttributeValue, Constraint, ConstraintMode, ConstraintOp
from iris.resources.execution import (
    CommandEntrypoint,
    CpuDevice,
    Device,
    Environment,
    GpuDevice,
    ResourceSpec,
    RuntimeEntrypoint,
    TpuDevice,
)
from iris.rpc import job_pb2


def device_to_proto(value: Device) -> job_pb2.DeviceConfig:
    if isinstance(value, CpuDevice):
        return job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice(variant=value.variant))
    if isinstance(value, GpuDevice):
        return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant=value.variant, count=value.count))
    tpu = job_pb2.TpuDevice(variant=value.variant)
    if value.topology:
        tpu.topology = value.topology
    if value.count:
        tpu.count = value.count
    return job_pb2.DeviceConfig(tpu=tpu)


def device_from_proto(value: job_pb2.DeviceConfig) -> Device | None:
    """Decode a device, treating an empty message as no device.

    A persisted device-less resource can reconstitute as a present ``DeviceConfig``
    with no oneof selected. That is the legacy wire encoding of no device.
    """
    match value.WhichOneof("device"):
        case "cpu":
            return CpuDevice(value.cpu.variant)
        case "gpu":
            return GpuDevice(value.gpu.variant, value.gpu.count or 1)
        case "tpu":
            return TpuDevice(value.tpu.variant, value.tpu.topology, value.tpu.count)
        case _:
            return None


def gpu_count_from_proto(value: job_pb2.DeviceConfig) -> int:
    """Return the GPU count from a legacy worker device message."""
    if not value.HasField("gpu"):
        return 0
    return value.gpu.count or 1


def tpu_count_from_proto(value: job_pb2.DeviceConfig) -> int:
    """Return the TPU count from a legacy worker device message."""
    return value.tpu.count if value.HasField("tpu") else 0


def resource_spec_to_proto(value: ResourceSpec) -> job_pb2.ResourceSpecProto:
    result = job_pb2.ResourceSpecProto()
    if value.cpu_millicores:
        result.cpu_millicores = value.cpu_millicores
    if value.memory:
        result.memory_bytes = value.memory
    if value.disk:
        result.disk_bytes = value.disk
    if value.device is not None:
        result.device.CopyFrom(device_to_proto(value.device))
    return result


def resource_spec_from_proto(value: job_pb2.ResourceSpecProto) -> ResourceSpec:
    return ResourceSpec(
        cpu=value.cpu_millicores / 1_000,
        memory=value.memory_bytes,
        disk=value.disk_bytes,
        device=device_from_proto(value.device),
    )


def environment_to_proto(value: Environment) -> job_pb2.EnvironmentConfig:
    return job_pb2.EnvironmentConfig(env_vars=dict(value.env_vars), setup_scripts=value.setup_scripts)


def environment_from_proto(value: job_pb2.EnvironmentConfig) -> Environment:
    return Environment(dict(value.env_vars), tuple(value.setup_scripts))


def runtime_entrypoint_to_proto(value: RuntimeEntrypoint) -> job_pb2.RuntimeEntrypoint:
    return job_pb2.RuntimeEntrypoint(
        setup_commands=value.setup_commands,
        run_command=job_pb2.CommandEntrypoint(argv=value.run_command.argv),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def runtime_entrypoint_from_proto(value: job_pb2.RuntimeEntrypoint) -> RuntimeEntrypoint:
    return RuntimeEntrypoint(
        setup_commands=tuple(value.setup_commands),
        run_command=CommandEntrypoint(tuple(value.run_command.argv)),
        workdir_files=dict(value.workdir_files),
        workdir_file_refs=dict(value.workdir_file_refs),
    )


def attribute_value_to_proto(value: AttributeValue) -> job_pb2.AttributeValue:
    result = job_pb2.AttributeValue()
    if isinstance(value.value, str):
        result.string_value = value.value
    elif isinstance(value.value, int):
        result.int_value = value.value
    else:
        result.float_value = value.value
    return result


def attribute_value_from_proto(value: job_pb2.AttributeValue) -> AttributeValue:
    match value.WhichOneof("value"):
        case "string_value":
            return AttributeValue(value.string_value)
        case "int_value":
            return AttributeValue(value.int_value)
        case "float_value":
            return AttributeValue(value.float_value)
        case _:
            return AttributeValue("")


def constraint_to_proto(value: Constraint) -> job_pb2.Constraint:
    result = job_pb2.Constraint(key=value.key, op=int(value.op), mode=int(value.mode))
    if value.op is ConstraintOp.IN:
        result.values.extend(attribute_value_to_proto(item) for item in value.values)
    elif value.values:
        result.value.CopyFrom(attribute_value_to_proto(value.values[0]))
    return result


def constraint_from_proto(value: job_pb2.Constraint) -> Constraint:
    op = ConstraintOp(value.op)
    if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
        values = ()
    elif op is ConstraintOp.IN:
        values = tuple(attribute_value_from_proto(item) for item in value.values)
    else:
        values = (attribute_value_from_proto(value.value),)
    return Constraint(value.key, op, values, ConstraintMode(value.mode))
