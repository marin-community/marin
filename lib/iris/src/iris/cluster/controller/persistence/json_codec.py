# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON codec for controller DB columns.

All serialization between native records and the JSON columns stored in the
controller SQLite database goes through this module.
"""

import functools
import json
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple, Protocol

from rigging.provenance import Provenance
from rigging.timing import Duration

from iris.cluster.constraints import AttributeValue, Constraint, ConstraintMode, ConstraintOp
from iris.resources.execution import (
    CommandEntrypoint,
    CpuDevice,
    Environment,
    GpuDevice,
    ResourceSpec,
    RuntimeEntrypoint,
    TpuDevice,
)
from iris.resources.job import (
    ContainerProfile,
    CoschedulingConfig,
    ExistingJobPolicy,
    JobPreemptionPolicy,
    JobSpec,
    PriorityBand,
)
from iris.resources.worker import WorkerMetadata


class WorkerAttributeRow(Protocol):
    """Structural shape of a ``worker_attributes`` SA row read by the value decoders."""

    key: str
    value_type: str
    str_value: str | None
    int_value: int | None
    float_value: float | None


class WorkerMetadataRow(Protocol):
    """Structural shape of the worker columns used by the metadata decoder."""

    md_hostname: str
    md_ip_address: str
    md_cpu_count: int
    md_memory_bytes: int
    md_disk_bytes: int
    md_tpu_name: str
    md_tpu_worker_hostnames: str
    md_tpu_worker_id: str
    md_tpu_chips_per_host_bounds: str
    md_gpu_count: int
    md_gpu_name: str
    md_gpu_memory_mb: int
    md_gce_instance_name: str
    md_gce_zone: str
    md_device_json: str | None
    md_provenance_json: str | None


# Maxsize for the JSON->proto decode caches. Sized to comfortably hold the
# distinct JSON column values in flight across a scheduling cycle.
_CODEC_CACHE_SIZE = 8192


# ---------------------------------------------------------------------------
# Persisted scalar helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Composite helpers for types that don't map 1:1 to a single JSON column
# ---------------------------------------------------------------------------


def resource_spec_from_scalars(cpu: int, mem: int, disk: int, device_json: str | None) -> ResourceSpec:
    """Reconstruct a native resource specification from persisted scalar columns."""
    return ResourceSpec(
        cpu=cpu / 1_000,
        memory=mem,
        disk=disk,
        device=_device_from_json(device_json) if device_json else None,
    )


def _attribute_value_to_json(value: AttributeValue) -> dict[str, object]:
    if isinstance(value.value, str):
        return {"string_value": value.value}
    if isinstance(value.value, int):
        return {"int_value": str(value.value)}
    return {"float_value": value.value}


def constraints_to_json(constraints: Iterable[Constraint]) -> str | None:
    """Serialize native constraints in the established protobuf-JSON shape."""
    items = []
    for constraint in constraints:
        item: dict[str, object] = {"key": constraint.key}
        if constraint.op is not ConstraintOp.EQ:
            item["op"] = int(constraint.op)
        if constraint.op is ConstraintOp.IN:
            item["values"] = [_attribute_value_to_json(value) for value in constraint.values]
        elif constraint.values:
            item["value"] = _attribute_value_to_json(constraint.values[0])
        if constraint.mode is not ConstraintMode.REQUIRED:
            item["mode"] = int(constraint.mode)
        items.append(item)
    return json.dumps(items) if items else None


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def constraints_from_json(constraints_json: str | None) -> list[Constraint]:
    """Deserialize a JSON array of constraints to native Constraint objects.

    Integer attribute values retain protobuf JSON's decimal-string representation.
    """
    if not constraints_json:
        return []
    constraints = []
    for item in json.loads(constraints_json):
        op = ConstraintOp(int(item.get("op", 0)))
        encoded_values = item.get("values", ()) if op is ConstraintOp.IN else (item.get("value", {}),)
        if op in (ConstraintOp.EXISTS, ConstraintOp.NOT_EXISTS):
            encoded_values = ()
        values = tuple(_attribute_value_from_json(value) for value in encoded_values)
        constraints.append(
            Constraint(
                key=str(item.get("key", "")),
                op=op,
                values=values,
                mode=ConstraintMode(int(item.get("mode", 0))),
            )
        )
    return constraints


def _attribute_value_from_json(value: dict[str, object]) -> AttributeValue:
    if "string_value" in value:
        return AttributeValue(str(value["string_value"]))
    if "int_value" in value:
        return AttributeValue(int(value["int_value"]))
    if "float_value" in value:
        return AttributeValue(float(value["float_value"]))
    return AttributeValue("")


def entrypoint_to_json(entrypoint: RuntimeEntrypoint) -> str:
    """Serialize a RuntimeEntrypoint, excluding inline workdir_files.

    The files live in ``job_workdir_files``, keyed by job id, so the JSON column stays
    small. An entrypoint rebuilt from this JSON therefore has no inline files until
    they are re-attached from that table.
    """
    value: dict[str, object] = {}
    if entrypoint.setup_commands:
        value["setup_commands"] = list(entrypoint.setup_commands)
    value["run_command"] = {}
    if entrypoint.run_command.argv:
        value["run_command"] = {"argv": list(entrypoint.run_command.argv)}
    if entrypoint.workdir_file_refs:
        value["workdir_file_refs"] = dict(sorted(entrypoint.workdir_file_refs.items()))
    return json.dumps(value)


def environment_to_json(environment: Environment) -> str:
    value: dict[str, object] = {}
    if environment.env_vars:
        value["env_vars"] = dict(sorted(environment.env_vars.items()))
    if environment.setup_scripts:
        value["setup_scripts"] = list(environment.setup_scripts)
    return json.dumps(value)


def runtime_entrypoint_from_json(value: str, workdir_files: Mapping[str, bytes]) -> RuntimeEntrypoint:
    """Decode a persisted runtime entrypoint and attach its separately stored files."""
    item = json.loads(value)
    return RuntimeEntrypoint(
        setup_commands=tuple(item.get("setup_commands", ())),
        run_command=CommandEntrypoint(tuple(item.get("run_command", {}).get("argv", ()))),
        workdir_files=workdir_files,
        workdir_file_refs=dict(item.get("workdir_file_refs", {})),
    )


def environment_from_json(value: str) -> Environment:
    """Decode a persisted native execution environment."""
    item = json.loads(value)
    return Environment(
        env_vars=dict(item.get("env_vars", {})),
        setup_scripts=tuple(item.get("setup_scripts", ())),
    )


def device_to_json(device: CpuDevice | GpuDevice | TpuDevice) -> str:
    if isinstance(device, CpuDevice):
        return json.dumps({"cpu": {"variant": device.variant}})
    if isinstance(device, GpuDevice):
        return json.dumps({"gpu": {"variant": device.variant, "count": device.count}})
    return json.dumps({"tpu": {"variant": device.variant, "topology": device.topology, "count": device.count}})


def _device_from_json(value: str) -> CpuDevice | GpuDevice | TpuDevice:
    item = json.loads(value)
    if "cpu" in item:
        return CpuDevice(variant=str(item["cpu"].get("variant", "")))
    if "gpu" in item:
        return GpuDevice(
            variant=str(item["gpu"].get("variant", "")),
            count=int(item["gpu"].get("count", 0)) or 1,
        )
    if "tpu" in item:
        return TpuDevice(
            variant=str(item["tpu"].get("variant", "")),
            topology=str(item["tpu"].get("topology", "")),
            count=int(item["tpu"].get("count", 0)),
        )
    raise ValueError("device JSON has no selected kind")


class DeviceCounts(NamedTuple):
    """GPU and TPU counts parsed from a `job_config.res_device_json` string."""

    gpu: int
    tpu: int


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def device_counts_from_json(device_json: str | None) -> DeviceCounts:
    """Cached parse of `job_config.res_device_json` into device counts.

    Per-tick scheduler usage aggregation calls this once per attempt row;
    the LRU amortizes the JSON parse across many rows that share the same
    device JSON string.
    """
    if not device_json:
        return DeviceCounts(gpu=0, tpu=0)
    device = _device_from_json(device_json)
    return DeviceCounts(
        gpu=device.count if isinstance(device, GpuDevice) else 0,
        tpu=device.count if isinstance(device, TpuDevice) else 0,
    )


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def device_variant_from_json(device_json: str | None) -> str | None:
    """Cached parse of ``job_config.res_device_json`` into a device variant string.

    Returns the GPU or TPU variant string (e.g. ``"v5p-64"``) or ``None`` for
    CPU-only jobs.  The LRU amortizes repeated parses of the same JSON string
    across many task rows within a scheduling cycle.
    """
    if not device_json:
        return None
    device = _device_from_json(device_json)
    if isinstance(device, (GpuDevice, TpuDevice)):
        return device.variant or None
    return None


# ---------------------------------------------------------------------------
# Row reconstructors
# ---------------------------------------------------------------------------


def resource_spec_from_job_row(job: Any) -> ResourceSpec:
    """Reconstruct a typed resource specification from native Job columns."""
    return resource_spec_from_scalars(
        job.res_cpu_millicores,
        job.res_memory_bytes,
        job.res_disk_bytes,
        job.res_device_json,
    )


def reconstruct_job_spec(job, *, workdir_files: dict[str, bytes]) -> JobSpec:
    """Reconstruct the typed Job specification persisted for ``job``."""
    return JobSpec(
        version=1,
        name=job.job_id.to_wire(),
        entrypoint=runtime_entrypoint_from_json(job.entrypoint_json, workdir_files),
        resources=resource_spec_from_job_row(job),
        environment=environment_from_json(job.environment_json),
        bundle_id=job.bundle_id,
        scheduling_timeout=(
            Duration.from_ms(job.scheduling_timeout_ms) if job.scheduling_timeout_ms is not None else None
        ),
        ports=tuple(job.ports_json),
        max_task_failures=job.max_task_failures,
        max_retries_failure=job.max_retries_failure,
        max_retries_preemption=job.max_retries_preemption,
        constraints=tuple(constraints_from_json(job.constraints_json)),
        coscheduling=(CoschedulingConfig(group_by=job.coscheduling_group_by) if job.has_coscheduling else None),
        replicas=job.num_tasks,
        timeout=Duration.from_ms(job.timeout_ms) if job.timeout_ms is not None else None,
        fail_if_exists=job.fail_if_exists,
        preemption_policy=JobPreemptionPolicy(job.preemption_policy),
        existing_job_policy=ExistingJobPolicy(job.existing_job_policy),
        priority_band=PriorityBand(job.priority_band),
        task_image=job.task_image,
        submit_argv=tuple(job.submit_argv_json),
        client_revision_date="",
        container_profile=ContainerProfile(job.container_profile),
    )


def worker_metadata_from_row(worker: WorkerMetadataRow, attributes: Mapping[str, str | int | float]) -> WorkerMetadata:
    """Reconstruct native worker metadata from scalar columns and attributes."""
    return WorkerMetadata(
        hostname=worker.md_hostname,
        ip_address=worker.md_ip_address,
        cpu_count=worker.md_cpu_count,
        memory_bytes=worker.md_memory_bytes,
        disk_bytes=worker.md_disk_bytes,
        tpu_name=worker.md_tpu_name,
        tpu_worker_hostnames=worker.md_tpu_worker_hostnames,
        tpu_worker_id=worker.md_tpu_worker_id,
        tpu_chips_per_host_bounds=worker.md_tpu_chips_per_host_bounds,
        gpu_count=worker.md_gpu_count,
        gpu_name=worker.md_gpu_name,
        gpu_memory_mb=worker.md_gpu_memory_mb,
        gce_instance_name=worker.md_gce_instance_name,
        gce_zone=worker.md_gce_zone,
        device=(
            _device_from_json(worker.md_device_json) if worker.md_device_json and worker.md_device_json != "{}" else None
        ),
        provenance=(
            Provenance.from_json(worker.md_provenance_json)
            if worker.md_provenance_json and worker.md_provenance_json != "{}"
            else None
        ),
        attributes={key: AttributeValue(value) for key, value in attributes.items()},
    )


def attribute_value_from_row(row: WorkerAttributeRow) -> str | int | float:
    """Decode the typed value from a ``worker_attributes`` row.

    A NULL ``str_value`` decodes to the empty string (the write path never stores one).
    """
    vtype = str(row.value_type)
    if vtype == "str":
        return str(row.str_value or "")
    if vtype == "int":
        return int(row.int_value)
    if vtype == "float":
        return float(row.float_value)
    raise ValueError(f"Unknown attribute value_type: {vtype!r}")


def decode_attribute_value(row: WorkerAttributeRow) -> tuple[str, str | int | float]:
    """Decode a worker_attributes row into a (key, value) pair."""
    return str(row.key), attribute_value_from_row(row)
