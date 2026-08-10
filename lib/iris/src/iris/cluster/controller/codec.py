# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON/proto codec for controller DB columns.

All serialization between protobuf messages and the JSON columns stored in the
controller SQLite database goes through this module.
"""

import functools
import json
from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol

from google.protobuf import json_format

from iris.cluster.constraints import Constraint, get_device_variant
from iris.cluster.types import get_gpu_count, get_tpu_count
from iris.rpc import controller_pb2, job_pb2


class WorkerAttributeRow(Protocol):
    """Structural shape of a ``worker_attributes`` SA row read by the value decoders."""

    key: str
    value_type: str
    str_value: str | None
    int_value: int | None
    float_value: float | None


# Shared kwargs for MessageToDict so every call site is consistent.
_TO_DICT_OPTS = dict(preserving_proto_field_name=True, use_integers_for_enums=True)

# Maxsize for the JSON->proto decode caches. Sized to comfortably hold the
# distinct JSON column values in flight across a scheduling cycle.
_CODEC_CACHE_SIZE = 8192


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def proto_to_json(msg) -> str:
    """Serialize any protobuf message to a JSON string (snake_case keys, integer enums)."""
    return json.dumps(json_format.MessageToDict(msg, **_TO_DICT_OPTS))


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def proto_from_json(json_str: str, proto_cls):
    """Deserialize a JSON string into a new protobuf message of *proto_cls*.

    Ignores unknown fields so jobs persisted before a field was removed still
    replay across a controller restart.
    """
    return json_format.ParseDict(json.loads(json_str), proto_cls(), ignore_unknown_fields=True)


# ---------------------------------------------------------------------------
# Composite helpers for types that don't map 1:1 to a single JSON column
# ---------------------------------------------------------------------------


def resource_spec_from_scalars(cpu: int, mem: int, disk: int, device_json: str | None) -> job_pb2.ResourceSpecProto:
    """Reconstruct a ResourceSpecProto from the scalar columns stored on jobs/job_config."""
    res = job_pb2.ResourceSpecProto(cpu_millicores=cpu, memory_bytes=mem, disk_bytes=disk)
    if device_json:
        res.device.CopyFrom(proto_from_json(device_json, job_pb2.DeviceConfig))
    return res


def constraints_to_json(constraints: Iterable[job_pb2.Constraint]) -> str | None:
    """Serialize a list of Constraint protos to a JSON array.  Returns None if empty."""
    items = [json_format.MessageToDict(c, **_TO_DICT_OPTS) for c in constraints]
    return json.dumps(items) if items else None


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def constraints_from_json(constraints_json: str | None) -> list[Constraint]:
    """Deserialize a JSON array of constraints to native Constraint objects.

    Goes through proto for JSON parsing, then normalizes via Constraint.from_proto
    (which strips/lowercases string values for case-insensitive comparison).
    Native Constraint is the canonical hot-path type — protos only at serialization.
    """
    if not constraints_json:
        return []
    return [
        Constraint.from_proto(json_format.ParseDict(item, job_pb2.Constraint())) for item in json.loads(constraints_json)
    ]


def entrypoint_to_json(ep: job_pb2.RuntimeEntrypoint) -> str:
    """Serialize a RuntimeEntrypoint, excluding inline workdir_files.

    The files live in ``job_workdir_files``, keyed by job id, so the JSON column stays
    small. An entrypoint rebuilt from this JSON therefore has no inline files until
    they are re-attached from that table.
    """
    d = json_format.MessageToDict(ep, **_TO_DICT_OPTS)
    d.pop("workdir_files", None)
    return json.dumps(d)


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
    device = proto_from_json(device_json, job_pb2.DeviceConfig)
    return DeviceCounts(gpu=get_gpu_count(device), tpu=get_tpu_count(device))


@functools.lru_cache(maxsize=_CODEC_CACHE_SIZE)
def device_variant_from_json(device_json: str | None) -> str | None:
    """Cached parse of ``job_config.res_device_json`` into a device variant string.

    Returns the GPU or TPU variant string (e.g. ``"v5p-64"``) or ``None`` for
    CPU-only jobs.  The LRU amortizes repeated parses of the same JSON string
    across many task rows within a scheduling cycle.
    """
    if not device_json:
        return None
    device = proto_from_json(device_json, job_pb2.DeviceConfig)
    return get_device_variant(device)


# ---------------------------------------------------------------------------
# Row -> proto reconstructors (build protos from already-fetched DB columns)
# ---------------------------------------------------------------------------


def resource_spec_from_job_row(job: Any) -> job_pb2.ResourceSpecProto:
    """Reconstruct a ResourceSpecProto from native job columns."""
    return resource_spec_from_scalars(
        job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json
    )


def reconstruct_launch_job_request(
    job, *, workdir_files: dict[str, bytes]
) -> controller_pb2.Controller.LaunchJobRequest:
    """Reconstruct a LaunchJobRequest proto from native job columns.

    ``job.entrypoint_json`` carries no inline workdir files, so the caller names them:
    ``reads.get_workdir_files(tx, job_id)`` for a request that will be *run* — a
    federated handoff delivers this request to the peer, and without the files the peer
    has no ``_callable_runner.py`` to exec — or ``{}`` for a request that only describes
    the job's shape.

    ``bundle_blob``, ``client_revision_date`` and ``federation`` are not stored (the
    bundle survives as ``bundle_id``; the other two belong to the submitting hop), so a
    reconstruction never carries them.
    """
    req = controller_pb2.Controller.LaunchJobRequest(
        name=job.name,
        bundle_id=job.bundle_id,
        max_task_failures=job.max_task_failures,
        max_retries_failure=job.max_retries_failure,
        max_retries_preemption=job.max_retries_preemption,
        replicas=job.num_tasks,
        preemption_policy=job.preemption_policy,
        existing_job_policy=job.existing_job_policy,
        priority_band=job.priority_band,
        task_image=job.task_image,
        container_profile=job.container_profile,
        fail_if_exists=job.fail_if_exists,
    )
    req.entrypoint.CopyFrom(proto_from_json(job.entrypoint_json, job_pb2.RuntimeEntrypoint))
    for filename, data in workdir_files.items():
        req.entrypoint.workdir_files[filename] = data
    req.environment.CopyFrom(proto_from_json(job.environment_json, job_pb2.EnvironmentConfig))
    req.resources.CopyFrom(
        resource_spec_from_scalars(job.res_cpu_millicores, job.res_memory_bytes, job.res_disk_bytes, job.res_device_json)
    )

    for c in constraints_from_json(job.constraints_json):
        req.constraints.append(c.to_proto())
    for port in job.ports_json:
        req.ports.append(port)
    for arg in job.submit_argv_json:
        req.submit_argv.append(arg)

    if job.has_coscheduling:
        req.coscheduling.CopyFrom(job_pb2.CoschedulingConfig(group_by=job.coscheduling_group_by))

    if job.scheduling_timeout_ms is not None and job.scheduling_timeout_ms > 0:
        req.scheduling_timeout.milliseconds = job.scheduling_timeout_ms

    if job.timeout_ms is not None and job.timeout_ms > 0:
        req.timeout.milliseconds = job.timeout_ms

    return req


def worker_metadata_to_proto(worker, attributes: dict) -> job_pb2.WorkerMetadata:
    """Reconstruct a WorkerMetadata proto from scalar columns and decoded attributes dict."""
    md = job_pb2.WorkerMetadata(
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
    )
    if worker.md_device_json and worker.md_device_json != "{}":
        md.device.CopyFrom(proto_from_json(worker.md_device_json, job_pb2.DeviceConfig))
    if worker.md_provenance_json and worker.md_provenance_json != "{}":
        md.provenance.CopyFrom(proto_from_json(worker.md_provenance_json, job_pb2.Provenance))
    for key, value in attributes.items():
        md.attributes[key].CopyFrom(python_value_to_attribute_value(value))
    return md


def python_value_to_attribute_value(value: str | int | float) -> job_pb2.AttributeValue:
    """Wrap a Python attribute value in its ``AttributeValue`` proto oneof.

    Unlike :meth:`constraints.AttributeValue.to_proto`, this does not normalize
    string values — it stores them verbatim, matching the worker_attributes
    round-trip.
    """
    av = job_pb2.AttributeValue()
    if isinstance(value, str):
        av.string_value = value
    elif isinstance(value, int):
        av.int_value = value
    elif isinstance(value, float):
        av.float_value = value
    return av


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
