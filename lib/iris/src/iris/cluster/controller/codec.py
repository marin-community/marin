# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON/proto codec for controller DB columns.

All serialization between protobuf messages and the JSON columns stored in the
controller SQLite database goes through this module.  controller.py,
transitions.py and service.py import from here — this module has no dependency
on any of them, sitting at the bottom of the import graph.
"""

import json
from collections.abc import Iterable

from google.protobuf import json_format

from iris.rpc import controller_pb2, job_pb2

# Shared kwargs for MessageToDict so every call site is consistent.
_TO_DICT_OPTS = dict(preserving_proto_field_name=True, use_integers_for_enums=True)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def proto_to_json(msg) -> str:
    """Serialize any protobuf message to a JSON string (snake_case keys, integer enums)."""
    return json.dumps(json_format.MessageToDict(msg, **_TO_DICT_OPTS))


def proto_from_json(json_str: str, proto_cls):
    """Deserialize a JSON string into a new protobuf message of *proto_cls*."""
    return json_format.ParseDict(json.loads(json_str), proto_cls())


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


def constraints_from_json(constraints_json: str | None) -> list[job_pb2.Constraint]:
    """Deserialize a JSON array of constraints back to proto objects."""
    if not constraints_json:
        return []
    return [json_format.ParseDict(item, job_pb2.Constraint()) for item in json.loads(constraints_json)]


def reservation_to_json(request: controller_pb2.Controller.LaunchJobRequest) -> str | None:
    """Serialize the reservation field of a LaunchJobRequest to JSON.  Returns None if absent."""
    if not request.HasField("reservation"):
        return None
    return json.dumps(json_format.MessageToDict(request.reservation, **_TO_DICT_OPTS))


def entrypoint_to_json(ep: job_pb2.RuntimeEntrypoint) -> str:
    """Serialize a RuntimeEntrypoint, excluding inline workdir_files (stored separately)."""
    d = json_format.MessageToDict(ep, **_TO_DICT_OPTS)
    d.pop("workdir_files", None)
    return json.dumps(d)


def reservation_entries_from_json(reservation_json: str | None) -> list[job_pb2.ReservationEntry]:
    """Deserialize reservation JSON back to a list of ReservationEntry protos."""
    if not reservation_json:
        return []
    data = json.loads(reservation_json)
    return [json_format.ParseDict(e, job_pb2.ReservationEntry()) for e in data.get("entries", [])]
