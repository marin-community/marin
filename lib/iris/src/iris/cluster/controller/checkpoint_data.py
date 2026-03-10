# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data types for controller checkpoint serialization.

Plain dataclasses with JSON serialization for persisting autoscaler state
(scaling groups, tracked workers) into the controller SQLite DB.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SliceSnapshotData:
    """Serializable snapshot of a single slice."""

    slice_id: str
    scale_group: str
    lifecycle: str
    vm_addresses: list[str] = field(default_factory=list)
    created_at_ms: int = 0
    last_active_ms: int = 0
    error_message: str = ""


@dataclass
class ScalingGroupSnapshotData:
    """Serializable snapshot of a scaling group."""

    name: str
    slices: list[SliceSnapshotData] = field(default_factory=list)
    consecutive_failures: int = 0
    backoff_until_ms: int = 0
    last_scale_up_ms: int = 0
    last_scale_down_ms: int = 0
    quota_exceeded_until_ms: int = 0
    quota_reason: str = ""


@dataclass
class TrackedWorkerSnapshotData:
    """Serializable snapshot of a tracked worker."""

    worker_id: str
    slice_id: str
    scale_group: str
    internal_address: str


def serialize_scaling_group(data: ScalingGroupSnapshotData) -> bytes:
    """Serialize a ScalingGroupSnapshotData to bytes for DB storage."""
    return json.dumps(asdict(data)).encode()


def deserialize_scaling_group(raw: bytes) -> ScalingGroupSnapshotData:
    """Deserialize a ScalingGroupSnapshotData from bytes."""
    d = json.loads(raw)
    d["slices"] = [SliceSnapshotData(**s) for s in d.get("slices", [])]
    return ScalingGroupSnapshotData(**d)
