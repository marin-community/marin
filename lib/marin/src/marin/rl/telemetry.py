# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Telemetry primitives for RL observability.

This module defines the durable event and reference schemas used by the RL
observability roadmap. The first implementation wave keeps the API small:
event records, step provenance, tracker/artifact references, and per-writer
event shards.
"""

import dataclasses
import datetime
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from levanter.utils.fsspec_utils import join_path
from rigging.filesystem import url_to_fs

from marin.utilities.json_encoder import CustomJsonEncoder

TELEMETRY_EVENT_SCHEMA_VERSION = "telemetry_event.v1"
TRACKER_RUN_REF_SCHEMA_VERSION = "rl_tracker_run_ref.v1"
ARTIFACT_REF_SCHEMA_VERSION = "rl_artifact_ref.v1"


class TrackerStream(StrEnum):
    """Named RL telemetry streams."""

    TRAINER = "trainer"
    ROLLOUT = "rollout"
    EVAL = "eval"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _fresh_event_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class StepProvenance:
    """Provenance for one telemetry record."""

    train_step: int | None = None
    rollout_step: int | None = None
    weight_step: int | None = None
    eval_sequence: int | None = None
    worker_index: int | None = None
    instance_id: str | None = None

    def as_dict(self, *, prefix: str | None = None) -> dict[str, int | str]:
        items: dict[str, int | str] = {}
        for field_name, value in dataclasses.asdict(self).items():
            if value is None:
                continue
            key = field_name if prefix is None else f"{prefix}.{field_name}"
            items[key] = value
        return items

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepProvenance":
        return cls(
            train_step=data.get("train_step"),
            rollout_step=data.get("rollout_step"),
            weight_step=data.get("weight_step"),
            eval_sequence=data.get("eval_sequence"),
            worker_index=data.get("worker_index"),
            instance_id=data.get("instance_id"),
        )


@dataclass(frozen=True)
class TelemetryEvent:
    """One durable RL telemetry event."""

    stream: TrackerStream
    event_type: str
    provenance: StepProvenance
    payload: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    event_id: str = field(default_factory=_fresh_event_id)
    created_at: str = field(default_factory=_utc_now_iso)
    schema_version: str = TELEMETRY_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "created_at": self.created_at,
            "run_id": self.run_id,
            "stream": self.stream.value,
            "event_type": self.event_type,
            "provenance": dataclasses.asdict(self.provenance),
            "payload": dict(self.payload),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, cls=CustomJsonEncoder)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TelemetryEvent":
        return cls(
            schema_version=str(data["schema_version"]),
            event_id=str(data["event_id"]),
            created_at=str(data["created_at"]),
            run_id=data.get("run_id"),
            stream=TrackerStream(data["stream"]),
            event_type=str(data["event_type"]),
            provenance=StepProvenance.from_dict(data.get("provenance", {})),
            payload=dict(data.get("payload", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "TelemetryEvent":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class TrackerRunRef:
    """Durable reference to an external tracker run."""

    stream: TrackerStream
    tracker_run_id: str
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    run_url: str | None = None
    worker_index: int | None = None
    schema_version: str = TRACKER_RUN_REF_SCHEMA_VERSION


@dataclass(frozen=True)
class ArtifactRef:
    """Durable reference to a file or artifact emitted by the RL run."""

    name: str
    path: str
    artifact_type: str
    stream: TrackerStream | None = None
    worker_index: int | None = None
    schema_version: str = ARTIFACT_REF_SCHEMA_VERSION


def event_shard_directory(metadata_path: str, run_id: str) -> str:
    return join_path(join_path(metadata_path, run_id), "events")


def event_shard_filename(
    stream: TrackerStream,
    *,
    instance_id: str,
    worker_index: int | None = None,
) -> str:
    if stream == TrackerStream.TRAINER:
        return f"train-{instance_id}.jsonl"
    if stream == TrackerStream.EVAL:
        return f"eval-{instance_id}.jsonl"
    if stream == TrackerStream.ROLLOUT:
        if worker_index is None:
            raise ValueError("worker_index is required for rollout event shards")
        return f"rollout-{worker_index}-{instance_id}.jsonl"
    raise ValueError(f"Unsupported tracker stream: {stream}")


def event_shard_path(
    metadata_path: str,
    run_id: str,
    stream: TrackerStream,
    *,
    instance_id: str,
    worker_index: int | None = None,
) -> str:
    return join_path(
        event_shard_directory(metadata_path, run_id),
        event_shard_filename(stream, instance_id=instance_id, worker_index=worker_index),
    )


class EventShardWriter:
    """Append-only writer for one RL telemetry shard.

    Each process owns exactly one shard. This avoids multi-process append
    contention while still keeping the event record line-oriented and easy to
    mirror into external trackers later.
    """

    def __init__(
        self,
        *,
        metadata_path: str,
        run_id: str,
        stream: TrackerStream,
        instance_id: str,
        worker_index: int | None = None,
    ):
        self.metadata_path = metadata_path
        self.run_id = run_id
        self.stream = stream
        self.instance_id = instance_id
        self.worker_index = worker_index
        self.directory = event_shard_directory(metadata_path, run_id)
        self.path = event_shard_path(
            metadata_path,
            run_id,
            stream,
            instance_id=instance_id,
            worker_index=worker_index,
        )

    def artifact_ref(self) -> ArtifactRef:
        return ArtifactRef(
            name=os.path.basename(self.path),
            path=self.path,
            artifact_type="event_shard",
            stream=self.stream,
            worker_index=self.worker_index,
        )

    def append(self, event: TelemetryEvent) -> None:
        if event.stream != self.stream:
            raise ValueError(f"event stream {event.stream} does not match shard stream {self.stream}")

        fs = url_to_fs(self.path)[0]
        fs.makedirs(self.directory, exist_ok=True)
        with fs.open(self.path, "a") as handle:
            handle.write(event.to_json())
            handle.write("\n")
