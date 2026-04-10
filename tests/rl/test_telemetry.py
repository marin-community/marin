# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import fsspec
import pytest

from marin.rl.telemetry import (
    EventShardWriter,
    StepProvenance,
    TelemetryEvent,
    TrackerStream,
    event_shard_path,
)


def test_telemetry_event_round_trips_through_json():
    event = TelemetryEvent(
        run_id="rl-test",
        stream=TrackerStream.ROLLOUT,
        event_type="rollout_batch_written",
        provenance=StepProvenance(
            rollout_step=12,
            weight_step=10,
            train_step=8,
            worker_index=2,
            instance_id="attempt-abc",
        ),
        payload={"avg_reward": 0.5, "lesson_id": "math"},
    )

    restored = TelemetryEvent.from_json(event.to_json())

    assert restored == event
    assert restored.schema_version == "telemetry_event.v1"
    assert restored.event_id


def test_event_shard_path_uses_per_writer_names():
    metadata_path = "gs://bucket/runs/metadata"

    assert (
        event_shard_path(
            metadata_path,
            "rl-test",
            TrackerStream.TRAINER,
            instance_id="attempt-abc",
        )
        == "gs://bucket/runs/metadata/rl-test/events/train-attempt-abc.jsonl"
    )
    assert (
        event_shard_path(
            metadata_path,
            "rl-test",
            TrackerStream.EVAL,
            instance_id="attempt-abc",
        )
        == "gs://bucket/runs/metadata/rl-test/events/eval-attempt-abc.jsonl"
    )
    assert (
        event_shard_path(
            metadata_path,
            "rl-test",
            TrackerStream.ROLLOUT,
            instance_id="attempt-abc",
            worker_index=3,
        )
        == "gs://bucket/runs/metadata/rl-test/events/rollout-3-attempt-abc.jsonl"
    )


def test_rollout_event_shard_path_requires_worker_index():
    with pytest.raises(ValueError, match="worker_index is required"):
        event_shard_path("/tmp/metadata", "rl-test", TrackerStream.ROLLOUT, instance_id="attempt-abc")


def test_event_shard_writer_appends_jsonl_records(tmp_path: Path):
    writer = EventShardWriter(
        metadata_path=str(tmp_path / "metadata"),
        run_id="rl-test",
        stream=TrackerStream.TRAINER,
        instance_id="attempt-abc",
    )

    first_event = TelemetryEvent(
        run_id="rl-test",
        stream=TrackerStream.TRAINER,
        event_type="worker_started",
        provenance=StepProvenance(instance_id="attempt-abc"),
        payload={"message": "trainer online"},
    )
    second_event = TelemetryEvent(
        run_id="rl-test",
        stream=TrackerStream.TRAINER,
        event_type="checkpoint_written",
        provenance=StepProvenance(instance_id="attempt-abc", train_step=5),
        payload={"checkpoint_step": 5},
    )

    writer.append(first_event)
    writer.append(second_event)

    fs = fsspec.filesystem("file")
    assert fs.exists(writer.path)
    with fs.open(writer.path) as handle:
        lines = [line for line in handle.read().splitlines() if line]

    restored = [TelemetryEvent.from_json(line) for line in lines]
    assert restored == [first_event, second_event]
    assert writer.artifact_ref().schema_version == "rl_artifact_ref.v1"
