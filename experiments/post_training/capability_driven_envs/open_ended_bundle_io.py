# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON records and Parquet rows for hydrated open-ended bundles."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO, Literal, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator
from taskforge.content_hash import canonical_json

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_packet_schema import OpenEndedReleaseStatus

BUNDLE_RECORD_FORMAT = "open-ended-task-bundle/1"
BUNDLE_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("format", pa.string(), nullable=False),
        pa.field("bundle_id", pa.string(), nullable=False),
        pa.field("bundle_digest", pa.string(), nullable=False),
        pa.field("split_group", pa.string(), nullable=False),
        pa.field("release_status", pa.string(), nullable=False),
        pa.field("bundle_json", pa.large_string(), nullable=False),
    ]
)


class OpenEndedBundleRecord(BaseModel):
    """One self-verifying bundle row suitable for JSON or Parquet storage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: Literal["open-ended-task-bundle/1"] = BUNDLE_RECORD_FORMAT
    bundle_id: str = Field(min_length=1)
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_group: str = Field(min_length=1)
    release_status: OpenEndedReleaseStatus
    bundle_json: str = Field(min_length=1)

    @model_validator(mode="after")
    def matches_bundle(self) -> Self:
        bundle = self.bundle()
        observed = (bundle.bundle_id, bundle.content_digest, bundle.split_group, bundle.release_status)
        expected = (self.bundle_id, self.bundle_digest, self.split_group, self.release_status)
        if observed != expected:
            raise ValueError(f"bundle row identity differs from its payload: expected {expected}, got {observed}")
        if self.bundle_json != canonical_json(bundle):
            raise ValueError("bundle_json is not Taskforge canonical JSON")
        return self

    def bundle(self) -> OpenEndedTaskBundle:
        """Decode and validate the hydrated bundle payload."""
        return OpenEndedTaskBundle.model_validate_json(self.bundle_json)


def bundle_record(bundle: OpenEndedTaskBundle) -> OpenEndedBundleRecord:
    """Return the canonical self-verifying record for ``bundle``."""
    return OpenEndedBundleRecord(
        bundle_id=bundle.bundle_id,
        bundle_digest=bundle.content_digest,
        split_group=bundle.split_group,
        release_status=bundle.release_status,
        bundle_json=canonical_json(bundle),
    )


def record_json(record: OpenEndedBundleRecord) -> str:
    """Return canonical JSON for one bundle record."""
    return canonical_json(record)


def load_record_json(value: str | bytes) -> OpenEndedBundleRecord:
    """Load one self-verifying bundle record from JSON."""
    return OpenEndedBundleRecord.model_validate_json(value)


def write_bundle_parquet(destination: str | Path | BinaryIO, bundles: Iterable[OpenEndedTaskBundle]) -> None:
    """Write bundles to one deterministic Parquet file ordered by bundle ID."""
    records = sorted((bundle_record(bundle) for bundle in bundles), key=lambda item: item.bundle_id)
    ids = [record.bundle_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("a bundle Parquet file cannot repeat bundle_id")
    table = pa.Table.from_pylist(
        [record.model_dump(mode="json") for record in records],
        schema=BUNDLE_PARQUET_SCHEMA,
    )
    pq.write_table(table, destination, compression="zstd", write_statistics=True)


def read_bundle_parquet(source: str | Path | BinaryIO) -> tuple[OpenEndedTaskBundle, ...]:
    """Read and verify every bundle in one canonical Parquet file."""
    table = pq.read_table(source)
    if not table.schema.equals(BUNDLE_PARQUET_SCHEMA, check_metadata=False):
        raise ValueError(f"unexpected open-ended bundle Parquet schema: {table.schema}")
    records = tuple(OpenEndedBundleRecord.model_validate(row) for row in table.to_pylist())
    ids = [record.bundle_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("a bundle Parquet file cannot repeat bundle_id")
    if ids != sorted(ids):
        raise ValueError("bundle Parquet rows must be ordered by bundle_id")
    return tuple(record.bundle() for record in records)


def record_from_mapping(row: Mapping[str, Any]) -> OpenEndedBundleRecord:
    """Validate a row supplied by a range-based Parquet reader such as Zephyr."""
    return OpenEndedBundleRecord.model_validate(row)
