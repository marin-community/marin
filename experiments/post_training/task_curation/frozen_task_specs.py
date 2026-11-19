# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read exact serialized TaskSpecs from a digest-bound frozen population."""

import hashlib
from collections.abc import Iterator

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath

TASK_SPEC_BATCH_SIZE = 16


class FrozenTaskSpecStream(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    specs_dir: str
    task_column: str
    raw_count: int
    task_spec_scan_sha256: str


def verified_task_spec_rows(source: FrozenTaskSpecStream) -> Iterator[str]:
    """Verify stored JSON cells in sorted Parquet-path and physical row order."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted((StoragePath(source.specs_dir) / "*.parquet").glob(), key=str):
        with path.open("rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(
                batch_size=TASK_SPEC_BATCH_SIZE, columns=[source.task_column]
            ):
                for encoded in batch.column(0).to_pylist():
                    if encoded is None:
                        raise ValueError(f"Frozen converted population contains a rejected row: {source.source_id}")
                    digest.update(encoded.encode())
                    count += 1
                    yield encoded
    if digest.hexdigest() != source.task_spec_scan_sha256 or count != source.raw_count:
        raise ValueError(f"Frozen TaskSpec scan changed: {source.source_id}")
