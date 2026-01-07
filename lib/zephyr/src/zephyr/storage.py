# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Storage abstractions for Zephyr chunk serialization.

This module provides InlineRef and StorageRef types for representing chunk data
that is either kept in memory (small) or spilled to storage (large).
"""

from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# Threshold for inline vs storage decision (1MB)
DEFAULT_SPILL_THRESHOLD_BYTES = 1 * 1024 * 1024


@dataclass
class InlineRef:
    """Data kept inline in memory (small chunks)."""

    data: list[Any]

    @property
    def count(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)


@dataclass
class StorageRef:
    """Reference to data stored in GCS/S3/local filesystem."""

    path: str
    count: int

    def __iter__(self) -> Iterator[dict]:
        from zephyr.readers import load_vortex

        return load_vortex(self.path)


# Union type for chunk references
ChunkRef = InlineRef | StorageRef


def _estimate_item_size(item: Any) -> int:
    """Estimate serialized size of an item."""
    if isinstance(item, dict):
        total = 0
        for k, v in item.items():
            total += _estimate_item_size(k) + _estimate_item_size(v)
        return total
    return sys.getsizeof(item)


class ChunkWriter:
    """Writes chunk data, choosing inline vs storage based on size threshold.

    Small chunks (< spill_threshold_bytes) are kept inline in memory.
    Large chunks are written to storage as Vortex files.

    Usage:
        writer = ChunkWriter(spill_path="/tmp/chunk.vortex")
        for item in items:
            writer.write(item)
        ref = writer.finish()  # Returns InlineRef or StorageRef
    """

    def __init__(
        self,
        spill_path: str,
        spill_threshold_bytes: int = DEFAULT_SPILL_THRESHOLD_BYTES,
    ):
        self.spill_path = spill_path
        self.spill_threshold_bytes = spill_threshold_bytes
        self._items: list[Any] = []
        self._size_estimate = 0

    def write(self, item: Any) -> None:
        """Add item to chunk."""
        self._items.append(item)
        self._size_estimate += _estimate_item_size(item)

    def finish(self) -> ChunkRef:
        """Finalize and return appropriate ref type."""
        if self._size_estimate < self.spill_threshold_bytes:
            return InlineRef(data=self._items)

        from zephyr.writers import write_vortex_file

        result = write_vortex_file(self._items, self.spill_path)
        return StorageRef(path=result["path"], count=result["count"])


class StorageManager:
    """Manages storage paths and cleanup for a job execution.

    Usage:
        with StorageManager() as storage:
            # ... use chunk_path/job_path to reference or write chunks
        # Job directory cleaned up on exit
    """

    def __init__(
        self,
        base_path: str | None = None,
        spill_threshold_bytes: int = DEFAULT_SPILL_THRESHOLD_BYTES,
    ):
        if base_path is None:
            prefix = os.environ.get("MARIN_PREFIX", "/tmp")
            base_path = f"{prefix}/zephyr/tmp/"

        self.base_path = base_path.rstrip("/")
        self.job_id = str(uuid.uuid4())[:8]
        self.spill_threshold_bytes = spill_threshold_bytes
        self._stage_idx = 0

    @property
    def job_path(self) -> str:
        return f"{self.base_path}/job_{self.job_id}"

    def chunk_path(self, shard_idx: int, chunk_idx: int) -> str:
        return f"{self.job_path}/stage_{self._stage_idx}/shard_{shard_idx:05d}_chunk_{chunk_idx:05d}.vortex"

    def create_writer(self, shard_idx: int, chunk_idx: int) -> ChunkWriter:
        """Create a ChunkWriter for the given shard/chunk."""
        path = self.chunk_path(shard_idx, chunk_idx)
        return ChunkWriter(spill_path=path, spill_threshold_bytes=self.spill_threshold_bytes)

    def next_stage(self) -> None:
        """Advance to next stage."""
        self._stage_idx += 1

    def cleanup(self) -> None:
        """Best-effort cleanup of job directory."""
        import fsspec

        try:
            fs, _ = fsspec.core.url_to_fs(self.job_path)
            if fs.exists(self.job_path):
                fs.rm(self.job_path, recursive=True)
        except Exception:
            pass  # ignore any failures during cleanup

    def __enter__(self) -> StorageManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.cleanup()
        return False
