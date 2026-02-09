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

"""File-based distributed queue implementation, mostly for testing."""

from __future__ import annotations

import logging
import pickle
import time
import uuid
from os import PathLike
from pathlib import Path
from typing import Any, TypeVar

import fsspec
from fray.v1.queue.base import Lease, Queue

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FileQueue(Queue[T]):
    """Tracks queue items using a set of lease directories and atomic mv operations.

    New items are inserted into the `pending` directory. Workers scan this directory for
    available items and move them to the `processing` directory. Items are then removed
    from the `processing` directory when completed.
    """

    def __init__(self, path: PathLike, fs_args: dict[str, Any] | None = None):
        """Initialize FileQueue.

        Args:
            path: Base path for the queue (e.g., "gs://my-bucket/queue-1" or "/tmp/queue-1")
            fs_args: Additional arguments for fsspec.filesystem
        """
        self.path = Path(path)
        self.fs_args = fs_args or {}

        self.fs, self.fs_path = fsspec.core.url_to_fs(path, **self.fs_args)

        self.pending_dir = self.path / "pending"
        self.processing_dir = self.path / "processing"

        self.fs.makedirs(self.pending_dir, exist_ok=True)
        self.fs.makedirs(self.processing_dir, exist_ok=True)

    def __getstate__(self):
        return {
            "path": self.path,
            "fs_args": self.fs_args,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def push(self, item: T) -> None:
        timestamp = time.time()
        unique_id = uuid.uuid4()
        filename = f"{timestamp:.6f}_{unique_id}.pkl"

        with self.fs.open(self.pending_dir / filename, "wb") as f:
            pickle.dump(item, f)

    def peek(self) -> T | None:
        files = sorted(self.fs.ls(str(self.pending_dir), detail=False))
        files = [f for f in files if f.rstrip("/") != str(self.pending_dir).rstrip("/")]

        if not files:
            return None

        with self.fs.open(files[0], "rb") as f:
            return pickle.load(f)

    def pop(self, lease_timeout: float = 60.0) -> Lease[T] | None:
        self._recover_expired_leases()

        try:
            files = sorted(self.fs.ls(str(self.pending_dir), detail=False))
            files = [f for f in files if f.rstrip("/") != str(self.pending_dir).rstrip("/")]
        except Exception:
            return None

        for file_path in files:
            filename = file_path.split("/")[-1]

            lease_expiry = time.time() + lease_timeout
            new_name = f"{filename}__expiry_{lease_expiry:.6f}"

            try:
                self.fs.mv(file_path, self.processing_dir / new_name)
                with self.fs.open(self.processing_dir / new_name, "rb") as f:
                    item = pickle.load(f)

                return Lease(item=item, lease_id=new_name, timestamp=time.time())
            except FileNotFoundError:
                # Someone else took it
                continue

        return None

    def _recover_expired_leases(self) -> None:
        """Check processing directory for expired leases and move them back to pending."""
        try:
            files = self.fs.ls(str(self.processing_dir), detail=False)
        except Exception:
            return

        now = time.time()
        for file_path in files:
            filename = file_path.split("/")[-1]
            try:
                parts = filename.split("__expiry_")
                if len(parts) != 2:
                    continue

                expiry = float(parts[1])

                if now > expiry:
                    logger.warning(f"Lease expired for {filename}, requeuing")
                    # Use timestamp 0.0 to put at front of queue
                    timestamp = 0.0
                    unique_id = uuid.uuid4()
                    new_filename = f"{timestamp:.6f}_{unique_id}.pkl"

                    try:
                        self.fs.mv(file_path, self.pending_dir / new_filename)
                    except FileNotFoundError:
                        # File was already moved or deleted by another process
                        pass
            except ValueError:
                continue

    def done(self, lease: Lease[T]) -> None:
        try:
            self.fs.rm(self.processing_dir / lease.lease_id)
        except FileNotFoundError:
            raise ValueError(f"Invalid lease: {lease.lease_id} not found (already done or expired)") from None

    def release(self, lease: Lease[T]) -> None:
        timestamp = 0.0
        unique_id = uuid.uuid4()
        filename = f"{timestamp:.6f}_{unique_id}.pkl"

        try:
            self.fs.mv(self.processing_dir / lease.lease_id, self.pending_dir / filename)
        except FileNotFoundError:
            raise ValueError(f"Invalid lease: {lease.lease_id} not found during release") from None
