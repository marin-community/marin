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

"""File-based distributed queue implementation using fsspec."""

from __future__ import annotations

import logging
import pickle
import threading
import time
import uuid
from typing import Any, TypeVar

import fsspec
from fray.queues.base import Lease, Queue

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FileQueue(Queue[T]):
    """Queue implementation using fsspec for storage and coordination.

    Supports any filesystem backend supported by fsspec (local, GCS, S3, etc.).
    Uses atomic moves (if supported by backend) or copy-delete for state transitions.
    """

    def __init__(self, path: str, fs_args: dict[str, Any] | None = None, monitor: bool = True):
        """Initialize FileQueue.

        Args:
            path: Base path for the queue (e.g., "gs://my-bucket/queue-1" or "/tmp/queue-1")
            fs_args: Additional arguments for fsspec.filesystem
            monitor: Whether to start the background lease monitor thread
        """
        self.path = path.rstrip("/")
        self.fs_args = fs_args or {}
        self._monitor = monitor

        # Initialize filesystem
        self.fs, self.fs_path = fsspec.core.url_to_fs(path, **self.fs_args)

        self.pending_dir = f"{self.path}/pending"
        self.processing_dir = f"{self.path}/processing"

        # Create directories
        self.fs.makedirs(self.pending_dir, exist_ok=True)
        self.fs.makedirs(self.processing_dir, exist_ok=True)

        # Start background monitor if requested
        self._stop_event = threading.Event()
        self._monitor_thread = None
        if self._monitor:
            self._monitor_thread = threading.Thread(target=self._monitor_leases, daemon=True)
            self._monitor_thread.start()

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle thread or event
        state.pop("_monitor_thread", None)
        state.pop("_stop_event", None)
        # Don't restart monitor on unpickle
        state["_monitor"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._monitor_thread = None

    def push(self, item: T) -> None:
        # Create unique filename with timestamp for ordering
        timestamp = time.time()
        unique_id = uuid.uuid4()
        filename = f"{timestamp:.6f}_{unique_id}.pkl"

        final_path = f"{self.pending_dir}/{filename}"

        # Write directly to final path (atomic on GCS/S3, usually atomic on local)
        with self.fs.open(final_path, "wb") as f:
            pickle.dump(item, f)

    def peek(self) -> T | None:
        # List pending files
        try:
            files = sorted(self.fs.ls(self.pending_dir, detail=False))
            # Filter out directory itself if returned
            files = [f for f in files if f.rstrip("/") != self.pending_dir.rstrip("/")]

            if not files:
                return None

            with self.fs.open(files[0], "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def pop(self, lease_timeout: float = 60.0) -> Lease[T] | None:
        # List pending files
        try:
            files = sorted(self.fs.ls(self.pending_dir, detail=False))
            files = [f for f in files if f.rstrip("/") != self.pending_dir.rstrip("/")]
        except Exception:
            return None

        for file_path in files:
            # Extract filename from path
            filename = file_path.split("/")[-1]

            # Try to acquire lease by moving to processing
            lease_expiry = time.time() + lease_timeout
            new_name = f"{filename}__expiry_{lease_expiry:.6f}"
            processing_path = f"{self.processing_dir}/{new_name}"

            try:
                # Atomic move if possible
                self.fs.mv(file_path, processing_path)

                # Successfully acquired
                with self.fs.open(processing_path, "rb") as f:
                    item = pickle.load(f)

                return Lease(item=item, lease_id=new_name, timestamp=time.time())
            except FileNotFoundError:
                # Someone else took it
                continue
            except Exception as e:
                logger.error(f"Error acquiring lease for {file_path}: {e}")
                continue

        return None

    def done(self, lease: Lease[T]) -> None:
        processing_path = f"{self.processing_dir}/{lease.lease_id}"
        try:
            self.fs.rm(processing_path)
        except FileNotFoundError:
            logger.warning(f"Lease {lease.lease_id} not found (already done or expired?)")

    def release(self, lease: Lease[T]) -> None:
        processing_path = f"{self.processing_dir}/{lease.lease_id}"

        # Extract original name (remove timeout suffix)
        original_name = lease.lease_id.split("__expiry_")[0]
        pending_path = f"{self.pending_dir}/{original_name}"

        try:
            self.fs.mv(processing_path, pending_path)
        except FileNotFoundError:
            logger.warning(f"Lease {lease.lease_id} not found during release")

    def size(self) -> int:
        try:
            pending = len(self.fs.ls(self.pending_dir, detail=False))
            processing = len(self.fs.ls(self.processing_dir, detail=False))
            return pending + processing
        except Exception:
            return 0

    def pending(self) -> int:
        try:
            return len(self.fs.ls(self.pending_dir, detail=False))
        except Exception:
            return 0

    def _monitor_leases(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(5.0)  # Check less frequently for remote FS
            try:
                now = time.time()
                try:
                    files = self.fs.ls(self.processing_dir, detail=False)
                except Exception:
                    continue

                for file_path in files:
                    filename = file_path.split("/")[-1]
                    try:
                        # Parse expiry from filename
                        parts = filename.split("__expiry_")
                        if len(parts) != 2:
                            continue

                        expiry = float(parts[1])

                        if now > expiry:
                            logger.warning(f"Lease expired for {filename}, requeuing")
                            original_name = parts[0]
                            pending_path = f"{self.pending_dir}/{original_name}"

                            # Move back to pending
                            try:
                                self.fs.mv(file_path, pending_path)
                            except FileNotFoundError:
                                pass  # Race condition

                    except ValueError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing lease {filename}: {e}")

            except Exception as e:
                logger.error(f"Error in lease monitor: {e}")

    def shutdown(self) -> None:
        self._stop_event.set()
        self._monitor_thread.join(timeout=1.0)
