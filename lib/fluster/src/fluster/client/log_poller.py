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

"""Log polling utility for streaming job logs."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fluster.client.client import FlusterClient, LogEntry
    from fluster.cluster.types import JobId


class LogPoller:
    """Background thread that polls for job logs and calls a handler.

    Example:
        def print_log(entry: LogEntry):
            print(f"[{entry.source}] {entry.data}")

        poller = LogPoller(client, job_id, print_log)
        poller.start()
        # ... wait for job ...
        poller.stop()
    """

    def __init__(
        self,
        client: FlusterClient,
        job_id: JobId,
        handler: Callable[[LogEntry], None],
        poll_interval: float = 1.0,
    ):
        """Initialize log poller.

        Args:
            client: FlusterClient to use for fetching logs
            job_id: Job ID to poll logs for
            handler: Callback function for each new log entry
            poll_interval: Seconds between polls
        """
        self._client = client
        self._job_id = job_id
        self._handler = handler
        self._poll_interval = poll_interval
        self._last_timestamp_ms = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling for logs in background thread."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Stop polling thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        self._stop_event.clear()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                logs = self._client.fetch_logs(
                    self._job_id,
                    start_ms=self._last_timestamp_ms,
                )
                for entry in logs:
                    if entry.timestamp_ms > self._last_timestamp_ms:
                        self._last_timestamp_ms = entry.timestamp_ms
                    self._handler(entry)
            except (ConnectionError, OSError, ValueError):
                pass

            self._stop_event.wait(self._poll_interval)
