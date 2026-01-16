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

import logging
import threading

from fluster.client.client import FlusterClient
from fluster.cluster.types import JobId

logger = logging.getLogger(__name__)


class LogPoller:
    """Background thread that polls for job logs and writes them to logger.info.

    Can be used as a context manager for automatic start/stop:

    Example:
        with LogPoller(client, job_id):
            # logs are automatically polled and written to logger.info
            # ... wait for job ...

    Or manually:
        poller = LogPoller(client, job_id)
        poller.start()
        # ... wait for job ...
        poller.stop()
    """

    def __init__(
        self,
        client: FlusterClient,
        job_id: JobId,
        poll_interval: float = 1.0,
    ):
        """Initialize log poller.

        Args:
            client: FlusterClient to use for fetching logs
            job_id: Job ID to poll logs for
            poll_interval: Seconds between polls
        """
        self._client = client
        self._job_id = job_id
        self._poll_interval = poll_interval
        self._last_timestamp_ms = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
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

    def __enter__(self) -> "LogPoller":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

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
                    logger.info("[%s] %s", entry.source, entry.data)
            except (ConnectionError, OSError, ValueError):
                pass

            self._stop_event.wait(self._poll_interval)
