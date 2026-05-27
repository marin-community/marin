# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCP preemption watcher: polls the metadata server and latches on confirmed preempt."""

import logging
import threading
from collections.abc import Callable

from rigging.timing import Duration

from iris.cluster.worker.env_probe import _get_gcp_metadata, _is_gcp_vm

logger = logging.getLogger(__name__)

_PREEMPTED_TRUE = "TRUE"


class PreemptWatcher:
    """Polls GCP instance metadata for preemption notice and fires ``on_preempt`` once.

    The watcher only latches on a confirmed ``"TRUE"`` body from the
    ``preempted`` metadata endpoint. Metadata-server errors, timeouts, and
    non-``TRUE`` responses keep polling; they never trigger the callback.
    Once latched, ``run`` returns — preemption is monotonic.

    On non-GCP hosts (no DMI signal), ``run`` returns immediately. The
    controller-side rule covers abrupt loss on every platform.
    """

    def __init__(
        self,
        on_preempt: Callable[[], None],
        poll: Duration = Duration.from_seconds(1.0),
        metadata_fetcher: Callable[[str], str | None] = _get_gcp_metadata,
        is_gcp_vm: Callable[[], bool] = _is_gcp_vm,
    ):
        self._on_preempt = on_preempt
        self._poll = poll
        self._fetch = metadata_fetcher
        self._is_gcp_vm = is_gcp_vm

    def run(self, stop_event: threading.Event | None = None) -> None:
        """Poll the metadata server until preempted or stopped."""
        stop = stop_event if stop_event is not None else threading.Event()
        if not self._is_gcp_vm():
            return
        poll_seconds = self._poll.to_seconds()
        while not stop.is_set():
            if self._fetch("preempted") == _PREEMPTED_TRUE:
                logger.warning("GCP preempt signal received from metadata server")
                self._on_preempt()
                return
            stop.wait(poll_seconds)
