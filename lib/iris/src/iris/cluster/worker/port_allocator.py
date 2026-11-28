# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Port allocation for task containers."""

import logging
import threading
from collections.abc import Callable
from contextlib import ExitStack

from rigging.port_lease import DYNAMIC_TCP_PORTS, claim_tcp_port, try_claim_free_tcp_port

logger = logging.getLogger(__name__)

# Worker bootstrap reserves the task range from automatic kernel assignment.
DEFAULT_TASK_PORT_RANGE: tuple[int, int] = (DYNAMIC_TCP_PORTS.start, DYNAMIC_TCP_PORTS.stop)


class PortAllocator:
    """Hold Iris task port claims until their containers exit."""

    def __init__(self, port_range: tuple[int, int] = DEFAULT_TASK_PORT_RANGE):
        self._range = port_range
        self._allocated: dict[int, Callable[[], None]] = {}
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        with self._lock, ExitStack() as pending:
            allocated: dict[int, Callable[[], None]] = {}
            for port in range(*self._range):
                if len(allocated) == count:
                    break
                if port in self._allocated:
                    continue
                if (release := try_claim_free_tcp_port(port)) is not None:
                    pending.callback(release)
                    allocated[port] = release
            if len(allocated) != count:
                logger.warning("Port allocation exhausted: no free ports in range %d-%d", *self._range)
                raise RuntimeError("No free ports available")
            self._allocated.update(allocated)
            pending.pop_all()
            return list(allocated)

    def reserve(self, ports: list[int]) -> None:
        """Mark an externally-known port set as taken.

        Used during worker restart to re-claim the host ports of an adopted
        container so the allocator never re-hands them to a new task.
        """
        with self._lock, ExitStack() as pending:
            adopted: dict[int, Callable[[], None]] = {}
            for port in ports:
                if port in self._allocated or port in adopted:
                    continue
                release = claim_tcp_port(port)
                pending.callback(release)
                adopted[port] = release
            self._allocated.update(adopted)
            pending.pop_all()

    def release(self, ports: list[int]) -> None:
        with self._lock:
            for port in ports:
                if (release := self._allocated.pop(port, None)) is not None:
                    release()
