# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Port allocation for task containers."""

import logging
import threading
from contextlib import ExitStack

from rigging.port_lease import DYNAMIC_TCP_PORTS, claim_tcp_port, lease_tcp_ports

logger = logging.getLogger(__name__)

# Worker bootstrap reserves the task range from automatic kernel assignment.
DEFAULT_TASK_PORT_RANGE: tuple[int, int] = (DYNAMIC_TCP_PORTS.start, DYNAMIC_TCP_PORTS.stop)


class PortAllocator:
    """Hold Iris task port claims until their containers exit."""

    def __init__(self, port_range: tuple[int, int] = DEFAULT_TASK_PORT_RANGE):
        self._range = port_range
        self._allocated: dict[int, ExitStack] = {}
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        with self._lock, ExitStack() as pending:
            allocated: dict[int, ExitStack] = {}
            for port in range(*self._range):
                if len(allocated) == count:
                    break
                if port in self._allocated:
                    continue
                lease = pending.enter_context(ExitStack())
                try:
                    lease.enter_context(lease_tcp_ports(candidates=(port,)))
                except RuntimeError:
                    continue
                allocated[port] = lease
            if len(allocated) != count:
                logger.warning("Port allocation exhausted: no free ports in range %d-%d", *self._range)
                raise RuntimeError("No free ports available")
            for port, lease in allocated.items():
                self._allocated[port] = lease.pop_all()
            return list(allocated)

    def reserve(self, ports: list[int]) -> None:
        """Mark an externally-known port set as taken.

        Used during worker restart to re-claim the host ports of an adopted
        container so the allocator never re-hands them to a new task.
        """
        with self._lock, ExitStack() as pending:
            adopted: dict[int, ExitStack] = {}
            for port in ports:
                if port in self._allocated or port in adopted:
                    continue
                lease = pending.enter_context(ExitStack())
                lease.enter_context(claim_tcp_port(port))
                adopted[port] = lease
            for port, lease in adopted.items():
                self._allocated[port] = lease.pop_all()

    def release(self, ports: list[int]) -> None:
        with self._lock:
            for port in ports:
                lease = self._allocated.pop(port, None)
                if lease is not None:
                    lease.close()
