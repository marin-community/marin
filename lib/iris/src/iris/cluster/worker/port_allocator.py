# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Port allocation for task containers."""

import logging
import socket
import threading

logger = logging.getLogger(__name__)

# Default task named-port range (end exclusive). Must stay below the kernel
# ephemeral floor (EPHEMERAL_PORT_RANGE in runtime/docker.py) so co-tenant
# outbound sockets can never squat an allocated port (#7392).
DEFAULT_TASK_PORT_RANGE: tuple[int, int] = (12000, 14000)


class PortAllocator:
    """Allocate task named ports from a range below the kernel ephemeral floor."""

    def __init__(self, port_range: tuple[int, int] = DEFAULT_TASK_PORT_RANGE):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self, count: int = 1) -> list[int]:
        with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    def reserve(self, ports: list[int]) -> None:
        """Mark an externally-known port set as taken.

        Used during worker restart to re-claim the host ports of an adopted
        container so the allocator never re-hands them to a new task.
        """
        with self._lock:
            self._allocated.update(ports)

    def release(self, ports: list[int]) -> None:
        with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        logger.warning("Port allocation exhausted: no free ports in range %d-%d", self._range[0], self._range[1])
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False
