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

"""Port allocation for worker jobs."""

import asyncio
import socket


class PortAllocator:
    """Allocate ephemeral ports for jobs.

    Tracks allocated ports to avoid conflicts.
    Ports are released when jobs terminate.
    """

    def __init__(self, port_range: tuple[int, int] = (30000, 40000)):
        self._range = port_range
        self._allocated: set[int] = set()
        self._lock = asyncio.Lock()

    async def allocate(self, count: int = 1) -> list[int]:
        """Allocate N unused ports."""
        async with self._lock:
            ports = []
            for _ in range(count):
                port = self._find_free_port()
                self._allocated.add(port)
                ports.append(port)
            return ports

    async def release(self, ports: list[int]) -> None:
        """Release allocated ports."""
        async with self._lock:
            for port in ports:
                self._allocated.discard(port)

    def _find_free_port(self) -> int:
        """Find an unused port in range."""
        for port in range(self._range[0], self._range[1]):
            if port in self._allocated:
                continue
            if self._is_port_free(port):
                return port
        raise RuntimeError("No free ports available")

    def _is_port_free(self, port: int) -> bool:
        """Check if port is free on host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False
