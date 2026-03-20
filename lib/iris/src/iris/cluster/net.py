# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Network utilities shared across the cluster package."""

from __future__ import annotations

import os
import socket
from pathlib import Path


def find_free_port(start: int = -1) -> int:
    """Find an available port.

    Args:
        start: Starting port for sequential scan. Default of -1 lets the kernel
            pick a random ephemeral port, which avoids collisions when multiple
            processes search for ports concurrently (e.g. pytest-xdist).
    """
    if start == -1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    for port in range(start, start + 1000):
        lock = Path(f"/tmp/iris/port_{port}")
        try:
            os.kill(int(lock.read_text()), 0)
            continue  # port locked by a live process
        except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
            pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                lock.parent.mkdir(parents=True, exist_ok=True)
                lock.write_text(str(os.getpid()))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{start + 1000}")
