# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared task-health protocol values."""

import os
from pathlib import Path

MAX_PORT = 65535
HEALTH_PATH = "/healthz"
HEALTH_PORT_FILE = "/tmp/iris/health-port"
HEALTH_FAILURE_COUNT_FILE = "/tmp/iris/health-failures"
HEALTH_TERMINATION_FILE = "/tmp/iris/health-termination-log"


def write_health_state(path: Path, value: str) -> None:
    """Replace probe state atomically, with read access for the task user."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o644)
    os.replace(temporary, path)
