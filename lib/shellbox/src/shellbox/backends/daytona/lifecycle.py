# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record provider identities for one process's owned Daytona sandbox requests."""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

LOGGER_NAME = "shellbox.backends.daytona.lifecycle"
logger = logging.getLogger(LOGGER_NAME)


class SandboxLifecycleEvent(StrEnum):
    CREATE_REQUESTED = "create_requested"
    CREATED = "created"
    DELETE_REQUESTED = "delete_requested"
    DELETED = "deleted"
    ABSENT = "absent"


@dataclass(frozen=True)
class SandboxLifecycle:
    event: SandboxLifecycleEvent
    requested_name: str
    sandbox_id: str | None = None
    schema_version: int = 1
    kind: str = "daytona_sandbox_lifecycle"


def record_sandbox_lifecycle(event: SandboxLifecycleEvent, name: str, sandbox_id: str | None = None) -> None:
    """Emit one versioned JSON message without credentials or task payloads."""
    logger.info("%s", json.dumps(asdict(SandboxLifecycle(event, name, sandbox_id)), sort_keys=True))


@contextmanager
def process_lifecycle_log(path: Path | None) -> Iterator[None]:
    """Capture this process's Daytona lifecycle events in a fresh local JSONL file."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous_level)
