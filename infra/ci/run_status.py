# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Status handoff for GitHub workflows that launch region-local jobs."""

import contextlib
import dataclasses
import enum
import json
import logging
from collections.abc import Generator
from dataclasses import dataclass

from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath

logger = logging.getLogger(__name__)


class _RunState(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class _RunStatusRecord:
    status: _RunState
    marin_prefix: str


def _write_run_status(path: str, record: _RunStatusRecord) -> None:
    destination = StoragePath(path)
    destination.parent.mkdirs(exist_ok=True)
    with atomic_rename(path) as temporary_path:
        with open_url(temporary_path, "w") as stream:
            json.dump(dataclasses.asdict(record), stream, sort_keys=True)
            stream.write("\n")
    logger.info("Wrote %s run status to %s", record.status, path)


@contextlib.contextmanager
def run_status(path: str | None, *, marin_prefix: str) -> Generator[None, None, None]:
    """Record running and terminal state at a workflow-owned storage path."""
    if path is None:
        yield
        return

    _write_run_status(path, _RunStatusRecord(_RunState.RUNNING, marin_prefix))
    try:
        yield
    except BaseException:
        _write_run_status(path, _RunStatusRecord(_RunState.FAILED, marin_prefix))
        raise
    else:
        _write_run_status(path, _RunStatusRecord(_RunState.SUCCEEDED, marin_prefix))
