# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable run status for jobs whose bulk outputs follow their Iris placement."""

import contextlib
import dataclasses
import datetime
import enum
import json
import logging
import subprocess
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from typing import Any

from google.protobuf import json_format
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.factory import open_url
from rigging.filesystem.storage_path import StoragePath

from iris.cluster.client.job_info import get_job_info

logger = logging.getLogger(__name__)
RUN_STATUS_SCHEMA_VERSION = 1


class RunState(enum.StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


_TERMINAL_RUN_STATES = frozenset({RunState.SUCCEEDED, RunState.FAILED})


@dataclass(frozen=True)
class RunStatusRecord:
    """Versioned status object persisted by :class:`RunStatusWriter`."""

    schema_version: int
    status: RunState
    marin_prefix: str
    requested_tpu_types: tuple[str, ...]
    job_id: str | None
    task_id: str | None
    attempt_id: int | None
    worker_id: str | None
    worker_region: str | None
    worker_device: dict[str, Any] | None
    started_at: str
    updated_at: str
    finished_at: str | None
    exit_code: int | None
    error_type: str | None


@dataclass
class RunStatusWriter:
    """Write one small status object outside a job's region-local output tree."""

    path: str
    output_prefix: str | None = None
    requested_tpu_types: Sequence[str] = ()
    started_at: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.UTC))

    def write(self, status: RunState, *, exit_code: int | None = None, error_type: str | None = None) -> None:
        now = datetime.datetime.now(datetime.UTC)
        info = get_job_info()
        device = None
        if info is not None and info.worker_device is not None:
            device = json_format.MessageToDict(info.worker_device, preserving_proto_field_name=True)
        record = RunStatusRecord(
            schema_version=RUN_STATUS_SCHEMA_VERSION,
            status=status,
            marin_prefix=self.output_prefix or marin_prefix(),
            requested_tpu_types=tuple(self.requested_tpu_types),
            job_id=str(info.job_id) if info is not None else None,
            task_id=str(info.task_id) if info is not None else None,
            attempt_id=info.attempt_id if info is not None else None,
            worker_id=info.worker_id if info is not None else None,
            worker_region=info.worker_region if info is not None else None,
            worker_device=device,
            started_at=self.started_at.isoformat(),
            updated_at=now.isoformat(),
            finished_at=now.isoformat() if status in _TERMINAL_RUN_STATES else None,
            exit_code=exit_code,
            error_type=error_type,
        )
        destination = StoragePath(self.path)
        destination.parent.mkdirs(exist_ok=True)
        with atomic_rename(self.path) as temporary_path:
            with open_url(temporary_path, "w") as stream:
                json.dump(dataclasses.asdict(record), stream, sort_keys=True)
                stream.write("\n")
        logger.info("Wrote run status %s to %s", status, self.path)


@contextlib.contextmanager
def run_status(
    path: str | None,
    *,
    output_prefix: str | None = None,
    requested_tpu_types: Sequence[str] = (),
) -> Generator[None, None, None]:
    """Write running and terminal status around a unit of work.

    Only task zero writes when called from a replicated Iris job, so a single
    fixed path remains useful for single- and multi-replica jobs.
    """

    info = get_job_info()
    writer = (
        RunStatusWriter(path, output_prefix=output_prefix, requested_tpu_types=requested_tpu_types)
        if path is not None and (info is None or info.task_index == 0)
        else None
    )
    if writer is not None:
        writer.write(RunState.RUNNING)
    try:
        yield
    except BaseException as exc:
        if writer is not None:
            exit_code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else None
            writer.write(RunState.FAILED, exit_code=exit_code, error_type=type(exc).__name__)
        raise
    else:
        if writer is not None:
            writer.write(RunState.SUCCEEDED, exit_code=0)


def run_command_with_status(
    command: Sequence[str],
    status_path: str,
    output_prefix: str | None,
    requested_tpu_types: Sequence[str],
) -> None:
    """Run ``command`` and preserve its exit status while writing durable status."""

    with run_status(
        status_path,
        output_prefix=output_prefix,
        requested_tpu_types=requested_tpu_types,
    ):
        completed = subprocess.run(list(command), check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
