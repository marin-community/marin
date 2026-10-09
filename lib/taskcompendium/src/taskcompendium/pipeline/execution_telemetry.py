# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source-local attribution of completed Zephyr executions and phase wall time."""

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.coordinator import ZephyrExecutionResult
from zephyr.dataset import Dataset

TELEMETRY_FILENAME = "telemetry.json"
"""Where ``SourceTelemetry.record`` writes below the source artifact."""


class TelemetryStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionTelemetry:
    operation: str
    ordinal: int
    execution_id: str
    status: TelemetryStatus
    wall_seconds: float
    counters: dict[str, int | float]
    error_type: str | None = None


@dataclass
class PhaseTelemetry:
    phase: str
    status: TelemetryStatus = TelemetryStatus.RUNNING
    wall_seconds: float = 0.0
    executions: list[ExecutionTelemetry] = field(default_factory=list)
    error_type: str | None = None


@dataclass
class SourceTelemetry:
    """Retain small final counter snapshots for one source invocation."""

    source: str
    source_artifact_path: str
    attempt_started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: TelemetryStatus = TelemetryStatus.RUNNING
    source_wall_seconds: float = 0.0
    phases: list[PhaseTelemetry] = field(default_factory=list)
    error_type: str | None = None

    @contextmanager
    def record(self) -> Iterator[None]:
        """Persist completed or partial evidence without retaining task payloads."""
        started = time.monotonic()
        try:
            yield
        except BaseException as error:
            self.status = TelemetryStatus.FAILED
            self.error_type = type(error).__name__
            raise
        else:
            self.status = TelemetryStatus.COMPLETED
        finally:
            self.source_wall_seconds = time.monotonic() - started
            path = StoragePath(self.source_artifact_path) / TELEMETRY_FILENAME
            with path.open("wt", auto_mkdir=True) as stream:
                json.dump({"schema_version": 1, **asdict(self)}, stream, indent=2, allow_nan=False)
                stream.write("\n")

    @contextmanager
    def phase(self, name: str) -> Iterator[PhaseTelemetry]:
        """Measure an enclosing logical phase, including driver and storage work."""
        phase = PhaseTelemetry(name)
        self.phases.append(phase)
        started = time.monotonic()
        try:
            yield phase
        except BaseException as error:
            phase.status = TelemetryStatus.FAILED
            phase.error_type = type(error).__name__
            raise
        else:
            phase.status = TelemetryStatus.COMPLETED
        finally:
            phase.wall_seconds = time.monotonic() - started


def execute_phase(
    context: ZephyrContext,
    dataset: Dataset,
    *,
    telemetry: PhaseTelemetry | None = None,
    operation: str = "execute",
    map_task_resources: ResourceConfig | None = None,
    reduce_task_resources: ResourceConfig | None = None,
) -> ZephyrExecutionResult:
    """Return the original execution result and retain only its final counters."""
    started = time.monotonic()
    try:
        result = context.execute(
            dataset, map_task_resources=map_task_resources, reduce_task_resources=reduce_task_resources
        )
    except BaseException as error:
        if telemetry is not None:
            # execute() exposes its ID only on success. Do not invent an ID or
            # final counters for an execution that did not return a result.
            telemetry.executions.append(
                ExecutionTelemetry(
                    operation,
                    len(telemetry.executions),
                    "",
                    TelemetryStatus.FAILED,
                    time.monotonic() - started,
                    {},
                    type(error).__name__,
                )
            )
        raise
    if telemetry is not None:
        telemetry.executions.append(
            ExecutionTelemetry(
                operation,
                len(telemetry.executions),
                result.execution_id,
                TelemetryStatus.COMPLETED,
                time.monotonic() - started,
                dict(result.counters),
            )
        )
    return result
