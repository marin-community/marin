# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version-aware object checks for EvalDash's PostgreSQL reconciler."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from marin.evaluation.records import RECORD_FILE, EvalRunRecord
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath

from .results_db import ObservationKind, RecordObservation, SourceState

_MAX_RECORD_READERS = 16


@dataclass(frozen=True)
class VerificationSchedule:
    """Timing for every object check in one reconciliation pass."""

    checked_at: datetime
    retry_after: float
    revalidate_after: float


def _utc(datetime_value: datetime) -> datetime:
    if datetime_value.tzinfo is None:
        return datetime_value.replace(tzinfo=UTC)
    return datetime_value.astimezone(UTC)


def _initial_next_verification(path: str, schedule: VerificationSchedule) -> datetime:
    """Spread newly cataloged objects across the configured revalidation window."""
    slots = max(1, int(schedule.revalidate_after // schedule.retry_after))
    phase = int.from_bytes(hashlib.blake2b(path.encode(), digest_size=4).digest()) % slots + 1
    return schedule.checked_at + timedelta(seconds=min(phase * schedule.retry_after, schedule.revalidate_after))


def _record_run_id(path: str) -> str:
    record_path = StoragePath(path)
    if record_path.name != RECORD_FILE or not record_path.parent.name:
        raise ValueError(f"record path must end with a run directory and {RECORD_FILE!r}: {path!r}")
    return record_path.parent.name


def _invalid_observation(
    path: str,
    version: str | None,
    schedule: VerificationSchedule,
    error: str,
) -> RecordObservation:
    return RecordObservation(
        path=path,
        object_version=version,
        verified_at=schedule.checked_at,
        next_verify_at=schedule.checked_at + timedelta(seconds=schedule.retry_after),
        kind=ObservationKind.CHANGED,
        run_id=_record_run_id(path),
        error=error,
    )


def _missing_observation(path: str, schedule: VerificationSchedule) -> RecordObservation:
    return RecordObservation(
        path=path,
        object_version=None,
        verified_at=schedule.checked_at,
        next_verify_at=schedule.checked_at + timedelta(seconds=schedule.retry_after),
        kind=ObservationKind.MISSING,
        run_id=_record_run_id(path),
        error="FileNotFoundError: record object is missing",
    )


def _read_changed_record(
    path: str,
    schedule: VerificationSchedule,
    *,
    initial: bool,
) -> RecordObservation:
    try:
        found = conditional_object(path).read()
    except Exception as exc:
        return _invalid_observation(path, None, schedule, f"{type(exc).__name__}: {exc}")
    if found is None:
        return _missing_observation(path, schedule)
    try:
        record = EvalRunRecord.model_validate_json(found.data)
        expected_run_id = _record_run_id(path)
        if record.run_id != expected_run_id:
            raise ValueError(f"path run ID {expected_run_id!r} does not match record run ID {record.run_id!r}")
    except Exception as exc:
        return _invalid_observation(path, found.version, schedule, f"{type(exc).__name__}: {exc}")
    next_verify_at = (
        _initial_next_verification(path, schedule)
        if initial
        else schedule.checked_at + timedelta(seconds=schedule.revalidate_after)
    )
    return RecordObservation(
        path=path,
        object_version=found.version,
        verified_at=schedule.checked_at,
        next_verify_at=next_verify_at,
        kind=ObservationKind.CHANGED,
        run_id=record.run_id,
        record=record,
    )


def _inspect_record(
    path: str,
    state: SourceState | None,
    schedule: VerificationSchedule,
) -> RecordObservation:
    if state is None:
        return _read_changed_record(path, schedule, initial=True)
    try:
        version = conditional_object(path).version()
    except Exception as exc:
        return RecordObservation(
            path=path,
            object_version=state.object_version,
            verified_at=state.last_verified_at,
            next_verify_at=schedule.checked_at + timedelta(seconds=schedule.retry_after),
            kind=ObservationKind.UNCHANGED,
            error=f"{type(exc).__name__}: {exc}",
        )
    if version is None:
        return _missing_observation(path, schedule)
    if version == state.object_version and state.error is None:
        return RecordObservation(
            path=path,
            object_version=version,
            verified_at=schedule.checked_at,
            next_verify_at=schedule.checked_at + timedelta(seconds=schedule.revalidate_after),
            kind=ObservationKind.UNCHANGED,
        )
    return _read_changed_record(path, schedule, initial=False)


def inspect_record_paths(
    paths: list[str],
    states: dict[str, SourceState],
    schedule: VerificationSchedule,
) -> list[RecordObservation]:
    """Read new objects and version-check catalog entries whose persisted deadline is due."""
    due = [
        path
        for path in paths
        if (state := states.get(path)) is None or _utc(state.next_verify_at) <= schedule.checked_at
    ]

    def inspect(path: str) -> RecordObservation:
        return _inspect_record(path, states.get(path), schedule)

    with ThreadPoolExecutor(max_workers=min(_MAX_RECORD_READERS, max(1, len(due)))) as executor:
        return list(executor.map(inspect, due))
