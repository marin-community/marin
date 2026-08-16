# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version-aware object checks for EvalDash's PostgreSQL reconciler."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from marin.evaluation.records import EvalRunRecord
from rigging.filesystem.conditional_object import conditional_object

from .results_db import RecordObservation, SourceState

_MAX_RECORD_READERS = 16


def _utc(datetime_value: datetime) -> datetime:
    if datetime_value.tzinfo is None:
        return datetime_value.replace(tzinfo=UTC)
    return datetime_value.astimezone(UTC)


def _initial_next_verification(path: str, now: datetime, interval: float, revalidate_after: float) -> datetime:
    """Spread a newly cataloged fleet's first recheck across its first day."""
    slots = max(1, int(revalidate_after // interval))
    phase = int.from_bytes(hashlib.blake2b(path.encode(), digest_size=4).digest()) % slots + 1
    return now + timedelta(seconds=min(phase * interval, revalidate_after))


def _record_run_id(path: str) -> str:
    return path.rstrip("/").rsplit("/", 2)[-2]


def _invalid_observation(
    path: str,
    version: str | None,
    now: datetime,
    interval: float,
    error: str,
) -> RecordObservation:
    return RecordObservation(
        path=path,
        object_version=version,
        verified_at=now,
        next_verify_at=now + timedelta(seconds=interval),
        changed=True,
        run_id=_record_run_id(path),
        error=error,
    )


def _missing_observation(path: str, now: datetime, interval: float) -> RecordObservation:
    return RecordObservation(
        path=path,
        object_version=None,
        verified_at=now,
        next_verify_at=now + timedelta(seconds=interval),
        changed=False,
        missing=True,
        run_id=_record_run_id(path),
        error="FileNotFoundError: record object is missing",
    )


def _read_changed_record(
    path: str,
    now: datetime,
    interval: float,
    revalidate_after: float,
    *,
    initial: bool,
) -> RecordObservation:
    try:
        found = conditional_object(path).read()
    except Exception as exc:
        return _invalid_observation(path, None, now, interval, f"{type(exc).__name__}: {exc}")
    if found is None:
        return _missing_observation(path, now, interval)
    try:
        record = EvalRunRecord.model_validate_json(found.data)
        expected_run_id = _record_run_id(path)
        if record.run_id != expected_run_id:
            raise ValueError(f"path run ID {expected_run_id!r} does not match record run ID {record.run_id!r}")
    except Exception as exc:
        return _invalid_observation(path, found.version, now, interval, f"{type(exc).__name__}: {exc}")
    next_verify_at = (
        _initial_next_verification(path, now, interval, revalidate_after)
        if initial
        else now + timedelta(seconds=revalidate_after)
    )
    return RecordObservation(
        path=path,
        object_version=found.version,
        verified_at=now,
        next_verify_at=next_verify_at,
        changed=True,
        run_id=record.run_id,
        record=record,
    )


def _inspect_record(
    path: str,
    state: SourceState | None,
    now: datetime,
    interval: float,
    revalidate_after: float,
) -> RecordObservation:
    if state is None:
        return _read_changed_record(path, now, interval, revalidate_after, initial=True)
    try:
        version = conditional_object(path).version()
    except Exception as exc:
        return RecordObservation(
            path=path,
            object_version=state.object_version,
            verified_at=state.last_verified_at,
            next_verify_at=now + timedelta(seconds=interval),
            changed=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    if version is None:
        return _missing_observation(path, now, interval)
    if version == state.object_version and state.error is None:
        return RecordObservation(
            path=path,
            object_version=version,
            verified_at=now,
            next_verify_at=now + timedelta(seconds=revalidate_after),
            changed=False,
        )
    return _read_changed_record(path, now, interval, revalidate_after, initial=False)


def inspect_record_paths(
    paths: list[str],
    states: dict[str, SourceState],
    now: datetime,
    interval: float,
    revalidate_after: float,
) -> list[RecordObservation]:
    """Read new objects and HEAD only catalog entries whose persisted deadline is due."""
    due = [path for path in paths if (state := states.get(path)) is None or _utc(state.next_verify_at) <= now]

    def inspect(path: str) -> RecordObservation:
        return _inspect_record(path, states.get(path), now, interval, revalidate_after)

    with ThreadPoolExecutor(max_workers=min(_MAX_RECORD_READERS, max(1, len(due)))) as executor:
        return list(executor.map(inspect, due))
