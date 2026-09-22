# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clear a shared uv cache after repeated task-local recoveries."""

import logging
import subprocess
import threading
from pathlib import Path

from rigging import telemetry
from rigging.telemetry.serialization import EventBody
from rigging.timing import Duration, Timestamp

from iris.cluster.runtime.env import UV_CACHE_PATH, UV_CACHE_RECOVERY_SIGNAL_PREFIX, cache_host_dirname

logger = logging.getLogger(__name__)

UV_CACHE_RECOVERY_THRESHOLD = 3
UV_CACHE_RECOVERY_WINDOW = Duration.from_minutes(30)
UV_CACHE_RECOVERY_INTERVAL = Duration.from_seconds(30)
UV_CACHE_CLEAN_TIMEOUT = Duration.from_minutes(6)
UV_CACHE_RECOVERY_OBSERVED_EVENT = "uv_cache_recovery_observed"
UV_CACHE_CLEARED_EVENT = "uv_cache_cleared"
_OBSERVED_RECOVERY_SIGNAL_PREFIX = ".iris-recovery-observed-"
_RECOVERY_FAILURES = telemetry.counter("iris_uv_cache_recovery_failures", unit="{failure}")


def _uv_cache_dir(cache_dir: Path) -> Path:
    return cache_dir / cache_host_dirname(UV_CACHE_PATH)


def _emit_event(name: str, **fields: str | int | float | bool) -> None:
    telemetry.event(name, EventBody(fields))


def _prune_and_count_recovery_signals(cache_dir: Path, now: Timestamp) -> int:
    cutoff = now.add_ms(-UV_CACHE_RECOVERY_WINDOW.to_ms())
    attempt_uids: set[str] = set()
    for signal in _uv_cache_dir(cache_dir).glob(f"{UV_CACHE_RECOVERY_SIGNAL_PREFIX}*"):
        if Timestamp.from_seconds(signal.stat().st_mtime) < cutoff:
            signal.unlink()
            continue
        if signal.name.startswith(_OBSERVED_RECOVERY_SIGNAL_PREFIX):
            attempt_uid = signal.name.removeprefix(_OBSERVED_RECOVERY_SIGNAL_PREFIX)
        else:
            attempt_uid = signal.name.removeprefix(UV_CACHE_RECOVERY_SIGNAL_PREFIX)
            _emit_event(UV_CACHE_RECOVERY_OBSERVED_EVENT, attempt_uid=attempt_uid)
            signal.replace(signal.with_name(f"{_OBSERVED_RECOVERY_SIGNAL_PREFIX}{attempt_uid}"))
        attempt_uids.add(attempt_uid)
    return len(attempt_uids)


def reconcile_uv_cache_recovery(
    cache_dir: Path,
    *,
    now: Timestamp,
    uv_executable: str = "uv",
) -> None:
    """Clear the cache when enough distinct tasks needed local recovery."""
    recovery_count = _prune_and_count_recovery_signals(cache_dir, now)
    if recovery_count < UV_CACHE_RECOVERY_THRESHOLD:
        return

    uv_cache_dir = _uv_cache_dir(cache_dir)
    subprocess.run(
        [uv_executable, "cache", "clean", "--cache-dir", str(uv_cache_dir)],
        check=True,
        timeout=UV_CACHE_CLEAN_TIMEOUT.to_seconds(),
    )
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    logger.warning("cleared uv cache %s after %d task recoveries", uv_cache_dir, recovery_count)
    _emit_event(
        UV_CACHE_CLEARED_EVENT,
        cache_path=str(uv_cache_dir),
        recovery_count=recovery_count,
        recovery_window_seconds=UV_CACHE_RECOVERY_WINDOW.to_seconds(),
    )


def run_uv_cache_recovery(cache_dir: Path, stop: threading.Event) -> None:
    """Watch task recovery signals and clear a repeatedly failing cache."""
    while not stop.is_set():
        try:
            reconcile_uv_cache_recovery(cache_dir, now=Timestamp.now())
        except (OSError, subprocess.SubprocessError) as error:
            logger.exception("uv cache recovery check failed for %s", cache_dir)
            _RECOVERY_FAILURES.add(1, attributes={"failure_kind": type(error).__name__})
        stop.wait(UV_CACHE_RECOVERY_INTERVAL.to_seconds())
