# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rotate suspect node-local uv caches without breaking live environments."""

import json
import logging
import os
import shutil
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from rigging import telemetry
from rigging.timing import Duration, Timestamp

from iris.cluster.runtime.env import UV_CACHE_PATH, UV_CACHE_REPAIR_MARKER, cache_host_dirname

logger = logging.getLogger(__name__)

UV_CACHE_MAINTENANCE_INTERVAL = Duration.from_seconds(30)
UV_CACHE_ROTATION_MIN_INTERVAL = Duration.from_hours(1)
UV_CACHE_RECLAIM_GRACE = Duration.from_minutes(5)

_GENERATION_PREFIX = ".iris-uv-cache-generation-"
_QUARANTINE_METADATA = ".iris-quarantine.json"
_LAST_ROTATION = ".iris-uv-cache-last-rotation.json"
_TEMP_PREFIX = ".iris-uv-cache-temp-"

_ROTATIONS = telemetry.counter("iris_uv_cache_rotations", unit="{rotation}")
_RECLAIMS = telemetry.counter("iris_uv_cache_reclaims", unit="{generation}")
_FAILURES = telemetry.counter("iris_uv_cache_maintenance_failures", unit="{failure}")


@dataclass(frozen=True)
class UvCacheMaintenanceResult:
    """Changes made by one maintenance pass."""

    rotated: Path | None = None
    reclaimed: tuple[Path, ...] = ()
    rotation_rate_limited: bool = False


@dataclass(frozen=True)
class _QuarantineMetadata:
    consumers: tuple[str, ...]
    rotation_complete: bool
    rotated_at_ms: int
    unused_since_ms: int | None


def _generation_path(cache_dir: Path) -> Path:
    return cache_dir / f"{_GENERATION_PREFIX}{uuid.uuid4().hex}"


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{_TEMP_PREFIX}{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_current_link(current: Path, generation: Path) -> None:
    temporary = current.with_name(f"{_TEMP_PREFIX}{uuid.uuid4().hex}")
    try:
        temporary.symlink_to(generation.name, target_is_directory=True)
        os.replace(temporary, current)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved_generation(cache_dir: Path, current: Path) -> Path:
    generation = current.resolve(strict=True)
    if generation.parent != cache_dir.resolve() or not generation.name.startswith(_GENERATION_PREFIX):
        raise ValueError(f"uv cache link {current} points outside the managed generation directory")
    if not generation.is_dir():
        raise ValueError(f"uv cache generation is not a directory: {generation}")
    return generation


def ensure_uv_cache_layout(cache_dir: Path) -> Path:
    """Return the current generation, migrating the legacy directory if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    current = cache_dir / cache_host_dirname(UV_CACHE_PATH)
    if current.is_symlink():
        return _resolved_generation(cache_dir, current)

    generation = _generation_path(cache_dir)
    if current.exists():
        if not current.is_dir():
            raise ValueError(f"uv cache path is not a directory: {current}")
        current.rename(generation)
    else:
        generation.mkdir()
    _replace_current_link(current, generation)
    return generation


def current_uv_cache_generation(cache_dir: Path) -> Path:
    """Return the concrete directory new tasks should bind mount."""
    return ensure_uv_cache_layout(cache_dir)


def cache_mount_path(cache_dir: Path, container_path: str) -> Path:
    """Resolve a cache mount, pinning uv mounts to their current generation."""
    if container_path == UV_CACHE_PATH:
        return current_uv_cache_generation(cache_dir)
    host_dir = cache_dir / cache_host_dirname(container_path)
    host_dir.mkdir(parents=True, exist_ok=True)
    return host_dir


def is_uv_cache_namespace(path: Path) -> bool:
    """Identify namespaces that must bypass generic age-based reclamation."""
    return path.name == cache_host_dirname(UV_CACHE_PATH) or path.name.startswith(_GENERATION_PREFIX)


def _quarantine_metadata(generation: Path) -> Path:
    return generation / _QUARANTINE_METADATA


def _write_quarantine(path: Path, metadata: _QuarantineMetadata) -> None:
    _write_json(path, asdict(metadata))


def _read_quarantine(path: Path) -> _QuarantineMetadata:
    value = json.loads(path.read_text())
    value["consumers"] = tuple(value["consumers"])
    return _QuarantineMetadata(**value)


def _last_rotation_ms(cache_dir: Path) -> int | None:
    path = cache_dir / _LAST_ROTATION
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["rotated_at_ms"])


def _rotation_is_rate_limited(cache_dir: Path, now: Timestamp) -> bool:
    last_rotation_ms = _last_rotation_ms(cache_dir)
    if last_rotation_ms is None:
        return False
    return now.epoch_ms() - last_rotation_ms < UV_CACHE_ROTATION_MIN_INTERVAL.to_ms()


def _rotate_uv_cache(
    cache_dir: Path,
    active_consumers: Callable[[], set[str]],
    now: Timestamp,
) -> Path:
    current = cache_dir / cache_host_dirname(UV_CACHE_PATH)
    old_generation = _resolved_generation(cache_dir, current)
    consumers_before = active_consumers()
    metadata_path = _quarantine_metadata(old_generation)
    _write_quarantine(
        metadata_path,
        _QuarantineMetadata(
            consumers=tuple(sorted(consumers_before)),
            rotation_complete=False,
            rotated_at_ms=now.epoch_ms(),
            unused_since_ms=None,
        ),
    )

    new_generation = _generation_path(cache_dir)
    new_generation.mkdir()
    _replace_current_link(current, new_generation)

    consumers_after = active_consumers()
    _write_quarantine(
        metadata_path,
        _QuarantineMetadata(
            consumers=tuple(sorted(consumers_before | consumers_after)),
            rotation_complete=True,
            rotated_at_ms=now.epoch_ms(),
            unused_since_ms=None,
        ),
    )
    _write_json(cache_dir / _LAST_ROTATION, {"rotated_at_ms": now.epoch_ms()})
    logger.warning("quarantined suspect uv cache %s; new tasks use %s", old_generation, new_generation)
    _ROTATIONS.add(1)
    return old_generation


def _reclaim_unused_generations(cache_dir: Path, active_consumers: set[str], now: Timestamp) -> tuple[Path, ...]:
    current = current_uv_cache_generation(cache_dir)
    reclaimed: list[Path] = []
    for generation in cache_dir.iterdir():
        if not generation.is_dir() or not generation.name.startswith(_GENERATION_PREFIX) or generation == current:
            continue
        metadata_path = _quarantine_metadata(generation)
        if not metadata_path.exists():
            continue
        metadata = _read_quarantine(metadata_path)
        if not metadata.rotation_complete:
            continue

        consumers = set(metadata.consumers)
        if consumers & active_consumers:
            if metadata.unused_since_ms is not None:
                _write_quarantine(metadata_path, replace(metadata, unused_since_ms=None))
            continue

        unused_since_ms = metadata.unused_since_ms
        if unused_since_ms is None:
            _write_quarantine(metadata_path, replace(metadata, unused_since_ms=now.epoch_ms()))
            continue
        if now.epoch_ms() - int(unused_since_ms) < UV_CACHE_RECLAIM_GRACE.to_ms():
            continue

        shutil.rmtree(generation)
        reclaimed.append(generation)
        logger.info("reclaimed quarantined uv cache %s", generation)
        _RECLAIMS.add(1)
    return tuple(reclaimed)


def maintain_uv_cache(
    cache_dir: Path,
    active_consumers: Callable[[], set[str]],
    *,
    now: Timestamp | None = None,
) -> UvCacheMaintenanceResult:
    """Rotate a signaled cache and reclaim quarantines with no live consumers."""
    observed_at = now or Timestamp.now()
    current = ensure_uv_cache_layout(cache_dir)
    marker = current / UV_CACHE_REPAIR_MARKER
    rotated: Path | None = None
    rate_limited = False
    if marker.exists():
        rate_limited = _rotation_is_rate_limited(cache_dir, observed_at)
        if not rate_limited:
            rotated = _rotate_uv_cache(cache_dir, active_consumers, observed_at)

    quarantines_exist = any(_quarantine_metadata(path).exists() for path in cache_dir.iterdir() if path.is_dir())
    reclaimed = _reclaim_unused_generations(cache_dir, active_consumers(), observed_at) if quarantines_exist else ()
    return UvCacheMaintenanceResult(rotated=rotated, reclaimed=reclaimed, rotation_rate_limited=rate_limited)


def run_uv_cache_maintenance(
    cache_dir: Path,
    active_consumers: Callable[[], set[str]],
    stop: threading.Event,
) -> None:
    """Maintain the node-local uv cache until shutdown."""
    while not stop.is_set():
        try:
            maintain_uv_cache(cache_dir, active_consumers)
        except Exception:
            logger.exception("uv cache maintenance failed for %s", cache_dir)
            _FAILURES.add(1)
        stop.wait(UV_CACHE_MAINTENANCE_INTERVAL.to_seconds())
