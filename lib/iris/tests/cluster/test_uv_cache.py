# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.runtime.env import UV_CACHE_REPAIR_MARKER
from iris.cluster.uv_cache import (
    UV_CACHE_RECLAIM_GRACE,
    UV_CACHE_ROTATION_MIN_INTERVAL,
    current_uv_cache_generation,
    maintain_uv_cache,
)
from rigging.timing import Timestamp


def test_legacy_uv_cache_becomes_a_pinned_generation(tmp_path):
    cache_dir = tmp_path / "cache"
    legacy = cache_dir / "uv-cache"
    legacy.mkdir(parents=True)
    (legacy / "artifact").write_text("cached")

    generation = current_uv_cache_generation(cache_dir)

    assert (cache_dir / "uv-cache").is_symlink()
    assert (generation / "artifact").read_text() == "cached"


def test_repair_rotates_new_tasks_without_reclaiming_live_generation(tmp_path):
    cache_dir = tmp_path / "cache"
    original = current_uv_cache_generation(cache_dir)
    (original / "artifact").write_text("cached")
    (original / UV_CACHE_REPAIR_MARKER).touch()

    result = maintain_uv_cache(
        cache_dir,
        lambda: {"running-attempt"},
        now=Timestamp.from_seconds(1_000),
    )

    replacement = current_uv_cache_generation(cache_dir)
    assert result.rotated == original
    assert replacement != original
    assert not (replacement / "artifact").exists()
    assert (original / "artifact").read_text() == "cached"
    assert original.is_dir()


def test_quarantine_is_reclaimed_after_consumers_exit_and_grace_elapses(tmp_path):
    cache_dir = tmp_path / "cache"
    original = current_uv_cache_generation(cache_dir)
    (original / UV_CACHE_REPAIR_MARKER).touch()
    maintain_uv_cache(cache_dir, lambda: {"attempt"}, now=Timestamp.from_seconds(1_000))

    maintain_uv_cache(cache_dir, set, now=Timestamp.from_seconds(1_100))
    assert original.is_dir()

    before_grace = Timestamp.from_seconds(1_100 + UV_CACHE_RECLAIM_GRACE.to_seconds() - 1)
    maintain_uv_cache(cache_dir, set, now=before_grace)
    assert original.is_dir()

    after_grace = Timestamp.from_seconds(1_100 + UV_CACHE_RECLAIM_GRACE.to_seconds())
    result = maintain_uv_cache(cache_dir, set, now=after_grace)
    assert result.reclaimed == (original,)
    assert not original.exists()


def test_repair_rate_limits_repeated_signals(tmp_path):
    cache_dir = tmp_path / "cache"
    original = current_uv_cache_generation(cache_dir)
    (original / UV_CACHE_REPAIR_MARKER).touch()
    maintain_uv_cache(cache_dir, set, now=Timestamp.from_seconds(1_000))
    replacement = current_uv_cache_generation(cache_dir)
    (replacement / UV_CACHE_REPAIR_MARKER).touch()

    result = maintain_uv_cache(cache_dir, set, now=Timestamp.from_seconds(1_001))

    assert result.rotation_rate_limited
    assert current_uv_cache_generation(cache_dir) == replacement

    retry_at = Timestamp.from_seconds(1_000 + UV_CACHE_ROTATION_MIN_INTERVAL.to_seconds())
    result = maintain_uv_cache(cache_dir, set, now=retry_at)
    assert result.rotated == replacement
