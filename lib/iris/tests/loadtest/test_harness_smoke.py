# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test for the autoscaler load-test harness.

Marked ``@pytest.mark.loadtest`` so it is excluded from the default CI run.
See `pyproject.toml` ``addopts`` and `lib/iris/tests/loadtest/conftest.py` for
the marker wiring.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from iris.loadtest.harness import DEFAULT_EVALUATION_INTERVAL, HarnessConfig, LoadtestHarness

RUN_SECONDS = 5.0


@pytest.mark.loadtest
@pytest.mark.timeout(120)
def test_harness_boots_against_snapshot_and_shuts_down_cleanly(
    snapshot_copy_dir: Path,
    snapshot_path: Path,
) -> None:
    """The harness boots, ticks the autoscaler for a few seconds, and shuts down.

    Also asserts the *original* snapshot file is untouched — the harness must
    only mutate the working copy.
    """
    original_mtime_before = snapshot_path.stat().st_mtime
    original_size_before = snapshot_path.stat().st_size

    config = HarnessConfig(evaluation_interval=DEFAULT_EVALUATION_INTERVAL)

    with LoadtestHarness(snapshot_copy_dir, config=config) as h:
        time.sleep(RUN_SECONDS)
        metrics = h.metrics()

    # The smoke run issues no scale-ups; the thread container should be empty.
    assert metrics.active_scale_up_threads == 0, f"expected 0 scale-up threads, got {metrics.active_scale_up_threads}"

    # Original snapshot file must not have been touched.
    original_mtime_after = snapshot_path.stat().st_mtime
    original_size_after = snapshot_path.stat().st_size
    assert original_mtime_after == original_mtime_before
    assert original_size_after == original_size_before
