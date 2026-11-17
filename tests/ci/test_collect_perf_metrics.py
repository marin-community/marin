# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.ci import collect_perf_metrics


def _line(text: str) -> str:
    return f"I20261003 07:57:03 140387406554816 marin.execution.step_runner {text}"


def test_stage_wall_seconds_sums_steps_and_separates_cached_from_missing_stages():
    lines = [
        _line("Step datakit-smoke/normalize_01f7b77e succeeded in 0:09:00.554201"),
        _line("Step datakit/quality/a_7b16eabf succeeded in 1:00:00"),
        _line("Step datakit/quality/b_1d2c3b4a succeeded in 0:30:00.5"),
        _line("Step = datakit/minhash/a_e6985ee7\tParams = {}"),
        _line("Step failed: datakit/minhash/a_e6985ee7 (status=FAILED)"),
        _line("Skip datakit/dedup_4bfbdbb1: already succeeded"),
    ]

    durations, cached = collect_perf_metrics.compute_stage_wall_seconds(lines)

    assert durations["normalize"] == pytest.approx(540.554201)
    assert durations["quality"] == pytest.approx(5400.5)
    assert cached == ["dedup"]
    assert durations["dedup"] == 0.0
    assert "minhash" not in durations
