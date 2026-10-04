# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from scripts.ci import collect_perf_metrics


def _line(step: str, elapsed: str) -> str:
    return f"I20261003 07:57:03 140387406554816 marin.execution.step_runner Step {step} succeeded in {elapsed}"


def test_stage_wall_seconds_sums_steps_per_stage_and_flags_missing_stages():
    lines = [
        _line("datakit-smoke/normalize_01f7b77e", "0:09:00.554201"),
        _line("datakit/quality/a_7b16eabf", "1:00:00"),
        _line("datakit/quality/b_1d2c3b4a", "0:30:00.5"),
        "I20261003 07:43:22 1 marin.execution.step_runner Step = datakit/minhash/a_e6985ee7\tParams = {}",
        "I20261003 07:43:22 1 marin.execution.step_runner Skip datakit/dedup_4bfbdbb1: already succeeded",
    ]

    durations, cached = collect_perf_metrics.compute_stage_wall_seconds(lines)

    assert durations["normalize"] == pytest.approx(540.554201)
    assert durations["quality"] == pytest.approx(5400.5)
    assert "minhash" in cached and durations["minhash"] == 0.0
    assert "dedup" in cached
    assert "normalize" not in cached
