# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.snowball_isoflop.measurements import warm_measurement


def test_warm_rates_exclude_compile_include_overheads_and_keep_early_drops():
    rows = [
        {
            "global_step": step,
            "train/loss": 2.0,
            "throughput/duration": 1000 if step == 0 else 10,
            "throughput/loading_time": 2,
            "throughput/hook_time": 3,
            "throughput/gc_time": 1,
            "memory/stats_devices_measured": 8,
            "memory/min_headroom_fraction": 0.2 if step < 7 else 0.1,
            "train/moe/sender_dropped": 1 if step == 0 else 0,
            "train/moe/receiver_dropped": 0,
        }
        for step in range(8)
    ]
    measurement = warm_measurement(rows, size=512, batch_size=8)
    assert measurement.warm_steps == (3, 4, 5, 6, 7)
    assert measurement.learner_positions_per_second_per_gpu == pytest.approx(26214.4)
    assert measurement.inclusive_positions_per_second_per_gpu == pytest.approx(16384)
    assert measurement.minimum_memory_headroom_fraction == pytest.approx(0.1)
    assert measurement.sender_dropped == 1
    assert measurement.receiver_dropped == 0
    for row in rows:
        row.pop("memory/min_headroom_fraction")
    assert warm_measurement(rows, size=512, batch_size=8).minimum_memory_headroom_fraction is None
