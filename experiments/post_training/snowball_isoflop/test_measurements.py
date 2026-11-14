# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.snowball_isoflop.measurements import training_history, warm_measurement


def test_history_reader_uses_mounted_credentials_without_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    (tmp_path / "WANDB_API_KEY").write_text("test-only-mounted-credential")
    rows = [{"global_step": 0, "train/loss": 2.0}]

    class Run:
        state = "finished"

        def scan_history(self, *, page_size, min_step, max_step):
            assert (page_size, min_step, max_step) == (100, 0, 20)
            return iter(rows)

    class Api:
        def __init__(self, *, timeout, api_key):
            assert timeout == 30
            if api_key != "test-only-mounted-credential":
                raise RuntimeError("W&B authentication requires the mounted credential")

        def run(self, path):
            assert path.endswith("-calibration-d512-assistant-seed0-n20")
            return Run()

    monkeypatch.setattr("experiments.post_training.snowball_isoflop.measurements.wandb.Api", Api)
    assert training_history("calibration-d512-assistant-seed0-n20", 20) == ("finished", rows)


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
