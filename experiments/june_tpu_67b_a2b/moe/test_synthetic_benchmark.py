# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging

from levanter.tracker.json_logger import JsonLoggerConfig, JsonLoggerTracker

from experiments.june_tpu_67b_a2b.moe.synthetic_benchmark import FIRST_TIMED_STEP, StepDurations


def test_step_durations_come_from_the_tracker_log():
    log = logging.getLogger(JsonLoggerConfig.logger_name)
    previous_level = log.level
    log.setLevel(logging.INFO)
    step_durations = StepDurations()
    log.addHandler(step_durations)
    try:
        tracker = JsonLoggerTracker(log)
        for step in range(5):
            tracker.log({"throughput/duration": 1.0 + step, "train/loss": 2.0}, step=step)
            tracker.log({"throughput/hook_time": 0.1}, step=step)
        tracker.log({"throughput/duration": 5.0}, step=4)
        tracker.log_summary({"throughput/flops_per_example": 1e9})
    finally:
        log.removeHandler(step_durations)
        log.setLevel(previous_level)
    assert step_durations.by_step == {step: 1.0 + step for step in range(FIRST_TIMED_STEP, 5)}
