# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from experiments.grug.fast_track.host_stall import HostStallSampler


def _busy_host_work(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_slow_stretch_is_logged_with_the_running_function(caplog):
    with caplog.at_level(logging.WARNING), HostStallSampler(threshold=0.2) as sampler:
        sampler.arm()
        _busy_host_work(0.4)
        sampler.disarm(step=7)
    assert "host stall" in caplog.text and "after step 7" in caplog.text
    assert "_busy_host_work" in caplog.text


def test_fast_stretch_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING), HostStallSampler(threshold=0.2) as sampler:
        sampler.arm()
        _busy_host_work(0.05)
        sampler.disarm(step=3)
    assert "host stall" not in caplog.text
