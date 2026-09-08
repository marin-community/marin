# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.math_eval.host_memory import cgroup_memory, host_memory_monitor


def files(root, current=20, peak=40, limit=100):
    for name, value in {"current": current, "peak": peak, "max": limit}.items():
        (root / f"memory.{name}").write_text(str(value))
    (root / "memory.events").write_text("low 0\nhigh 0\nmax 2\noom 1\noom_kill 1\n")


def test_memory_monitor_preserves_before_after_and_error_without_masking_task_failure(tmp_path):
    files(tmp_path)
    with pytest.raises(LookupError, match="native failure"):
        with host_memory_monitor(expected_limit_bytes=100, root=tmp_path) as receipt:
            files(tmp_path, current=50, peak=70)
            raise LookupError("native failure")
    assert receipt["status"] == "error"
    assert receipt["before"]["current_bytes"] == 20
    assert receipt["after"]["kernel_peak_bytes"] == 70
    assert receipt["observed_current_peak_bytes"] == 50
    assert receipt["after"]["events"]["oom_kill"] == 1
    assert receipt["samples"] == 2


def test_cgroup_limit_mismatch_fails_before_native_work(tmp_path):
    files(tmp_path, limit=101)
    with pytest.raises(ValueError, match="limit"):
        cgroup_memory(tmp_path, 100)
