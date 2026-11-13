# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest

import levanter.callbacks._metrics as metrics


def test_memory_headroom_uses_the_most_constrained_gpu_across_processes(monkeypatch):
    device = SimpleNamespace(
        memory_stats=lambda: {"bytes_in_use": 20 * 2**30, "peak_bytes_in_use": 30 * 2**30, "bytes_limit": 40 * 2**30}
    )
    recorded = []
    monkeypatch.setattr(metrics.jax, "local_devices", lambda: [device])
    monkeypatch.setattr(
        metrics.multihost_utils,
        "process_allgather",
        lambda local: np.stack([local, np.array([[20, 39, 40]], np.float32)]),
    )
    monkeypatch.setattr(metrics.levanter.tracker, "log", lambda values, *, step: recorded.append((values, step)))
    metrics.log_device_memory(SimpleNamespace(step=7))
    values, step = recorded[0]
    assert step == 7
    assert values["memory/min_headroom_fraction"] == pytest.approx(0.025)
    assert values["memory/gpu0/peak_gib"] == 30
    assert values["memory/gpu1/peak_gib"] == 39
