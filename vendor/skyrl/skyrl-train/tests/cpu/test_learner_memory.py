# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from skyrl_train.learner_memory import LearnerCudaMetrics


def test_host_bank_lifetimes_survive_phase_failure(fake_cuda, monkeypatch, delivered_telemetry):
    # Native active is allocator ownership, including pending events, rather
    # than a Python tensor census. A backload can leave a reusable cached bank.
    stats = {
        "allocated_bytes.current": 128,
        "active_bytes.current": 64,
        "allocations.current": 2,
        "active_requests.current": 1,
        "num_host_alloc": 2,
        "num_host_free": 0,
    }
    monkeypatch.setattr(torch.cuda, "host_memory_stats", lambda: dict(stats))
    memory = LearnerCudaMetrics(enabled=True, rank=7)
    with pytest.raises(RuntimeError, match="offload failed"):
        with memory.span("offload_to_cpu", step=1):
            # A second live bank consumes the previous cached bank; native
            # allocated bytes stay constant while active bytes increase.
            stats.update({"active_bytes.current": 128, "active_requests.current": 2})
            raise RuntimeError("offload failed")

    rows = delivered_telemetry.select("cuda_memory_observation", rank="7", phase="offload_to_cpu", step="1")
    assert [(row["attributes"]["boundary"], row["attributes"]["outcome"]) for row in rows] == [
        ("enter", "started"),
        ("exit", "failure"),
    ]
    assert [
        (row["body"]["host_allocated_bytes"], row["body"]["host_active_bytes"], row["body"]["host_cached_bytes"])
        for row in rows
    ] == [(128, 64, 64), (128, 128, 0)]
    assert rows[-1]["body"]["host_active_requests"] == 2
    assert rows[-1]["body"]["host_alloc_count"] == 2
    assert rows[-1]["body"]["host_free_count"] == 0


def test_uninitialized_host_allocator_keeps_cuda_observations(fake_cuda, monkeypatch, delivered_telemetry):
    monkeypatch.setattr(torch.cuda, "host_memory_stats", lambda: {})
    memory = LearnerCudaMetrics(enabled=True, rank=3)
    with memory.span("backload_to_gpu", step=0):
        fake_cuda.use_memory(200, 300)

    rows = delivered_telemetry.select("cuda_memory_observation", rank="3", phase="backload_to_gpu")
    assert len(rows) == 2
    assert rows[-1]["body"]["peak_allocated_bytes"] == 200
    assert "host_active_bytes" not in rows[-1]["body"]
