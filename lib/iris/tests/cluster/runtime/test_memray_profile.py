# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import threading
from contextlib import contextmanager

import pytest
from iris.cluster.runtime.profile import _run_memray_profile
from iris.rpc import job_pb2


@contextmanager
def _allocations_during_profile():
    stop = threading.Event()

    def allocate() -> None:
        allocations: list[bytearray] = []
        while not stop.is_set():
            allocations.append(bytearray(1024))
            if len(allocations) == 1_000:
                allocations.clear()

    thread = threading.Thread(target=allocate, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "profile_format",
    [
        job_pb2.MemoryProfile.FLAMEGRAPH,
        job_pb2.MemoryProfile.TABLE,
        job_pb2.MemoryProfile.STATS,
        job_pb2.MemoryProfile.RAW,
    ],
)
@pytest.mark.memray
def test_run_memray_profile_returns_nonempty_output(profile_format):
    config = job_pb2.MemoryProfile(format=profile_format, leaks=False)

    with _allocations_during_profile():
        result = _run_memray_profile(str(os.getpid()), duration_seconds=1, memory_config=config)

    assert result


@pytest.mark.memray
def test_run_memray_profile_stats_returns_valid_json():
    config = job_pb2.MemoryProfile(format=job_pb2.MemoryProfile.STATS, leaks=False)

    with _allocations_during_profile():
        result = _run_memray_profile(str(os.getpid()), duration_seconds=1, memory_config=config)

    assert "total_num_allocations" in json.loads(result)
