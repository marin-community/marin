# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8s resource quantity parsers."""

from datetime import UTC, datetime

import pytest
from iris.cluster.backends.k8s.tasks import _parse_pod_proc_status
from iris.cluster.platforms.k8s.types import parse_k8s_cpu, parse_k8s_quantity, parse_k8s_timestamp

# One capture of the in-pod /proc reader. comm carries a space to exercise the
# "index from the last ')'" parse; CLK_TCK=100, PAGE_SIZE=4096. Between the two
# stat samples utime+stime advances 15 ticks over a 0.5s uptime gap => 300 mcores.
# starttime 6000 ticks at 1000.5s uptime => 940.5s process uptime.
_POD_PROC_STATUS_SAMPLE = """@@hostname
task-pod-abc
@@uptime1
1000.50 900.00
@@stat1
1 (python 3.11) S 0 1 1 0 -1 4194560 12345 0 0 0 5000 2000 0 0 20 0 8 0 6000 0 0 0
@@uptime2
1001.00 900.40
@@stat2
1 (python 3.11) S 0 1 1 0 -1 4194560 12400 0 0 0 5010 2005 0 0 20 0 8 0 6000 0 0 0
@@statm
100000 25000 5000 1 0 90000 0
@@threads
Threads:\t8
@@fds
42
@@memtotal
MemTotal:       16000000 kB
@@nproc
8
@@clktck
100
@@pagesize
4096
"""


def test_parse_pod_proc_status():
    info = _parse_pod_proc_status(_POD_PROC_STATUS_SAMPLE)
    assert info.hostname == "task-pod-abc"
    assert info.pid == 1
    assert info.thread_count == 8
    assert info.open_fd_count == 42
    assert info.cpu_count == 8
    assert info.cpu_millicores == 300
    assert info.uptime_ms == 940500
    assert info.memory_rss_bytes == 25000 * 4096
    assert info.memory_vms_bytes == 100000 * 4096
    assert info.memory_total_bytes == 16000000 * 1024


def test_parse_pod_proc_status_tolerates_empty_capture():
    # A stripped-down container may lack some /proc entries; parsing must not raise
    # and unknown counters fall back to zero rather than a bogus value.
    info = _parse_pod_proc_status("@@hostname\npod-x\n@@clktck\n100\n@@pagesize\n4096\n")
    assert info.hostname == "pod-x"
    assert info.cpu_millicores == 0
    assert info.uptime_ms == 0
    assert info.thread_count == 0


@pytest.mark.parametrize(
    "value, expected",
    [
        ("250m", 250),
        ("1", 1000),
        ("0.5", 500),
        ("2500m", 2500),
        ("0", 0),
        ("4", 4000),
    ],
)
def test_parse_k8s_cpu(value: str, expected: int):
    assert parse_k8s_cpu(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("512Mi", 512 * 1024**2),
        ("1Gi", 1024**3),
        ("100Ki", 100 * 1024),
        ("1000", 1000),
        ("2G", 2 * 1000**3),
        ("1Ti", 1024**4),
        ("500M", 500 * 1000**2),
    ],
)
def test_parse_k8s_memory(value: str, expected: int):
    assert parse_k8s_quantity(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2024-01-01T00:00:00Z", datetime(2024, 1, 1, tzinfo=UTC)),
        ("2024-01-01T00:00:00+00:00", datetime(2024, 1, 1, tzinfo=UTC)),
        (
            "2024-01-01T00:00:00.123456Z",
            datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=UTC),
        ),
        # The Kubernetes API emits nanosecond precision; fromisoformat truncates
        # to microseconds (the case the removed manual-truncation block handled).
        (
            "2024-01-01T00:00:00.123456789Z",
            datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=UTC),
        ),
    ],
)
def test_parse_k8s_timestamp(value: str, expected: datetime):
    assert parse_k8s_timestamp(value) == expected


def test_parse_k8s_timestamp_rejects_malformed():
    with pytest.raises(ValueError):
        parse_k8s_timestamp("not-a-timestamp")
