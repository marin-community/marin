# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stream-based log-level heuristic used by task log capture."""

import pytest
from finelog.rpc import logging_pb2
from iris.cluster.log_keys import classify_log_level


@pytest.mark.parametrize(
    "source,data,expected",
    [
        # Prefix-free lines default from the stream they came from.
        ("stdout", "sys.path: ['', '/app']", logging_pb2.LOG_LEVEL_INFO),
        ("stdout", "running user command", logging_pb2.LOG_LEVEL_INFO),
        ("stderr", "Traceback (most recent call last):", logging_pb2.LOG_LEVEL_ERROR),
        # iris injects failure lines under the synthetic "error" source.
        ("error", "Container was OOM killed by the kernel", logging_pb2.LOG_LEVEL_ERROR),
        # Build output and unrecognized streams stay UNKNOWN (visible in every filter).
        ("build", "Resolved 412 packages", logging_pb2.LOG_LEVEL_UNKNOWN),
    ],
)
def test_stream_default_level(source, data, expected):
    assert classify_log_level(source, data) == expected


@pytest.mark.parametrize(
    "source,data,expected",
    [
        # A glog prefix wins over the stream default, in both directions.
        ("stderr", "I20260102 12:34:56 worker starting up", logging_pb2.LOG_LEVEL_INFO),
        ("stdout", "E20260102 12:44:05 something blew up", logging_pb2.LOG_LEVEL_ERROR),
        ("stderr", "W20260102 12:44:05 deprecated flag", logging_pb2.LOG_LEVEL_WARNING),
    ],
)
def test_parsed_prefix_overrides_stream_default(source, data, expected):
    assert classify_log_level(source, data) == expected


@pytest.mark.parametrize(
    "data",
    [
        " 45%|####5     | 450/1000 [00:12<00:15,  3.21it/s]",
        "100%|##########| 1000/1000 [05:12<00:00,  3.21it/s]",
        # A description prefixes the bar, so the frame is not at the line start.
        "Loading shards:  45%|####5     | 450/1000 [00:12<00:15,  3.21it/s]",
        # An unstarted bar has no rate to report.
        "  0%|          | 0/1000 [00:00<?, ?it/s]",
        # tqdm without a known total renders no bar and no percentage.
        "450it [00:12,  3.21it/s]",
        # Below one iteration per second tqdm inverts the rate.
        " 45%|####5     | 450/1000 [00:12<00:15,  1.20s/it]",
        # A postfix trails the rate inside the brackets.
        " 45%|####5     | 450/1000 [00:12<00:15,  3.21it/s, loss=1.23]",
        # tqdm redraws with a bare carriage return, so one captured line can hold
        # several frames.
        "\r 10%|#         | 100/1000 [00:03<00:27,  3.21it/s]\r 20%|##        | 200/1000 [00:06<00:24,  3.21it/s]",
        # An hour-long run's elapsed clock grows a third field.
        " 45%|####5     | 450/1000 [1:00:12<1:13:20,  0.12it/s]",
    ],
)
def test_progress_bar_on_stderr_is_informational(data):
    assert classify_log_level("stderr", data) == logging_pb2.LOG_LEVEL_INFO


@pytest.mark.parametrize(
    "data",
    [
        "450it [00:12,  3.21it/s]",
        # A glog prefix reclassifies any other source, but not this one.
        "I20260102 12:34:56 task exited with status 1",
        "Task failed:\nI20260102 12:34:56 shutting down",
    ],
)
def test_injected_failures_ignore_their_text(data):
    # The "error" source is iris asserting the task failed; text never overrides it.
    assert classify_log_level("error", data) == logging_pb2.LOG_LEVEL_ERROR


@pytest.mark.parametrize(
    "data",
    [
        "RuntimeError: shard missing",
        "Traceback (most recent call last):",
        # A percentage alone is not a progress bar.
        "GPU utilization at 45% for the last 30s",
        # Nor is a bracketed clock with no rate after it.
        "worker heartbeat late [00:12]",
        # A bracketed clock followed by prose, not a rate.
        "checkpoint 2026-07-09 [12:34, restoring shards]",
    ],
)
def test_real_stderr_lines_stay_errors(data):
    assert classify_log_level("stderr", data) == logging_pb2.LOG_LEVEL_ERROR
