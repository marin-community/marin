# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the stream-based log-level heuristic used by task log capture."""

import pytest
from finelog.rpc import logging_pb2
from iris.cluster.log_classify import classify_log_level


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
