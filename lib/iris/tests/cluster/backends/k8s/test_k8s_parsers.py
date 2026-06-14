# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8s resource quantity parsers."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from iris.cluster.backends.k8s.types import parse_k8s_cpu, parse_k8s_quantity, parse_k8s_timestamp


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
        ("2024-01-01T00:00:00Z", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        ("2024-01-01T00:00:00+00:00", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        (
            "2024-01-01T00:00:00.123456Z",
            datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
        ),
        # The Kubernetes API emits nanosecond precision; fromisoformat truncates
        # to microseconds (the case the removed manual-truncation block handled).
        (
            "2024-01-01T00:00:00.123456789Z",
            datetime(2024, 1, 1, 0, 0, 0, 123456, tzinfo=timezone.utc),
        ),
    ],
)
def test_parse_k8s_timestamp(value: str, expected: datetime):
    assert parse_k8s_timestamp(value) == expected


def test_parse_k8s_timestamp_rejects_malformed():
    with pytest.raises(ValueError):
        parse_k8s_timestamp("not-a-timestamp")
