# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8s resource quantity parsers."""

from __future__ import annotations

import pytest
from iris.cluster.providers.k8s.types import parse_k8s_cpu, parse_k8s_quantity


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
