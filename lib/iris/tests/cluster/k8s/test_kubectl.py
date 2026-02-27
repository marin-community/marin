# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for kubectl K8s resource parsers and top_pod method."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from iris.cluster.k8s.kubectl import Kubectl, _parse_k8s_cpu, _parse_k8s_memory


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
    assert _parse_k8s_cpu(value) == expected


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
    assert _parse_k8s_memory(value) == expected


def _make_completed_process(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_top_pod_parses_container_output():
    """top_pod should parse the first line with >= 4 columns from `kubectl top pod --containers`."""
    kubectl = Kubectl(namespace="iris")
    fake_output = "iris-task-abc123  task  250m  512Mi\n"

    with patch.object(kubectl, "run", return_value=_make_completed_process(stdout=fake_output)):
        result = kubectl.top_pod("iris-task-abc123")

    assert result == (250, 512 * 1024**2)


def test_top_pod_returns_none_on_failure():
    """top_pod should return None when metrics-server is unavailable (non-zero exit)."""
    kubectl = Kubectl(namespace="iris")
    with patch.object(
        kubectl,
        "run",
        return_value=_make_completed_process(returncode=1, stderr="error: Metrics API not available"),
    ):
        assert kubectl.top_pod("iris-task-abc123") is None


def test_top_pod_returns_none_on_empty_output():
    """top_pod should return None when the command succeeds but produces no parseable lines."""
    kubectl = Kubectl(namespace="iris")
    with patch.object(kubectl, "run", return_value=_make_completed_process(stdout="")):
        assert kubectl.top_pod("iris-task-abc123") is None


def test_top_pod_multiline_takes_first():
    """When multiple containers are listed, top_pod returns the first parseable line."""
    kubectl = Kubectl(namespace="iris")
    fake_output = "my-pod  sidecar  100m  128Mi\nmy-pod  main  500m  1Gi\n"
    with patch.object(kubectl, "run", return_value=_make_completed_process(stdout=fake_output)):
        result = kubectl.top_pod("my-pod")

    assert result == (100, 128 * 1024**2)
