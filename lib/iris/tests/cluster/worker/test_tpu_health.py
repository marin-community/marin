# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker-failure stderr pattern detection."""

import pytest
from iris.cluster.worker.tpu_health import TPU_INIT_FAILURE_PATTERNS, detect_tpu_init_failure


@pytest.mark.parametrize(
    ("stderr_lines", "expected_pattern"),
    [
        # TPU init-time bad-node signatures.
        (["Couldn't open iommu group /dev/vfio/0"], "Couldn't open iommu group"),
        (["open(/dev/vfio/0): Device or resource busy"], "open(/dev/vfio"),
        (["jax.errors.JaxRuntimeError: Failed to initialize TPU system"], "Failed to initialize TPU system"),
        (["RuntimeError: TPU initialization failed"], "TPU initialization failed"),
        (["No accelerator found on this host"], "No accelerator found"),
        # JAX-distributed-RPC unavailability (sibling self-exit, issue #5753).
        (
            ['UNKNOWN:Error received from peer {grpc_message:"Socket closed", grpc_status:14}'],
            'grpc_message:"Socket closed"',
        ),
        (["E0511 23:00:57 gRPC Socket closed grpc_status:14"], "gRPC Socket closed"),
        # JAX raw-signal-handler / SIGABRT from peer loss.
        (["E0511 23:00:57 1234 abort.cc:42] RAW: Raising signal 6 with default behavior"], "RAW: Raising signal"),
        (["Fatal Python error: Aborted"], "Fatal Python error: Aborted"),
    ],
)
def test_detect_returns_first_matching_pattern(stderr_lines, expected_pattern):
    assert detect_tpu_init_failure(stderr_lines) == expected_pattern


def test_detect_returns_none_for_unrelated_stderr():
    lines = [
        "Traceback (most recent call last):",
        'ValueError: bad user config: expected "foo"',
        "",
    ]
    assert detect_tpu_init_failure(lines) is None


def test_detect_skips_empty_lines():
    assert detect_tpu_init_failure(["", "RAW: Raising signal 11 with default behavior"]) == "RAW: Raising signal"


def test_detect_returns_first_match_when_multiple_lines_match():
    lines = [
        "unrelated chatter",
        "gRPC Socket closed grpc_status:14",
        "Fatal Python error: Aborted",
    ]
    # First matching line wins (iteration order over stderr_lines).
    assert detect_tpu_init_failure(lines) == "gRPC Socket closed"


def test_all_patterns_are_unique():
    assert len(set(TPU_INIT_FAILURE_PATTERNS)) == len(TPU_INIT_FAILURE_PATTERNS)
