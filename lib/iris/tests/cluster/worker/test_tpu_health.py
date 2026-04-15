# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for TPU bad-node stderr pattern detection."""

import pytest

from iris.cluster.worker.tpu_health import (
    TPU_INIT_FAILURE_PATTERNS,
    detect_tpu_init_failure,
)


@pytest.mark.parametrize(
    "line",
    [
        # Exact failure from #4783:
        "jax.errors.JaxRuntimeError: UNKNOWN: TPU initialization failed: "
        "open(/dev/vfio/0): Device or resource busy: Device or resource busy; "
        "Couldn't open iommu group /dev/vfio/0",
        # libtpu's older init-failure wording
        "Failed to initialize TPU system: some backend error",
        # JAX surface when the VM booted without a TPU attached
        "RuntimeError: No accelerator found on this host",
        # vfio path-only hit
        "libtpu: open(/dev/vfio/0) returned -1",
    ],
)
def test_detects_known_bad_node_signatures(line: str) -> None:
    assert detect_tpu_init_failure([line]) is not None


def test_detects_from_mixed_log_tail() -> None:
    tail = [
        "normal startup log line",
        "another info line",
        "Couldn't open iommu group /dev/vfio/0",
        "subsequent error traceback frame",
    ]
    pattern = detect_tpu_init_failure(tail)
    assert pattern == "Couldn't open iommu group"


def test_returns_none_on_unrelated_stderr() -> None:
    tail = [
        "Traceback (most recent call last):",
        'ValueError: bad user config: expected "foo"',
        "",
    ]
    assert detect_tpu_init_failure(tail) is None


def test_empty_input() -> None:
    assert detect_tpu_init_failure([]) is None


def test_ignores_empty_lines() -> None:
    # Empty strings should not be mistaken for matches and should not crash.
    assert detect_tpu_init_failure(["", None or ""]) is None


def test_all_patterns_are_discoverable() -> None:
    # Sanity: every declared pattern must be detected when it appears verbatim
    # in a line. Guards against accidental pattern-list / detector drift.
    for pattern in TPU_INIT_FAILURE_PATTERNS:
        line = f"prefix noise {pattern} trailing noise"
        assert detect_tpu_init_failure([line]) == pattern
