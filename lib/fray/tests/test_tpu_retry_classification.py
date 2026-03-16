# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from fray.v2.ray_backend.tpu import _is_retryable_tpu_startup_error


def test_retryable_tpu_startup_error_detects_nested_accelerator_failure():
    inner = RuntimeError("No accelerator found. Please run on a TPU or GPU.")
    outer = RuntimeError("worker failed")
    outer.__cause__ = inner

    assert _is_retryable_tpu_startup_error(outer)


def test_retryable_tpu_startup_error_ignores_user_code_failures():
    error = RuntimeError("shape mismatch while constructing batch")

    assert not _is_retryable_tpu_startup_error(error)
