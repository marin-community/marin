# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for temporarily modifying environment variables."""

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def temporary_env_vars(env_vars: dict[str, str]) -> Iterator[None]:
    """Context manager to temporarily set environment variables.

    Args:
        env_vars: Dictionary of environment variable names to values to set.

    On exit, restores the original environment.
    """
    old_env = {k: os.environ.get(k) for k in env_vars}
    os.environ.update(env_vars)
    try:
        yield
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
