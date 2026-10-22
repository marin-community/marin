# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for zephyr.memory_budget."""

import os
import subprocess
import sys

from zephyr.memory_budget import native_thread_env


def test_native_thread_env_sizes_pools_in_fresh_interpreter():
    probe = "import polars as pl, pyarrow as pa; print(pl.thread_pool_size(), pa.cpu_count())"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        env={**os.environ, **native_thread_env(1.5)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "2 2"
