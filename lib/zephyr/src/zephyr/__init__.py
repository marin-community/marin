# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr: Lightweight dataset library for distributed data processing."""

import logging

from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import (
    CounterSnapshot,
    WorkerContext,
    ZephyrContext,
    ZephyrExecutionResult,
    zephyr_worker_ctx,
)
from zephyr.expr import Expr, col, lit
from zephyr.plan import compute_plan
from zephyr.readers import InputFileSpec, load_file, load_jsonl, load_parquet, load_vortex, load_zip_members
from zephyr.writers import atomic_rename, write_jsonl_file, write_levanter_cache, write_parquet_file, write_vortex_file

logger = logging.getLogger(__name__)


__all__ = [
    "Dataset",
    "Expr",
    "InputFileSpec",
    "ShardInfo",
    "WorkerContext",
    "ZephyrContext",
    "ZephyrExecutionResult",
    "atomic_rename",
    "col",
    "compute_plan",
    "counters",
    "lit",
    "load_file",
    "load_jsonl",
    "load_parquet",
    "load_vortex",
    "load_zip_members",
    "write_jsonl_file",
    "write_levanter_cache",
    "write_parquet_file",
    "write_vortex_file",
    "zephyr_worker_ctx",
]
