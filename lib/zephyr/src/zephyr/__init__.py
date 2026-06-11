# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr: Lightweight dataset library for distributed data processing."""

import logging

from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import (
    ZephyrContext,
    ZephyrExecutionResult,
)
from zephyr.expr import Expr, col, lit
from zephyr.plan import compute_plan
from zephyr.readers import (
    DEFAULT_FILE_PATH_COLUMN,
    InputFileSpec,
    load_file,
    load_jsonl,
    load_parquet,
    load_vortex,
    load_zip_members,
)
from zephyr.worker_context import CounterSnapshot, WorkerContext, zephyr_worker_ctx
from zephyr.writers import write_jsonl_file, write_parquet_file, write_vortex_file

logger = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_FILE_PATH_COLUMN",
    "Dataset",
    "Expr",
    "InputFileSpec",
    "ShardInfo",
    "WorkerContext",
    "ZephyrContext",
    "ZephyrExecutionResult",
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
    "write_parquet_file",
    "write_vortex_file",
    "zephyr_worker_ctx",
]
