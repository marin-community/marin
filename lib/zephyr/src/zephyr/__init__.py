# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr: Lightweight dataset library for distributed data processing."""

import logging

from zephyr.dataset import Dataset
from zephyr.execution import DiskChunk, WorkerContext, ZephyrContext, zephyr_worker_ctx
from zephyr.expr import Expr, col, lit
from zephyr.plan import ExecutionHint, compute_plan
from zephyr.readers import InputFileSpec, load_file, load_jsonl, load_parquet, load_vortex, load_zip_members
from zephyr.writers import atomic_rename, write_jsonl_file, write_levanter_cache, write_parquet_file, write_vortex_file

logger = logging.getLogger(__name__)


__all__ = [
    "Dataset",
    "ExecutionHint",
    "Expr",
    "InputFileSpec",
    "WorkerContext",
    "ZephyrContext",
    "atomic_rename",
    "col",
    "compute_plan",
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
