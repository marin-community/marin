# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Zephyr: Lightweight dataset library for distributed data processing."""

import logging

from zephyr.backends import Backend, BackendConfig
from zephyr.dataset import Dataset
from zephyr.expr import Expr, col, lit
from zephyr.plan import ExecutionHint, PhysicalPlan, PhysicalStage, compute_plan
from zephyr.readers import InputFileSpec, load_file, load_jsonl, load_parquet, load_vortex, load_zip_members
from zephyr.writers import atomic_rename, write_jsonl_file, write_levanter_cache, write_parquet_file, write_vortex_file

logger = logging.getLogger(__name__)


__all__ = [
    "Backend",
    "BackendConfig",
    "Dataset",
    "ExecutionHint",
    "Expr",
    "InputFileSpec",
    "PhysicalPlan",
    "PhysicalStage",
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
]
