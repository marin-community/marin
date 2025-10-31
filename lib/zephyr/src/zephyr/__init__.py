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

"""ml-flow: Lightweight dataset library for distributed data processing."""

import logging
from contextvars import ContextVar
from dataclasses import replace

from zephyr.backend_factory import create_backend, flow_backend, set_flow_backend
from zephyr.backends import Backend, RayBackend, SyncBackend, ThreadPoolBackend
from zephyr.dataset import Dataset, load_file, load_jsonl, load_parquet, load_zip_members
from zephyr.worker_pool import WorkerPool, WorkerPoolConfig
from zephyr.writers import write_jsonl_file

logger = logging.getLogger(__name__)


__all__ = [
    "Backend",
    "Dataset",
    "RayBackend",
    "SyncBackend",
    "ThreadPoolBackend",
    "WorkerPool",
    "WorkerPoolConfig",
    "create_backend",
    "flow_backend",
    "load_file",
    "load_jsonl",
    "load_parquet",
    "load_zip_members",
    "set_flow_backend",
    "write_jsonl_file",
]
