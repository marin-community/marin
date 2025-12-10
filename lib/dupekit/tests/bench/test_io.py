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

"""
Benchmarks focusing on End-to-End I/O performance:
Disk -> Parquet Parsing -> Memory -> Rust -> Output Data.
"""

import pytest
import pyarrow.parquet as pq
from typing import Any
import dupekit


def test_rust_native(benchmark: Any, small_parquet_path: str) -> None:
    """
    Baseline: Rust reads file from disk, parses Parquet, transforms, returns RecordBatch.
    """

    def _run() -> int:
        return len(dupekit.process_native(small_parquet_path))

    assert benchmark(_run) > 0


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1_000_000_000, id="giant"),
        pytest.param(1024, id="small"),
    ],
)
def test_arrow_io_pipeline(benchmark: Any, small_parquet_path: str, batch_size: int) -> None:
    """
    Python End-to-End: Python reads file -> Stream of RecordBatches -> Rust (called per batch).
    Includes Parquet parsing overhead and Python loop overhead.
    """

    def _pipeline() -> int:
        batches = pq.ParquetFile(small_parquet_path).iter_batches(batch_size=batch_size)
        return sum(len(dupekit.process_arrow_batch(b)) for b in batches)

    assert benchmark(_pipeline) > 0


def test_dicts_loop_io(benchmark: Any, small_parquet_path: str) -> None:
    """
    Python End-to-End: Read File -> List[dict] -> Loop calling Rust per item -> List[dict].
    Slowest Python approach (Baseline for worst case).
    """

    def _pipeline() -> int:
        docs = pq.read_table(small_parquet_path).to_pylist()
        return len([dupekit.process_dicts_loop(doc) for doc in docs])

    assert benchmark(_pipeline) > 0
