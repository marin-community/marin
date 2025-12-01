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
import pyarrow as pa
from typing import Any
import dupekit


@pytest.fixture(scope="module")
def small_parquet_file(tmp_path_factory: pytest.TempPathFactory, parquet_file: str) -> str:
    """
    Creates a smaller slice (250k rows) of the main parquet file for faster benchmarking.
    """
    fn = tmp_path_factory.mktemp("data_io") / "subset.parquet"
    pf = pq.ParquetFile(parquet_file)
    first_batch = next(pf.iter_batches(batch_size=250_000))
    table = pa.Table.from_batches([first_batch])
    pq.write_table(table, fn)
    path_str = str(fn)

    # Warm up OS cache
    with open(path_str, "rb") as f:
        while f.read(1024**2):
            pass

    return path_str


def test_rust_native(benchmark: Any, small_parquet_file: str) -> None:
    """
    Baseline: Rust reads file from disk, parses Parquet, transforms, returns RecordBatch.
    """

    def _run() -> int:
        res = dupekit.process_native(small_parquet_file)
        return len(res)

    result = benchmark(_run)
    assert result > 0


def test_arrow_small(benchmark: Any, small_parquet_file: str) -> None:
    """
    Python End-to-End: Python reads file -> Stream of RecordBatches -> Rust (called per batch).
    Includes Parquet parsing overhead and Python loop overhead.
    """

    def _pipeline() -> int:
        pf = pq.ParquetFile(small_parquet_file)
        count = 0
        for b in pf.iter_batches(batch_size=1024):
            res = dupekit.process_arrow_batch(b)
            count += len(res)
        return count

    result = benchmark(_pipeline)
    assert result > 0


def test_arrow_giant(benchmark: Any, small_parquet_file: str) -> None:
    """
    Python End-to-End: Python reads file -> Stream of RecordBatches (Giant) -> Rust (on the whole data).
    Includes Parquet parsing overhead.
    """

    def _pipeline() -> int:
        pf = pq.ParquetFile(small_parquet_file)
        count = 0
        for b in pf.iter_batches(batch_size=1_000_000_000):
            res = dupekit.process_arrow_batch(b)
            count += len(res)
        return count

    result = benchmark(_pipeline)
    assert result > 0


def test_dicts_loop_io(benchmark: Any, small_parquet_file: str) -> None:
    """
    Python End-to-End: Read File -> List[dict] -> Loop calling Rust per item -> List[dict].
    Slowest Python approach (Baseline for worst case).
    """

    def _pipeline() -> int:
        table = pq.read_table(small_parquet_file)
        docs = table.to_pylist()

        res_list = []
        for doc in docs:
            res_list.append(dupekit.process_dicts_loop(doc))
        return len(res_list)

    result = benchmark(_pipeline)
    assert result > 0
