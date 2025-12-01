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
Benchmarks focusing purely on Memory -> Rust FFI overhead.
Benchmarks include the cost of converting Arrow data to target format (Dicts, Structs) if applicable.
"""

import pytest
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Any
import dupekit


@pytest.fixture(scope="module")
def in_memory_table(tmp_path_factory: pytest.TempPathFactory, parquet_file: str) -> pa.Table:
    """
    Loads 250k rows into memory once.
    """
    pf = pq.ParquetFile(parquet_file)
    first_batch = next(pf.iter_batches(batch_size=250_000))
    table = pa.Table.from_batches([first_batch])
    return table


def test_arrow_giant(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> One Giant RecordBatch -> Rust -> One Giant RecordBatch.
    """

    def _pipeline() -> int:
        batch = in_memory_table.combine_chunks().to_batches()[0]
        res = dupekit.process_arrow_batch(batch)
        return len(res)

    result = benchmark(_pipeline)
    assert result > 0


def test_arrow_small(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Stream of Batches (Size 1024) -> Rust (called per batch) -> Arrow Batches.
    """

    def _pipeline() -> int:
        count = 0
        for b in in_memory_table.to_batches(max_chunksize=1024):
            res = dupekit.process_arrow_batch(b)
            count += len(res)
        return count

    result = benchmark(_pipeline)
    assert result > 0


def test_arrow_tiny(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Stream of Batches (Size 1) -> Rust (called per batch) -> Arrow Batches.
    """

    def _pipeline() -> int:
        count = 0
        for b in in_memory_table.to_batches(max_chunksize=1):
            res = dupekit.process_arrow_batch(b)
            count += len(res)
        return count

    result = benchmark(_pipeline)
    assert result > 0


def test_rust_structs(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Converts to list of Rust 'Document' Classes -> Rust -> List of Rust Classes.
    """

    def _pipeline() -> int:
        pylist = in_memory_table.to_pylist()
        docs = [dupekit.Document(row["id"], row["text"]) for row in pylist]
        res = dupekit.process_rust_structs(docs)
        return len(res)

    result = benchmark(_pipeline)
    assert result > 0


def test_dicts_batch(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Converts to List[dict] -> Rust -> List[dict].
    """

    def _pipeline() -> int:
        docs = in_memory_table.to_pylist()
        res = dupekit.process_dicts_batch(docs)
        return len(res)

    result = benchmark(_pipeline)
    assert result > 0


def test_dicts_loop(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> List[dict] -> Python Loop calls Rust per item -> List[dict].
    """

    def _pipeline() -> int:
        docs = in_memory_table.to_pylist()

        res_list = []
        for doc in docs:
            res_list.append(dupekit.process_dicts_loop(doc))
        return len(res_list)

    result = benchmark(_pipeline)
    assert result > 0


def test_dicts_batched_stream(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Stream of Batches (1024) -> List[dict] -> Rust -> List[dict].
    """

    def _pipeline() -> int:
        total = 0
        for batch in in_memory_table.to_batches(max_chunksize=1024):
            docs = batch.to_pylist()
            res = dupekit.process_dicts_batch(docs)
            total += len(res)
        return total

    result = benchmark(_pipeline)
    assert result > 0
