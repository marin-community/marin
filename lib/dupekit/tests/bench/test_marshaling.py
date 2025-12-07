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
    return pa.Table.from_batches([first_batch])


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(1_000_000_000, id="giant"),
        pytest.param(1024, id="small"),
        pytest.param(1, id="tiny"),
    ],
)
def test_arrow_marshaling(benchmark: Any, in_memory_table: pa.Table, batch_size: int | None) -> None:
    """
    Benchmarks Python Memory -> Rust -> Arrow Batches with different chunk sizes.
    batch_size=None simulates the "Giant" case (one massive batch).
    """

    def _pipeline() -> int:
        return sum(len(dupekit.process_arrow_batch(b)) for b in in_memory_table.to_batches(max_chunksize=batch_size))

    assert benchmark(_pipeline) > 0


@pytest.mark.parametrize(
    "batch_size",
    [
        pytest.param(None, id="batch"),
        pytest.param(1024, id="batched_stream"),
    ],
)
def test_dicts_vectorized(benchmark: Any, in_memory_table: pa.Table, batch_size: int | None) -> None:
    """
    Benchmarks Python Memory -> List[dict] -> Rust -> List[dict] (Vectorized).
    Rust iterates the list internally.
    """

    def _pipeline() -> int:
        if batch_size is None:
            batches = [in_memory_table.to_pylist()]
        else:
            batches = (b.to_pylist() for b in in_memory_table.to_batches(max_chunksize=batch_size))
        return sum(len(dupekit.process_dicts_batch(batch)) for batch in batches)

    assert benchmark(_pipeline) > 0


def test_dicts_loop(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Benchmarks Python Memory -> List[dict] -> Python Loop calls Rust per item -> List[dict].
    """

    def _pipeline() -> int:
        return len([dupekit.process_dicts_loop(d) for d in in_memory_table.to_pylist()])

    assert benchmark(_pipeline) > 0


def test_rust_structs(benchmark: Any, in_memory_table: pa.Table) -> None:
    """
    Python Memory -> Converts to list of Rust 'Document' Classes -> Rust -> List of Rust Classes.
    """

    def _pipeline() -> int:
        docs = [dupekit.Document(row["id"], row["text"]) for row in in_memory_table.to_pylist()]
        return len(dupekit.process_rust_structs(docs))

    assert benchmark(_pipeline) > 0
