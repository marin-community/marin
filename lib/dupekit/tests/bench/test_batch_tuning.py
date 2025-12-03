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

import pytest
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Any
import dupekit


@pytest.fixture(scope="module")
def in_memory_table(parquet_file: str) -> pa.Table:
    """
    Loads 100k rows into memory.
    """
    pf = pq.ParquetFile(parquet_file)
    first_batch = next(pf.iter_batches(batch_size=100_000))
    return pa.Table.from_batches([first_batch])


@pytest.mark.parametrize("batch_size", [1, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
def test_arrow_batch_sizes(benchmark: Any, in_memory_table: pa.Table, batch_size: int) -> None:
    """
    Benchmarks the effect of PyArrow batch size on marshaling throughput.
    """

    def _pipeline() -> int:
        return sum(len(dupekit.process_arrow_batch(b)) for b in in_memory_table.to_batches(max_chunksize=batch_size))

    assert benchmark(_pipeline) > 0
