# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import pyarrow as pa
from typing import Any
import dupekit


@pytest.mark.parametrize("batch_size", [1, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
def test_arrow_batch_sizes(benchmark: Any, in_memory_table: pa.Table, batch_size: int) -> None:
    """
    Benchmarks the effect of PyArrow batch size on marshaling throughput.
    """

    def _pipeline() -> int:
        return sum(len(dupekit.process_arrow_batch(b)) for b in in_memory_table.to_batches(max_chunksize=batch_size))

    assert benchmark(_pipeline) > 0
