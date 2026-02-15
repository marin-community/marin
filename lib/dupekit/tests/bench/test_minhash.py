# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
from typing import Any
import dupekit
from dupekit import Transformation

# Python is slow, can't use too many rows
BENCHMARK_ROWS = 1000


def rust_minhash_pipeline(batch: pa.RecordBatch) -> int:
    pipeline = [
        Transformation.CleanText(input_col="text", output_col="clean"),
        Transformation.MinHash(input_col="clean", output_col="sig", num_perms=286, ngram_size=5, seed=42),
        Transformation.MinHashLSH(input_col="sig", output_col="buckets", num_bands=26),
    ]
    res = dupekit.transform(batch, pipeline)
    return len(res)


def test_bench_rust_minhash(benchmark: Any, sample_batch: pa.RecordBatch) -> None:
    batch = sample_batch.slice(length=BENCHMARK_ROWS)
    benchmark(rust_minhash_pipeline, batch)
