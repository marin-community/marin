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
import logging

import pyarrow as pa
from typing import Any
import dupekit
from dupekit import Transformation

try:
    # Use the internal _minhash_lsh function to benchmark the Datasketch logic directly
    # without the Zephyr Dataset API wrapper overhead or type checks.
    from marin.processing.classification.deduplication.minhash_lsh import _minhash_lsh
except ImportError:
    # These benchmark tests are parsed by no run by the build
    logging.warn("Couldn't import marin")

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


def python_minhash_pipeline_wrapper(batch: pa.RecordBatch) -> int:
    records = batch.to_pylist()
    count = 0
    for _ in _minhash_lsh(records, vector_length=286, num_bands=26, shingle_size=5):
        count += 1
    return count


def test_bench_rust_minhash(benchmark: Any, sample_batch: pa.RecordBatch) -> None:
    batch = sample_batch.slice(length=BENCHMARK_ROWS)
    benchmark(rust_minhash_pipeline, batch)


def test_bench_python_minhash(benchmark: Any, sample_batch: pa.RecordBatch) -> None:
    batch = sample_batch.slice(length=BENCHMARK_ROWS)
    benchmark(python_minhash_pipeline_wrapper, batch)
