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

from typing import Any
from collections.abc import Callable

import pytest
import pyarrow.parquet as pq
import hashlib
import dupekit


@pytest.fixture(scope="module")
def text_samples(parquet_file: str) -> list[bytes]:
    table = pq.read_table(parquet_file)
    texts = table["text"][:10000].to_pylist()
    return [t.encode("utf-8") for t in texts]


def _py_blake2b(text: bytes) -> bytes:
    return hashlib.blake2b(text).digest()


@pytest.mark.parametrize(
    "func, mode",
    [
        pytest.param(_py_blake2b, "scalar", id="python_blake2b"),
        pytest.param(dupekit.hash_blake2, "scalar", id="rust_blake2"),
        pytest.param(dupekit.hash_blake3, "scalar", id="rust_blake3"),
        pytest.param(dupekit.hash_xxh3_128, "scalar", id="rust_xxh3_128"),
        pytest.param(dupekit.hash_xxh3_64, "scalar", id="rust_xxh3_64_scalar"),
        pytest.param(dupekit.hash_xxh3_64_batch, "batch", id="rust_xxh3_64_batch"),
    ],
)
def test_hashing_throughput(benchmark: Any, text_samples: list[bytes], func: Callable, mode: str) -> None:

    def _run() -> list[Any]:
        if mode == "batch":
            return func(text_samples)
        return [func(x) for x in text_samples]

    benchmark(_run)
