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

import pytest
import pyarrow.parquet as pq
import hashlib
import dupekit


@pytest.fixture(scope="module")
def text_samples(parquet_file: str) -> list[bytes]:
    table = pq.read_table(parquet_file)
    texts = table["text"][:10000].to_pylist()
    return [t.encode("utf-8") for t in texts]


def test_hash_python_blake2b(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[bytes]:
        res = []
        for text in text_samples:
            h = hashlib.blake2b(text)
            res.append(h.digest())
        return res

    benchmark(_run)


def test_hash_rust_blake2(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[list[int]]:
        res = []
        for text in text_samples:
            res.append(dupekit.hash_blake2(text))
        return res

    benchmark(_run)


def test_hash_rust_blake3(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[list[int]]:
        res = []
        for text in text_samples:
            res.append(dupekit.hash_blake3(text))
        return res

    benchmark(_run)


def test_hash_rust_xxh3_128(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[int]:
        res = []
        for text in text_samples:
            res.append(dupekit.hash_xxh3_128(text))
        return res

    benchmark(_run)


def test_hash_rust_xxh3_64_scalar(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[int]:
        res = []
        for text in text_samples:
            res.append(dupekit.hash_xxh3_64(text))
        return res

    benchmark(_run)


def test_hash_rust_xxh3_64_batch(benchmark: Any, text_samples: list[bytes]) -> None:

    def _run() -> list[int]:
        return dupekit.hash_xxh3_64_batch(text_samples)

    benchmark(_run)
