# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from collections.abc import Callable

import pytest
import pyarrow as pa
import hashlib
import dupekit


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
def test_hashing_throughput(benchmark: Any, sample_batch: pa.RecordBatch, func: Callable, mode: str) -> None:
    # Use the sample_batch fixture (10k rows) and convert to bytes
    text_samples = [t.as_py().encode("utf-8") for t in sample_batch["text"]]

    def _run() -> list[Any]:
        if mode == "batch":
            return func(text_samples)
        return [func(x) for x in text_samples]

    benchmark(_run)
