# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dupekit import hash_blake2, hash_blake3, hash_xxh3_64, hash_xxh3_128, hash_xxh3_64_batch


def test_blake2_compliance():
    inputs = [
        b"",
        b"hello",
        b"The quick brown fox jumps over the lazy dog",
        b"\x00\xff" * 100,
    ]

    for data in inputs:
        rust_result_list = hash_blake2(data)
        rust_result_bytes = bytes(rust_result_list)
        assert len(rust_result_bytes) == 64

        # Parity with python
        py_result_bytes = hashlib.blake2b(data).digest()
        assert rust_result_bytes == py_result_bytes


def test_xxh3_64_batch():
    # Batch vs. rowwise xxh3_64 parity
    inputs = [b"one", b"two", b"three", b"four"]
    expected = [hash_xxh3_64(x) for x in inputs]
    actual = hash_xxh3_64_batch(inputs)
    assert actual == expected


def test_blake3_vector():
    # Catch un-intentional regressions
    assert bytes(hash_blake3(b"hello")).hex() == "ea8f163db38682925e4491c5e58d4bb3506ef8c14eb78a86e908c5624a67200f"


def test_xxh3_64_vector():
    # Catch un-intentional regressions
    assert hash_xxh3_64(b"hello") == 10760762337991515389


def test_xxh3_128_vector():
    # Catch un-intentional regressions
    assert hash_xxh3_128(b"hello") == 241804000618833338782870102822322583576
