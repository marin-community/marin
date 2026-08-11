# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hashing and sort-key helpers for Zephyr shard routing."""

import msgspec
import xxhash

_encoder = msgspec.msgpack.Encoder(order="deterministic")


def encode_key(obj: object) -> bytes:
    """Encode a routing key to its canonical msgpack bytes.

    The same encoding backs both shard routing and the binary sort-key column
    written by the shuffle, so a key always hashes to the shard whose chunks
    store it.
    """
    return _encoder.encode(obj)


def hash_encoded_key(encoded: bytes) -> int:
    """Hash key bytes produced by :func:`encode_key`.

    Callers that already hold the encoded key use this instead of
    :func:`deterministic_hash` to avoid encoding it a second time.
    """
    return xxhash.xxh3_64_intdigest(encoded)


def deterministic_hash(obj: object) -> int:
    """Compute a deterministic hash for an object."""
    return hash_encoded_key(encode_key(obj))
