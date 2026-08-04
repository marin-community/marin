# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hashing and sort-key helpers for Zephyr shard routing."""

import msgspec
import xxhash


def deterministic_hash(obj: object) -> int:
    """Compute a deterministic hash for an object."""
    s = msgspec.msgpack.encode(obj, order="deterministic")
    return xxhash.xxh3_64_intdigest(s)
