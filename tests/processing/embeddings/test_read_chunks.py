# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.processing.embeddings.read_chunks import rows_per_chunk


def test_rows_per_chunk_shrinks_as_rows_grow():
    """A megabyte-document source must get far fewer rows per chunk than a small one."""
    target = 64 * 1024 * 1024
    small = rows_per_chunk(num_rows=10_000, nbytes=10_000 * 2_000, target_bytes=target)
    large = rows_per_chunk(num_rows=10_000, nbytes=10_000 * 5_000_000, target_bytes=target)
    assert large < small
    assert large >= 1, "never zero: a single oversized row must still be materializable"


def test_rows_per_chunk_stays_within_the_byte_target():
    target = 1_000_000
    avg = 250_000
    n = rows_per_chunk(num_rows=100, nbytes=100 * avg, target_bytes=target)
    assert n * avg <= target


def test_rows_per_chunk_handles_an_empty_batch():
    assert rows_per_chunk(num_rows=0, nbytes=0, target_bytes=1_000) == 1
