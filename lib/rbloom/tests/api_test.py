#!/usr/bin/env python3
import io
import os
from hashlib import sha256
from pickle import dumps

from rbloom import Bloom


def hash_func(obj):
    """Convert arbitrary objects to i128 integers for hashing."""
    h = sha256(dumps(obj)).digest()
    return int.from_bytes(h[:16], "big", signed=True)


def test_bloom(bloom: Bloom):
    assert not bloom
    assert bloom.approx_items == 0.0

    bloom.add(hash_func('foo'))
    assert bloom
    assert bloom.approx_items > 0.0

    bloom.add(hash_func('bar'))

    assert hash_func('foo') in bloom
    assert hash_func('bar') in bloom
    assert hash_func('baz') not in bloom

    bloom.update([hash_func('baz'), hash_func('qux')])
    assert hash_func('baz') in bloom
    assert hash_func('qux') in bloom

    other = bloom.copy()
    assert other == bloom
    assert other is not bloom

    other.clear()
    assert not other
    assert other.approx_items == 0.0

    other.update([hash_func('foo'), hash_func('bar'), hash_func('baz'), hash_func('qux')])
    assert other == bloom

    other.update(hash_func(str(i).encode()*500) for i in range(100000))
    for i in range(100000):
        assert hash_func(str(i).encode()*500) in other
    assert bloom != other
    assert bloom & other == bloom
    assert bloom | other == other

    bloom &= other
    assert bloom < other

    orig = bloom.copy()
    bloom |= other
    assert bloom == other
    assert bloom > orig
    assert bloom >= orig
    assert bloom.issuperset(other)
    assert orig <= bloom
    assert orig.issubset(bloom)
    assert bloom >= bloom
    assert bloom.issuperset(bloom)
    assert bloom <= bloom
    assert bloom.issubset(bloom)

    bloom = orig.copy()
    bloom.update(other)
    assert bloom == other
    assert bloom > orig

    bloom = orig.copy()
    assert other == bloom.union(other)
    assert bloom == bloom.intersection(other)

    bloom.intersection_update(other)
    assert bloom == orig

    # TEST FILE PERSISTENCE
    i = 0
    while os.path.exists(f'test{i}.bloom'):
        i += 1
    filename = f'test{i}.bloom'

    try:
        # save and load from file path
        bloom.save(filename)
        bloom2 = Bloom.load(filename)
        assert bloom == bloom2
    finally:
        # remove the file
        os.remove(filename)

    # TEST FILE OBJECT PERSISTENCE
    buffer = io.BytesIO()
    bloom.save(buffer)
    buffer.seek(0)
    bloom3 = Bloom.load(buffer)
    assert bloom == bloom3

    # TEST bytes PERSISTENCE
    bloom_bytes = bloom.save_bytes()
    assert type(bloom_bytes) == bytes
    bloom4 = Bloom.load_bytes(bloom_bytes)
    assert bloom == bloom4


def test_chunked_write():
    """Test chunked writing with large bloom filters and custom file objects."""

    # Create a large bloom filter to ensure chunking happens
    # At 1% FPR, 10M items requires ~11.5MB, at 0.01% requires ~23MB, etc.
    large_bloom = Bloom(10_000_000, 0.01)

    # Add some test data
    for i in range(1000):
        large_bloom.add(hash_func(f"test_item_{i}"))

    # Test 1: Verify chunked write to BytesIO works correctly
    buffer = io.BytesIO()
    large_bloom.save(buffer)
    buffer.seek(0)
    loaded = Bloom.load(buffer)
    assert large_bloom == loaded

    # Verify all test items are present
    for i in range(1000):
        assert hash_func(f"test_item_{i}") in loaded

    # Test 2: Custom file object that tracks write calls
    class ChunkTracker(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.write_calls = []

        def write(self, data):
            self.write_calls.append(len(data))
            return super().write(data)

    tracker = ChunkTracker()
    large_bloom.save(tracker)

    # Verify multiple writes occurred (k value + at least one chunk)
    assert len(tracker.write_calls) >= 2, f"Expected multiple writes, got {len(tracker.write_calls)}"

    # First write should be k value (8 bytes)
    assert tracker.write_calls[0] == 8, f"First write should be 8 bytes (k value), got {tracker.write_calls[0]}"

    # Verify data integrity
    tracker.seek(0)
    reloaded = Bloom.load(tracker)
    assert large_bloom == reloaded

    # Test 3: Very large filter that definitely requires chunking (>32MB of data)
    huge_bloom = Bloom(100_000_000, 0.01)  # ~115MB filter
    for i in range(100):
        huge_bloom.add(hash_func(f"huge_item_{i}"))

    huge_tracker = ChunkTracker()
    huge_bloom.save(huge_tracker)

    # With 32MB chunks, a ~115MB filter should have 4+ writes (1 for k + 3+ chunks)
    assert len(huge_tracker.write_calls) >= 4, \
        f"Expected 4+ writes for large filter, got {len(huge_tracker.write_calls)}"

    # Verify each chunk after k value is <= 32MB
    CHUNK_SIZE = 32 * 1024 * 1024
    for i, size in enumerate(huge_tracker.write_calls[1:]):  # Skip k value
        assert size <= CHUNK_SIZE, \
            f"Chunk {i} is {size} bytes, exceeds 32MB limit"

    # Verify data integrity for huge filter
    huge_tracker.seek(0)
    huge_reloaded = Bloom.load(huge_tracker)
    assert huge_bloom == huge_reloaded
    for i in range(100):
        assert hash_func(f"huge_item_{i}") in huge_reloaded

    print("✓ Chunked write tests passed")


def operations_with_self():
    bloom = Bloom(1000, 0.1)
    bloom.add(hash_func('foo'))
    assert hash_func('foo') in bloom
    bloom |= bloom
    bloom &= bloom
    bloom.update(bloom)
    bloom.update(bloom, bloom)
    bloom.update(bloom, [hash_func('bob')], bloom)
    assert hash_func('foo') in bloom
    assert hash_func('bob') in bloom

    bloom.intersection_update(bloom)
    bloom.intersection_update(bloom, bloom)
    bloom.intersection_update(bloom, [hash_func('foo')], bloom)
    assert hash_func('foo') in bloom
    assert hash_func('bob') not in bloom
    assert bloom == bloom
    assert bloom <= bloom
    assert bloom >= bloom
    assert not (bloom > bloom)
    assert not (bloom < bloom)
    assert bloom.issubset(bloom)
    assert bloom.issuperset(bloom)
    assert bloom.union(bloom) == bloom
    assert bloom.union(bloom, bloom) == bloom
    assert bloom.intersection(bloom) == bloom
    assert bloom.intersection(bloom, bloom) == bloom


def api_suite():
    assert repr(Bloom(27_000, 0.0317)) == "<Bloom size_in_bits=193960 approx_items=0.0>"

    test_bloom(Bloom(13242, 0.0000001))
    test_bloom(Bloom(9874124, 0.01))
    test_bloom(Bloom(2837, 0.5))

    operations_with_self()
    test_chunked_write()

    print('All API tests passed')


if __name__ == '__main__':
    api_suite()
