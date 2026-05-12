# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``CachedProto`` TypeDecorator on the SA Core ``schema``.

Covers bytes-keyed identity reuse, 8192-entry bound, 25% eviction
batches, thread-safe decode, and per-row None passthrough. The global
cache is class-wide and shared across every ``CachedProto`` instance,
so we exercise that explicitly.
"""

import threading

import pytest
from iris.cluster.controller.schema import CachedProto
from iris.rpc.job_pb2 import LoginRequest, WorkerTaskStatus


@pytest.fixture(autouse=True)
def _clear_global_cache():
    """Reset the process-wide cache around each test for isolation."""
    with CachedProto._global_lock:
        CachedProto._global_cache.clear()
    yield
    with CachedProto._global_lock:
        CachedProto._global_cache.clear()


def _make_login_blob(token: str) -> bytes:
    msg = LoginRequest()
    msg.identity_token = token
    return msg.SerializeToString()


def test_bind_param_serializes_proto():
    decoder = CachedProto(LoginRequest)
    msg = LoginRequest()
    msg.identity_token = "abc"
    assert decoder.process_bind_param(msg, None) == msg.SerializeToString()


def test_bind_param_none_passthrough():
    decoder = CachedProto(LoginRequest)
    assert decoder.process_bind_param(None, None) is None


def test_result_value_none_passthrough():
    decoder = CachedProto(LoginRequest)
    assert decoder.process_result_value(None, None) is None


def test_identical_bytes_share_python_identity():
    decoder = CachedProto(LoginRequest)
    blob = _make_login_blob("hello")
    first = decoder.process_result_value(blob, None)
    second = decoder.process_result_value(blob, None)
    assert first is second
    assert first.identity_token == "hello"


def test_distinct_bytes_produce_distinct_objects():
    decoder = CachedProto(LoginRequest)
    a = decoder.process_result_value(_make_login_blob("one"), None)
    b = decoder.process_result_value(_make_login_blob("two"), None)
    assert a is not b
    assert a.identity_token == "one"
    assert b.identity_token == "two"


def test_global_cache_shared_across_instances():
    """Today's ProtoCache is a singleton across all blob columns;
    CachedProto must preserve that — even for different message_cls."""
    decoder_a = CachedProto(LoginRequest)
    decoder_b = CachedProto(LoginRequest)
    blob = _make_login_blob("share")
    first = decoder_a.process_result_value(blob, None)
    second = decoder_b.process_result_value(blob, None)
    assert first is second


def test_global_cache_shared_across_message_cls():
    """The cache is keyed only by bytes; if two columns of different
    proto types happen to receive identical bytes, the cache returns
    the first decoded object. This matches today's behaviour where
    ProtoCache is a single dict per process."""
    decoder_login = CachedProto(LoginRequest)
    decoder_status = CachedProto(WorkerTaskStatus)
    # An empty proto serializes to b"" for both types.
    blob = b""
    first = decoder_login.process_result_value(blob, None)
    second = decoder_status.process_result_value(blob, None)
    # Bytes-keyed cache returns the same object even though the second
    # call asked for a different class — matches today's ProtoCache.
    assert first is second


def test_eviction_drops_oldest_quarter_when_full():
    """Filling the cache to _MAX_SIZE and inserting one more entry
    should drop _MAX_SIZE // 4 oldest entries in a single batch,
    leaving _MAX_SIZE - _MAX_SIZE // 4 + 1 entries."""
    decoder = CachedProto(LoginRequest)
    for i in range(CachedProto._MAX_SIZE):
        decoder.process_result_value(_make_login_blob(f"tok-{i}"), None)
    assert len(CachedProto._global_cache) == CachedProto._MAX_SIZE

    # Insert one more distinct entry; eviction fires before insertion.
    decoder.process_result_value(_make_login_blob("tok-overflow"), None)
    expected = CachedProto._MAX_SIZE - (CachedProto._MAX_SIZE // 4) + 1
    assert len(CachedProto._global_cache) == expected


def test_concurrent_decode_returns_same_identity():
    """Four threads decoding the same blob concurrently should all
    receive the same Python object — the second-check-under-lock
    preserves is-identity even on a cache-miss race."""
    decoder = CachedProto(LoginRequest)
    blob = _make_login_blob("contended")
    results: list = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        results.append(decoder.process_result_value(blob, None))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 4
    first = results[0]
    for other in results[1:]:
        assert other is first
