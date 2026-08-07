# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.cache import PersistentKvCache


def test_store_then_load_round_trips_bytes(tmp_path):
    cache = PersistentKvCache(directory=str(tmp_path))
    cache.store("k", b"object-code")
    assert cache.load("k") == b"object-code"


def test_load_of_an_absent_key_is_none(tmp_path):
    assert PersistentKvCache(directory=str(tmp_path)).load("missing") is None


def test_store_overwrites_the_previous_value(tmp_path):
    cache = PersistentKvCache(directory=str(tmp_path))
    cache.store("k", b"first")
    cache.store("k", b"second")
    assert cache.load("k") == b"second"


def test_distinct_keys_are_distinct_objects(tmp_path):
    cache = PersistentKvCache(directory=str(tmp_path), suffix=".o")
    cache.store("a", b"aaa")
    cache.store("b", b"bbb")

    assert cache.load("a") == b"aaa"
    assert cache.load("b") == b"bbb"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.o", "b.o"]


def test_store_leaves_no_staging_file_behind(tmp_path):
    """A completed store renames its staged temp into place rather than leaving it."""
    cache = PersistentKvCache(directory=str(tmp_path), suffix=".o")
    cache.store("k", b"payload")
    assert [p.name for p in tmp_path.iterdir()] == ["k.o"]
