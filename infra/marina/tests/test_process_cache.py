# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from marina.process_cache import ProcessCache


def test_concurrent_callers_share_one_resource_per_specification() -> None:
    cache = ProcessCache()
    ready = Barrier(4)
    initialized = []

    def resource() -> object:
        value = object()
        initialized.append(value)
        return value

    def request(_revision: int) -> object:
        ready.wait()
        return cache.get_or_create("example.models", ("model-v1", 4), resource)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(request, range(4)))
    assert len(initialized) == 1
    assert all(value is initialized[0] for value in results)
    assert cache.get_or_create("example.models", ("model-v2", 4), resource) is not results[0]
    assert cache.get_or_create("other.models", ("model-v1", 4), resource) is not results[0]


def test_failed_resource_initialization_can_be_retried() -> None:
    cache = ProcessCache()

    def unavailable() -> object:
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        cache.get_or_create("example.models", "model-v1", unavailable)
    recovered = object()
    assert cache.get_or_create("example.models", "model-v1", lambda: recovered) is recovered
