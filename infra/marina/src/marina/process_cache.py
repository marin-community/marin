# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lazy resources shared across application revisions in one Marina worker."""

import threading
from collections.abc import Callable, Hashable
from typing import cast


class ProcessCache:
    """Retain resources until process exit, serializing their initialization.

    Callers own their namespace and must include every resource setting in the
    key. A namespace/key pair must always produce the same value type, including
    across package revisions. Factories must not reenter the cache.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[tuple[str, Hashable], object] = {}

    def get_or_create[T](self, namespace: str, key: Hashable, factory: Callable[[], T]) -> T:
        """Return the shared resource, retrying initialization after a failure."""
        with self._lock:
            identity = (namespace, key)
            if identity not in self._values:
                self._values[identity] = factory()
            return cast(T, self._values[identity])


process_cache = ProcessCache()
