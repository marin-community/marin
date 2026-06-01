# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ObjectRef: a driver-local handle to a (future) task result.

Unlike Ray's distributed object store, a fakeray ``ObjectRef`` is backed by a
plain ``concurrent.futures.Future`` living in the driver process. This is
sufficient because the values smallpond passes between tasks are *path
descriptors* (``DataSet`` objects holding parquet paths), not bulk data — the
terabytes live on the shared filesystem and never traverse the ref.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field


@dataclass(eq=False)
class ObjectRef:
    """Handle to the result of a remote call.

    ``eq=False`` keeps identity-based hashing so refs are usable as dict keys
    and set members, matching Ray's ObjectRef semantics.
    """

    id: str
    future: Future = field(repr=False)

    def _done(self) -> bool:
        return self.future.done()

    def _result(self, timeout: float | None = None) -> object:
        # Re-raises the task's exception if the future failed.
        return self.future.result(timeout=timeout)

    def __repr__(self) -> str:
        state = "ready" if self.future.done() else "pending"
        return f"ObjectRef({self.id[:8]}, {state})"
