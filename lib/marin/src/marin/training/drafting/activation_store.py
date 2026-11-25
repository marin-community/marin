# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store the per-row safetensors shared by prefill and Speculators."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from time import sleep, time
from uuid import uuid4

from rigging.filesystem.conditional_object import VersionedBytes, conditional_object
from rigging.filesystem.distributed_lock import create_lock, lease_refresh
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration, ExponentialBackoff

OBJECT_STORE_BACKEND = "object-store"


@dataclass(frozen=True)
class _ActivationGeneration:
    generation: str
    started_at: float


def activation_path(root: StoragePath, row_index: int) -> StoragePath:
    """Return the activation object for a row in the prepared Arrow dataset."""
    return root / f"hs_{row_index}.safetensors"


def get_or_build_activation_cache(
    root: StoragePath, *, reusable_for: Duration, populate: Callable[[StoragePath], None]
) -> StoragePath:
    """Reuse a complete generation for `reusable_for` from the start of its build."""
    marker = conditional_object(str(root / "complete.json"))

    def reusable_generation(entry: VersionedBytes | None) -> StoragePath | None:
        if entry is None:
            return None
        record = _ActivationGeneration(**json.loads(entry.data))
        if time() - record.started_at < reusable_for.to_seconds():
            return root / record.generation
        return None

    if path := reusable_generation(marker.read()):
        return path

    root.mkdirs()
    lease = create_lock(str(root / "populate.lock"))
    backoff = ExponentialBackoff(initial=1.0, maximum=30.0)
    while not lease.try_acquire():
        if path := reusable_generation(marker.read()):
            return path
        sleep(backoff.next_interval())

    try:
        with lease_refresh(lease):
            previous = marker.read()
            if path := reusable_generation(previous):
                return path
            # Start the reuse window before any row can age in object storage.
            record = _ActivationGeneration(uuid4().hex, time())
            path = root / record.generation
            path.mkdirs()
            populate(path)
        lease.refresh()
        marker.write(json.dumps(asdict(record)).encode(), expected_version=previous.version if previous else None)
        return path
    finally:
        lease.release()
