# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr context drivers for tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from fray.client import Client
from fray.types import ResourceConfig

from zephyr.execution import ZephyrContext


@contextmanager
def memory_store_context(client: Client, tmp_path: Path, *, max_workers: int = 2) -> Iterator[ZephyrContext]:
    context = ZephyrContext(
        client=client,
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="256m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="memory-store-test",
    )
    with context:
        yield context
