# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream pinned Hugging Face rows without materializing a corpus."""

import json
from collections.abc import Iterator
from importlib import import_module
from itertools import islice
from typing import Any, Protocol, cast

from datasets import load_dataset
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.pipeline.models import GeneratedSource, HFSource, SnapshotSource


class GeneratorModule(Protocol):
    def generate_rows(self, limit: int) -> Iterator[dict[str, Any]]: ...


def source_rows(source: HFSource | GeneratedSource | SnapshotSource, limit: int | None) -> Iterator[dict[str, Any]]:
    """Stream pinned rows in source order; ``None`` reads a finite source to its end."""
    if isinstance(source, GeneratedSource):
        if limit is None:
            raise ValueError("Generated sources require an explicit finite row limit")
        yield from islice(cast(GeneratorModule, import_module(source.module)).generate_rows(limit), limit)
        return
    if isinstance(source, SnapshotSource):
        with StoragePath(source.path).open("rt") as stream:
            yield from islice((json.loads(line) for line in stream if line.strip()), limit)
        return
    if len(source.revision) != 40 or any(character not in "0123456789abcdef" for character in source.revision):
        raise ValueError("HF source revisions must be full commit hashes")
    dataset = load_dataset(source.dataset, source.config, revision=source.revision, split=source.split, streaming=True)
    yield from islice(dataset, limit)
