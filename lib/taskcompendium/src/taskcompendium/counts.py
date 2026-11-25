# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the rows or tasks in pinned source payloads, per format the pipeline accepts.

Each format uses its cheapest exact measurement: parquet footer metadata (no
data-page read), streamed non-empty lines for JSONL — gzipped or plain, exactly
the lines the JSONL reader yields — and the top-level element count of a JSON
record array. Every function reads through :class:`StoragePath`, so remote
payloads — a pinned ``hf://datasets/<repo>@<revision>/...`` path among them —
count identically to local ones. Formats without a fixed row count (procedural
generators) and dialects whose line count lies (CSV quoting, XML) report
``None`` rather than an approximation.
"""

import json
from collections.abc import Mapping

from pyarrow import parquet as pq
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import open_file

from taskcompendium.pipeline.inputs import SourceFormat


def count_parquet_rows(path: StoragePath) -> int:
    """The file's row count from footer metadata, without reading data pages."""
    with path.open("rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def count_jsonl_rows(path: StoragePath) -> int:
    """The file's non-empty line count, decompressing by extension like the JSONL reader."""
    count = 0
    with open_file(str(path), "rt") as stream:
        for line in stream:
            if line.strip():
                count += 1
    return count


def count_json_elements(path: StoragePath) -> int:
    """The element count of the file's top-level JSON record array."""
    with path.open("rt") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON record array in {path}")
    return len(payload)


def count_source_rows(path: StoragePath, source_format: SourceFormat) -> int | None:
    """The payload's row count for the formats with an exact cheap measurement, else ``None``."""
    if source_format is SourceFormat.PARQUET:
        return count_parquet_rows(path)
    if source_format is SourceFormat.JSONL:
        return count_jsonl_rows(path)
    if source_format is SourceFormat.JSON:
        return count_json_elements(path)
    return None


def count_source_files(files: Mapping[str, StoragePath], source_format: SourceFormat) -> dict[str, int | None]:
    """Per-file row counts for one declared source's files, keyed by relative path."""
    return {name: count_source_rows(path, source_format) for name, path in sorted(files.items())}
