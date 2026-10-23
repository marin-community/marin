# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Immutable upstream row coordinates with explicitly recorded file ordering.

An offset in our source manifest is not automatically a Dataset Viewer offset.
Viewer coordinates require the viewer's converted-Parquet ordering and matching
revision. Original files and converted files are distinct coordinate systems.
"""

import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlencode

class OffsetNamespace(StrEnum):
    SOURCE_MANIFEST = "source_manifest"
    HF_VIEWER_PARQUET = "hf_viewer_parquet"


@dataclass(frozen=True)
class SourceRef:
    dataset: str
    config: str
    split: str
    revision: str


@dataclass(frozen=True)
class SourceFile:
    path: str
    num_rows: int


@dataclass(frozen=True)
class SourceManifest:
    source_ref: SourceRef
    files: tuple[SourceFile, ...]
    ordering_evidence: str
    offset_namespace: OffsetNamespace = OffsetNamespace.SOURCE_MANIFEST

    def __post_init__(self) -> None:
        if not self.ordering_evidence or not self.source_ref.revision:
            raise ValueError("Pin the source revision and document the file ordering")
        if not self.files or any(type(file.num_rows) is not int or file.num_rows < 0 for file in self.files):
            raise ValueError("Provide a nonempty ordered file manifest with nonnegative row counts")
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("Repeated file paths make physical row locators ambiguous")

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RowLocator:
    source_ref: SourceRef
    manifest_sha256: str
    row_offset: int
    file_path: str
    file_row_offset: int
    offset_namespace: OffsetNamespace


def locate_file_row(manifest: SourceManifest, file_path: str, file_row_offset: int) -> RowLocator:
    """Translate an original file coordinate using the manifest's explicit order."""
    if type(file_row_offset) is not int:
        raise TypeError("File row offset must be an integer")
    start = 0
    for file in manifest.files:
        if file.path == file_path:
            if not 0 <= file_row_offset < file.num_rows:
                raise IndexError(f"Row {file_row_offset} is outside {file_path}")
            return RowLocator(
                source_ref=manifest.source_ref,
                manifest_sha256=manifest.sha256,
                row_offset=start + file_row_offset,
                file_path=file_path,
                file_row_offset=file_row_offset,
                offset_namespace=manifest.offset_namespace,
            )
        start += file.num_rows
    raise KeyError(f"File absent from source manifest: {file_path}")


def locate_split_row(manifest: SourceManifest, row_offset: int) -> RowLocator:
    """Resolve a split offset to its exact file row, including empty file entries."""
    if type(row_offset) is not int:
        raise TypeError("Split row offset must be an integer")
    ends = []
    total = 0
    for file in manifest.files:
        total += file.num_rows
        ends.append(total)
    if not 0 <= row_offset < total:
        raise IndexError(f"Row {row_offset} is outside the split of {total} rows")
    index = bisect_right(ends, row_offset)
    start = ends[index - 1] if index else 0
    return locate_file_row(manifest, manifest.files[index].path, row_offset - start)


def validate_locator_mapping(manifest: SourceManifest, source_row: dict) -> RowLocator:
    """Validate public coordinates against a pinned ordered-file catalog entry."""
    if source_row["source_ref"] != manifest.sha256:
        raise ValueError("Source row reference differs from its manifest")
    if type(source_row["row_offset"]) is not int:
        raise TypeError("Split row offset must be an integer")
    locator = locate_file_row(manifest, source_row["file"], source_row["file_row_offset"])
    if locator.row_offset != source_row["row_offset"]:
        raise ValueError("Split row offset disagrees with the ordered file prefix and file row")
    return locator


def viewer_row_url(locator: RowLocator) -> str:
    """Build a current-viewer lookup only for proven converted-Parquet offsets."""
    if locator.offset_namespace != OffsetNamespace.HF_VIEWER_PARQUET:
        raise ValueError("Original-source offsets do not establish Dataset Viewer ordering")
    source = locator.source_ref
    query = urlencode(
        {
            "dataset": source.dataset,
            "config": source.config,
            "split": source.split,
            "offset": locator.row_offset,
            "length": 1,
        }
    )
    return f"https://datasets-server.huggingface.co/rows?{query}"


def verify_viewer_response(locator: RowLocator, response: dict, response_revision: str) -> dict[str, Any]:
    """Reject stale, partial or truncated viewer content before reconstructing IDs."""
    viewer_row_url(locator)
    if response_revision != locator.source_ref.revision:
        raise ValueError("Dataset Viewer revision differs from the pinned source revision")
    if response.get("partial"):
        raise ValueError("Partial Dataset Viewer export cannot establish complete split coordinates")
    rows = response["rows"]
    if len(rows) != 1 or rows[0]["row_idx"] != locator.row_offset:
        raise ValueError("Dataset Viewer returned a different row offset")
    if rows[0].get("truncated_cells"):
        raise ValueError("Dataset Viewer truncated source values")
    return rows[0]["row"]
