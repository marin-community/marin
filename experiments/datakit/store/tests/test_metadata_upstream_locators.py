# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from experiments.datakit.store.upstream_locators import (
    OffsetNamespace,
    SourceFile,
    SourceManifest,
    SourceRef,
    locate_file_row,
    locate_split_row,
    validate_locator_mapping,
    verify_viewer_response,
    viewer_row_url,
)


def test_offsets_follow_explicit_manifest_order_and_roundtrip_across_empty_files():
    manifest = SourceManifest(
        SourceRef("provider/dataset", "config", "train", "pinned-commit"),
        (SourceFile("z.parquet", 2), SourceFile("empty.parquet", 0), SourceFile("a.parquet", 3)),
        "provider original ordered train file list at pinned-commit",
    )
    expected = [("z.parquet", 0), ("z.parquet", 1), ("a.parquet", 0), ("a.parquet", 1), ("a.parquet", 2)]
    for offset, (path, file_offset) in enumerate(expected):
        physical = locate_file_row(manifest, path, file_offset)
        assert physical.row_offset == offset
        assert locate_split_row(manifest, offset) == physical
        public = {"source_ref": manifest.sha256, "row_offset": offset, "file": path, "file_row_offset": file_offset}
        assert validate_locator_mapping(manifest, public) == physical
        with pytest.raises(ValueError, match="disagrees"):
            validate_locator_mapping(manifest, {**public, "row_offset": offset + 1})
    changed = replace(manifest, files=tuple(reversed(manifest.files)))
    assert locate_file_row(changed, "a.parquet", 0).row_offset == 0
    assert changed.sha256 != manifest.sha256


def test_viewer_coordinates_require_matching_namespace_revision_and_complete_raw_row():
    original = SourceManifest(
        SourceRef("provider/dataset", "config", "train", "pinned-commit"),
        (SourceFile("raw.parquet", 4),),
        "explicit original-source file order",
    )
    with pytest.raises(ValueError, match="ordering"):
        viewer_row_url(locate_split_row(original, 2))
    viewer = replace(
        original,
        offset_namespace=OffsetNamespace.HF_VIEWER_PARQUET,
        ordering_evidence="converted parquet metadata ordering verified against the viewer revision",
    )
    locator = locate_split_row(viewer, 2)
    payload = {
        "partial": False,
        "rows": [{"row_idx": 2, "row": {"text": "complete source text"}, "truncated_cells": []}],
    }
    assert verify_viewer_response(locator, payload, "pinned-commit") == {"text": "complete source text"}
    with pytest.raises(ValueError, match="revision"):
        verify_viewer_response(locator, payload, "newer-commit")
    with pytest.raises(ValueError, match="truncated"):
        verify_viewer_response(
            locator, {**payload, "rows": [{**payload["rows"][0], "truncated_cells": [0]}]}, "pinned-commit"
        )
