# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
from dataclasses import replace
from email.message import Message
from urllib.response import addinfourl

import pytest

from experiments.datakit.scripts.metadata_upstream_locators import (
    OffsetNamespace,
    SourceFile,
    SourceManifest,
    SourceRef,
    card_file_memberships,
    locate_file_row,
    locate_split_row,
    probe_viewer_row,
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


def test_pinned_card_memberships_preserve_config_ambiguity_and_distinct_splits():
    card = {
        "configs": [
            {"config_name": "all", "data_files": "data/*/*/*.parquet"},
            {
                "config_name": "eng",
                "data_files": [
                    {"split": "train", "path": "data/eng/train/*.parquet"},
                    {"split": "test", "path": "data/eng/test/*.parquet"},
                ],
            },
        ]
    }
    files = ["data/eng/test/first.parquet", "data/eng/train/second.parquet", "README.md"]
    matches = card_file_memberships(card, files)
    assert [(match.config, match.split) for match in matches[files[0]]] == [("all", "train"), ("eng", "test")]
    assert [(match.config, match.split) for match in matches[files[1]]] == [("all", "train"), ("eng", "train")]
    assert matches["README.md"] == []


@pytest.mark.parametrize("ids", [["document-1"], []])
def test_viewer_probe_requires_a_reconstructed_document(monkeypatch, ids):
    manifest = SourceManifest(
        SourceRef("provider/dataset", "config", "train", "pinned-commit"),
        (SourceFile("original.parquet", 1),),
        "verified original/viewer file identity",
        OffsetNamespace.HF_VIEWER_PARQUET,
    )
    payload = {"partial": False, "rows": [{"row_idx": 0, "row": {"ids": ids}, "truncated_cells": []}]}
    headers = Message()
    headers["X-Revision"] = "pinned-commit"
    stream = addinfourl(io.BytesIO(json.dumps(payload).encode()), headers, "https://datasets-server.huggingface.co/rows")
    monkeypatch.setattr("experiments.datakit.scripts.metadata_upstream_locators.urlopen", lambda *args, **kwargs: stream)
    result = probe_viewer_row(manifest, 0, normalized_ids=lambda row: row["ids"], expected_ids=ids)
    assert result["status"] == ("passed" if ids else "mismatch")
    assert result["actual_ids"] == ids
