# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the release plan boundary (DESIGN.md §3, M2c scaffold)."""

import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages.graph import (
    ReleasePlan,
    load_plan,
    plan_ref,
    stage_output_root,
    write_plan,
)


def _plan(**overrides) -> ReleasePlan:
    defaults = dict(
        release_id="testrelease",
        source_revision="a" * 40,
        release_root="s3://bucket/atlas/preview-release/testrelease",
        catalog_root="s3://bucket/catalog/production",
        prior={
            "release_id": "priorrelease",
            "load_id": "b" * 32,
            "snapshot_uri": "s3://bucket/snapshots/prior",
            "snapshot_sha256": "c" * 64,
            "catalog_uri": "s3://bucket/catalog/prior",
        },
        targets=("Marin:fixture",),
        expected_source_count=2,
        expected_browse_source_count=2,
        expected_contract_count=1,
        expected_contract_tasks=10,
        expected_selected_panels=3,
        region_prefix="s3://bucket-region/",
    )
    defaults.update(overrides)
    return ReleasePlan(**defaults)


def test_plan_validates_shape():
    _plan().validate_shape()
    with pytest.raises(ValueError, match="32-hex"):
        _plan(
            prior={
                "release_id": "x",
                "load_id": "short",
                "snapshot_uri": "u",
                "snapshot_sha256": "s",
                "catalog_uri": "c",
            }
        ).validate_shape()
    with pytest.raises(ValueError, match="region_prefix"):
        _plan(region_prefix="s3://bucket-region").validate_shape()
    with pytest.raises(ValueError, match="kind"):
        _plan(kind="other").validate_shape()


def test_write_plan_seals_write_once(tmp_path):
    root = str(tmp_path / "release")
    first = write_plan(_plan(), root)
    assert first["uri"].endswith("release-plan.json")
    stored = json.loads(StoragePath(first["uri"]).read_text())
    assert stored["release_id"] == "testrelease" and stored["prior"]["load_id"] == "b" * 32
    loaded = load_plan(Path(first["uri"]))
    assert loaded.targets == ("Marin:fixture",)
    # Byte-identical rerun is idempotent; a changed plan refuses.
    assert write_plan(_plan(), root)["sha256"] == first["sha256"]
    with pytest.raises(ValueError, match="different content"):
        write_plan(_plan(expected_source_count=3), root)


def test_stage_paths_derive_from_the_plan():
    plan = _plan()
    assert stage_output_root(plan, "census").endswith("testrelease/census")
    assert plan_ref(plan.release_root, "census/receipt.json")["uri"].endswith("census/receipt.json")
