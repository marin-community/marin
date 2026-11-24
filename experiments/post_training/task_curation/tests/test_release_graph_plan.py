# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the release plan boundary (DESIGN.md §3, M2c scaffold)."""

import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.graph as graph
from experiments.post_training.task_curation.release_stages.graph import (
    load_plan,
    plan_fingerprint,
    plan_ref,
    stage_output_root,
    write_plan,
)
from experiments.post_training.task_curation.tests.release_graph_fixtures import fixture_plan


def test_plan_validates_shape():
    fixture_plan().validate_shape()
    with pytest.raises(ValueError, match="32-hex"):
        fixture_plan(
            prior={
                "release_id": "x",
                "load_id": "short",
                "snapshot_uri": "u",
                "snapshot_sha256": "s",
                "catalog_uri": "c",
            }
        ).validate_shape()
    with pytest.raises(ValueError, match="region_prefix"):
        fixture_plan(region_prefix="s3://bucket-region").validate_shape()
    with pytest.raises(ValueError, match="kind"):
        fixture_plan(kind="other").validate_shape()
    with pytest.raises(ValueError, match="artifact_version"):
        fixture_plan(artifact_version="v1").validate_shape()
    with pytest.raises(ValueError, match="populations"):
        fixture_plan(populations={}).validate_shape()
    with pytest.raises(ValueError, match="scan region"):
        fixture_plan(scan_regions={"alpha": "s3://bucket-region/alpha"}).validate_shape()
    with pytest.raises(ValueError, match="rebound"):
        fixture_plan(rebound_source="Marin:fixture", rebound_tasks=0).validate_shape()
    with pytest.raises(ValueError, match="production_version"):
        fixture_plan(
            service={**fixture_plan().service.model_dump(mode="json"), "production_version": 0}
        ).validate_shape()
    with pytest.raises(ValueError, match="measured dependency"):
        fixture_plan(measured={**fixture_plan().measured.model_dump(mode="json"), "sha256": ""}).validate_shape()


def test_write_plan_seals_write_once(tmp_path):
    root = str(tmp_path / "release")
    first = write_plan(fixture_plan(), root)
    assert first["uri"].endswith("release-plan.json")
    stored = json.loads(StoragePath(first["uri"]).read_text())
    assert stored["release_id"] == "testrelease" and stored["prior"]["load_id"] == "b" * 32
    assert stored["pins"]["prior_declarations"]["uri"].endswith("prior/declarations.json")
    loaded = load_plan(Path(first["uri"]))
    assert loaded.targets == ("Marin:fixture",)
    assert loaded.service.preview_base_version == 30
    # Byte-identical rerun is idempotent; a changed plan refuses.
    assert write_plan(fixture_plan(), root)["sha256"] == first["sha256"]
    with pytest.raises(ValueError, match="different content"):
        write_plan(fixture_plan(expected_source_count=3), root)


def test_stage_paths_derive_from_the_plan():
    plan = fixture_plan()
    assert stage_output_root(plan, "census").endswith("testrelease/census")
    assert plan_ref(plan.release_root, "census/receipt.json")["uri"].endswith("census/receipt.json")


def test_plan_fingerprint_is_stable_and_content_bound():
    plan = fixture_plan()
    assert plan_fingerprint(plan) == plan_fingerprint(fixture_plan())
    assert plan_fingerprint(plan) != plan_fingerprint(fixture_plan(expected_source_count=3))


def test_registered_difficulty_seal_binds_only_the_named_release(tmp_path, monkeypatch):
    reference = {"uri": "s3://bucket/sealed/manifest.json", "sha256": "a" * 64}
    registry = tmp_path / "difficulty_snapshots.json"
    registry.write_text(json.dumps({"r41": reference}))
    monkeypatch.setattr(graph, "DIFFICULTY_SNAPSHOTS", registry)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(fixture_plan(release_id="r41").model_dump_json())
    assert graph.load_plan(plan_file).difficulty_snapshot.model_dump() == reference
    plan_file.write_text(fixture_plan(release_id="r40").model_dump_json())
    assert graph.load_plan(plan_file).difficulty_snapshot is None
    plan_file.write_text(
        fixture_plan(release_id="r41", difficulty_snapshot={"uri": "other", "sha256": "b" * 64}).model_dump_json()
    )
    with pytest.raises(ValueError, match="registered difficulty seal"):
        graph.load_plan(plan_file)
