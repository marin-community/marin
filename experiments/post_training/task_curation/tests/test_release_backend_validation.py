# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the backend-validation stage (the graph's survey producer)."""

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import backend_validation as stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

REVISION = "a" * 40


def _census_inputs(tmp_path: Path, populations: dict[str, str]) -> ArtifactRef:
    document = {
        "kind": "atlas-population-census-inputs",
        "release_id": "testrelease",
        "source_revision": REVISION,
        "rows": [
            {"source_id": target, "declaration": {"id": target}, "population_path": population}
            for target, population in sorted(populations.items())
        ],
    }
    raw = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    path = StoragePath(str(tmp_path / "census-inputs.json"))
    path.write_bytes(raw, auto_mkdir=True)
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _population(tmp_path: Path, target: str, *, task_count: int = 10) -> str:
    """A local population record artifact: a SourceTextPopulation with its contract summary."""
    population = str(tmp_path / "population")
    record = {
        "result": {
            "path": population,
            "source_id": target,
            "dataset_id": "open-athena/fixture-dataset",
            "dataset_revision": "d" * 40,
            "source_version": "fixture-v1",
            "converter_revision": "c" * 40,
            "text_renderer_version": "renderer-v1",
            "task_count": task_count,
            "text_sha256": "t" * 64,
            "task_specs_sha256": "e" * 64,
            "main_output_dir": f"{population}/outputs/main",
            "specs_output_dir": f"{population}/specs",
            "converter_rejections": {},
            "admission_rejections": {},
        }
    }
    StoragePath(f"{population}/.artifact.json").write_text(json.dumps(record), auto_mkdir=True)
    return population


def _inputs(
    tmp_path: Path, *, targets: tuple[str, ...] = ("Marin:fixture",), task_count: int = 10
) -> stage.BackendValidationInputs:
    populations = {target: _population(tmp_path, target, task_count=task_count) for target in targets}
    return stage.BackendValidationInputs(
        release_id="testrelease",
        source_revision=REVISION,
        snapshot_id="assembly-test-v1",
        catalog_root="s3://bucket/catalog",
        output_root=str(tmp_path / "release"),
        targets=targets,
        census_inputs=_census_inputs(tmp_path, populations),
        grading_environments={
            target: {"assessed_tasks": task_count, "evidence_url": "s3://bucket/evidence"} for target in targets
        },
        integrity_contracts={
            target: {
                "task_specs_sha256": "e" * 64,
                "groups": [{"id": "root-delivery", "count": task_count, "present": True}],
            }
            for target in targets
        },
        runtime_defaults={"agent": {"network": "allow"}, "grader": {"network": "deny"}},
    )


def test_dry_run_verifies_the_campaign_records_against_the_census_artifact(tmp_path):
    inputs = _inputs(tmp_path)
    proof = stage.run_backend_validation(inputs, dry_run=True, repo_root=tmp_path)
    assert proof["release_id"] == "testrelease" and proof["dry_run"] is True
    assert proof["task_count"] == 10


def test_dry_run_refuses_a_grading_record_that_disagrees_with_the_census(tmp_path):
    inputs = _inputs(tmp_path, task_count=10)
    broken = dataclasses.replace(inputs, grading_environments={"Marin:fixture": {"assessed_tasks": 9}})
    with pytest.raises(ValueError, match="assesses 9 tasks"):
        stage.run_backend_validation(broken, dry_run=True, repo_root=tmp_path)


def test_dry_run_refuses_contract_groups_that_disagree_with_the_census(tmp_path):
    inputs = _inputs(tmp_path)
    broken = dataclasses.replace(
        inputs,
        integrity_contracts={
            "Marin:fixture": {
                "task_specs_sha256": "e" * 64,
                "groups": [{"id": "root-delivery", "count": 7, "present": True}],
            }
        },
    )
    with pytest.raises(ValueError, match="contract record covers 7"):
        stage.run_backend_validation(broken, dry_run=True, repo_root=tmp_path)


def test_dry_run_refuses_specs_that_disagree_with_the_census(tmp_path):
    inputs = _inputs(tmp_path)
    broken = dataclasses.replace(
        inputs,
        integrity_contracts={
            "Marin:fixture": {
                "task_specs_sha256": "f" * 64,
                "groups": [{"id": "root-delivery", "count": 10, "present": True}],
            }
        },
    )
    with pytest.raises(ValueError, match="pins specs"):
        stage.run_backend_validation(broken, dry_run=True, repo_root=tmp_path)


def test_manifest_binds_the_release_and_the_frozen_populations(tmp_path):
    inputs = _inputs(tmp_path, targets=("Marin:fixture", "Marin:other2"))
    manifest = stage.build_census_manifest(
        inputs,
        task_counts={"Marin:fixture": 10, "Marin:other2": 4},
        specs_paths={t: [str(tmp_path / "specs" / f"{t}.parquet")] for t in ("Marin:fixture", "Marin:other2")},
    )
    assert manifest["kind"] == "atlas-backend-census-inputs"
    assert manifest["snapshot_id"] == "assembly-test-v1"
    assert manifest["source_revision"] == REVISION
    assert manifest["source_ids"] == ["Marin:fixture", "Marin:other2"]
    assert [row["expected_tasks"] for row in manifest["inputs"]] == [10, 4]
    assert manifest["integrity_contracts"]["Marin:fixture"] == inputs.integrity_contracts["Marin:fixture"]


def test_missing_census_row_refuses(tmp_path):
    inputs = _inputs(tmp_path)
    broken = dataclasses.replace(inputs, targets=("Marin:other",))
    with pytest.raises(ValueError, match="targets carry no census row"):
        stage.run_backend_validation(broken, dry_run=True, repo_root=tmp_path)
