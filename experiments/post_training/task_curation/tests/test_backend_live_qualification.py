# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the backend validation-receipt assembly (the maintained live-qualification path)."""

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from experiments.post_training.task_curation.backend_live_qualification import (
    MANIFEST_KIND,
    REMOTE_BACKEND_NAMES,
    ValidationInputs,
    read_manifest,
    run_validation,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

SOURCE = "Lego-X:Terminal-Lego-15k"


def _init_repo(root: Path) -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    root.mkdir(parents=True, exist_ok=True)
    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (root / "marker.txt").write_text("one\n")
    git("add", "marker.txt")
    git("commit", "--quiet", "-m", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def _manifest_document(revision: str, snapshot: str = "assembly-fixture-v1", source_id: str = SOURCE) -> dict:
    return {
        "kind": MANIFEST_KIND,
        "source_revision": revision,
        "snapshot_id": snapshot,
        "catalog_root": "s3://fixture/catalog",
        "source_ids": [source_id],
        "inputs": [
            {"source_id": source_id, "paths": ["/fixture/pop.parquet"], "column": "task_spec", "expected_tasks": 15}
        ],
        "grading_environments": {},
        "integrity_contracts": {},
        "runtime_defaults": {
            "agent": {"image": "a", "cpus": 1, "memory_mb": 1},
            "grader": {"image": "g", "cpus": 1, "memory_mb": 1},
        },
        "unassessed_reasons": {},
        "deployment_limits": {},
    }


def _inputs(tmp_path: Path, manifest_document: dict) -> ValidationInputs:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_document))
    return ValidationInputs(
        release_id="m3-dryrun",
        source_revision=manifest_document["source_revision"],
        snapshot_id=manifest_document["snapshot_id"],
        catalog_root=manifest_document["catalog_root"],
        output_root=str(tmp_path / "release" / "backend-validation"),
        targets=(SOURCE,),
        manifest=ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        expected_task_count=15,
    )


@dataclass
class _FakeSource:
    source_id: str
    task_count: int


@dataclass
class _FakeSurvey:
    sources: list


def _fake_survey_builder(manifest: dict, revision: str, disk_pool_root):
    source_id = manifest["inputs"][0]["source_id"]
    return _FakeSurvey(sources=[_FakeSource(source_id=source_id, task_count=15)])


def test_read_manifest_checks_bindings_and_coverage(tmp_path):
    revision = "a" * 40
    inputs = _inputs(tmp_path, _manifest_document(revision))
    manifest = read_manifest(inputs)
    assert manifest["kind"] == MANIFEST_KIND
    # A drifted binding refuses: the inputs pin the original manifest, the file moved.
    drifted = _manifest_document(revision)
    drifted["snapshot_id"] = "assembly-other-v9"
    drifted_ref = _seal_drifted(tmp_path, drifted)
    with pytest.raises(ValueError, match="binds snapshot"):
        read_manifest(dataclasses.replace(inputs, manifest=drifted_ref))
    missing = _manifest_document(revision, source_id="Marin:other")
    missing_ref = _seal_drifted(tmp_path, missing)
    with pytest.raises(ValueError, match="must cover exactly the release targets"):
        read_manifest(dataclasses.replace(inputs, manifest=missing_ref))


def _seal_drifted(tmp_path: Path, document: dict) -> ArtifactRef:
    path = tmp_path / f"manifest-{document['snapshot_id'] or document['source_ids'][0]}.json"
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_dry_run_verifies_with_zero_writes(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _inputs(tmp_path, _manifest_document(revision))
    proof = run_validation(inputs, dry_run=True, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["sql_writes"] == 0
    assert proof["targets"] == [SOURCE]
    assert not (tmp_path / "release" / "backend-validation").exists() or not any(
        (tmp_path / "release" / "backend-validation").iterdir()
    )


def test_write_seals_the_survey_and_the_validation_receipt(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _inputs(tmp_path, _manifest_document(revision))

    def fake_probe(backend: str, **kwargs) -> dict:
        return {"backend": backend, "state": "supported", "detail": "fixture pull ok"}

    receipt = run_validation(
        inputs,
        dry_run=False,
        repo_root=tmp_path / "repo",
        survey_builder=_fake_survey_builder,
        pull_probe=fake_probe,
        feeds=_feeds(tmp_path),
    )
    assert receipt["kind"] == "atlas-m3-dryrun-backend-validation"
    assert receipt["sources"] == [SOURCE] and receipt["task_count"] == 15
    assert [record["backend"] for record in receipt["pull_probes"]] == list(REMOTE_BACKEND_NAMES)
    survey_path = Path(receipt["survey"]["uri"])
    assert hashlib.sha256(survey_path.read_bytes()).hexdigest() == receipt["survey"]["sha256"]
    receipt_path = Path(receipt["receipt"]["uri"])
    stored = json.loads(receipt_path.read_text())
    assert stored["survey"]["sha256"] == receipt["survey"]["sha256"]
    assert stored["completed_at"]
    # The receipt seals write-once: a rerun either is byte-identical (the same
    # second) or refuses — either way the sealed document never changes.
    sealed = receipt_path.read_bytes()
    try:
        run_validation(
            inputs,
            dry_run=False,
            repo_root=tmp_path / "repo",
            survey_builder=_fake_survey_builder,
            pull_probe=fake_probe,
            feeds=_feeds(tmp_path),
        )
    except ValueError as error:
        assert "already exists with different content" in str(error)
    assert receipt_path.read_bytes() == sealed


def test_write_refuses_a_survey_that_does_not_cover_the_targets(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _inputs(tmp_path, _manifest_document(revision))

    def wrong_builder(manifest: dict, revision: str, disk_pool_root):
        return _FakeSurvey(sources=[_FakeSource(source_id="Marin:other", task_count=15)])

    with pytest.raises(ValueError, match=r"covers .* expected exactly the release targets"):
        run_validation(
            inputs,
            dry_run=False,
            repo_root=tmp_path / "repo",
            survey_builder=wrong_builder,
            pull_probe=lambda backend, **kwargs: {"backend": backend, "state": "supported", "detail": "x"},
            feeds=_feeds(tmp_path),
        )


def test_write_refuses_a_measured_task_total_drift(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _inputs(tmp_path, _manifest_document(revision))

    def drift_builder(manifest: dict, revision: str, disk_pool_root):
        return _FakeSurvey(sources=[_FakeSource(source_id=SOURCE, task_count=14)])

    with pytest.raises(ValueError, match="measures 14 tasks, expected 15"):
        run_validation(
            inputs,
            dry_run=False,
            repo_root=tmp_path / "repo",
            survey_builder=drift_builder,
            pull_probe=lambda backend, **kwargs: {"backend": backend, "state": "supported", "detail": "x"},
            feeds=_feeds(tmp_path),
        )


def test_write_refuses_a_moved_checkout(tmp_path):
    """The manifest pins the revision it was sealed at; a moved HEAD refuses."""
    revision = _init_repo(tmp_path / "repo")
    manifest = _manifest_document(revision)
    inputs = _inputs(tmp_path, manifest)
    (tmp_path / "repo" / "marker.txt").write_text("two\n")
    subprocess.run(["git", "add", "marker.txt"], cwd=tmp_path / "repo", check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "two"], cwd=tmp_path / "repo", check=True, capture_output=True)
    with pytest.raises(ValueError, match="not the expected release revision"):
        run_validation(
            inputs,
            dry_run=False,
            repo_root=tmp_path / "repo",
            survey_builder=_fake_survey_builder,
            feeds=_feeds(tmp_path),
        )
