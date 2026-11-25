# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained declarations and catalog-staging stages (R40 gate)."""

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from finestore.rl.curation import QualifiedSourceDeclaration, SourceOrigin, source_display_name, source_identifier

from experiments.post_training.task_curation.release_stages import catalog_stage
from experiments.post_training.task_curation.release_stages import declarations as declarations_stage
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

SOURCE_A = "Marin:fixture-a"
SOURCE_B = "Marin:fixture-b"


def _declaration(identifier: str) -> QualifiedSourceDeclaration:
    return QualifiedSourceDeclaration(
        id=(source_id := source_identifier(SourceOrigin.MARIN, identifier, None, None)),
        display_name=source_display_name(source_id),
        origin="Marin",
        build_path="fixture",
        upstream_dataset=identifier,
        upstream_name=identifier,
        current_release={"repo": "fixture", "revision": "1", "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=2,
        within_source_dedup=deduplication_evidence("fixture", "1", "1", count=2),
        notes="",
        dataset_id="fixture",
        dataset_revision="1",
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": "1", "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
    )


def _registry(declarations: dict[str, QualifiedSourceDeclaration]):
    def sources_fn():
        return {
            source_id: SimpleNamespace(
                info=SimpleNamespace(id=source_id, oracle_filtration=None), declaration=declaration
            )
            for source_id, declaration in declarations.items()
        }

    return sources_fn


def _seal_json(tmp_path: Path, name: str, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _init_repo(root: Path, revision_file: str = "x") -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    root.mkdir(parents=True, exist_ok=True)
    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (root / revision_file).write_text("one\n")
    git("add", revision_file)
    git("commit", "--quiet", "-m", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def _declaration_inputs(tmp_path: Path, prior: ArtifactRef, revision: str) -> declarations_stage.DeclarationInputs:
    return declarations_stage.DeclarationInputs(
        release_id="testrelease",
        source_revision=revision,
        output_root=str(tmp_path / "release" / "declarations"),
        expected_source_count=2,
        targets=(SOURCE_B,),
        prior_declarations=prior,
    )


def test_declarations_write_seals_document_and_refuses_rerun(tmp_path, monkeypatch):
    declarations = {SOURCE_A: _declaration("fixture-a"), SOURCE_B: _declaration("fixture-b")}
    monkeypatch.setattr(declarations_stage, "require_catalog_declaration", lambda declaration: None)
    prior = _seal_json(tmp_path, "prior-declarations.json", {SOURCE_A: declarations[SOURCE_A].model_dump(mode="json")})
    revision = _init_repo(tmp_path / "repo")
    inputs = _declaration_inputs(tmp_path, prior, revision)
    proof = declarations_stage.run_declarations(
        inputs, dry_run=False, repo_root=tmp_path / "repo", sources_fn=_registry(declarations), feeds=_feeds(tmp_path)
    )
    document = json.loads((tmp_path / "release" / "declarations" / "declarations.json").read_text())
    assert set(document) == {SOURCE_A, SOURCE_B}
    assert proof["source_count"] == 2 and proof["targets"] == [SOURCE_B]
    with pytest.raises(ValueError, match="already exists"):
        declarations_stage.run_declarations(
            inputs,
            dry_run=False,
            repo_root=tmp_path / "repo",
            sources_fn=_registry(declarations),
            feeds=_feeds(tmp_path),
        )


def test_declarations_dry_run_writes_nothing(tmp_path, monkeypatch):
    declarations = {SOURCE_A: _declaration("fixture-a"), SOURCE_B: _declaration("fixture-b")}
    monkeypatch.setattr(declarations_stage, "require_catalog_declaration", lambda declaration: None)
    prior = _seal_json(tmp_path, "prior-declarations.json", {SOURCE_A: declarations[SOURCE_A].model_dump(mode="json")})
    revision = _init_repo(tmp_path / "repo")
    inputs = _declaration_inputs(tmp_path, prior, revision)
    proof = declarations_stage.run_declarations(
        inputs, dry_run=True, repo_root=tmp_path / "repo", sources_fn=_registry(declarations), feeds=_feeds(tmp_path)
    )
    assert proof["dry_run"] is True and proof["source_count"] == 2
    assert not (tmp_path / "release" / "declarations").exists() or not any(
        (tmp_path / "release" / "declarations").iterdir()
    )


def test_declarations_refusals(tmp_path, monkeypatch):
    declarations = {SOURCE_A: _declaration("fixture-a"), SOURCE_B: _declaration("fixture-b")}
    monkeypatch.setattr(declarations_stage, "require_catalog_declaration", lambda declaration: None)
    prior = _seal_json(tmp_path, "prior-declarations.json", {SOURCE_A: declarations[SOURCE_A].model_dump(mode="json")})
    revision = _init_repo(tmp_path / "repo")
    registry = _registry(declarations)

    with pytest.raises(ValueError, match="dirty"):
        (tmp_path / "repo" / "x").write_text("dirty\n")
        declarations_stage.run_declarations(
            _declaration_inputs(tmp_path, prior, revision),
            dry_run=True,
            repo_root=tmp_path / "repo",
            sources_fn=registry,
            feeds=_feeds(tmp_path),
        )
    _init_repo(tmp_path / "repo2")
    head2 = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path / "repo2", capture_output=True, text=True, check=True
    ).stdout.strip()
    with pytest.raises(ValueError, match="does not contain the plan's pinned revision"):
        declarations_stage.run_declarations(
            _declaration_inputs(tmp_path, prior, "b" * 40),
            dry_run=True,
            repo_root=tmp_path / "repo2",
            sources_fn=registry,
            feeds=_feeds(tmp_path),
        )
    wrong_count = declarations_stage.DeclarationInputs(
        release_id="testrelease",
        source_revision=head2,
        output_root=str(tmp_path / "release" / "declarations"),
        expected_source_count=3,
        targets=(SOURCE_B,),
        prior_declarations=prior,
    )
    with pytest.raises(ValueError, match="expected 3"):
        declarations_stage.run_declarations(
            wrong_count, dry_run=True, repo_root=tmp_path / "repo2", sources_fn=registry, feeds=_feeds(tmp_path)
        )
    missing_target = declarations_stage.DeclarationInputs(
        release_id="testrelease",
        source_revision=head2,
        output_root=str(tmp_path / "release" / "declarations"),
        expected_source_count=2,
        targets=("Marin:missing",),
        prior_declarations=prior,
    )
    with pytest.raises(ValueError, match="unknown to the prior and current"):
        declarations_stage.run_declarations(
            missing_target, dry_run=True, repo_root=tmp_path / "repo2", sources_fn=registry, feeds=_feeds(tmp_path)
        )
    drifted_prior = _seal_json(
        tmp_path,
        "drifted-prior.json",
        {SOURCE_A: {**declarations[SOURCE_A].model_dump(mode="json"), "dataset_revision": "2"}},
    )
    with pytest.raises(ValueError, match="drifted"):
        declarations_stage.run_declarations(
            _declaration_inputs(tmp_path, drifted_prior, head2),
            dry_run=True,
            repo_root=tmp_path / "repo2",
            sources_fn=registry,
            feeds=_feeds(tmp_path),
        )


def test_declarations_refuse_unplanned_removals_and_allow_planned_ones(tmp_path, monkeypatch):
    declarations = {SOURCE_A: _declaration("fixture-a"), SOURCE_B: _declaration("fixture-b")}
    monkeypatch.setattr(declarations_stage, "require_catalog_declaration", lambda declaration: None)
    prior = _seal_json(tmp_path, "prior-declarations.json", {SOURCE_A: declarations[SOURCE_A].model_dump(mode="json")})
    revision = _init_repo(tmp_path / "repo")
    # the current set drops the prior's only source: a refusal unless the plan removes it
    dropped = _registry({SOURCE_B: _declaration("fixture-b")})
    unannounced = _declaration_inputs(tmp_path, prior, revision)
    unannounced = dataclasses.replace(unannounced, expected_source_count=1)
    with pytest.raises(ValueError, match="absent from the current set"):
        declarations_stage.run_declarations(
            unannounced,
            dry_run=True,
            repo_root=tmp_path / "repo",
            sources_fn=dropped,
            feeds=_feeds(tmp_path),
        )
    planned = dataclasses.replace(
        _declaration_inputs(tmp_path, prior, revision),
        removed_sources=(SOURCE_A,),
        expected_source_count=1,
    )
    proof = declarations_stage.run_declarations(
        planned, dry_run=True, repo_root=tmp_path / "repo", sources_fn=dropped, feeds=_feeds(tmp_path)
    )
    assert proof["source_count"] == 1


def _staged_documents(tmp_path: Path) -> dict[str, dict]:
    return {
        "declarations.json": {SOURCE_A: {}, SOURCE_B: {}},
        "publication.json": {
            "backend": {
                "uri": str(tmp_path / "backend.json"),
                "sha256": (
                    hashlib.sha256(
                        json.dumps({"sources": [{"source_id": SOURCE_A}, {"source_id": SOURCE_B}]}).encode()
                    ).hexdigest()
                ),
            },
            "identity": {
                "uri": str(tmp_path / "identity.json"),
                "sha256": (
                    hashlib.sha256(json.dumps({"fingerprints": {SOURCE_A: {}, SOURCE_B: {}}}).encode()).hexdigest()
                ),
            },
        },
        "history.json": {
            "load_id": "b" * 32,
            "prior_load_id": "a" * 32,
            "selected_pointer_changed": False,
            "prior_item_count": 100,
        },
        "browse.json": {
            "status": "PASS",
            "source_count": 2,
            "target_load_id": "b" * 32,
            "source_load_id": "a" * 32,
        },
        "fresh-qc.json": {"load_id": "b" * 32, "snapshot_id": "assembly-test-v1", "receipts": []},
        "contracts.json": {"contracts": [{"input": {"measured_tasks": 4}}]},
        "backend.json": {"sources": [{"source_id": SOURCE_A}, {"source_id": SOURCE_B}]},
        "identity.json": {"fingerprints": {SOURCE_A: {}, SOURCE_B: {}}},
    }


def _sealed_staged_inputs(tmp_path: Path, **overrides) -> catalog_stage.CatalogStageInputs:
    documents = _staged_documents(tmp_path)
    # The publication refs point at their own digest-bound documents.
    (tmp_path / "backend.json").write_text(json.dumps(documents["backend.json"]))
    (tmp_path / "identity.json").write_text(json.dumps(documents["identity.json"]))
    refs = {}
    file_for = {
        "declarations": "declarations.json",
        "publication": "publication.json",
        "history_receipt": "history.json",
        "prior_qc_completion": "browse.json",
        "fresh_qc_index": "fresh-qc.json",
        "contracts": "contracts.json",
    }
    for field, name in file_for.items():
        path = tmp_path / name
        # Write only absent files: refusal cases tamper with documents first and the
        # refs must bind the tampered bytes.
        if not path.exists():
            path.write_text(json.dumps(documents[name]))
        refs[field] = ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    inputs = catalog_stage.CatalogStageInputs(
        release_id="testrelease",
        prior_load_id="a" * 32,
        target_load_id="b" * 32,
        snapshot_id="assembly-test-v1",
        source_revision="c" * 40,
        catalog_root=str(tmp_path / "catalog"),
        output_root=str(tmp_path / "release" / "catalog-stage"),
        expected_source_count=2,
        expected_browse_source_count=2,
        expected_contract_count=1,
        expected_contract_tasks=4,
        **refs,
    )
    if overrides:
        return dataclasses.replace(inputs, **overrides)
    return inputs


class _FakeSnapshot:
    def __init__(self, snapshot_id: str) -> None:
        self.snapshot_id = snapshot_id


class _FakeCatalog:
    def __init__(self, snapshot_id: str, source_ids: list[str]) -> None:
        self._snapshot = _FakeSnapshot(snapshot_id)
        self._source_ids = source_ids

    def current(self) -> _FakeSnapshot:
        return self._snapshot

    def sources(self, snapshot_id: str) -> list:
        return [SimpleNamespace(id=source_id) for source_id in self._source_ids]


def test_catalog_stage_dry_run_verifies_and_writes_nothing(tmp_path, monkeypatch):
    inputs = _sealed_staged_inputs(tmp_path)
    monkeypatch.setattr(
        catalog_stage, "CurationCatalog", lambda root: _FakeCatalog("assembly-test-v1", [SOURCE_A, SOURCE_B])
    )
    proof = catalog_stage.run_catalog_stage(inputs, dry_run=True, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["source_count"] == 2
    assert not (tmp_path / "release" / "catalog-stage").exists() or not any(
        (tmp_path / "release" / "catalog-stage").iterdir()
    )


def test_catalog_stage_write_seals_plan(tmp_path, monkeypatch):
    inputs = _sealed_staged_inputs(tmp_path)
    monkeypatch.setattr(
        catalog_stage, "CurationCatalog", lambda root: _FakeCatalog("assembly-test-v1", [SOURCE_A, SOURCE_B])
    )
    proof = catalog_stage.run_catalog_stage(inputs, dry_run=False, feeds=_feeds(tmp_path))
    plan = json.loads((tmp_path / "release" / "catalog-stage" / "catalog-plan.json").read_text())
    assert plan["target_load_id"] == "b" * 32
    assert proof["receipt"]["uri"].endswith("catalog-plan.json")
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["kind"] == catalog_stage.alerts.KIND_FINISH


def test_catalog_stage_refusals(tmp_path, monkeypatch):
    inputs = _sealed_staged_inputs(tmp_path)
    monkeypatch.setattr(
        catalog_stage, "CurationCatalog", lambda root: _FakeCatalog("assembly-test-v1", [SOURCE_A, SOURCE_B])
    )
    # Wrong load IDs in the history receipt.
    history = json.loads((tmp_path / "history.json").read_text())
    history["load_id"] = "c" * 32
    (tmp_path / "history.json").write_text(json.dumps(history))
    bad_history = _sealed_staged_inputs(tmp_path)
    with pytest.raises(ValueError, match="targets"):
        catalog_stage.run_catalog_stage(bad_history, dry_run=True, feeds=_feeds(tmp_path))
    # Snapshot moved.
    (tmp_path / "history.json").write_text(json.dumps(_staged_documents(tmp_path)["history.json"]))
    monkeypatch.setattr(catalog_stage, "CurationCatalog", lambda root: _FakeCatalog("assembly-other-v9", [SOURCE_A]))
    with pytest.raises(ValueError, match="catalog snapshot is"):
        catalog_stage.run_catalog_stage(inputs, dry_run=True, feeds=_feeds(tmp_path))
    monkeypatch.setattr(
        catalog_stage, "CurationCatalog", lambda root: _FakeCatalog("assembly-test-v1", [SOURCE_A, SOURCE_B])
    )
    # Contracts disagree with the planned measured total.
    contracts = {"contracts": [{"input": {"measured_tasks": 9}}]}
    (tmp_path / "contracts.json").write_text(json.dumps(contracts))
    bad_contracts = _sealed_staged_inputs(tmp_path)
    with pytest.raises(ValueError, match="measure 9 tasks"):
        catalog_stage.run_catalog_stage(bad_contracts, dry_run=True, feeds=_feeds(tmp_path))
    # Migration not in the in-repo schema.
    (tmp_path / "contracts.json").write_text(json.dumps({"contracts": [{"input": {"measured_tasks": 4}}]}))
    with pytest.raises(ValueError, match="not part of the in-repo schema"):
        catalog_stage.run_catalog_stage(
            _sealed_staged_inputs(tmp_path, migrations=("ALTER TABLE atlas_items ADD COLUMN not_in_schema_v40 TEXT",)),
            dry_run=True,
            feeds=_feeds(tmp_path),
        )
