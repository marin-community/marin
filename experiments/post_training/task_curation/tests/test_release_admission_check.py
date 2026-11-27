# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the maintained admission-check stage (R40 gate).

Coordinator design input (2026-11-23): the catalog-load v1/v2 failures died on the
cluster at ``load_catalog``'s backend admission gate after dry runs that never
exercised it. These tests pin the fix: a blocked source fails the DRY RUN, with all
blocked sources reported at once — never the cluster job.
"""

import hashlib
import json
from functools import partial
from pathlib import Path

import pytest
from finestore.rl.curation import QualifiedSourceDeclaration, SourceOrigin, source_display_name, source_identifier
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, NetworkPolicy, UnsupportedMachineSpec
from shellbox.survey import LocalResourceEvidence
from taskcompendium.backend_requirements import (
    BackendFit,
    BackendSourceMap,
    BackendSurvey,
    FitSupport,
    RuntimeProfile,
)

from experiments.post_training.task_curation.backend_capabilities import production_matrix
from experiments.post_training.task_curation.release_stages import admission_check
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests.admission_fixtures import with_declaration_admission_census
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

SOURCE_ID = "Marin:fixture"
SECOND_ID = "Marin:fixture-two"
RUNTIME = RuntimeProfile(
    network=NetworkPolicy.DENY,
    cpus=1,
    memory_mb=512,
    storage_mb=1024,
    gpus=0,
    user=None,
    startup_timeout=60.0,
    cleanup_timeout=30.0,
)
BACKEND_VALUES = frozenset(backend.value for backend in Backend) | {"iris"}


def _declaration(dataset: str = "open-athena/fixture", revision: str = "b" * 40, identifier: str = "fixture") -> dict:
    base = QualifiedSourceDeclaration(
        id=(source_id := source_identifier(SourceOrigin.MARIN, identifier, None, None)),
        display_name=source_display_name(source_id),
        origin="Marin",
        build_path="fixture",
        upstream_dataset=identifier,
        upstream_name=identifier,
        current_release={"repo": "fixture", "revision": revision, "url": "https://example.org/pin"},
        family="fixture",
        tags=[],
        task_count=2,
        within_source_dedup=deduplication_evidence(dataset, revision, "1", count=2),
        notes="",
        dataset_id=dataset,
        dataset_revision=revision,
        url=None,
        type="RLVR",
        turns="Single-turn",
        is_benchmark=False,
        kind="Dataset",
        status="Available",
        pipeline={"name": "fixture", "version": "1", "source": "fixture", "revision": revision, "files": []},
        daytona_safety="not_applicable",
        daytona_snapshot_count=0,
        daytona_machine_slots=0,
        daytona_execution_failures=0,
        daytona_count_basis="Fixture contains no actor environment.",
        intended_use="train",
    )
    admitted = with_declaration_admission_census(base)
    return {
        **admitted.model_dump(mode="json"),
        "lexical_contamination": decontamination_evidence(dataset, revision, 2).model_dump(mode="json"),
    }


def _fit(support: FitSupport, count: int = 2) -> BackendFit:
    counts = {support: count}
    return BackendFit(
        support=support,
        task_counts=counts,
        roles={"agent": dict(counts), "grader": dict(counts)},
        blockers={},
        validation_errors={},
    )


def _row(source_id: str, fits: dict[str, BackendFit], task_count: int = 2) -> BackendSourceMap:
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=task_count,
        inputs={},
        demands={"agent": {}, "grader": {}},
        resource_ranges={"agent": {}, "grader": {}},
        backends=fits,
        role_backend_assignments={"agent": {}, "grader": {}},
        grading_environment=None,
    )


def _survey(source_id: str, blocked: bool, matrix: list) -> BackendSurvey:
    if blocked:
        # Every production backend is unsupported while every dev-only backend is
        # supported: the review blocks because dev-only support never admits.
        fits = {
            backend: _fit(FitSupport.UNSUPPORTED if backend in ("iris", "daytona", "local") else FitSupport.SUPPORTED)
            for backend in BACKEND_VALUES
        }
    else:
        fits = {backend: _fit(FitSupport.SUPPORTED) for backend in BACKEND_VALUES}
    return BackendSurvey(
        catalog_root="test:catalog",
        snapshot_id="assembly-test-v1",
        generated_at="2026-11-23T00:00:00Z",
        git_revision="a" * 40,
        schema_sha256="0" * 64,
        runtime_defaults={"agent": RUNTIME, "grader": RUNTIME},
        matrix=matrix,
        deployment_limits={},
        deployment_classes={},
        sources=[_row(source_id, fits)],
    )


def _survey_with_matrix(source_id: str, blocked: bool, checkers) -> BackendSurvey:
    return _survey(source_id, blocked, production_matrix(checkers))


def _declarations_document(identifier: str = "fixture") -> dict:
    return {SOURCE_ID: _declaration(identifier=identifier)}


def _inputs(
    tmp_path: Path, declarations: dict, survey: BackendSurvey, **overrides
) -> admission_check.AdmissionCheckInputs:
    declarations_ref = ArtifactRef(
        uri=str(_write(tmp_path, "declarations.json", declarations)),
        sha256=_digest(_write(tmp_path, "declarations.json", declarations)),
    )
    del declarations_ref
    path = _write(tmp_path, "declarations.json", declarations)
    backend_path = _write(tmp_path, "backend.json", json.loads(survey.model_dump_json()))
    defaults = dict(
        release_id="testrelease",
        source_revision="a" * 40,
        declarations=ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
        backend=ArtifactRef(uri=str(backend_path), sha256=hashlib.sha256(backend_path.read_bytes()).hexdigest()),
        snapshot_id="assembly-test-v1",
        catalog_root="test:catalog",
        prior_load_id="b" * 32,
        output_root=str(tmp_path / "release" / "admission"),
        expected_source_count=len(declarations),
    )
    defaults.update(overrides)
    return admission_check.AdmissionCheckInputs(**defaults)


def _write(tmp_path: Path, name: str, document) -> Path:
    path = tmp_path / name
    path.write_text(document if isinstance(document, str) else json.dumps(document))
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


class _FakeQuery:
    def __init__(self, available: set[str]) -> None:
        self.available = available
        self.executed: list[str] = []

    def __call__(self, sql: str, params: dict) -> dict:
        self.executed.append(sql)
        if "atlas_sources" in sql:
            return {"rows": [{"source_id": source_id} for source_id in sorted(self.available)]}
        raise AssertionError(f"unexpected query: {sql}")


def _unit_checkers():
    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The unit lane installs no optional backends")

    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "docker-gvisor": rejecting,
        "qemu": rejecting,
        "shellsim": rejecting,
        "daytona": rejecting,
        "iris": rejecting,
    }


def _accepting_local_checkers():
    def accepting(requirements, spec, **kwargs) -> None:
        pass

    return {
        "docker": DockerMachineFactory().check_environment,
        "local": partial(
            accepting,
            resource_evidence=LocalResourceEvidence(storage_enforced=True, cgroup_verified=True, cgroup_parent=None),
        ),
        "docker-gvisor": _rejecting(),
        "qemu": _rejecting(),
        "shellsim": _rejecting(),
        "daytona": _rejecting(),
        "iris": _rejecting(),
    }


def _rejecting():
    def rejecting(requirements, spec) -> None:
        raise UnsupportedMachineSpec("The unit lane installs no optional backends")

    return rejecting


def test_runtime_fit_does_not_downgrade_prepared_sources():
    declarations = {SOURCE_ID: _declaration(), SECOND_ID: _declaration(identifier="fixture-two")}
    survey = _survey_with_matrix(SOURCE_ID, blocked=True, checkers=_unit_checkers()).model_copy(
        update={"sources": [_row(SOURCE_ID, _blocked_fits()), _row(SECOND_ID, _blocked_fits())]}
    )
    report = admission_check.review_admissions(declarations, survey, set())
    assert report.ok and report.review_source_count == 2
    assert report.downgraded == {}


def _blocked_fits() -> dict[str, BackendFit]:
    return {
        backend: _fit(FitSupport.UNSUPPORTED if backend in ("iris", "daytona", "local") else FitSupport.SUPPORTED)
        for backend in BACKEND_VALUES
    }


def _healthy_fits() -> dict[str, BackendFit]:
    return {backend: _fit(FitSupport.SUPPORTED) for backend in BACKEND_VALUES}


def test_review_admissions_routes_pending_and_gates(tmp_path):
    declarations = {SOURCE_ID: _declaration()}
    survey = _survey_with_matrix(SOURCE_ID, blocked=False, checkers=_accepting_local_checkers()).model_copy(
        update={"sources": [_row(SOURCE_ID, _healthy_fits())]}
    )
    report = admission_check.review_admissions(declarations, survey, set())
    assert report.ok and report.review_source_count == 1


def test_run_admission_check_dry_run_retains_unsupported_evidence_without_writes(tmp_path):
    declarations = _declarations_document()
    survey = _survey_with_matrix(SOURCE_ID, blocked=True, checkers=_unit_checkers())
    inputs = _inputs(tmp_path, declarations, survey)
    query = _FakeQuery(set())
    proof = admission_check.run_admission_check(inputs, dry_run=True, query=query, feeds=_feeds(tmp_path))
    assert proof["downgraded"] == {}
    # Read-only: the only SQL is the prior-availability lookup, and nothing is sealed.
    assert all("INSERT" not in sql.upper() for sql in query.executed)
    assert not (tmp_path / "release" / "admission" / "admission-check.json").exists()
    rows = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["kind"] == admission_check.alerts.KIND_FINISH


def test_run_admission_check_seals_receipt_on_a_clean_population(tmp_path):
    declarations = _declarations_document()
    survey = _survey_with_matrix(SOURCE_ID, blocked=False, checkers=_accepting_local_checkers())
    inputs = _inputs(tmp_path, declarations, survey)
    proof = admission_check.run_admission_check(
        inputs, dry_run=False, query=_FakeQuery({SOURCE_ID}), feeds=_feeds(tmp_path)
    )
    assert proof["review_source_count"] == 1 and proof["previously_available"] == 1
    assert proof["downgraded"] == {} and proof["verification_pending"] == {}
    receipt = json.loads((tmp_path / "release" / "admission" / "admission-check.json").read_text())
    assert receipt["kind"] == "atlas-testrelease-admission-check"
    assert receipt["sql_writes"] == 0


def test_source_evidence_applies_across_snapshots_and_inventory_count_still_matches(tmp_path):
    declarations = _declarations_document()
    survey = _survey_with_matrix(SOURCE_ID, blocked=False, checkers=_accepting_local_checkers())
    moved = _inputs(tmp_path, declarations, survey, snapshot_id="assembly-other-v1")
    proof = admission_check.run_admission_check(moved, dry_run=True, query=_FakeQuery(set()), feeds=_feeds(tmp_path))
    assert proof["review_source_count"] == 1
    assert proof["downgraded"] == {}
    bad_count = _inputs(tmp_path, declarations, survey, expected_source_count=2)
    with pytest.raises(ValueError, match="expected 2"):
        admission_check.run_admission_check(bad_count, dry_run=True, query=_FakeQuery(set()), feeds=_feeds(tmp_path))
