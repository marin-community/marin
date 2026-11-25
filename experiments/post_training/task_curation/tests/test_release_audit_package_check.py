# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the strict-audit, package-preview and release-check stages (R40 gate).

The audit tests drive a fake dict-backed catalog reader against tiny sealed
declaration/backend/identity documents; the release-check tests inject fake
checker callables. The real maintained ``prepare_preview`` and maintained
checker module own their own behavior tests; these tests pin the typed stage
envelopes around them.
"""

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from finestore.rl.curation import QualifiedSourceDeclaration, SourceOrigin, source_display_name, source_identifier
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, NetworkPolicy, UnsupportedMachineSpec
from taskcompendium.backend_requirements import BackendFit, BackendSourceMap, BackendSurvey, FitSupport, RuntimeProfile
from taskcompendium.source_fingerprints import DuplicateGroup, DuplicateGroupMember, SourceIdentitySurvey

from experiments.post_training.task_curation.backend_capabilities import production_matrix
from experiments.post_training.task_curation.release_stages import package_preview, release_check, strict_audit
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from infra.marina.apps.rl_data_catalog.server.curriculum import TOTAL_METRICS
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence

SOURCE_A = "Marin:fixture-a"
SOURCE_B = "Marin:fixture-b"
SOURCE_IDS = (SOURCE_A, SOURCE_B)
REVISION = "c" * 40
SNAPSHOT_ID = "assembly-test-v1"
RELEASE_ID = "testrelease"
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


def _declarations_document() -> dict[str, dict]:
    return {source_id: _declaration(source_id.split(":")[1]).model_dump(mode="json") for source_id in SOURCE_IDS}


def _rejecting(requirements, spec) -> None:
    raise UnsupportedMachineSpec("The unit lane installs no optional backends")


def _checkers() -> dict:
    return {
        "docker": DockerMachineFactory().check_environment,
        "local": check_local_environment,
        "docker-gvisor": _rejecting,
        "qemu": _rejecting,
        "shellsim": _rejecting,
        "daytona": _rejecting,
        "iris": _rejecting,
    }


def _fit(support: FitSupport) -> BackendFit:
    counts = {support: 2}
    return BackendFit(
        support=support,
        task_counts=counts,
        roles={"agent": dict(counts), "grader": dict(counts)},
        blockers={},
        validation_errors={},
    )


def _backend_row(source_id: str) -> BackendSourceMap:
    fits = {backend: _fit(FitSupport.SUPPORTED) for backend in BACKEND_VALUES}
    return BackendSourceMap(
        source_id=source_id,
        basis="converted_spec_census",
        task_count=2,
        inputs={},
        demands={"agent": {}, "grader": {}},
        resource_ranges={"agent": {}, "grader": {}},
        backends=fits,
        role_backend_assignments={"agent": {}, "grader": {}},
        grading_environment=None,
    )


def _survey(git_revision: str = REVISION) -> BackendSurvey:
    return BackendSurvey(
        catalog_root="test:catalog",
        snapshot_id=SNAPSHOT_ID,
        generated_at="2026-11-23T00:00:00Z",
        git_revision=git_revision,
        schema_sha256="0" * 64,
        runtime_defaults={"agent": RUNTIME, "grader": RUNTIME},
        matrix=production_matrix(_checkers()),
        deployment_limits={},
        deployment_classes={},
        sources=[_backend_row(source_id) for source_id in SOURCE_IDS],
    )


def _identity_survey() -> SourceIdentitySurvey:
    group = DuplicateGroup(
        group_id="fixture-duplicate-group",
        canonical_source_id=SOURCE_A,
        members=[
            DuplicateGroupMember(source_id=SOURCE_A, overlap_with_canonical=1.0),
            DuplicateGroupMember(source_id=SOURCE_B, overlap_with_canonical=None),
        ],
    )
    return SourceIdentitySurvey(
        catalog_root="test:catalog",
        snapshot_id=SNAPSHOT_ID,
        generated_at="2026-11-23T00:00:00Z",
        fingerprints={source_id: None for source_id in SOURCE_IDS},
        reviews=[],
        groups=[group],
        judge=None,
    )


def _curriculum_document(release_id: str, source_id: str) -> dict:
    taxonomy = [
        {"subject_id": "fixture-subject", "node_id": "", "parent_id": None, "kind": "subject", "name": "Fixture Subject"}
    ]
    totals = {metric: 0 for metric in TOTAL_METRICS}
    return {
        "release_id": release_id,
        "taxonomy": taxonomy,
        "sources": [{"source_id": source_id, "alignment": None, "statistics": [], "confidence_statistics": []}],
        "views": {use: {"hierarchy": taxonomy, "totals": totals, "area_counts": []} for use in ("all", "train", "eval")},
    }


def _seal_json(tmp_path: Path, name: str, document: Any) -> ArtifactRef:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _feeds(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "alerts.jsonl",)


def _last_alert_kind(tmp_path: Path) -> str:
    return json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[-1])["kind"]


def _init_repo(root: Path) -> str:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    root.mkdir(parents=True, exist_ok=True)
    git("init", "--quiet")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (root / "x").write_text("one\n")
    git("add", "x")
    git("commit", "--quiet", "-m", "one")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


class _FakeReader:
    def __init__(self, documents: dict[str, Any]) -> None:
        self.documents = documents
        self.requested: list[str] = []

    def get(self, path: str) -> Any:
        self.requested.append(path)
        return self.documents[path]


def _audit_fixture(tmp_path: Path, *, local: bool = True, **overrides) -> SimpleNamespace:
    """Sealed fixture documents plus a fake reader serving a consistent catalog."""
    survey = _survey()
    identity = _identity_survey()
    declarations = _declarations_document()
    group_document = identity.groups[0].model_dump(mode="json")
    backend_rows = {row.source_id: row.model_dump(mode="json") for row in survey.sources}
    served_rows = [
        {
            **declarations[source_id],
            "backend_capabilities": backend_rows[source_id],
            "duplicate_group": group_document,
            "partial_overlaps": [],
        }
        for source_id in SOURCE_IDS
    ]
    served = {
        "sources": served_rows,
        "snapshot_id": SNAPSHOT_ID,
        "created_at": "2026-11-23T00:00:00Z",
        "synced_at": "2026-11-23T00:00:00Z",
        "daytona_snapshot_limit": 4,
    }
    documents: dict[str, Any] = {
        "api/sources": served,
        "api/backends": {**survey.model_dump(mode="json"), "survey_id": "fixture-survey", "matrix": []},
        "api/curriculum/releases": {"releases": [{"release_id": "fixture-curriculum"}]},
        "api/curriculum/invariants?release=fixture-curriculum": _curriculum_document("fixture-curriculum", SOURCE_A),
    }
    if not local:
        documents["api/source-duplicates"] = {"groups": [group_document], "partial_overlaps": {}, "stored_groups": []}
    reader = _FakeReader(documents)
    origins: list[str] = []

    def factory(origin: str) -> _FakeReader:
        origins.append(origin)
        return reader

    inputs = strict_audit.StrictAuditInputs(
        release_id=RELEASE_ID,
        source_revision=REVISION,
        origin="https://fixture.example/a/preview/v/1/",
        output_root=str(tmp_path / "release" / "audit"),
        local=local,
        declarations=_seal_json(tmp_path, "declarations.json", declarations),
        backend=_seal_json(tmp_path, "backend.json", survey.model_dump(mode="json")),
        identity=_seal_json(tmp_path, "identity.json", identity.model_dump(mode="json")),
        expected_source_count=2,
        retained_tasks=4,
        phase="before",
        **overrides,
    )
    return SimpleNamespace(
        inputs=inputs,
        factory=factory,
        reader=reader,
        origins=origins,
        served=served,
        survey=survey,
        declarations=declarations,
    )


def test_strict_audit_pass_seals_receipt(tmp_path):
    fixture = _audit_fixture(tmp_path)
    proof = strict_audit.run_strict_audit(
        fixture.inputs, dry_run=False, feeds=_feeds(tmp_path), reader_factory=fixture.factory
    )
    receipt = json.loads((tmp_path / "release" / "audit" / "strict-audit-before.json").read_text())
    assert receipt["kind"] == f"atlas-{RELEASE_ID}-strict-audit"
    assert receipt["phase"] == "before" and receipt["mismatches"] == {} and receipt["errors"] == []
    assert receipt["source_count"] == 2 and receipt["origin"] == fixture.inputs.origin
    assert fixture.origins == [fixture.inputs.origin]
    by_name = {check["name"]: check for check in receipt["checks"]}
    assert by_name["served-population"]["sources"] == 2
    assert by_name["served-population"]["retained_tasks"] == 4
    assert by_name["identity-duplicates"]["status"] == "skipped-local"
    assert by_name["curriculum-invariants"]["releases"] == 1
    assert by_name["browser-evidence"]["browser_checks"] == strict_audit.SKIPPED_BROWSER
    assert proof["receipt"]["sha256"] == hashlib.sha256(Path(proof["receipt"]["uri"]).read_bytes()).hexdigest()
    assert _last_alert_kind(tmp_path) == strict_audit.alerts.KIND_FINISH


def test_strict_audit_reports_all_mismatches_at_once(tmp_path):
    fixture = _audit_fixture(tmp_path)
    fixture.served["sources"][0]["dataset_revision"] = "999"
    fixture.reader.documents["api/backends"]["deployment_limits"] = {"stale": True}
    with pytest.raises(ValueError) as failure:
        strict_audit.run_strict_audit(
            fixture.inputs, dry_run=False, feeds=_feeds(tmp_path), reader_factory=fixture.factory
        )
    message = str(failure.value)
    assert "Strict audit refused" in message
    assert SOURCE_A in message and "dataset_revision" in message
    assert "api/backends" in message and "deployment_limits" in message
    assert SOURCE_B not in message
    assert _last_alert_kind(tmp_path) == strict_audit.alerts.KIND_FAIL
    assert not (tmp_path / "release" / "audit" / "strict-audit-before.json").exists()


def test_strict_audit_snapshot_mismatch(tmp_path):
    fixture = _audit_fixture(tmp_path)
    fixture.served["snapshot_id"] = "assembly-other-v1"
    with pytest.raises(ValueError, match="served snapshot"):
        strict_audit.run_strict_audit(
            fixture.inputs, dry_run=True, feeds=_feeds(tmp_path), reader_factory=fixture.factory
        )


def test_strict_audit_dry_run_skips_browser_and_writes_nothing(tmp_path):
    fixture = _audit_fixture(tmp_path)
    proof = strict_audit.run_strict_audit(
        fixture.inputs, dry_run=True, feeds=_feeds(tmp_path), reader_factory=fixture.factory
    )
    assert proof["dry_run"] is True
    by_name = {check["name"]: check for check in proof["checks"]}
    assert by_name["browser-evidence"]["browser_checks"] == strict_audit.SKIPPED_BROWSER
    assert not (tmp_path / "release" / "audit").exists()
    assert _last_alert_kind(tmp_path) == strict_audit.alerts.KIND_FINISH


def test_strict_audit_runs_browser_evidence_when_injected(tmp_path, monkeypatch):
    fixture = _audit_fixture(tmp_path, local=False)
    browser_calls, evidence_calls = [], []

    def fake_browser_checks(reader, root, expected, shots, *, local, stored_groups):
        browser_calls.append((reader, root, expected, shots, local, stored_groups))
        return {"scope": "representative"}

    def fake_evidence_sweep(page, root, sources):
        evidence_calls.append((page, root, sources))
        return ["fixture-view"]

    monkeypatch.setattr(strict_audit, "browser_checks", fake_browser_checks)
    monkeypatch.setattr(strict_audit, "evidence_sweep", fake_evidence_sweep)
    page = object()
    proof = strict_audit.run_strict_audit(
        fixture.inputs,
        dry_run=False,
        feeds=_feeds(tmp_path),
        reader_factory=fixture.factory,
        page_factory=lambda: page,
    )
    reader, root, expected, shots, local, stored_groups = browser_calls[0]
    assert reader is fixture.reader and local is False and stored_groups == []
    assert root == f"{fixture.inputs.origin.rstrip('/')}/"
    assert expected is fixture.served and shots == tmp_path / "release" / "audit" / "browser-shots"
    assert evidence_calls == [(page, root, fixture.served["sources"])]
    receipt = json.loads(Path(proof["receipt"]["uri"]).read_text())
    by_name = {check["name"]: check for check in receipt["checks"]}
    assert by_name["browser-evidence"]["browser_checks"] == {"scope": "representative"}
    assert by_name["browser-evidence"]["evidence_views"] == ["fixture-view"]
    assert by_name["identity-duplicates"]["status"] == "checked"


def test_strict_audit_hosted_duplicate_drift_refuses(tmp_path):
    fixture = _audit_fixture(tmp_path, local=False)
    fixture.reader.documents["api/source-duplicates"]["groups"] = []
    with pytest.raises(ValueError, match="served duplicate groups differ"):
        strict_audit.run_strict_audit(
            fixture.inputs, dry_run=True, feeds=_feeds(tmp_path), reader_factory=fixture.factory
        )


def test_strict_audit_input_validation(tmp_path):
    fixture = _audit_fixture(tmp_path)
    with pytest.raises(ValueError, match="phase must be one of"):
        strict_audit.run_strict_audit(
            dataclasses.replace(fixture.inputs, phase="during"),
            dry_run=True,
            feeds=_feeds(tmp_path),
            reader_factory=fixture.factory,
        )
    with pytest.raises(ValueError, match="retained_tasks"):
        strict_audit.run_strict_audit(
            dataclasses.replace(fixture.inputs, retained_tasks=0),
            dry_run=True,
            feeds=_feeds(tmp_path),
            reader_factory=fixture.factory,
        )


def _package_inputs(tmp_path: Path, revision: str, **overrides) -> package_preview.PackagePreviewInputs:
    defaults: dict[str, Any] = {
        "release_id": RELEASE_ID,
        "source_revision": revision,
        "output_root": str(tmp_path / "release" / "package"),
        "oracle_inventory": _seal_json(tmp_path, "oracle-inventory.json", {"schema_version": 2, "rows": []}),
        "source_counts": _seal_json(tmp_path, "source-counts.json", {"sources": []}),
    }
    defaults.update(overrides)
    return package_preview.PackagePreviewInputs(**defaults)


def test_package_preview_write_stages_package_and_seals_receipt(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    revision = _init_repo(repo)
    inputs = _package_inputs(tmp_path, revision)
    staged: list[tuple[Path, Path, Path | None, Path | None]] = []

    def fake_prepare(checkout, output, oracle_inventory=None, source_counts=None):
        staged.append((checkout, Path(output), oracle_inventory, source_counts))
        dist = Path(output) / "dist"
        dist.mkdir(parents=True)
        (dist / "bundled-model-provenance.json").write_text(
            json.dumps({"source_revision": revision, "preview_only": True, "source_counts": {"sources": 2}})
        )

    monkeypatch.setattr(package_preview, "prepare_preview", fake_prepare)
    proof = package_preview.run_package_preview(inputs, dry_run=False, repo_root=repo, feeds=_feeds(tmp_path))
    planned = tmp_path / "release" / "package" / "preview-package"
    assert staged == [(repo, planned, Path(inputs.oracle_inventory.uri), Path(inputs.source_counts.uri))]
    receipt = json.loads((tmp_path / "release" / "package" / "preview-package-receipt.json").read_text())
    assert receipt["kind"] == package_preview.receipt_kind(RELEASE_ID)
    assert receipt["package"]["uri"] == str(planned)
    assert receipt["packaged"]["source_revision"] == revision
    assert receipt["oracle_inventory"]["uri"] == inputs.oracle_inventory.uri
    assert receipt["source_counts"]["uri"] == inputs.source_counts.uri
    assert proof["receipt"]["sha256"] == hashlib.sha256(Path(proof["receipt"]["uri"]).read_bytes()).hexdigest()
    assert _last_alert_kind(tmp_path) == package_preview.alerts.KIND_FINISH


def test_package_preview_dry_run_writes_nothing(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    revision = _init_repo(repo)
    inputs = _package_inputs(tmp_path, revision)

    def failing_prepare(*args, **kwargs):
        raise AssertionError("prepare_preview must not run in a dry run")

    monkeypatch.setattr(package_preview, "prepare_preview", failing_prepare)
    proof = package_preview.run_package_preview(inputs, dry_run=True, repo_root=repo, feeds=_feeds(tmp_path))
    assert proof["package"]["uri"] == str(tmp_path / "release" / "package" / "preview-package")
    assert proof["dry_run"] is True and proof["packaged"] is None
    assert not (tmp_path / "release" / "package").exists()


def test_package_preview_refusals(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    revision = _init_repo(repo)
    repo2 = tmp_path / "repo2"
    repo2_revision = _init_repo(repo2)
    planned = tmp_path / "release" / "package" / "preview-package"
    inputs = _package_inputs(tmp_path, revision)

    planned.mkdir(parents=True)
    with pytest.raises(ValueError, match="already exists"):
        package_preview.run_package_preview(inputs, dry_run=True, repo_root=repo, feeds=_feeds(tmp_path))
    planned.rmdir()

    (repo / "x").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        package_preview.run_package_preview(inputs, dry_run=True, repo_root=repo, feeds=_feeds(tmp_path))

    with pytest.raises(ValueError, match="does not contain the plan's pinned revision"):
        package_preview.run_package_preview(
            dataclasses.replace(inputs, source_revision="d" * 40), dry_run=True, repo_root=repo2, feeds=_feeds(tmp_path)
        )

    with pytest.raises(ValueError, match="outside the source checkout"):
        package_preview.run_package_preview(
            dataclasses.replace(inputs, output_root=str(repo / "dist")),
            dry_run=True,
            repo_root=repo,
            feeds=_feeds(tmp_path),
        )

    # The refs bind the originally sealed bytes; rewriting the document breaks the digest.
    (tmp_path / "oracle-inventory.json").write_text(json.dumps({"schema_version": 3, "rows": []}))
    with pytest.raises(ValueError, match="sealed reference digest mismatch"):
        package_preview.run_package_preview(
            dataclasses.replace(inputs, source_revision=repo2_revision),
            dry_run=True,
            repo_root=repo2,
            feeds=_feeds(tmp_path),
        )


def _preflight_receipt_document(revision: str, production_version: int = 41) -> dict:
    return {
        "version": 2,
        "phase": "preflight",
        "passed": True,
        "source_revision": revision,
        "inputs": {"source_counts_sha256": "0" * 64, "source_counts_json": "{}"},
        "production_version": production_version,
        "preview_base_version": 7,
        "package": {},
        "evidence_before": {},
        "expected_sources": {},
        "browser": {},
        "api_invariants": {},
    }


def _package_receipt(
    tmp_path: Path, revision: str = REVISION, count: int = 2
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef, Path]:
    oracle_ref = _seal_json(tmp_path, "oracle-inventory.json", {"schema_version": 2, "rows": []})
    counts_ref = _seal_json(tmp_path, "source-counts.json", {"sources": []})
    package_dir = tmp_path / "staged" / "preview-package"
    package_dir.mkdir(parents=True)
    document = {
        "kind": package_preview.receipt_kind(RELEASE_ID),
        "release_id": RELEASE_ID,
        "source_revision": revision,
        "package": {"uri": str(package_dir)},
        "packaged": {
            "source_revision": revision,
            "source_counts": {"sources": count},
            "oracle_inventory": {"sources": count},
        },
        "oracle_inventory": {"uri": oracle_ref.uri, "sha256": oracle_ref.sha256},
        "source_counts": {"uri": counts_ref.uri, "sha256": counts_ref.sha256},
    }
    return _seal_json(tmp_path, "staged/package-receipt.json", document), oracle_ref, counts_ref, package_dir


class _FakeChecker:
    def __init__(self, phase: str, revision: str, passed: bool = True) -> None:
        self.phase = phase
        self.revision = revision
        self.passed = passed
        self.calls: list[tuple[Path, list[str], str]] = []

    def __call__(self, repo_root: Path, argv: list[str], receipt_path: str) -> None:
        self.calls.append((repo_root, argv, receipt_path))
        if self.phase is not None:
            Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
            Path(receipt_path).write_text(
                json.dumps({"phase": self.phase, "passed": self.passed, "source_revision": self.revision})
            )


def _release_check_inputs(
    tmp_path: Path, phase: str, name: str = "release-check", **overrides
) -> release_check.ReleaseCheckInputs:
    defaults: dict[str, Any] = {
        "release_id": RELEASE_ID,
        "source_revision": REVISION,
        "phase": phase,
        "output_root": str(tmp_path / "release" / name),
        "service_url": "https://fixture.example",
        "applet_id": "12345678-1234-5678-1234-567812345678",
        "expected_source_count": 2,
        "production_version": 41,
    }
    defaults.update(overrides)
    return release_check.ReleaseCheckInputs(**defaults)


def test_release_check_preflight_dry_run_plans_without_execution(tmp_path):
    package_ref, oracle_ref, counts_ref, package_dir = _package_receipt(tmp_path)
    checker = _FakeChecker("preflight", REVISION)
    inputs = _release_check_inputs(
        tmp_path, "preflight", package=package_ref, oracle_inventory=oracle_ref, source_counts=counts_ref
    )
    proof = release_check.run_release_check(
        inputs, dry_run=True, repo_root=tmp_path, feeds=_feeds(tmp_path), checker=checker
    )
    assert checker.calls == []
    planned = proof["planned_checker"]
    assert planned["module"] == release_check.CHECKER_MODULE
    argv = planned["arguments"]
    assert argv[argv.index("--dist") + 1] == str(package_dir)
    assert argv[argv.index("--source-counts") + 1] == counts_ref.uri
    assert argv[argv.index("--oracle-inventory") + 1] == oracle_ref.uri
    assert argv[argv.index("--production-version") + 1] == "41"
    assert not (tmp_path / "release" / "release-check" / "checker").exists()
    assert not (tmp_path / "release" / "release-check" / "release-check-preflight.json").exists()


def test_release_check_passes_the_bundle_attestation_to_the_checker(tmp_path, monkeypatch):
    """The in-region bundle has no git checkout; the checker takes the attested revision."""
    package_ref, oracle_ref, counts_ref, _ = _package_receipt(tmp_path)
    inputs = _release_check_inputs(
        tmp_path, "preflight", package=package_ref, oracle_inventory=oracle_ref, source_counts=counts_ref
    )
    checker = _FakeChecker("preflight", REVISION)
    run = lambda: release_check.run_release_check(  # noqa: E731
        inputs, dry_run=True, repo_root=tmp_path, feeds=_feeds(tmp_path), checker=checker
    )
    monkeypatch.delenv("ATLAS_RELEASE_SOURCE_REVISION", raising=False)
    assert "--attested-revision" not in run()["planned_checker"]["arguments"]
    monkeypatch.setenv("ATLAS_RELEASE_SOURCE_REVISION", REVISION)
    argv = run()["planned_checker"]["arguments"]
    assert argv[argv.index("--attested-revision") + 1] == REVISION


def test_release_check_preflight_write_delegates_and_seals(tmp_path):
    package_ref, oracle_ref, counts_ref, _ = _package_receipt(tmp_path)
    checker = _FakeChecker("preflight", REVISION)
    inputs = _release_check_inputs(
        tmp_path, "preflight", package=package_ref, oracle_inventory=oracle_ref, source_counts=counts_ref
    )
    repo = tmp_path / "repo"
    proof = release_check.run_release_check(
        inputs, dry_run=False, repo_root=repo, feeds=_feeds(tmp_path), checker=checker
    )
    repo_arg, _argv, receipt_path = checker.calls[0]
    assert repo_arg == repo
    assert receipt_path == str(tmp_path / "release" / "release-check" / "checker" / "preflight.json")
    stage_receipt = json.loads((tmp_path / "release" / "release-check" / "release-check-preflight.json").read_text())
    assert stage_receipt["kind"] == f"atlas-{RELEASE_ID}-release-check"
    assert stage_receipt["phase"] == "preflight" and stage_receipt["passed"] is True
    assert stage_receipt["target"] == {
        "id": "12345678-1234-5678-1234-567812345678",
        "service_url": "https://fixture.example",
    }
    assert proof["receipt"]["sha256"] == hashlib.sha256(Path(proof["receipt"]["uri"]).read_bytes()).hexdigest()
    assert _last_alert_kind(tmp_path) == release_check.alerts.KIND_FINISH


def test_release_check_phase_gating_refusals(tmp_path):
    package_ref, oracle_ref, counts_ref, _ = _package_receipt(tmp_path)
    preflight_ref = _seal_json(tmp_path, "preflight-receipt.json", _preflight_receipt_document(REVISION))
    with pytest.raises(ValueError, match="requires the package, oracle-inventory and source-counts"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "preflight", package=package_ref),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="must not name a published revision"):
        release_check.run_release_check(
            _release_check_inputs(
                tmp_path,
                "preflight",
                package=package_ref,
                oracle_inventory=oracle_ref,
                source_counts=counts_ref,
                published_revision=7,
            ),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="requires the sealed preflight receipt"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="runs the published revision"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", package=package_ref, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="requires the blocking hosted receipt"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "postpublish", preflight_receipt=preflight_ref, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )


def test_release_check_hosted_delegates_with_baseline_and_seals(tmp_path):
    preflight_ref = _seal_json(tmp_path, "preflight-receipt.json", _preflight_receipt_document(REVISION))
    checker = _FakeChecker("hosted", REVISION)
    inputs = _release_check_inputs(tmp_path, "hosted", preflight_receipt=preflight_ref, published_revision=7)
    release_check.run_release_check(
        inputs, dry_run=False, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path), checker=checker
    )
    _, argv, receipt_path = checker.calls[0]
    assert argv[argv.index("--revision") + 1] == "7"
    assert argv[argv.index("--baseline") + 1] == preflight_ref.uri
    assert "--dist" not in argv
    assert receipt_path.endswith("checker/hosted.json")
    stage_receipt = json.loads((tmp_path / "release" / "release-check" / "release-check-hosted.json").read_text())
    assert stage_receipt["phase"] == "hosted" and stage_receipt["passed"] is True
    assert stage_receipt["target"]["id"] == "12345678-1234-5678-1234-567812345678"


def test_release_check_postpublish_plans_full_sweep(tmp_path):
    preflight_ref = _seal_json(tmp_path, "preflight-receipt.json", _preflight_receipt_document(REVISION))
    hosted_ref = _seal_json(
        tmp_path, "hosted-receipt.json", {"phase": "hosted", "passed": True, "source_revision": REVISION}
    )
    checker = _FakeChecker("postpublish", REVISION)
    inputs = _release_check_inputs(
        tmp_path, "postpublish", preflight_receipt=preflight_ref, hosted_receipt=hosted_ref, published_revision=7
    )
    proof = release_check.run_release_check(
        inputs, dry_run=True, repo_root=tmp_path, feeds=_feeds(tmp_path), checker=checker
    )
    assert checker.calls == []
    assert "--full-browser-sweep" in proof["planned_checker"]["arguments"]
    release_check.run_release_check(
        inputs, dry_run=False, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path), checker=checker
    )
    assert "--full-browser-sweep" in checker.calls[0][1]


def test_release_check_refuses_failed_or_stale_prior_receipts(tmp_path):
    preflight_document = _preflight_receipt_document(REVISION)
    failed = _seal_json(tmp_path, "failed-preflight.json", {**preflight_document, "passed": False})
    with pytest.raises(ValueError, match="did not pass"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=failed, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    stale = _seal_json(tmp_path, "stale-preflight.json", {**preflight_document, "source_revision": "d" * 40})
    with pytest.raises(ValueError, match="pins revision"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=stale, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    wrong_phase = _seal_json(tmp_path, "wrong-phase.json", {**preflight_document, "phase": "hosted"})
    with pytest.raises(ValueError, match="prior receipt phase is"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=wrong_phase, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    drifted = _seal_json(
        tmp_path, "drifted-preflight.json", _preflight_receipt_document(REVISION, production_version=42)
    )
    with pytest.raises(ValueError, match="different production version"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=drifted, published_revision=7),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )


def test_release_check_refuses_mismatched_produced_receipts(tmp_path):
    package_ref, oracle_ref, counts_ref, _ = _package_receipt(tmp_path)
    wrong_phase = _FakeChecker("hosted", REVISION)
    with pytest.raises(ValueError, match="checker receipt phase is"):
        release_check.run_release_check(
            _release_check_inputs(
                tmp_path,
                "preflight",
                name="rc-wrong-phase",
                package=package_ref,
                oracle_inventory=oracle_ref,
                source_counts=counts_ref,
            ),
            dry_run=False,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
            checker=wrong_phase,
        )
    failed = _FakeChecker("preflight", REVISION, passed=False)
    with pytest.raises(ValueError, match="preflight check failed"):
        release_check.run_release_check(
            _release_check_inputs(
                tmp_path,
                "preflight",
                name="rc-failed",
                package=package_ref,
                oracle_inventory=oracle_ref,
                source_counts=counts_ref,
            ),
            dry_run=False,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
            checker=failed,
        )
    silent = _FakeChecker(None, REVISION)
    with pytest.raises(ValueError, match="left no receipt"):
        release_check.run_release_check(
            _release_check_inputs(
                tmp_path,
                "preflight",
                name="rc-silent",
                package=package_ref,
                oracle_inventory=oracle_ref,
                source_counts=counts_ref,
            ),
            dry_run=False,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
            checker=silent,
        )
    assert _last_alert_kind(tmp_path) == release_check.alerts.KIND_FAIL
