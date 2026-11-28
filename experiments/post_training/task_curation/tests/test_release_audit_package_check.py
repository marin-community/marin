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
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages import (
    guard,
    live_applet,
    package_inputs,
    package_preview,
    release_check,
    strict_audit,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.sample_qc_artifacts import SampleInventory
from experiments.post_training.task_curation.tests.admission_fixtures import actor_census_input
from experiments.post_training.task_curation.tests.release_audit_fixtures import (
    REVISION,
    SNAPSHOT_ID,
    SOURCE_A,
    SOURCE_B,
    SOURCE_IDS,
    backend_survey,
    declarations_document,
    identity_survey,
)
from infra.marina.apps.rl_data_catalog.prepare_preview import oracle_inventory_payload
from infra.marina.apps.rl_data_catalog.server.curriculum import TOTAL_METRICS
from infra.marina.apps.rl_data_catalog.server.source_counts import (
    ACTORLESS_POLICY_SHA256,
    NO_ORACLE_POLICY_SHA256,
    SourceCounts,
)

RELEASE_ID = "testrelease"


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
    survey = backend_survey()
    identity = identity_survey()
    declarations = declarations_document()
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


@pytest.mark.parametrize("has_population", [False, True])
def test_audit_identity_coverage_distinguishes_excluded_inventory_from_population(tmp_path, has_population):
    fixture = _audit_fixture(tmp_path)
    excluded = fixture.declarations[SOURCE_B]
    excluded["status"] = "Excluded"
    excluded["actor_census_input"] = actor_census_input("fixture", "1", "1", 2) if has_population else None
    fixture.served["sources"][1].update(excluded)
    identity = identity_survey().model_copy(update={"fingerprints": {SOURCE_A: None}, "groups": []})
    for row in fixture.served["sources"]:
        row["duplicate_group"] = None
        row["backend_capabilities"] = strict_audit.served_backend_row(
            next(
                source.model_dump(mode="json", exclude_unset=True)
                for source in fixture.survey.sources
                if source.source_id == row["id"]
            ),
            fixture.declarations[row["id"]],
        )
    mismatches = strict_audit.audit_served_population(
        fixture.served,
        fixture.declarations,
        fixture.survey,
        identity,
        expected_source_count=2,
        retained_tasks=4,
    )
    assert bool(mismatches) is has_population
    if has_population:
        assert SOURCE_B in mismatches["population-coverage"][0]


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


def test_strict_audit_compares_survey_rows_as_sealed(tmp_path):
    """Regression (R40 #53): a survey sealed before ``BackendFit.blocker_reasons`` existed
    is served without it; the audit must not compare the model's default against it."""
    survey = backend_survey().model_dump(mode="json")
    for row in survey["sources"]:
        for fit in row["backends"].values():
            del fit["blocker_reasons"]
    fixture = _audit_fixture(tmp_path)
    rows = {row["source_id"]: row for row in survey["sources"]}
    for served in fixture.served["sources"]:
        served["backend_capabilities"] = rows[served["id"]]
    inputs = dataclasses.replace(fixture.inputs, backend=_seal_json(tmp_path, "backend-presealed.json", survey))
    proof = strict_audit.run_strict_audit(inputs, dry_run=True, feeds=_feeds(tmp_path), reader_factory=fixture.factory)
    assert proof["mismatches"] == {}


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


def _census_row(source_id: str, *, measured: bool = True, status: str = "Available") -> dict:
    if not measured:
        return {
            "source_id": source_id,
            "declaration": {"status": status, "intended_use": "train", "actor_census_input": None},
            "inventory": None,
            "inventory_reference": None,
            "oracle_tasks": 0,
            "static_blocked_tasks": None,
            "static_reason_counts": {},
            "static_audit": None,
        }
    binding = actor_census_input(f"fixture/{source_id}", "a" * 40, "1", 3)
    inventory = SampleInventory(
        source_id=source_id,
        dataset_id=binding.dataset_id,
        dataset_revision=binding.dataset_revision,
        converter_version=binding.converter_version,
        input_artifact="s3://fixture/population",
        task_specs_sha256=binding.task_specs_sha256,
        scanned_task_specs_sha256=binding.task_specs_sha256,
        retained_tasks=3,
        sample_seed=0,
        sample_size=3,
        panel_path="s3://fixture/panel.parquet",
        panel_sha256="b" * 64,
        evidence_url="s3://fixture/inventory.json",
        oracle_tasks=0,
        execution_qualification="not_assessed",
    )
    return {
        "source_id": source_id,
        "declaration": {
            "status": status,
            "intended_use": "train",
            "actor_census_input": binding.model_dump(mode="json"),
        },
        "inventory": inventory.model_dump(mode="json"),
        "inventory_reference": {"uri": f"s3://fixture/{source_id}/summary.json", "sha256": "c" * 64},
        "oracle_tasks": 0,
        "static_blocked_tasks": 1,
        "static_reason_counts": {"fixture": 1},
        "static_audit": {"uri": f"s3://fixture/{source_id}/static.json", "sha256": "d" * 64},
    }


def _census_proof(rows: list[dict]) -> dict:
    return {"sources": len(rows), "results": rows}


CITATIONS = (
    {"uri": "memory://rulings/no-oracle.md", "sha256": NO_ORACLE_POLICY_SHA256},
    {"uri": "memory://rulings/actorless.md", "sha256": ACTORLESS_POLICY_SHA256},
)
PRIOR_COUNTS = {"sources": [{"policy": CITATIONS[0], "actorless_policy": CITATIONS[1]}]}

FIXTURE_CENSUS = [
    _census_row("Marin:first"),
    _census_row("Marin:second"),
    _census_row("Marin:dropped", measured=False, status="Excluded"),
]


def _package_inputs(tmp_path: Path, revision: str, **overrides) -> package_preview.PackagePreviewInputs:
    defaults: dict[str, Any] = {
        "release_id": RELEASE_ID,
        "source_revision": revision,
        "output_root": str(tmp_path / "release" / "package"),
        "census": _seal_json(tmp_path, "census-receipt.json", _census_proof(FIXTURE_CENSUS)),
        "prior_source_counts": _seal_json(tmp_path, "prior-source-counts.json", PRIOR_COUNTS),
        "expected_source_count": len(FIXTURE_CENSUS),
        "derived_root": "memory://release/package-inputs",
    }
    defaults.update(overrides)
    return package_preview.PackagePreviewInputs(**defaults)


def test_package_inputs_derive_from_the_census_and_fail_closed():
    derived = package_inputs.derive_package_inputs(_census_proof(FIXTURE_CENSUS), len(FIXTURE_CENSUS), CITATIONS)
    assert (derived.census_sources, derived.counted_sources) == (3, 2)
    assert derived.unmeasured_sources == ("Marin:dropped",)
    observations = oracle_inventory_payload(derived.oracle_inventory)
    counts = SourceCounts.model_validate_json(derived.source_counts)
    assert [row["source_id"] for row in observations["sources"]] == ["Marin:first", "Marin:second"]
    assert [row.source_id for row in counts.sources] == ["Marin:first", "Marin:second"]
    assert all(row.static_blocked_tasks == 1 for row in counts.sources)
    assert {(row.policy.uri, row.actorless_policy.uri) for row in counts.sources} == {
        (CITATIONS[0]["uri"], CITATIONS[1]["uri"])
    }
    # A plan source missing from the census fails closed.
    with pytest.raises(ValueError, match="the census covers 3 sources"):
        package_inputs.derive_package_inputs(_census_proof(FIXTURE_CENSUS), len(FIXTURE_CENSUS) + 1, CITATIONS)
    # An Available source must have a measured population.
    unmeasured_available = [*FIXTURE_CENSUS[:2], _census_row("Marin:dropped", measured=False)]
    with pytest.raises(ValueError, match="Available source Marin:dropped has no measured census population"):
        package_inputs.derive_package_inputs(_census_proof(unmeasured_available), 3, CITATIONS)


def test_package_preview_write_derives_seals_and_stages_its_inputs(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    revision = _init_repo(repo)
    inputs = _package_inputs(tmp_path, revision)
    derived = package_inputs.derive_package_inputs(_census_proof(FIXTURE_CENSUS), len(FIXTURE_CENSUS), CITATIONS)
    staged: list[tuple[Path, Path, Path | None, Path | None]] = []

    def fake_prepare(checkout, output, oracle_inventory=None, source_counts=None):
        staged.append((checkout, Path(output), oracle_inventory, source_counts))
        assert oracle_inventory.read_bytes() == derived.oracle_inventory
        assert source_counts.read_bytes() == derived.source_counts
        dist = Path(output) / "dist"
        dist.mkdir(parents=True)
        (dist / "bundled-model-provenance.json").write_text(
            json.dumps({"source_revision": revision, "preview_only": True, "source_counts": {"sources": 2}})
        )

    monkeypatch.setattr(package_preview, "prepare_preview", fake_prepare)
    proof = package_preview.run_package_preview(inputs, dry_run=False, repo_root=repo, feeds=_feeds(tmp_path))
    planned = tmp_path / "release" / "package" / "preview-package"
    staged_inputs = tmp_path / "release" / "package" / package_preview.STAGED_INPUTS_DIR_NAME
    assert staged == [(repo, planned, staged_inputs / "oracle-inventory.json", staged_inputs / "source-counts.json")]
    receipt = json.loads((tmp_path / "release" / "package" / "preview-package-receipt.json").read_text())
    assert receipt["kind"] == package_preview.receipt_kind(RELEASE_ID)
    assert receipt["package"]["uri"] == str(planned)
    assert receipt["packaged"]["source_revision"] == revision
    assert (receipt["census_sources"], receipt["counted_sources"]) == (3, 2)
    for name, raw in (("oracle_inventory", derived.oracle_inventory), ("source_counts", derived.source_counts)):
        sealed = StoragePath(receipt[name]["uri"]).read_bytes()
        assert sealed == raw and receipt[name]["sha256"] == hashlib.sha256(raw).hexdigest()
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

    # The census ref binds the originally sealed bytes; rewriting the document breaks the digest.
    (tmp_path / "census-receipt.json").write_text(json.dumps(_census_proof(FIXTURE_CENSUS[:1])))
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
    tmp_path: Path, revision: str = REVISION, count: int = 2, census_sources: int = 2
) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef, Path]:
    oracle_ref = _seal_json(tmp_path, "oracle-inventory.json", {"schema_version": 2, "rows": []})
    counts_ref = _seal_json(tmp_path, "source-counts.json", {"sources": []})
    package_dir = tmp_path / "staged" / "preview-package"
    package_dir.mkdir(parents=True)
    document = {
        "kind": package_preview.receipt_kind(RELEASE_ID),
        "release_id": RELEASE_ID,
        "source_revision": revision,
        "code_revision": revision,
        "package": {"uri": str(package_dir)},
        "census_sources": census_sources,
        "counted_sources": 2,
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


LIVE_VERSION = 39
PUBLISHED_VERSION = LIVE_VERSION + 1


@pytest.fixture(autouse=True)
def _live_preview(monkeypatch):
    monkeypatch.setattr(live_applet, "live_applet", lambda _url, _app: live_applet.LiveApplet(LIVE_VERSION, "public"))
    # The release-check fixtures run the checker from a checkout at REVISION.
    monkeypatch.setattr(release_check, "repo_head", lambda _root: REVISION)


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
        "work_root": str(tmp_path / "work" / name),
    }
    defaults.update(overrides)
    return release_check.ReleaseCheckInputs(**defaults)


def _preflight_stage_receipt(
    tmp_path: Path, name: str = "preflight", production_version: int = 41, **overrides
) -> ArtifactRef:
    """A sealed preflight stage receipt binding its sealed maintained checker receipt."""
    checker = _seal_json(tmp_path, f"{name}-checker.json", _preflight_receipt_document(REVISION, production_version))
    document = {
        "phase": "preflight",
        "passed": True,
        "source_revision": REVISION,
        "checker_receipt": {"uri": checker.uri, "sha256": checker.sha256},
        **overrides,
    }
    return _seal_json(tmp_path, f"{name}-stage.json", document)


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
    staged = tmp_path / "work" / "release-check" / "release-check-preflight" / "inputs"
    assert argv[argv.index("--source-counts") + 1] == str(staged / "source-counts.json")
    assert argv[argv.index("--oracle-inventory") + 1] == str(staged / "oracle-inventory.json")
    assert argv[argv.index("--production-version") + 1] == "41"
    assert not (tmp_path / "release" / "release-check" / "checker").exists()
    assert not (tmp_path / "release" / "release-check" / "release-check-preflight.json").exists()
    assert not (tmp_path / "work").exists()


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
    repo_arg, argv, receipt_path = checker.calls[0]
    assert repo_arg == repo
    # The checker reads and writes local files: digest-verified inputs staged beside its receipt.
    work = tmp_path / "work" / "release-check" / "release-check-preflight"
    assert receipt_path == str(work / "checker" / "preflight.json")
    for flag, ref in (("--source-counts", counts_ref), ("--oracle-inventory", oracle_ref)):
        assert Path(argv[argv.index(flag) + 1]).read_bytes() == Path(ref.uri).read_bytes()
    stage_receipt = json.loads((tmp_path / "release" / "release-check" / "release-check-preflight.json").read_text())
    sealed_checker = tmp_path / "release" / "release-check" / "checker" / "preflight.json"
    assert sealed_checker.read_bytes() == Path(receipt_path).read_bytes()
    assert stage_receipt["checker_receipt"] == {
        "uri": str(sealed_checker),
        "sha256": hashlib.sha256(sealed_checker.read_bytes()).hexdigest(),
    }
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
    preflight_ref = _preflight_stage_receipt(tmp_path)
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
            _release_check_inputs(tmp_path, "hosted"),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="requires the published revision"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=preflight_ref),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    with pytest.raises(ValueError, match="runs the published revision"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", package=package_ref),
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
    preflight_ref = _preflight_stage_receipt(tmp_path)
    checker = _FakeChecker("hosted", REVISION)
    inputs = _release_check_inputs(
        tmp_path, "hosted", preflight_receipt=preflight_ref, published_revision=PUBLISHED_VERSION
    )
    release_check.run_release_check(
        inputs, dry_run=False, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path), checker=checker
    )
    _, argv, receipt_path = checker.calls[0]
    # RELEASE.md: the hosted check verifies the newly published revision, never the
    # preceding live one (R40 #51), and keeps the live mode.
    assert argv[argv.index("--revision") + 1] == str(PUBLISHED_VERSION)
    assert argv[argv.index("--expected-mode") + 1] == "public"
    baseline = Path(argv[argv.index("--baseline") + 1])
    assert (
        baseline == tmp_path / "work" / "release-check" / "release-check-hosted" / "inputs" / "preflight-baseline.json"
    )
    assert json.loads(baseline.read_text()) == _preflight_receipt_document(REVISION)
    assert "--dist" not in argv
    assert receipt_path.endswith("checker/hosted.json")
    stage_receipt = json.loads((tmp_path / "release" / "release-check" / "release-check-hosted.json").read_text())
    assert stage_receipt["phase"] == "hosted" and stage_receipt["passed"] is True
    assert stage_receipt["target"]["id"] == "12345678-1234-5678-1234-567812345678"


def test_release_check_postpublish_diffs_against_the_staged_hosted_receipt(tmp_path):
    preflight_ref = _preflight_stage_receipt(tmp_path)
    hosted_checker = _seal_json(tmp_path, "hosted-checker.json", {"phase": "hosted", "passed": True, "revision": 7})
    hosted_ref = _seal_json(
        tmp_path,
        "hosted-receipt.json",
        {
            "phase": "hosted",
            "passed": True,
            "source_revision": REVISION,
            "checker_receipt": {"uri": hosted_checker.uri, "sha256": hosted_checker.sha256},
        },
    )
    checker = _FakeChecker("postpublish", REVISION)
    inputs = _release_check_inputs(
        tmp_path, "postpublish", preflight_receipt=preflight_ref, hosted_receipt=hosted_ref, published_revision=7
    )
    proof = release_check.run_release_check(
        inputs, dry_run=True, repo_root=tmp_path, feeds=_feeds(tmp_path), checker=checker
    )
    assert checker.calls == []
    arguments = proof["planned_checker"]["arguments"]
    assert "--postpublish" in arguments
    release_check.run_release_check(
        inputs, dry_run=False, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path), checker=checker
    )
    argv = checker.calls[0][1]
    staged = Path(argv[argv.index("--hosted-receipt") + 1])
    assert staged.read_bytes() == Path(hosted_checker.uri).read_bytes()


def test_release_check_refuses_failed_or_stale_prior_receipts(tmp_path):
    failed = _preflight_stage_receipt(tmp_path, "failed", passed=False)
    with pytest.raises(ValueError, match="did not pass"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=failed, published_revision=PUBLISHED_VERSION),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    stale = _preflight_stage_receipt(tmp_path, "stale", source_revision="d" * 40)
    with pytest.raises(ValueError, match="pins revision"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=stale, published_revision=PUBLISHED_VERSION),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    wrong_phase = _preflight_stage_receipt(tmp_path, "wrong-phase", phase="hosted")
    with pytest.raises(ValueError, match="prior receipt phase is"):
        release_check.run_release_check(
            _release_check_inputs(
                tmp_path, "hosted", preflight_receipt=wrong_phase, published_revision=PUBLISHED_VERSION
            ),
            dry_run=True,
            repo_root=tmp_path,
            feeds=_feeds(tmp_path),
        )
    drifted = _preflight_stage_receipt(tmp_path, "drifted", production_version=42)
    with pytest.raises(ValueError, match="different production version"):
        release_check.run_release_check(
            _release_check_inputs(tmp_path, "hosted", preflight_receipt=drifted, published_revision=PUBLISHED_VERSION),
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


def test_package_preview_refuses_the_in_region_bundle(tmp_path, monkeypatch):
    """The package and the release checks share a workstation disk; the bundle has neither."""
    repo = tmp_path / "repo"
    revision = _init_repo(repo)
    monkeypatch.setenv(guard.ATTESTED_REVISION_ENV, revision)
    with pytest.raises(ValueError, match="workstation worktree"):
        package_preview.run_package_preview(
            _package_inputs(tmp_path, revision), dry_run=True, repo_root=repo, feeds=_feeds(tmp_path)
        )


def test_release_check_binds_the_package_to_the_census_and_its_counted_sources(tmp_path):
    for name, overrides, message in (
        ("census", {"census_sources": 3}, "derives from a census of 3 sources, expected 2"),
        ("counted", {"count": 1}, "carries 1 count rows, the census measured 2"),
    ):
        root = tmp_path / name
        root.mkdir()
        package_ref, oracle_ref, counts_ref, _ = _package_receipt(root, **overrides)
        inputs = _release_check_inputs(
            root, "preflight", package=package_ref, oracle_inventory=oracle_ref, source_counts=counts_ref
        )
        with pytest.raises(ValueError, match=message):
            release_check.run_release_check(inputs, dry_run=True, repo_root=root, feeds=_feeds(root))


def test_strict_audit_reads_the_serving_projection_of_sealed_rows():
    """The load serves typed backend Pending, rebinds grading evidence and withholds content."""
    available = {
        "id": "Marin:fixture",
        "status": "Available",
        "grading_environment": {"evidence_url": "s3://fixture/current.json"},
    }
    pending = {"status": "Pending", "stage": strict_audit.PendingStage.BACKEND_PREPARATION_PENDING.value}
    assert strict_audit.served_status_matches(pending, available)
    assert not strict_audit.served_status_matches({"status": "Pending", "stage": "Snapshot repair queued"}, available)
    assert not strict_audit.served_status_matches({"status": "Excluded"}, available)

    survey_row = {"source_id": "Marin:fixture", "basis": "matched", "task_count": 3, "backends": {}}
    survey_row["grading_environment"] = {"evidence_url": "s3://fixture/survey-time.json"}
    served = strict_audit.served_backend_row(survey_row, available)
    assert served["grading_environment"] == available["grading_environment"]
    withheld = {**available, "data_restriction": "private_always"}
    assert strict_audit.served_backend_row(survey_row, withheld) == strict_audit.backend_metadata(served)


def test_package_preview_records_the_fix_forward_checkout_beside_the_cut(tmp_path, monkeypatch):
    """R40: the package builds from a checkout past the plan's cut; provenance pins that checkout."""
    repo = tmp_path / "repo"
    cut = _init_repo(repo)
    (repo / "x").write_text("fix forward\n")
    subprocess.run(["git", "commit", "--quiet", "-am", "fix"], cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
    head = head.stdout.strip()

    def fake_prepare(checkout, output, oracle_inventory=None, source_counts=None):
        dist = Path(output) / "dist"
        dist.mkdir(parents=True)
        (dist / "bundled-model-provenance.json").write_text(json.dumps({"source_revision": head}))

    monkeypatch.setattr(package_preview, "prepare_preview", fake_prepare)
    proof = package_preview.run_package_preview(
        _package_inputs(tmp_path, cut), dry_run=False, repo_root=repo, feeds=_feeds(tmp_path)
    )
    assert (proof["source_revision"], proof["code_revision"]) == (cut, head)
