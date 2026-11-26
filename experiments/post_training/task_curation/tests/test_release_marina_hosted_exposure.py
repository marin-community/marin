# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the marina-release, hosted-check and exposure-probe stages (R40 gate).

These are the final stages of the scripted release pipeline: publish and seal the
preview applet, verify the published revision through its installed API, then probe
the promoted public release anonymously. Every test pins the data-driven contracts:
expectations come from sealed documents, never from counts in the stage source.
"""

import hashlib
import json
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from finestore.rl.status_taxonomy import PendingStage

from experiments.post_training.task_curation.release_stages import (
    exposure_probe,
    hosted_check,
    live_applet,
    marina_release,
)
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef
from experiments.post_training.task_curation.tests import release_audit_fixtures as audit_fixtures
from infra.marina.apps.rl_data_catalog.server.content_access import LICENSE_CONTENT_MARKER, backend_metadata

RELEASE_ID = "testrelease"
APPLET_ID = "12345678-1234-5678-1234-567812345678"
SOURCE_A = "Marin:fixture-a"
SOURCE_B = "Marin:fixture-b"
SERVICE_URL = "https://marina.example.test"
APP_ORIGIN = "https://applets.example.test"
PUBLIC_ROOT = "https://atlas.example.test"
BASE_VERSION = 30
NEW_VERSION = 31
GRANT = {"source_id": SOURCE_A, "upstream_url": "https://upstream.example.test/fixture", "license_id": "cc-by-4.0"}


def _seal_json(tmp_path: Path, name: str, document) -> ArtifactRef:
    path = tmp_path / name
    path.write_text(json.dumps(document))
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


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


def _last_alert(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[-1])


# --- marina_release fixtures -------------------------------------------------------------------


def _release_receipt(tmp_path: Path, name: str, phase: str, revision: str, passed: bool = True) -> ArtifactRef:
    document = {
        "version": 2,
        "phase": phase,
        "passed": passed,
        "gate": "browser-sweep",
        "source_revision": revision,
        "revision": None,
        "started_at": "2026-11-23T00:00:00+00:00",
        "finished_at": "2026-11-23T00:01:00+00:00",
        "target": {"id": APPLET_ID, "origin": APP_ORIGIN, "service_url": SERVICE_URL},
        "production_id": "production-applet",
        "blocking": True,
    }
    return _seal_json(tmp_path, name, document)


def _stage_package(tmp_path: Path) -> tuple[Path, list[dict]]:
    """A minimal packaged applet directory plus its provenance module records."""
    root = tmp_path / "preview-package"
    (root / "dist").mkdir(parents=True, exist_ok=True)
    (root / "applet.toml").write_text('title = "Atlas Preview"\ndescription = "Fixture preview applet"\n')
    (root / "dist" / "index.html").write_text("<!doctype html><html><body>fixture</body></html>\n")
    bundled = root / "server" / "bundled"
    bundled.mkdir(parents=True, exist_ok=True)
    body = b"# fixture bundled module\n"
    (bundled / "models.py").write_bytes(body)
    modules = [
        {
            "source": "lib/fixture/models.py",
            "source_sha256": hashlib.sha256(body).hexdigest(),
            "packaged": "server/bundled/models.py",
            "packaged_sha256": hashlib.sha256(body).hexdigest(),
            "transformation": "Unchanged",
        }
    ]
    return root, modules


def _package_receipt(tmp_path: Path, name: str, package_root: Path, modules: list[dict], revision: str) -> ArtifactRef:
    document = {
        "kind": f"atlas-{RELEASE_ID}-preview-package",
        "release_id": RELEASE_ID,
        "source_revision": revision,
        "code_revision": revision,
        "package": {"uri": str(package_root)},
        "packaged": {"source_revision": revision, "preview_only": True, "modules": modules},
        "oracle_inventory": None,
        "source_counts": None,
        "dry_run": False,
    }
    return _seal_json(tmp_path, name, document)


def _prior_manifest(tmp_path: Path, name: str = "prior-manifest.json", revision: int = BASE_VERSION) -> ArtifactRef:
    return _seal_json(
        tmp_path,
        name,
        {
            "kind": "atlas-priorrelease-deployment-manifest",
            "release_id": "priorrelease",
            "revision": revision,
            "app_id": APPLET_ID,
            "mode": "private",
            "url": f"{SERVICE_URL}/a/{APPLET_ID}/v/{revision}/",
            "source_sha": "b" * 40,
            "sealed_at": "2026-11-22T00:00:00+00:00",
        },
    )


@pytest.fixture(autouse=True)
def _live_preview(monkeypatch):
    """The preview is live at the base revision the hosted check recorded."""
    monkeypatch.setattr(live_applet, "live_applet", lambda _url, _app: live_applet.LiveApplet(BASE_VERSION, "private"))


def _marina_inputs(tmp_path: Path, revision: str, **overrides) -> marina_release.MarinaReleaseInputs:
    package_root, modules = _stage_package(tmp_path)
    defaults = dict(
        release_id=RELEASE_ID,
        source_revision=revision,
        applet_id=APPLET_ID,
        service_url=SERVICE_URL,
        package=_package_receipt(tmp_path, "package-receipt.json", package_root, modules, revision),
        preflight_receipt=_release_receipt(tmp_path, "preflight-receipt.json", "preflight", revision),
        hosted_receipt=_release_receipt(tmp_path, "hosted-receipt.json", "hosted", revision),
        prior_manifest=_prior_manifest(tmp_path),
        base_version=BASE_VERSION,
        access_mode="private",
        output_root=str(tmp_path / "release" / "marina"),
    )
    defaults.update(overrides)
    return marina_release.MarinaReleaseInputs(**defaults)


def _fake_publish(calls: list[bytes]):
    def publish(payload: bytes) -> dict:
        calls.append(payload)
        return {
            "id": APPLET_ID,
            "version": NEW_VERSION,
            "url": f"{SERVICE_URL}/a/{APPLET_ID}/v/{NEW_VERSION}/",
            "mode": "private",
        }

    return publish


def test_marina_release_dry_run_verifies_receipts_with_zero_writes(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _marina_inputs(tmp_path, revision)
    proof = marina_release.run_marina_release(inputs, dry_run=True, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["sql_writes"] == 0
    assert proof["base_version"] == BASE_VERSION and proof["prior_manifest"]["uri"].endswith("prior-manifest.json")
    assert not (tmp_path / "release" / "marina").exists() or not any((tmp_path / "release" / "marina").iterdir())
    assert _last_alert(tmp_path)["kind"] == marina_release.alerts.KIND_FINISH


def test_marina_release_write_seals_release_and_manifest_documents(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _marina_inputs(tmp_path, revision)
    calls: list[bytes] = []
    proof = marina_release.run_marina_release(
        inputs, dry_run=False, repo_root=tmp_path / "repo", publish=_fake_publish(calls), feeds=_feeds(tmp_path)
    )
    assert len(calls) == 1 and calls[0]
    assert proof["publish"] == {
        "id": APPLET_ID,
        "version": NEW_VERSION,
        "url": f"{SERVICE_URL}/a/{APPLET_ID}/v/{NEW_VERSION}/",
        "mode": "private",
    }
    release = json.loads((tmp_path / "release" / "marina" / "release-v1.json").read_text())
    manifest = json.loads((tmp_path / "release" / "marina" / "deployment-manifest.json").read_text())
    assert release["publish"] == proof["publish"] and release["dry_run"] is False
    assert manifest["kind"] == f"atlas-{RELEASE_ID}-deployment-manifest"
    assert manifest["revision"] == NEW_VERSION and manifest["base_version"] == BASE_VERSION
    assert manifest["app_id"] == APPLET_ID and manifest["mode"] == "private" and manifest["source_sha"] == revision
    assert set(manifest["receipts"]) == {"preflight", "hosted", "package"}
    assert manifest["prior_manifest"]["sha256"] == inputs.prior_manifest.sha256
    assert proof["manifest"]["uri"].endswith("deployment-manifest.json")
    assert _last_alert(tmp_path)["kind"] == marina_release.alerts.KIND_FINISH


def test_marina_release_continues_the_live_public_revision_and_refuses_a_moved_one(tmp_path, monkeypatch):
    """R40: publish against the recorded live revision in the live mode; never change the mode."""
    revision = _init_repo(tmp_path / "repo")
    monkeypatch.setattr(live_applet, "live_applet", lambda _url, _app: live_applet.LiveApplet(BASE_VERSION, "public"))
    inputs = _marina_inputs(tmp_path, revision, access_mode="public", prior_manifest=None)
    seen: list[dict] = []

    def publish(payload: bytes) -> dict:
        seen.append({"mode": inputs.access_mode, "base": inputs.base_version})
        return {
            "id": APPLET_ID,
            "version": NEW_VERSION,
            "url": f"{SERVICE_URL}/a/{APPLET_ID}/v/{NEW_VERSION}/",
            "mode": "public",
        }

    proof = marina_release.run_marina_release(
        inputs, dry_run=False, repo_root=tmp_path / "repo", publish=publish, feeds=_feeds(tmp_path)
    )
    assert seen == [{"mode": "public", "base": BASE_VERSION}]
    assert proof["access_mode"] == "public" and proof["publish"]["mode"] == "public"
    monkeypatch.setattr(
        live_applet, "live_applet", lambda _url, _app: live_applet.LiveApplet(BASE_VERSION + 1, "public")
    )
    moved = _marina_inputs(tmp_path / "moved", revision, access_mode="public", prior_manifest=None)
    with pytest.raises(ValueError, match="moved to v31"):
        marina_release.run_marina_release(
            moved, dry_run=False, repo_root=tmp_path / "repo", publish=publish, feeds=_feeds(tmp_path)
        )
    assert len(seen) == 1


def test_marina_release_refuses_rerun_without_republishing(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _marina_inputs(tmp_path, revision)
    calls: list[bytes] = []
    marina_release.run_marina_release(
        inputs, dry_run=False, repo_root=tmp_path / "repo", publish=_fake_publish(calls), feeds=_feeds(tmp_path)
    )
    with pytest.raises(ValueError, match="already exists"):
        marina_release.run_marina_release(
            inputs, dry_run=False, repo_root=tmp_path / "repo", publish=_fake_publish(calls), feeds=_feeds(tmp_path)
        )
    assert len(calls) == 1
    assert _last_alert(tmp_path)["kind"] == marina_release.alerts.KIND_FAIL


def test_marina_release_refusals(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    inputs = _marina_inputs(tmp_path, revision)
    with pytest.raises(ValueError, match="positive"):
        marina_release.run_marina_release(
            replace(inputs, base_version=0), dry_run=True, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path)
        )
    # A failed preflight receipt.
    failed = _release_receipt(tmp_path, "failed-preflight.json", "preflight", revision, passed=False)
    with pytest.raises(ValueError, match="did not pass"):
        marina_release.run_marina_release(
            _marina_inputs(tmp_path, revision, preflight_receipt=failed),
            dry_run=True,
            repo_root=tmp_path / "repo",
            feeds=_feeds(tmp_path),
        )
    # A receipt whose bytes no longer match its sealed digest.
    tampered = _release_receipt(tmp_path, "tampered-receipt.json", "preflight", revision)
    (tmp_path / "tampered-receipt.json").write_text(json.dumps({"phase": "preflight", "passed": True}))
    with pytest.raises(ValueError, match="digest mismatch"):
        marina_release.run_marina_release(
            _marina_inputs(tmp_path, revision, preflight_receipt=tampered),
            dry_run=True,
            repo_root=tmp_path / "repo",
            feeds=_feeds(tmp_path),
        )
    # A receipt from the wrong release-check phase.
    wrong_phase = _release_receipt(tmp_path, "wrong-phase.json", "hosted", revision)
    with pytest.raises(ValueError, match="records phase"):
        marina_release.run_marina_release(
            _marina_inputs(tmp_path, revision, preflight_receipt=wrong_phase),
            dry_run=True,
            repo_root=tmp_path / "repo",
            feeds=_feeds(tmp_path),
        )
    # Prior-manifest continuity: the live revision and the applet must match.
    stale_prior = _prior_manifest(tmp_path, "stale-prior.json", revision=BASE_VERSION - 1)
    with pytest.raises(ValueError, match="base_version"):
        marina_release.run_marina_release(
            _marina_inputs(tmp_path, revision, prior_manifest=stale_prior),
            dry_run=True,
            repo_root=tmp_path / "repo",
            feeds=_feeds(tmp_path),
        )
    foreign_prior = _seal_json(
        tmp_path,
        "foreign-prior.json",
        {
            "kind": "atlas-priorrelease-deployment-manifest",
            "revision": BASE_VERSION,
            "app_id": "87654321-4321-8765-4321-87654321fedcba",
            "mode": "private",
        },
    )
    with pytest.raises(ValueError, match="different applet"):
        marina_release.run_marina_release(
            _marina_inputs(tmp_path, revision, prior_manifest=foreign_prior),
            dry_run=True,
            repo_root=tmp_path / "repo",
            feeds=_feeds(tmp_path),
        )
    # A dirty checkout.
    (tmp_path / "repo" / "marker.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        marina_release.run_marina_release(inputs, dry_run=True, repo_root=tmp_path / "repo", feeds=_feeds(tmp_path))


def test_marina_release_refuses_packaged_module_drift(tmp_path):
    revision = _init_repo(tmp_path / "repo")
    package_root, modules = _stage_package(tmp_path)
    package = _package_receipt(tmp_path, "package-receipt.json", package_root, modules, revision)
    inputs = _marina_inputs(tmp_path, revision, package=package)
    (package_root / "server" / "bundled" / "models.py").write_bytes(b"# drifted module\n")
    calls: list[bytes] = []
    with pytest.raises(ValueError, match="differs from the package provenance"):
        marina_release.run_marina_release(
            inputs, dry_run=False, repo_root=tmp_path / "repo", publish=_fake_publish(calls), feeds=_feeds(tmp_path)
        )
    assert calls == []


# --- hosted_check fixtures ---------------------------------------------------------------------

CODE_REVISION = "d" * 40
HOSTED_ROOT = f"{APP_ORIGIN}/a/{APPLET_ID}/v/{NEW_VERSION}/"


def _deployment_manifest() -> dict:
    return {
        "kind": f"atlas-{RELEASE_ID}-deployment-manifest",
        "release_id": RELEASE_ID,
        "revision": NEW_VERSION,
        "base_version": BASE_VERSION,
        "url": HOSTED_ROOT,
        "app_id": APPLET_ID,
        "mode": "public",
        "source_sha": audit_fixtures.REVISION,
        "code_revision": CODE_REVISION,
        "service_url": SERVICE_URL,
        "package": {"root": "fixture:package"},
        "receipts": {"preflight": {}, "hosted": {}, "package": {}},
        "sealed_at": "2026-11-23T00:00:00+00:00",
    }


def _served_documents(declarations: dict, survey, identity) -> dict[str, dict]:
    """What the installed API serves for the sealed fixture documents (the strict audit's projection)."""
    group = identity.groups[0].model_dump(mode="json")
    backend_rows = {row.source_id: row.model_dump(mode="json") for row in survey.sources}
    sources = [
        {
            **declarations[source_id],
            "backend_capabilities": backend_rows[source_id],
            "duplicate_group": group,
            "partial_overlaps": [],
        }
        for source_id in audit_fixtures.SOURCE_IDS
    ]
    return {
        hosted_check.SOURCES_PATH: {"sources": sources, "snapshot_id": audit_fixtures.SNAPSHOT_ID},
        hosted_check.BACKENDS_PATH: {**survey.model_dump(mode="json"), "survey_id": "fixture-survey", "matrix": []},
        hosted_check.DUPLICATES_PATH: {"groups": [group], "partial_overlaps": {}, "stored_groups": []},
        hosted_check.PROVENANCE_PATH: {"source_revision": CODE_REVISION, "modules": []},
    }


def _hosted_fixture(tmp_path: Path, **overrides) -> tuple[hosted_check.HostedCheckInputs, dict[str, dict]]:
    manifest = overrides.pop("manifest", _deployment_manifest())
    declarations = audit_fixtures.declarations_document()
    survey = audit_fixtures.backend_survey()
    identity = audit_fixtures.identity_survey()
    inputs = hosted_check.HostedCheckInputs(
        release_id=RELEASE_ID,
        source_revision=audit_fixtures.REVISION,
        revision=NEW_VERSION,
        root=HOSTED_ROOT,
        declarations=_seal_json(tmp_path, "declarations.json", declarations),
        backend=_seal_json(tmp_path, "backend.json", survey.model_dump(mode="json")),
        identity=_seal_json(tmp_path, "identity.json", identity.model_dump(mode="json")),
        expected_source_count=2,
        retained_tasks=4,
        release_manifest=_seal_json(tmp_path, "deployment-manifest.json", manifest),
        output_root=str(tmp_path / "release" / "hosted"),
        **overrides,
    )
    return inputs, _served_documents(declarations, survey, identity)


class _FakeApi:
    def __init__(self, documents: dict[str, dict]) -> None:
        self.documents = documents
        self.requested: list[str] = []

    def __call__(self, path: str) -> dict:
        self.requested.append(path)
        return self.documents[path]


class _FakeBrowserPage:
    def __init__(self, status: int = 200, errors: list[str] | None = None) -> None:
        self.status = status
        self.errors = errors or []
        self.visited: list[str] = []

    def goto(self, url: str):
        self.visited.append(url)
        return SimpleNamespace(status=self.status)

    def javascript_errors(self) -> list[str]:
        return list(self.errors)


def test_hosted_check_dry_run_verifies_documents_without_requests(tmp_path):
    inputs, _ = _hosted_fixture(tmp_path)
    api = _FakeApi({})
    proof = hosted_check.run_hosted_check(inputs, dry_run=True, api_client=api, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["requests"] == 0 and api.requested == []
    assert proof["browser_checks"] == "skipped"
    assert not (tmp_path / "release" / "hosted").exists() or not any((tmp_path / "release" / "hosted").iterdir())
    assert _last_alert(tmp_path)["kind"] == hosted_check.alerts.KIND_FINISH


def test_hosted_check_seals_receipt_on_a_clean_pass(tmp_path):
    inputs, documents = _hosted_fixture(tmp_path)
    page = _FakeBrowserPage()
    proof = hosted_check.run_hosted_check(
        inputs,
        dry_run=False,
        api_client=_FakeApi(documents),
        page_factory=lambda: nullcontext(page),
        feeds=_feeds(tmp_path),
    )
    assert proof["mismatches"] == {} and proof["browser_checks"] == "executed"
    assert page.visited == [HOSTED_ROOT]
    receipt = json.loads((tmp_path / "release" / "hosted" / "hosted-check.json").read_text())
    assert receipt["kind"] == f"atlas-{RELEASE_ID}-hosted-check"
    assert receipt["revision"] == NEW_VERSION and receipt["mismatches"] == {}
    assert receipt["origin"] == HOSTED_ROOT
    assert [check["check"] for check in receipt["checks"]] == [
        hosted_check.SOURCES_PATH,
        hosted_check.BACKENDS_PATH,
        hosted_check.DUPLICATES_PATH,
        hosted_check.PROVENANCE_PATH,
        "browser",
    ]
    assert proof["receipt"]["uri"].endswith("hosted-check.json")
    assert _last_alert(tmp_path)["kind"] == hosted_check.alerts.KIND_FINISH


def test_hosted_check_accepts_the_serving_projection_of_withheld_and_downgraded_sources(tmp_path):
    """Regression (R40 #49): the API serves a withheld source's backend metadata only, the
    load's typed backend Pending downgrade, and the declaration's grading evidence; the old
    field-by-field comparison with the raw sealed documents refused every such source."""
    declarations = audit_fixtures.declarations_document()
    declarations[audit_fixtures.SOURCE_A]["exclusion_claims"] = [{"reason": "license-ban"}]
    survey = audit_fixtures.backend_survey()
    identity = audit_fixtures.identity_survey()
    inputs, _ = _hosted_fixture(tmp_path)
    inputs = replace(inputs, declarations=_seal_json(tmp_path, "declarations-withheld.json", declarations))
    documents = _served_documents(declarations, survey, identity)
    rows = {row["id"]: row for row in documents[hosted_check.SOURCES_PATH]["sources"]}
    rows[audit_fixtures.SOURCE_A]["backend_capabilities"] = backend_metadata(
        rows[audit_fixtures.SOURCE_A]["backend_capabilities"]
    )
    rows[audit_fixtures.SOURCE_A]["notes"] = LICENSE_CONTENT_MARKER
    rows[audit_fixtures.SOURCE_B].update(status="Pending", stage=PendingStage.BACKEND_VERIFICATION_PENDING.value)
    proof = hosted_check.run_hosted_check(inputs, dry_run=False, api_client=_FakeApi(documents), feeds=_feeds(tmp_path))
    assert proof["mismatches"] == {}


def test_hosted_check_reports_every_source_mismatch_at_once(tmp_path):
    inputs, documents = _hosted_fixture(tmp_path)
    served = documents[hosted_check.SOURCES_PATH]
    served["sources"][1]["status"] = "Excluded"
    served["sources"].append({**served["sources"][0], "id": "Marin:fixture-c"})
    with pytest.raises(ValueError) as failure:
        hosted_check.run_hosted_check(inputs, dry_run=False, api_client=_FakeApi(documents), feeds=_feeds(tmp_path))
    message = str(failure.value)
    assert message.startswith("Hosted check failed with")
    assert f"{audit_fixtures.SOURCE_B}: served status differs" in message and "Marin:fixture-c" in message
    assert not (tmp_path / "release" / "hosted" / "hosted-check.json").exists()
    assert _last_alert(tmp_path)["kind"] == hosted_check.alerts.KIND_FAIL


def test_hosted_check_refuses_a_served_applet_from_another_checkout(tmp_path):
    inputs, documents = _hosted_fixture(tmp_path)
    documents[hosted_check.PROVENANCE_PATH] = {"source_revision": "e" * 40}
    with pytest.raises(ValueError, match="expected the published packaging checkout"):
        hosted_check.run_hosted_check(inputs, dry_run=False, api_client=_FakeApi(documents), feeds=_feeds(tmp_path))


def test_hosted_check_refuses_backend_document_drift(tmp_path):
    inputs, documents = _hosted_fixture(tmp_path)
    documents[hosted_check.BACKENDS_PATH]["snapshot_id"] = "other-snapshot"
    with pytest.raises(ValueError, match="served snapshot_id differs from the sealed backend survey"):
        hosted_check.run_hosted_check(inputs, dry_run=False, api_client=_FakeApi(documents), feeds=_feeds(tmp_path))


def test_hosted_check_dry_run_refuses_manifest_binding_drift(tmp_path):
    inputs, _ = _hosted_fixture(tmp_path, manifest=dict(_deployment_manifest(), revision=NEW_VERSION + 1))
    with pytest.raises(ValueError, match="binds revision"):
        hosted_check.run_hosted_check(inputs, dry_run=True, feeds=_feeds(tmp_path))


def test_hosted_check_browser_gate_refuses_javascript_errors(tmp_path):
    inputs, documents = _hosted_fixture(tmp_path)
    page = _FakeBrowserPage(errors=["boom"])
    with pytest.raises(ValueError, match="JavaScript error"):
        hosted_check.run_hosted_check(
            inputs,
            dry_run=False,
            api_client=_FakeApi(documents),
            page_factory=lambda: nullcontext(page),
            feeds=_feeds(tmp_path),
        )


# --- exposure_probe fixtures -------------------------------------------------------------------


def _withheld_routes() -> dict[str, tuple[int, dict]]:
    prefix = f"/v/{NEW_VERSION}/api/"
    row = {
        "data_restriction": "license-exclusion",
        "content_withheld": True,
        "withheld_reason": "license exclusion",
        "content": None,
        "marker": "withheld",
    }
    return {
        prefix + "difficulty-attempts": (200, dict(row)),
        prefix + "difficulty-rollout": (200, dict(row)),
        prefix + "curriculum/tasks": (200, {"tasks": [{"task_id": "task-1", "task_preview": LICENSE_CONTENT_MARKER}]}),
        prefix + "curriculum/task": (200, {"task": {"task_id": "task-1", "content_withheld": True}}),
        prefix + "assessment": (200, {"report": {"evidence": {"items": [{"task_id": "task-1"}]}}}),
        prefix + "evidence": (200, {"packet": {"content_withheld": True}, "raw_review": {"content_withheld": True}}),
    }


class _Response:
    def __init__(self, status_code: int, document: dict) -> None:
        self.status_code = status_code
        self._document = document

    def json(self) -> dict:
        return self._document


class _FakeHttp:
    def __init__(self, routes: dict[str, tuple[int, dict]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, headers: dict[str, str]):
        self.calls.append((url, dict(headers)))
        status, document = self.routes[urlsplit(url).path]
        return _Response(status, document)


class _FakeDisplayPage:
    def __init__(self, fail_text: bool = False) -> None:
        self.visited: list[str] = []
        self.links: list[str] = []
        self.texts: list[str] = []
        self.fail_text = fail_text

    def goto(self, url: str):
        self.visited.append(url)
        return SimpleNamespace(status=200)

    def visible_link(self, href: str) -> None:
        self.links.append(href)

    def body_contains(self, text: str) -> None:
        if self.fail_text:
            raise AssertionError("license text not rendered")
        self.texts.append(text)


def _exposure_inputs(tmp_path: Path, **overrides) -> exposure_probe.ExposureProbeInputs:
    defaults = dict(
        release_id=RELEASE_ID,
        revision=NEW_VERSION,
        root=PUBLIC_ROOT,
        withheld_sources=(SOURCE_A, SOURCE_B),
        publication_grants=(GRANT,),
        input_check_receipt=None,
        output_root=str(tmp_path / "release" / "exposure"),
    )
    defaults.update(overrides)
    return exposure_probe.ExposureProbeInputs(**defaults)


def test_exposure_probe_dry_run_makes_no_requests(tmp_path):
    http = _FakeHttp(_withheld_routes())
    inputs = _exposure_inputs(tmp_path)
    proof = exposure_probe.run_exposure_probe(inputs, dry_run=True, http=http, feeds=_feeds(tmp_path))
    assert proof["dry_run"] is True and proof["requests"] == 0 and http.calls == []
    assert proof["browser_checks"] == "skipped"
    assert not (tmp_path / "release" / "exposure").exists() or not any((tmp_path / "release" / "exposure").iterdir())
    assert _last_alert(tmp_path)["kind"] == exposure_probe.alerts.KIND_FINISH


def test_exposure_probe_passes_anonymously_and_seals_receipt(tmp_path):
    http = _FakeHttp(_withheld_routes())
    inputs = _exposure_inputs(tmp_path)
    proof = exposure_probe.run_exposure_probe(inputs, dry_run=False, http=http, feeds=_feeds(tmp_path))
    assert proof["status"] == "PASS" and proof["anonymous"] is True
    assert proof["license_withheld_sources"] == 2 and proof["content_route_checks"] == 12
    assert proof["public_display_approvals"] == 1
    receipt = json.loads((tmp_path / "release" / "exposure" / "exposure-probe.json").read_text())
    assert receipt["kind"] == f"atlas-{RELEASE_ID}-anonymous-publication-exposure"
    assert receipt["status"] == "PASS" and receipt["revision"] == NEW_VERSION
    assert receipt["root"] == f"{PUBLIC_ROOT}/v/{NEW_VERSION}/"
    assert receipt["anonymous"] is True
    assert receipt["public_display_approvals"] == [
        {"source_id": SOURCE_A, "license_id": "cc-by-4.0", "upstream_url": GRANT["upstream_url"], "status": "skipped"}
    ]
    assert receipt["content_route_checks"][0]["content_withheld"] is True
    assert _last_alert(tmp_path)["kind"] == exposure_probe.alerts.KIND_FINISH


def test_exposure_probe_never_sends_credentials(tmp_path):
    http = _FakeHttp(_withheld_routes())
    inputs = _exposure_inputs(tmp_path, withheld_sources=(SOURCE_A,))
    exposure_probe.run_exposure_probe(inputs, dry_run=False, http=http, feeds=_feeds(tmp_path))
    assert http.calls
    for url, headers in http.calls:
        assert headers == {}
        assert "Authorization" not in headers and "Cookie" not in headers
        assert url.startswith(f"{PUBLIC_ROOT}/v/{NEW_VERSION}/")


def test_exposure_probe_refuses_leaked_content_naming_the_route(tmp_path):
    routes = _withheld_routes()
    routes[f"/v/{NEW_VERSION}/api/curriculum/task"] = (
        200,
        {"task": {"task_id": "task-1", "content_withheld": False, "task_preview": "secret content"}},
    )
    http = _FakeHttp(routes)
    inputs = _exposure_inputs(tmp_path)
    with pytest.raises(ValueError, match=f"api/curriculum/task for {SOURCE_A}"):
        exposure_probe.run_exposure_probe(inputs, dry_run=False, http=http, feeds=_feeds(tmp_path))
    assert not (tmp_path / "release" / "exposure" / "exposure-probe.json").exists()
    assert _last_alert(tmp_path)["kind"] == exposure_probe.alerts.KIND_FAIL


def test_exposure_probe_refuses_unexpected_response_keys(tmp_path):
    routes = _withheld_routes()
    routes[f"/v/{NEW_VERSION}/api/difficulty-attempts"] = (
        200,
        {"content_withheld": True, "content": "leaked", "task_preview": "leaked too"},
    )
    http = _FakeHttp(routes)
    inputs = _exposure_inputs(tmp_path)
    with pytest.raises(ValueError, match="unexpected keys"):
        exposure_probe.run_exposure_probe(inputs, dry_run=False, http=http, feeds=_feeds(tmp_path))


def test_exposure_probe_verifies_public_display(tmp_path):
    http = _FakeHttp(_withheld_routes())
    page = _FakeDisplayPage()
    inputs = _exposure_inputs(tmp_path, withheld_sources=(SOURCE_A,))
    exposure_probe.run_exposure_probe(
        inputs, dry_run=False, http=http, page_factory=lambda: nullcontext(page), feeds=_feeds(tmp_path)
    )
    receipt = json.loads((tmp_path / "release" / "exposure" / "exposure-probe.json").read_text())
    assert receipt["public_display_approvals"] == [
        {
            "source_id": SOURCE_A,
            "license_id": "cc-by-4.0",
            "upstream_url": GRANT["upstream_url"],
            "status": "verified",
        }
    ]
    assert page.visited == [f"{PUBLIC_ROOT}/v/{NEW_VERSION}/source.html#source_id=Marin%3Afixture-a"]
    assert page.links == [GRANT["upstream_url"]]
    assert page.texts == ["CC-BY-4.0"]


def test_exposure_probe_refuses_failed_public_display(tmp_path):
    http = _FakeHttp(_withheld_routes())
    page = _FakeDisplayPage(fail_text=True)
    inputs = _exposure_inputs(tmp_path, withheld_sources=(SOURCE_A,))
    with pytest.raises(ValueError, match=f"public display for {SOURCE_A}"):
        exposure_probe.run_exposure_probe(
            inputs, dry_run=False, http=http, page_factory=lambda: nullcontext(page), feeds=_feeds(tmp_path)
        )
    assert _last_alert(tmp_path)["kind"] == exposure_probe.alerts.KIND_FAIL


def test_exposure_probe_refuses_input_check_receipt_mismatch(tmp_path):
    receipt = _seal_json(tmp_path, "input-check.json", {"withheld_sources": [SOURCE_A]})
    inputs = _exposure_inputs(tmp_path, input_check_receipt=receipt)
    with pytest.raises(ValueError, match="withhold"):
        exposure_probe.run_exposure_probe(inputs, dry_run=True, feeds=_feeds(tmp_path))


def test_exposure_probe_refuses_empty_typed_lists(tmp_path):
    with pytest.raises(ValueError, match="withheld_sources must be non-empty"):
        exposure_probe.run_exposure_probe(
            _exposure_inputs(tmp_path, withheld_sources=()), dry_run=True, feeds=_feeds(tmp_path)
        )
    with pytest.raises(ValueError, match="publication_grants must be non-empty"):
        exposure_probe.run_exposure_probe(
            _exposure_inputs(tmp_path, publication_grants=()), dry_run=True, feeds=_feeds(tmp_path)
        )
