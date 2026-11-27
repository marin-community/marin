# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hosted check: verify the published preview through its installed API.

One maintained module replaces the per-release hosted verification scripts (see
INVENTORY.md). The verifier is data-driven: every expectation comes from the typed
inputs and the sealed documents they reference — the declaration set, the backend
survey, the identity survey and the deployment manifest — never from counts or
shapes hard-coded here. All mismatches are collected and reported at once, like the
admission check.

The served sources, backend document and duplicate survey are compared with the
sealed documents through the strict audit's serving projection (the same functions
``strict_audit`` runs), so the hosted check accepts exactly what the API serves for
withheld sources, rebound grading evidence and the load's typed backend Pending
stages, and nothing else. The served applet provenance
(``bundled-model-provenance.json``) must pin the packaging checkout the deployment
manifest records, and the typed root must be the manifest's published URL.

The API client is injected; the default binds the maintained ``marina_request`` to
the typed root (IAP bearer token handled there). The graph supplies a client bound
to the applet origin when the release needs a shared transport. The browser gate is
also injected (the graph supplies a configured Playwright page adapter); without it
the receipt records ``browser_checks: "skipped"`` honestly.
"""

import argparse
import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from marina.client import marina_request
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.backend_requirements import BackendSurvey
from taskcompendium.source_fingerprints import SourceIdentitySurvey

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages import strict_audit
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

HOSTED_CHECK_SUFFIX = "hosted-check"
DEPLOYMENT_MANIFEST_SUFFIX = "deployment-manifest"
SOURCES_PATH = "api/sources"
BACKENDS_PATH = "api/backends"
DUPLICATES_PATH = "api/source-duplicates"
PROVENANCE_PATH = "bundled-model-provenance.json"

ApiClient = Callable[[str], Any]
PageFactory = Callable[[], AbstractContextManager["BrowserPage"]]


class BrowserPage(Protocol):
    """The graph-supplied browser adapter (a configured Playwright page)."""

    def goto(self, url: str) -> Any: ...
    def javascript_errors(self) -> list[str]: ...


@dataclass(frozen=True)
class HostedCheckInputs:
    """Typed inputs for one hosted verification; every value is a graph output."""

    release_id: str
    source_revision: str
    revision: int
    root: str
    declarations: ArtifactRef
    backend: ArtifactRef
    identity: ArtifactRef
    expected_source_count: int
    retained_tasks: int
    release_manifest: ArtifactRef
    output_root: str

    def validate(self) -> None:
        """Refuse malformed inputs before any read runs."""
        if not self.release_id or not self.source_revision or not self.root:
            raise ValueError("release_id, source_revision and root are required")
        if not self.output_root:
            raise ValueError("output_root is required")
        if self.revision <= 0:
            raise ValueError(f"revision must be positive, got {self.revision}")
        if self.expected_source_count <= 0 or self.retained_tasks <= 0:
            raise ValueError("expected_source_count and retained_tasks must be positive")


def seal(path: str, value: dict) -> ArtifactRef:
    """Write-once seal with byte-identical readback; returns the artifact reference."""
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    target = StoragePath(path)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError(f"sealed artifact {path} exists with different content; use a new output root")
    else:
        target.write_bytes(raw, auto_mkdir=True)
    if target.read_bytes() != raw:
        raise ValueError(f"seal readback mismatch at {path}")
    return ArtifactRef(uri=str(target), sha256=hashlib.sha256(raw).hexdigest())


def default_api_client(root: str) -> ApiClient:
    """The maintained Marina client bound to the hosted base URL."""
    return lambda path: marina_request(root, "GET", "/" + path)


def fetch_document(client: ApiClient, path: str) -> dict:
    served = client(path)
    if not isinstance(served, dict):
        raise ValueError(f"{path}: served document is not an object")
    return served


def verify_manifest_binding(inputs: HostedCheckInputs, manifest: dict) -> list[str]:
    """The deployment manifest must bind this release, revision and source revision."""
    mismatches: list[str] = []
    if manifest.get("release_id") != inputs.release_id:
        mismatches.append(
            f"release manifest binds release {manifest.get('release_id')!r}, expected {inputs.release_id!r}"
        )
    if manifest.get("revision") != inputs.revision:
        mismatches.append(f"release manifest binds revision {manifest.get('revision')!r}, expected {inputs.revision}")
    if manifest.get("source_sha") != inputs.source_revision:
        mismatches.append(
            f"release manifest pins source revision {manifest.get('source_sha')!r}, expected {inputs.source_revision!r}"
        )
    expected_kind = f"atlas-{inputs.release_id}-{DEPLOYMENT_MANIFEST_SUFFIX}"
    if manifest.get("kind") != expected_kind:
        mismatches.append(f"release manifest kind is {manifest.get('kind')!r}, expected {expected_kind!r}")
    return mismatches


def _subject_reasons(mismatches: dict[str, list[str]]) -> list[str]:
    return [f"{subject}: {reason}" for subject, reasons in sorted(mismatches.items()) for reason in reasons]


def check_served_population(
    inputs: HostedCheckInputs,
    served: dict,
    declarations: dict[str, dict],
    survey: BackendSurvey,
    identity: SourceIdentitySurvey,
) -> tuple[list[dict], list[str]]:
    """The served sources against the sealed documents, through the serving projection."""
    mismatches = strict_audit.audit_served_population(
        served,
        declarations,
        survey,
        identity,
        expected_source_count=inputs.expected_source_count,
        retained_tasks=inputs.retained_tasks,
    )
    checks = [
        {
            "check": SOURCES_PATH,
            "sources": len(served["sources"]),
            "retained_tasks": strict_audit.served_retained_tasks(served),
            "fields": list(strict_audit.DECLARATION_FIELDS),
        }
    ]
    return checks, _subject_reasons(mismatches)


def check_backends(served: dict, survey: BackendSurvey) -> tuple[list[dict], list[str]]:
    """The served backend document against the sealed backend survey."""
    reasons = strict_audit.audit_backends_document(survey, served)
    return (
        [{"check": BACKENDS_PATH, "fields": list(strict_audit.BACKEND_DOCUMENT_FIELDS)}],
        [f"{BACKENDS_PATH}: {reason}" for reason in reasons],
    )


def check_duplicates(served_sources: dict, identity: SourceIdentitySurvey, served: dict) -> tuple[list[dict], list[str]]:
    """The served duplicate survey against the sealed identity survey."""
    mismatches = strict_audit.audit_identity_duplicates(served_sources, identity, served)
    return [{"check": DUPLICATES_PATH, "groups": len(identity.groups)}], _subject_reasons(mismatches)


def check_provenance(provenance: dict, inputs: HostedCheckInputs, manifest: dict) -> tuple[list[dict], list[str]]:
    """The served applet must be the packaged checkout the manifest published at the typed root."""
    mismatches: list[str] = []
    if provenance.get("source_revision") != manifest.get("code_revision"):
        mismatches.append(
            f"{PROVENANCE_PATH}: served applet pins revision {provenance.get('source_revision')!r},"
            f" expected the published packaging checkout {manifest.get('code_revision')!r}"
        )
    if inputs.root != manifest.get("url"):
        mismatches.append(f"hosted root {inputs.root!r} is not the published URL {manifest.get('url')!r}")
    checks = [{"check": PROVENANCE_PATH, "source_revision": provenance.get("source_revision")}]
    return checks, mismatches


def browser_gate(page_factory: PageFactory | None, root: str) -> tuple[dict, list[str]]:
    """The minimal browser gate: load the root once and refuse JavaScript errors."""
    if page_factory is None:
        return {"browser_checks": "skipped"}, []
    with page_factory() as page:
        response = page.goto(root)
        errors = list(page.javascript_errors())
    status = getattr(response, "status", None)
    failures: list[str] = []
    if status != 200:
        failures.append(f"browser gate: the root returned HTTP {status}")
    failures.extend(f"browser gate: JavaScript error {error}" for error in errors)
    return {"browser_checks": "executed", "http_status": status, "javascript_errors": len(errors)}, failures


def _report(mismatches: list[str]) -> str:
    return f"Hosted check failed with {len(mismatches)} mismatch(es): {'; '.join(mismatches)}"


def run_hosted_check(
    inputs: HostedCheckInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
    api_client: ApiClient | None = None,
    page_factory: PageFactory | None = None,
) -> dict:
    """Verify the hosted release; ``dry_run`` checks the typed documents with zero requests."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "hosted_check", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        declarations = read_ref(inputs.declarations)
        survey = BackendSurvey.model_validate(read_ref(inputs.backend))
        identity = SourceIdentitySurvey.model_validate(read_ref(inputs.identity))
        manifest = read_ref(inputs.release_manifest)
        if len(declarations) != inputs.expected_source_count:
            raise ValueError(f"{len(declarations)} declarations, expected {inputs.expected_source_count}")
        mismatches = verify_manifest_binding(inputs, manifest)
        checks: list[dict] = []
        request_count = 0
        if not dry_run:
            client = api_client if api_client is not None else default_api_client(inputs.root)
            served_sources = fetch_document(client, SOURCES_PATH)
            served_backends = fetch_document(client, BACKENDS_PATH)
            served_duplicates = fetch_document(client, DUPLICATES_PATH)
            provenance = fetch_document(client, PROVENANCE_PATH)
            request_count += 4
            for document_checks, document_mismatches in (
                check_served_population(inputs, served_sources, declarations, survey, identity),
                check_backends(served_backends, survey),
                check_duplicates(served_sources, identity, served_duplicates),
                check_provenance(provenance, inputs, manifest),
            ):
                checks.extend(document_checks)
                mismatches.extend(document_mismatches)
            browser_summary, browser_failures = browser_gate(page_factory, inputs.root)
            checks.append({"check": "browser", **browser_summary})
            mismatches.extend(browser_failures)
        if mismatches:
            raise ValueError(_report(mismatches))
        proof: dict = {
            "kind": f"atlas-{inputs.release_id}-{HOSTED_CHECK_SUFFIX}",
            "release_id": inputs.release_id,
            "source_revision": inputs.source_revision,
            "revision": inputs.revision,
            "origin": inputs.root,
            "checks": checks,
            "browser_checks": "skipped" if dry_run or page_factory is None else "executed",
            "mismatches": {},
            "requests": request_count,
            "dry_run": dry_run,
            "sql_writes": 0,
        }
        if not dry_run:
            receipt = {
                "kind": proof["kind"],
                "release_id": inputs.release_id,
                "revision": inputs.revision,
                "origin": inputs.root,
                "mismatches": {},
                "checks": checks,
                "release_manifest": {"uri": inputs.release_manifest.uri, "sha256": inputs.release_manifest.sha256},
                "sealed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            }
            sealed = seal(f"{inputs.output_root}/hosted-check.json", receipt)
            proof["receipt"] = {"uri": sealed.uri, "sha256": sealed.sha256}
        alerts.stage_finished(
            inputs.release_id,
            "hosted_check",
            {"mode": mode, "revision": inputs.revision, **proof.get("receipt", {})},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "hosted_check", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON HostedCheckInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    for name in ("declarations", "backend", "identity", "release_manifest"):
        document[name] = ArtifactRef(**document[name])
    inputs = HostedCheckInputs(**document)
    proof = run_hosted_check(inputs, dry_run=args.mode == "dry-run")
    label = "HOSTED-CHECK-DRY-RUN" if args.mode == "dry-run" else "HOSTED-CHECK-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
