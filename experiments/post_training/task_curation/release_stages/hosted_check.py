# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hosted check: verify the published preview through its installed API.

One maintained module replaces the per-release hosted verification scripts (see
INVENTORY.md). The verifier is data-driven: every expectation comes from the typed
inputs and the sealed documents they reference — the declaration set, the backend
survey, the identity survey and the deployment manifest — never from counts or
shapes hard-coded here. All mismatches are collected and reported at once, like the
admission check.

The served release record (``marina-release.json`` beside the API routes) must bind
the typed revision and the manifest's app id, and must carry the release's selected
backend map with a matching SHA-256: the record's map digest must equal the
canonical digest of the sealed backend survey document, and both must equal the
digest of the record's own map content.

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

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

HOSTED_CHECK_SUFFIX = "hosted-check"
DEPLOYMENT_MANIFEST_SUFFIX = "deployment-manifest"
SOURCES_PATH = "api/sources"
BACKENDS_PATH = "api/backends"
DUPLICATES_PATH = "api/source-duplicates"
RELEASE_RECORD_PATH = "marina-release.json"

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
        if self.expected_source_count <= 0:
            raise ValueError(f"expected_source_count must be positive, got {self.expected_source_count}")


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


def canonical_sha256(value: Any) -> str:
    """The canonical-JSON digest used for served documents and sealed survey content."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fetch_document(client: ApiClient, path: str) -> dict:
    served = client(path)
    if not isinstance(served, dict):
        raise ValueError(f"{path}: served document is not an object")
    return served


def shared_field_mismatches(expected: dict, served: dict, label: str) -> list[str]:
    """Fields present in both documents whose values disagree."""
    return [
        f"{label}: field {name} is {served[name]!r}, expected {expected[name]!r}"
        for name in sorted(set(expected) & set(served))
        if served[name] != expected[name]
    ]


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


def check_sources(served: dict, declarations: dict, expected_count: int) -> tuple[list[dict], list[str]]:
    """The served sources must be exactly the declared population, field by field."""
    rows = served.get("sources")
    if not isinstance(rows, list):
        return [], [f"{SOURCES_PATH}: served document carries no sources list"]
    mismatches: list[str] = []
    served_ids = [row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]
    if len(served_ids) != len(rows):
        mismatches.append(f"{SOURCES_PATH}: {len(rows) - len(served_ids)} served rows carry no id")
    if set(served_ids) != set(declarations):
        missing = sorted(set(declarations) - set(served_ids))
        unexpected = sorted(set(served_ids) - set(declarations))
        mismatches.append(
            f"{SOURCES_PATH}: served sources differ from the declarations"
            f" (missing {missing[:5]}, unexpected {unexpected[:5]})"
        )
    if len(rows) != expected_count:
        mismatches.append(f"{SOURCES_PATH}: {len(rows)} served sources, expected {expected_count}")
    for row in rows:
        source_id = row.get("id") if isinstance(row, dict) else None
        if isinstance(source_id, str) and source_id in declarations:
            mismatches.extend(shared_field_mismatches(declarations[source_id], row, f"{SOURCES_PATH}[{source_id}]"))
    return [{"check": SOURCES_PATH, "sources": len(rows)}], mismatches


def check_backends(served: dict, survey: dict) -> tuple[list[dict], list[str]]:
    """The served backend document must agree with the sealed backend survey fields."""
    shared = set(served) & set(survey)
    if not shared:
        return [], [f"{BACKENDS_PATH}: served document shares no fields with the backend survey document"]
    return (
        [{"check": BACKENDS_PATH, "shared_fields": sorted(shared)}],
        shared_field_mismatches(survey, served, BACKENDS_PATH),
    )


def check_duplicates(served: dict, identity: dict) -> tuple[list[dict], list[str]]:
    """The served duplicate survey must agree with the sealed identity survey fields."""
    shared = set(served) & set(identity)
    if not shared:
        return [], [f"{DUPLICATES_PATH}: served document shares no fields with the identity survey document"]
    return (
        [{"check": DUPLICATES_PATH, "shared_fields": sorted(shared)}],
        shared_field_mismatches(identity, served, DUPLICATES_PATH),
    )


def check_release_record(
    record: dict, inputs: HostedCheckInputs, manifest: dict, survey: dict
) -> tuple[list[dict], list[str]]:
    """The served release record must bind the revision, app id and selected backend map."""
    mismatches: list[str] = []
    if record.get("revision") != inputs.revision:
        mismatches.append(f"{RELEASE_RECORD_PATH}: revision is {record.get('revision')!r}, expected {inputs.revision}")
    if record.get("app_id") != manifest.get("app_id"):
        mismatches.append(
            f"{RELEASE_RECORD_PATH}: app id is {record.get('app_id')!r}, expected {manifest.get('app_id')!r}"
        )
    digest = record.get("selected_backend_map_sha256")
    backend_map = record.get("selected_backend_map")
    if not isinstance(backend_map, dict) or not isinstance(digest, str):
        mismatches.append(f"{RELEASE_RECORD_PATH}: the record carries no selected backend map with its digest")
    else:
        if canonical_sha256(backend_map) != digest:
            mismatches.append(
                f"{RELEASE_RECORD_PATH}: the recorded backend map digest does not match the record's map content"
            )
        if digest != canonical_sha256(survey):
            mismatches.append(
                f"{RELEASE_RECORD_PATH}: the record's backend map digest differs from the backend survey document"
            )
    checks = [
        {
            "check": RELEASE_RECORD_PATH,
            "revision": record.get("revision"),
            "app_id": record.get("app_id"),
            "selected_backend_map_sha256": digest,
        }
    ]
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
        survey = read_ref(inputs.backend)
        identity = read_ref(inputs.identity)
        manifest = read_ref(inputs.release_manifest)
        if len(declarations) != inputs.expected_source_count:
            raise ValueError(f"{len(declarations)} declarations, expected {inputs.expected_source_count}")
        mismatches = verify_manifest_binding(inputs, manifest)
        checks: list[dict] = []
        request_count = 0
        if not dry_run:
            client = api_client if api_client is not None else default_api_client(inputs.root)
            request_count += 1
            source_checks, source_mismatches = check_sources(
                fetch_document(client, SOURCES_PATH), declarations, inputs.expected_source_count
            )
            request_count += 1
            backend_checks, backend_mismatches = check_backends(fetch_document(client, BACKENDS_PATH), survey)
            request_count += 1
            duplicate_checks, duplicate_mismatches = check_duplicates(fetch_document(client, DUPLICATES_PATH), identity)
            request_count += 1
            record_checks, record_mismatches = check_release_record(
                fetch_document(client, RELEASE_RECORD_PATH), inputs, manifest, survey
            )
            browser_summary, browser_failures = browser_gate(page_factory, inputs.root)
            checks.extend(
                [
                    *source_checks,
                    *backend_checks,
                    *duplicate_checks,
                    *record_checks,
                    {"check": "browser", **browser_summary},
                ]
            )
            mismatches.extend(
                [*source_mismatches, *backend_mismatches, *duplicate_mismatches, *record_mismatches, *browser_failures]
            )
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
