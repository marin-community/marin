# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exposure probe: anonymous exposure check of the promoted public release.

One maintained module replaces the per-release anonymous-exposure scripts (see
INVENTORY.md). The probe requests the public production root at ``{root}/v/{revision}/``
strictly anonymously: the module passes no credentials, and the default session
refuses environment proxies and credential files outright.

For every license-withheld source the probe verifies the content routes withhold
content (difficulty attempts and rollout with the probe marker, the curriculum
sample and detail, and the assessment evidence route), with the response-key
allowlist the reference scripts pinned. For every publication grant it verifies the
upstream link and license render publicly, through the graph-supplied browser
adapter when one is provided; otherwise the receipt records the display check as
skipped. Any failure is reported with the failed routes listed; the receipt seals
only on a full pass.
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
from urllib.parse import urlencode

import requests
from rigging.filesystem.storage_path import StoragePath

import experiments.post_training.task_curation.release_stages.alerts as alerts
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref
from infra.marina.apps.rl_data_catalog.server.content_access import (
    LICENSE_CONTENT_MARKER,
    content_is_withheld,
    license_content_excluded,
)
from infra.marina.apps.rl_data_catalog.server.publication_scope import (
    PUBLICATION_DECISIONS,
    publication_scope_decision,
)

EXPOSURE_SUFFIX = "anonymous-publication-exposure"
PROBE_MARKER = "exposure-guard-probe"
REQUEST_TIMEOUT = 60
WITHHELD_RESPONSE_KEYS = frozenset({"data_restriction", "content_withheld", "withheld_reason", "content", "marker"})

HttpGet = Callable[[str, dict[str, str]], Any]
PageFactory = Callable[[], AbstractContextManager["PublicDisplayPage"]]


class PublicDisplayPage(Protocol):
    """The graph-supplied browser adapter for the public source pages.

    ``visible_link`` and ``body_contains`` raise when the rendered page fails the
    assertion (the Playwright ``expect`` contract).
    """

    def goto(self, url: str) -> Any: ...
    def visible_link(self, href: str) -> None: ...
    def body_contains(self, text: str) -> None: ...


@dataclass(frozen=True)
class ExposureProbeInputs:
    """Typed inputs for one anonymous exposure probe; every value is a graph output."""

    release_id: str
    revision: int
    root: str
    withheld_sources: tuple[str, ...]
    publication_grants: tuple[dict, ...]
    input_check_receipt: ArtifactRef | None
    output_root: str

    def validate(self) -> None:
        """Refuse malformed inputs before any request runs."""
        if not self.release_id or not self.root or not self.output_root:
            raise ValueError("release_id, root and output_root are required")
        if self.revision <= 0:
            raise ValueError(f"revision must be positive, got {self.revision}")
        if not self.withheld_sources:
            raise ValueError("withheld_sources must be non-empty: the probe exists to verify withholding")
        if len(set(self.withheld_sources)) != len(self.withheld_sources):
            raise ValueError("withheld_sources repeats a source id")
        if not self.publication_grants:
            raise ValueError("publication_grants must be non-empty: the probe verifies visible attribution")
        for position, grant in enumerate(self.publication_grants):
            missing = [key for key in ("source_id", "upstream_url", "license_id") if not grant.get(key)]
            if missing:
                raise ValueError(f"publication grant {position} lacks {missing}")


def exposure_lists(declarations: dict[str, dict]) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    """The license-withheld sources and public-display grants of the release's declaration cut.

    The withheld list is every declaration whose content the served app withholds; the
    grants are the recorded public-display decisions, each bound to its exact declared
    release. Fails closed on a withheld source the probe cannot verify (one withheld
    for a reason other than its license carries a different marker) and on a grant
    whose source is missing from the cut, withheld, or declared at another release.
    """
    withheld = tuple(sorted(source_id for source_id, row in declarations.items() if content_is_withheld(row)))
    unverifiable = [source_id for source_id in withheld if not license_content_excluded(declarations[source_id])]
    if unverifiable:
        raise ValueError(f"the probe verifies only license withholding; withheld for another reason: {unverifiable}")
    grants: list[dict] = []
    for decision in PUBLICATION_DECISIONS:
        declaration = declarations.get(decision.source_id)
        if declaration is None:
            raise ValueError(f"publication grant {decision.source_id!r} names a source outside the declaration cut")
        if publication_scope_decision(declaration) != decision:
            raise ValueError(f"publication grant {decision.source_id!r} does not match its declared release")
        if content_is_withheld(declaration):
            raise ValueError(f"publication grant {decision.source_id!r} names a withheld source")
        grants.append(decision.model_dump(mode="json"))
    return withheld, tuple(grants)


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


def version_root(root: str, revision: int) -> str:
    """The public revision root the probe requests: ``{root}/v/{revision}/``."""
    return f"{root.rstrip('/')}/v/{revision}/"


def anonymous_http() -> HttpGet:
    """An anonymous GET: environment proxies and credentials disabled, no auth headers, no cookies."""
    session = requests.Session()
    session.trust_env = False
    if "Authorization" in session.headers or len(session.cookies) > 0:
        raise AssertionError("the exposure probe session must start without credentials")

    def get(url: str, headers: dict[str, str]) -> Any:
        if "Authorization" in headers or "Cookie" in headers:
            raise AssertionError("the exposure probe must never send credentials")
        return session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

    return get


def verify_input_check(inputs: ExposureProbeInputs) -> None:
    """The typed withheld list must match the exposure input-check receipt when one is bound."""
    if inputs.input_check_receipt is None:
        return
    document = read_ref(inputs.input_check_receipt)
    recorded = document.get("withheld_sources")
    if not isinstance(recorded, list) or sorted(recorded) != sorted(inputs.withheld_sources):
        raise ValueError(
            f"the input check receipt withholds {len(recorded) if isinstance(recorded, list) else 'no'} sources,"
            f" the typed inputs withhold {len(inputs.withheld_sources)}"
        )


def _fetch_json(
    http: HttpGet, url: str, headers: dict[str, str], source_id: str, route: str, failures: list[str]
) -> dict | None:
    response = http(url, headers)
    if response.status_code != 200:
        failures.append(f"{route} for {source_id} returned HTTP {response.status_code}")
        return None
    document = response.json()
    if not isinstance(document, dict):
        failures.append(f"{route} for {source_id} returned a non-object document")
        return None
    return document


def _check_withheld_row(row: dict, source_id: str, route: str, failures: list[str]) -> None:
    unexpected = sorted(set(row) - WITHHELD_RESPONSE_KEYS)
    if unexpected:
        failures.append(f"{route} for {source_id} exposes unexpected keys {unexpected}")
    if row.get("content_withheld") is not True:
        failures.append(f"{route} for {source_id} does not record content_withheld true")
    if row.get("content") is not None:
        failures.append(f"{route} for {source_id} leaks content")


def probe_withheld_source(
    http: HttpGet, base: str, source_id: str, headers: dict[str, str]
) -> tuple[list[dict], list[str]]:
    """The content routes for one withheld source; returns check rows and failures."""
    checks: list[dict] = []
    failures: list[str] = []

    def fetch(route: str, query: dict[str, str]) -> dict | None:
        return _fetch_json(http, base + route + "?" + urlencode(query), headers, source_id, route, failures)

    params = {"source_id": source_id, "measurement_id": PROBE_MARKER}
    for endpoint in ("difficulty-attempts", "difficulty-rollout"):
        route = f"api/{endpoint}"
        query = {**params, **({"attempt_id": PROBE_MARKER} if endpoint == "difficulty-rollout" else {})}
        before = len(failures)
        row = fetch(route, query)
        if row is not None:
            _check_withheld_row(row, source_id, route, failures)
            if len(failures) == before:
                checks.append({"source_id": source_id, "endpoint": endpoint, "content_withheld": True})
    before = len(failures)
    route = "api/curriculum/tasks"
    document = fetch(route, {"source_id": source_id, "limit": "1"})
    tasks = document.get("tasks") if document is not None else None
    if document is not None and not isinstance(tasks, list):
        failures.append(f"{route} for {source_id} carries no tasks list")
    if isinstance(tasks, list):
        unredacted = [
            task.get("task_id")
            for task in tasks
            if isinstance(task, dict) and task.get("task_preview") != LICENSE_CONTENT_MARKER
        ]
        if unredacted:
            failures.append(f"{route} for {source_id} renders unredacted sample previews for {unredacted[:3]}")
    if len(failures) == before and isinstance(tasks, list):
        checks.append({"source_id": source_id, "endpoint": "curriculum/tasks", "stored_samples_checked": len(tasks)})
    if isinstance(tasks, list) and tasks:
        before = len(failures)
        route = "api/curriculum/task"
        task_id = tasks[0].get("task_id") if isinstance(tasks[0], dict) else None
        if not isinstance(task_id, str) or not task_id:
            failures.append(f"{route} for {source_id}: the sample task carries no task_id")
        else:
            detail = fetch(route, {"source_id": source_id, "task_id": task_id})
            task = detail.get("task") if detail is not None else None
            if not isinstance(task, dict) or task.get("content_withheld") is not True:
                failures.append(f"{route} for {source_id} renders the sample task unwithheld")
            if len(failures) == before:
                checks.append({"source_id": source_id, "endpoint": "curriculum/task", "content_withheld": True})
    before = len(failures)
    route = "api/assessment"
    response = http(base + route + "?" + urlencode({"source_id": source_id, "limit": "1"}), headers)
    if response.status_code not in (200, 404):
        failures.append(f"{route} for {source_id} returned HTTP {response.status_code}")
    else:
        checks.append({"source_id": source_id, "endpoint": "assessment", "status": response.status_code})
        if response.status_code == 200:
            _probe_assessment_evidence(http, base, source_id, headers, response.json(), failures, checks)
    return checks, failures


def _probe_assessment_evidence(
    http: HttpGet,
    base: str,
    source_id: str,
    headers: dict[str, str],
    document: Any,
    failures: list[str],
    checks: list[dict],
) -> None:
    route = "api/assessment"
    items = document.get("report", {}).get("evidence", {}).get("items") if isinstance(document, dict) else None
    if not isinstance(items, list):
        failures.append(f"{route} for {source_id} carries no evidence items")
        return
    if not items:
        return
    route = "api/evidence"
    task_id = items[0].get("task_id") if isinstance(items[0], dict) else None
    if not isinstance(task_id, str) or not task_id:
        failures.append(f"{route} for {source_id}: the first evidence item carries no task_id")
        return
    before = len(failures)
    evidence = _fetch_json(
        http,
        base + route + "?" + urlencode({"source_id": source_id, "task_id": task_id}),
        headers,
        source_id,
        route,
        failures,
    )
    if evidence is not None:
        packet = evidence.get("packet")
        review = evidence.get("raw_review")
        withheld = (
            isinstance(packet, dict)
            and packet.get("content_withheld") is True
            and isinstance(review, dict)
            and review.get("content_withheld") is True
        )
        if not withheld:
            failures.append(f"{route} for {source_id} renders unwithheld QC evidence")
        if len(failures) == before:
            checks.append({"source_id": source_id, "endpoint": "evidence", "content_withheld": True})


def _approval(grant: dict, status: str) -> dict:
    return {
        "source_id": grant["source_id"],
        "license_id": grant["license_id"],
        "upstream_url": grant["upstream_url"],
        "status": status,
    }


def verify_public_display(
    page_factory: PageFactory | None, grants: tuple[dict, ...], base: str
) -> tuple[list[dict], list[str]]:
    """Verify each grant's upstream link and license render publicly; skipped without a browser."""
    if page_factory is None:
        return [_approval(grant, "skipped") for grant in grants], []
    approvals: list[dict] = []
    failures: list[str] = []
    with page_factory() as page:
        for grant in grants:
            source_id = grant["source_id"]
            url = base + "source.html#" + urlencode({"source_id": source_id})
            response = page.goto(url)
            status = getattr(response, "status", None)
            if status != 200:
                failures.append(f"source page for {source_id} returned HTTP {status}")
                continue
            try:
                page.visible_link(grant["upstream_url"])
                page.body_contains(str(grant["license_id"]).upper())
            except (AssertionError, ValueError) as error:
                failures.append(f"public display for {source_id}: {error}")
                continue
            approvals.append(_approval(grant, "verified"))
    return approvals, failures


def run_exposure_probe(
    inputs: ExposureProbeInputs,
    *,
    dry_run: bool,
    feeds: tuple | None = None,
    http: HttpGet | None = None,
    page_factory: PageFactory | None = None,
) -> dict:
    """Probe the promoted public release; ``dry_run`` verifies the typed inputs with zero requests."""
    alert_feeds = feeds if feeds is not None else (alerts.WATCHDOG_FEED,)
    mode = "dry-run" if dry_run else "write"
    alerts.stage_started(inputs.release_id, "exposure_probe", {"mode": mode}, alert_feeds)
    try:
        inputs.validate()
        verify_input_check(inputs)
        if dry_run:
            proof = {
                "kind": f"atlas-{inputs.release_id}-{EXPOSURE_SUFFIX}",
                "release_id": inputs.release_id,
                "revision": inputs.revision,
                "root": version_root(inputs.root, inputs.revision),
                "withheld_sources": len(inputs.withheld_sources),
                "publication_grants": len(inputs.publication_grants),
                "requests": 0,
                "browser_checks": "skipped",
                "dry_run": True,
                "sql_writes": 0,
            }
            alerts.stage_finished(inputs.release_id, "exposure_probe", {"mode": mode}, alert_feeds)
            return proof
        base = version_root(inputs.root, inputs.revision)
        get = http if http is not None else anonymous_http()
        headers: dict[str, str] = {}
        checks: list[dict] = []
        failures: list[str] = []
        for source_id in sorted(inputs.withheld_sources):
            source_checks, source_failures = probe_withheld_source(get, base, source_id, headers)
            checks.extend(source_checks)
            failures.extend(source_failures)
        approvals, display_failures = verify_public_display(page_factory, inputs.publication_grants, base)
        failures.extend(display_failures)
        if failures:
            raise ValueError(f"Anonymous exposure probe failed {len(failures)} check(s): {'; '.join(failures)}")
        receipt = {
            "kind": f"atlas-{inputs.release_id}-{EXPOSURE_SUFFIX}",
            "status": "PASS",
            "revision": inputs.revision,
            "root": base,
            "license_withheld_sources": len(inputs.withheld_sources),
            "content_route_checks": checks,
            "public_display_approvals": approvals,
            "anonymous": True,
            "sealed_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        sealed = seal(f"{inputs.output_root}/exposure-probe.json", receipt)
        proof = {
            "kind": receipt["kind"],
            "status": "PASS",
            "revision": inputs.revision,
            "root": base,
            "license_withheld_sources": len(inputs.withheld_sources),
            "content_route_checks": len(checks),
            "public_display_approvals": len(approvals),
            "anonymous": True,
            "dry_run": False,
            "sql_writes": 0,
            "receipt": {"uri": sealed.uri, "sha256": sealed.sha256},
        }
        alerts.stage_finished(
            inputs.release_id,
            "exposure_probe",
            {"mode": mode, "routes": len(checks), **proof["receipt"]},
            alert_feeds,
        )
        return proof
    except Exception as error:
        alerts.stage_failed(inputs.release_id, "exposure_probe", {"mode": mode, "error": str(error)[:500]}, alert_feeds)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Path to a JSON ExposureProbeInputs document")
    parser.add_argument("--mode", choices=("dry-run", "write"), default="dry-run")
    args = parser.parse_args()
    document = json.loads(args.inputs.read_text(encoding="utf-8"))
    if document.get("input_check_receipt") is not None:
        document["input_check_receipt"] = ArtifactRef(**document["input_check_receipt"])
    inputs = ExposureProbeInputs(**document)
    proof = run_exposure_probe(inputs, dry_run=args.mode == "dry-run")
    label = "EXPOSURE-PROBE-DRY-RUN" if args.mode == "dry-run" else "EXPOSURE-PROBE-SEALED"
    print(f"{label} {json.dumps(proof)}", flush=True)


if __name__ == "__main__":
    main()
