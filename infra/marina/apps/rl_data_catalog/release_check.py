# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check a landed Atlas preview package or published revision and retain one JSON receipt."""

import argparse
import ast
import functools
import hashlib
import importlib
import json
import logging
import queue
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests
from finestore.rl.status_taxonomy import taxonomy_tables
from marina.client import MarinaRequestError, applet_query, bearer_token, marina_request, marina_token_provider
from marina.testing import test_database_environment
from playwright.sync_api import Page, Route, sync_playwright
from shellbox.requirements import Requirement

from .prepare_preview import (
    APP_SOURCE,
    IMPORT_RELOCATIONS,
    PREVIEW_APP_ID,
    PRODUCTION_APP_ID,
    RELOCATED_SERVER_FILES,
    oracle_inventory_payload,
)
from .readability import readability_smoke, readability_sweep
from .release_browse import browse_invariants
from .release_browser import (
    LAYOUTS,
    curriculum_domain_sweep,
    curriculum_release_sweep,
    curriculum_sweep,
    evidence_cases,
    evidence_smoke,
    evidence_sweep,
    report_failed_request,
    report_failed_response,
    source_sweep,
)
from .release_curriculum import curriculum_invariants
from .release_evidence import (
    SqlQuery,
    catalog_state,
    projected_contract_inventory,
    projected_duplicate_inventory,
    projected_sources,
    stored_duplicate_groups,
    verify_preservation,
)
from .release_gateway import CandidateApi, bounded_read_query
from .release_receipts import (
    BrowserScope,
    CommandResult,
    PreflightBaseline,
    ReleaseFailure,
    ReleaseInputs,
    ReleaseReceipt,
    ReleaseTarget,
)
from .server.app import table_source_record
from .server.curriculum import (
    AREA_SQL,
    CONFIDENCE_AREA_SQL,
    RELEASE_SQL,
    TASKS_SQL,
    CoverageQuery,
    area_parameters,
    area_response,
    coverage_document,
    invariant_document,
    source_restrictions,
    task_parameters,
    task_response,
)
from .server.derived_contract import PROFILE_FIELDS
from .server.json_digest import json_sha256
from .server.source_counts import SourceCounts

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def landed_cut(attested_revision: str | None) -> str:
    """Refuse uncommitted source and cuts absent from the freshly fetched integration branch.

    An in-region release bundle is a copied workspace, not a checkout: its submitter
    verified the landing and passes the attested revision, which replaces the git checks.
    """
    if attested_revision is not None:
        if len(attested_revision) != 40 or any(c not in "0123456789abcdef" for c in attested_revision):
            raise ValueError(f"attested revision is not a full commit sha: {attested_revision!r}")
        return attested_revision
    if git("status", "--porcelain"):
        raise ValueError("Release checks require a clean committed checkout")
    if git("branch", "--show-current") == "main":
        raise ValueError("Run Atlas release checks in a dedicated worktree, never main")
    subprocess.run(["git", "fetch", "origin", "atlas-env-audit"], cwd=ROOT, check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/atlas-env-audit"], cwd=ROOT, check=True)
    return git("rev-parse", "HEAD")


def check_workspace_imports() -> None:
    """Reject editable workspace dependencies resolved from a different checkout."""
    for name, relative in (
        ("finestore", "lib/finestore/src/finestore/__init__.py"),
        ("taskcompendium", "lib/taskcompendium/src/taskcompendium/__init__.py"),
        ("marina", "infra/marina/src/marina/__init__.py"),
    ):
        location = importlib.import_module(name).__file__
        if location is None or Path(location).resolve() != ROOT / relative:
            raise ValueError(f"{name} resolves outside this checkout; sync its own virtual environment")


def check_package(package: Path, head: str, count_raw: bytes, inventory_raw: bytes) -> dict[str, Any]:
    """Verify preview identity and source/model/count bytes against the landed cut."""
    if tomllib.loads((package / "applet.toml").read_text())["applet_id"] != PREVIEW_APP_ID:
        raise ValueError("The package must declare the preview UUID")
    metadata = json.loads((package / "dist/bundled-model-provenance.json").read_text())
    if metadata["source_revision"] != head or metadata["preview_only"] is not True:
        raise ValueError("The preview package does not identify the checked landed cut")
    for module in metadata["modules"]:
        for path, expected in (
            (ROOT / module["source"], module["source_sha256"]),
            (package / module["packaged"], module["packaged_sha256"]),
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Bundled module provenance differs: {path}")
    if metadata["source_counts"]["source_sha256"] != hashlib.sha256(count_raw).hexdigest():
        raise ValueError("The package count inputs differ from --source-counts")
    inventory = json.loads((package / "dist/measurements/oracle-inventory.json").read_text())
    if inventory != oracle_inventory_payload(inventory_raw):
        raise ValueError("Packaged oracle inventory differs from --oracle-inventory")
    inventory_rows = {row["source_id"]: row for row in inventory["sources"]}
    for observation in SourceCounts.model_validate_json(count_raw).sources:
        record = inventory_rows.get(observation.source_id)
        if (
            record is None
            or record["input"] != observation.input.model_dump(mode="json")
            or record["oracle_tasks"] != observation.oracle_tasks
        ):
            raise ValueError(f"Count and oracle inventory inputs differ: {observation.source_id}")
    assignment = next(
        node
        for node in ast.parse((package / "server/count_evidence.py").read_text()).body
        if isinstance(node, ast.Assign)
    )
    packaged_counts = ast.literal_eval(assignment.value)
    if SourceCounts.model_validate_json(packaged_counts) != SourceCounts.model_validate_json(count_raw):
        raise ValueError("Packaged count observations differ from their explicit inputs")
    for tracked in git("ls-files", str(APP_SOURCE)).splitlines():
        relative = str(Path(tracked).relative_to(APP_SOURCE))
        if "tests" in Path(relative).parts or relative in {
            "applet.toml",
            "server/count_evidence.py",
            "dist/measurements/oracle-inventory.json",
        }:
            continue
        expected = (ROOT / tracked).read_bytes()
        if relative in RELOCATED_SERVER_FILES:
            for before, after in IMPORT_RELOCATIONS.items():
                expected = expected.replace(before.encode(), after.encode())
        if (package / relative).read_bytes() != expected:
            raise ValueError(f"Package source differs from the checked landed cut: {relative}")
    return metadata


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib override parameter
        LOGGER.debug(format, *args)


@contextmanager
def static_preview(dist: Path) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(dist)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class CatalogReader:
    """Cache immutable-revision API responses during one pointer-checked browser sweep."""

    def __init__(self, root: str, session: requests.Session, query: SqlQuery | None = None) -> None:
        self.query = functools.partial(bounded_read_query, query) if query is not None else None
        self.root = root
        self.session = session
        self.provider = marina_token_provider()
        self.cache: dict[str, Any] = {}
        self.releases: dict[str, Any] = {}
        self.candidate_api = CandidateApi(query) if query is not None else None

    @staticmethod
    def cache_key(path: str) -> str:
        url = urlsplit(path)
        query = urlencode(sorted((key, value) for key, value in parse_qsl(url.query) if value))
        return url.path + ("?" + query if query else "")

    def get(self, path: str) -> Any:
        url = urlsplit(path)
        key = self.cache_key(path)
        if key not in self.cache:
            params = dict(parse_qsl(url.query))
            if self.candidate_api is not None and url.path in {
                "api/sources",
                "api/source-statuses",
                "api/source",
                "api/assessment",
                "api/evidence",
                "api/qc/defects",
                "api/qc/inventory",
                "api/difficulty-attempts",
                "api/difficulty-rollout",
            }:
                data = self.candidate_api.get(key)
            elif self.candidate_api is not None and url.path == "api/taxonomy":
                # The candidate's pages render from the tables its own API will serve;
                # the preceding revision may serve an older table set (R40: v39 had no
                # backend tables, so every Backends page failed the sweep).
                data = candidate_taxonomy()
            elif self.query is not None and url.path == "api/source-labels":
                catalog = self.get("api/sources")
                data = {
                    "sources": [
                        {key: row[key] for key in ("id", "display_name", "origin", "build_path")}
                        for row in catalog["sources"]
                    ],
                    "snapshot_id": catalog["snapshot_id"],
                }
            elif self.query is not None and url.path in {
                "api/curriculum",
                "api/curriculum/area",
                "api/curriculum/tasks",
                "api/curriculum/invariants",
            }:
                release = params.get("release", "")
                if release not in self.releases:
                    self.releases[release] = self.query(RELEASE_SQL, {"release": release})["rows"][0]
                metadata = self.releases[release]
                filters = CoverageQuery(
                    params.get("intended_use", "all"), params.get("source_id", ""), params.get("release", "")
                )
                subject, node, scope = (
                    params.get("subject_id", ""),
                    params.get("node_id", ""),
                    params.get("scope", "all"),
                )
                if url.path == "api/curriculum/invariants":
                    data = invariant_document(lambda sql, bound: self.query(sql, bound)["rows"], metadata)
                elif url.path == "api/curriculum":
                    data = coverage_document(
                        lambda sql, bound: self.query(sql, bound)["rows"], metadata, filters, subject
                    )
                elif url.path == "api/curriculum/area":
                    bound = area_parameters(metadata, filters, subject, node, scope)
                    rows = self.query(AREA_SQL, bound)["rows"]
                    data = area_response(metadata, rows)
                    if metadata["manifest"].get("confidence"):
                        bound["levels"] = len(metadata["manifest"]["confidence"]["levels"])
                        data["confidence_sources"] = self.query(CONFIDENCE_AREA_SQL, bound)["rows"]
                else:
                    limit = int(params.get("limit", "50"))
                    bound = task_parameters(
                        metadata,
                        filters,
                        subject,
                        node,
                        scope,
                        params.get("outcome", "all"),
                        params.get("cursor", ""),
                        limit,
                        params.get("sample_pool") == "true",
                    )
                    rows = self.query(TASKS_SQL, bound)["rows"]
                    data = task_response(metadata, rows, limit, source_restrictions(rows))
            else:
                data = marina_request(self.root, "GET", "/" + key, session=self.session, token_provider=self.provider)
            self.cache[key] = data
        return self.cache[key]


CURRICULUM_LATENCY_LIMIT = 10.0
"""Seconds one whole-subject sample-pool page may take on the real catalog (R40 #54)."""


def curriculum_latency(
    reader: CatalogReader, releases: list[dict[str, Any]], clock: Any = time.monotonic
) -> dict[str, Any]:
    """Time every subject's sample-pool page of the selected curriculum; refuse a slow one.

    The curriculum page requests ``limit=1000`` sample pools per subject. R40's query
    exceeded the 60 s gateway timeout for half the subjects and only the postpublish
    full sweep found it, after the publish. Each page is read uncached.
    """
    selected = [release["release_id"] for release in releases if release.get("selected")]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one selected curriculum release, found {selected}")
    release_id = selected[0]
    hierarchy = reader.get(f"api/curriculum?{urlencode({'release': release_id})}")["hierarchy"]
    subjects = sorted({node["subject_id"] for node in hierarchy if node.get("kind") == "subject"})
    timings: dict[str, float | None] = {}
    for subject in subjects:
        path = "api/curriculum/tasks?" + urlencode(
            {"intended_use": "all", "release": release_id, "sample_pool": "true", "limit": "1000", "subject_id": subject}
        )
        reader.cache.pop(reader.cache_key(path), None)
        started = clock()
        try:
            reader.get(path)
        except MarinaRequestError as error:
            if error.status_code != 504:
                raise
            # The gateway's statement timeout: the page is slow, name it with the others.
            timings[subject] = None
            continue
        timings[subject] = round(clock() - started, 2)
    slow = {
        subject: seconds for subject, seconds in timings.items() if seconds is None or seconds > CURRICULUM_LATENCY_LIMIT
    }
    if slow:
        raise ValueError(f"Curriculum sample-pool pages exceed {CURRICULUM_LATENCY_LIMIT} s: {slow}")
    return {"release_id": release_id, "limit_seconds": CURRICULUM_LATENCY_LIMIT, "subjects": timings}


def candidate_taxonomy() -> dict[str, Any]:
    """The taxonomy tables this release's API serves (the server's ``/taxonomy`` route)."""
    return taxonomy_tables(requirement_values=[requirement.value for requirement in Requirement])


def browser_api_response(reader: CatalogReader, path: str, sources: list[dict[str, Any]]) -> tuple[int, Any]:
    """Deliver the real empty-assessment response without excusing missing required data."""
    try:
        return 200, reader.get(path)
    except MarinaRequestError as error:
        parsed = urlsplit(path)
        source_id = dict(parse_qsl(parsed.query)).get("source_id")
        source = next((row for row in sources if row["id"] == source_id), None)
        if error.status_code == 404 and parsed.path == "api/taxonomy":
            # The canonical taxonomy tables ship with this release: the served
            # revision predates the endpoint, so the checker answers the route
            # from the candidate's own code - the same tables the released API
            # serves.
            return 200, candidate_taxonomy()
        if (
            error.status_code == 404
            and parsed.path == "api/assessment"
            and source is not None
            and not (source.get("assessment") or {}).get("pipeline_key")
        ):
            return 404, {"detail": str(error)}
        if error.status_code == 504:
            return retry_transient_read(reader, path, error)
        raise


def retry_transient_read(reader: CatalogReader, path: str, error: MarinaRequestError) -> tuple[int, Any]:
    """Retry a cold-plan statement timeout with the established bounded read policy.

    A cold sample-pool plan can exceed the gateway statement timeout once under
    sweep concurrency; page fetches are idempotent reads, so the same bounded
    read-retry policy the staged loaders run applies. A genuinely slow route
    still fails on the last attempt.
    """
    last: MarinaRequestError = error
    for _ in (2, 3):
        time.sleep(5)
        try:
            return 200, reader.get(path)
        except MarinaRequestError as retry_error:
            if retry_error.status_code != 504:
                raise
            last = retry_error
    raise last


def inventory_timestamp_projection(inventory: dict[str, Any]) -> dict[str, Any]:
    """Compare timestamp instants across SQL-gateway strings and native API datetimes."""

    def timestamp(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return value
        if parsed.tzinfo is None:
            return value
        return parsed.astimezone(UTC).isoformat(timespec="microseconds")

    return {
        **inventory,
        **{key: timestamp(inventory[key]) for key in ("created_at", "synced_at") if key in inventory},
        "sources": [
            {**source, **({"updated_at": timestamp(source["updated_at"])} if "updated_at" in source else {})}
            for source in inventory["sources"]
        ],
    }


class SweepFamily(StrEnum):
    """The browser sweep's page families; a post-publish diff re-sweeps a changed family in full."""

    SOURCES = "sources"
    CURRICULUM = "curriculum"
    EVIDENCE = "evidence"
    READABILITY = "readability"


@dataclass(frozen=True)
class SweepUnit:
    """One independent slice of the browser sweep, run on its own page."""

    family: SweepFamily
    label: str
    run: Callable[[Page], list[dict[str, Any]]]


SWEEP_WORKERS = 16
"""Pages swept in parallel: each worker owns a browser and runs units on fresh contexts."""
EVIDENCE_CASES_PER_UNIT = 8
CURRICULUM_DOMAINS_PER_UNIT = 8


def hosted_sweep_baseline(path: Path, revision: int) -> dict[str, Any]:
    """The passing hosted receipt's full browser sweep for this revision."""
    hosted = json.loads(path.read_text())
    if hosted.get("phase") != "hosted" or hosted.get("passed") is not True or hosted.get("revision") != revision:
        raise ValueError(f"{path} is not a passing hosted receipt for revision {revision}")
    browser = hosted.get("browser") or {}
    if browser.get("scope") != BrowserScope.FULL.value or not browser.get("api_response_sha256"):
        raise ValueError("The hosted receipt carries no full browser sweep to compare against")
    return browser


POSTPUBLISH_READ_WORKERS = 16


def changed_api_keys(reader: CatalogReader, recorded: dict[str, str]) -> list[str]:
    """Every response the hosted sweep read whose served content has changed since, re-read uncached."""

    def current(key: str) -> str:
        reader.cache.pop(key, None)
        status, data = browser_api_response(reader, key, [])
        return json_sha256(data) if status == 200 else f"status:{status}"

    keys = sorted(recorded)
    with ThreadPoolExecutor(max_workers=POSTPUBLISH_READ_WORKERS) as pool:
        digests = dict(zip(keys, pool.map(current, keys), strict=True))
    return [key for key in keys if digests[key] != recorded[key]]


def sweep_family(key: str) -> SweepFamily:
    """The page family whose full sweep covers a served response."""
    path = urlsplit(key).path
    if path.startswith("api/curriculum"):
        return SweepFamily.CURRICULUM
    if path in ("api/sources", "api/source-statuses", "api/source-labels"):
        return SweepFamily.SOURCES
    if path.startswith("api/"):
        return SweepFamily.EVIDENCE
    return SweepFamily.READABILITY


def pending_release_sources(reader: CatalogReader) -> set[str]:
    """The candidate presentations the sweep covers, from the served package's manifest.

    The manifest ships with the package, not the API. A browser fetch of it from a page
    that never navigated is cross-origin and dies on CORS (R40 #62), so the reader reads it
    server-side in both the local and hosted modes.
    """
    manifest = reader.get("measurements/pending-filtered-releases.json")
    return {row["source_id"] for row in manifest["sources"]}


def browser_checks(
    reader: CatalogReader,
    root: str,
    expected: dict[str, Any],
    shots: Path,
    *,
    local: bool,
    stored_groups: list[dict[str, Any]],
    scope: BrowserScope = BrowserScope.REPRESENTATIVE,
    full_families: frozenset[SweepFamily] = frozenset(),
    workers: int = 0,
) -> dict[str, Any]:
    """The browser sweep at ``scope``, with ``full_families`` swept in full regardless.

    The sweep runs as independent per-layout units on ``workers`` parallel pages
    (``SWEEP_WORKERS`` when 0).
    """
    started = time.monotonic()
    workers = workers or SWEEP_WORKERS
    releases = reader.get("api/curriculum/releases")["releases"]
    inventory = reader.get("measurements/oracle-inventory.json")
    errors: list[str] = []
    if not local:
        survey = reader.get("api/source-duplicates")
        expected_groups = {
            row["duplicate_group"]["group_id"]: row["duplicate_group"]
            for row in expected["sources"]
            if row["duplicate_group"]
        }
        if sorted(survey["groups"], key=lambda group: group["group_id"]) != sorted(
            expected_groups.values(), key=lambda group: group["group_id"]
        ):
            raise ValueError("Hosted duplicate groups differ from the frozen-review projection")
        expected_notes = {row["id"]: row["partial_overlaps"] for row in expected["sources"] if row["partial_overlaps"]}
        if survey["partial_overlaps"] != expected_notes:
            raise ValueError("Hosted partial-overlap notes differ from their directional measurements")
        if sorted(survey["stored_groups"], key=lambda group: group["group_id"]) != stored_groups:
            raise ValueError("Hosted duplicate API omitted or changed archived stored groups")

    def route_request(route: Route) -> None:
        parsed = urlsplit(route.request.url)
        path = parsed.path[len(urlsplit(root).path) :]
        if path.startswith("api/"):
            try:
                status, data = browser_api_response(
                    reader, path + ("?" + parsed.query if parsed.query else ""), expected["sources"]
                )
                if status != 200:
                    route.fulfill(status=status, json=data)
                    return
            except Exception as error:
                # Playwright reconstructs callback errors; custom constructors can mask the cause.
                raise RuntimeError(f"Release API {route.request.url}: {error}") from error
            if local and path == "api/source":
                row = next(item for item in expected["sources"] if item["id"] == data["declaration"]["id"])
                data = {
                    **data,
                    "declaration": {**data["declaration"], **{key: row.get(key) for key in PROFILE_FIELDS}},
                    "duplicate_group": row["duplicate_group"],
                    "partial_overlaps": row["partial_overlaps"],
                    "daytona_snapshot_limit": expected["daytona_snapshot_limit"],
                }
            if path == "api/sources":
                projection = dict(parse_qsl(parsed.query)).get("projection", "complete")
                projected = (
                    {**expected, "sources": [table_source_record(source) for source in expected["sources"]]}
                    if projection == "table"
                    else expected
                )
                if local:
                    data = projected
                elif inventory_timestamp_projection(data) != inventory_timestamp_projection(projected):
                    raise ValueError("Hosted source API differs from the input-bound baseline projection")
            route.fulfill(json=data)
        elif not local:
            route.continue_(
                headers={**route.request.headers, "Authorization": "Bearer " + bearer_token(reader.provider)}
            )
        else:
            route.continue_()

    full = {family for family in SweepFamily if family in full_families or scope is BrowserScope.FULL}
    pending = pending_release_sources(reader)
    units = sweep_units(reader, expected, releases, shots, full, pending, root=root)
    outcomes = run_sweep_units(units, root, route_request, errors, workers)
    by_family = {
        family: [row for unit, rows in outcomes if unit.family is family for row in rows] for family in SweepFamily
    }
    readability_views = by_family[SweepFamily.READABILITY]
    violations = sum(len(view["findings"]) for view in readability_views)
    if violations:
        raise ValueError(f"{violations} readability violations; see {shots}/readability/*/readability.json")
    if errors:
        raise ValueError(f"Browser JavaScript errors: {errors}")
    return {
        "scope": scope.value,
        "full_families": sorted(family.value for family in full),
        "workers": workers,
        "units": len(units),
        "seconds": time.monotonic() - started,
        "readability": {"violations": violations, "views": len(readability_views)},
        "source_views": by_family[SweepFamily.SOURCES],
        "curriculum_views": by_family[SweepFamily.CURRICULUM],
        "evidence_views": by_family[SweepFamily.EVIDENCE],
        "api_response_sha256": {key: json_sha256(value) for key, value in list(reader.cache.items())},
        "inventory_sha256": json_sha256(inventory),
        "errors": errors,
    }


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def sweep_units(
    reader: CatalogReader,
    expected: dict[str, Any],
    releases: list[dict[str, Any]],
    shots: Path,
    full: set[SweepFamily],
    pending: set[str],
    *,
    root: str,
) -> list[SweepUnit]:
    """Split the sweep into per-layout slices; full families also split their domains and cases."""
    sources = expected["sources"]
    units: list[SweepUnit] = []
    selected = next((row for row in releases if row["selected"]), releases[0])
    domains = sorted(
        {
            node["subject_id"]
            for node in reader.get(f"api/curriculum?{urlencode({'release': selected['release_id']})}")["hierarchy"]
            if node["kind"] == "subject"
        }
    )
    cases = evidence_cases(sources, pending)
    for layout in LAYOUTS:
        layouts = (layout,)
        name = f"{layout[0]}-{layout[1]}"
        units.append(
            SweepUnit(
                SweepFamily.SOURCES,
                f"sources {name}",
                partial(source_sweep, root=root, sources=sources, shots=shots, layouts=layouts),
            )
        )
        if SweepFamily.CURRICULUM in full:
            units.append(
                SweepUnit(
                    SweepFamily.CURRICULUM,
                    f"curriculum releases {name}",
                    partial(curriculum_release_sweep, root=root, releases=releases, layouts=layouts),
                )
            )
            for chunk in chunks(domains, CURRICULUM_DOMAINS_PER_UNIT):
                units.append(
                    SweepUnit(
                        SweepFamily.CURRICULUM,
                        f"curriculum domains {chunk[0]}-{chunk[-1]} {name}",
                        partial(curriculum_domain_sweep, root=root, shots=shots, layouts=layouts, domains=chunk),
                    )
                )
        else:
            units.append(
                SweepUnit(
                    SweepFamily.CURRICULUM,
                    f"curriculum {name}",
                    partial(curriculum_sweep, root=root, releases=releases, shots=shots, layouts=layouts),
                )
            )
        if SweepFamily.EVIDENCE in full:
            for chunk in chunks(cases, EVIDENCE_CASES_PER_UNIT):
                units.append(
                    SweepUnit(
                        SweepFamily.EVIDENCE,
                        f"evidence {chunk[0]['id']} {name}",
                        partial(evidence_sweep, root=root, sources=sources, layouts=layouts, cases=chunk),
                    )
                )
        else:
            units.append(
                SweepUnit(
                    SweepFamily.EVIDENCE,
                    f"evidence {name}",
                    partial(evidence_smoke, root=root, sources=sources, layouts=layouts),
                )
            )
        readability = readability_sweep if SweepFamily.READABILITY in full else readability_smoke
        units.append(
            SweepUnit(
                SweepFamily.READABILITY,
                f"readability {name}",
                partial(_readability_views, readability, root=root, shots=shots / "readability" / name, layouts=layouts),
            )
        )
    return units


def _readability_views(sweep: Callable[..., dict[str, Any]], page: Page, **kwargs: Any) -> list[dict[str, Any]]:
    return sweep(page, **kwargs)["views"]


def run_sweep_units(
    units: list[SweepUnit],
    root: str,
    route_request: Callable[[Route], None],
    errors: list[str],
    workers: int,
) -> list[tuple[SweepUnit, list[dict[str, Any]]]]:
    """Run the units on a bounded pool of pages; every failure is reported with its unit."""
    pending: queue.SimpleQueue[int] = queue.SimpleQueue()
    for index in range(len(units)):
        pending.put(index)
    results: dict[int, list[dict[str, Any]]] = {}
    failures: list[str] = []

    def worker() -> None:
        # Sync Playwright is per-thread: each worker starts its own driver and browser.
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                while True:
                    try:
                        index = pending.get_nowait()
                    except queue.Empty:
                        return
                    unit = units[index]
                    context = browser.new_context()
                    try:
                        context.route(root + "**", route_request)
                        page = context.new_page()
                        page.set_default_navigation_timeout(60000)
                        page.on("pageerror", lambda error, label=unit.label: errors.append(f"{label}: {error}"))
                        page.on("response", report_failed_response)
                        page.on("requestfailed", report_failed_request)
                        results[index] = unit.run(page)
                    except Exception as error:
                        failures.append(f"{unit.label}: {type(error).__name__}: {error}")
                    finally:
                        context.close()
            finally:
                browser.close()

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(units)))) as pool:
        for future in [pool.submit(worker) for _ in range(max(1, min(workers, len(units))))]:
            future.result()
    if failures:
        raise ValueError(f"{len(failures)} browser sweep unit(s) failed: " + " | ".join(failures))
    return [(unit, results[index]) for index, unit in enumerate(units)]


def run_command(
    command: list[str], label: str, artifacts: Path, receipt: ReleaseReceipt, *, env: dict[str, str] | None = None
) -> None:
    LOGGER.info("Gate: %s", label)
    receipt.gate = label
    log = artifacts / f"{label}.log"
    with log.open("x") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, env=env)
    receipt.commands.append(CommandResult(label, command, result.returncode, str(log)))
    if result.returncode:
        raise RuntimeError(f"{label} failed (exit {result.returncode}); see {log}")


def app_details(service_url: str, app_id: str, session: requests.Session) -> dict[str, Any]:
    value = marina_request(service_url, "GET", f"/api/marina/applets/{app_id}", session=session)
    if not isinstance(value, dict):
        raise ValueError("Marina app details are not a JSON object")
    return value


def run_checks(args: argparse.Namespace, artifacts: Path, receipt: ReleaseReceipt) -> None:
    receipt.gate = "workspace-environment"
    check_workspace_imports()
    receipt.gate = "landed-cut"
    head = landed_cut(args.attested_revision)
    receipt.source_revision = head
    baseline = PreflightBaseline.model_validate_json(args.baseline.read_bytes()) if args.baseline else None
    if baseline and baseline.production_version != args.production_version:
        raise ValueError("The production hold must match the preflight baseline")
    if not args.source_counts and baseline is None:
        raise ValueError("A count manifest or successful preflight baseline is required")
    count_raw = args.source_counts.read_bytes() if args.source_counts else baseline.inputs.source_counts_json.encode()
    counts = SourceCounts.model_validate_json(count_raw)
    if baseline and hashlib.sha256(count_raw).hexdigest() != baseline.inputs.source_counts_sha256:
        raise ValueError("Hosted count inputs must match the preflight baseline")
    inputs = ReleaseInputs(hashlib.sha256(count_raw).hexdigest(), count_raw.decode())
    receipt.inputs = inputs
    package = args.dist.resolve() if args.dist else None
    if package:
        receipt.gate = "package-inputs"
        inventory_raw = args.oracle_inventory.read_bytes()
        receipt.inputs = replace(inputs, oracle_inventory_sha256=hashlib.sha256(inventory_raw).hexdigest())
        receipt.package = check_package(package, head, count_raw, inventory_raw)
        run_command(["uv", "run", "--no-sync", "marina", "validate", str(package)], "package", artifacts, receipt)
    # The code gates bind the checkout, not the revision: a hosted check at the
    # preflight's own checkout is covered by that passing preflight receipt.
    if args.postpublish or (baseline is not None and baseline.source_revision == head):
        receipt.code_gates_from = baseline.source_revision if baseline is not None else None
    else:
        lint_paths = git(
            "ls-files", str(APP_SOURCE), "docs/references/rl-data-atlas.md", "docs/references/task-curation.md"
        ).splitlines()
        run_command(["./infra/pre-commit.py", *lint_paths], "lint", artifacts, receipt)
        with test_database_environment() as environment:
            run_command(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "pytest",
                    "infra/marina/tests/test_rl_data_catalog.py",
                    "infra/marina/tests/test_private_tier.py",
                    "experiments/post_training/task_curation/tests/test_oracle_admitted_release.py",
                    str(APP_SOURCE / "tests"),
                    "-q",
                ],
                "api-count-tests",
                artifacts,
                receipt,
                env=environment,
            )
            run_command(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "marina",
                    "journey",
                    "rl_data_catalog",
                    "--apps-dir",
                    str(ROOT / "infra/marina/apps"),
                    "-k",
                    "not full_browser",
                    "--shots",
                    str(artifacts / "journeys"),
                ],
                "browser-journeys",
                artifacts,
                receipt,
                env=environment,
            )
    with requests.Session() as session, ExitStack() as resources:
        receipt.gate = "production-version"
        production = app_details(args.service_url, PRODUCTION_APP_ID, session)
        if production["current_version"] != args.production_version:
            raise ValueError("Production changed from the explicitly expected revision")
        details = app_details(args.service_url, PREVIEW_APP_ID, session)
        if details["mode"] != args.expected_mode:
            raise ValueError(f"Atlas mode changed from the explicitly expected {args.expected_mode!r}")
        revision = args.revision or details["current_version"]
        provider = marina_token_provider()
        query = applet_query(args.service_url, PREVIEW_APP_ID, session=session, token_provider=provider)
        reader = CatalogReader(
            f"{args.app_origin.rstrip('/')}/a/{PREVIEW_APP_ID}/v/{revision}/", session, query if package else None
        )
        if reader.candidate_api is not None:
            resources.callback(reader.candidate_api.close)
        receipt.gate = "catalog-history-baseline"
        before = catalog_state(query)
        if args.identity_measurement:
            selected = [row for row in before["identity_selection"] if row["load_id"] == before["selected"]["load_id"]]
            if selected != [{"load_id": before["selected"]["load_id"], "measurement_id": args.identity_measurement}]:
                raise ValueError("Selected overlap evidence differs from the explicitly pinned measurement")
        actual = reader.get("api/sources")
        if package or args.refresh_snapshot:
            expected = projected_duplicate_inventory(
                projected_contract_inventory(projected_sources(actual, counts), query),
                query,
                before["selected"]["load_id"],
            )
        else:
            if baseline is None:
                raise ValueError("Hosted checks require a preflight baseline")
            expected = baseline.expected_sources
        receipt.evidence_before = before
        receipt.production_version = args.production_version
        receipt.preview_base_version = details["current_version"]
        receipt.expected_sources = expected
        receipt.gate = "curriculum-api-invariants"
        receipt.api_invariants = curriculum_invariants(reader.get, reader.get("api/curriculum/releases")["releases"])
        receipt.api_invariants["evidence_browsers"] = browse_invariants(reader.get, expected["sources"])
        if package:
            receipt.gate = "curriculum-latency"
            receipt.api_invariants["curriculum_latency"] = curriculum_latency(
                reader, reader.get("api/curriculum/releases")["releases"]
            )
        stored_groups = stored_duplicate_groups(query, before["selected"]["load_id"])
        if baseline:
            verify_preservation(baseline.evidence_before, before, refresh_snapshot=args.refresh_snapshot)
            provenance = reader.get("bundled-model-provenance.json")
            if provenance != baseline.package:
                raise ValueError("Hosted package provenance differs from the checked preflight package")
        if package:
            reader.cache["measurements/oracle-inventory.json"] = json.loads(
                (package / "dist/measurements/oracle-inventory.json").read_text()
            )
            receipt.gate = "local-browser-sweep"
            with static_preview(package / "dist") as root:
                receipt.gate = "browser-sweep"
                receipt.browser = browser_checks(
                    reader,
                    root,
                    expected,
                    artifacts / "screenshots",
                    local=True,
                    stored_groups=stored_groups,
                    scope=BrowserScope.REPRESENTATIVE,
                )
        else:
            if baseline is None:
                raise ValueError("Hosted checks require a preflight baseline")
            full_families: frozenset[SweepFamily] = frozenset()
            if args.postpublish:
                receipt.gate = "postpublish-diff"
                hosted = hosted_sweep_baseline(args.hosted_receipt, args.revision)
                changed = changed_api_keys(reader, hosted["api_response_sha256"])
                full_families = frozenset(sweep_family(key) for key in changed)
                receipt.api_invariants["postpublish_diff"] = {
                    "hosted_keys": len(hosted["api_response_sha256"]),
                    "changed_keys": changed,
                    "full_families": sorted(full_families),
                }
            receipt.gate = "hosted-browser-sweep"
            # The hosted check sweeps the new revision in full once; post-publish
            # re-sweeps in full only the families whose served content changed since.
            receipt.browser = browser_checks(
                reader,
                reader.root,
                expected,
                artifacts / "screenshots",
                local=False,
                stored_groups=stored_groups,
                scope=BrowserScope.REPRESENTATIVE if args.postpublish else BrowserScope.FULL,
                full_families=full_families,
            )
            if receipt.browser["inventory_sha256"] != baseline.browser["inventory_sha256"]:
                raise ValueError("Hosted oracle inventory differs from the preflight input")
        receipt.gate = "catalog-history-readback"
        after = catalog_state(query)
        receipt.gate = "history-preservation"
        verify_preservation(before, after)
        receipt.evidence_after = after
        receipt.gate = "production-version-after"
        production_after = app_details(args.service_url, PRODUCTION_APP_ID, session)
        if production_after["current_version"] != args.production_version:
            raise ValueError("Production changed during the release check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dist", type=Path, help="Prepared preview PACKAGE directory, including applet.toml")
    mode.add_argument("--revision", type=int, help="Published immutable preview revision")
    parser.add_argument("--source-counts", type=Path, help="Explicit pinned count manifest (required for --dist)")
    parser.add_argument("--oracle-inventory", type=Path, help="Explicit pinned inventory (required for --dist)")
    parser.add_argument(
        "--identity-measurement", help="Explicit selected overlap survey SHA-256 for an intended refresh"
    )
    parser.add_argument("--baseline", type=Path, help="Successful preflight receipt (required for --revision)")
    parser.add_argument("--production-version", type=int, required=True)
    parser.add_argument(
        "--expected-mode",
        choices=("private", "public"),
        default="private",
        help="Expected target access mode; this checker never changes it",
    )
    parser.add_argument(
        "--refresh-snapshot", help="Explicit intended selected catalog snapshot; historical evidence must still remain"
    )
    parser.add_argument("--service-url", default="https://marina.oa.dev")
    parser.add_argument("--app-origin", default="https://applets.marina.oa.dev")
    parser.add_argument(
        "--postpublish",
        action="store_true",
        help="Re-check the published revision against its hosted check: re-sweep changed families in full",
    )
    parser.add_argument(
        "--hosted-receipt", type=Path, help="The revision's passing hosted receipt (required for --postpublish)"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--attested-revision",
        help="The landed revision an in-region release bundle was attested at (replaces the git checks)",
    )
    args = parser.parse_args()
    if args.dist and (not args.source_counts or not args.oracle_inventory):
        parser.error("--dist requires --source-counts and --oracle-inventory")
    if args.revision and not args.baseline:
        parser.error("--revision requires --baseline")
    if args.postpublish and (not args.revision or not args.hosted_receipt):
        parser.error("--postpublish requires --revision, its preflight --baseline and its --hosted-receipt")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    artifacts = args.receipt.with_suffix("")
    artifacts.mkdir(parents=True)
    (artifacts / "screenshots").mkdir()
    receipt = ReleaseReceipt(
        phase="postpublish" if args.postpublish else "preflight" if args.dist else "hosted",
        revision=args.revision,
        started_at=datetime.now(UTC).isoformat(),
        target=ReleaseTarget(PREVIEW_APP_ID, args.app_origin, args.service_url),
        production_id=PRODUCTION_APP_ID,
    )
    with args.receipt.open("x") as output:
        try:
            run_checks(args, artifacts, receipt)
            receipt.passed = True
        except Exception as error:
            receipt.error = ReleaseFailure(receipt.gate, type(error).__name__, str(error))
            LOGGER.error("Release gate failed: %s", error)
        receipt.finished_at = datetime.now(UTC).isoformat()
        json.dump(asdict(receipt), output, indent=2)
        output.write("\n")
    LOGGER.info("Receipt: %s", args.receipt)
    raise SystemExit(0 if receipt.passed else 1)


if __name__ == "__main__":
    main()
