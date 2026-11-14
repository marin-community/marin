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
import subprocess
import threading
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

import requests
from marina.client import applet_query, bearer_token, marina_request, marina_token_provider
from playwright.sync_api import Route, sync_playwright

from .prepare_preview import (
    APP_SOURCE,
    IMPORT_RELOCATIONS,
    PREVIEW_APP_ID,
    PRODUCTION_APP_ID,
    RELOCATED_SERVER_FILES,
    oracle_inventory_payload,
)
from .release_browser import curriculum_sweep, evidence_sweep, source_sweep
from .release_evidence import (
    SqlQuery,
    catalog_state,
    json_sha256,
    projected_duplicate_inventory,
    projected_sources,
    stored_duplicate_groups,
    verify_preservation,
)
from .release_receipts import (
    CommandResult,
    PreflightBaseline,
    ReleaseFailure,
    ReleaseInputs,
    ReleaseReceipt,
    ReleaseTarget,
)
from .server.curriculum import (
    AREA_SQL,
    RELEASE_SQL,
    TASKS_SQL,
    CoverageQuery,
    area_parameters,
    area_response,
    display_hierarchy,
    task_parameters,
    task_response,
)
from .server.source_counts import SourceCounts

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[4]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def landed_cut() -> str:
    """Refuse uncommitted source and cuts absent from the freshly fetched integration branch."""
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
        self.query = query
        self.root = root
        self.session = session
        self.provider = marina_token_provider()
        self.cache: dict[str, Any] = {}
        self.releases: dict[str, Any] = {}

    def get(self, path: str) -> Any:
        url = urlsplit(path)
        query = urlencode(sorted((key, value) for key, value in parse_qsl(url.query) if value))
        key = url.path + ("?" + query if query else "")
        if key not in self.cache:
            params = dict(parse_qsl(url.query))
            if self.query is not None and url.path == "api/source-labels":
                catalog = self.get("api/sources")
                data = {
                    "sources": [
                        {key: row[key] for key in ("id", "display_name", "origin", "build_path")}
                        for row in catalog["sources"]
                    ],
                    "snapshot_id": catalog["snapshot_id"],
                }
            elif self.query is not None and url.path in {"api/curriculum/area", "api/curriculum/tasks"}:
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
                if url.path == "api/curriculum/area":
                    rows = self.query(AREA_SQL, area_parameters(metadata, filters, subject, node, scope))["rows"]
                    data = area_response(metadata, rows)
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
                    )
                    data = task_response(metadata, self.query(TASKS_SQL, bound)["rows"], limit)
            else:
                data = marina_request(self.root, "GET", "/" + key, session=self.session, token_provider=self.provider)
                if self.query is not None and url.path == "api/curriculum":
                    assert isinstance(data, dict)
                    data = {**data, "hierarchy": display_hierarchy(data["hierarchy"])}
            self.cache[key] = data
        return self.cache[key]


def browser_checks(
    reader: CatalogReader,
    root: str,
    expected: dict[str, Any],
    shots: Path,
    *,
    local: bool,
    stored_groups: list[dict[str, Any]],
) -> dict[str, Any]:
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
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            context = browser.new_context()

            def route_request(route: Route) -> None:
                parsed = urlsplit(route.request.url)
                path = parsed.path[len(urlsplit(root).path) :]
                if path.startswith("api/"):
                    data = reader.get(path + ("?" + parsed.query if parsed.query else ""))
                    if local and path == "api/source":
                        row = next(item for item in expected["sources"] if item["id"] == data["declaration"]["id"])
                        data = {
                            **data,
                            "duplicate_group": row["duplicate_group"],
                            "partial_overlaps": row["partial_overlaps"],
                            "daytona_snapshot_limit": expected["daytona_snapshot_limit"],
                        }
                    if path == "api/sources":
                        if local:
                            data = expected
                        elif data != expected:
                            raise ValueError("Hosted source API differs from the input-bound baseline projection")
                    route.fulfill(json=data)
                elif not local:
                    route.continue_(
                        headers={**route.request.headers, "Authorization": "Bearer " + bearer_token(reader.provider)}
                    )
                else:
                    route.continue_()

            context.route(root + "**", route_request)
            page = context.new_page()
            page.set_default_navigation_timeout(60000)
            page.on("pageerror", lambda error: errors.append(str(error)))
            sources = source_sweep(page, root, expected["sources"], shots)
            curriculum = curriculum_sweep(page, root, releases, shots)
            evidence = evidence_sweep(page, root, expected["sources"])
            if errors:
                raise ValueError(f"Browser JavaScript errors: {errors}")
            return {
                "source_views": sources,
                "curriculum_views": curriculum,
                "evidence_views": evidence,
                "api_response_sha256": {key: json_sha256(value) for key, value in reader.cache.items()},
                "inventory_sha256": json_sha256(inventory),
                "errors": errors,
            }
        finally:
            browser.close()


def run_command(command: list[str], label: str, artifacts: Path, receipt: ReleaseReceipt) -> None:
    LOGGER.info("Gate: %s", label)
    receipt.gate = label
    log = artifacts / f"{label}.log"
    with log.open("x") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
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
    head = landed_cut()
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
    lint_paths = git(
        "ls-files", str(APP_SOURCE), "docs/references/rl-data-atlas.md", "docs/references/task-curation.md"
    ).splitlines()
    run_command(["./infra/pre-commit.py", *lint_paths], "lint", artifacts, receipt)
    run_command(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "infra/marina/tests/test_rl_data_catalog.py",
            str(APP_SOURCE / "tests"),
            "-q",
        ],
        "api-count-tests",
        artifacts,
        receipt,
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
            "--shots",
            str(artifacts / "journeys"),
        ],
        "browser-journeys",
        artifacts,
        receipt,
    )
    with requests.Session() as session:
        receipt.gate = "production-version"
        production = app_details(args.service_url, PRODUCTION_APP_ID, session)
        if production["current_version"] != args.production_version:
            raise ValueError("Production changed from the explicitly expected revision")
        details = app_details(args.service_url, PREVIEW_APP_ID, session)
        if details["mode"] != "private":
            raise ValueError("The preview app is no longer private")
        revision = args.revision or details["current_version"]
        provider = marina_token_provider()
        query = applet_query(args.service_url, PREVIEW_APP_ID, session=session, token_provider=provider)
        reader = CatalogReader(
            f"{args.app_origin.rstrip('/')}/a/{PREVIEW_APP_ID}/v/{revision}/", session, query if package else None
        )
        before = catalog_state(query)
        actual = reader.get("api/sources")
        if package or args.refresh_snapshot:
            expected = projected_duplicate_inventory(
                projected_sources(actual, counts), query, before["selected"]["load_id"]
            )
        else:
            if baseline is None:
                raise ValueError("Hosted checks require a preflight baseline")
            expected = baseline.expected_sources
        receipt.evidence_before = before
        receipt.production_version = args.production_version
        receipt.preview_base_version = details["current_version"]
        receipt.expected_sources = expected
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
            with static_preview(package / "dist") as root:
                receipt.gate = "browser-sweep"
                receipt.browser = browser_checks(
                    reader, root, expected, artifacts / "screenshots", local=True, stored_groups=stored_groups
                )
        else:
            if baseline is None:
                raise ValueError("Hosted checks require a preflight baseline")
            receipt.gate = "hosted-browser-sweep"
            receipt.browser = browser_checks(
                reader, reader.root, expected, artifacts / "screenshots", local=False, stored_groups=stored_groups
            )
            if receipt.browser["inventory_sha256"] != baseline.browser["inventory_sha256"]:
                raise ValueError("Hosted oracle inventory differs from the preflight input")
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
    parser.add_argument("--baseline", type=Path, help="Successful preflight receipt (required for --revision)")
    parser.add_argument("--production-version", type=int, required=True)
    parser.add_argument(
        "--refresh-snapshot", help="Explicit intended selected catalog snapshot; historical evidence must still remain"
    )
    parser.add_argument("--service-url", default="https://marina.oa.dev")
    parser.add_argument("--app-origin", default="https://applets.marina.oa.dev")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.dist and (not args.source_counts or not args.oracle_inventory):
        parser.error("--dist requires --source-counts and --oracle-inventory")
    if args.revision and not args.baseline:
        parser.error("--revision requires --baseline")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    artifacts = args.receipt.with_suffix("")
    artifacts.mkdir(parents=True)
    (artifacts / "screenshots").mkdir()
    receipt = ReleaseReceipt(
        phase="preflight" if args.dist else "hosted",
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
