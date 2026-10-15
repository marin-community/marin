# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``marina`` command: check manifests, build frontends, run the kernel."""

import json
import os
import subprocess
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import click
import pytest
import uvicorn

from marina.applets import (
    APPLET_MANIFEST,
    AppletMode,
    AppletPackage,
    AppletStore,
    package_applet,
    parse_applet_manifest,
    read_applet_package,
    validate_backend_import,
)
from marina.client import DEFAULT_MARINA_URL, MarinaRequestError, marina_request, publish_applet
from marina.database_setup import ensure_applet_provisioning
from marina.db import (
    DatabaseSpec,
    UrlDatabase,
    database_from_env,
    migration_lock,
    runner_lock,
)
from marina.jobs import execute_jobs
from marina.journey_plugin import DEFAULT_SHOTS_DIR, JOURNEYS_DIR
from marina.journeys import LOOPBACK, running_kernel
from marina.local_postgres import docker_database
from marina.preview import install_preview_sources
from marina.releases import (
    RELEASE_CONFIG,
    RELEASE_METADATA,
    ReleaseSource,
    clean_frontend_build,
    release_applet_id,
    release_source,
    require_current_main,
    require_tracked_package,
    source_status,
)
from marina.runtime import load_runtime_config
from marina.server import CLOUD_RUN_SERVICE_ENV, MarinaConfig, create_app
from marina.sources import discover_apps
from marina.table_load import read_table, table_statements

# Source inspection and previews use this directory; production serving uses Postgres.
DEFAULT_APPS_DIR = Path(__file__).resolve().parents[2] / "apps"
# Where `marina dev` reads app data: one directory per app, mirroring the production bucket.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / ".data"
DEFAULT_PORT = 8080


class _AppletValidationCheck(StrEnum):
    PACKAGE = "package"
    INLINE_SCRIPTS = "inline_scripts"
    BACKEND_IMPORT = "backend_import"
    JOB_IMPORT = "job_import"
    BACKEND_FACTORY_EXECUTION = "backend_factory_execution"
    MIGRATION_EXECUTION = "migration_execution"
    BROWSER = "browser"


@dataclass(frozen=True)
class _AppletValidationReport:
    files: tuple[str, ...]
    file_count: int
    byte_size: int
    digest: str
    checks: tuple[_AppletValidationCheck, ...]
    not_checked: tuple[_AppletValidationCheck, ...]


@click.group()
def cli() -> None:
    """Marina: many small apps, one process."""


@cli.command()
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
def check(apps_dir: Path) -> None:
    """Parse every app manifest and print what would be served."""
    for app in discover_apps(apps_dir):
        built = "built" if (app.dist / "index.html").is_file() else "not built"
        click.echo(f"{app.name:16} {built:10} {app.title}")


@cli.command()
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--only", multiple=True, help="Build only these apps.")
def build(apps_dir: Path, only: tuple[str, ...]) -> None:
    """Run each app's build_command in its directory."""
    for app in discover_apps(apps_dir):
        if only and app.name not in only:
            continue
        if app.build_command is None:
            continue
        click.echo(f"== {app.name}: {app.build_command}")
        subprocess.run(app.build_command, shell=True, cwd=app.root, check=True)


@cli.command()
@click.option(
    "--require-apps", is_flag=True, help="Require the explicitly released named applications before runtime cutover."
)
def migrate(require_apps: bool) -> None:
    """Upgrade Marina's registry; application schemas migrate only during release."""
    database = database_from_env(os.environ)
    if database is None:
        raise click.UsageError("no database configured: set MARINA_DATABASE_URL or CLOUDSQL_CONNECTION")
    _migrate_registry(database)
    if require_apps:
        store = AppletStore(database)
        try:
            actual = {app.name: app.id for app in store.active_applets()}
            load_runtime_config().require_registrations({name: applet_id for name, applet_id in actual.items() if name})
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
        finally:
            store.engine.dispose()


def _migrate_registry(database: DatabaseSpec) -> None:
    with migration_lock(database):
        store = AppletStore(database)
        try:
            ensure_applet_provisioning(store.engine)
            store.migrate()
        finally:
            store.engine.dispose()


@cli.command("run")
@click.argument("runner_name")
def run_jobs(runner_name: str) -> None:
    """Execute the current released modules assigned to a base-runtime runner."""
    database = database_from_env(os.environ)
    if database is None:
        raise click.UsageError("no database configured: set MARINA_DATABASE_URL or CLOUDSQL_CONNECTION")
    with runner_lock(database, runner_name) as acquired:
        if not acquired:
            click.echo(f"runner {runner_name}: another execution holds the lease; skipping")
            return
        store = AppletStore(database)
        try:
            execute_jobs(store, load_runtime_config(), runner_name)
        except (OSError, RuntimeError, ValueError) as error:
            raise click.ClickException(str(error)) from error
        finally:
            store.engine.dispose()


def _publish_package(
    app_dir: Path, *, build: bool, extra_files: dict[str, bytes] | None = None
) -> tuple[bytes, AppletPackage]:
    manifest_path = app_dir / APPLET_MANIFEST
    if not manifest_path.is_file():
        raise click.UsageError(f"{app_dir} has no {APPLET_MANIFEST}")
    manifest = parse_applet_manifest(manifest_path.read_bytes())
    if build and manifest.build_command is not None:
        subprocess.run(manifest.build_command, shell=True, cwd=app_dir, check=True)
    elif build and not (app_dir / "dist" / "index.html").is_file():
        raise click.UsageError("dist/index.html is absent and applet.toml has no build_command")
    try:
        payload = package_applet(app_dir, extra_files=extra_files)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    return payload, read_applet_package(payload)


def _validation_report(package: AppletPackage, *, imports_checked: bool) -> _AppletValidationReport:
    files = tuple(sorted(package.files))
    checks = [_AppletValidationCheck.PACKAGE, _AppletValidationCheck.INLINE_SCRIPTS]
    not_checked = [_AppletValidationCheck.BROWSER]
    if package.manifest.python_entrypoint is not None:
        not_checked = [
            _AppletValidationCheck.BACKEND_FACTORY_EXECUTION,
            _AppletValidationCheck.MIGRATION_EXECUTION,
            *not_checked,
        ]
        if imports_checked:
            checks.append(_AppletValidationCheck.BACKEND_IMPORT)
        else:
            not_checked.insert(0, _AppletValidationCheck.BACKEND_IMPORT)
    if package.manifest.jobs:
        (checks if imports_checked else not_checked).append(_AppletValidationCheck.JOB_IMPORT)
    return _AppletValidationReport(
        files=files,
        file_count=len(files),
        byte_size=package.byte_size,
        digest=package.digest.hex(),
        checks=tuple(checks),
        not_checked=tuple(not_checked),
    )


def _print_validation_report(package: AppletPackage, *, imports_checked: bool, json_output: bool) -> None:
    report = _validation_report(package, imports_checked=imports_checked)
    if json_output:
        click.echo(json.dumps(asdict(report), sort_keys=True))
        return
    click.echo(f"{report.file_count} files, {report.byte_size} bytes, sha256 {report.digest}")
    click.echo(f"Checks: {', '.join(report.checks)}")
    click.echo(f"Not checked: {', '.join(report.not_checked)}")
    for path in report.files:
        click.echo(path)


@cli.command()
@click.argument("app_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--build/--no-build", default=True, help="Run a declared build_command before packaging.")
@click.option("--json", "json_output", is_flag=True, help="Print the validation report as JSON.")
def validate(app_dir: Path, build: bool, json_output: bool) -> None:
    """Validate an applet package and import its backend in the Marina runtime."""
    _payload, package = _publish_package(app_dir, build=build)
    imports_checked = package.manifest.python_entrypoint is not None or bool(package.manifest.jobs)
    if imports_checked:
        try:
            validate_backend_import(package)
        except ValueError as error:
            raise click.UsageError(str(error)) from error
    _print_validation_report(package, imports_checked=imports_checked, json_output=json_output)


def _print_publish_result(result: dict[str, object], service_url: str, *, json_output: bool) -> None:
    if not isinstance(result.get("url"), str) or not urlparse(str(result["url"])).netloc:
        result["url"] = service_url.rstrip("/") + str(result["path"])
    if json_output:
        click.echo(json.dumps(result, sort_keys=True))
        return
    click.echo(result["url"])


def _serve_local_applet(payload: bytes, *, applet_id: str, mode: AppletMode | None, json_output: bool) -> None:
    try:
        with docker_database() as database_url:
            database = UrlDatabase(database_url)
            _migrate_registry(database)
            with tempfile.TemporaryDirectory(prefix="marina-local-applet-") as directory:
                root = Path(directory)
                apps_dir = root / "apps"
                data_root = root / "data"
                apps_dir.mkdir()
                data_root.mkdir()
                config = MarinaConfig(
                    data_root=str(data_root),
                    iap_audience=None,
                    database=database,
                    runtime=replace(load_runtime_config(), named_readers=()),
                )
                with running_kernel(config) as kernel:
                    result = publish_applet(
                        kernel.origin,
                        payload,
                        applet_id=applet_id,
                        base_version=0,
                        mode=mode.value if mode is not None else None,
                    )
                    _print_publish_result(result, kernel.origin, json_output=json_output)
                    click.echo("Serving until Ctrl-C; Postgres and Marina will be removed on exit.", err=True)
                    try:
                        threading.Event().wait()
                    except KeyboardInterrupt:
                        return
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip() if isinstance(error.stderr, bytes) else error.stderr
        raise click.ClickException(detail or str(error)) from error
    except (OSError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


@cli.command()
@click.argument("app_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--base-version", type=int, help="Expected live revision; use 0 for the first release.")
@click.option("--id", "explicit_id", help="Stable UUID for an applet without a release.toml declaration.")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--mode", type=click.Choice([mode.value for mode in AppletMode]), default=None)
@click.option("--local", is_flag=True, help="Preview using disposable Postgres and Marina.")
@click.option("--dry-run", is_flag=True, help="Build and validate without releasing.")
@click.option("--json", "json_output", is_flag=True)
def release(
    app_dir: Path,
    base_version: int | None,
    explicit_id: str | None,
    service_url: str,
    mode: str | None,
    local: bool,
    dry_run: bool,
    json_output: bool,
) -> None:
    """Build and release one application to its declared stable UUID."""
    if local and dry_run:
        raise click.UsageError("--local cannot be combined with --dry-run")
    if not local and not dry_run and base_version is None:
        raise click.UsageError("Supply --base-version; use 0 for the first release")
    publishing = not local and not dry_run
    registered = (app_dir / RELEASE_CONFIG).exists()
    guarded = publishing and registered
    try:
        source = release_source(app_dir) if guarded else None
        if guarded:
            require_current_main(app_dir)
        manifest = parse_applet_manifest((app_dir / APPLET_MANIFEST).read_bytes())
        extra_files = {RELEASE_METADATA: json.dumps(asdict(source)).encode()} if source is not None else None
        build_context = clean_frontend_build(app_dir) if manifest.build_command is not None else nullcontext()
        with build_context:
            payload, package = _publish_package(app_dir, build=True, extra_files=extra_files)
        if guarded:
            require_tracked_package(
                app_dir, package.files.keys() - {RELEASE_METADATA}, build_command=package.manifest.build_command
            )
        if package.manifest.python_entrypoint is not None or package.manifest.jobs:
            validate_backend_import(package)
        if dry_run:
            _print_validation_report(
                package,
                imports_checked=package.manifest.python_entrypoint is not None or bool(package.manifest.jobs),
                json_output=json_output,
            )
            return
        if local:
            _serve_local_applet(
                payload,
                applet_id=release_applet_id(app_dir) if registered else str(uuid4()),
                mode=AppletMode(mode) if mode is not None else None,
                json_output=json_output,
            )
            return
        if registered:
            applet_id = release_applet_id(app_dir)
            if explicit_id is not None and explicit_id != applet_id:
                raise ValueError("--id differs from the source directory's release.toml")
        elif explicit_id is not None:
            applet_id = explicit_id
        elif base_version == 0:
            applet_id = str(uuid4())
        else:
            raise ValueError("Supply --id for updates, or commit a release.toml declaration")
        if guarded and source is not None and require_current_main(app_dir) != source.commit:
            raise RuntimeError("The checkout changed while building; retry the release.")
        result = publish_applet(service_url, payload, applet_id=applet_id, base_version=base_version, mode=mode)
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip() if isinstance(error.stderr, bytes) else error.stderr
        raise click.ClickException(detail or str(error)) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if source is not None:
        result["source"] = asdict(source)
    _print_publish_result(result, service_url, json_output=json_output)


@cli.command("release-status")
@click.argument("app_dir", type=click.Path(path_type=Path, file_okay=False, exists=True))
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--json", "json_output", is_flag=True)
def release_status(app_dir: Path, service_url: str, json_output: bool) -> None:
    """Compare an applet's live source tree with this checkout's committed source."""
    try:
        applet_id = release_applet_id(app_dir)
        details = _applet_details(service_url, applet_id)
        version = details["current_version"]
        assert isinstance(version, int)
        origin = _applet_origin(service_url, details)
        try:
            metadata = marina_request(origin, "GET", f"/a/{applet_id}/v/{version}/marina-release.json")
        except MarinaRequestError as error:
            if error.status_code != 404:
                raise
            metadata = None
        if metadata is not None:
            assert isinstance(metadata, dict)
        actual = ReleaseSource(**metadata) if metadata is not None else None
        result = source_status(app_dir, applet_id, version, actual)
    except (OSError, RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    if json_output:
        click.echo(json.dumps(asdict(result), sort_keys=True))
    else:
        click.echo(f"{applet_id} v{version}: {result.state}")
        expected = result.checkout
        click.echo(f"Checkout: {expected.commit} {expected.path} (tree {expected.tree})")
        if actual is not None:
            click.echo(f"Released: {actual.commit} {actual.path} (tree {actual.tree})")


@cli.group()
def applets() -> None:
    """Inspect and manage published applets."""


def _applet_details(service_url: str, applet_id: str) -> dict[str, object]:
    details = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
    if not isinstance(details, dict) or not isinstance(details.get("authenticated_url"), str):
        raise RuntimeError("Marina returned invalid applet details")
    return details


def _applet_origin(service_url: str, details: dict[str, object]) -> str:
    assert isinstance(details["authenticated_url"], str)
    parsed = urlparse(details["authenticated_url"])
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else service_url.rstrip("/")


@applets.command("list")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--json", "json_output", is_flag=True)
def list_applets(service_url: str, json_output: bool) -> None:
    """List active applets."""
    try:
        result = marina_request(service_url, "GET", "/api/marina/applets")
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    if not isinstance(result, dict) or not isinstance(result.get("applets"), list):
        raise click.ClickException("Marina returned an invalid applet list")
    if json_output:
        click.echo(json.dumps(result, sort_keys=True))
        return
    for applet in result["applets"]:
        click.echo(f"{applet['name']}  v{applet['version']}  {applet['mode']}  {applet['title']}  {applet['path']}")


@applets.command("versions")
@click.argument("applet_id")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--json", "json_output", is_flag=True)
def list_applet_versions(applet_id: str, service_url: str, json_output: bool) -> None:
    """List the retained revisions of one applet."""
    try:
        result = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("current_version"), int)
        or not isinstance(result.get("versions"), list)
    ):
        raise click.ClickException("Marina returned invalid applet details")
    if json_output:
        click.echo(json.dumps(result, sort_keys=True))
        return
    for revision in result["versions"]:
        current = " current" if revision["version"] == result["current_version"] else ""
        click.echo(
            f"v{revision['version']}{current}  {revision['published_at']}  "
            f"{revision['published_by']}  {revision['byte_size']} bytes"
        )


@applets.command("mode")
@click.argument("applet_id")
@click.argument("mode", type=click.Choice([mode.value for mode in AppletMode]))
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def set_applet_mode(applet_id: str, mode: str, service_url: str) -> None:
    """Change an applet's access mode without publishing a new revision."""
    try:
        details = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
        if not isinstance(details, dict) or not isinstance(details.get("current_version"), int):
            raise RuntimeError("Marina returned invalid applet details")
        result = marina_request(
            service_url,
            "PUT",
            f"/api/marina/applets/{applet_id}/mode",
            json_body={"mode": mode, "base_version": details["current_version"]},
        )
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    if not isinstance(result, dict) or not isinstance(result.get("url"), str):
        raise click.ClickException("Marina returned invalid applet mode details")
    click.echo(result["url"])


@applets.command("rollback")
@click.argument("applet_id")
@click.argument("version", type=int)
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def rollback_applet(applet_id: str, version: int, service_url: str) -> None:
    """Move an applet's current URL to a retained revision."""
    try:
        details = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
        if not isinstance(details, dict) or not isinstance(details.get("current_version"), int):
            raise RuntimeError("Marina returned invalid applet details")
        marina_request(
            service_url,
            "PUT",
            f"/api/marina/applets/{applet_id}/current",
            json_body={"version": version, "base_version": details["current_version"]},
        )
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"{applet_id} now points to revision {version}")


@applets.command("archive")
@click.argument("applet_id")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def archive_applet(applet_id: str, service_url: str) -> None:
    """Hide an applet while preserving its schema and retained revisions."""
    try:
        marina_request(service_url, "DELETE", f"/api/marina/applets/{applet_id}")
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"archived {applet_id}")


@applets.command("sql")
@click.argument("applet_id")
@click.argument("sql")
@click.option("--parameters", default="{}", help="JSON object bound to SQL parameters.")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def applet_sql(applet_id: str, sql: str, parameters: str, service_url: str) -> None:
    """Execute one parameterized statement as an applet's database role."""
    try:
        values = json.loads(parameters)
        if not isinstance(values, dict):
            raise ValueError("--parameters must be a JSON object")
        origin = _applet_origin(service_url, _applet_details(service_url, applet_id))
        result = marina_request(
            origin,
            "POST",
            f"/a/{applet_id}/query",
            json_body={"sql": sql, "parameters": values},
        )
    except (RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))


@applets.group("table")
def applet_tables() -> None:
    """Load local tabular files into an applet schema."""


@applet_tables.command("load")
@click.argument("applet_id")
@click.argument("table_name")
@click.argument("source", type=click.Path(path_type=Path, dir_okay=False, exists=True))
@click.option("--replace", is_flag=True, help="Drop and recreate the table before loading.")
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
def load_applet_table(applet_id: str, table_name: str, source: Path, replace: bool, service_url: str) -> None:
    """Load JSON, JSONL, CSV, or Parquet rows into an applet table."""
    try:
        table = read_table(source)
        schema_sql, inserts = table_statements(table_name, table, replace=replace)
        origin = _applet_origin(service_url, _applet_details(service_url, applet_id))
        for statement in schema_sql:
            marina_request(
                origin,
                "POST",
                f"/a/{applet_id}/query",
                json_body={"sql": statement, "parameters": {}},
            )
        for statement, parameters in inserts:
            marina_request(
                origin,
                "POST",
                f"/a/{applet_id}/query",
                json_body={"sql": statement, "parameters": parameters},
            )
    except (RuntimeError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(f"loaded {table.num_rows} rows into {table_name}")


@cli.command()
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--data-root", default=str(DEFAULT_DATA_ROOT), show_default=True, help="Directory or gs:// URL.")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True)
@click.option("--host", default=LOOPBACK, show_default=True)
def dev(apps_dir: Path, data_root: str, port: int, host: str) -> None:
    """Serve the apps on loopback with the anonymous admin identity."""
    with docker_database() as database_url:
        database = UrlDatabase(database_url)
        install_preview_sources(apps_dir, database)
        config = MarinaConfig(data_root=data_root, iap_audience=None, database=database)
        uvicorn.run(create_app(config), host=host, port=port)


@cli.command()
@click.argument("apps", nargs=-1)
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--data-root", default=str(DEFAULT_DATA_ROOT), show_default=True)
@click.option("--shots", default=str(DEFAULT_SHOTS_DIR), show_default=True, help="Screenshot directory.")
@click.option("--video", is_flag=True, help="Record a video per journey.")
@click.option("--headed", is_flag=True, help="Show the browser.")
@click.option("-k", "keyword", default=None, help="Only journeys matching this pytest -k expression.")
def journey(
    apps: tuple[str, ...], apps_dir: Path, data_root: str, shots: str, video: bool, headed: bool, keyword: str | None
) -> None:
    """Walk through apps in a real browser: run apps/*/journeys against an in-process kernel."""
    targets = [apps_dir / name / JOURNEYS_DIR for name in apps] or sorted(apps_dir.glob(f"*/{JOURNEYS_DIR}"))
    missing = [str(target) for target in targets if not target.is_dir()]
    if missing:
        raise click.UsageError(f"no journeys at {missing}")
    args = [
        *map(str, targets),
        "-p",
        "marina.journey_plugin",
        "--journeys",
        f"--journey-apps-dir={apps_dir}",
        f"--journey-data-root={data_root}",
        f"--journey-shots={shots}",
        "-o",
        "addopts=",
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]
    if video:
        args.append("--journey-video")
    if headed:
        args.append("--journey-headed")
    if keyword:
        args += ["-k", keyword]
    raise SystemExit(pytest.main(args))


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--local-publisher", help="Owner ID for one-time bootstrap through a loopback operator kernel.")
def serve(host: str, local_publisher: str | None) -> None:
    """Serve for production: configuration from the environment, bound on all interfaces."""
    config = MarinaConfig.from_env()
    if local_publisher is not None:
        if host != LOOPBACK or config.iap_audience is not None or os.environ.get(CLOUD_RUN_SERVICE_ENV):
            raise click.UsageError("--local-publisher requires --host 127.0.0.1 outside Cloud Run and IAP")
        config = replace(config, local_publisher=local_publisher)
    click.echo(
        f"marina: released packages from Postgres, data from {config.data_root}, "
        f"IAP audience {config.iap_audience or 'unset'}"
    )
    uvicorn.run(create_app(config), host=host, port=int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    cli()
