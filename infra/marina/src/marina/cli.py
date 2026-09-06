# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``marina`` command: check manifests, build frontends, run the kernel."""

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from urllib.parse import urlparse

import click
import pytest
import uvicorn

from marina.applets import (
    APPLET_MANIFEST,
    AppletPackage,
    AppletStore,
    package_applet,
    parse_applet_manifest,
    read_applet_package,
)
from marina.apps import is_python_app, migration, services_for
from marina.client import DEFAULT_MARINA_URL, marina_request, publish_applet
from marina.database_setup import APPLET_READER_ROLE, ensure_applet_provisioning
from marina.db import (
    DatabaseSpec,
    UrlDatabase,
    database_from_env,
    deployment_runner_lock,
    grant_read,
    migration_lock,
    runner_lock,
    schema_name,
)
from marina.journey_plugin import DEFAULT_SHOTS_DIR, JOURNEYS_DIR
from marina.journeys import running_kernel
from marina.local_postgres import docker_database
from marina.manifest import AppManifest, JobRunner, discover_apps, job_runners
from marina.server import APPS_DIR_ENV, MarinaConfig, create_app
from marina.table_load import read_table, table_statements

# The environment wins so the deployed image (MARINA_APPS_DIR=/app/apps) runs `marina migrate` unchanged.
DEFAULT_APPS_DIR = Path(os.environ.get(APPS_DIR_ENV) or Path(__file__).resolve().parents[2] / "apps")
# Where `marina dev` reads app data: one directory per app, mirroring the production bucket.
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / ".data"
DEFAULT_PORT = 8080


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
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--only", multiple=True, help="Migrate only these apps.")
@click.option("--reader", help="Postgres role to grant read access on every schema migrated.")
def migrate(apps_dir: Path, only: tuple[str, ...], reader: str | None) -> None:
    """Run each Python app's migrate() against its schema on the database the environment names."""
    database = database_from_env(os.environ)
    if database is None:
        raise click.UsageError("no database configured: set MARINA_DATABASE_URL or CLOUDSQL_CONNECTION")
    _migrate_apps(discover_apps(apps_dir), database, only, reader)


def _migrate_apps(apps: list[AppManifest], database: DatabaseSpec, only: tuple[str, ...], reader: str | None) -> None:
    with migration_lock(database):
        applet_store = AppletStore(database)
        try:
            ensure_applet_provisioning(applet_store.engine)
            applet_store.migrate()
        finally:
            applet_store.engine.dispose()
        for app in apps:
            if only and app.name not in only:
                continue
            if not is_python_app(app):
                continue
            run = migration(app)
            if run is None:
                continue
            click.echo(f"== {app.name}: migrate")
            engine = services_for(app, "", database).engine()
            try:
                run(engine)
                grant_read(engine, schema_name(app.name), APPLET_READER_ROLE)
                if reader:
                    grant_read(engine, schema_name(app.name), reader)
            finally:
                engine.dispose()


def _runner(runner_name: str, apps: list[AppManifest]) -> JobRunner:
    for runner in job_runners(apps):
        if runner.name == runner_name:
            return runner
    raise click.UsageError(f"unknown job runner {runner_name!r}")


@cli.command("run")
@click.argument("runner_name")
@click.option("--apps-dir", type=click.Path(path_type=Path), default=DEFAULT_APPS_DIR, show_default=True)
@click.option("--reader", help="Postgres role to grant read access on every schema migrated.")
@click.option("--migrate-only", is_flag=True, help="Apply migrations without running app jobs.")
def run_jobs(runner_name: str, apps_dir: Path, reader: str | None, migrate_only: bool) -> None:
    """Migrate, then execute every app job assigned to RUNNER_NAME."""
    apps = discover_apps(apps_dir)
    selected = _runner(runner_name, apps)
    database = database_from_env(os.environ)
    if database is None:
        raise click.UsageError("no database configured: set MARINA_DATABASE_URL or CLOUDSQL_CONNECTION")

    lock = deployment_runner_lock(database, runner_name) if migrate_only else runner_lock(database, runner_name)
    with lock as acquired:
        if not acquired:
            click.echo(f"runner {runner_name}: another execution holds the lease; skipping")
            return
        _migrate_apps(apps, database, (), reader)
        if migrate_only:
            return

        runner_secrets = {secret for bound in selected.jobs for secret in bound.job.secrets}
        failures: list[str] = []
        for bound in selected.jobs:
            click.echo(f"== {bound.qualified_name}: {' '.join(bound.job.command)}")
            child_env = dict(os.environ)
            for secret in runner_secrets - set(bound.job.secrets):
                child_env.pop(secret, None)
            try:
                completed = subprocess.run(
                    bound.job.command,
                    cwd=bound.app.root,
                    env=child_env,
                    timeout=bound.job.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                click.echo(f"{bound.qualified_name}: {exc}", err=True)
                failures.append(bound.qualified_name)
                continue
            if completed.returncode != 0:
                failures.append(bound.qualified_name)
        if failures:
            raise click.ClickException(f"job failures: {', '.join(failures)}")


def _publish_package(app_dir: Path, *, build: bool) -> tuple[bytes, AppletPackage]:
    manifest_path = app_dir / APPLET_MANIFEST
    if not manifest_path.is_file():
        raise click.UsageError(f"{app_dir} has no {APPLET_MANIFEST}")
    manifest = parse_applet_manifest(manifest_path.read_bytes())
    if build and manifest.build_command is not None:
        subprocess.run(manifest.build_command, shell=True, cwd=app_dir, check=True)
    elif build and not (app_dir / "dist" / "index.html").is_file():
        raise click.UsageError("dist/index.html is absent and applet.toml has no build_command")
    try:
        payload = package_applet(app_dir)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    return payload, read_applet_package(payload)


def _print_dry_run(package: AppletPackage, *, json_output: bool) -> None:
    files = sorted(package.files)
    report = {
        "files": files,
        "file_count": len(files),
        "byte_size": package.byte_size,
        "digest": package.digest.hex(),
    }
    if json_output:
        click.echo(json.dumps(report, sort_keys=True))
        return
    click.echo(f"{report['file_count']} files, {report['byte_size']} bytes, sha256 {report['digest']}")
    for path in files:
        click.echo(path)


def _print_publish_result(result: dict[str, object], service_url: str, *, json_output: bool) -> None:
    if not isinstance(result.get("url"), str) or not urlparse(str(result["url"])).netloc:
        result["url"] = service_url.rstrip("/") + str(result["path"])
    if json_output:
        click.echo(json.dumps(result, sort_keys=True))
        return
    click.echo(result["url"])


def _serve_local_applet(payload: bytes, *, json_output: bool) -> None:
    try:
        with docker_database() as database_url:
            database = UrlDatabase(database_url)
            _migrate_apps([], database, (), None)
            with tempfile.TemporaryDirectory(prefix="marina-local-applet-") as directory:
                root = Path(directory)
                apps_dir = root / "apps"
                data_root = root / "data"
                apps_dir.mkdir()
                data_root.mkdir()
                config = MarinaConfig(
                    apps_dir=apps_dir,
                    data_root=str(data_root),
                    iap_audience=None,
                    database=database,
                )
                with running_kernel(config) as kernel:
                    result = publish_applet(kernel.origin, payload)
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
@click.option("--url", "service_url", default=DEFAULT_MARINA_URL, show_default=True)
@click.option("--update", "applet_id", default=None, help="Applet UUID to update.")
@click.option("--base-version", type=int, default=None, help="Current version required by an update.")
@click.option("--build/--no-build", default=True, help="Run a declared build_command before packaging.")
@click.option("--local", is_flag=True, help="Run this applet against disposable Postgres and Marina.")
@click.option("--dry-run", is_flag=True, help="Validate and print package contents without publishing.")
@click.option("--json", "json_output", is_flag=True, help="Print the publish result as JSON.")
def publish(
    app_dir: Path,
    service_url: str,
    applet_id: str | None,
    base_version: int | None,
    build: bool,
    local: bool,
    dry_run: bool,
    json_output: bool,
) -> None:
    """Publish a built applet directory and print its URL."""
    if local:
        if dry_run:
            raise click.UsageError("--local cannot be combined with --dry-run")
        if applet_id is not None or base_version is not None:
            raise click.UsageError("--local cannot be combined with --update or --base-version")
    payload, package = _publish_package(app_dir, build=build)
    if local:
        _serve_local_applet(payload, json_output=json_output)
        return
    if dry_run:
        _print_dry_run(package, json_output=json_output)
        return
    if (applet_id is None) != (base_version is None):
        raise click.UsageError("--update and --base-version must be supplied together")
    try:
        result = publish_applet(service_url, payload, applet_id=applet_id, base_version=base_version)
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error
    _print_publish_result(result, service_url, json_output=json_output)


@cli.group()
def applets() -> None:
    """Inspect and manage published applets."""


def _applet_origin(service_url: str, applet_id: str) -> str:
    details = marina_request(service_url, "GET", f"/api/marina/applets/{applet_id}")
    if not isinstance(details, dict) or not isinstance(details.get("url"), str):
        raise RuntimeError("Marina returned invalid applet details")
    parsed = urlparse(details["url"])
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
        click.echo(f"{applet['name']}  v{applet['version']}  {applet['title']}  {applet['path']}")


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
        origin = _applet_origin(service_url, applet_id)
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
        origin = _applet_origin(service_url, applet_id)
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
@click.option("--host", default="127.0.0.1", show_default=True)
def dev(apps_dir: Path, data_root: str, port: int, host: str) -> None:
    """Serve the apps on loopback with the anonymous admin identity."""
    config = MarinaConfig(
        apps_dir=apps_dir, data_root=data_root, iap_audience=None, database=database_from_env(os.environ)
    )
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
def serve() -> None:
    """Serve for production: configuration from the environment, bound on all interfaces."""
    config = MarinaConfig.from_env(DEFAULT_APPS_DIR)
    click.echo(
        f"marina: apps from {config.apps_dir}, data from {config.data_root}, "
        f"IAP audience {config.iap_audience or 'unset'}"
    )
    uvicorn.run(create_app(config), host="0.0.0.0", port=int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    cli()
