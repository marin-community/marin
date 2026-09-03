# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``marina`` command: check manifests, build frontends, run the kernel."""

import os
import subprocess
from pathlib import Path

import click
import pytest
import uvicorn

from marina.apps import is_python_app, migration, services_for
from marina.db import database_from_env, grant_read, schema_name
from marina.journey_plugin import DEFAULT_SHOTS_DIR, JOURNEYS_DIR
from marina.manifest import discover_apps
from marina.server import APPS_DIR_ENV, MarinaConfig, create_app

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
    for app in discover_apps(apps_dir):
        if only and app.name not in only:
            continue
        if not is_python_app(app):
            continue
        run = migration(app)
        if run is None:
            continue
        click.echo(f"== {app.name}: migrate")
        engine = services_for(app, "", database).engine()
        run(engine)
        if reader:
            grant_read(engine, schema_name(app.name), reader)


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
