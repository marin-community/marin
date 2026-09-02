# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``marina`` command: check manifests, build frontends, run the kernel."""

import os
import subprocess
from pathlib import Path

import click
import uvicorn

from marina.manifest import discover_apps
from marina.server import APPS_DIR_ENV, MarinaConfig, create_app

DEFAULT_APPS_DIR = Path(__file__).resolve().parents[2] / "apps"
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
        click.echo(f"{app.name:16} {app.app_class:8} {built:10} {app.title}")


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
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
def dev(apps_dir: Path, port: int, host: str) -> None:
    """Serve the apps on loopback with the anonymous admin identity."""
    config = MarinaConfig(apps_dir=apps_dir, iap_audience=None)
    uvicorn.run(create_app(config), host=host, port=port)


@cli.command()
def serve() -> None:
    """Serve for production: configuration from the environment, bound on all interfaces."""
    os.environ.setdefault(APPS_DIR_ENV, str(DEFAULT_APPS_DIR))
    config = MarinaConfig.from_env()
    click.echo(f"marina: apps from {config.apps_dir}, IAP audience {config.iap_audience or 'unset'}")
    uvicorn.run(create_app(config), host="0.0.0.0", port=int(os.environ.get("PORT", DEFAULT_PORT)))


if __name__ == "__main__":
    cli()
