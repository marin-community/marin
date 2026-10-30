# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a directory as a saved Python playground project."""

import io
import json
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import click
import requests
from marina.client import DEFAULT_MARINA_URL, DEFAULT_TIMEOUT, LOCAL_HOSTS, bearer_token, marina_request
from pydantic import BaseModel

from .server.app import MAX_FILES, MAX_INPUT_BYTES

EXCLUDED_DIRECTORIES = frozenset({"__pycache__", "node_modules", "output", "venv"})
APPLET_ID = tomllib.loads(Path(__file__).with_name("applet.toml").read_text())["applet_id"]


class AppletLocation(BaseModel):
    authenticated_url: str


def directory_archive(directory: Path) -> bytes:
    """Archive project files, excluding hidden paths, caches and generated output."""
    payload = io.BytesIO()
    total = 0
    count = 0
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory)
            if any(part.startswith(".") or part in EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            if path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise click.ClickException(f"Symlinks are not supported: {relative}")
            if not path.is_file():
                continue
            size = path.stat().st_size
            if total + size > MAX_INPUT_BYTES:
                raise click.ClickException("Directory exceeds the 5 MiB input limit")
            data = path.read_bytes()
            total += len(data)
            count += 1
            if total > MAX_INPUT_BYTES or count > MAX_FILES:
                raise click.ClickException("Directory exceeds the 5 MiB / 100 file input limit")
            archive.writestr(relative.as_posix(), data)
    return payload.getvalue()


@click.group()
def cli() -> None:
    """Manage Python playground projects."""


@cli.command()
@click.argument("directory", type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--title", default=None, help="Project name; defaults to the directory name.")
@click.option("--entrypoint", default="main.py", show_default=True)
@click.option("--package", "packages", multiple=True, help="Python requirement; repeat to override requirements.txt.")
@click.option(
    "--url", default=DEFAULT_MARINA_URL, show_default=True, help="Marina origin, including a local preview origin."
)
def upload(directory: Path, title: str | None, entrypoint: str, packages: tuple[str, ...], url: str) -> None:
    """Create a new saved project from DIRECTORY and print its ID and URL as JSON."""
    origin = url.rstrip("/")
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path:
        raise click.ClickException("--url must be a Marina origin without an application path")
    location = AppletLocation.model_validate(marina_request(origin, "GET", f"/api/marina/applets/{APPLET_ID}"))
    app_url = urljoin(origin + "/", location.authenticated_url).rstrip("/") + "/"
    headers = {"Content-Type": "application/zip"}
    if urlparse(app_url).hostname not in LOCAL_HOSTS:
        headers["Authorization"] = f"Bearer {bearer_token()}"
    params = [("title", title or directory.resolve().name), ("entrypoint", entrypoint)]
    params.extend(("packages", package) for package in packages)
    response = requests.post(
        app_url + "api/projects/import",
        headers=headers,
        params=params,
        data=directory_archive(directory),
        timeout=DEFAULT_TIMEOUT,
    )
    if not response.ok:
        raise click.ClickException(f"Upload failed ({response.status_code}): {response.text}")
    project = response.json()
    click.echo(
        json.dumps(
            {
                "id": project["id"],
                "revision": project["revision"],
                "url": f"{app_url}#/projects/{project['id']}",
            }
        )
    )


if __name__ == "__main__":
    cli()
