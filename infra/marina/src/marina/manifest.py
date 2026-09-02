# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The app manifest: what ``apps/<name>/app.toml`` declares and how the kernel finds it.

An app is a directory under the apps root holding an ``app.toml``. The manifest carries
only what the kernel needs to serve the app: its display name, the origins its page may
fetch, and how to build its frontend. Unknown keys are an error so a typo cannot silently
disable something.
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

MANIFEST_FILE = "app.toml"
DIST_DIR = "dist"
APP_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
KNOWN_KEYS = frozenset({"title", "description", "connect_src", "build_command"})


@dataclass(frozen=True)
class AppManifest:
    name: str
    title: str
    description: str
    root: Path
    connect_src: tuple[str, ...] = ()
    build_command: str | None = None

    @property
    def path(self) -> str:
        """The URL prefix the app is served under."""
        return f"/{self.name}/"

    @property
    def dist(self) -> Path:
        """Where the built frontend lives; served verbatim with index.html as the SPA fallback."""
        return self.root / DIST_DIR


def load_manifest(app_dir: Path) -> AppManifest:
    """Parse ``app_dir/app.toml``; raise ValueError on a missing, unknown, or malformed key."""
    name = app_dir.name
    if not APP_NAME_PATTERN.match(name):
        raise ValueError(f"app directory {name!r} must match {APP_NAME_PATTERN.pattern}")
    manifest_path = app_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"{app_dir} has no {MANIFEST_FILE}")
    raw = tomllib.loads(manifest_path.read_text())
    unknown = set(raw) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"{manifest_path}: unknown keys {sorted(unknown)}")
    for key in ("title", "description"):
        if key not in raw:
            raise ValueError(f"{manifest_path}: missing required key {key!r}")
    return AppManifest(
        name=name,
        title=raw["title"],
        description=raw["description"],
        root=app_dir,
        connect_src=tuple(raw.get("connect_src", [])),
        build_command=raw.get("build_command"),
    )


def discover_apps(apps_dir: Path) -> list[AppManifest]:
    """Load every app under ``apps_dir`` in name order.

    Directories starting with ``_`` or ``.`` are not apps. Any other directory without a
    manifest is an error rather than silently skipped.
    """
    if not apps_dir.is_dir():
        raise ValueError(f"apps directory {apps_dir} does not exist")
    return [
        load_manifest(child)
        for child in sorted(apps_dir.iterdir())
        if child.is_dir() and not child.name.startswith(("_", "."))
    ]
