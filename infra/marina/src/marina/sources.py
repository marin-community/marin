# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect local application source without installing it into the runtime."""

from pathlib import Path

from marina.applets import parse_applet_manifest
from marina.manifest import MANIFEST_FILE, AppManifest


def load_manifest(app_dir: Path) -> AppManifest:
    manifest = parse_applet_manifest((app_dir / MANIFEST_FILE).read_bytes())
    return AppManifest(
        name=manifest.name or app_dir.name,
        title=manifest.title,
        description=manifest.description,
        root=app_dir,
        connect_src=manifest.connect_src,
        build_command=manifest.build_command,
        data_url=manifest.data_url,
        jobs=manifest.jobs,
        agent=manifest.agent,
    )


def discover_apps(apps_dir: Path) -> list[AppManifest]:
    """Inspect local source directories; never register or release them."""
    return [
        load_manifest(child)
        for child in sorted(apps_dir.iterdir())
        if child.is_dir() and not child.name.startswith(("_", "."))
    ]
