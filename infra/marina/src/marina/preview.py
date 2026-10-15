# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install source packages into an explicitly chosen local preview database."""

from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from rigging.server_auth import ANONYMOUS_ADMIN

from marina.applets import AppletStore, package_applet, read_applet_package
from marina.database_setup import ensure_applet_provisioning
from marina.db import DatabaseSpec
from marina.releases import RELEASE_CONFIG, release_applet_id
from marina.sources import discover_apps


def install_preview_sources(apps_dir: Path, database: DatabaseSpec) -> None:
    """Install built source apps in a disposable preview registry."""
    store = AppletStore(database)
    try:
        ensure_applet_provisioning(store.engine)
        store.migrate()
        existing = {applet.id: applet.current_version for applet in store.active_applets()}
        for source in discover_apps(apps_dir):
            applet_id = (
                UUID(release_applet_id(source.root))
                if (source.root / RELEASE_CONFIG).exists()
                else uuid5(NAMESPACE_URL, f"marina-preview:{source.name}")
            )
            package = read_applet_package(package_applet(source.root))
            owner = ANONYMOUS_ADMIN.user_id
            store.publish(package, owner, applet_id, existing.get(applet_id, 0), frozenset({owner}))
    finally:
        store.engine.dispose()
