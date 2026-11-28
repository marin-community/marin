# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The preview applet's live revision and access mode, read at stage time.

The release tail continues whatever revision is live and never changes the access
mode: the hosted check verifies the live revision, the publish compare-and-sets
against it, and both record the mode they observed.
"""

from dataclasses import dataclass

import requests

from infra.marina.apps.rl_data_catalog.release_check import app_details


@dataclass(frozen=True)
class LiveApplet:
    """One observation of an applet's live state."""

    version: int
    mode: str


def live_applet(service_url: str, applet_id: str) -> LiveApplet:
    """Read the applet's current revision and access mode from Marina."""
    with requests.Session() as session:
        details = app_details(service_url, applet_id, session)
    version, mode = details.get("current_version"), details.get("mode")
    if not isinstance(version, int) or version <= 0 or mode not in ("private", "public"):
        raise ValueError(f"Marina reports no usable live state for {applet_id}: {version!r}, {mode!r}")
    return LiveApplet(version=version, mode=mode)
