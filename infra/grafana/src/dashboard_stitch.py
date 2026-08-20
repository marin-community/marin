# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stitch shared panel and link fragments into dashboard JSON at image build time.

A dashboard's ``panels`` array can hold a stitch marker instead of a full panel
body: ``{"id": N, "gridPos": {...}, "panelRef": "<fragment-name>"}``. Every other
field of the rendered panel — type, title, description, datasource, fieldConfig,
options, targets — comes from ``panels/<fragment-name>.json``, the single source of
truth for a panel shared across dashboards. ``id`` and ``gridPos`` stay
dashboard-local: they are the only two things that legitimately vary by placement.

This keeps ``dashboards/*.json`` file-provisioned and git-reviewable end to end —
no Grafana library-panel API, no runtime sync, no new credential — while killing
copy-pasted panel bodies that drift out of sync with the bridge's actual schema.
Dashboard links use ``{"linkRef": "<fragment-name>"}`` markers resolved from
``SHARED_LINKS`` for the same reason.
"""

import argparse
import json
from pathlib import Path

PANEL_REF_KEY = "panelRef"
LINK_REF_KEY = "linkRef"

_CLUSTER_CAPACITY_LINK = {
    "asDropdown": False,
    "icon": "dashboard",
    "includeVars": True,
    "keepTime": True,
    "targetBlank": False,
    "title": "Cluster capacity",
    "type": "link",
    "url": "/d/marin-cluster-capacity",
}
_SHARED_LINKS = {
    "cluster_capacity": _CLUSTER_CAPACITY_LINK,
    "cluster_capacity_without_vars": {**_CLUSTER_CAPACITY_LINK, "includeVars": False},
}


def load_panel_fragments(panels_dir: Path) -> dict[str, dict]:
    """Read every panels/<name>.json fragment, keyed by its filename stem."""
    return {path.stem: json.loads(path.read_text()) for path in panels_dir.glob("*.json")}


def _stitch_panels(panels: list[dict], fragments: dict[str, dict]) -> list[dict]:
    """Resolve markers in one panels array, descending into collapsed rows.

    Grafana nests a collapsed row's children under the row's own ``panels``, so a
    fragment placed inside a row has to be resolved there too.
    """
    resolved = []
    for panel in panels:
        ref = panel.get(PANEL_REF_KEY)
        if ref is None:
            if panel.get("panels"):
                panel = {**panel, "panels": _stitch_panels(panel["panels"], fragments)}
            resolved.append(panel)
            continue
        if ref not in fragments:
            raise KeyError(f"panel {panel.get('id')} references unknown panel fragment {ref!r}")
        resolved.append({**fragments[ref], "id": panel["id"], "gridPos": panel["gridPos"]})
    return resolved


def _stitch_links(links: list[dict]) -> list[dict]:
    resolved = []
    for link in links:
        ref = link.get(LINK_REF_KEY)
        if ref is None:
            resolved.append(link)
            continue
        if ref not in _SHARED_LINKS:
            raise KeyError(f"unknown dashboard link fragment {ref!r}")
        resolved.append(_SHARED_LINKS[ref])
    return resolved


def stitch_dashboard(source: dict, fragments: dict[str, dict]) -> dict:
    """Replace panelRef and linkRef markers with their fragment bodies.

    Raises:
        KeyError: A panel or link references an unknown fragment.
    """
    dashboard = {**source, "panels": _stitch_panels(source["panels"], fragments)}
    if "links" in source:
        dashboard["links"] = _stitch_links(source["links"])
    return dashboard


def stitch_all(src_dir: Path, panels_dir: Path) -> dict[str, dict]:
    """Stitch every dashboard JSON file in src_dir, keyed by filename."""
    fragments = load_panel_fragments(panels_dir)
    return {path.name: stitch_dashboard(json.loads(path.read_text()), fragments) for path in src_dir.glob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", type=Path, required=True, help="Directory of dashboard JSON with panelRef markers.")
    parser.add_argument("--panels-dir", type=Path, required=True, help="Directory of shared panel fragment JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write stitched dashboard JSON to.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, dashboard in stitch_all(args.src_dir, args.panels_dir).items():
        (args.out_dir / name).write_text(json.dumps(dashboard, indent=2) + "\n")


if __name__ == "__main__":
    main()
