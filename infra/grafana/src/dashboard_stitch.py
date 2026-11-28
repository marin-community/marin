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

A ``panelGroupRef`` expands an ordered array from a fragment's ``panels`` field,
including groups inside collapsed rows. Shared dashboard fragments live in
``panels/``; a dashboard's single-use fragments live in a directory named after
the dashboard (``dashboards/<stem>/``). Templating SQL expands the tokens in
``_SHARED_SQL`` at build time and raises on any other ``{{token}}`` so an
undeclared name cannot ship literally to Grafana; Grafana's own ``{{from}}`` and
``{{to}}`` macros pass through.

Targets with ``targetRef`` share endpoint parameters. Time-series targets default to
time/series/value columns; each target specifies its view, format and other columns.
"""

import argparse
import json
import re
from pathlib import Path

from rl_producer_policy import RL_PRODUCER_ROW_NAMES_SQL

PANEL_REF_KEY = "panelRef"
LINK_REF_KEY = "linkRef"
TARGET_REF_KEY = "targetRef"
PANEL_GROUP_REF_KEY = "panelGroupRef"

_SQL_TOKEN = re.compile(r"\{\{(\w+)\}\}")
# Grafana's own dashboard macros inside templating SQL pass through untouched.
GRAFANA_TEMPLATE_MACROS = frozenset({"from", "to"})
_SHARED_SQL = {"{{rl_producer_row_names}}": RL_PRODUCER_ROW_NAMES_SQL}

_TIMESERIES_COLUMNS = [
    {"selector": "t", "text": "time", "type": "timestamp_epoch"},
    {"selector": "series", "text": "series", "type": "string"},
    {"selector": "value", "text": "value", "type": "number"},
]

_RANGE_PARAMS = (("from", "${__from}"), ("to", "${__to}"), ("bucket_ms", "${__interval_ms}"))
_SHARED_TARGET_PARAMS = {
    "rl_run": (("clusters", "${cluster:csv}"), ("run", "${run}"), *_RANGE_PARAMS),
    "rl_execution": (
        ("clusters", "${cluster:csv}"),
        ("run", "${run}"),
        ("job", "${job}"),
        ("executions", "${execution:csv}"),
        *_RANGE_PARAMS,
    ),
}

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
_FLEET_ACCELERATORS_LINK = {
    "asDropdown": False,
    "icon": "dashboard",
    "includeVars": True,
    "keepTime": True,
    "targetBlank": False,
    "title": "Fleet accelerators",
    "type": "link",
    "url": "/d/marin-accel",
}
_FLEET_HEALTH_LINK = {
    "asDropdown": False,
    "icon": "dashboard",
    "includeVars": True,
    "keepTime": True,
    "targetBlank": False,
    "title": "Fleet health",
    "type": "link",
    "url": "/d/marin-clusters",
}
_RL_POST_TRAINING_LINK = {
    "asDropdown": False,
    "icon": "dashboard",
    "includeVars": True,
    "keepTime": True,
    "targetBlank": False,
    "title": "RL Post-training",
    "type": "link",
    "url": "/d/marin-rl-runs",
}
_SHARED_LINKS = {
    "cluster_capacity": _CLUSTER_CAPACITY_LINK,
    "cluster_capacity_without_vars": {**_CLUSTER_CAPACITY_LINK, "includeVars": False},
    "fleet_accelerators": _FLEET_ACCELERATORS_LINK,
    "fleet_accelerators_without_vars": {**_FLEET_ACCELERATORS_LINK, "includeVars": False},
    "fleet_health": _FLEET_HEALTH_LINK,
    "fleet_health_without_vars": {**_FLEET_HEALTH_LINK, "includeVars": False},
    "rl_post_training": _RL_POST_TRAINING_LINK,
}


def load_panel_fragments(panels_dir: Path) -> dict[str, dict]:
    """Read every panels/<name>.json fragment, keyed by its filename stem."""
    return {path.stem: json.loads(path.read_text()) for path in panels_dir.glob("*.json")}


def _stitch_target(target: dict) -> dict:
    ref = target.get(TARGET_REF_KEY)
    if ref is None:
        return target
    if ref not in _SHARED_TARGET_PARAMS:
        raise KeyError(f"unknown target parameter set {ref!r}")
    params = (*_SHARED_TARGET_PARAMS[ref], ("view", target["view"]))
    columns = target.get("columns", _TIMESERIES_COLUMNS) if target["format"] == "timeseries" else target["columns"]
    return {
        "refId": target["refId"],
        "type": "json",
        "source": "url",
        "format": target["format"],
        "parser": "backend",
        "url": target["url"],
        "url_options": {"method": "GET", "params": [{"key": key, "value": value} for key, value in params]},
        "columns": [dict(column) for column in columns],
    }


def _stitch_panels(panels: list[dict], fragments: dict[str, dict]) -> list[dict]:
    """Resolve markers in one panels array, descending into collapsed rows.

    Grafana nests a collapsed row's children under the row's own ``panels``, so a
    fragment placed inside a row has to be resolved there too.
    """
    resolved = []
    for panel in panels:
        group_ref = panel.get(PANEL_GROUP_REF_KEY)
        if group_ref is not None:
            resolved.extend(_stitch_panels(fragments[group_ref]["panels"], fragments))
            continue
        ref = panel.get(PANEL_REF_KEY)
        if ref is not None:
            if ref not in fragments:
                raise KeyError(f"panel {panel.get('id')} references unknown panel fragment {ref!r}")
            panel = {**fragments[ref], "id": panel["id"], "gridPos": panel["gridPos"]}
        if panel.get("panels"):
            panel = {**panel, "panels": _stitch_panels(panel["panels"], fragments)}
        if "targets" in panel:
            panel = {**panel, "targets": [_stitch_target(target) for target in panel["targets"]]}
        resolved.append(panel)
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
        resolved.append(dict(_SHARED_LINKS[ref]))
    return resolved


def _stitch_sql_fragments(value, fragments: dict[str, str]):
    if isinstance(value, str):
        for token, replacement in fragments.items():
            value = value.replace(token, replacement)
        unresolved = {name for name in _SQL_TOKEN.findall(value) if name not in GRAFANA_TEMPLATE_MACROS}
        if unresolved:
            raise ValueError(f"unresolved templating SQL token(s) {sorted(unresolved)}; declare them in _SHARED_SQL")
        return value
    if isinstance(value, list):
        return [_stitch_sql_fragments(item, fragments) for item in value]
    if isinstance(value, dict):
        return {key: _stitch_sql_fragments(item, fragments) for key, item in value.items()}
    return value


def stitch_dashboard(source: dict, fragments: dict[str, dict]) -> dict:
    """Resolve panel groups, shared panels, links, targets and picker SQL.

    Raises:
        KeyError: A panel, link or target references an unknown fragment.
        ValueError: Templating SQL contains a token outside _SHARED_SQL and
            Grafana's template macros.
    """
    dashboard = dict(source)
    dashboard["panels"] = _stitch_panels(source["panels"], fragments)
    if "templating" in source:
        dashboard["templating"] = _stitch_sql_fragments(source["templating"], _SHARED_SQL)
    if "links" in source:
        dashboard["links"] = _stitch_links(source["links"])
    return dashboard


def stitch_all(src_dir: Path, panels_dir: Path) -> dict[str, dict]:
    """Stitch every dashboard JSON file in src_dir, keyed by filename.

    Each dashboard also resolves fragments from its dashboard-local directory
    ``<src_dir>/<stem>/``; a name in both directories is an error.
    """
    shared = load_panel_fragments(panels_dir)
    stitched = {}
    for path in src_dir.glob("*.json"):
        fragments = dict(shared)
        local_dir = src_dir / path.stem
        if local_dir.is_dir():
            for name, fragment in load_panel_fragments(local_dir).items():
                if name in fragments:
                    raise KeyError(f"fragment {name!r} exists in both {panels_dir} and {local_dir}")
                fragments[name] = fragment
        stitched[path.name] = stitch_dashboard(json.loads(path.read_text()), fragments)
    return stitched


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
