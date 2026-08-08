# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load reproducible SQL workloads from checked-in Grafana dashboard JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_UNRESOLVED_MACRO = re.compile(r"\$\{|\{\{")


@dataclass(frozen=True)
class DashboardQuery:
    name: str
    panel_title: str
    target: str
    sql: str


@dataclass(frozen=True)
class DashboardCorpus:
    uid: str
    title: str
    refresh: str
    queries: tuple[DashboardQuery, ...]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def render_sql(
    sql: str,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    clusters: tuple[str, ...],
) -> str:
    """Resolve the Grafana macros used by Finelog dashboards to fixed values."""
    if start_ms >= end_ms:
        raise ValueError("start_ms must be less than end_ms")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    if not clusters:
        raise ValueError("at least one cluster is required")
    replacements = {
        "${cluster:sqlstring}": ",".join(_sql_string(cluster) for cluster in clusters),
        "${__interval_ms}": str(interval_ms),
        "{{from}}": f"to_timestamp_millis({start_ms})",
        "{{to}}": f"to_timestamp_millis({end_ms})",
        "now()": f"to_timestamp_millis({end_ms})",
    }
    rendered = sql
    for macro, value in replacements.items():
        rendered = rendered.replace(macro, value)
    if _UNRESOLVED_MACRO.search(rendered):
        raise ValueError(f"dashboard query retains an unresolved macro: {rendered}")
    return rendered


def _query_name(title: str, target: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    if not slug:
        raise ValueError(f"panel title has no usable query name: {title!r}")
    return slug if target == "A" else f"{slug}_{target.lower()}"


def _panels(panels: list[object]) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        flattened.append(panel)
        nested = panel.get("panels")
        if isinstance(nested, list):
            flattened.extend(_panels(nested))
    return flattened


def load_dashboard_corpus(
    path: Path,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    clusters: tuple[str, ...],
) -> DashboardCorpus:
    """Extract and render every Finelog SQL target from a dashboard."""
    document = json.loads(path.read_text())
    queries: list[DashboardQuery] = []
    seen_names: set[str] = set()
    for panel in _panels(document.get("panels", [])):
        title = panel.get("title")
        targets = panel.get("targets", [])
        if not isinstance(title, str) or not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, dict):
                continue
            target_name = target.get("refId", "A")
            options = target.get("url_options", {})
            params = options.get("params", []) if isinstance(options, dict) else []
            sql_values = [
                param.get("value") for param in params if isinstance(param, dict) and param.get("key") == "sql"
            ]
            if not sql_values:
                continue
            if len(sql_values) != 1 or not isinstance(sql_values[0], str) or not isinstance(target_name, str):
                raise ValueError(f"panel {title!r} target has an invalid SQL parameter")
            name = _query_name(title, target_name)
            if name in seen_names:
                raise ValueError(f"duplicate dashboard query name: {name}")
            seen_names.add(name)
            queries.append(
                DashboardQuery(
                    name=name,
                    panel_title=title,
                    target=target_name,
                    sql=render_sql(
                        sql_values[0],
                        start_ms=start_ms,
                        end_ms=end_ms,
                        interval_ms=interval_ms,
                        clusters=clusters,
                    ),
                )
            )
    if not queries:
        raise ValueError(f"dashboard contains no Finelog SQL targets: {path}")
    return DashboardCorpus(
        uid=str(document.get("uid", "")),
        title=str(document.get("title", "")),
        refresh=str(document.get("refresh", "")),
        queries=tuple(queries),
    )
