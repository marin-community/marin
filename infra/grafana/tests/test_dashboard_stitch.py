# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for stitching shared panel fragments into dashboard JSON."""

import json

import pytest
from dashboard_stitch import load_panel_fragments, stitch_all, stitch_dashboard

FRAGMENT = {"type": "table", "title": "Shared panel", "targets": [{"refId": "A"}]}


def test_stitch_dashboard_merges_fragment_with_local_id_and_grid_pos():
    grid = {"h": 8, "w": 12, "x": 0, "y": 0}
    params = [{"key": "identity", "value": "${identity}"}, {"key": "to", "value": "${__to}"}]
    fragment = {**FRAGMENT, "targets": [{"refId": "A", "url_options": {"params": params}}]}
    source = {
        "panels": [
            {"id": 7, "gridPos": grid, "panelRef": "shared"},
            {"id": 8, "gridPos": grid, "panelRef": "shared", "vars": {"identity": "${run}"}},
        ]
    }

    plain, mapped = stitch_dashboard(source, {"shared": fragment})["panels"]

    assert plain == {**fragment, "id": 7, "gridPos": grid}
    mapped_params = [{"key": "identity", "value": "${run}"}, {"key": "to", "value": "${__to}"}]
    assert mapped == {
        **fragment,
        "targets": [{"refId": "A", "url_options": {"params": mapped_params}}],
        "id": 8,
        "gridPos": grid,
    }


def test_stitch_dashboard_expands_target_refs_in_fragments_and_collapsed_rows():
    columns = [{"selector": "t", "text": "t", "type": "timestamp"}]
    short = {"refId": "A", "targetRef": "rl_run", "url": "/v1/rl/overview", "view": "spans", "format": "table"}
    fragment = {"type": "timeseries", "targets": [{**short, "columns": columns, "filterExpression": "x == 1"}]}
    source = {
        "panels": [
            {"id": 1, "gridPos": {"y": 0}, "panelRef": "shared"},
            {"id": 2, "type": "row", "panels": [{"id": 3, "targets": [{**short, "columns": columns}]}]},
        ]
    }

    first, row = stitch_dashboard(source, {"shared": fragment})["panels"]

    params = [
        {"key": "clusters", "value": "${cluster:csv}"},
        {"key": "run", "value": "${run}"},
        {"key": "from", "value": "${__from}"},
        {"key": "to", "value": "${__to}"},
        {"key": "bucket_ms", "value": "${__interval_ms}"},
        {"key": "view", "value": "spans"},
    ]
    expanded = {
        "refId": "A",
        "type": "json",
        "source": "url",
        "format": "table",
        "parser": "backend",
        "url": "/v1/rl/overview",
        "url_options": {"method": "GET", "params": params},
        "columns": columns,
    }
    assert first["targets"] == [{**expanded, "filterExpression": "x == 1"}]
    assert row["panels"] == [{"id": 3, "targets": [expanded]}]


def test_stitch_dashboard_leaves_non_ref_panels_untouched():
    inline_panel = {"id": 1, "type": "row", "title": "Section"}
    source = {"panels": [inline_panel]}
    assert stitch_dashboard(source, {})["panels"] == [inline_panel]


@pytest.mark.parametrize(
    ("link_ref", "expected_title", "expected_url", "expected_include_vars"),
    [
        ("async_rl", "RL Post-training (async)", "/d/marin-async-rl", True),
        ("cluster_capacity", "Cluster capacity", "/d/marin-cluster-capacity", True),
        ("fleet_accelerators_without_vars", "Fleet accelerators", "/d/marin-accel", False),
        ("fleet_health", "Fleet health", "/d/marin-clusters", True),
    ],
)
def test_stitch_dashboard_resolves_shared_links(
    link_ref: str, expected_title: str, expected_url: str, expected_include_vars: bool
):
    source = {"links": [{"linkRef": link_ref}], "panels": []}

    (link,) = stitch_dashboard(source, {})["links"]

    assert link["title"] == expected_title
    assert link["url"] == expected_url
    assert link["includeVars"] is expected_include_vars


def test_stitch_dashboard_returns_independent_shared_links():
    source = {"links": [{"linkRef": "async_rl"}], "panels": []}
    first = stitch_dashboard(source, {})
    first["links"][0]["title"] = "changed"

    second = stitch_dashboard(source, {})

    assert second["links"][0]["title"] == "RL Post-training (async)"


def test_stitch_dashboard_rejects_an_unknown_fragment_name():
    source = {"panels": [{"id": 7, "gridPos": {}, "panelRef": "missing"}]}
    with pytest.raises(KeyError, match="missing"):
        stitch_dashboard(source, {})


def test_load_panel_fragments_keys_by_filename_stem(tmp_path):
    (tmp_path / "control_plane_components.json").write_text(json.dumps(FRAGMENT))
    assert load_panel_fragments(tmp_path) == {"control_plane_components": FRAGMENT}


def test_stitch_all_processes_every_dashboard_in_a_directory(tmp_path):
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir()
    (panels_dir / "shared.json").write_text(json.dumps(FRAGMENT))
    (tmp_path / "a.json").write_text(json.dumps({"panels": [{"id": 1, "gridPos": {"h": 1}, "panelRef": "shared"}]}))
    (tmp_path / "b.json").write_text(json.dumps({"panels": []}))

    dashboards = stitch_all(tmp_path, panels_dir)

    assert set(dashboards) == {"a.json", "b.json"}
    assert dashboards["a.json"]["panels"][0]["title"] == "Shared panel"
    assert dashboards["b.json"]["panels"] == []
