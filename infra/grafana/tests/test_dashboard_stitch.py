# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for stitching shared panel fragments into dashboard JSON."""

import json

import pytest
from dashboard_stitch import load_panel_fragments, stitch_all, stitch_dashboard

FRAGMENT = {"type": "table", "title": "Shared panel", "targets": [{"refId": "A"}]}


def test_stitch_dashboard_merges_fragment_with_local_id_and_grid_pos():
    source = {
        "panels": [
            {"id": 7, "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}, "panelRef": "shared"},
        ]
    }
    (panel,) = stitch_dashboard(source, {"shared": FRAGMENT})["panels"]
    assert panel == {**FRAGMENT, "id": 7, "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}}


def test_stitch_dashboard_leaves_non_ref_panels_untouched():
    inline_panel = {"id": 1, "type": "row", "title": "Section"}
    source = {"panels": [inline_panel]}
    assert stitch_dashboard(source, {})["panels"] == [inline_panel]


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
