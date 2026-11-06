# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from marin.execution.lazy import ArtifactStep
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_scoring as scoring


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "SELECTION_ROOT", str(tmp_path))
    monkeypatch.setattr(presets, "artifact_prefix", lambda cluster: str(tmp_path))
    root = tmp_path / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "recipe": {"heldout": {"version": "2026.10.23.1", "rows": 910}},
                "validation_rows": 100,
                "full_validation_rows": 910,
                "grader_kind_counts": {"train": {"remote_judge": 0}, "validation": {}, "full_validation": {}},
            }
        )
    )
    panel_name, panel_version = presets.DEFAULT_TEST_PANEL.split("@")
    panel = tmp_path / panel_name / panel_version
    panel.mkdir(parents=True)
    (panel / "summary.json").write_text(
        json.dumps(
            {
                "validation_rows": 100,
                "full_validation_rows": 1500,
                "grader_kind_counts": {"validation": {}, "full_validation": {}},
                "recipe": {
                    "pass_rates": {
                        "uri": str(tmp_path / "test-panel-rates" / scoring.PASS_RATES_FILENAME),
                        "rows": 1500,
                        "model_revision": presets.MODEL_REVISION,
                        "version": panel_version,
                    }
                },
            }
        )
    )
    return ArtifactStep.adopt("tests/candidate", "2026.10.26", str(root))


@pytest.fixture
def source_provenance(monkeypatch):
    monkeypatch.setattr(
        snowball, "_provenance", lambda **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )


@pytest.fixture
def synthetic_current_controls(monkeypatch):
    monkeypatch.setattr(
        gate,
        "CONTROLS",
        {
            name: tuple(replace(pin, revision=presets.PRESET_REVISION) for pin in pins)
            for name, pins in gate.CONTROLS.items()
        },
    )
