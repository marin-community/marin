# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from marin.execution.lazy import ArtifactStep
from rigging.provenance import Provenance

from experiments.post_training import snowball_rl as snowball
from experiments.post_training import snowball_rl_presets as presets


@pytest.fixture
def selection(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "SELECTION_ROOT", str(tmp_path))
    monkeypatch.setenv(presets.JUDGE_KEY, "fixture")
    root = tmp_path / "snowball-100k" / presets.DEFAULT_SELECTION_VERSION
    root.mkdir(parents=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "recipe": {"heldout": {"version": "2026.10.23.1", "rows": 910}},
                "validation_rows": 100,
                "full_validation_rows": 910,
                "grader_kind_counts": {"train": {"remote_judge": 1}},
            }
        )
    )
    return ArtifactStep.adopt("tests/candidate", "2026.10.26", str(root))


@pytest.fixture
def source_provenance(monkeypatch):
    monkeypatch.setattr(
        snowball, "_provenance", lambda **kwargs: Provenance("tree-one", "commit-one", False, "branch", "test")
    )
