# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.utils.tracking import Tracking


@pytest.mark.parametrize("wandb_enabled", [False, True])
def test_tracker_saves_the_actual_wandb_link_and_metric_files(tmp_path, monkeypatch, wandb_enabled):
    url = "https://wandb.example/team/project/runs/issued-id"
    run = SimpleNamespace(
        id="issued-id",
        url=url,
        define_metric=lambda *args, **kwargs: None,
        log=lambda **kwargs: None,
        finish=lambda **kwargs: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kwargs: run, Settings=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    config = OmegaConf.create({"trainer": {"export_path": str(tmp_path / "exports")}})
    backends = ["files", *(["wandb"] if wandb_enabled else [])]
    with Tracking("project", "name", backends, config) as tracker:
        path = tmp_path / "exports/tracking.json"
        assert path.exists() == wandb_enabled
        if wandb_enabled:
            assert json.loads(path.read_text()) == {"wandb_url": url}
        tracker.log({"loss": 1.25}, step=2)
    assert json.loads((tmp_path / "exports/metrics/step-000002.json").read_text())["loss"] == 1.25
