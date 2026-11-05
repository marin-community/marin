# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from skyrl_train.utils.tracking import Tracking


def test_tracker_saves_the_actual_wandb_link_and_metric_files(tmp_path, monkeypatch):
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
    with Tracking("project", "name", ["files", "wandb"], config):
        assert json.loads((tmp_path / "exports/tracking.json").read_text()) == {"wandb_url": url}
