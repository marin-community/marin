"""Record the initialized W&B identity without requiring a live service."""

import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
from skyrl_train.utils import tracking


def test_initialized_wandb_identity_is_durable_before_training(tmp_path, monkeypatch):
    run = SimpleNamespace(entity="resolved-team", project="resolved-project", id="server-run-id",
                          url="https://wandb.ai/resolved-team/resolved-project/runs/server-run-id",
                          define_metric=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: run,
                                                            Settings=lambda **kwargs: kwargs))
    monkeypatch.setattr(tracking, "require_secret", lambda name: "test-only")
    monkeypatch.setattr(tracking.ray, "is_initialized", lambda: False)
    root = tmp_path / "not-yet-created"
    tracking.Tracking("requested-project", "human-run-name", "wandb",
                      OmegaConf.create({"trainer": {"export_path": str(root)}}))
    assert json.loads((root / "wandb-run.json").read_text()) == {
        "entity": run.entity, "project": run.project, "id": run.id, "url": run.url,
    }
