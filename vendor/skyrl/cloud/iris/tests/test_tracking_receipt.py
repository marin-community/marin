"""Record the initialized W&B identity without requiring a live service."""

import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
from rigging.training_journal import TrainingJournal, set_training_journal, verify_training_journals
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


def test_local_history_survives_provider_logging_and_shutdown_failure(tmp_path, monkeypatch):
    class OfflineRun:
        id = "run-id"
        entity = "team"
        project = "project"
        url = "https://wandb.ai/team/project/runs/run-id"

        def define_metric(self, *args, **kwargs):
            pass

        def log(self, **kwargs):
            raise ConnectionError("provider unavailable")

        def finish(self, **kwargs):
            raise ConnectionError("provider unavailable")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **kwargs: OfflineRun(),
                                                            Settings=lambda **kwargs: kwargs))
    monkeypatch.setattr(tracking, "require_secret", lambda name: "test-only")
    monkeypatch.setattr(tracking.ray, "is_initialized", lambda: False)
    journal = TrainingJournal(str(tmp_path / "telemetry"), "run-id", "tracking", "trainer")
    set_training_journal(journal)
    tracker = tracking.Tracking("project", "run-name", "wandb", OmegaConf.create({"trainer": {
        "export_path": str(tmp_path / "exports"), "wandb_entity": "team",
    }}))
    with tracker:
        with journal.events.span("train"):
            tracker.log({"loss": 0.5}, step=1, commit=False)
            tracker.log({"loss": 0.25}, step=2, commit=True)
    receipts = verify_training_journals(str(tmp_path / "telemetry"), {"trainer"}, {"train"})
    assert receipts[0]["steps"] == [1, 2]
    rows = [json.loads(line) for line in (journal.destination / "history.jsonl").read_text().splitlines()]
    assert [(row["step"], row["metrics"], row["commit"]) for row in rows] == [
        (1, {"loss": 0.5}, False), (2, {"loss": 0.25}, True),
    ]
