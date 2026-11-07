# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from skyrl_train.utils.tracking import Tracking


def test_tracker_keeps_failed_attempt_identity_and_wandb_link_on_retry(tmp_path, monkeypatch):
    launches = []

    def init(**kwargs):
        launches.append(kwargs)
        if len(launches) == 1:
            raise ConnectionError("W&B initialization failed")
        identifier = kwargs.get("id", "issued-id")
        return SimpleNamespace(
            id=identifier,
            url=f"https://wandb.example/team/project/runs/{identifier}",
            define_metric=lambda *args, **kwargs: None,
            log=lambda **kwargs: None,
            finish=lambda **kwargs: None,
        )

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init, Settings=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    monkeypatch.setattr(
        "skyrl_train.utils.tracking.get_job_info", lambda: SimpleNamespace(job_id="/tests/tracker-trainer")
    )
    failed = tmp_path / "exports/attempts/failed-attempt/tracking.json"
    before = datetime.now(timezone.utc)
    for attempt in ("failed-attempt", "current-attempt"):
        config = OmegaConf.create(
            {
                "trainer": {"export_path": str(tmp_path / "exports")},
                "runtime": {
                    "run_id": "native-run",
                    "attempt_id": attempt,
                    "coordinator_iris_job_id": "/tests/coordinator",
                },
            }
        )
        if attempt == "failed-attempt":
            with pytest.raises(ConnectionError, match="W&B initialization failed"):
                Tracking("project", "display-name", ["files", "wandb"], config)
            retained = failed.read_bytes()
        else:
            with Tracking("project", "display-name", ["files", "wandb"], config):
                pass
        record = json.loads((tmp_path / f"exports/attempts/{attempt}/tracking.json").read_text())
        assert {key: record[key] for key in ("run_id", "attempt_id", "iris_job_id", "coordinator_iris_job_id")} == {
            "run_id": "native-run",
            "attempt_id": attempt,
            "iris_job_id": "/tests/tracker-trainer",
            "coordinator_iris_job_id": "/tests/coordinator",
        }
        assert record["wandb_url"] == (
            None if attempt == "failed-attempt" else f"https://wandb.example/team/project/runs/{attempt}"
        )
        assert before <= datetime.fromisoformat(record["started_at"]) <= datetime.now(timezone.utc)
    assert failed.read_bytes() == retained
    assert [(row.get("id"), row["group"], row["resume"]) for row in launches] == [
        ("failed-attempt", "native-run", "allow"),
        ("current-attempt", "native-run", "allow"),
    ]
    assert [row["config"]["attempt_id"] for row in launches] == ["failed-attempt", "current-attempt"]
