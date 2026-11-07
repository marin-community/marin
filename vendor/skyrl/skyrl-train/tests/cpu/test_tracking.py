# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import fsspec
from skyrl_train.io import io
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
                    "iris_job_id": "/tests/tracker-trainer",
                },
            }
        )
        startup = {
            "run_id": "native-run",
            "attempt_id": attempt,
            "iris_job_id": "/tests/tracker-trainer",
            "coordinator_iris_job_id": "/tests/coordinator",
            "started_at": before.isoformat(),
            "engine": {"vllm_version": "fixture-vllm"},
            "wandb_url": None,
        }
        path = tmp_path / f"exports/attempts/{attempt}/tracking.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(startup))
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
        assert record == startup | {"wandb_url": record["wandb_url"]}
    assert failed.read_bytes() == retained
    assert [(row.get("id"), row["group"], row["resume"]) for row in launches] == [
        ("failed-attempt", "native-run", "allow"),
        ("current-attempt", "native-run", "allow"),
    ]
    assert [row["config"]["attempt_id"] for row in launches] == ["failed-attempt", "current-attempt"]
    assert all(row["config"]["iris_job_id"] == "/tests/tracker-trainer" for row in launches)


def test_metrics_cleanup_uses_object_store_listing_paths(monkeypatch):
    export_root = "s3://tracker-test/exports"
    filesystem = fsspec.filesystem("memory")
    list_directory = filesystem.ls

    def list_objects(path, detail=True, **kwargs):
        entries = list_directory(path, detail=detail, **kwargs)
        return entries if detail else [entry.removeprefix("s3://") for entry in entries]

    monkeypatch.setattr(filesystem, "ls", list_objects)
    resolve = io.filesystem_and_path
    monkeypatch.setattr(
        io, "filesystem_and_path", lambda uri: (filesystem, uri) if uri.startswith("s3://") else resolve(uri)
    )
    for step in (0, 8):
        io.write_bytes_atomic(f"{export_root}/metrics/step-{step:06d}.json", json.dumps({"old": 1}).encode())
    config = OmegaConf.create({"trainer": {"export_path": export_root}, "runtime": {"attempt_id": "new"}})
    with Tracking("project", "name", "files", config) as tracker:
        tracker.start(resumed_step=None)
        tracker.log({"loss": 0.25}, step=0)
    assert not io.exists(f"{export_root}/metrics/step-000008.json")
    assert json.loads(io.read_bytes(f"{export_root}/metrics/step-000000.json")) == {
        "step": 0,
        "attempt_id": "new",
        "loss": 0.25,
    }
