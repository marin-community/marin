"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest
import yaml

from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.launch import ExportedPolicy, LaunchOutcome, LaunchState, execute_launch
from cloud.iris.iris_backend import supervise_iris_job
from iris.resources.state import JobState
from rigging.timing import Timestamp
from cloud.iris.tests.test_launch_config import _raw_config
from marinskyrl.checkpoint_paths import policy_export_path


@dataclass
class RecordingBackend:
    outcome: LaunchOutcome = LaunchOutcome("/user/job", "succeeded", 0)
    validated: bool = False
    launched: bool = False
    launched_config: dict | None = None
    exported: bool = False

    def validate(self, _config_path: Path) -> None:
        self.validated = True

    def launch(self, config_path: Path) -> LaunchOutcome:
        self.launched = True
        self.launched_config = yaml.safe_load(config_path.read_text())
        return self.outcome

    def export_terminal_policy(self, _config_path: Path) -> None:
        self.exported = True


def _config_path(tmp_path: Path, *, submission: str = "wait") -> Path:
    config = _raw_config()
    config["run"]["submission"] = submission
    config["inputs"]["train_data"] = [
        {
            "uri": "s3://data/gsm8k",
            "identity": "sha256:gsm8k",
            "local_path": "/tmp/data/gsm8k",
            "relative_path": "train.parquet",
            "kind": "directory",
        }
    ]
    path = tmp_path / "launch.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_prepare_validates_without_submitting(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    result = execute_launch(_config_path(tmp_path, submission="prepare"), backend=backend)

    assert result.state is LaunchState.PREPARED
    assert backend.validated
    assert not backend.launched


def test_wait_launches_exports_and_records_the_terminal_model(tmp_path: Path, monkeypatch) -> None:
    status = SimpleNamespace(
        state=JobState.SUCCEEDED,
        started_at=Timestamp.from_seconds(datetime(2026, 11, 7, 1, tzinfo=timezone.utc).timestamp()),
        finished_at=Timestamp.from_seconds(datetime(2026, 11, 7, 2, tzinfo=timezone.utc).timestamp()),
    )

    class NativeStatusBackend(RecordingBackend):
        def launch(self, config_path):
            super().launch(config_path)
            job = SimpleNamespace(job_id="/user/job", wait=lambda **kwargs: status)
            return supervise_iris_job(job)

    backend = NativeStatusBackend()
    writes: list[tuple[str, dict]] = []
    model = ExportedPolicy(
        policy_export_uri="s3://runs/smoke/exports/global_step_8/policy",
        global_step=8,
        tokenizer_uri="tokenizer",
        tokenizer_revision="revision",
        checkpoint_root="s3://runs/smoke/checkpoints",
        terminal_manifest_uri="s3://runs/smoke/terminal.json",
    )
    monkeypatch.setattr(
        "cloud.iris.launch.StoragePath.exists",
        lambda path: str(path).endswith("resolved.yaml"),
    )
    monkeypatch.setattr("cloud.iris.launch._exported_policy", lambda _config: model)
    monkeypatch.setattr("cloud.iris.launch.write_json", lambda uri, payload: writes.append((uri, payload)))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.SUCCEEDED
    assert result.model == model
    assert all(
        payload["result"]["started_at"] == "2026-11-07T01:00:00+00:00"
        and payload["result"]["finished_at"] == "2026-11-07T02:00:00+00:00"
        for _, payload in writes
    )
    assert backend.launched and backend.exported
    assert backend.launched_config is not None
    assert backend.launched_config["inputs"]["train_data"][0]["kind"] == "directory"
    assert [uri for uri, _ in writes] == [
        "s3://runs/smoke/exports/attempts/attempt-1/receipt.json",
        "s3://runs/smoke/terminal.json",
    ]


def test_detach_submits_without_exporting(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    result = execute_launch(_config_path(tmp_path, submission="detach"), backend=backend)

    assert result.state is LaunchState.SUBMITTED
    assert backend.launched
    assert not backend.exported


def test_wait_does_not_commit_nonterminal_zero_exit_outcome(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend(outcome=LaunchOutcome("/user/still-running", "submitted", 0))
    writes: list[tuple[str, dict]] = []
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    monkeypatch.setattr("cloud.iris.launch.write_json", lambda uri, payload: writes.append((uri, payload)))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.FAILED
    assert result.failure == "Iris job reached submitted"
    assert not backend.exported
    assert [uri for uri, _ in writes] == ["s3://runs/smoke/exports/attempts/attempt-1/receipt.json"]


@dataclass
class LocalLaunchBackend:
    """Iris I/O boundary with local resolved-config and export fixtures."""

    export_jobs: list[str] = field(default_factory=list)
    outcome: LaunchOutcome = LaunchOutcome("/user/training", "succeeded", 0)

    def validate(self, _config_path: Path) -> None:
        raise AssertionError("this lifecycle test submits training")

    def launch(self, config_path: Path) -> LaunchOutcome:
        config = yaml.safe_load(config_path.read_text())
        artifacts = config["artifacts"]
        resolved = Path(artifacts["resolved_config_uri"])
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(config_path.read_bytes())
        checkpoint_root = Path(artifacts["checkpoint_root"])
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / CHECKPOINT_MARKER_FILENAME).write_text("3\n")
        return self.outcome

    def export_terminal_policy(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text())
        output = policy_export_path(config["artifacts"]["export_root"], 3)
        self.export_jobs.append(output)
        root = Path(output)
        root.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (root / name).write_text("{}")


@pytest.mark.parametrize(
    "export_hf,outcome",
    [
        (None, LaunchOutcome("/user/training", "succeeded", 0)),
        (True, LaunchOutcome("/user/training", "succeeded", 0)),
        (None, LaunchOutcome("/user/training", "failed", 1)),
    ],
    ids=["default-native", "explicit-hf", "failed-native"],
)
def test_launch_persists_attempt_result_and_exports_only_on_opt_in(tmp_path: Path, export_hf: bool | None, outcome):
    path = _config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    if export_hf is None:
        config["run"].pop("export_hf")
    else:
        config["run"]["export_hf"] = export_hf
    config["skyrl"]["trainer"].update(max_steps=3, eval_interval=3, ckpt_interval=3)
    config["artifacts"] = {
        "checkpoint_root": str(tmp_path / "checkpoints"),
        "export_root": str(tmp_path / "exports"),
        "attempts_root": str(tmp_path / "attempts"),
        "resolved_config_uri": str(tmp_path / "resolved.yaml"),
        "terminal_manifest_uri": str(tmp_path / "terminal.json"),
    }
    path.write_text(yaml.safe_dump(config))
    backend = LocalLaunchBackend(outcome=outcome)

    result = execute_launch(path, backend=backend)
    attempt = json.loads((tmp_path / "exports/attempts/attempt-1/receipt.json").read_text())
    assert not (tmp_path / "attempts").exists()
    if outcome.exit_code:
        assert result.state is LaunchState.FAILED
        assert attempt["result"]["state"] == "failed"
        assert attempt["result"]["failure"] == "Iris job reached failed"
        assert not (tmp_path / "terminal.json").exists()
        assert backend.export_jobs == []
        return
    assert result.state is LaunchState.SUCCEEDED
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal == attempt
    assert terminal["config"] == yaml.safe_load((tmp_path / "resolved.yaml").read_text())
    assert terminal["result"]["state"] == "succeeded"
    if export_hf:
        assert backend.export_jobs == [str(tmp_path / "exports/global_step_3/policy")]
        assert result.model is not None and result.model.global_step == 3
        assert terminal["result"]["model"]["policy_export_uri"] == backend.export_jobs[0]
    else:
        assert backend.export_jobs == []
        assert result.model is None and terminal["result"]["model"] is None
        assert not (tmp_path / "exports/global_step_3").exists()
