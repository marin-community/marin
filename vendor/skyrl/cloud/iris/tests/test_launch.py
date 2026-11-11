"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from cloud.iris.launch import ExportedPolicy, LaunchOutcome, LaunchState, execute_launch
from cloud.iris.launch_config import load_launch_config
from cloud.iris.tests.test_launch_config import _raw_config


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

    def export_pending_policies(self, _config_path: Path) -> None:
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
    backend = RecordingBackend()
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
    assert backend.launched and backend.exported
    assert backend.launched_config is not None
    assert backend.launched_config["inputs"]["train_data"][0]["kind"] == "directory"
    assert [uri for uri, _ in writes] == [
        "s3://runs/smoke/attempts/attempt-1.json",
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
    assert [uri for uri, _ in writes] == ["s3://runs/smoke/attempts/attempt-1.json"]


def test_completed_run_is_reused_only_while_all_exports_remain_complete(tmp_path: Path) -> None:
    config = yaml.safe_load(_config_path(tmp_path).read_text())
    run_root = tmp_path / "run"
    config["artifacts"] = {
        "checkpoint_root": str(run_root / "checkpoints"),
        "export_root": str(run_root / "exports"),
        "attempts_root": str(run_root / "attempts"),
        "resolved_config_uri": str(run_root / "resolved.yaml"),
        "terminal_manifest_uri": str(run_root / "terminal.json"),
    }
    config["skyrl"]["trainer"]["hf_export_steps"] = [10, 125]
    config_path = tmp_path / "completed.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (run_root / "checkpoints").mkdir(parents=True)
    (run_root / "checkpoints" / "latest_ckpt_global_step.txt").write_text("125")
    for step in (10, 125):
        policy = run_root / "exports" / f"global_step_{step}" / "policy"
        policy.mkdir(parents=True)
        (policy / "config.json").write_text("{}")
        (policy / "tokenizer.json").write_text("{}")
        (policy / "model.safetensors").write_bytes(b"weights")
    manifest = {
        "config": OmegaConf.to_container(load_launch_config(config_path), resolve=True),
        "result": {"state": "succeeded", "iris_job_id": "/user/prior", "iris_job_state": "succeeded"},
    }
    (run_root / "terminal.json").write_text(json.dumps(manifest))
    backend = RecordingBackend()

    result = execute_launch(config_path, backend=backend)

    assert result.state is LaunchState.SUCCEEDED
    assert result.iris_job_id == "/user/prior"
    assert result.model is not None and result.model.global_step == 125
    assert not backend.launched

    (run_root / "exports" / "global_step_10" / "policy" / "model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="no safetensors weights"):
        execute_launch(config_path, backend=backend)
    assert not backend.launched
