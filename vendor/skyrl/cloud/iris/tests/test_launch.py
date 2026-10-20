"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from cloud.iris.launch import ExportedPolicy, LaunchOutcome, LaunchState, execute_launch
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
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
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
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
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
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    result = execute_launch(_config_path(tmp_path, submission="detach"), backend=backend)

    assert result.state is LaunchState.SUBMITTED
    assert backend.launched
    assert not backend.exported


def test_wait_does_not_commit_nonterminal_zero_exit_outcome(tmp_path: Path, monkeypatch) -> None:
    backend = RecordingBackend(outcome=LaunchOutcome("/user/still-running", "submitted", 0))
    writes: list[tuple[str, dict]] = []
    monkeypatch.setattr("cloud.iris.launch.runtime_bundle_inputs", lambda _commit: ())
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    monkeypatch.setattr("cloud.iris.launch.write_json", lambda uri, payload: writes.append((uri, payload)))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.FAILED
    assert result.failure == "Iris job reached submitted"
    assert not backend.exported
    assert [uri for uri, _ in writes] == ["s3://runs/smoke/attempts/attempt-1.json"]
