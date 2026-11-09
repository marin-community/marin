"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import fsspec
import numpy as np
import pytest
import yaml
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName
from safetensors.numpy import save

from cloud.iris import hf_model_cache
from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, HuggingFaceSnapshotFile, load_model_manifest
from cloud.iris.launch import ExportedPolicy, LaunchOutcome, LaunchState, execute_launch
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
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
    config = raw_launch_config()
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


@dataclass
class LocalLaunchBackend:
    """Iris I/O boundary with local resolved-config and export fixtures."""

    export_jobs: list[str] = field(default_factory=list)

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
        return LaunchOutcome("/user/training", "succeeded", 0)

    def export_terminal_policy(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text())
        output = policy_export_path(config["artifacts"]["export_root"], 3)
        self.export_jobs.append(output)
        root = Path(output)
        root.mkdir(parents=True)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            (root / name).write_text("{}")


@pytest.mark.parametrize("export_hf", [None, True], ids=["default-native", "explicit-hf"])
def test_completed_launch_defaults_to_native_result_and_exports_only_on_opt_in(tmp_path: Path, export_hf: bool | None):
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
    backend = LocalLaunchBackend()

    result = execute_launch(path, backend=backend)

    assert result.state is LaunchState.SUCCEEDED
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    attempt = json.loads((tmp_path / "attempts/attempt-1.json").read_text())
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


def _hub_config_path(tmp_path: Path, *, submission: str = "detach") -> Path:
    path = _config_path(tmp_path, submission=submission)
    config = yaml.safe_load(path.read_text())
    config["inputs"]["model"].update(
        uri="org/policy",
        identity="a" * 40,
        local_path="org/policy",
        tokenizer_uri="org/policy",
        tokenizer_revision="a" * 40,
    )
    config["iris"]["allocation"]["num_nodes"] = 2
    skyrl = config["skyrl"]
    skyrl["trainer"]["algorithm"].update(
        advantage_estimator="uniform",
        distillation={
            "objective": "sampled_reverse_kl",
            "routing_plan": "opd",
            "coefficient": 1.0,
            "reward_mode": "replace",
        },
    )
    skyrl["teachers"] = {
        "local": {
            "source": "local_inference",
            "placement": "pinned",
            "backend": "vllm",
            "evidence": "chosen_token",
            "model": {"path": "org/teacher", "revision": "b" * 40},
            "resources": {"num_nodes": 1, "gpus_per_node": 8, "tensor_parallel_size": 1, "colocation_group": "teacher"},
        },
        "external": {
            "source": "openai_compatible",
            "placement": "external",
            "evidence": "chosen_token",
            "model": {"path": "org/external", "revision": "c" * 40},
            "endpoints": [{"url": "https://teacher.example/v1", "max_concurrency": 8}],
            "tokenizer_fingerprint": "sha256:" + "d" * 64,
            "max_sequence_length": 1024,
            "request_timeout_seconds": 120,
        },
    }
    skyrl["teacher_routing"] = {
        "opd": {"revision": "v1", "routes": {name: {"teacher": name, "weight": 1.0} for name in ("local", "external")}}
    }
    path.write_text(yaml.safe_dump(config))
    return path


def _hub_snapshot(tmp_path: Path) -> HuggingFaceSnapshot:
    filesystem = fsspec.filesystem("memory")
    root = f"{tmp_path.name}/hub"
    payloads = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors": save({"weight": np.arange(4, dtype=np.float32)}),
    }
    files = []
    for name, payload in payloads.items():
        filesystem.pipe_file(f"{root}/{name}", payload)
        files.append(HuggingFaceSnapshotFile(name, len(payload), hashlib.sha256(payload).hexdigest()))
    return HuggingFaceSnapshot(filesystem, root, tuple(files))


@pytest.mark.parametrize(("hub_failure", "shared_model"), [(False, False), (False, True), (True, False)])
def test_cold_policy_and_local_teachers_are_mirrored_before_gpu_submission(
    tmp_path, monkeypatch, hub_failure, shared_model
):
    path = _hub_config_path(tmp_path)
    if shared_model:
        config = yaml.safe_load(path.read_text())
        config["skyrl"]["teachers"]["local"]["model"] = {"path": "org/policy", "revision": "a" * 40}
        path.write_text(yaml.safe_dump(config))
    expected_models = ["org/policy"] if shared_model else ["org/policy", "org/teacher"]
    snapshot = _hub_snapshot(tmp_path)
    mirror_root = tmp_path / "mirrors"
    monkeypatch.setattr(
        hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(mirror_root / prefix)
    )
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    opened = []

    def open_snapshot(model_id, _revision):
        opened.append(model_id)
        if hub_failure and model_id == "org/teacher":
            raise OSError("Hub unavailable")
        return snapshot

    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)

    class MirroredBackend(RecordingBackend):
        def launch(self, config_path):
            config = yaml.safe_load(config_path.read_text())
            mirrors = config["inputs"]["hugging_face_models"]
            assert {model["model_id"] for model in mirrors} == set(expected_models)
            for model in mirrors:
                manifest = load_model_manifest(model["uri"])
                assert manifest.identity == model["identity"]
                assert Path(model["uri"], "model.safetensors").read_bytes() == save(
                    {"weight": np.arange(4, dtype=np.float32)}
                )
            return super().launch(config_path)

    backend = MirroredBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        if hub_failure:
            with pytest.raises(OSError, match="Hub unavailable"):
                execute_launch(path, backend=backend)
            assert not backend.launched
        else:
            result = execute_launch(path, backend=backend)
            assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)
    assert opened == expected_models


@pytest.mark.parametrize("submission", ["prepare", "detach"])
def test_hub_models_are_not_downloaded_on_a_local_launcher(tmp_path, monkeypatch, submission):
    backend = RecordingBackend()
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    path = _hub_config_path(tmp_path, submission=submission)
    previous = get_job_info()
    set_job_info(None)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    try:
        if submission == "prepare":
            result = execute_launch(path, backend=backend)
            assert result.state is LaunchState.PREPARED
        else:
            with pytest.raises(RuntimeError, match="Iris CPU coordinator"):
                execute_launch(path, backend=backend)
        assert not backend.launched
    finally:
        set_job_info(previous)
