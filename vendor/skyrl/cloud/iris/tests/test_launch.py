"""Behavior tests for the single-config launch lifecycle."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import numpy as np
import pytest
import yaml
from finestore.rl.run_records import read_attempt
from finestore.testing.simulated_s3 import simulated_s3
from rigging.filesystem.storage_path import StoragePath
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.types import JobName
from omegaconf import OmegaConf
from safetensors.numpy import save

from cloud.iris import hf_model_cache, task_runtime
from cloud.iris.artifacts import CHECKPOINT_MARKER_FILENAME
from cloud.iris.task_runtime import prepare_draft_model
from marinskyrl.speculative_decoding import SpeculatorModelConfig
from cloud.iris.hf_model_cache import HuggingFaceSnapshot, load_model_manifest
from cloud.iris.launch import ExportedPolicy, LaunchOutcome, LaunchState, execute_launch
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from cloud.iris.tests.model_snapshot_fixtures import memory_model_snapshot
from cloud.iris.iris_backend import supervise_iris_job
from iris.resources.state import JobState
from rigging.timing import Timestamp
from marinskyrl.checkpoint_paths import policy_export_path


@pytest.fixture
def run_bucket():
    with simulated_s3(latency=0.0) as filesystem:
        yield filesystem


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
    config["artifacts"].update(
        checkpoint_root=str(tmp_path / "checkpoints"),
        export_root=str(tmp_path / "exports"),
        attempts_root=str(tmp_path / "attempts"),
        resolved_config_uri=str(tmp_path / "resolved.yaml"),
        terminal_manifest_uri=str(tmp_path / "terminal.json"),
    )
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


def test_prepare_validates_without_submitting(tmp_path: Path) -> None:
    backend = RecordingBackend()

    result = execute_launch(_config_path(tmp_path, submission="prepare"), backend=backend)

    assert result.state is LaunchState.PREPARED
    assert backend.validated
    assert not backend.launched


def test_detach_submits_without_exporting(tmp_path: Path) -> None:
    backend = RecordingBackend()

    result = execute_launch(_config_path(tmp_path, submission="detach"), backend=backend)

    assert result.state is LaunchState.SUBMITTED
    assert backend.launched
    assert not backend.exported


def test_wait_does_not_commit_nonterminal_zero_exit_outcome(tmp_path: Path) -> None:
    backend = RecordingBackend(outcome=LaunchOutcome("/user/still-running", "submitted", 0))

    result = execute_launch(_config_path(tmp_path), backend=backend)

    assert result.state is LaunchState.FAILED
    assert result.failure == "Iris job reached submitted"
    assert not backend.exported
    record = read_attempt(str(tmp_path / "exports"), "attempt-1")
    assert record is not None
    assert record.native.result.model_dump(mode="json") == result.model_dump(mode="json")
    assert not StoragePath(str(tmp_path / "terminal.json")).exists()


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
        status = SimpleNamespace(
            state=JobState(self.outcome.job_state),
            started_at=Timestamp.from_seconds(datetime(2026, 11, 7, 1, tzinfo=timezone.utc).timestamp()),
            finished_at=Timestamp.from_seconds(datetime(2026, 11, 7, 2, tzinfo=timezone.utc).timestamp()),
        )
        return supervise_iris_job(SimpleNamespace(job_id=self.outcome.job_id, wait=lambda **_kwargs: status))

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
    saved = read_attempt(str(tmp_path / "exports"), "attempt-1")
    assert saved is not None
    attempt = saved.native.model_dump(mode="json")
    assert saved.native.result.started_at == "2026-11-07T01:00:00+00:00"
    assert saved.native.result.finished_at == "2026-11-07T02:00:00+00:00"
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


@pytest.mark.parametrize("separate_tokenizer", [False, True])
@pytest.mark.parametrize(("hub_failure", "shared_model"), [(False, False), (False, True), (True, False)])
def test_cold_policy_and_local_teachers_are_mirrored_before_gpu_submission(
    tmp_path, monkeypatch, hub_failure, shared_model, separate_tokenizer
):
    path = _hub_config_path(tmp_path)
    if separate_tokenizer:
        config = yaml.safe_load(path.read_text())
        config["inputs"]["model"]["tokenizer_uri"] = "org/tokenizer"
        path.write_text(yaml.safe_dump(config))
    if shared_model:
        config = yaml.safe_load(path.read_text())
        config["skyrl"]["teachers"]["local"]["model"] = {"path": "org/policy", "revision": "a" * 40}
        path.write_text(yaml.safe_dump(config))
    expected_models = ["org/policy"] if shared_model else ["org/policy", "org/teacher"]
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    mirror_root = tmp_path / "mirrors"
    monkeypatch.setattr(
        hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(mirror_root / prefix)
    )
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    def open_snapshot(model_id, _revision):
        if hub_failure and model_id == "org/teacher":
            raise OSError("Hub unavailable")
        if separate_tokenizer and not shared_model and model_id == "org/policy":
            return HuggingFaceSnapshot(
                snapshot.filesystem,
                snapshot.root,
                tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
            )
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
                assert manifest.revision == model["resolved_revision"]
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


@pytest.mark.parametrize("tokenizer_uri", ["org/policy", ""])
def test_policy_without_a_separate_tokenizer_rejects_model_only_mirrors(tmp_path, monkeypatch, tokenizer_uri):
    path = _hub_config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    config["inputs"]["model"]["tokenizer_uri"] = tokenizer_uri
    path.write_text(yaml.safe_dump(config))
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    model_only = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
    )
    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", lambda *_: model_only)
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_, **__: str(tmp_path / "mirror"))
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    backend = RecordingBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        with pytest.raises(ValueError, match="tokenizer"):
            execute_launch(path, backend=backend)
        assert not backend.launched
    finally:
        set_job_info(previous)


def test_hub_draft_is_mirrored_and_rewritten_before_gpu_submission(tmp_path, monkeypatch):
    path = _config_path(tmp_path, submission="detach")
    config = yaml.safe_load(path.read_text())
    config["skyrl"]["trainer"]["placement"]["colocate_all"] = False
    config["iris"]["allocation"]["num_nodes"] = 2
    config["skyrl"]["generator"]["speculative_decoding"] = {
        "model": {"source_uri": "hf://org/draft", "source_identity": "d" * 40},
        "num_speculative_tokens": 3,
        "method": "eagle3",
    }
    path.write_text(yaml.safe_dump(config))
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    snapshot = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.startswith("tokenizer")),
    )
    opened = []

    def open_snapshot(model_id, revision):
        opened.append((model_id, revision))
        return snapshot

    monkeypatch.setattr(hf_model_cache, "_open_hugging_face_snapshot", open_snapshot)
    monkeypatch.setattr(hf_model_cache, "marin_temp_bucket", lambda *_, **__: str(tmp_path / "draft"))
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)

    class DraftBackend(RecordingBackend):
        def launch(self, config_path):
            resolved = yaml.safe_load(config_path.read_text())
            draft = SpeculatorModelConfig.from_mapping(
                resolved["skyrl"]["generator"]["speculative_decoding"]["model"], context="draft"
            )
            manifest = load_model_manifest(draft.source_uri)
            assert manifest.tokenizer_mode == "policy"
            assert draft.source_identity == manifest.identity
            assert Path(draft.source_uri, "model.safetensors").is_file()
            assert not Path(draft.source_uri, "tokenizer.json").exists()
            assert resolved["inputs"]["hugging_face_models"][0]["uri"] == draft.source_uri
            assert prepare_draft_model(draft, cache_ttl_days=14, cache_source_prefix="s3://region/run") == draft
            return super().launch(config_path)

    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        result = execute_launch(path, backend=DraftBackend())
        assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)
    assert opened == [("org/draft", "d" * 40)]


def test_unsupported_policy_layout_retains_node_fallback(tmp_path, monkeypatch):
    path = _hub_config_path(tmp_path)
    snapshot = memory_model_snapshot(f"{tmp_path.name}/hub")
    unsupported = HuggingFaceSnapshot(
        snapshot.filesystem,
        snapshot.root,
        tuple(entry for entry in snapshot.files if not entry.path.endswith(".safetensors")),
    )
    monkeypatch.setattr(
        hf_model_cache,
        "_open_hugging_face_snapshot",
        lambda model_id, _: unsupported if model_id == "org/policy" else snapshot,
    )
    monkeypatch.setattr(
        hf_model_cache, "marin_temp_bucket", lambda _ttl, *, prefix, source_prefix: str(tmp_path / "mirrors" / prefix)
    )
    monkeypatch.setattr("cloud.iris.launch.StoragePath.exists", lambda _path: False)
    backend = RecordingBackend()
    previous = get_job_info()
    set_job_info(JobInfo(task_id=JobName.from_wire("/user/coordinator/0")))
    try:
        result = execute_launch(path, backend=backend)
        assert result.state is LaunchState.SUBMITTED
    finally:
        set_job_info(previous)
    assert [model["model_id"] for model in backend.launched_config["inputs"]["hugging_face_models"]] == ["org/teacher"]
    downloaded = []
    node_snapshot = tmp_path / "fallback"
    node_snapshot.mkdir()

    def download(model_id, *, revision, allow_patterns):
        downloaded.append((model_id, revision))
        return node_snapshot

    monkeypatch.setattr(task_runtime, "download_hugging_face_snapshot", download)
    monkeypatch.setattr(task_runtime.huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "node-cache"))
    task_runtime.stage_hugging_face_models(OmegaConf.create(backend.launched_config))
    assert downloaded == [("org/policy", "a" * 40)]
