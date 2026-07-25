# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    HarborRunConfig,
    HarborTrial,
    _restore_completed_trials,
    _upload_trials,
    _write_samples,
    run_harbor,
)
from marin.evaluation.records import RunStatus
from marin.evaluation.runner import EvaluationError
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath


def _running_model() -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )


def test_materialize_harbor_dataset_downloads_hf_revision_as_local_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / ".gitattributes").write_text("*.gz filter=lfs")
    (snapshot / "task-one").mkdir()
    calls: list[dict] = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr("marin.evaluation.harbor.dataset.snapshot_download", download)

    path = materialize_harbor_dataset(
        "hf://DCAgent2/terminal_bench_2",
        "main",
        tmp_path / "workdir",
        hf_token=None,
    )

    assert path == Path(snapshot)
    assert calls == [
        {
            "repo_id": "DCAgent2/terminal_bench_2",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": str(tmp_path / "workdir" / "hf_dataset"),
            "cache_dir": str(tmp_path / "workdir" / "hf_cache"),
            "token": False,
        }
    ]
    assert not (snapshot / ".gitattributes").exists()


def test_write_samples_uses_a_path_safe_name_for_hf_dataset(tmp_path):
    trial = HarborTrial(task_id="task-one", reward=0.0, status="completed", trajectory=None, error=None)

    path = _write_samples([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert path == str(tmp_path / "samples_harbor.parquet")
    assert Path(path).exists()


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_harbor_trials_round_trip_through_remote_storage(protocol, tmp_path, monkeypatch):
    class RemoteMemoryFileSystem(MemoryFileSystem):
        pass

    RemoteMemoryFileSystem.protocol = protocol
    remote_fs = RemoteMemoryFileSystem()

    def remote_url_to_fs(url: str, **_kwargs):
        path = StoragePath(url)
        assert path.scheme == protocol
        return remote_fs, "/".join(part for part in (path.netloc, path.key) if part)

    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", remote_url_to_fs)

    trial_dir = tmp_path / "source" / "trial-one"
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "result.json").write_text('{"task_name": "trial-one"}')
    (trial_dir / "agent" / "trajectory.json").write_text('{"steps": []}')
    incomplete_dir = tmp_path / "source" / "incomplete"
    incomplete_dir.mkdir()
    (incomplete_dir / "partial.json").write_text("{}")

    output_dir = f"{protocol}://eval-bucket/run"
    _upload_trials(trial_dir.parent, output_dir)

    restored_dir = tmp_path / "restored"
    restored = _restore_completed_trials(output_dir, restored_dir)

    assert restored == 1
    assert (restored_dir / "trial-one" / "result.json").read_text() == '{"task_name": "trial-one"}'
    assert (restored_dir / "trial-one" / "agent" / "trajectory.json").read_text() == '{"steps": []}'
    assert not (restored_dir / "incomplete").exists()


def test_run_harbor_derives_url_and_served_name_from_running_model(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config_file: Path) -> None:
        captured.update(json.loads(config_file.read_text()))
        trial_dir = Path(captured["jobs_dir"]) / captured["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", lambda *args, **kwargs: None)
    model = _running_model()

    result = run_harbor(
        model,
        HarborRunConfig(dataset=f"toy-{tmp_path.name}", version="1.0", agent="terminus-2"),
        str(tmp_path),
        hf_token=None,
    )

    assert captured["model_name"] == "hosted_vllm/qwen3-0.6b"
    assert captured["agent_kwargs"]["api_base"] == model.endpoint.base_url
    assert result.total_trials == 1
    assert result.accuracy == 1.0


def test_harbor_executor_fails_when_trial_contains_exception_info(tmp_path, monkeypatch):
    def run_driver(config_file: Path) -> None:
        config = json.loads(config_file.read_text())
        trial_dir = Path(config["jobs_dir"]) / config["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 0.0}},
                    "exception_info": {
                        "exception_type": "AgentError",
                        "exception_message": "model request failed",
                    },
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", lambda *args, **kwargs: None)
    executor = HarborExecutor(HarborRunConfig(dataset=f"toy-{tmp_path.name}", version="1.0", agent="terminus-2"))

    with pytest.raises(EvaluationError) as exc_info:
        executor(_running_model(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.FAILED
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(config_file: Path) -> None:
        config = json.loads(config_file.read_text())
        trial_dir = Path(config["jobs_dir"]) / config["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 0.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner._run_driver", run_driver)
    monkeypatch.setattr("marin.evaluation.harbor.runner.materialize_harbor_dataset", lambda *args, **kwargs: None)
    executor = HarborExecutor(HarborRunConfig(dataset=f"toy-{tmp_path.name}", version="1.0", agent="terminus-2"))

    outcome = executor(_running_model(), str(tmp_path), {})

    assert outcome.metrics[executor.config.dataset]["accuracy"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 0
