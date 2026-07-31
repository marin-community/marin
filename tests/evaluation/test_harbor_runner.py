# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborDatasetKind,
    ValidatedHarborConfig,
)
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    HarborTrial,
    _restore_completed_trials,
    _upload_trials,
    _write_samples,
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


def _validated_config(
    *,
    dataset_kind: HarborDatasetKind = HarborDatasetKind.HARBOR_REGISTRY,
    dataset_selector: str = "aime",
    dataset_revision: str | None = "1.0",
    workspace_dataset_path: Path | None = None,
    agent: str = "terminus-2",
) -> ValidatedHarborConfig:
    return ValidatedHarborConfig(
        stable_policy_json='{"opaque":"policy"}',
        digest=f"sha256:{'1' * 64}",
        dataset_kind=dataset_kind,
        dataset_selector=dataset_selector,
        dataset_revision=dataset_revision,
        workspace_dataset_path=workspace_dataset_path,
        agent=agent,
        environment="daytona",
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
        _validated_config(
            dataset_kind=HarborDatasetKind.HUGGING_FACE,
            dataset_selector="DCAgent2/terminal_bench_2",
            dataset_revision="main",
        ),
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


def test_materialize_harbor_dataset_rebases_local_path_onto_worker_workspace(tmp_path, monkeypatch):
    worker_workspace = tmp_path / "worker"
    dataset = worker_workspace / "policies" / "tasks"
    dataset.mkdir(parents=True)
    config = _validated_config(
        dataset_kind=HarborDatasetKind.LOCAL,
        dataset_selector="tasks",
        dataset_revision=None,
        workspace_dataset_path=Path("policies/tasks"),
    )
    monkeypatch.setattr(
        "marin.evaluation.harbor.dataset.find_project_root",
        lambda: worker_workspace,
    )

    assert (
        materialize_harbor_dataset(
            config,
            tmp_path / "workdir",
            hf_token=None,
        )
        == dataset
    )


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


def test_harbor_executor_passes_opaque_policy_and_runtime_overlay_to_driver(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config, overlay, driver_env) -> None:
        captured["config"] = config
        captured["overlay"] = overlay
        captured["env"] = driver_env
        trial_dir = Path(overlay.jobs_dir) / overlay.job_name / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-harbor")
    model = _running_model()

    executor = HarborExecutor(
        _validated_config(
            dataset_selector=f"toy-{tmp_path.name}",
        ),
        task_limit=7,
        model_agent_kwargs={"extra_body": "{}"},
        secret_env_keys=("DAYTONA_API_KEY",),
    )
    outcome = executor(
        model,
        str(tmp_path),
        {"DAYTONA_API_KEY": "daytona-key"},
    )

    assert captured["config"] is executor.config
    assert captured["overlay"].endpoint_url == model.endpoint.base_url
    assert captured["overlay"].served_model == "qwen3-0.6b"
    assert captured["overlay"].task_limit == 7
    assert captured["overlay"].model_agent_kwargs == {"extra_body": "{}"}
    assert captured["env"]["DAYTONA_API_KEY"] == "daytona-key"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert outcome.metrics[f"toy-{tmp_path.name}"]["accuracy"] == 1.0


def _harbor_executor(dataset: str) -> HarborExecutor:
    return HarborExecutor(
        _validated_config(dataset_selector=dataset),
        task_limit=None,
        model_agent_kwargs={},
    )


def test_harbor_executor_fails_when_trial_contains_exception_info(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env) -> None:
        assert isinstance(driver_env, dict)
        trial_dir = Path(overlay.jobs_dir) / overlay.job_name / "trial-one"
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

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"failed-{tmp_path.name}")

    with pytest.raises(EvaluationError) as exc_info:
        executor(_running_model(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.FAILED
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env) -> None:
        assert isinstance(driver_env, dict)
        trial_dir = Path(overlay.jobs_dir) / overlay.job_name / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 0.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"zero-{tmp_path.name}")

    outcome = executor(_running_model(), str(tmp_path), {})

    assert outcome.metrics[executor.config.record_dataset]["accuracy"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 0
