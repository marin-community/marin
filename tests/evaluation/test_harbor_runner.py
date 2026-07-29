# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborEnvironmentConfig,
    HarborRetryConfig,
    HarborRunConfig,
    HarborVerifierConfig,
    adapt_job_config,
    harbor_job_config,
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


def test_harbor_job_config_adapts_the_external_harbor_contract():
    config = harbor_job_config(
        job_name="job",
        run=HarborRunConfig(
            dataset="aime",
            revision="1.0",
            agent=HarborAgentConfig(
                name="opencode",
                max_output_tokens=16384,
                max_timeout=7200,
                setup_timeout=600,
                kwargs={
                    "model_info": {"max_input_tokens": 64512},
                    "opencode_config": {"compaction": {"auto": False}},
                },
            ),
            environment=HarborEnvironmentConfig(
                environment_type="daytona",
                cpus=2,
                memory_mb=8192,
                storage_mb=8192,
            ),
            attempts=3,
            retry=HarborRetryConfig(max_retries=6, min_wait=2.0, max_wait=90.0),
            verifier=HarborVerifierConfig(max_timeout=14400),
        ),
    )

    adapted = adapt_job_config(
        config,
        job_name="job",
        jobs_dir="/tmp/jobs",
        dataset_path=None,
        endpoint_url="https://iris.example/v1",
        served_model="served-model",
        task_limit=None,
        model_agent_kwargs={},
    )

    assert adapted["datasets"] == [{"name": "aime", "version": "1.0", "n_tasks": None}]
    assert adapted["n_attempts"] == 3
    assert adapted["retry"]["min_wait_sec"] == 2.0
    assert adapted["retry"]["max_wait_sec"] == 90.0
    assert adapted["environment"]["override_cpus"] == 2
    assert adapted["environment"]["override_memory_mb"] == 8192
    assert adapted["environment"]["override_storage_mb"] == 8192
    assert adapted["verifier"]["max_timeout_sec"] == 14400
    agent = adapted["agents"][0]
    assert agent["model_name"] == "hosted_vllm/served-model"
    assert agent["override_setup_timeout_sec"] == 600
    assert agent["kwargs"]["api_base"] == "https://iris.example/v1"
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 64512
    assert agent["kwargs"]["model_info"]["max_output_tokens"] == 16384
    opencode_config = agent["kwargs"]["opencode_config"]
    assert opencode_config["compaction"] == {"auto": False}
    assert opencode_config["provider"]["hosted_vllm"] == {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Hosted vLLM",
        "options": {"baseURL": "https://iris.example/v1"},
    }


def test_harbor_executor_normalizes_a_completed_external_trial(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(command, *, check, env) -> None:
        assert check
        job_config = json.loads(Path(command[-1]).read_text())
        captured["job_config"] = job_config
        captured["env"] = env
        trial_dir = Path(job_config["jobs_dir"]) / job_config["job_name"] / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.subprocess.run", run_driver)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-harbor")
    model = _running_model()

    executor = HarborExecutor(
        harbor_job_config(
            job_name="toy",
            run=HarborRunConfig(
                dataset=f"toy-{tmp_path.name}",
                revision="1.0",
                agent=HarborAgentConfig(name="terminus-2"),
                environment=HarborEnvironmentConfig(environment_type="daytona"),
            ),
        ),
        task_limit=None,
        model_agent_kwargs={},
        secret_env_keys=("DAYTONA_API_KEY",),
    )
    outcome = executor(
        model,
        str(tmp_path),
        {"DAYTONA_API_KEY": "daytona-key"},
    )

    assert captured["job_config"]["agents"][0]["kwargs"]["api_base"] == model.endpoint.base_url
    assert captured["job_config"]["agents"][0]["model_name"] == "hosted_vllm/qwen3-0.6b"
    assert captured["env"]["DAYTONA_API_KEY"] == "daytona-key"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert outcome.metrics[f"toy-{tmp_path.name}"]["accuracy"] == 1.0


def _harbor_executor(dataset: str) -> HarborExecutor:
    return HarborExecutor(
        harbor_job_config(
            job_name=dataset,
            run=HarborRunConfig(
                dataset=dataset,
                revision="1.0",
                agent=HarborAgentConfig(name="terminus-2"),
                environment=HarborEnvironmentConfig(environment_type="daytona"),
            ),
        ),
        task_limit=None,
        model_agent_kwargs={},
    )


def test_harbor_executor_fails_when_trial_contains_exception_info(tmp_path, monkeypatch):
    def run_driver(command, *, check, env) -> None:
        assert check and isinstance(env, dict)
        config = json.loads(Path(command[-1]).read_text())
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

    monkeypatch.setattr("marin.evaluation.harbor.runner.subprocess.run", run_driver)
    executor = _harbor_executor(f"failed-{tmp_path.name}")

    with pytest.raises(EvaluationError) as exc_info:
        executor(_running_model(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.FAILED
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(command, *, check, env) -> None:
        assert check and isinstance(env, dict)
        config = json.loads(Path(command[-1]).read_text())
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

    monkeypatch.setattr("marin.evaluation.harbor.runner.subprocess.run", run_driver)
    executor = _harbor_executor(f"zero-{tmp_path.name}")

    outcome = executor(_running_model(), str(tmp_path), {})

    assert outcome.metrics[executor.config.dataset]["accuracy"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 0
