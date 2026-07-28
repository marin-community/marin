# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import rigging.filesystem.factory as filesystem_factory
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborAgentConfig,
    HarborDriverConfig,
    HarborEnvironmentConfig,
    HarborRetryConfig,
    HarborRunConfig,
    HarborVerifierConfig,
    native_job_config,
)
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    HarborTrial,
    _restore_completed_trials,
    _upload_trials,
    _write_samples,
    run_harbor,
)
from marin.evaluation.records import EvalRef, HarborRef, RunStatus
from marin.evaluation.runner import EvaluationError
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath
from rigging.testing import memory_filesystem_and_resolver


def _running_model() -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_materialize_harbor_dataset_stages_only_selected_remote_tasks(protocol, tmp_path, monkeypatch):
    remote_fs, resolve = memory_filesystem_and_resolver(protocol, filesystem_factory.url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", resolve)
    remote = StoragePath(f"{protocol}://regional-cache/benchmark")
    for task_name in ("task-b", "task-a"):
        root = f"regional-cache/benchmark/{task_name}"
        remote_fs.makedirs(f"{root}/environment", exist_ok=True)
        remote_fs.makedirs(f"{root}/tests", exist_ok=True)
        remote_fs.pipe(f"{root}/task.toml", b"version = '1.0'")
        remote_fs.pipe(f"{root}/environment/Dockerfile", b"FROM scratch")
        remote_fs.pipe(f"{root}/tests/test.sh", b"true")
    remote_fs.pipe("regional-cache/benchmark/README.md", b"metadata, not a task")
    remote_fs.pipe("regional-cache/benchmark/incomplete/task.toml", b"version = '1.0'")

    path = materialize_harbor_dataset(str(remote), tmp_path / "workdir", task_limit=1)

    assert path == tmp_path / "workdir" / "harbor_dataset"
    assert (path / "task-a" / "environment" / "Dockerfile").read_text() == "FROM scratch"
    assert (path / "task-a" / "tests" / "test.sh").read_text() == "true"
    assert not (path / "task-b").exists()
    assert not (path / "incomplete").exists()
    assert not (path / "README.md").exists()


def test_materialize_harbor_dataset_rejects_direct_hugging_face_reads(tmp_path):
    with pytest.raises(ValueError):
        materialize_harbor_dataset("hf://DCAgent2/terminal_bench_2", tmp_path, task_limit=None)


def test_write_samples_uses_a_path_safe_name_for_hf_dataset(tmp_path):
    trial = HarborTrial(task_id="task-one", reward=0.0, status="completed", trajectory=None, error=None)

    path = _write_samples([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert path == str(tmp_path / "samples_harbor.parquet")
    assert Path(path).exists()


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_harbor_trials_round_trip_through_remote_storage(protocol, tmp_path, monkeypatch):
    _remote_fs, resolve = memory_filesystem_and_resolver(protocol, filesystem_factory.url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", resolve)

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


def test_native_job_config_translates_the_external_harbor_contract():
    config = HarborDriverConfig(
        job_name="job",
        jobs_dir="/tmp/jobs",
        dataset_path=None,
        endpoint_url="https://iris.example/v1",
        served_model="served-model",
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

    native = native_job_config(config)

    assert native["datasets"] == [{"name": "aime", "version": "1.0", "n_tasks": None}]
    assert native["n_attempts"] == 3
    assert native["retry"]["min_wait_sec"] == 2.0
    assert native["retry"]["max_wait_sec"] == 90.0
    assert native["environment"]["override_cpus"] == 2
    assert native["environment"]["override_memory_mb"] == 8192
    assert native["environment"]["override_storage_mb"] == 8192
    assert native["verifier"]["max_timeout_sec"] == 14400
    agent = native["agents"][0]
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


def test_run_harbor_normalizes_a_completed_external_trial(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(command, *, check, env) -> None:
        assert check
        driver_config = HarborDriverConfig.from_dict(json.loads(Path(command[-1]).read_text()))
        captured["driver_config"] = driver_config
        captured["env"] = env
        trial_dir = Path(driver_config.jobs_dir) / driver_config.job_name / "trial-one"
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

    result = run_harbor(
        model,
        HarborRunConfig(
            dataset=f"toy-{tmp_path.name}",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type="daytona"),
        ),
        str(tmp_path),
        driver_env={"DAYTONA_API_KEY": "daytona-key"},
    )

    assert captured["driver_config"].endpoint_url == model.endpoint.base_url
    assert captured["driver_config"].served_model == "qwen3-0.6b"
    assert captured["env"]["DAYTONA_API_KEY"] == "daytona-key"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert result.total_trials == 1
    assert result.accuracy == 1.0


def test_harbor_executor_fails_when_trial_contains_exception_info(tmp_path, monkeypatch):
    def run_driver(command, *, check, env) -> None:
        assert check and isinstance(env, dict)
        config = HarborDriverConfig.from_dict(json.loads(Path(command[-1]).read_text()))
        trial_dir = Path(config.jobs_dir) / config.job_name / "trial-one"
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
    executor = HarborExecutor(
        HarborRunConfig(
            dataset=f"failed-{tmp_path.name}",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type="daytona"),
        )
    )

    with pytest.raises(EvaluationError) as exc_info:
        executor(_running_model(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.FAILED
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(command, *, check, env) -> None:
        assert check and isinstance(env, dict)
        config = HarborDriverConfig.from_dict(json.loads(Path(command[-1]).read_text()))
        trial_dir = Path(config.jobs_dir) / config.job_name / "trial-one"
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
    executor = HarborExecutor(
        HarborRunConfig(
            dataset=f"zero-{tmp_path.name}",
            revision="1.0",
            agent=HarborAgentConfig(name="terminus-2"),
            environment=HarborEnvironmentConfig(environment_type="daytona"),
        )
    )

    outcome = executor(_running_model(), str(tmp_path), {})

    assert outcome.metrics[executor.config.dataset]["accuracy"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 0


def test_harbor_executor_reports_the_resolved_dataset_artifact(tmp_path, monkeypatch):
    mirror_uri = "s3://regional/artifacts/terminal-bench/immutable-commit/2026.07.27"
    remote_fs, resolve_path = memory_filesystem_and_resolver("s3", filesystem_factory.url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", resolve_path)
    task_root = "regional/artifacts/terminal-bench/immutable-commit/2026.07.27/task-one"
    remote_fs.makedirs(f"{task_root}/environment", exist_ok=True)
    remote_fs.makedirs(f"{task_root}/tests", exist_ok=True)
    remote_fs.pipe(f"{task_root}/task.toml", b"version = '1.0'")
    remote_fs.pipe(f"{task_root}/environment/Dockerfile", b"FROM scratch")
    remote_fs.pipe(f"{task_root}/tests/test.sh", b"true")

    def run_driver(command, *, check, env) -> None:
        assert check and isinstance(env, dict)
        driver_config = HarborDriverConfig.from_dict(json.loads(Path(command[-1]).read_text()))
        dataset_path = Path(driver_config.dataset_path)
        assert (dataset_path / "task-one" / "environment" / "Dockerfile").read_text() == "FROM scratch"
        trial_dir = Path(driver_config.jobs_dir) / driver_config.job_name / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                }
            )
        )

    dataset_artifact = ArtifactStep(
        name="evaluation/harbor-datasets/terminal-bench/immutable-commit",
        version="2026.07.27",
        artifact_type=Artifact,
        run=lambda _config: None,
        build_config=lambda _ctx: {},
    )
    monkeypatch.setattr("marin.evaluation.harbor.runner.resolve", lambda _artifact: Artifact(path=mirror_uri))
    monkeypatch.setattr("marin.evaluation.harbor.runner.subprocess.run", run_driver)
    config = HarborRunConfig(
        dataset="DCAgent2/terminal_bench_2",
        revision="immutable-commit",
        agent=HarborAgentConfig(name="terminus-2"),
        environment=HarborEnvironmentConfig(environment_type="daytona"),
    )
    executor = HarborExecutor(
        config=config,
        dataset_artifact=dataset_artifact,
        record_ref=EvalRef(
            name="tb2-lite",
            mechanism="harbor",
            harbor=HarborRef(
                dataset=config.dataset,
                version=config.revision,
                agent=config.agent.name,
                env=config.environment.environment_type,
                repository=config.dataset,
                commit=config.revision,
            ),
        ),
    )

    outcome = executor(_running_model(), str(tmp_path), {})

    assert outcome.eval_ref is not None
    assert outcome.eval_ref.harbor is not None
    assert outcome.eval_ref.harbor.mirror_uri == mirror_uri
