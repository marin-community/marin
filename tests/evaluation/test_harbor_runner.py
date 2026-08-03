# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor import runner
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborDatasetKind,
    ValidatedHarborConfig,
)
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    HarborTrial,
    _job_dir,
    _migrate_legacy_scored_trials,
    _read_trials,
    _sample_for,
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
    trial = HarborTrial(task_id="task-one", reward=0.0, status="completed", trajectory_uri=None, error=None)

    path = _write_samples([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert path == str(tmp_path / "samples_harbor.parquet")
    assert Path(path).exists()


def test_read_trials_references_trajectory_in_place(tmp_path):
    """A trial's trajectory is referenced at its durable job-tree path, not copied elsewhere."""
    job_dir = tmp_path / "harbor_jobs" / "job"
    with_trajectory = job_dir / "trial-one"
    (with_trajectory / "agent").mkdir(parents=True)
    (with_trajectory / "result.json").write_text(
        json.dumps({"task_name": "task-one", "verifier_result": {"rewards": {"reward": 1.0}}})
    )
    (with_trajectory / "agent" / "trajectory.json").write_text('{"steps": []}')
    without_trajectory = job_dir / "trial-two"
    without_trajectory.mkdir(parents=True)
    (without_trajectory / "result.json").write_text(json.dumps({"task_name": "task-two"}))

    trials = _read_trials(StoragePath(str(job_dir)))

    by_task = {trial.task_id: trial for trial in trials}
    assert by_task["task-one"].trajectory_uri == str(with_trajectory / "agent" / "trajectory.json")
    assert by_task["task-two"].trajectory_uri is None
    # The sample carries the in-place URI; no separate trajectories/ copy is written.
    sample = _sample_for(by_task["task-one"], "aime")
    assert sample.trajectory_uri == str(with_trajectory / "agent" / "trajectory.json")


def _memory_remote(protocol: str, monkeypatch) -> None:
    """Route ``protocol://`` reads and writes to a fresh in-memory filesystem.

    Patches both the ``StoragePath`` factory (glob/read/write verbs) and the ``url_to_fs`` bound in
    the runner (the sample-parquet writer), so the whole executor path stays off real object storage.
    """

    class RemoteMemoryFileSystem(MemoryFileSystem):
        @classmethod
        def _strip_protocol(cls, path):
            # s3fs/gcsfs strip their scheme from a full URL; MemoryFileSystem only strips
            # ``memory://``. Match the real backends so a ``s3://``/``gs://`` URL and a bare
            # ``bucket/key`` resolve to the same store key.
            if isinstance(path, str) and path.startswith(f"{protocol}://"):
                path = path[len(f"{protocol}://") :]
            return MemoryFileSystem._strip_protocol(path)

    RemoteMemoryFileSystem.protocol = protocol
    RemoteMemoryFileSystem.store = {}
    RemoteMemoryFileSystem.pseudo_dirs = [""]
    remote_fs = RemoteMemoryFileSystem()

    def remote_url_to_fs(url: str, **_kwargs):
        path = StoragePath(url)
        assert path.scheme == protocol
        return remote_fs, "/".join(part for part in (path.netloc, path.key) if part)

    def remote_open_url(url: str, mode: str = "rb", **kwargs):
        fs, path = remote_url_to_fs(url)
        return fs.open(path, mode, **kwargs)

    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", remote_url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.open_url", remote_open_url)
    monkeypatch.setattr(runner, "url_to_fs", remote_url_to_fs)


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_completed_trial_is_durable_across_driver_termination_and_restored(protocol, tmp_path, monkeypatch):
    """A trial that finishes before the driver dies is durable at the remote path and restored intact.

    Harbor writes each trial straight to the ``output_dir`` jobs tree, so a driver killed before it
    returns leaves the completed trial on durable storage. A resumed run whose driver produces nothing
    new must still report that trial, proving the runner reads it back from the durable path rather
    than depending on a clean full-job return or a post-run upload sweep.
    """
    _memory_remote(protocol, monkeypatch)
    output_dir = f"{protocol}://eval-bucket-{tmp_path.name}/run"
    executor = _harbor_executor(f"resume-{tmp_path.name}")

    captured: dict = {}

    def dying_driver(config, overlay, driver_env) -> None:
        captured["jobs_dir"] = overlay.jobs_dir
        captured["job_name"] = overlay.job_name
        trial = StoragePath(overlay.jobs_dir) / overlay.job_name / "trial-one"
        (trial / "result.json").write_text(
            json.dumps({"task_name": "trial-one", "verifier_result": {"rewards": {"reward": 1.0}}})
        )
        (trial / "agent" / "trajectory.json").write_text('{"steps": []}')
        raise RuntimeError("preempted before seal")

    monkeypatch.setattr(runner, "run_harbor_driver", dying_driver)
    with pytest.raises(EvaluationError) as exc_info:
        executor(_running_model(), output_dir, {})
    assert exc_info.value.status is RunStatus.FAILED

    durable = StoragePath(captured["jobs_dir"]) / captured["job_name"] / "trial-one" / "result.json"
    assert durable.exists()

    def resumed_driver(config, overlay, driver_env) -> None:
        # Harbor's own resume finds the durable trial and writes nothing new this run.
        return None

    monkeypatch.setattr(runner, "run_harbor_driver", resumed_driver)
    outcome = executor(_running_model(), output_dir, {})

    # The resumed driver produced no trials, so total==1 means the durable trial was read back.
    assert outcome.metrics[executor.config.record_dataset]["total"] == 1.0
    assert outcome.metrics[executor.config.record_dataset]["accuracy"] == 1.0
    assert StoragePath(f"{output_dir}/samples_harbor.parquet").exists()
    assert StoragePath(f"{output_dir}/harbor_result.json").exists()


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


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_legacy_harbor_resume_imports_only_scored_trials(protocol, tmp_path, monkeypatch):
    _memory_remote(protocol, monkeypatch)
    output_dir = f"{protocol}://eval-bucket-{tmp_path.name}/run"
    legacy_trials = StoragePath(output_dir) / "harbor_trials"
    scored = legacy_trials / "scored"
    unscored = legacy_trials / "unscored"
    (scored / "result.json").write_text(
        json.dumps({"task_name": "scored", "verifier_result": {"rewards": {"reward": 1.0}}})
    )
    (scored / "agent" / "trajectory.json").write_text('{"steps": []}')
    (unscored / "result.json").write_text(json.dumps({"task_name": "unscored", "exception_info": {}}))
    (legacy_trials / "corrupt" / "result.json").write_text("not-json")

    job_dir = _job_dir(output_dir, "resume-job")
    _migrate_legacy_scored_trials(output_dir, job_dir)

    assert (job_dir / "scored" / "result.json").exists()
    assert (job_dir / "scored" / "agent" / "trajectory.json").exists()
    assert not (job_dir / "unscored" / "result.json").exists()
    assert not (job_dir / "corrupt" / "result.json").exists()


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
