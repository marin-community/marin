# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from finestore.reader import CompositeReader
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor import driver_config, runner
from marin.evaluation.harbor.dataset import materialize_harbor_dataset
from marin.evaluation.harbor.driver_config import (
    HarborBackendsUnavailable,
    HarborDatasetKind,
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
)
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    HarborTrial,
    _read_trials,
    _write_archive,
)
from marin.evaluation.records import RunStatus
from marin.evaluation.runner import EvaluationError
from marin.inference.iris import InferenceBackendState, RemoteInferenceSession
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem import StoragePath
from rigging.filesystem import factory as rigging_factory


def _running_model() -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(
            base_url="https://iris.example/proxy/t/token/serve.model/v1",
            model="qwen3-0.6b",
        )
    )


def _inference_session() -> RemoteInferenceSession:
    return RemoteInferenceSession(
        model=_running_model(),
        jobs=(),
        endpoint_name="/serve/test",
        endpoint_health_timeout_seconds=1800.0,
        streaming=True,
        tensor_parallel_size=1,
        backend_name="vllm",
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


def test_write_archive_writes_agentic_samples(tmp_path):
    trial = HarborTrial(
        task_id="task-one", trial_id="trial-1", reward=0.0, status="completed", trajectory_path=None, error=None
    )

    root = _write_archive([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert root == str(tmp_path)
    rows = CompositeReader(str(tmp_path)).scan("samples").to_pylist()
    assert len(rows) == 1
    assert rows[0]["doc_id"] == "task-one"
    assert rows[0]["trial_id"] == "trial-1"
    assert rows[0]["kind"] == "agentic"


def test_read_trials_and_archive_captures_trajectory(tmp_path):
    """A trial's trajectory is archived once, referenced by a finestore:// URI, and its steps flattened."""
    job_dir = tmp_path / "harbor_jobs" / "job"
    with_trajectory = job_dir / "trial-one"
    (with_trajectory / "agent").mkdir(parents=True)
    (with_trajectory / "result.json").write_text(
        json.dumps({"task_name": "task-one", "verifier_result": {"rewards": {"reward": 1.0}}})
    )
    (with_trajectory / "agent" / "trajectory.json").write_text(
        json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "hi"}]})
    )
    without_trajectory = job_dir / "trial-two"
    without_trajectory.mkdir(parents=True)
    (without_trajectory / "result.json").write_text(json.dumps({"task_name": "task-two"}))

    trials = _read_trials(StoragePath(str(job_dir)))

    by_task = {trial.task_id: trial for trial in trials}
    assert by_task["task-one"].trajectory_path == str(with_trajectory / "agent" / "trajectory.json")
    assert by_task["task-one"].trial_id == "trial-one"
    assert by_task["task-two"].trajectory_path is None

    archive_root = str(tmp_path / "archive")
    _write_archive(trials, "aime", archive_root)
    reader = CompositeReader(archive_root)
    samples = {row["doc_id"]: row for row in reader.scan("samples").to_pylist()}
    # The archived sample references its trajectory by a finestore:// URI, not the job-tree path.
    assert samples["task-one"]["trajectory_uri"].startswith("finestore://blobs/")
    assert reader.resolve(samples["task-one"]["trajectory_uri"]) is not None
    steps = reader.scan("steps").to_pylist()
    assert len(steps) == 1 and steps[0]["step_id"] == 1


def _memory_remote(protocol: str, monkeypatch) -> None:
    """Route ``protocol://`` reads and writes to a fresh in-memory filesystem; leave local paths alone.

    Patches rigging's factory (``url_to_fs``/``open_url``), which every read and write resolves through
    at call time -- ``StoragePath`` verbs, ``atomic_rename``, and the finestore archive writer -- so the
    remote object-store leg stays off real storage. Non-``protocol`` URLs (the pod-local ``jobs_dir`` the
    isolated driver writes to) fall through to the real factory so genuine local I/O still hits disk.
    """
    real_url_to_fs = rigging_factory.url_to_fs
    real_open_url = rigging_factory.open_url

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

    def remote_url_to_fs(url: str, **kwargs):
        path = StoragePath(url)
        if path.scheme != protocol:
            return real_url_to_fs(url, **kwargs)
        return remote_fs, "/".join(part for part in (path.netloc, path.key) if part)

    def remote_open_url(url: str, mode: str = "rb", **kwargs):
        if StoragePath(url).scheme != protocol:
            return real_open_url(url, mode, **kwargs)
        fs, path = remote_url_to_fs(url)
        return fs.open(path, mode, **kwargs)

    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", remote_url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.open_url", remote_open_url)


@pytest.mark.parametrize("protocol", ["gs", "s3"])
def test_completed_run_is_durable_on_remote_and_resumed_on_a_fresh_pod(protocol, tmp_path, monkeypatch):
    """A finished run's trials land on the durable ``output_dir`` tree and a fresh pod resumes them.

    Upstream Harbor writes each trial to a local ``jobs_dir`` (it does plain local-filesystem I/O and
    cannot write to a ``gs://``/``s3://`` URI), so Marin uploads the finished job tree to the remote
    ``output_dir`` for durability. A re-launch on a pod with no local scratch must still report those
    trials, proving they are hydrated back from the durable remote copy rather than surviving in
    pod-local state.
    """
    _memory_remote(protocol, monkeypatch)
    output_dir = f"{protocol}://eval-bucket-{tmp_path.name}/run"
    executor = _harbor_executor(f"resume-{tmp_path.name}")

    def completing_driver(config, overlay, driver_env, _backend_state) -> None:
        # The isolated driver writes to the local jobs_dir; it never sees the remote output_dir.
        assert StoragePath(overlay.jobs_dir).scheme != protocol
        trial = Path(overlay.jobs_dir) / overlay.job_name / "trial-one"
        (trial / "agent").mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text(
            json.dumps({"task_name": "trial-one", "verifier_result": {"rewards": {"reward": 1.0}}})
        )
        (trial / "agent" / "trajectory.json").write_text('{"steps": []}')

    # First pod runs to completion; Marin uploads the finished tree to the durable remote path.
    monkeypatch.setattr(runner, "_HARBOR_WORKDIR", tmp_path / "pod-one")
    monkeypatch.setattr(runner, "run_harbor_driver", completing_driver)
    first = executor(_inference_session(), output_dir, {})
    assert first.metrics[executor.config.record_dataset]["total"] == 1.0
    assert StoragePath(f"{output_dir}/harbor_jobs").exists()

    # A fresh pod with no local scratch and a driver that writes nothing new must still report the
    # trial, so it can only have come from the durable remote copy hydrated back down.
    monkeypatch.setattr(runner, "_HARBOR_WORKDIR", tmp_path / "pod-two")
    monkeypatch.setattr(runner, "run_harbor_driver", lambda *args, **kwargs: None)
    second = executor(_inference_session(), output_dir, {})

    assert second.metrics[executor.config.record_dataset]["total"] == 1.0
    assert second.metrics[executor.config.record_dataset]["accuracy"] == 1.0
    assert StoragePath(f"{output_dir}/SEALED").exists()
    assert StoragePath(f"{output_dir}/harbor_result.json").exists()


def test_managed_harbor_pauses_and_resumes_after_inference_recovers(tmp_path, monkeypatch):
    output_dir = str(tmp_path / "run")
    executor = _harbor_executor(f"managed-{tmp_path.name}")

    class RecoveringSession:
        model = _running_model()
        recovery_waits = 0
        unavailable = True

        def backend_state(self) -> InferenceBackendState:
            if self.unavailable:
                return InferenceBackendState.RECOVERING
            return InferenceBackendState.READY

        def wait_until_ready(self) -> None:
            self.recovery_waits += 1
            self.unavailable = False

    session = RecoveringSession()
    driver_starts = 0

    def run_driver(_config, overlay, _driver_env, backend_state) -> None:
        nonlocal driver_starts
        driver_starts += 1
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        completed_result = job_dir / "trial-one" / "result.json"
        completed_result.parent.mkdir(parents=True, exist_ok=True)
        if not completed_result.exists():
            completed_result.write_text(
                json.dumps({"task_name": "trial-one", "verifier_result": {"rewards": {"reward": 1.0}}})
            )
        zero_reward_result = job_dir / "trial-three" / "result.json"
        zero_reward_result.parent.mkdir(parents=True, exist_ok=True)
        if not zero_reward_result.exists():
            zero_reward_result.write_text(
                json.dumps({"task_name": "trial-three", "verifier_result": {"rewards": {"reward": 0.0}}})
            )
        interrupted_result = job_dir / "trial-two" / "result.json"
        if driver_starts == 1:
            interrupted_result.parent.mkdir(parents=True, exist_ok=True)
            interrupted_result.write_text(
                json.dumps(
                    {
                        "task_name": "trial-two",
                        "verifier_result": None,
                        "exception_info": {"exception_type": "InternalServerError"},
                    }
                )
            )
        else:
            assert completed_result.exists()
            assert zero_reward_result.exists()
            assert not interrupted_result.exists()
            interrupted_result.parent.mkdir(parents=True, exist_ok=True)
            interrupted_result.write_text(
                json.dumps({"task_name": "trial-two", "verifier_result": {"rewards": {"reward": 1.0}}})
            )
        if backend_state() is InferenceBackendState.RECOVERING:
            raise HarborBackendsUnavailable("inference backends are not ready")

    monkeypatch.setattr(runner, "run_harbor_driver", run_driver)

    outcome = executor(session, output_dir, {})

    assert session.recovery_waits == 1
    assert driver_starts == 2
    assert outcome.metrics[executor.config.record_dataset] == {
        "accuracy": 2 / 3,
        "mean_reward": 2 / 3,
        "solved": 2.0,
        "total": 3.0,
    }


def test_harbor_driver_terminates_when_dependency_becomes_unavailable(tmp_path, monkeypatch):
    terminated_return_codes: list[int | None] = []
    terminate_process_group = driver_config.terminate_process_group

    def terminate(process, *, grace_period):
        terminate_process_group(process, grace_period=grace_period)
        terminated_return_codes.append(process.returncode)

    monkeypatch.setattr(driver_config, "terminate_process_group", terminate)
    monkeypatch.setattr(driver_config, "_driver_command", lambda *_args: ["sleep", "60"])
    monkeypatch.setattr(driver_config, "_BACKEND_POLL_SECONDS", 0.01)

    def backend_state() -> InferenceBackendState:
        return InferenceBackendState.RECOVERING

    with pytest.raises(HarborBackendsUnavailable):
        driver_config.run_harbor_driver(
            _validated_config(),
            HarborRuntimeOverlay(
                job_name="driver-stop",
                jobs_dir=str(tmp_path / "jobs"),
                dataset_path=None,
                endpoint_url="https://iris.example/capability/v1",
                served_model="model",
                task_limit=1,
                model_agent_kwargs={},
            ),
            {},
            backend_state,
        )

    assert len(terminated_return_codes) == 1
    assert terminated_return_codes[0] is not None


def test_harbor_driver_classifies_fast_failure_from_unavailable_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_config, "_driver_command", lambda *_args: ["false"])

    def backend_state() -> InferenceBackendState:
        return InferenceBackendState.RECOVERING

    with pytest.raises(HarborBackendsUnavailable):
        driver_config.run_harbor_driver(
            _validated_config(),
            HarborRuntimeOverlay(
                job_name="driver-failure",
                jobs_dir=str(tmp_path / "jobs"),
                dataset_path=None,
                endpoint_url="https://iris.example/capability/v1",
                served_model="model",
                task_limit=1,
                model_agent_kwargs={},
            ),
            {},
            backend_state,
        )


def test_harbor_executor_passes_opaque_policy_and_runtime_overlay_to_driver(tmp_path, monkeypatch):
    captured: dict = {}

    def run_driver(config, overlay, driver_env, _backend_state) -> None:
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
    session = _inference_session()
    model = session.model

    executor = HarborExecutor(
        _validated_config(
            dataset_selector=f"toy-{tmp_path.name}",
        ),
        task_limit=7,
        model_agent_kwargs={"extra_body": "{}"},
        secret_env_keys=("DAYTONA_API_KEY",),
    )
    outcome = executor(
        session,
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
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
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
        executor(_inference_session(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.FAILED
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
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

    outcome = executor(_inference_session(), str(tmp_path), {})

    assert outcome.metrics[executor.config.record_dataset]["accuracy"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 0
