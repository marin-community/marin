# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from finestore.reader import ReadView
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
from rigging.filesystem.conditional_object import ConditionalWriteError, VersionedBytes


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


def _write_job_record(job_dir: Path, n_total_trials: int) -> None:
    """Harbor's own job-level bookkeeping: the count the coverage denominator comes from."""
    job_dir.mkdir(parents=True, exist_ok=True)
    job_dir.joinpath("result.json").write_text(json.dumps({"n_total_trials": n_total_trials}))


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
        task_id="task-one",
        trial_id="trial-1",
        reward=0.0,
        scored=True,
        status="completed",
        trajectory_path=None,
        error=None,
    )

    root = _write_archive([trial], "hf://DCAgent2/terminal_bench_2", str(tmp_path))

    assert root == str(tmp_path)
    rows = ReadView(str(tmp_path)).scan("samples").to_pylist()
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
    reader = ReadView(archive_root)
    samples = {row["doc_id"]: row for row in reader.scan("samples").to_pylist()}
    # The archived sample references its trajectory by a finestore:// URI, not the job-tree path.
    assert samples["task-one"]["trajectory_uri"].startswith("finestore://blobs/")
    assert reader.resolve(samples["task-one"]["trajectory_uri"]) is not None
    steps = reader.scan("steps").to_pylist()
    assert len(steps) == 1 and steps[0]["step_id"] == 1


def _memory_remote(protocol: str, monkeypatch) -> None:
    """Route ``protocol://`` reads and writes to a fresh in-memory filesystem.

    Patches rigging's factory (``url_to_fs``/``open_url``), which every read and write resolves through
    at call time -- ``StoragePath`` verbs, ``atomic_rename``, and the finestore archive writer -- so the
    whole executor path stays off real object storage.
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
    versions: dict[str, int] = {}

    def remote_url_to_fs(url: str, **_kwargs):
        path = StoragePath(url)
        assert path.scheme == protocol
        return remote_fs, "/".join(part for part in (path.netloc, path.key) if part)

    def remote_open_url(url: str, mode: str = "rb", **kwargs):
        fs, path = remote_url_to_fs(url)
        return fs.open(path, mode, **kwargs)

    class MemoryConditionalObject:
        def __init__(self, path: str) -> None:
            self.path = path

        def read(self) -> VersionedBytes | None:
            _, key = remote_url_to_fs(self.path)
            if not remote_fs.exists(key):
                return None
            return VersionedBytes(remote_fs.cat(key), str(versions[self.path]))

        def write(self, data: bytes, *, expected_version: str | None) -> str:
            current = versions.get(self.path)
            if (None if current is None else str(current)) != expected_version:
                raise ConditionalWriteError(f"stale version for {self.path}")
            version = (current or 0) + 1
            _, key = remote_url_to_fs(self.path)
            remote_fs.pipe(key, data)
            versions[self.path] = version
            return str(version)

    monkeypatch.setattr("rigging.filesystem.factory.url_to_fs", remote_url_to_fs)
    monkeypatch.setattr("rigging.filesystem.factory.open_url", remote_open_url)
    monkeypatch.setattr("finestore.commit.conditional_object", MemoryConditionalObject)


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

    def dying_driver(config, overlay, driver_env, _backend_state) -> None:
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
        executor(_inference_session(), output_dir, {})
    assert exc_info.value.status is RunStatus.FAILED

    durable = StoragePath(captured["jobs_dir"]) / captured["job_name"] / "trial-one" / "result.json"
    assert durable.exists()

    def resumed_driver(config, overlay, driver_env, _backend_state) -> None:
        # Harbor's own resume finds the durable trial and writes nothing new this run.
        return None

    monkeypatch.setattr(runner, "run_harbor_driver", resumed_driver)
    outcome = executor(_inference_session(), output_dir, {})

    # The resumed driver produced no trials, so total==1 means the durable trial was read back.
    assert outcome.metrics[executor.config.record_dataset]["total"] == 1.0
    assert outcome.metrics[executor.config.record_dataset]["accuracy"] == 1.0
    assert ReadView(output_dir).is_sealed()
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
        _write_job_record(job_dir, 3)
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
        "attempted": 3.0,
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


def test_harbor_executor_fails_when_too_few_trials_were_graded(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
        assert isinstance(driver_env, dict)
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 1)
        trial_dir = job_dir / "trial-one"
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

    # A trial that errored is an ungraded item, not a wrong answer: the only trial here is ungraded,
    # so the run graded 0% of what it attempted and is rejected as an infrastructure failure.
    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage["failed-" + tmp_path.name].errors == {"AgentError": 1}
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["failed_trials"] == 1
    assert result["errors"] == {"AgentError": 1}


def test_harbor_executor_admits_a_batch_that_clears_the_completion_gate(tmp_path, monkeypatch):
    """One timed-out trial in twenty does not discard the other nineteen. The run keeps its aggregate
    and its error distribution, and the aggregate is over the trials a verifier actually graded."""

    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 20)
        for index in range(20):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            if index == 0:
                trial_dir.joinpath("result.json").write_text(
                    json.dumps(
                        {
                            "task_name": f"task-{index}",
                            "verifier_result": None,
                            "exception_info": {"exception_type": "AgentTimeoutError"},
                        }
                    )
                )
                continue
            trial_dir.joinpath("result.json").write_text(
                json.dumps(
                    {
                        "task_name": f"task-{index}",
                        "verifier_result": {"rewards": {"reward": 1.0 if index % 2 else 0.0}},
                    }
                )
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"gated-{tmp_path.name}")

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    metrics = outcome.metrics[dataset]
    assert metrics["total"] == 19.0
    assert metrics["attempted"] == 20.0
    # 10 of the 19 graded trials solved: dividing by 20 instead would publish the worst case as the
    # estimate, scoring the timed-out trial as a wrong answer.
    assert metrics["accuracy"] == pytest.approx(10 / 19)
    coverage = outcome.coverage[dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (20, 19)
    assert coverage.errors == {"AgentTimeoutError": 1}


def test_an_unreadable_job_record_reports_unknown_coverage_rather_than_complete(tmp_path, monkeypatch):
    """Without Harbor's job record there is no denominator. Falling back to the number of results
    found would certify exactly the interrupted runs as complete, so coverage stays unreported and
    the completion gate -- which has no rate to test -- does not reject the batch either."""

    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        for index in range(4):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_dir.joinpath("result.json").write_text(
                json.dumps({"task_name": f"task-{index}", "verifier_result": {"rewards": {"reward": 1.0}}})
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"unknown-{tmp_path.name}")

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    coverage = outcome.coverage[dataset]
    assert coverage.n_attempted is None
    assert coverage.n_scored == 4
    # No attempted count means no "attempted" metric to publish, rather than one equal to the scored
    # count, which would read as complete.
    assert "attempted" not in outcome.metrics[dataset]


def test_harbor_attempted_trials_come_from_the_job_record_not_the_result_glob(tmp_path, monkeypatch):
    """A trial that dies before writing a result leaves no file. Counting result files would report
    perfect coverage for exactly the runs that lost the most trials."""

    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 20)
        for index in range(19):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_dir.joinpath("result.json").write_text(
                json.dumps({"task_name": f"task-{index}", "verifier_result": {"rewards": {"reward": 1.0}}})
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"missing-{tmp_path.name}")

    outcome = executor(_inference_session(), str(tmp_path), {})

    coverage = outcome.coverage[executor.config.record_dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (20, 19)
    assert coverage.errors == {"no_result_written": 1}


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
        assert isinstance(driver_env, dict)
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 1)
        trial_dir = job_dir / "trial-one"
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
