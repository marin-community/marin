# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from marin.evaluation.harbor import driver_config, runner
from marin.evaluation.harbor.agent_context import (
    DEFAULT_MODEL_INFO,
    reconciled_model_info,
    served_model_info,
)
from marin.evaluation.harbor.dataset import local_harbor_dataset_path
from marin.evaluation.harbor.driver_config import (
    HarborBackendsUnavailable,
    HarborDatasetKind,
    HarborErrorTaxonomy,
    HarborRuntimeOverlay,
    ValidatedHarborConfig,
)
from marin.evaluation.harbor.runner import (
    HarborExecutor,
    _read_trial,
    _read_trials,
)
from marin.evaluation.records import BenchmarkMetadataRef, BenchmarkMetricRef, MetricKind, RunStatus
from marin.evaluation.runner import EvaluationError
from marin.inference.iris import InferenceBackendState, RemoteInferenceSession
from marin.inference.types import OpenAIEndpoint, RunningModel
from rigging.filesystem.conditional_object import ConditionalWriteError, VersionedBytes
from rigging.filesystem.storage_path import StoragePath

_ERROR_TAXONOMY = HarborErrorTaxonomy(
    infrastructure=frozenset({"InfrastructureError", "InternalServerError"}),
    agent=frozenset({"AgentError", "AgentTimeoutError"}),
    passthrough=frozenset({"PassthroughError"}),
    undecided=frozenset({"VerifierTimeoutError"}),
    commit="1" * 40,
)


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
    n_benchmark: int = 1,
    trials_per_task: int = 1,
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
        error_taxonomy=_ERROR_TAXONOMY,
        max_input_tokens=32768,
        max_output_tokens=8192,
        benchmark=BenchmarkMetadataRef(
            schema_version=1,
            task=f"hf://{dataset_selector}" if dataset_kind == HarborDatasetKind.HUGGING_FACE else dataset_selector,
            primary_metric="reward",
            metric_kind=MetricKind.CONTINUOUS,
            metrics=(
                BenchmarkMetricRef(
                    name="reward",
                    source_name="reward",
                    kind=MetricKind.CONTINUOUS,
                    higher_is_better=True,
                ),
            ),
            n_benchmark=n_benchmark,
            n_attempted=n_benchmark,
        ),
        trials_per_task=trials_per_task,
    )


def test_reconciled_model_info_takes_context_limits_from_the_served_model():
    served = served_model_info(max_model_len=1048576, max_gen_toks=393216)

    assert reconciled_model_info(served, None) == {
        "max_input_tokens": 1048576,
        "max_output_tokens": 393216,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }


def test_reconciled_model_info_keeps_harbor_defaults_when_the_model_states_no_limits():
    served = served_model_info(max_model_len=None, max_gen_toks=None)

    assert reconciled_model_info(served, {"input_cost_per_token": 1.5}) == {
        **DEFAULT_MODEL_INFO,
        "input_cost_per_token": 1.5,
    }


@pytest.mark.parametrize("policy_max_input_tokens", [64512, 65536])
def test_reconciled_model_info_keeps_a_policy_limit_within_the_served_window(policy_max_input_tokens):
    served = served_model_info(max_model_len=65536, max_gen_toks=16384)

    resolved = reconciled_model_info(served, {"max_input_tokens": policy_max_input_tokens})

    assert (resolved["max_input_tokens"], resolved["max_output_tokens"]) == (policy_max_input_tokens, 16384)


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ({"max_input_tokens": 64512}, r"64512.*serve\.max_model_len is only 32768"),
        ({"max_output_tokens": 16384}, r"16384.*generation\.max_gen_toks is only 8192"),
    ],
)
def test_reconciled_model_info_rejects_a_policy_limit_above_the_served_window(policy, message):
    served = served_model_info(max_model_len=32768, max_gen_toks=8192)

    with pytest.raises(ValueError, match=message):
        reconciled_model_info(served, policy)


def _write_job_record(job_dir: Path, n_total_trials: int, config: ValidatedHarborConfig) -> None:
    """Harbor's own job-level bookkeeping: the count the coverage denominator comes from."""
    job_dir.mkdir(parents=True, exist_ok=True)
    job_dir.joinpath("result.json").write_text(
        json.dumps(
            {
                "n_total_trials": n_total_trials,
                "benchmark_metadata": [config.benchmark.model_dump(mode="json")],
            }
        )
    )


def test_local_harbor_dataset_path_rebases_onto_worker_workspace(tmp_path, monkeypatch):
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

    assert local_harbor_dataset_path(config) == dataset


def test_read_trials_reads_every_result(tmp_path):
    job_dir = tmp_path / "harbor_jobs" / "job"
    first_trial = job_dir / "trial-one"
    first_trial.mkdir(parents=True)
    (first_trial / "result.json").write_text(
        json.dumps({"task_name": "task-one", "verifier_result": {"rewards": {"reward": 1.0}}})
    )
    second_trial = job_dir / "trial-two"
    second_trial.mkdir(parents=True)
    (second_trial / "result.json").write_text(json.dumps({"task_name": "task-two"}))

    trials = _read_trials(StoragePath(str(job_dir)), _ERROR_TAXONOMY)

    assert [(trial.reward, trial.scored) for trial in trials] == [(1.0, True), (0.0, False)]


@pytest.mark.parametrize(
    ("exception_type", "verifier_result", "expected_scored", "expected_error_type"),
    [
        (None, {"rewards": {"reward": 1.0}}, True, None),
        ("InfrastructureError", {"rewards": {"reward": 1.0}}, False, "InfrastructureError"),
        ("AgentError", None, True, "AgentError"),
        ("PassthroughError", {"rewards": {"reward": 0.5}}, True, "PassthroughError"),
        ("PassthroughError", None, False, "PassthroughError"),
        ("VerifierTimeoutError", {"rewards": {"reward": 0.5}}, False, "VerifierTimeoutError"),
        ("VerifierTimeoutError", None, False, "VerifierTimeoutError"),
        ("NewHarborError", {"rewards": {"reward": 1.0}}, False, "unknown:NewHarborError"),
    ],
)
def test_read_trial_applies_harbor_error_taxonomy(
    tmp_path, exception_type, verifier_result, expected_scored, expected_error_type
):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result = {"task_name": "task", "verifier_result": verifier_result}
    if exception_type is not None:
        result["exception_info"] = {"exception_type": exception_type, "exception_message": "failed"}
    result_file = trial_dir / "result.json"
    result_file.write_text(json.dumps(result))

    trial = _read_trial(StoragePath(str(result_file)), _ERROR_TAXONOMY)

    assert trial.scored is expected_scored
    assert (trial.error or {}).get("type") == expected_error_type


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
        job_dir = StoragePath(overlay.jobs_dir) / overlay.job_name
        (job_dir / "result.json").write_text(
            json.dumps({"n_total_trials": 1, "benchmark_metadata": [config.benchmark.model_dump(mode="json")]})
        )
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
    assert outcome.canonical_metrics[executor.config.record_dataset]["reward"] == 1.0
    assert (StoragePath.parse(output_dir) / "harbor_result.json").exists()


def test_managed_harbor_pauses_and_resumes_after_inference_recovers(tmp_path, monkeypatch):
    output_dir = str(tmp_path / "run")
    executor = _harbor_executor(f"managed-{tmp_path.name}", n_benchmark=4)

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
        _write_job_record(job_dir, 4, _config)
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
        agent_failure_result = job_dir / "trial-four" / "result.json"
        agent_failure_result.parent.mkdir(parents=True, exist_ok=True)
        if not agent_failure_result.exists():
            agent_failure_result.write_text(
                json.dumps(
                    {
                        "task_name": "trial-four",
                        "verifier_result": None,
                        "exception_info": {"exception_type": "AgentError"},
                    }
                )
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
            assert agent_failure_result.exists()
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
        "mean_reward": 0.5,
        "solved": 2.0,
        "total": 4.0,
        "attempted": 4.0,
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
                archive_root=str(tmp_path / "archive"),
                archive_dataset="dataset",
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
                archive_root=str(tmp_path / "archive"),
                archive_dataset="dataset",
            ),
            {},
            backend_state,
        )


@pytest.mark.parametrize("dataset_kind", [HarborDatasetKind.HARBOR_REGISTRY, HarborDatasetKind.HUGGING_FACE])
def test_harbor_executor_passes_opaque_policy_and_runtime_overlay_to_driver(tmp_path, monkeypatch, dataset_kind):
    captured: dict = {}

    def run_driver(config, overlay, driver_env, _backend_state) -> None:
        captured["config"] = config
        captured["overlay"] = overlay
        captured["env"] = driver_env
        _write_job_record(Path(overlay.jobs_dir) / overlay.job_name, 1, config)
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

    selector = (
        f"toy-{tmp_path.name}" if dataset_kind == HarborDatasetKind.HARBOR_REGISTRY else f"example/{tmp_path.name}"
    )
    executor = HarborExecutor(
        _validated_config(
            dataset_kind=dataset_kind,
            dataset_selector=selector,
        ),
        task_limit=7,
        model_agent_kwargs={"extra_body": "{}"},
        secret_env_keys=("DAYTONA_API_KEY",),
    )
    env_vars = {"DAYTONA_API_KEY": "daytona-key"}
    if dataset_kind == HarborDatasetKind.HUGGING_FACE:
        env_vars["HF_TOKEN"] = "hf-key"
    outcome = executor(
        session,
        str(tmp_path),
        env_vars,
    )

    assert captured["config"] is executor.config
    assert captured["overlay"].endpoint_url == model.endpoint.base_url
    assert captured["overlay"].served_model == "qwen3-0.6b"
    assert captured["overlay"].task_limit == 7
    assert captured["overlay"].model_agent_kwargs == {"extra_body": "{}"}
    assert captured["overlay"].archive_root == str(tmp_path)
    assert captured["overlay"].archive_dataset == executor.config.record_dataset
    assert captured["overlay"].dataset_path is None
    assert captured["env"]["DAYTONA_API_KEY"] == "daytona-key"
    if dataset_kind == HarborDatasetKind.HUGGING_FACE:
        assert captured["env"]["HF_TOKEN"] == "hf-key"
    assert "OPENAI_API_KEY" not in captured["env"]
    assert outcome.canonical_metrics[executor.config.record_dataset]["reward"] == 1.0


def _harbor_executor(dataset: str, *, n_benchmark: int = 1, trials_per_task: int = 1) -> HarborExecutor:
    return HarborExecutor(
        _validated_config(dataset_selector=dataset, n_benchmark=n_benchmark, trials_per_task=trials_per_task),
        task_limit=None,
        model_agent_kwargs={},
    )


def test_harbor_executor_fails_when_too_few_trials_were_graded(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
        assert isinstance(driver_env, dict)
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 1, _config)
        trial_dir = job_dir / "trial-one"
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "task_name": "trial-one",
                    "verifier_result": {"rewards": {"reward": 0.0}},
                    "exception_info": {
                        "exception_type": "InfrastructureError",
                        "exception_message": "model request failed",
                    },
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"failed-{tmp_path.name}")

    with pytest.raises(EvaluationError) as exc_info:
        executor(_inference_session(), str(tmp_path), {})

    # Infrastructure errors are ungraded: the run scored 0% of its attempted trials and fails the gate.
    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage["failed-" + tmp_path.name].errors == {"InfrastructureError": 1}
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["unscored_trials"] == 1
    assert result["errors"] == {"InfrastructureError": 1}


def test_harbor_executor_counts_agent_failure_without_verifier_as_zero_reward(tmp_path, monkeypatch):

    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 20, _config)
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
    executor = _harbor_executor(f"gated-{tmp_path.name}", n_benchmark=20)

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    metrics = outcome.metrics[dataset]
    assert metrics["total"] == 20.0
    assert metrics["attempted"] == 20.0
    assert outcome.canonical_metrics[dataset]["reward"] == pytest.approx(10 / 20)
    assert outcome.canonical_metrics[dataset]["reward_stderr"] == pytest.approx(0.1147078669)
    assert outcome.tasks is not None
    assert outcome.tasks[0].benchmark == executor.config.benchmark
    coverage = outcome.coverage[dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (20, 20)
    assert coverage.errors == {"AgentTimeoutError": 1}
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["errors"] == {"AgentTimeoutError": 1}


def test_harbor_executor_counts_undecided_error_against_completion_gate(tmp_path, monkeypatch):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 20, _config)
        for index in range(20):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            result = {"task_name": f"task-{index}", "verifier_result": {"rewards": {"reward": 1.0}}}
            if index == 0:
                result = {
                    "task_name": f"task-{index}",
                    "verifier_result": None,
                    "exception_info": {"exception_type": "VerifierTimeoutError"},
                }
            trial_dir.joinpath("result.json").write_text(json.dumps(result))

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"undecided-{tmp_path.name}", n_benchmark=20)

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    assert outcome.metrics[dataset]["total"] == 19.0
    coverage = outcome.coverage[dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (20, 19)
    assert coverage.errors == {"VerifierTimeoutError": 1}


def test_harbor_executor_rejects_unknown_error_name(tmp_path, monkeypatch):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 1, _config)
        trial_dir = job_dir / "trial-one"
        trial_dir.mkdir(parents=True)
        trial_dir.joinpath("result.json").write_text(
            json.dumps(
                {
                    "task_name": "task-one",
                    "verifier_result": {"rewards": {"reward": 1.0}},
                    "exception_info": {"exception_type": "NewHarborError"},
                }
            )
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"unknown-{tmp_path.name}")

    with pytest.raises(EvaluationError) as exc_info:
        executor(_inference_session(), str(tmp_path), {})

    assert exc_info.value.status is RunStatus.INFRA_FAILED
    assert exc_info.value.coverage[executor.config.record_dataset].errors == {"unknown:NewHarborError": 1}


@pytest.mark.parametrize("exception_type", ["AgentTimeoutError", "PassthroughError"])
def test_harbor_executor_preserves_scored_errors(tmp_path, monkeypatch, exception_type):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 10, _config)
        for index in range(10):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "task_name": f"task-{index}",
                "verifier_result": {"rewards": {"reward": 1.0 if index >= 7 else 0.0}},
            }
            if index < 4:
                result["exception_info"] = {"exception_type": exception_type}
            trial_dir.joinpath("result.json").write_text(json.dumps(result))

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"timeout-{tmp_path.name}", n_benchmark=10)

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    assert outcome.metrics[dataset]["total"] == 10.0
    assert outcome.canonical_metrics[dataset]["reward"] == pytest.approx(0.3)
    coverage = outcome.coverage[dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (10, 10)
    assert coverage.errors == {exception_type: 4}
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["unscored_trials"] == 0


def test_preflight_benchmark_count_matches_job_metadata(tmp_path, monkeypatch):

    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 4, _config)
        for index in range(4):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_dir.joinpath("result.json").write_text(
                json.dumps({"task_name": f"task-{index}", "verifier_result": {"rewards": {"reward": 1.0}}})
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"known-{tmp_path.name}", n_benchmark=4)

    outcome = executor(_inference_session(), str(tmp_path), {})

    dataset = executor.config.record_dataset
    coverage = outcome.coverage[dataset]
    assert coverage.n_benchmark == 4
    assert coverage.n_attempted == 4
    assert coverage.n_scored == 4
    assert outcome.metrics[dataset]["attempted"] == 4


def test_harbor_attempt_count_includes_repeated_trials_per_task(tmp_path, monkeypatch):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 6, _config)
        for index in range(6):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_dir.joinpath("result.json").write_text(
                json.dumps({"task_name": f"task-{index // 3}", "verifier_result": {"rewards": {"reward": 1.0}}})
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"repeated-{tmp_path.name}", n_benchmark=2, trials_per_task=3)

    outcome = executor(_inference_session(), str(tmp_path), {})

    coverage = outcome.coverage[executor.config.record_dataset]
    assert (coverage.n_benchmark, coverage.n_attempted, coverage.n_scored) == (6, 6, 6)


def test_harbor_executor_rejects_job_metadata_that_differs_from_preflight(tmp_path, monkeypatch):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        changed = _config.benchmark.model_copy(update={"n_benchmark": 2, "n_attempted": 2})
        job_dir.mkdir(parents=True, exist_ok=True)
        job_dir.joinpath("result.json").write_text(
            json.dumps({"n_total_trials": 2, "benchmark_metadata": [changed.model_dump(mode="json")]})
        )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"mismatch-{tmp_path.name}")

    with pytest.raises(EvaluationError, match="benchmark metadata differs from preflight"):
        executor(_inference_session(), str(tmp_path), {})


def test_harbor_missing_results_reduce_scored_not_intended_count(tmp_path, monkeypatch):
    def run_driver(_config, overlay, _driver_env, _backend_state) -> None:
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 20, _config)
        for index in range(19):
            trial_dir = job_dir / f"trial-{index}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            trial_dir.joinpath("result.json").write_text(
                json.dumps({"task_name": f"task-{index}", "verifier_result": {"rewards": {"reward": 1.0}}})
            )

    monkeypatch.setattr("marin.evaluation.harbor.runner.run_harbor_driver", run_driver)
    executor = _harbor_executor(f"missing-{tmp_path.name}", n_benchmark=20)

    outcome = executor(_inference_session(), str(tmp_path), {})

    coverage = outcome.coverage[executor.config.record_dataset]
    assert (coverage.n_attempted, coverage.n_scored) == (20, 19)
    assert coverage.errors == {"no_result_written": 1}


def test_harbor_executor_accepts_zero_reward_without_exception_info(tmp_path, monkeypatch):
    def run_driver(_config, overlay, driver_env, _backend_state) -> None:
        assert isinstance(driver_env, dict)
        job_dir = Path(overlay.jobs_dir) / overlay.job_name
        _write_job_record(job_dir, 1, _config)
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

    assert outcome.canonical_metrics[executor.config.record_dataset]["reward"] == 0.0
    result = json.loads((tmp_path / "harbor_result.json").read_text())
    assert result["unscored_trials"] == 0
