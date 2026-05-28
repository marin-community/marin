# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace
from typing import cast

import pytest
from fray.types import JobStatus
from marin.rl.environments.inference_ctx import VLLMSamplingConfig, vLLMInferenceContextConfig
from marin.rl.orchestration import (
    _HostedRuntime,
    _rollout_schedule_ledger_path,
    _run_rl_coordinator,
    _train_worker_entry,
)
from marin.rl.rl_job import RLJobConfig, RunConfig
from marin.rl.rollout_schedule import derive_worker_seed
from marin.rl.rollout_storage import RolloutStorageConfig, StorageType
from marin.rl.rollout_worker import RolloutTrackerConfig


class _FakeHostedActor:
    def __init__(self, name: str, shutdown_calls: list[str]):
        self.handle = name
        self._name = name
        self._shutdown_calls = shutdown_calls

    def shutdown(self) -> None:
        self._shutdown_calls.append(self._name)


class _FakeJobHandle:
    def __init__(
        self,
        job_id: str,
        *,
        current_status: JobStatus = JobStatus.RUNNING,
        wait_status: JobStatus = JobStatus.SUCCEEDED,
    ):
        self.job_id = job_id
        self._status = current_status
        self._wait_status = wait_status
        self.terminate_calls = 0

    def wait(self, timeout=None, *, raise_on_failure=True):
        del timeout, raise_on_failure
        self._status = self._wait_status
        return self._wait_status

    def status(self):
        return self._status

    def terminate(self):
        self.terminate_calls += 1
        self._status = JobStatus.STOPPED


class _FakeClient:
    def __init__(self):
        self.submissions = []
        self.handles = []

    def submit(self, request):
        self.submissions.append(request)
        if request.name.endswith("-train"):
            handle = _FakeJobHandle(request.name, current_status=JobStatus.RUNNING, wait_status=JobStatus.SUCCEEDED)
        else:
            handle = _FakeJobHandle(request.name, current_status=JobStatus.RUNNING, wait_status=JobStatus.RUNNING)
        self.handles.append(handle)
        return handle


class _FakeActorMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return SimpleNamespace(result=lambda: None)


class _FakeRunStateHandle:
    def __init__(self):
        self.mark_completed = _FakeActorMethod()
        self.mark_failed = _FakeActorMethod()


class _FakeRLJob:
    def __init__(self, _config):
        pass

    def to_worker_configs(self):
        return _FakeWorkerConfig(seed=7, run_id="train"), _FakeWorkerConfig(
            seed=11,
            run_id="rl-test",
            inference_config=vLLMInferenceContextConfig(
                model_name="test-model",
                max_model_len=16,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.5,
                sampling_params=VLLMSamplingConfig(),
                seed=11,
            ),
            tracker_config=RolloutTrackerConfig(project="marin_iris_rl_debug", name="shared-rollout-name"),
        )


@dataclasses.dataclass(frozen=True)
class _FakeWorkerConfig:
    seed: int
    run_id: str
    worker_index: int = 0
    inference_config: object | None = None
    tracker_config: object | None = None
    weight_transfer: object = dataclasses.field(default_factory=lambda: SimpleNamespace(debug_weight_transfer=False))
    trainer: object = dataclasses.field(
        default_factory=lambda: SimpleNamespace(
            checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)),
        )
    )


def test_rollout_schedule_ledger_path_uses_file_rollout_storage_path():
    file_config = SimpleNamespace(
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.FILE,
            path="gs://marin/rl/run/rollouts/",
        )
    )
    memory_config = SimpleNamespace(
        rollout_storage=RolloutStorageConfig(
            storage_type=StorageType.IN_MEMORY,
            queue_name="rollouts",
        )
    )

    assert (
        _rollout_schedule_ledger_path(cast(RLJobConfig, file_config))
        == "gs://marin/rl/run/rollouts/_rollout_schedule_ledger"
    )
    assert _rollout_schedule_ledger_path(cast(RLJobConfig, memory_config)) is None


def test_run_rl_coordinator_shuts_down_hosted_actors_when_child_job_fails(monkeypatch):
    shutdown_calls: list[str] = []
    client = _FakeClient()
    hosted_actors = [
        _FakeHostedActor("curriculum", shutdown_calls),
        _FakeHostedActor("run-state", shutdown_calls),
        _FakeHostedActor("weight-transfer", shutdown_calls),
    ]
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=hosted_actors),
    )
    monkeypatch.setattr(
        "marin.rl.orchestration.wait_all",
        lambda _jobs, raise_on_failure: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    assert shutdown_calls == ["weight-transfer", "run-state", "curriculum"]


def test_run_rl_coordinator_stops_rollouts_after_trainer_success(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    wait_all_calls: list[list[str]] = []
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=2,
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr(
        "marin.rl.orchestration.wait_all",
        lambda jobs, raise_on_failure: wait_all_calls.append([job.job_id for job in jobs]),
    )

    _run_rl_coordinator(config)

    assert len(client.handles) == 3
    train_handle, rollout0_handle, rollout1_handle = client.handles
    assert train_handle.status() == JobStatus.SUCCEEDED
    assert rollout0_handle.terminate_calls == 1
    assert rollout1_handle.terminate_calls == 1
    assert wait_all_calls == [[rollout0_handle.job_id, rollout1_handle.job_id]]


def test_run_rl_coordinator_uses_run_config_ram_overrides(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            train_ram="300g",
            inference_ram="300g",
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    assert client.submissions[0].resources.ram == "300g"
    assert client.submissions[1].resources.ram == "300g"


def test_run_rl_coordinator_uses_run_config_zone_for_child_tpu_jobs(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v6e-8",
            inference_tpu_type="v6e-8",
            num_rollout_workers=1,
            regions=["us-east1"],
            zone="us-east1-d",
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    assert client.submissions[0].resources.zone == "us-east1-d"
    assert client.submissions[1].resources.zone == "us-east1-d"


def test_run_rl_coordinator_enables_unbuffered_logs_for_debug_checkpointer(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            regions=["us-central1"],
        ),
    )

    class _DebugRLJob(_FakeRLJob):
        def to_worker_configs(self):
            trainer = SimpleNamespace(checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=True)))
            return _FakeWorkerConfig(seed=7, run_id="train", trainer=trainer), _FakeWorkerConfig(
                seed=11, run_id="rollout"
            )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _DebugRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    for request in client.submissions:
        assert request.environment.env_vars["PYTHONUNBUFFERED"] == "1"
        assert request.environment.env_vars["JAX_TRACEBACK_FILTERING"] == "off"
        assert request.environment.env_vars["JAX_LOGGING_LEVEL"] == "INFO"
        assert (
            request.environment.env_vars["JAX_DEBUG_LOG_MODULES"]
            == "jax.experimental.array_serialization.serialization,"
            "jax.experimental.array_serialization.tensorstore_impl,jax._src.distributed"
        )
        assert request.environment.env_vars["JAX_INCLUDE_FULL_TRACEBACKS_IN_LOCATIONS"] == "1"
        assert request.environment.env_vars["TF_CPP_MIN_LOG_LEVEL"] == "0"
        assert request.environment.env_vars["TF_CPP_MAX_VLOG_LEVEL"] == "1"
        assert (
            request.environment.env_vars["TF_CPP_VMODULE"] == "coordination_service=2,coordination_service_agent=2,tsl=1"
        )


def test_run_rl_coordinator_enables_transfer_debug_env_vars(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=1,
            regions=["us-central1"],
        ),
    )

    class _DebugRLJob(_FakeRLJob):
        def to_worker_configs(self):
            trainer = SimpleNamespace(checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)))
            weight_transfer = SimpleNamespace(debug_weight_transfer=True)
            return _FakeWorkerConfig(
                seed=7,
                run_id="train",
                trainer=trainer,
                weight_transfer=weight_transfer,
            ), _FakeWorkerConfig(
                seed=11,
                run_id="rollout",
                weight_transfer=weight_transfer,
            )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _DebugRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 2
    for request in client.submissions:
        assert request.environment.env_vars["PYTHONUNBUFFERED"] == "1"
        assert request.environment.env_vars["JAX_TRACEBACK_FILTERING"] == "off"
        assert request.environment.env_vars["TF_CPP_MIN_LOG_LEVEL"] == "0"


def test_run_rl_coordinator_assigns_stable_rollout_wandb_names(monkeypatch):
    client = _FakeClient()
    hosted_runtime = _HostedRuntime(runtime=SimpleNamespace(), hosted_actors=[])
    config = SimpleNamespace(
        run_id="rl-test",
        resolved_instance_id="rl-test-instance",
        pip_dependency_groups=["math"],
        run_config=RunConfig(
            train_tpu_type="v5p-8",
            inference_tpu_type="v5p-8",
            num_rollout_workers=2,
            regions=["us-central1"],
        ),
    )

    monkeypatch.setattr("marin.rl.orchestration.current_client", lambda: client)
    monkeypatch.setattr("marin.rl.orchestration.RLJob", _FakeRLJob)
    monkeypatch.setattr(
        "marin.rl.orchestration._create_runtime_handles",
        lambda _client, _config: hosted_runtime,
    )
    monkeypatch.setattr("marin.rl.orchestration.wait_all", lambda _jobs, raise_on_failure: None)

    _run_rl_coordinator(config)

    assert len(client.submissions) == 3
    rollout0_config = client.submissions[1].entrypoint.callable_entrypoint.args[0]
    rollout1_config = client.submissions[2].entrypoint.callable_entrypoint.args[0]

    # Per-worker seeds are derived via `derive_worker_seed(base, worker_index)`
    # (jax.random.fold_in). Adjacent workers get fully decorrelated seeds — they
    # are NOT base+0 and base+1 as the old implementation produced.
    expected_seed_0 = derive_worker_seed(11, 0)
    expected_seed_1 = derive_worker_seed(11, 1)
    assert expected_seed_0 != expected_seed_1  # sanity: derivation is decorrelating
    assert rollout0_config.run_id == "rl-test-rollout-0"
    assert rollout0_config.seed == expected_seed_0
    assert rollout0_config.inference_config.seed == expected_seed_0
    assert rollout0_config.worker_index == 0
    assert rollout0_config.tracker_config.name == "rl-test-rollout-0"
    assert rollout1_config.run_id == "rl-test-rollout-1"
    assert rollout1_config.seed == expected_seed_1
    assert rollout1_config.inference_config.seed == expected_seed_1
    assert rollout1_config.worker_index == 1
    assert rollout1_config.tracker_config.name == "rl-test-rollout-1"


def test_train_worker_entry_does_not_mark_run_state_failed_on_attempt_crash(monkeypatch):
    runtime = SimpleNamespace(run_state=_FakeRunStateHandle())

    class _CrashingTrainWorker:
        def __init__(self, config, runtime):
            del config, runtime

        def train(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("marin.rl.orchestration.TrainWorker", _CrashingTrainWorker)

    with pytest.raises(RuntimeError, match="boom"):
        _train_worker_entry(train_config=SimpleNamespace(), runtime=runtime)

    assert runtime.run_state.mark_failed.calls == []
    assert runtime.run_state.mark_completed.calls == []
