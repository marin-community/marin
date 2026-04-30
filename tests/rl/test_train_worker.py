# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import fsspec
from marin.rl.rl_losses import RLOOLoss
from marin.rl.telemetry import TelemetryEvent, TrackerStream
from marin.rl.train_worker import (
    BatchPrepTiming,
    InitialRolloutState,
    TrainWorker,
    _initial_rollout_state,
    _resume_safe_weight_transfer_metrics,
    _training_step_timing_metrics,
)
from marin.rl.weight_transfer.base import WeightTransferServerMetrics


def test_drop_bootstrap_model_references_clears_reference_model_when_kl_disabled():
    worker = TrainWorker.__new__(TrainWorker)
    model = object()
    worker.loss_module = RLOOLoss(kl_coef=0.0)
    worker.initial_model = model
    worker.reference_model = model

    worker._drop_bootstrap_model_references()

    assert worker.initial_model is None
    assert worker.reference_model is None


def test_drop_bootstrap_model_references_preserves_reference_model_when_kl_enabled():
    worker = TrainWorker.__new__(TrainWorker)
    model = object()
    worker.loss_module = RLOOLoss(kl_coef=0.01)
    worker.initial_model = model
    worker.reference_model = model

    worker._drop_bootstrap_model_references()

    assert worker.initial_model is None
    assert worker.reference_model is model


def test_record_train_step_updates_replay_buffer_and_shared_run_state():
    recorded_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            recorded_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    worker = TrainWorker.__new__(TrainWorker)
    worker.replay_buffer = SimpleNamespace(set_current_step=recorded_steps.append)
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())

    worker._record_train_step(7)

    assert recorded_steps == [7, 7]


def test_initialize_telemetry_writes_trainer_event_shard_and_registers_artifact(tmp_path):
    registered_artifacts = []

    class _FakeRemoteMethod:
        def remote(self, artifact_ref):
            registered_artifacts.append(artifact_ref)
            return SimpleNamespace(result=lambda: None)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        metadata_path=str(tmp_path / "metadata"),
        run_id="rl-test",
        root_run_id="rl-test",
        instance_id="attempt-abc",
    )
    worker._runtime = SimpleNamespace(run_state=SimpleNamespace(register_artifact_ref=_FakeRemoteMethod()))
    worker._event_writer = None

    worker._initialize_telemetry()

    assert worker._event_writer is not None
    assert len(registered_artifacts) == 1

    fs = fsspec.filesystem("file")
    assert fs.exists(worker._event_writer.path)
    with fs.open(worker._event_writer.path) as handle:
        events = [TelemetryEvent.from_json(line) for line in handle.read().splitlines() if line]

    assert len(events) == 1
    assert events[0].stream == TrackerStream.TRAINER
    assert events[0].event_type == "worker_started"
    assert events[0].run_id == "rl-test"
    assert events[0].payload == {"worker_role": "trainer"}


def test_initial_rollout_state_uses_bootstrap_weights_for_fresh_run():
    assert _initial_rollout_state(0) == InitialRolloutState(weight_step=-1, published_train_step=None)


def test_initial_rollout_state_reuses_recovered_step_for_resumed_run():
    assert _initial_rollout_state(68) == InitialRolloutState(weight_step=68, published_train_step=68)


def test_seed_initial_rollout_state_updates_replay_buffer_only_for_fresh_run():
    recorded_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            recorded_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    worker = TrainWorker.__new__(TrainWorker)
    worker.replay_buffer = SimpleNamespace(set_current_step=recorded_steps.append)
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())

    worker._seed_initial_rollout_state(InitialRolloutState(weight_step=-1, published_train_step=None))

    assert recorded_steps == [-1]


def test_seed_initial_rollout_state_publishes_resumed_step():
    recorded_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            recorded_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    worker = TrainWorker.__new__(TrainWorker)
    worker.replay_buffer = SimpleNamespace(set_current_step=recorded_steps.append)
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())

    worker._seed_initial_rollout_state(InitialRolloutState(weight_step=68, published_train_step=68))

    assert recorded_steps == [68, 68]


def test_train_bootstraps_fresh_run_with_step_minus_one(monkeypatch):
    served_weight_steps: list[int] = []
    waited_weight_steps: list[int] = []
    replay_steps: list[int] = []
    published_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            published_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    class _FakeReplayLoader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTransferServer:
        def serve_weights(self, weight_step: int, model: object) -> None:
            served_weight_steps.append(weight_step)

        def cleanup(self) -> None:
            return None

    class _FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del config, optimizer, loss_fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def initial_state(self, key, *, model):
            del key
            return SimpleNamespace(step=0, model=model)

        def train(self, state, data_loader) -> None:
            del state, data_loader

    monkeypatch.setattr("marin.rl.train_worker.Trainer", _FakeTrainer)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        run_id="resume-test",
        trainer=SimpleNamespace(
            checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)),
            num_train_steps=10,
            seed=0,
        ),
        optimizer=SimpleNamespace(build=lambda num_steps: object()),
        weight_transfer=SimpleNamespace(debug_weight_transfer=False, sync_interval_steps=1),
    )
    worker.loss_module = SimpleNamespace(create_loss_fn=lambda reference_model, _: lambda model, batch, key: 0.0)
    worker.reference_model = object()
    worker.initial_model = object()
    worker.replay_buffer = SimpleNamespace(set_current_step=replay_steps.append)
    worker.replay_loader = _FakeReplayLoader()
    worker.transfer_server = _FakeTransferServer()
    worker.data_loader = object()
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())
    worker._drop_bootstrap_model_references = lambda: None
    worker._configure_training_hooks = lambda trainer: None
    worker._wait_for_initial_rollouts = lambda *, weight_step: waited_weight_steps.append(weight_step) or True
    worker.stop = lambda: None

    worker.train()

    assert replay_steps == [-1]
    assert published_steps == []
    assert served_weight_steps == [-1]
    assert waited_weight_steps == [-1]


def test_train_reuses_recovered_step_on_resume(monkeypatch):
    served_weight_steps: list[int] = []
    waited_weight_steps: list[int] = []
    replay_steps: list[int] = []
    published_steps: list[int] = []

    class _FakeRemoteMethod:
        def remote(self, step: int) -> None:
            published_steps.append(step)

    class _FakeRunState:
        update_train_step = _FakeRemoteMethod()

    class _FakeReplayLoader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTransferServer:
        def serve_weights(self, weight_step: int, model: object) -> None:
            served_weight_steps.append(weight_step)

        def cleanup(self) -> None:
            return None

    class _FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del config, optimizer, loss_fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def initial_state(self, key, *, model):
            del key
            return SimpleNamespace(step=68, model=model)

        def train(self, state, data_loader) -> None:
            del state, data_loader

    monkeypatch.setattr("marin.rl.train_worker.Trainer", _FakeTrainer)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        run_id="resume-test",
        trainer=SimpleNamespace(
            checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)),
            num_train_steps=100,
            seed=0,
        ),
        optimizer=SimpleNamespace(build=lambda num_steps: object()),
        weight_transfer=SimpleNamespace(debug_weight_transfer=False, sync_interval_steps=1),
    )
    worker.loss_module = SimpleNamespace(create_loss_fn=lambda reference_model, _: lambda model, batch, key: 0.0)
    worker.reference_model = object()
    worker.initial_model = object()
    worker.replay_buffer = SimpleNamespace(set_current_step=replay_steps.append)
    worker.replay_loader = _FakeReplayLoader()
    worker.transfer_server = _FakeTransferServer()
    worker.data_loader = object()
    worker._runtime = SimpleNamespace(run_state=_FakeRunState())
    worker._drop_bootstrap_model_references = lambda: None
    worker._configure_training_hooks = lambda trainer: None
    worker._wait_for_initial_rollouts = lambda *, weight_step: waited_weight_steps.append(weight_step) or True
    worker.stop = lambda: None

    worker.train()

    assert replay_steps == [68]
    assert published_steps == [68]
    assert served_weight_steps == [68]
    assert waited_weight_steps == [68]


def test_resume_safe_weight_transfer_metrics_counts_bootstrap_and_sync_hooks():
    assert _resume_safe_weight_transfer_metrics(step=0, sync_interval_steps=1) == {
        "total_transfers": 2,
        "successful_transfers": 2,
    }
    assert _resume_safe_weight_transfer_metrics(step=6, sync_interval_steps=3) == {
        "total_transfers": 4,
        "successful_transfers": 4,
    }


def test_training_step_timing_metrics_keep_step_duration_separate_from_batch_prep():
    metrics = _training_step_timing_metrics(
        step_duration=17.28,
        batch_prep_timing=BatchPrepTiming(fetch_time=40.0, batch_time=1.0, shard_time=0.26),
    )

    assert metrics == {
        "throughput/train_step_duration_seconds": 17.28,
        "throughput/forward_backward_duration_seconds": 17.28,
        "throughput/rollout_wait_duration_seconds": 40.0,
        "throughput/batch_create_duration_seconds": 1.0,
        "throughput/batch_shard_duration_seconds": 0.26,
        "throughput/batch_prep_duration_seconds": 41.26,
        "throughput/iteration_duration_seconds": 58.54,
    }


def test_weight_transfer_hook_logs_global_and_attempt_metrics(monkeypatch):
    logged_metrics: list[tuple[dict[str, float | int], int]] = []
    served_weights: list[tuple[int, object]] = []

    class _FakeTransferServer:
        def serve_weights(self, weight_id: int, model: object) -> None:
            served_weights.append((weight_id, model))

        def get_metrics(self) -> WeightTransferServerMetrics:
            return WeightTransferServerMetrics(total_transfers=8, successful_transfers=8, failed_transfers=0)

    class _FakeTracker:
        def log(self, metrics: dict[str, float | int], *, step: int) -> None:
            logged_metrics.append((metrics, step))

    monkeypatch.setattr("marin.rl.train_worker.time.time", lambda: 100.0)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(weight_transfer=SimpleNamespace(sync_interval_steps=1))
    worker.transfer_server = _FakeTransferServer()

    trainer = SimpleNamespace(tracker=_FakeTracker())
    model = object()
    info = SimpleNamespace(step=67, state=SimpleNamespace(model=model), loss=1.23)

    worker.weight_transfer_hook(trainer, info)

    assert served_weights == [(67, model)]
    assert len(logged_metrics) == 1
    metrics, step = logged_metrics[0]
    assert step == 67
    assert metrics["weight_transfer/attempt_total_transfers"] == 8
    assert metrics["weight_transfer/attempt_successful_transfers"] == 8
    assert metrics["weight_transfer/attempt_failed_transfers"] == 0
    assert metrics["weight_transfer/total_transfers"] == 69
    assert metrics["weight_transfer/successful_transfers"] == 69
    assert metrics["weight_transfer/serve_time_seconds"] == 0.0


def test_checkpoint_debug_snapshot_includes_replay_buffer_and_transfer_state():
    worker = TrainWorker.__new__(TrainWorker)
    worker.replay_buffer = SimpleNamespace(get_stats=lambda: {"total_size": 128}, _current_step=251)
    worker.transfer_server = SimpleNamespace(
        get_debug_snapshot=lambda: {"latest_store": {"stored_arrow_bytes": 1024}, "latest_transfer_metrics": {"a": 1}}
    )

    assert worker._checkpoint_debug_snapshot() == {
        "replay_buffer": {"total_size": 128, "current_step": 251},
        "weight_transfer": {
            "latest_store": {"stored_arrow_bytes": 1024},
            "latest_transfer_metrics": {"a": 1},
        },
    }
