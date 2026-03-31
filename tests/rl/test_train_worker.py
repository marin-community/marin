# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from marin.rl.rl_losses import RLOOLoss
from marin.rl.train_worker import (
    BatchPrepTiming,
    TrainWorker,
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
