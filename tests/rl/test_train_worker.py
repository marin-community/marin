# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from types import SimpleNamespace

import pytest
from levanter.lora import LoraConfig
from marin.rl.lora_manifest import build_rl_run_manifest
from marin.rl.rl_losses import RLOOLoss
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
            return SimpleNamespace(eval_model=object())

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
        lora=None,
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
            return SimpleNamespace(eval_model=object())

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
        lora=None,
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


def test_train_uses_trainable_filter_for_lora_runs(monkeypatch):
    initial_state_calls: list[dict[str, object]] = []
    manifest_writes: list[bool] = []
    logged_summaries: list[dict[str, float | int]] = []
    served_weights: list[tuple[int, object]] = []
    merged_model = object()
    trainable_model = object()

    class _FakeReplayLoader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTransferServer:
        def serve_weights(self, weight_step: int, model: object) -> None:
            served_weights.append((weight_step, model))

        def cleanup(self) -> None:
            return None

    class _FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del config, optimizer, loss_fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def initial_state(self, key, *, model, is_trainable):
            del key
            initial_state_calls.append({"model": model, "is_trainable": is_trainable})
            return SimpleNamespace(step=0, model=model, trainable_model=trainable_model)

        def train(self, state, data_loader) -> None:
            del data_loader
            return SimpleNamespace(eval_model=state.model)

    monkeypatch.setattr("marin.rl.train_worker.Trainer", _FakeTrainer)
    monkeypatch.setattr("marin.rl.train_worker.merge_lora_modules", lambda model: merged_model)
    monkeypatch.setattr("marin.rl.train_worker.parameter_count", lambda model: 3 if model is trainable_model else 12)
    monkeypatch.setattr("marin.rl.train_worker.levanter.tracker.log_summary", logged_summaries.append)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        run_id="lora-test",
        trainer=SimpleNamespace(
            checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)),
            num_train_steps=10,
            seed=0,
        ),
        optimizer=SimpleNamespace(build=lambda num_steps: object()),
        weight_transfer=SimpleNamespace(debug_weight_transfer=False, sync_interval_steps=1),
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj"]),
        rollout_policy_format="merged",
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model"},
        inference_type="vllm",
        run_manifest_path="/tmp/rl_run_manifest.json",
    )
    worker.loss_module = SimpleNamespace(create_loss_fn=lambda reference_model, _: lambda model, batch, key: 0.0)
    worker.reference_model = object()
    worker.initial_model = object()
    worker.trainable_model_filter = "lora-filter"
    worker.replay_buffer = SimpleNamespace(set_current_step=lambda step: None)
    worker.replay_loader = _FakeReplayLoader()
    worker.transfer_server = _FakeTransferServer()
    worker.data_loader = object()
    worker._runtime = SimpleNamespace(
        run_state=SimpleNamespace(update_train_step=SimpleNamespace(remote=lambda step: None))
    )
    worker._drop_bootstrap_model_references = lambda: None
    worker._configure_training_hooks = lambda trainer: None
    worker._wait_for_initial_rollouts = lambda *, weight_step: True
    worker._validate_run_manifest_for_resume = lambda trainer: None
    worker._write_run_manifest = lambda: manifest_writes.append(True)
    worker.stop = lambda: None

    worker.train()

    assert manifest_writes == [True]
    assert initial_state_calls == [{"model": worker.initial_model, "is_trainable": "lora-filter"}]
    assert served_weights == [(-1, merged_model)]
    assert logged_summaries == [
        {"rollout_policy_format": "merged", "reference_mode": "base"},
        {"parameter_count": 12, "trainable_parameter_count": 3, "fraction_trainable": 0.25},
    ]


def test_export_lora_artifacts_uses_expected_paths_and_helpers(monkeypatch):
    exported_calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    logged_summaries: list[dict[str, str]] = []

    def _record_adapter_export(*args, **kwargs):
        exported_calls.append(("adapter", args, kwargs))

    def _record_merged_export(*args, **kwargs):
        exported_calls.append(("merged", args, kwargs))

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj"]),
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        adapter_artifacts_path="gs://marin-us-central1/rl/adapter_artifacts",
        merged_hf_export_path="gs://marin-us-central1/rl/exports/merged",
    )
    worker.tokenizer = object()
    worker._hf_export_converter = lambda: "converter"

    monkeypatch.setattr("marin.rl.train_worker.save_peft_pretrained", _record_adapter_export)
    monkeypatch.setattr("marin.rl.train_worker.save_merged_hf_model", _record_merged_export)
    monkeypatch.setattr("marin.rl.train_worker.levanter.tracker.log_summary", logged_summaries.append)

    model = object()
    worker._export_lora_artifacts(model)

    assert exported_calls == [
        (
            "adapter",
            (
                model,
                worker.config.lora,
                "hf://meta-llama/Llama-3.1-8B",
                "gs://marin-us-central1/rl/adapter_artifacts/final",
            ),
            {"tokenizer": worker.tokenizer},
        ),
        (
            "merged",
            (
                model,
                "converter",
                "gs://marin-us-central1/rl/exports/merged/final",
            ),
            {},
        ),
    ]
    assert logged_summaries == [
        {
            "adapter_artifacts_path": "gs://marin-us-central1/rl/adapter_artifacts/final",
            "merged_hf_export_path": "gs://marin-us-central1/rl/exports/merged/final",
        }
    ]


def test_export_lora_artifacts_requires_hf_compatible_base_for_adapter_export(monkeypatch):
    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj"]),
        initial_checkpoint="/tmp/checkpoints/run/step-10",
        adapter_artifacts_path="gs://marin-us-central1/rl/adapter_artifacts",
        merged_hf_export_path=None,
    )

    monkeypatch.setattr("marin.rl.train_worker.is_hf_checkpoint", lambda path: False)

    with pytest.raises(ValueError, match="HF-compatible base checkpoint"):
        worker._export_lora_artifacts(object())


def test_train_exports_lora_artifacts_after_training(monkeypatch):
    exported_models: list[object] = []
    eval_model = object()

    class _FakeReplayLoader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _FakeTransferServer:
        def serve_weights(self, weight_step: int, model: object) -> None:
            del weight_step, model

        def cleanup(self) -> None:
            return None

    class _FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del config, optimizer, loss_fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def initial_state(self, key, *, model, is_trainable):
            del key, is_trainable
            return SimpleNamespace(step=0, model=model, trainable_model=object())

        def train(self, state, data_loader):
            del state, data_loader
            return SimpleNamespace(eval_model=eval_model)

    monkeypatch.setattr("marin.rl.train_worker.Trainer", _FakeTrainer)
    monkeypatch.setattr("marin.rl.train_worker.parameter_count", lambda model: 4 if model is eval_model else 16)
    monkeypatch.setattr("marin.rl.train_worker.levanter.tracker.log_summary", lambda summary: None)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        run_id="lora-export-test",
        trainer=SimpleNamespace(
            checkpointer=SimpleNamespace(debug=SimpleNamespace(enabled=False)),
            num_train_steps=10,
            seed=0,
        ),
        optimizer=SimpleNamespace(build=lambda num_steps: object()),
        weight_transfer=SimpleNamespace(debug_weight_transfer=False, sync_interval_steps=1),
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj"]),
        rollout_policy_format="merged",
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model"},
        inference_type="vllm",
        run_manifest_path="/tmp/rl_run_manifest.json",
    )
    worker.loss_module = SimpleNamespace(create_loss_fn=lambda reference_model, _: lambda model, batch, key: 0.0)
    worker.reference_model = object()
    worker.initial_model = object()
    worker.trainable_model_filter = "lora-filter"
    worker.replay_buffer = SimpleNamespace(set_current_step=lambda step: None)
    worker.replay_loader = _FakeReplayLoader()
    worker.transfer_server = _FakeTransferServer()
    worker.data_loader = object()
    worker._runtime = SimpleNamespace(
        run_state=SimpleNamespace(update_train_step=SimpleNamespace(remote=lambda step: None))
    )
    worker._drop_bootstrap_model_references = lambda: None
    worker._configure_training_hooks = lambda trainer: None
    worker._wait_for_initial_rollouts = lambda *, weight_step: True
    worker._validate_run_manifest_for_resume = lambda trainer: None
    worker._write_run_manifest = lambda: None
    worker._export_lora_artifacts = exported_models.append
    worker.stop = lambda: None

    worker.train()

    assert exported_models == [eval_model]


def test_write_run_manifest_persists_expected_lora_metadata(tmp_path):
    manifest_path = tmp_path / "artifacts" / "rl_run_manifest.json"

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model", "hidden_dim": 128},
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj", "v_proj"]),
        rollout_policy_format="merged",
        inference_type="vllm",
        run_manifest_path=str(manifest_path),
    )

    worker._write_run_manifest()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["manifest_version"] == 1
    assert manifest["initial_checkpoint"] == "hf://meta-llama/Llama-3.1-8B"
    assert manifest["rollout_policy_format"] == "merged"
    assert manifest["reference_mode"] == "base"
    assert manifest["inference_type"] == "vllm"
    assert manifest["lora_config"]["r"] == 8
    assert manifest["lora_config"]["alpha"] == 16.0
    assert manifest["lora_config"]["target_modules"] == ["q_proj", "v_proj"]
    assert manifest["lora_config_fingerprint"] is not None


def test_validate_run_manifest_for_resume_accepts_matching_manifest(monkeypatch):
    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model", "hidden_dim": 128},
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj", "v_proj"]),
        rollout_policy_format="merged",
        inference_type="vllm",
        run_manifest_path="/tmp/rl_run_manifest.json",
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(load_checkpoint=None, initialize_from=None),
        checkpoint_path="/tmp/checkpoints/run",
    )
    manifest = build_rl_run_manifest(
        initial_checkpoint=worker.config.initial_checkpoint,
        model_config=worker.config.model,
        lora_config=worker.config.lora,
        rollout_policy_format=worker.config.rollout_policy_format,
        inference_type=worker.config.inference_type,
    )

    monkeypatch.setattr("marin.rl.train_worker.discover_latest_checkpoint", lambda path: f"{path}/step_10")
    monkeypatch.setattr("marin.rl.train_worker.read_rl_run_manifest", lambda path: manifest)

    worker._validate_run_manifest_for_resume(trainer)


@pytest.mark.parametrize(
    ("manifest_overrides", "field_name"),
    [
        ({"initial_checkpoint": "hf://meta-llama/Llama-3.1-70B"}, "initial_checkpoint"),
        ({"model_config_fingerprint": "deadbeefdeadbeef"}, "model_config_fingerprint"),
        ({"lora_config_fingerprint": "cafebabecafebabe"}, "lora_config_fingerprint"),
        ({"inference_type": "levanter"}, "inference_type"),
        ({"rollout_policy_format": "adapter"}, "rollout_policy_format"),
        ({"reference_mode": "adapter"}, "reference_mode"),
    ],
)
def test_validate_run_manifest_for_resume_rejects_mismatches(monkeypatch, manifest_overrides, field_name):
    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model", "hidden_dim": 128},
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj", "v_proj"]),
        rollout_policy_format="merged",
        inference_type="vllm",
        run_manifest_path="/tmp/rl_run_manifest.json",
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(load_checkpoint=None, initialize_from=None),
        checkpoint_path="/tmp/checkpoints/run",
    )
    expected_manifest = build_rl_run_manifest(
        initial_checkpoint=worker.config.initial_checkpoint,
        model_config=worker.config.model,
        lora_config=worker.config.lora,
        rollout_policy_format=worker.config.rollout_policy_format,
        inference_type=worker.config.inference_type,
    )
    manifest = dataclasses.replace(expected_manifest, **manifest_overrides)

    monkeypatch.setattr("marin.rl.train_worker.discover_latest_checkpoint", lambda path: f"{path}/step_10")
    monkeypatch.setattr("marin.rl.train_worker.read_rl_run_manifest", lambda path: manifest)

    with pytest.raises(ValueError, match=field_name):
        worker._validate_run_manifest_for_resume(trainer)


def test_validate_run_manifest_for_resume_requires_manifest(monkeypatch):
    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        initial_checkpoint="hf://meta-llama/Llama-3.1-8B",
        model={"name": "toy-model", "hidden_dim": 128},
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj", "v_proj"]),
        rollout_policy_format="merged",
        inference_type="vllm",
        run_manifest_path="/tmp/rl_run_manifest.json",
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(load_checkpoint=None, initialize_from=None),
        checkpoint_path="/tmp/checkpoints/run",
    )

    monkeypatch.setattr("marin.rl.train_worker.discover_latest_checkpoint", lambda path: f"{path}/step_10")
    monkeypatch.setattr(
        "marin.rl.train_worker.read_rl_run_manifest",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )

    with pytest.raises(ValueError, match="requires run manifest"):
        worker._validate_run_manifest_for_resume(trainer)


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


def test_weight_transfer_hook_merges_lora_weights_before_serving(monkeypatch):
    served_weights: list[tuple[int, object]] = []
    merged_model = object()

    class _FakeTransferServer:
        def serve_weights(self, weight_id: int, model: object) -> None:
            served_weights.append((weight_id, model))

        def get_metrics(self) -> WeightTransferServerMetrics:
            return WeightTransferServerMetrics(total_transfers=8, successful_transfers=8, failed_transfers=0)

    class _FakeTracker:
        def log(self, metrics: dict[str, float | int], *, step: int) -> None:
            del metrics, step

    monkeypatch.setattr("marin.rl.train_worker.merge_lora_modules", lambda model: merged_model)
    monkeypatch.setattr("marin.rl.train_worker.time.time", lambda: 100.0)

    worker = TrainWorker.__new__(TrainWorker)
    worker.config = SimpleNamespace(
        lora=LoraConfig(r=8, alpha=16.0, target_modules=["q_proj"]),
        rollout_policy_format="merged",
        weight_transfer=SimpleNamespace(sync_interval_steps=1),
    )
    worker.transfer_server = _FakeTransferServer()

    trainer = SimpleNamespace(tracker=_FakeTracker())
    info = SimpleNamespace(step=67, state=SimpleNamespace(model=object()), loss=1.23)

    worker.weight_transfer_hook(trainer, info)

    assert served_weights == [(67, merged_model)]


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
