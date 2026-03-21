# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    AlternatingRunState,
    MaterializedBatchesManifest,
    PolicyManifest,
    RunStatus,
    read_policy_manifest,
)
from marin.rl.alternating.training_phase import export_policy_only, run_training_phase


@dataclass(frozen=True)
class DummyCheckpointer:
    base_path: str = "unused"
    append_run_id_to_base_path: bool = True

    def expanded_path(self, run_id: str) -> str:
        if self.append_run_id_to_base_path:
            return f"{self.base_path}/{run_id}"
        return self.base_path


@dataclass(frozen=True)
class DummyTrainer:
    train_batch_size: int = 4
    checkpointer: DummyCheckpointer = DummyCheckpointer()
    seed: int = 0
    id: str | None = None
    num_train_steps: int = 0
    load_checkpoint: bool | None = None
    load_checkpoint_path: str | None = None
    device_mesh: object | None = None
    compute_axis_mapping: dict[str, str] | None = None
    parameter_axis_mapping: dict[str, str] | None = None


@dataclass(frozen=True)
class DummyInference:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"


class RecordingOptimizer:
    def __init__(self):
        self.calls: list[int] = []

    def build(self, num_train_steps: int) -> object:
        self.calls.append(num_train_steps)
        return object()


class DummyLoss:
    kl_coef: float = 0.0

    def create_loss_fn(self, reference_model, train_model):
        del reference_model, train_model

        def _loss_fn(model, batch, key):
            del model, batch, key
            return 0.0

        return _loss_fn


def _make_config(tmp_path, optimizer: RecordingOptimizer) -> AlternatingRLConfig:
    return AlternatingRLConfig(
        run_id="alt-train-phase-test",
        shared_root=tmp_path.as_posix(),
        image_digest="image@sha256:test",
        seed=23,
        cluster=AlternatingClusterConfig(
            tpu_name="test-pod",
            tpu_type="v5p-8",
            zone="us-east5-a",
            num_hosts=1,
            local_tensor_parallel_size=1,
        ),
        quotas=AlternatingPhaseQuotaConfig(
            steps_per_phase=2,
            num_train_steps=10,
            groups_per_training_step=1,
            eval_examples_per_lesson=4,
        ),
        trainer=DummyTrainer(
            checkpointer=DummyCheckpointer(),
            compute_axis_mapping={},
            parameter_axis_mapping={},
        ),
        model=object(),
        optimizer=optimizer,
        loss=DummyLoss(),
        curriculum=SimpleNamespace(max_seq_len=128),
        inference=DummyInference(),
        replay_buffer=SimpleNamespace(alpha=1.0, max_samples=1, filter_out_groups_with_no_variance=False),
        tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        initial_checkpoint="meta-llama/Llama-3.1-8B-Instruct",
    )


def test_training_phase_uses_full_run_optimizer_schedule_and_run_specific_checkpoint_root(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=3,
        policy_version=3,
        source_global_step=5,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(3),
        current_levanter_checkpoint_path=f"{tmp_path.as_posix()}/resume/step-5",
        current_sampling_manifest=paths.sampling_manifest_path(3),
        current_materialized_manifest=paths.materialized_manifest_path(3),
        last_completed_phase=2,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=3,
        policy_version=3,
        input_rollout_paths=[],
        num_rollout_groups=2,
        num_individual_rollouts=8,
        num_training_batches=2,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[
            paths.materialized_batch_path(3, 0),
            paths.materialized_batch_path(3, 1),
        ],
    )

    captured_trainer_config = {}
    discovered_paths: list[str] = []
    exported_checkpoints: list[str] = []

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.levanter.initialize",
        lambda trainer_config: None,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: object(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._build_reference_model",
        lambda config, trainer_config, tokenizer, resume_checkpoint_path: object(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.barrier_sync",
        lambda: None,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.jax.process_index",
        lambda: 0,
    )

    def _discover_latest_checkpoint(path: str) -> str:
        discovered_paths.append(path)
        return f"{path}/step-7"

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.discover_latest_checkpoint",
        _discover_latest_checkpoint,
    )

    def _export_checkpoint(convert_config) -> None:
        exported_checkpoints.append(convert_config.checkpoint_path)

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.export_lm_to_hf.main",
        _export_checkpoint,
    )

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            captured_trainer_config["config"] = config
            captured_trainer_config["optimizer"] = optimizer
            captured_trainer_config["loss_fn"] = loss_fn

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def initial_state(self, training_key, model):
            del training_key, model
            return "initial-state"

        def train(self, initial_state, loader):
            del initial_state, loader

    monkeypatch.setattr("marin.rl.alternating.training_phase.Trainer", FakeTrainer)

    policy_manifest_path = run_training_phase(config, paths, state, manifest)

    trainer_config = captured_trainer_config["config"]
    assert trainer_config.num_train_steps == 7
    assert trainer_config.load_checkpoint_path == state.current_levanter_checkpoint_path
    assert optimizer.calls == [config.quotas.num_train_steps]
    expected_checkpoint_root = f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train"
    assert discovered_paths == [expected_checkpoint_root]
    assert exported_checkpoints == [f"{expected_checkpoint_root}/step-7"]

    policy_manifest = read_policy_manifest(policy_manifest_path)
    assert isinstance(policy_manifest, PolicyManifest)
    assert policy_manifest.policy_version == 4
    assert policy_manifest.phase_id == state.phase_id
    assert policy_manifest.source_global_step == 7
    assert policy_manifest.levanter_checkpoint_path == f"{expected_checkpoint_root}/step-7"


def test_resumed_zero_kl_training_skips_initial_checkpoint_load(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    trainer_config = config.trainer
    tokenizer = type("Tokenizer", (), {"__len__": lambda self: 128})()
    captured_checkpoints: list[str | None] = []

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.load_model_from_checkpoint",
        lambda **kwargs: captured_checkpoints.append(kwargs["checkpoint"]) or object(),
    )

    from marin.rl.alternating.training_phase import _build_reference_model

    _build_reference_model(
        config,
        trainer_config,
        tokenizer,
        resume_checkpoint_path="/tmp/resume/step-5",
    )

    assert captured_checkpoints == [None]


def test_export_only_recovery_writes_policy_manifest_from_latest_checkpoint(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=2,
        policy_version=2,
        source_global_step=4,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(2),
        current_levanter_checkpoint_path=f"{tmp_path.as_posix()}/resume/step-4",
        current_sampling_manifest=paths.sampling_manifest_path(2),
        current_materialized_manifest=paths.materialized_manifest_path(2),
        last_completed_phase=1,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=2,
        policy_version=2,
        input_rollout_paths=[],
        num_rollout_groups=2,
        num_individual_rollouts=8,
        num_training_batches=2,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[
            paths.materialized_batch_path(2, 0),
            paths.materialized_batch_path(2, 1),
        ],
    )
    exported_checkpoints: list[str] = []

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.barrier_sync",
        lambda: None,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.jax.process_index",
        lambda: 0,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.discover_latest_checkpoint",
        lambda _path: f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train/step-6",
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._checkpoint_step",
        lambda _path: 6,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.export_lm_to_hf.main",
        lambda convert_config: exported_checkpoints.append(convert_config.checkpoint_path),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.levanter.tracker.log",
        lambda metrics, step: None,
    )

    policy_manifest_path = export_policy_only(config, paths, state, manifest)

    assert exported_checkpoints == [f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train/step-6"]
    policy_manifest = read_policy_manifest(policy_manifest_path)
    assert policy_manifest.policy_version == 3
    assert policy_manifest.phase_id == state.phase_id
    assert policy_manifest.source_global_step == 6
