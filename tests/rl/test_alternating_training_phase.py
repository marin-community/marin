# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

from levanter.tracker.wandb import WandbConfig

from marin.rl.alternating import (
    AlternatingClusterConfig,
    AlternatingPhaseQuotaConfig,
    AlternatingRLConfig,
    AlternatingRunPaths,
    AlternatingRunState,
    MaterializedBatchesManifest,
    PolicyBootstrapFormat,
    PolicyManifest,
    RunStatus,
    read_policy_manifest,
    read_run_state,
    write_run_state,
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
    tracker: object | None = None
    num_train_steps: int = 0
    load_checkpoint: bool | None = None
    load_checkpoint_path: str | None = None
    device_mesh: object | None = None
    compute_axis_mapping: dict[str, str] | None = None
    parameter_axis_mapping: dict[str, str] | None = None


@dataclass(frozen=True)
class DummyInference:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"


class FakeTokenizer:
    def __len__(self) -> int:
        return 128

    def decode(self, tokens, *, skip_special_tokens=False):
        del skip_special_tokens
        return ",".join(str(token) for token in tokens)


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
        model=SimpleNamespace(flops_per_token=lambda vocab_size, tokens_per_example: None),
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

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.levanter.initialize",
        lambda trainer_config: None,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
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
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._checkpoint_metadata_exists",
        lambda _path: False,
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._checkpoint_step",
        lambda _path: 7,
    )

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            captured_trainer_config["config"] = config
            captured_trainer_config["optimizer"] = optimizer
            captured_trainer_config["loss_fn"] = loss_fn
            self.config = config

        def add_hook(self, fn, *, every=1):
            del fn, every

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

    policy_manifest = read_policy_manifest(policy_manifest_path)
    assert isinstance(policy_manifest, PolicyManifest)
    assert policy_manifest.policy_version == 4
    assert policy_manifest.phase_id == state.phase_id
    assert policy_manifest.source_global_step == 7
    assert policy_manifest.policy_path == f"{expected_checkpoint_root}/step-7"
    assert policy_manifest.levanter_checkpoint_path == f"{expected_checkpoint_root}/step-7"
    assert policy_manifest.bootstrap_format == PolicyBootstrapFormat.LEVANTER_CHECKPOINT


def test_training_phase_waits_for_checkpoint_visibility_before_export(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=1,
        policy_version=1,
        source_global_step=0,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=1,
        policy_version=1,
        input_rollout_paths=[],
        num_rollout_groups=1,
        num_individual_rollouts=4,
        num_training_batches=1,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[paths.materialized_batch_path(1, 0)],
    )
    trainer_checkpoint_root = f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train"
    metadata_visibility = iter([False, False, True])

    monkeypatch.setattr("marin.rl.alternating.training_phase.levanter.initialize", lambda trainer_config: None)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._build_reference_model",
        lambda config, trainer_config, tokenizer, resume_checkpoint_path: object(),
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase.barrier_sync", lambda: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase.jax.process_index", lambda: 0)
    monkeypatch.setattr("marin.rl.alternating.training_phase.discover_latest_checkpoint", lambda _path: None)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._checkpoint_metadata_exists",
        lambda _path: next(metadata_visibility),
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase.time.sleep", lambda _seconds: None)

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            self.config = config
            del optimizer, loss_fn

        def add_hook(self, fn, *, every=1):
            del fn, every

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

    policy_manifest = read_policy_manifest(policy_manifest_path)
    assert policy_manifest.policy_path == f"{trainer_checkpoint_root}/step-1"
    assert policy_manifest.levanter_checkpoint_path == f"{trainer_checkpoint_root}/step-1"
    assert policy_manifest.bootstrap_format == PolicyBootstrapFormat.LEVANTER_CHECKPOINT


def test_training_phase_sets_distinct_alternating_wandb_run_metadata(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    config = replace(
        config,
        trainer=replace(
            config.trainer,
            tracker=WandbConfig(
                project="alternate_rl",
                name="base-run",
                tags=["rl", "math"],
            ),
        ),
    )
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=1,
        policy_version=1,
        source_global_step=0,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=1,
        policy_version=1,
        input_rollout_paths=[],
        num_rollout_groups=1,
        num_individual_rollouts=4,
        num_training_batches=1,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[paths.materialized_batch_path(1, 0)],
    )
    captured_trainer_config = {}

    monkeypatch.setattr("marin.rl.alternating.training_phase.levanter.initialize", lambda trainer_config: None)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._build_reference_model",
        lambda config, trainer_config, tokenizer, resume_checkpoint_path: object(),
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase.barrier_sync", lambda: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase.jax.process_index", lambda: 0)
    monkeypatch.setattr("marin.rl.alternating.training_phase.discover_latest_checkpoint", lambda _path: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase._checkpoint_metadata_exists", lambda _path: True)

    class FakeTracker:
        def log(self, metrics, *, step, commit=None):
            del metrics, step, commit

        def finish(self):
            return None

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del optimizer, loss_fn
            captured_trainer_config["config"] = config
            self.config = config
            self.tracker = FakeTracker()

        def add_hook(self, fn, *, every=1):
            del fn, every

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

    run_training_phase(config, paths, state, manifest)

    trainer_tracker = captured_trainer_config["config"].tracker
    assert isinstance(trainer_tracker, WandbConfig)
    assert trainer_tracker.name == "alt-train-phase-test-alternating-train"
    assert trainer_tracker.group == "alt-train-phase-test"
    assert trainer_tracker.tags == ["rl", "math", "alternating", "alternating-train"]
    assert trainer_tracker.save_code is False


def test_training_phase_installs_async_rl_metric_hooks(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=1,
        policy_version=1,
        source_global_step=0,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=1,
        policy_version=1,
        input_rollout_paths=[],
        num_rollout_groups=1,
        num_individual_rollouts=4,
        num_training_batches=1,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[paths.materialized_batch_path(1, 0)],
    )
    hook_calls: list[dict[str, object]] = []

    monkeypatch.setattr("marin.rl.alternating.training_phase.levanter.initialize", lambda trainer_config: None)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._build_reference_model",
        lambda config, trainer_config, tokenizer, resume_checkpoint_path: object(),
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase.barrier_sync", lambda: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase.jax.process_index", lambda: 0)
    monkeypatch.setattr("marin.rl.alternating.training_phase.discover_latest_checkpoint", lambda _path: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase._checkpoint_metadata_exists", lambda _path: True)

    def _record_hooks(
        trainer,
        *,
        tokenizer,
        tokens_per_example,
        flops_per_example,
        batch_schedule,
        batch_prep_time,
        sample_previews,
    ):
        hook_calls.append(
            {
                "trainer": trainer,
                "tokenizer": tokenizer,
                "tokens_per_example": tokens_per_example,
                "flops_per_example": flops_per_example,
                "batch_schedule": batch_schedule,
                "batch_prep_time": batch_prep_time,
                "sample_previews": sample_previews,
            }
        )

    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.configure_rl_training_metric_hooks",
        _record_hooks,
    )

    class FakeTracker:
        def log(self, metrics, *, step, commit=None):
            del metrics, step, commit

        def finish(self):
            return None

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            del optimizer, loss_fn
            self.config = config
            self.tracker = FakeTracker()

        def add_hook(self, fn, *, every=1):
            del fn, every

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

    run_training_phase(config, paths, state, manifest)

    assert len(hook_calls) == 1
    assert hook_calls[0]["tokens_per_example"] == config.curriculum.max_seq_len
    assert hook_calls[0]["batch_schedule"] == config.trainer.train_batch_size


def test_training_phase_finishes_tracker_after_logging_phase_metrics(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.TRAINING,
        phase_id=1,
        policy_version=1,
        source_global_step=0,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )
    manifest = MaterializedBatchesManifest(
        phase_id=1,
        policy_version=1,
        input_rollout_paths=[],
        num_rollout_groups=1,
        num_individual_rollouts=4,
        num_training_batches=1,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[paths.materialized_batch_path(1, 0)],
    )
    tracker_events: list[tuple[str, int | None, dict[str, float] | None]] = []

    monkeypatch.setattr("marin.rl.alternating.training_phase.levanter.initialize", lambda trainer_config: None)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.AutoTokenizer.from_pretrained",
        lambda _name: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase._build_reference_model",
        lambda config, trainer_config, tokenizer, resume_checkpoint_path: object(),
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase.barrier_sync", lambda: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase.jax.process_index", lambda: 0)
    monkeypatch.setattr("marin.rl.alternating.training_phase.discover_latest_checkpoint", lambda _path: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase._checkpoint_metadata_exists", lambda _path: True)

    class FakeTracker:
        def log(self, metrics, *, step, commit=None):
            del commit
            tracker_events.append(("log", step, metrics))

        def finish(self):
            tracker_events.append(("finish", None, None))

    class FakeTrainer:
        def __init__(self, *, config, optimizer, loss_fn):
            self.config = config
            del optimizer, loss_fn
            self.tracker = FakeTracker()

        def add_hook(self, fn, *, every=1):
            del fn, every

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

    run_training_phase(config, paths, state, manifest)

    assert tracker_events[-1] == ("finish", None, None)
    assert tracker_events[0][0] == "log"
    assert tracker_events[0][1] == 1
    assert tracker_events[0][2]["alternating/phase_id"] == 1
    assert tracker_events[0][2]["alternating/source_global_step"] == 1


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
        "marin.rl.alternating.training_phase.levanter.tracker.log",
        lambda metrics, step: None,
    )

    policy_manifest_path = export_policy_only(config, paths, state, manifest)

    policy_manifest = read_policy_manifest(policy_manifest_path)
    assert policy_manifest.policy_version == 3
    assert policy_manifest.phase_id == state.phase_id
    assert policy_manifest.source_global_step == 6
    assert policy_manifest.policy_path == f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train/step-6"
    assert policy_manifest.bootstrap_format == PolicyBootstrapFormat.LEVANTER_CHECKPOINT


def test_export_policy_only_recovers_failed_run_state(monkeypatch, tmp_path):
    optimizer = RecordingOptimizer()
    config = _make_config(tmp_path, optimizer)
    paths = AlternatingRunPaths.from_config(config)
    state = AlternatingRunState(
        run_id=config.run_id,
        status=RunStatus.FAILED,
        phase_id=1,
        policy_version=1,
        source_global_step=2,
        num_hosts=1,
        tpu_name=config.cluster.tpu_name,
        tpu_type=config.cluster.tpu_type,
        zone=config.cluster.zone,
        image_digest=config.image_digest,
        current_policy_manifest_path=paths.policy_manifest_path(1),
        current_levanter_checkpoint_path=None,
        current_sampling_manifest=paths.sampling_manifest_path(1),
        current_materialized_manifest=paths.materialized_manifest_path(1),
        last_completed_phase=0,
    )
    write_run_state(paths.run_state_path, state)
    manifest = MaterializedBatchesManifest(
        phase_id=1,
        policy_version=1,
        input_rollout_paths=[],
        num_rollout_groups=2,
        num_individual_rollouts=8,
        num_training_batches=2,
        global_batch_size=config.global_batch_size,
        max_seq_len=128,
        batch_paths=[
            paths.materialized_batch_path(1, 0),
            paths.materialized_batch_path(1, 1),
        ],
    )
    latest_checkpoint = f"{paths.levanter_checkpoints_root}/{config.run_id}-alternating-train/step-4"

    monkeypatch.setattr("marin.rl.alternating.training_phase.barrier_sync", lambda: None)
    monkeypatch.setattr("marin.rl.alternating.training_phase.jax.process_index", lambda: 0)
    monkeypatch.setattr(
        "marin.rl.alternating.training_phase.discover_latest_checkpoint", lambda _path: latest_checkpoint
    )
    monkeypatch.setattr("marin.rl.alternating.training_phase._checkpoint_step", lambda _path: 4)
    monkeypatch.setattr("marin.rl.alternating.training_phase.levanter.tracker.log", lambda metrics, step: None)

    policy_manifest_path = export_policy_only(config, paths, state, manifest)
    recovered_state = read_run_state(paths.run_state_path)

    assert policy_manifest_path == paths.policy_manifest_path(2)
    assert recovered_state.status == RunStatus.TRAINING
    assert recovered_state.current_levanter_checkpoint_path == latest_checkpoint

    write_run_state(paths.run_state_path, state)

    second_manifest_path = export_policy_only(config, paths, state, manifest)
    second_recovered_state = read_run_state(paths.run_state_path)

    assert second_manifest_path == policy_manifest_path
    assert second_recovered_state.status == RunStatus.TRAINING
    assert second_recovered_state.current_levanter_checkpoint_path == latest_checkpoint
