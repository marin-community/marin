# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest
from fray.types import ResourceConfig
from levanter.checkpoint import CheckpointDebugConfig
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig
from marin.execution.artifact import PathMetadata
from marin.execution.executor import ExecutorStep, output_path_of
from marin.rl.curriculum import CurriculumConfig
from marin.rl.model_utils import is_hf_checkpoint
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    RLStepConfig,
    _build_rl_job_config,
    _run_rl_experiment_step,
    config_class_path,
    executor_main_config_for_rl_experiment,
    executor_step_resources_for_rl_experiment,
    launcher_region_for_rl_experiment,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


@dataclasses.dataclass(frozen=True)
class _EmptyConfig:
    pass


@dataclasses.dataclass(frozen=True)
class _FakeRuntimeLmConfig:
    max_seq_len: int = 0
    tokenizer: str = ""
    attn_backend: object | None = None

    @classmethod
    def from_hf_config(cls, _hf_config):
        return cls()


def _noop(_config: _EmptyConfig) -> None:
    return None


@pytest.fixture(autouse=True)
def _default_launcher_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")


def _test_config(
    *,
    train_resources: ResourceConfig | None = None,
    rollout_resources: ResourceConfig | None = None,
    train_attention_backend: AttentionBackend | None = None,
    inflight_weight_updates: bool = False,
    delete_previous_temporary_checkpoint_after_save: bool = True,
    checkpoint_debug: CheckpointDebugConfig | None = None,
) -> RLExperimentConfig:
    return RLExperimentConfig(
        model_config=ModelConfig(
            name=MODEL_NAME,
            type="llama",
            artifact=MODEL_NAME,
            config_class_path=config_class_path(LlamaConfig),
        ),
        rl_loss=RLOOLoss(
            kl_coef=0.0,
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="test",
        inflight_weight_updates=inflight_weight_updates,
        train_resources=train_resources or ResourceConfig.with_tpu("v5p-8"),
        rollout_resources=rollout_resources or ResourceConfig.with_tpu("v5p-8"),
        train_attention_backend=train_attention_backend,
        delete_previous_temporary_checkpoint_after_save=delete_previous_temporary_checkpoint_after_save,
        checkpoint_debug=checkpoint_debug or CheckpointDebugConfig(),
    )


def _test_curriculum() -> CurriculumConfig:
    return CurriculumConfig(
        lessons={},
        eval_frequency=1,
        micro_eval_frequency=1,
        actor_name="curriculum-test",
        eval_n_examples=1,
        max_seq_len=16,
    )


def test_executor_step_regions_follow_current_launcher_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")

    resources = executor_step_resources_for_rl_experiment(_test_config())

    assert resources.regions == ["us-east5"]


def test_non_v5p_executor_step_regions_follow_current_launcher_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    resources = executor_step_resources_for_rl_experiment(
        _test_config(
            train_resources=ResourceConfig.with_tpu("v6e-4"),
            rollout_resources=ResourceConfig.with_tpu("v6e-4"),
        )
    )

    assert resources.regions == ["europe-west4"]


def test_executor_main_config_uses_current_launcher_region_prefix(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")

    executor_config = executor_main_config_for_rl_experiment(_test_config())

    assert executor_config.prefix == "gs://marin-us-east5"


def test_launcher_region_raises_when_root_region_conflicts_with_requested_compute(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    monkeypatch.setattr(
        "marin.rl.placement.infer_tpu_variant_regions_from_iris",
        lambda variants: ["us-central1", "us-east5"],
    )

    with pytest.raises(ValueError, match="current launcher region"):
        launcher_region_for_rl_experiment(_test_config())


def test_launcher_region_uses_gpu_rollout_region_when_it_matches_tpu_capacity(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(
        "marin.rl.placement.infer_tpu_variant_regions_from_iris",
        lambda variants: ["us-east5"],
    )

    config = _test_config(
        rollout_resources=ResourceConfig.with_gpu(
            "H100",
            count=4,
            cpu=32,
            ram="240g",
            disk="128g",
            regions=["us-east5"],
        )
    )

    assert launcher_region_for_rl_experiment(config) == "us-east5"


def test_zone_pinned_tpu_resources_use_zone_region_for_launcher_and_prefix(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(
        "marin.rl.placement.infer_tpu_variant_regions_from_iris",
        lambda variants: ["us-central1", "us-east5"],
    )

    config = _test_config(
        train_resources=ResourceConfig.with_tpu("v5p-8", zone="us-east5-d"),
        rollout_resources=ResourceConfig.with_tpu("v5p-8", zone="us-east5-d"),
    )

    assert launcher_region_for_rl_experiment(config) == "us-east5"
    assert executor_main_config_for_rl_experiment(config).prefix == "gs://marin-us-east5"


def test_launcher_region_rejects_conflicting_zone_and_regions(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setattr(
        "marin.rl.placement.infer_tpu_variant_regions_from_iris",
        lambda variants: ["us-central1", "us-east5"],
    )

    config = _test_config(
        train_resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"], zone="us-east5-d"),
        rollout_resources=ResourceConfig.with_tpu("v5p-8", zone="us-east5-d"),
    )

    with pytest.raises(ValueError, match="conflicts with requested regions"):
        launcher_region_for_rl_experiment(config)


def test_make_rl_step_uses_model_step_artifact_root_as_dependency(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    model_step = ExecutorStep(name="models/test-llama", fn=_noop, config=_EmptyConfig())
    config = dataclasses.replace(
        _test_config(),
        model_config=ModelConfig(
            name=MODEL_NAME,
            type="llama",
            artifact=model_step,
            config_class_path=config_class_path(LlamaConfig),
        ),
    )

    step = make_rl_step(name="rl-test", config=config, curriculum=_test_curriculum())

    assert step.config.model_path == output_path_of(model_step)
    assert step.config.experiment_config.model_config.artifact == output_path_of(model_step)


def test_is_hf_checkpoint_recognizes_gcs_hf_exports(monkeypatch):
    hf_files = {
        "gs://marin-us-central1/models/test-model/hf/config.json",
    }
    monkeypatch.setattr("marin.rl.model_utils.fsspec_exists", lambda path: path in hf_files)

    assert is_hf_checkpoint("gs://marin-us-central1/models/test-model/hf")
    assert not is_hf_checkpoint("gs://marin-us-central1/checkpoints/test-run")


def test_build_rl_job_config_resolves_runtime_output_paths(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(),
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.trainer.checkpointer.base_path == "gs://marin-us-central1/rl_testing/rl-test/checkpoints"
    assert job_config.rollout_storage.path == "gs://marin-us-central1/rl_testing/rl-test/rollouts"
    assert job_config.inference_config.load_format == "runai_streamer"
    assert job_config.inference_config.canonical_model_name == MODEL_NAME
    assert job_config.model.attn_backend == AttentionBackend.SPLASH


def test_build_rl_job_config_uses_dummy_load_format_for_non_object_store_model_path(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(),
        curriculum=_test_curriculum(),
        model_path=MODEL_NAME,
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.inference_config.load_format == "dummy"


def test_build_rl_job_config_rejects_gpu_inflight_vllm(monkeypatch):
    monkeypatch.delenv("MARIN_PREFIX", raising=False)

    with pytest.raises(ValueError, match="does not yet support inflight_weight_updates"):
        _build_rl_job_config(
            name="rl-test",
            config=_test_config(
                inflight_weight_updates=True,
                rollout_resources=ResourceConfig.with_gpu(
                    "H100",
                    count=4,
                    cpu=32,
                    ram="240g",
                    disk="128g",
                    regions=["us-central1"],
                ),
            ),
            curriculum=_test_curriculum(),
            model_path=MODEL_NAME,
            output_path="gs://marin-us-central1/rl_testing/rl-test",
        )


def test_build_rl_job_config_uses_default_attention_backend_for_gpu_training(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(
            train_resources=ResourceConfig.with_gpu(
                "H100",
                count=4,
                cpu=32,
                ram="240g",
                disk="128g",
                regions=["us-central1"],
            ),
            rollout_resources=ResourceConfig.with_gpu(
                "H100",
                count=4,
                cpu=32,
                ram="240g",
                disk="128g",
                regions=["us-central1"],
            ),
        ),
        curriculum=_test_curriculum(),
        model_path=MODEL_NAME,
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.model.attn_backend == AttentionBackend.DEFAULT


def test_build_rl_job_config_preserves_explicit_train_attention_backend(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(
            train_resources=ResourceConfig.with_gpu(
                "H100",
                count=4,
                cpu=32,
                ram="240g",
                disk="128g",
                regions=["us-central1"],
            ),
            rollout_resources=ResourceConfig.with_gpu(
                "H100",
                count=4,
                cpu=32,
                ram="240g",
                disk="128g",
                regions=["us-central1"],
            ),
            train_attention_backend=AttentionBackend.JAX_FLASH,
        ),
        curriculum=_test_curriculum(),
        model_path=MODEL_NAME,
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.model.attn_backend == AttentionBackend.JAX_FLASH


def test_build_rl_job_config_propagates_resource_configs(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(
            train_resources=ResourceConfig.with_tpu("v5p-8", ram="300g"),
            rollout_resources=ResourceConfig.with_gpu(
                "H100",
                count=4,
                cpu=32,
                ram="240g",
                disk="128g",
            ),
        ),
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.run_config.train_resources.device.kind == "tpu"
    assert job_config.run_config.train_resources.ram == "300g"
    assert job_config.run_config.train_resources.regions == ["us-central1"]
    assert job_config.run_config.rollout_resources.device.kind == "gpu"
    assert job_config.run_config.rollout_resources.ram == "240g"
    assert job_config.run_config.rollout_resources.regions == ["us-central1"]
    assert job_config.inference_config.device_kind == "gpu"


def test_build_rl_job_config_propagates_checkpoint_controls_and_instance_id(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(
            train_resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
            rollout_resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
            delete_previous_temporary_checkpoint_after_save=False,
            checkpoint_debug=CheckpointDebugConfig(
                enabled=True,
                log_interval=15.0,
                dump_stacks_after=45.0,
            ),
        ),
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/rl-test",
        instance_id="rl-test-instance",
    )

    assert job_config.instance_id == "rl-test-instance"
    assert job_config.run_config.train_resources.regions == ["us-central1"]
    assert job_config.run_config.rollout_resources.regions == ["us-central1"]
    assert job_config.curriculum.actor_name == "curriculum-rl-test-instance"
    assert job_config.weight_transfer.coordinator_name == "wt-coord-rl-test-instance"
    assert not job_config.trainer.checkpointer.delete_previous_temporary_checkpoint_after_save
    assert job_config.trainer.checkpointer.debug.enabled
    assert job_config.trainer.checkpointer.debug.log_interval == 15.0
    assert job_config.trainer.checkpointer.debug.dump_stacks_after == 45.0


def test_run_rl_experiment_step_returns_serializable_path_metadata(monkeypatch):
    calls = {}

    class _FakeRLJob:
        def __init__(self, config):
            calls["job_config"] = config

        def run(self, name):
            calls["name"] = name
            return object()

    runtime_config = _test_config()
    step_config = RLStepConfig(
        name="exec-gcs-small-test",
        experiment_config=runtime_config,
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/exec-gcs-small-test",
    )

    def _fake_build_rl_job_config(**kwargs):
        calls["build_kwargs"] = kwargs
        return "job-config"

    monkeypatch.setattr("marin.rl.rl_experiment_utils._build_rl_job_config", _fake_build_rl_job_config)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.RLJob", _FakeRLJob)

    result = _run_rl_experiment_step(step_config)

    assert calls["build_kwargs"]["name"] == "exec-gcs-small-test"
    assert calls["build_kwargs"]["instance_id"].startswith("exec-gcs-small-test-")
    assert calls["job_config"] == "job-config"
    assert calls["name"] == "exec-gcs-small-test"
    assert result == PathMetadata(path="gs://marin-us-central1/rl_testing/exec-gcs-small-test")
