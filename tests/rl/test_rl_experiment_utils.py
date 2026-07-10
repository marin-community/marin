# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import pytest
import rigging.filesystem as fs
from levanter.models.llama import LlamaConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl.curriculum import CurriculumConfig
from marin.rl.kl_regularization import KLConfig, KLMode
from marin.rl.model_utils import is_hf_checkpoint
from marin.rl.rl_experiment_utils import (
    ModelConfig,
    RLExperimentConfig,
    RLStepConfig,
    _build_rl_job_config,
    _run_rl_experiment_step,
    config_class_path,
    executor_step_resources_for_rl_experiment,
    launcher_region_for_rl_experiment,
    make_rl_step,
)
from marin.rl.rl_losses import RLOOLoss
from marin.training.training import LevanterCheckpoint

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# A minimal DataConfig covering just the regions these tests exercise (mirrors the
# relevant entries in config/marin.yaml). eu-west4 -> europe-west4 stands in for any
# non-identity bucket-to-region normalization; the mapping itself isn't what's under
# test, so it's bound explicitly rather than resolved from the ambient cluster config.
_TEST_DATA_CONFIG = fs.DataConfig(
    scheme="gs",
    region_buckets={
        "us-central1": fs.BucketSpec(name="marin-us-central1", store=fs.StoreType.GCS),
        "us-east5": fs.BucketSpec(name="marin-us-east5", store=fs.StoreType.GCS),
        "europe-west4": fs.BucketSpec(name="marin-eu-west4", store=fs.StoreType.GCS),
    },
)


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
    # marin_region() checks GCE instance metadata before MARIN_PREFIX; on a real
    # GCP host that resolves a real region and silently overrides the prefix
    # these tests set, so pin it to "not on GCP" for hermetic region resolution.
    monkeypatch.setattr(fs, "region_from_metadata", lambda: None)
    # Bind a fixed DataConfig instead of resolving the ambient cluster config, so
    # these tests are independent of the host's config search path (a per-user
    # ~/.config/marin/clusters override, cwd, missing config/ dir, etc).
    with fs.use_data_config(_TEST_DATA_CONFIG):
        yield


def _test_config(
    *,
    train_tpu_type: str,
    inference_tpu_type: str,
) -> RLExperimentConfig:
    return RLExperimentConfig(
        model_config=ModelConfig(
            name=MODEL_NAME,
            type="llama",
            artifact=MODEL_NAME,
            config_class_path=config_class_path(LlamaConfig),
        ),
        rl_loss=RLOOLoss(
            kl=KLConfig(mode=KLMode.NONE, beta=0.0),
            clip_epsilon_low=0.2,
            clip_epsilon_high=0.28,
            synchronous=True,
            do_trainer_inference_mismatch_importance_sampling=True,
            tis_importance_sampling_ratio_max=2.0,
            do_overlong_filtering=True,
            vocab_tile_size=32064,
        ),
        experiment_name_suffix="test",
        train_tpu_type=train_tpu_type,
        inference_tpu_type=inference_tpu_type,
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

    resources = executor_step_resources_for_rl_experiment(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8")
    )

    assert resources.regions == ["us-east5"]


def test_non_v5p_executor_step_regions_follow_current_launcher_region(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    resources = executor_step_resources_for_rl_experiment(
        _test_config(train_tpu_type="v6e-4", inference_tpu_type="v6e-4")
    )

    assert resources.regions == ["europe-west4"]


def test_launcher_region_raises_when_root_region_conflicts_with_requested_compute(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-eu-west4")

    monkeypatch.setattr(
        "marin.rl.placement.infer_tpu_variant_regions_from_iris",
        lambda variants: ["us-central1", "us-east5"],
    )

    with pytest.raises(ValueError, match="current launcher region"):
        launcher_region_for_rl_experiment(_test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8"))


def test_make_rl_step_uses_model_checkpoint_path_as_dependency(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    model_checkpoint = ArtifactStep(
        name="models/test-llama",
        version="2026.06.28",
        artifact_type=LevanterCheckpoint,
        run=_noop,
        build_config=lambda _ctx: _EmptyConfig(),
    )
    config = dataclasses.replace(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8"),
        model_config=ModelConfig(
            name=MODEL_NAME,
            type="llama",
            artifact=model_checkpoint,
            config_class_path=config_class_path(LlamaConfig),
        ),
    )

    step = make_rl_step(name="rl-test", config=config, curriculum=_test_curriculum(), version="2026.06.28")

    # The model checkpoint is a dependency, and the step's config resolves to its path.
    assert step.deps == (model_checkpoint,)
    expected_path = model_checkpoint.path("gs://marin-us-central1")
    step_config = materialized_config(step, "gs://marin-us-central1")
    assert step_config.model_path == expected_path
    assert step_config.experiment_config.model_config.artifact == expected_path


def test_is_hf_checkpoint_recognizes_gcs_hf_exports(monkeypatch):
    hf_files = {
        "gs://marin-us-central1/models/test-model/hf/config.json",
    }
    monkeypatch.setattr("rigging.filesystem.StoragePath.exists", lambda self: str(self) in hf_files)

    assert is_hf_checkpoint("gs://marin-us-central1/models/test-model/hf")
    assert not is_hf_checkpoint("gs://marin-us-central1/checkpoints/test-run")


def test_build_rl_job_config_resolves_runtime_output_paths(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8"),
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.trainer.checkpointer.base_path == "gs://marin-us-central1/rl_testing/rl-test/checkpoints"
    assert job_config.rollout_storage.path == "gs://marin-us-central1/rl_testing/rl-test/rollouts"
    assert job_config.inference_config.engine.load_format == "runai_streamer"
    assert job_config.inference_config.engine.canonical_model_name == MODEL_NAME


def test_build_rl_job_config_keeps_rollout_policy_out_of_vllm_fallback_sampling(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    config = dataclasses.replace(
        _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8"),
        n_generations_per_prompt=16,
        train_decoding_top_k=4096,
    )
    job_config = _build_rl_job_config(
        name="rl-test",
        config=config,
        curriculum=_test_curriculum(),
        model_path=MODEL_NAME,
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.inference_config.engine.max_model_len == config.max_input_tokens + config.max_output_tokens
    assert job_config.inference_config.fallback_sampling.top_k is None
    assert job_config.inference_config.fallback_sampling.stop_strings == ["<|eot_id|>"]


def test_build_rl_job_config_uses_dummy_load_format_for_non_object_store_model_path(monkeypatch):
    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            self.default_hf_config = SimpleNamespace(vocab_size=32000)

    monkeypatch.setattr("marin.rl.rl_experiment_utils._resolve_config_class", lambda _path: _FakeRuntimeLmConfig)
    monkeypatch.setattr("marin.rl.rl_experiment_utils.HFCheckpointConverter", _FakeConverter)

    job_config = _build_rl_job_config(
        name="rl-test",
        config=_test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8"),
        curriculum=_test_curriculum(),
        model_path=MODEL_NAME,
        output_path="gs://marin-us-central1/rl_testing/rl-test",
    )

    assert job_config.inference_config.engine.load_format == "dummy"
    assert job_config.inference_config.engine.canonical_model_name == MODEL_NAME


def test_run_rl_experiment_step_returns_a_path_ref(monkeypatch):
    calls = {}

    class _FakeRLJob:
        def __init__(self, config):
            calls["job_config"] = config

        def run(self, name):
            calls["name"] = name
            return object()

    runtime_config = _test_config(train_tpu_type="v5p-8", inference_tpu_type="v5p-8")
    step_config = RLStepConfig(
        name="exec-gcs-small-test",
        experiment_config=runtime_config,
        curriculum=_test_curriculum(),
        model_path="gs://marin-us-central1/models/test-model",
        output_path="gs://marin-us-central1/rl_testing/exec-gcs-small-test",
        resources=executor_step_resources_for_rl_experiment(runtime_config),
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
    assert result == Artifact(path="gs://marin-us-central1/rl_testing/exec-gcs-small-test")
