# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import draccus
import pytest
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.formats import SerializedChatLmDatasetFormat
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.qwen import Qwen3Config
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.timing import Duration

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.sft import launcher
from experiments.sft.configs import snowball_h100, snowball_round1
from experiments.sft.launcher import LevanterSft, PreparedData, PreparedModel, sft_model


def test_levanter_sft_refuses_grug_without_hf_export():
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    recipe = LevanterSft(replace(snowball_h100.SPEC, model=grug))
    with pytest.raises(ValueError, match=r"GrugModel.*Hugging Face export"):
        sft_model(recipe, snowball_h100.RESOURCES)


def test_prepared_data_uses_the_manifest_and_matches_the_model_tokenizer(tmp_path):
    artifact = ArtifactStep.adopt(name="mixture", version="2026.11.19", source=str(tmp_path), kind=Artifact)
    source = PreparedData(artifact, "mix.json", snowball_h100.HF_MODEL)
    spec = replace(snowball_h100.SPEC, datasets=(), prepared_data=source)
    with pytest.raises(ValueError, match=r"prepared_data\.tokenizer"):
        replace(spec, prepared_data=replace(source, tokenizer="different-tokenizer"))
    step = sft_model(LevanterSft(spec), snowball_h100.RESOURCES).step
    assert step.deps == (artifact,)
    assert "mixture@2026.11.19/mix.json" in step.fingerprint_payload()
    ctx = StepContext.for_run(str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    path = tmp_path / source.file
    data = LmDataConfig(
        tokenizer=spec.model.tokenizer_cache_key(), cache_catalog="/catalog.json", auto_build_caches=False
    )
    path.write_text(json.dumps(draccus.encode(data)))
    config = step.build_config(ctx)
    assert config.train_config.data == data
    assert config.auto_build_caches is False
    path.write_text(json.dumps(draccus.encode(replace(data, tokenizer="different-tokenizer"))))
    with pytest.raises(ValueError, match="file tokenizer"):
        step.build_config(ctx)


def test_prepared_data_matches_the_resolved_artifact_tokenizer(tmp_path):
    weights = ArtifactStep.adopt(name="weights", version="2026.11.19", source=str(tmp_path / "weights"), kind=Artifact)
    data = ArtifactStep.adopt(name="mixture", version="2026.11.19", source=str(tmp_path), kind=Artifact)
    source = PreparedData(data, "mix.json", str(tmp_path / "weights"))
    spec = replace(snowball_h100.SPEC, model=PreparedModel(weights), datasets=(), prepared_data=source)
    step = sft_model(LevanterSft(spec), snowball_h100.RESOURCES).step
    ctx = StepContext.for_run(str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps)
    manifest = tmp_path / source.file
    manifest.write_text(json.dumps(draccus.encode(LmDataConfig(tokenizer=source.tokenizer))))

    assert step.build_config(ctx).train_config.data.tokenizer == source.tokenizer
    other = str(tmp_path / "other")
    spec = replace(spec, prepared_data=replace(source, tokenizer=other))
    manifest.write_text(json.dumps(draccus.encode(LmDataConfig(tokenizer=other))))
    mismatched = sft_model(LevanterSft(spec), snowball_h100.RESOURCES).step
    with pytest.raises(ValueError, match="resolved SFT model tokenizer"):
        mismatched.build_config(ctx)


def test_snowball_round1_binds_the_published_mixture(tmp_path, monkeypatch):
    model = sft_model(snowball_round1.SFT, snowball_round1.RESOURCES)
    training = model.step
    [manifest] = training.deps
    output = manifest.path(str(tmp_path))
    manifest.run(manifest.build_config(StepContext.for_run(output, str(tmp_path))))
    context = StepContext.for_run(
        str(tmp_path / "training"), str(tmp_path), runtime_args=training.runtime_args, deps=training.deps
    )
    pod_config = training.build_config(context)
    config = pod_config.train_config
    assert config.data.tokenizer == snowball_round1.TOKENIZER
    assert config.data.cache_catalog == (
        "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json"
    )
    assert len(config.data.components) == 377
    assert isinstance(config.data.components["sft/source/agenttrove"].format, SerializedChatLmDatasetFormat)
    assert len(config.data.train_weights) == 19
    assert [phase for phase, _ in config.data.train_weights][:3] == [0, 8000, 16000]
    assert config.trainer.num_train_steps == 23_872
    assert model.relative_path == "hf/step-23871"

    submitted = {}

    def fake_remote(_fn, **kwargs):
        submitted.update(kwargs)
        return lambda _config: None

    monkeypatch.setattr(launcher, "remote", fake_remote)
    training.run(pod_config)
    assert submitted["timeout"] == Duration.from_seconds(72_000)
    assert submitted["env_vars"]["XLA_FLAGS"].split() == [
        "--xla_gpu_shard_autotuning=false",
        "--xla_gpu_nccl_termination_timeout_seconds=600",
    ]


def test_snowball_round1_refuses_a_mesh_that_does_not_fit_resources():
    with pytest.raises(ValueError, match="mesh does not fit SFT resources"):
        sft_model(snowball_round1.SFT, snowball_h100.RESOURCES)
    with pytest.raises(ValueError, match="must be divisible by the mesh microbatch"):
        sft_model(snowball_round1.SFT, replace(snowball_round1.RESOURCES, replicas=64))


def test_qwen3_hf_reload_keeps_explicit_runtime_settings():
    requested = Qwen3Config(gradient_checkpointing=False, scan_layers=False)
    reloaded = Qwen3Config(max_seq_len=8192)
    config = preserve_hf_runtime_settings(reloaded, requested)
    assert config == replace(reloaded, gradient_checkpointing=False, scan_layers=False)
