# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import draccus
import pytest
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.formats import SerializedChatLmDatasetFormat
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.training.levanter import GPU_NCCL_TERMINATION_TIMEOUT_FLAG
from marin.training.training import DEFAULT_HF_CHECKPOINTS_PATH
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


@pytest.mark.parametrize(
    ("overrides", "error"),
    [({"xla_flags": ("--flag",)}, "xla_flags"), ({"per_device_parallelism": 1}, "per_device_parallelism")],
)
def test_grug_rejects_levanter_runtime_settings(overrides, error):
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    spec = replace(snowball_h100.SPEC, model=grug, **overrides)
    with pytest.raises(ValueError, match=error):
        launcher.sft_step(spec, snowball_h100.RESOURCES).fingerprint()


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        ("declared", r"prepared_data\.tokenizer must match"),
        ("manifest", "file tokenizer does not match"),
        ("resolved", "resolved SFT model tokenizer"),
        ("epochs", "prepared_data requires num_train_steps"),
    ],
)
def test_prepared_data_requires_matching_tokenizers_and_fixed_steps(tmp_path, scenario, error):
    artifact = ArtifactStep.adopt(name="mixture", version="2026.11.19", source=str(tmp_path), kind=Artifact)
    model = snowball_h100.SPEC.model
    source = PreparedData(artifact, "mix.json", snowball_h100.HF_MODEL)
    if scenario == "resolved":
        weights = ArtifactStep.adopt(
            name="weights", version="2026.11.19", source=str(tmp_path / "weights"), kind=Artifact
        )
        model = PreparedModel(weights)
    if scenario == "declared":
        source = replace(source, tokenizer="different-tokenizer")
    tokenizer = "different-tokenizer" if scenario == "manifest" else source.tokenizer
    manifest = tmp_path / source.file
    manifest.write_text(json.dumps(draccus.encode(LmDataConfig(tokenizer=tokenizer))))
    steps = {"num_train_steps": None, "num_train_epochs": 1} if scenario == "epochs" else {}
    with pytest.raises(ValueError, match=error):
        spec = replace(snowball_h100.SPEC, model=model, datasets=(), prepared_data=source, **steps)
        step = sft_model(LevanterSft(spec), snowball_h100.RESOURCES).step
        ctx = StepContext.for_run(
            str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        step.build_config(ctx)


def test_snowball_round1_binds_the_published_mixture(tmp_path, monkeypatch):
    model = sft_model(snowball_round1.SFT, snowball_round1.RESOURCES)
    training = model.step
    [manifest] = training.deps
    output = manifest.path(str(tmp_path))
    manifest_config = manifest.build_config(StepContext.for_run(output, str(tmp_path)))
    manifest.run(manifest_config)
    with pytest.raises(ValueError, match="SHA-256 pin"):
        manifest.run({**manifest_config, "sha256": "0" * 64})
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
    assert model.relative_path == f"{DEFAULT_HF_CHECKPOINTS_PATH}/step-23871"

    submitted = {}

    def fake_remote(_fn, **kwargs):
        submitted.update(kwargs)
        return lambda _config: None

    monkeypatch.setattr(launcher, "remote", fake_remote)
    training.run(pod_config)
    assert submitted["timeout"] == Duration.from_seconds(72_000)
    flags = submitted["env_vars"]["XLA_FLAGS"].split()
    assert "--xla_gpu_shard_autotuning=false" in flags
    assert any(flag.startswith(f"--{GPU_NCCL_TERMINATION_TIMEOUT_FLAG}=") for flag in flags)


def test_snowball_round1_refuses_a_mesh_that_does_not_fit_resources(tmp_path):
    step = sft_model(snowball_round1.SFT, snowball_h100.RESOURCES).step
    [manifest] = step.deps
    output = manifest.path(str(tmp_path))
    manifest.run(manifest.build_config(StepContext.for_run(output, str(tmp_path))))
    with pytest.raises(ValueError, match="mesh does not fit SFT resources"):
        materialized_config(step, str(tmp_path))
    with pytest.raises(ValueError, match="must be divisible by the mesh microbatch"):
        step = sft_model(snowball_round1.SFT, replace(snowball_round1.RESOURCES, replicas=64)).step
        materialized_config(step, str(tmp_path))
    with pytest.raises(ValueError, match="must be divisible by the mesh microbatch"):
        default_mesh = LevanterSft(replace(snowball_round1.SPEC, mesh=None))
        step = sft_model(default_mesh, snowball_round1.RESOURCES).step
        materialized_config(step, str(tmp_path))
