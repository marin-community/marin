# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
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
from experiments.post_training.snowball_sft import (
    HF_MODEL,
    MIXTURE_SFT,
    MIXTURE_TOKENIZER,
    OPENTHOUGHTS_SFT,
    SNOWBALL_SFT,
)
from experiments.sft import launcher
from experiments.sft.launcher import PreparedData, PreparedModel


def test_levanter_sft_refuses_grug_without_hf_export():
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    preset = replace(OPENTHOUGHTS_SFT, recipe=replace(OPENTHOUGHTS_SFT.recipe, model=grug))
    with pytest.raises(ValueError, match=r"Build a Hugging Face export for the GrugModel trainer"):
        preset.model()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [({"xla_flags": ("--flag",)}, "xla_flags"), ({"per_device_parallelism": 1}, "per_device_parallelism")],
)
def test_grug_rejects_levanter_runtime_settings(overrides, error):
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    spec = replace(OPENTHOUGHTS_SFT.recipe, model=grug, **overrides)
    with pytest.raises(ValueError, match=error):
        launcher.sft_step(spec, OPENTHOUGHTS_SFT.resources).fingerprint()


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
    model = OPENTHOUGHTS_SFT.recipe.model
    source = PreparedData(artifact, "mix.json", HF_MODEL)
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
        spec = replace(OPENTHOUGHTS_SFT.recipe, model=model, datasets=(), prepared_data=source, **steps)
        step = replace(OPENTHOUGHTS_SFT, recipe=spec).model().step
        ctx = StepContext.for_run(
            str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        step.build_config(ctx)


def test_snowball_round1_binds_the_published_mixture(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    model = MIXTURE_SFT.model()
    training = model.step
    [manifest] = training.deps
    submitted = {}

    def fake_remote(_fn, **kwargs):
        submitted.update(kwargs)
        return lambda _config: None

    monkeypatch.setattr(launcher, "remote", fake_remote)
    args = ["snowball_sft", "--preset", MIXTURE_SFT.name]
    for flags, exit_code in [(("--accelerator", "4xH100"), 2), ((), 0)]:
        monkeypatch.setattr(sys, "argv", [*args, *flags])
        with pytest.raises(SystemExit) as exit_info:
            launcher.run_sft_cli(SNOWBALL_SFT)
        assert exit_info.value.code == exit_code
    assert "preset fixes its own allocation" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
    assert not submitted

    monkeypatch.setattr(sys, "argv", [*args, "--run"])
    with pytest.raises(SystemExit) as exit_info:
        launcher.run_sft_cli(SNOWBALL_SFT)
    assert exit_info.value.code == 0

    output = manifest.path(str(tmp_path))
    manifest_config = manifest.build_config(StepContext.for_run(output, str(tmp_path)))
    with pytest.raises(ValueError, match="SHA-256 pin"):
        manifest.run({**manifest_config, "sha256": "0" * 64})
    context = StepContext.for_run(
        str(tmp_path / "training"), str(tmp_path), runtime_args=training.runtime_args, deps=training.deps
    )
    pod_config = training.build_config(context)
    config = pod_config.train_config
    assert config.data.tokenizer == MIXTURE_TOKENIZER
    assert config.data.cache_catalog == (
        "s3://marin-us-east-02a/marin/grug-sft/sft-pipe-repro/full-mixture/2026.10.24.2/cache_catalog.json"
    )
    assert len(config.data.components) == 377
    assert isinstance(config.data.components["sft/source/agenttrove"].format, SerializedChatLmDatasetFormat)
    assert len(config.data.train_weights) == 19
    assert [phase for phase, _ in config.data.train_weights][:3] == [0, 250, 500]
    assert config.trainer.train_batch_size == 256
    assert model.relative_path == f"{DEFAULT_HF_CHECKPOINTS_PATH}/step-203"

    assert submitted["timeout"] == Duration.from_seconds(72_000)
    flags = submitted["env_vars"]["XLA_FLAGS"].split()
    assert "--xla_gpu_shard_autotuning=false" in flags
    assert any(flag.startswith(f"--{GPU_NCCL_TERMINATION_TIMEOUT_FLAG}=") for flag in flags)

    step = replace(MIXTURE_SFT, resources=OPENTHOUGHTS_SFT.resources).model().step
    with pytest.raises(ValueError, match="mesh does not fit SFT resources"):
        materialized_config(step, str(tmp_path))
    with pytest.raises(ValueError, match="must be divisible by the mesh microbatch"):
        preset = replace(MIXTURE_SFT, recipe=replace(MIXTURE_SFT.recipe, batch_size=7))
        step = preset.model().step
        materialized_config(step, str(tmp_path))
    with pytest.raises(ValueError, match="must be divisible by the mesh microbatch"):
        preset = replace(MIXTURE_SFT, recipe=replace(MIXTURE_SFT.recipe, mesh=None, batch_size=7))
        step = preset.model().step
        materialized_config(step, str(tmp_path))
