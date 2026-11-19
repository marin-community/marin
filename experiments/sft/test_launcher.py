# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from dataclasses import replace

import draccus
import numpy as np
import pytest
from haliax import Axis
from levanter.data.text.datasets import LmDataConfig, dataset_for_component
from levanter.data.text.formats import SerializedChatLmDatasetFormat
from levanter.store.cache import SerialCacheWriter
from levanter.utils.thread_utils import blocking_wait
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.training.levanter import GPU_NCCL_TERMINATION_TIMEOUT_FLAG
from rigging.timing import Duration

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.post_training.snowball_sft import (
    HF_MODEL,
    MIXTURE_262K_32NODE,
    MIXTURE_SPEC,
    MIXTURE_TOKENIZER,
    OPENTHOUGHTS_32K_4NODE,
    OPENTHOUGHTS_SPEC,
    SNOWBALL_SFT,
)
from experiments.sft import launcher
from experiments.sft.launcher import PreparedData, PreparedModel


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (None, "Build a Hugging Face export for the GrugModel trainer"),
        ({"xla_flags": ("--flag",)}, "xla_flags"),
        ({"per_device_parallelism": 1}, "per_device_parallelism"),
    ],
)
def test_sft_refuses_incompatible_grug_settings(overrides, error):
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    spec = replace(OPENTHOUGHTS_SPEC, model=grug, **(overrides or {}))
    with pytest.raises(ValueError, match=error):
        if overrides is None:
            replace(OPENTHOUGHTS_32K_4NODE, recipe=spec).model()
        else:
            launcher.sft_step(spec, OPENTHOUGHTS_32K_4NODE.resources).fingerprint()


def test_openthoughts_keeps_one_conversation_per_context_window(tmp_path):
    training = materialized_config(OPENTHOUGHTS_32K_4NODE.model().step, str(tmp_path)).train_config
    [component] = training.data.components.values()
    conversations = [
        {"input_ids": np.arange(10, 22, dtype=np.int32), "assistant_masks": np.arange(12) >= 2},
        {"input_ids": np.array([31, 32, 33], dtype=np.int32), "assistant_masks": np.array([False, True, True])},
        {"input_ids": np.array([41, 42, 43], dtype=np.int32), "assistant_masks": np.array([False, True, True])},
    ]
    with SerialCacheWriter(str(tmp_path / "cache"), conversations[0]) as writer:
        writer.write_batch(conversations)
    dataset = dataset_for_component(
        component, Axis("position", 8), writer.result(), eos_id=None, block_cross_document_attention=True
    )
    assert blocking_wait(dataset.async_len()) == 3
    long, first_short, second_short = blocking_wait(dataset.get_batch([0, 1, 2]))
    np.testing.assert_array_equal(long.tokens, np.arange(10, 18))
    np.testing.assert_array_equal(np.flatnonzero(long.loss_weight), np.arange(1, 7))
    np.testing.assert_array_equal(first_short.tokens, [31, 32, 33, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(second_short.tokens, [41, 42, 43, 0, 0, 0, 0, 0])


@pytest.mark.parametrize(
    ("change", "message"),
    [("mesh", "mesh does not fit SFT resources"), ("batch", "must be divisible by the mesh microbatch")],
)
def test_sft_plan_refuses_an_invalid_gpu_mesh_or_batch(monkeypatch, capsys, change, message):
    preset = (
        replace(MIXTURE_262K_32NODE, resources=OPENTHOUGHTS_32K_4NODE.resources)
        if change == "mesh"
        else replace(MIXTURE_262K_32NODE, recipe=replace(MIXTURE_SPEC, batch_size=7))
    )
    monkeypatch.setattr(sys, "argv", ["snowball_sft", "--preset", preset.name])
    with pytest.raises(SystemExit) as exit_info:
        launcher.main((preset,))
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


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
    model = OPENTHOUGHTS_SPEC.model
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
        spec = replace(OPENTHOUGHTS_SPEC, model=model, datasets=(), prepared_data=source, **steps)
        step = replace(OPENTHOUGHTS_32K_4NODE, recipe=spec).model().step
        ctx = StepContext.for_run(
            str(tmp_path / "output"), str(tmp_path), runtime_args=step.runtime_args, deps=step.deps
        )
        step.build_config(ctx)


def test_snowball_262k_mixture_binds_the_published_mixture(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    model = MIXTURE_262K_32NODE.model()
    training = model.step
    [manifest] = training.deps
    submitted = {}

    def fake_remote(_fn, **kwargs):
        submitted.update(kwargs)
        return lambda _config: None

    monkeypatch.setattr("marin.experiment.train.remote", fake_remote)
    args = ["snowball_sft", "--preset", MIXTURE_262K_32NODE.name]
    for flags, exit_code in [(("--accelerator", "4xH100"), 2), ((), 0)]:
        monkeypatch.setattr(sys, "argv", [*args, *flags])
        with pytest.raises(SystemExit) as exit_info:
            launcher.main(SNOWBALL_SFT)
        assert exit_info.value.code == exit_code
    assert "preset fixes its own allocation" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
    assert not submitted

    monkeypatch.setattr(sys, "argv", [*args, "--run"])
    with pytest.raises(SystemExit) as exit_info:
        launcher.main(SNOWBALL_SFT)
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
    assert isinstance(config.data.components["sft/source/agenttrove"].format, SerializedChatLmDatasetFormat)
    assert config.trainer.train_batch_size == 256

    assert submitted["timeout"] == Duration.from_seconds(72_000)
    flags = submitted["env_vars"]["XLA_FLAGS"].split()
    assert "--xla_gpu_shard_autotuning=false" in flags
    assert any(flag.startswith(f"--{GPU_NCCL_TERMINATION_TIMEOUT_FLAG}=") for flag in flags)
