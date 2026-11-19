# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import draccus
import pytest
from fray.types import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig
from experiments.june_tpu_67b_a2b.moe.sft_launch import GrugModel
from experiments.sft.configs import mochi, snowball_h100, snowball_round1
from experiments.sft.launcher import LevanterSft, PreparedData, sft_model


def test_levanter_sft_returns_final_hf_export():
    model = sft_model(snowball_h100.SFT, snowball_h100.RESOURCES)
    assert model.relative_path == "hf/step-9"
    assert model.tokenizer_uri is None


def test_levanter_sft_refuses_grug_without_hf_export():
    grug = GrugModel(
        model=GrugModelConfig(vocab_size=32), tokenizer_path="fixture", init_from="/checkpoint", expert_parallel=1
    )
    recipe = LevanterSft(replace(snowball_h100.SPEC, model=grug))
    with pytest.raises(ValueError, match=r"GrugModel.*Hugging Face export"):
        sft_model(recipe, snowball_h100.RESOURCES)


def test_mochi_sft_refuses_resources_that_do_not_fit_its_mesh():
    with pytest.raises(ValueError, match="GB200 allocation derived from dp_racks"):
        sft_model(mochi.SFT, ResourceConfig.with_cpu())


def test_prepared_data_uses_the_manifest_and_matches_the_model_tokenizer(tmp_path):
    artifact = ArtifactStep.adopt(name="mixture", version="2026.11.19", source=str(tmp_path), kind=Artifact)
    source = PreparedData(artifact, "mix.json", snowball_h100.HF_MODEL)
    spec = replace(snowball_h100.SPEC, datasets=(), prepared_data=source)
    with pytest.raises(ValueError, match=r"prepared_data\.tokenizer"):
        replace(spec, prepared_data=replace(source, tokenizer="different-tokenizer"))
    step = sft_model(LevanterSft(spec), snowball_h100.RESOURCES).step
    assert step.deps == (artifact,)
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


def test_snowball_round1_binds_the_published_mixture(tmp_path):
    model = sft_model(snowball_round1.SFT, snowball_round1.RESOURCES)
    training = model.step
    [manifest] = training.deps
    output = manifest.path(str(tmp_path))
    manifest.run(manifest.build_config(StepContext.for_run(output, str(tmp_path))))
    context = StepContext.for_run(
        str(tmp_path / "training"), str(tmp_path), runtime_args=training.runtime_args, deps=training.deps
    )
    config = training.build_config(context).train_config
    assert config.data.tokenizer == snowball_round1.TOKENIZER
    assert config.data.cache_catalog.endswith("/2026.10.24.2/cache_catalog.json")
    assert len(config.data.components) == 377
    assert len(config.data.train_weights) == 19
    assert config.trainer.num_train_steps == 23_872
    assert model.relative_path == "hf/step-23871"
