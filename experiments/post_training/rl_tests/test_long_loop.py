# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace
from pathlib import Path

import draccus
import pytest
import yaml
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.experiment.cli import graph_handles

from experiments.grug.moe_hero_ep.launch_hero_sft import SFT_TOKENIZER
from experiments.post_training.long_loop import build_arm
from experiments.post_training.mochi_rl import MOCHI
from experiments.post_training.snowball_rl import SNOWBALL
from experiments.post_training.snowball_sft import MIXTURE_262K_32NODE
from experiments.sft.launcher import PreparedData

pytestmark = pytest.mark.usefixtures("source_provenance")


@pytest.mark.parametrize("model", ("mochi", "snowball"))
def test_arm_loads_versioned_mix_and_keeps_unbuilt_sft_dependency(
    model, tmp_path, monkeypatch, selection, mochi_sft_presets
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "outputs"))
    sft = mochi_sft_presets["bounded"] if model == "mochi" else MIXTURE_262K_32NODE
    table = MOCHI if model == "mochi" else SNOWBALL
    tokenizer = SFT_TOKENIZER if model == "mochi" else sft.recipe.model.tokenizer_cache_key()
    data = LmDataConfig(
        tokenizer=tokenizer,
        components={"v5": DatasetComponent(cache_dir="/regional/v5", flat_cache=True)},
        train_weights={"v5": 1.0},
        auto_build_caches=False,
    )
    root = tmp_path / "v5"
    root.mkdir()
    (root / "mix.json").write_text(json.dumps(draccus.encode(data)))
    mix = PreparedData(ArtifactStep.adopt("tests/sft-v5", "2026.11.25", str(root)), "mix.json", tokenizer)
    arm = build_arm(
        sft,
        table.by_name["train-long"],
        table=table,
        data_mix=mix,
        version="2026.11.25.1",
        sft_steps=2,
        rl_steps=2,
    )
    [run] = arm.runs
    handles = graph_handles([run])
    [training] = [step for step in handles if mix.step in step.deps]
    run.lower()
    config = materialized_config(training, str(tmp_path))
    if model == "mochi":
        original = sft.recipe.train_artifact(sft.resources)
        baseline = original.build_config(StepContext.for_fingerprint(original.runtime_args.keys(), original.deps))
        assert config.optimizer == baseline.optimizer
        assert config.trainer.trainer.train_batch_size == baseline.trainer.trainer.train_batch_size
        assert config.stop_step == 393493
        train_config = config
        [export] = [step for step in run.deps if training in step.deps]
        expected_model_uri = f"{export.path(str(tmp_path))}/hf"
    else:
        train_config = config.train_config
        assert train_config.optimizer == sft.recipe.optimizer
        assert train_config.trainer.train_batch_size == sft.recipe.batch_size
        assert train_config.trainer.num_train_steps == 2
        assert training in run.deps
        expected_model_uri = f"{training.path(str(tmp_path))}/hf/step-1"
    assert train_config.data.components == data.components
    assert train_config.data.train_weights == data.train_weights
    launch = yaml.safe_load(materialized_config(run, str(tmp_path)).launch_config_yaml)
    assert launch["inputs"]["model"]["uri"] == expected_model_uri
    assert not Path(expected_model_uri).exists()
    assert launch["skyrl"]["trainer"]["hf_export_checkpoint"] == "best"
    assert launch["skyrl"]["trainer"]["max_steps"] == 2
    assert "best_checkpoint" in {callback["type"] for callback in launch["skyrl"]["trainer"]["callbacks"]}


def test_data_mix_changes_get_separate_sft_outputs(tmp_path):
    source = ArtifactStep.adopt("tests/sft-mix", "2026.11.25.1", str(tmp_path))
    first = PreparedData(source, "v5.json", MIXTURE_262K_32NODE.recipe.model.tokenizer_cache_key())
    second = replace(first, step=replace(source, version="2026.11.25.2"), file="v7.json")
    spec = MIXTURE_262K_32NODE.recipe
    v5 = spec.with_data_mix(first).trained(MIXTURE_262K_32NODE.resources).step
    v7 = spec.with_data_mix(second).trained(MIXTURE_262K_32NODE.resources).step
    assert v5.path() != v7.path()
    assert v5.fingerprint() != v7.fingerprint()
