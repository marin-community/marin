# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral wiring tests for the five-way Snowball SFT campaign."""

import pytest
from levanter.data.text.formats import ChatLmDatasetFormat, LossWeightTransform, PrebuiltLmDatasetFormat
from levanter.models.snowball import SnowballConfig
from marin.execution.lazy import materialized_config

from experiments.sft.configs import snowball_lce_final
from experiments.sft.launcher import HFModel

_PREFIX = "s3://test-prefix"
_VERSION = "2026.09.08.99"


@pytest.fixture
def local_base(monkeypatch):
    config = SnowballConfig(max_seq_len=262_144, qk_mult=1.5703)
    ref = "open-athena/test-snowball@deadbeef"
    source = HFModel(
        model_ref=ref,
        tokenizer_path=ref,
        model_type="snowball",
        model_config=config,
        trainer_mesh=snowball_lce_final._TRAIN_MESH,
        use_explicit_mesh_axes=True,
    )
    monkeypatch.setattr(snowball_lce_final, "_base_model", lambda _base: (source, config, ref))
    return config, ref


def test_opencode_is_packed_fixed_eot_child_of_thinking(local_base):
    model_config, tokenizer = local_base
    step = snowball_lce_final.build_opencode("qk157", _VERSION)
    cache, thinking = step.deps

    assert cache.name == "tokenized/grug-a2b-agentic-sft-eot"
    assert thinking.name.endswith("/qk157/thinking")
    assert any(dep.name.endswith("/qk157/chat") for dep in thinking.deps)

    pod = materialized_config(step, _PREFIX)
    train = pod.train_config
    component = train.data.components["grug_a2b_agentic_sft_eot"]
    assert train.trainer.num_train_steps == 1_888
    assert train.initialize_model_from_checkpoint_path.endswith("/qk157/thinking/2026.09.08.99/checkpoints")
    assert train.model == model_config
    assert train.data.tokenizer == tokenizer
    assert component.pack is True
    assert component.packed_slice_strategy == "right"
    assert isinstance(component.format, PrebuiltLmDatasetFormat)
    assert component.format.loss_weight_transform is LossWeightTransform.SHIFT_LEFT


def test_nemotron_is_independent_thinking_child_with_historical_cache(local_base):
    model_config, tokenizer = local_base
    step = snowball_lce_final.build_nemotron_terminal("qk157", _VERSION)
    cache, thinking = step.deps

    assert cache.adopt_source == snowball_lce_final._NEMOTRON_CACHE_SOURCE
    assert thinking.name.endswith("/qk157/thinking")
    assert not any("opencode" in dep.name for dep in step.deps)

    pod = materialized_config(step, _PREFIX)
    train = pod.train_config
    component = train.data.components["nemotron_terminal_full"]
    assert train.trainer.num_train_steps == 1_888
    assert train.initialize_model_from_checkpoint_path.endswith("/qk157/thinking/2026.09.08.99/checkpoints")
    assert train.model == model_config
    assert train.data.tokenizer == tokenizer
    assert isinstance(component.format, ChatLmDatasetFormat)
    assert component.format.chat_template == snowball_lce_final.MARIN_CHAT_TEMPLATE
    assert component.packed_slice_strategy == "left"


def test_nemotron_file_selection_is_frozen():
    files = snowball_lce_final._NEMOTRON_PARQUET_FILES
    assert len(files) == 29
    assert len(set(files)) == 29
    assert "dataset_adapters/swe.parquet" in files
    assert "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet" in files


def test_campaign_uses_canonical_optimizer_registration():
    assert snowball_lce_final.GrugMoeAdamHConfig.__module__ == "experiments.grug.moe.optimizer"
