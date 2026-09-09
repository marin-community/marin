# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioral wiring tests for the five-way Snowball SFT campaign."""

import pytest
from levanter.data.text.formats import ChatLmDatasetFormat, LossWeightTransform, PrebuiltLmDatasetFormat
from marin.execution.lazy import StepContext, materialized_config

from experiments.sft import launcher as sft_launcher
from experiments.sft.configs import snowball_lce_final
from experiments.sft.validate_snowball_caches import _EXPECTED_OPENCODE_STEPS, _OPENCODE_EPOCHS

_PREFIX = "s3://test-prefix"
_VERSION = "2026.09.08.99"


@pytest.fixture
def local_base():
    model, conversion = snowball_lce_final._base_model("qk157")
    return model.conversion.model, conversion


def test_opencode_is_packed_fixed_eot_child_of_thinking(local_base):
    model_config, conversion = local_base
    step = snowball_lce_final.build_opencode("qk157", _VERSION)
    cache = step.deps[0]
    thinking = next(dep for dep in step.deps if dep.name.endswith("/qk157/thinking"))

    assert cache.name == "tokenized/grug-a2b-agentic-sft-eot"
    assert thinking.name.endswith("/qk157/thinking")
    assert any(dep.name.endswith("/qk157/chat") for dep in thinking.deps)

    train = materialized_config(step, _PREFIX)
    component = train.data.components["grug_a2b_agentic_sft_eot"]
    assert train.steps == 1_888
    assert train.init_from_path.endswith("/qk157/thinking/2026.09.08.99/checkpoints")
    assert train.model == model_config
    assert train.data.tokenizer == f"{_PREFIX}/{conversion.step.name}/{conversion.step.version}"
    assert component.pack is True
    assert component.packed_slice_strategy == "right"
    assert isinstance(component.format, PrebuiltLmDatasetFormat)
    assert component.format.loss_weight_transform is LossWeightTransform.SHIFT_LEFT


def test_nemotron_is_independent_thinking_child_with_historical_cache(local_base):
    model_config, conversion = local_base
    step = snowball_lce_final.build_nemotron_terminal("qk157", _VERSION)
    cache = step.deps[0]
    thinking = next(dep for dep in step.deps if dep.name.endswith("/qk157/thinking"))

    assert cache.adopt_source == snowball_lce_final._NEMOTRON_CACHE_SOURCE
    assert thinking.name.endswith("/qk157/thinking")
    assert not any("opencode" in dep.name for dep in step.deps)

    train = materialized_config(step, _PREFIX)
    component = train.data.components["nemotron_terminal_full"]
    assert train.steps == 1_888
    assert train.init_from_path.endswith("/qk157/thinking/2026.09.08.99/checkpoints")
    assert train.model == model_config
    assert train.data.tokenizer == f"{_PREFIX}/{conversion.step.name}/{conversion.step.version}"
    assert isinstance(component.format, ChatLmDatasetFormat)
    assert component.format.chat_template == snowball_lce_final.MARIN_CHAT_TEMPLATE
    assert component.packed_slice_strategy == "left"


def test_nemotron_file_selection_is_frozen():
    files = snowball_lce_final._NEMOTRON_PARQUET_FILES
    assert len(files) == 29
    assert len(set(files)) == 29
    assert "dataset_adapters/swe.parquet" in files
    assert "synthetic_tasks/skill_based/medium/model_training/data_filtered.parquet" in files


def test_training_topology_matches_historical_grug_recipe(local_base):
    train = materialized_config(snowball_lce_final.build_smoke("qk157", _VERSION), _PREFIX)

    assert train.expert_parallel == 8
    assert train.grug_trainer.model_axis_size == 1
    assert train.batch_size == 64


def test_all_five_base_revisions_are_immutable():
    assert len(snowball_lce_final._BASE_REVISIONS) == 5
    for repository, revision in snowball_lce_final._BASE_REVISIONS.values():
        assert repository.startswith("open-athena/snowball-67b-a2b-base-")
        assert revision is not None
        assert len(revision) == 40


def test_each_base_uses_a_pinned_conversion_for_weights_and_tokenizer():
    model, conversion = snowball_lce_final._base_model("qk157")

    config = materialized_config(conversion.step, _PREFIX)
    assert config.hf_id == "open-athena/snowball-67b-a2b-base-262k-qk157"
    assert len(config.hf_revision) == 40
    assert model.conversion is conversion
    assert model.init_deps() == (conversion.step,)
    assert model.resolve_tokenizer(StepContext.for_fingerprint((), (conversion.step,))) == (
        f"{conversion.step.name}@{conversion.step.version}"
    )


def test_data_stage_builds_and_gates_both_shared_prefix_caches(local_base):
    step = snowball_lce_final.build_prefix_caches("qk157", _VERSION)
    chat_cache, thinking_cache = step.deps

    assert chat_cache.name.startswith("tokenized/wildchat_386k-chat-")
    assert thinking_cache.name.startswith("tokenized/nemotron_science_think-chat-")
    assert chat_cache.version == thinking_cache.version

    config = step.build_config(StepContext.for_fingerprint((), step.deps))
    assert config.chat_tokens == 257 * 32_768 * 64
    assert config.thinking_tokens == 630 * 32_768 * 64

    class MismatchedContext:
        is_fingerprint = False

        @staticmethod
        def resolved(_cache):
            return type("ResolvedCache", (), {"num_train_tokens": 1})()

    with pytest.raises(ValueError, match=r"chat cache has 1 tokens \(1 steps\); expected 257"):
        snowball_lce_final._checked_cache_tokens(
            MismatchedContext(), chat_cache, stage="chat", expected_steps=257  # type: ignore[arg-type]
        )


def test_smoke_reload_strictly_initializes_from_native_smoke(local_base):
    model_config, conversion = local_base
    step = snowball_lce_final.build_smoke_reload("qk157", _VERSION)

    assert step.name.endswith("/qk157/hf-smoke-reload")
    assert any(dep.name.endswith("/qk157/hf-smoke") for dep in step.deps)
    train = materialized_config(step, _PREFIX)
    assert train.steps == 1
    assert train.init_from_path.endswith("/qk157/hf-smoke/2026.09.08.99/checkpoints")
    assert train.model == model_config
    assert train.data.tokenizer == f"{_PREFIX}/{conversion.step.name}/{conversion.step.version}"


def test_cache_preflight_matches_frozen_opencode_length():
    assert _OPENCODE_EPOCHS == 5
    assert _EXPECTED_OPENCODE_STEPS == 1_888


def test_all_stage_shares_one_thinking_parent(local_base):
    step = snowball_lce_final.build_all("qk157", _VERSION)
    opencode, nemotron = step.deps

    assert opencode.name.endswith("/qk157/opencode")
    assert nemotron.name.endswith("/qk157/nemotron-terminal")
    opencode_thinking = next(dep for dep in opencode.deps if dep.name.endswith("/qk157/thinking"))
    nemotron_thinking = next(dep for dep in nemotron.deps if dep.name.endswith("/qk157/thinking"))
    assert opencode_thinking is nemotron_thinking

    manifest = materialized_config(step, _PREFIX)
    assert manifest.base == "qk157"
    assert manifest.opencode_path.endswith("/qk157/opencode/2026.09.08.99")
    assert manifest.nemotron_terminal_path.endswith("/qk157/nemotron-terminal/2026.09.08.99")


def test_prefix_stages_gate_historical_epoch_lengths(local_base, monkeypatch):
    specs = []

    def capture(spec, _resources):
        specs.append(spec)
        return object()

    monkeypatch.setattr(snowball_lce_final, "sft_step", capture)
    snowball_lce_final.build_thinking("qk157", _VERSION)

    assert [(spec.num_train_epochs, spec.expected_epoch_steps) for spec in specs] == [(1, 257), (1, 630)]

    class MismatchedContext:
        is_fingerprint = False

        @staticmethod
        def resolved(_cache):
            return type("ResolvedCache", (), {"num_train_tokens": 1})()

    with pytest.raises(ValueError, match="resolve to 1 steps; expected 257"):
        sft_launcher._resolve_epoch_steps(MismatchedContext(), specs[0], object())  # type: ignore[arg-type]
