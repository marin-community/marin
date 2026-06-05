# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RULER eval configuration."""

import pytest
from fray.cluster import ResourceConfig
from marin.execution.types import ExecutorStep

from experiments.evals.evals import _ruler_context_lengths_for_model, default_ruler_eval, ruler_effective_context_length
from experiments.evals.task_configs import RULER_MAX_GENERATION_TOKENS


def test_ruler_context_lengths_include_exact_model_length() -> None:
    """RULER lengths already include output-token room in generated samples."""
    selected_lengths = _ruler_context_lengths_for_model((4096, 8192), model_max_length=4096)

    assert selected_lengths == (4096,)


def test_ruler_context_lengths_reject_too_short_model() -> None:
    """RULER should fail fast when no benchmark length fits."""
    with pytest.raises(ValueError):
        _ruler_context_lengths_for_model((4096, 8192), model_max_length=2048)


def test_ruler_effective_context_length_respects_sliding_window() -> None:
    """Sliding-window models should use the attention window for RULER."""
    assert ruler_effective_context_length(max_seq_len=8192) == 8192
    assert ruler_effective_context_length(max_seq_len=8192, sliding_window=4096) == 4096
    assert ruler_effective_context_length(max_seq_len=4096, sliding_window=8192) == 4096


def test_default_ruler_eval_sets_metadata_and_context_kwargs() -> None:
    """RULER needs lm-eval metadata plus vLLM and lm-eval length settings."""
    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=8192,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        discover_latest_checkpoint=False,
    )

    config = step.config
    assert config.evaluator == "lm_evaluation_harness"
    assert config.engine_kwargs == {
        "max_model_len": 8192,
        "max_length": 8192,
        "max_gen_toks": RULER_MAX_GENERATION_TOKENS,
    }
    assert len(config.evals) == 1
    assert config.evals[0].name == "ruler"
    assert config.evals[0].num_fewshot == 0
    assert config.evals[0].metadata == {"max_seq_lengths": [4096, 8192]}
    assert step.fn.env_vars["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"


def test_default_ruler_eval_does_not_mutate_engine_kwargs() -> None:
    """Caller-provided engine kwargs should be copied before adding RULER lengths."""
    engine_kwargs = {"max_model_len": 8192}

    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=8192,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        engine_kwargs=engine_kwargs,
        discover_latest_checkpoint=False,
    )

    assert engine_kwargs == {"max_model_len": 8192}
    assert step.config.engine_kwargs == {
        "max_model_len": 8192,
        "max_length": 8192,
        "max_gen_toks": RULER_MAX_GENERATION_TOKENS,
    }


def test_default_ruler_eval_forwards_ruler_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """RULER workers should receive selected eval runtime env vars."""
    monkeypatch.setenv("HF_TOKEN", "hf-token")

    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=4096,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        discover_latest_checkpoint=False,
    )

    assert step.fn.env_vars["HF_TOKEN"] == "hf-token"
    assert step.fn.env_vars["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] == "1"


def test_default_ruler_eval_sets_tokenizer() -> None:
    """RULER can use a tokenizer separate from the checkpoint path."""
    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=8192,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        tokenizer="stanford-crfm/marin-tokenizer",
        discover_latest_checkpoint=False,
    )

    assert step.config.engine_kwargs == {
        "max_model_len": 8192,
        "max_length": 8192,
        "max_gen_toks": RULER_MAX_GENERATION_TOKENS,
        "tokenizer": "stanford-crfm/marin-tokenizer",
    }


def test_default_ruler_eval_rejects_conflicting_tokenizers() -> None:
    """Tokenizer should have one clear source."""
    with pytest.raises(ValueError):
        default_ruler_eval(
            "gs://example/checkpoints/model/hf/step-100",
            model_max_length=8192,
            resource_config=ResourceConfig.with_cpu(cpu=1),
            engine_kwargs={"tokenizer": "existing-tokenizer"},
            tokenizer="different-tokenizer",
            discover_latest_checkpoint=False,
        )


def test_default_ruler_eval_rejects_native_grug_step() -> None:
    """Raw Grug train steps are not vLLM-loadable RULER inputs."""

    class GrugLaunchConfig:
        pass

    GrugLaunchConfig.__module__ = "experiments.grug.moe.launch"
    grug_step = ExecutorStep(name="grug/test", fn=None, config=GrugLaunchConfig())

    with pytest.raises(ValueError):
        default_ruler_eval(
            grug_step,
            model_max_length=4096,
            resource_config=ResourceConfig.with_cpu(cpu=1),
            discover_latest_checkpoint=False,
        )
    with pytest.raises(ValueError):
        default_ruler_eval(
            grug_step.as_input_name(),
            model_max_length=4096,
            resource_config=ResourceConfig.with_cpu(cpu=1),
            discover_latest_checkpoint=False,
        )


def test_default_ruler_eval_reserves_chat_template_room() -> None:
    """Chat wrapping adds tokens after RULER generates raw-length samples."""
    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=8192,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        apply_chat_template=True,
        chat_template_token_buffer=256,
        discover_latest_checkpoint=False,
    )

    config = step.config
    assert config.engine_kwargs == {
        "max_model_len": 4352,
        "max_length": 4352,
        "max_gen_toks": RULER_MAX_GENERATION_TOKENS,
    }
    assert config.evals[0].metadata == {"max_seq_lengths": [4096]}


def test_default_ruler_eval_can_select_subtasks() -> None:
    """RULER can run a small explicit subtask set."""
    step = default_ruler_eval(
        "gs://example/checkpoints/model/hf/step-100",
        model_max_length=8192,
        resource_config=ResourceConfig.with_cpu(cpu=1),
        task_names=("niah_single_1", "ruler_vt"),
        max_gen_toks=64,
        discover_latest_checkpoint=False,
    )

    config = step.config
    assert [task.name for task in config.evals] == ["niah_single_1", "ruler_vt"]
    assert [task.task_alias for task in config.evals] == ["niah_single_1", "ruler_vt"]
    assert config.evals[0].metadata == {"max_seq_lengths": [4096, 8192]}
    assert config.engine_kwargs["max_gen_toks"] == 64


def test_default_ruler_eval_rejects_short_explicit_context_kwargs() -> None:
    """Explicit context kwargs should not be silently enlarged."""
    with pytest.raises(ValueError):
        default_ruler_eval(
            "gs://example/checkpoints/model/hf/step-100",
            model_max_length=8192,
            resource_config=ResourceConfig.with_cpu(cpu=1),
            engine_kwargs={"max_model_len": 4096},
            discover_latest_checkpoint=False,
        )
