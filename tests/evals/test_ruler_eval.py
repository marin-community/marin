# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RULER eval configuration."""

import pytest
from fray.cluster import ResourceConfig

from experiments.evals.evals import _ruler_context_lengths_for_model, default_ruler_eval


def test_ruler_context_lengths_include_exact_model_length() -> None:
    """RULER lengths already include output-token room in generated samples."""
    selected_lengths = _ruler_context_lengths_for_model((4096, 8192), model_max_length=4096)

    assert selected_lengths == (4096,)


def test_ruler_context_lengths_reject_too_short_model() -> None:
    """RULER should fail fast when no benchmark length fits."""
    with pytest.raises(ValueError):
        _ruler_context_lengths_for_model((4096, 8192), model_max_length=2048)


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
    assert config.engine_kwargs == {"max_model_len": 8192, "max_length": 8192}
    assert len(config.evals) == 1
    assert config.evals[0].name == "ruler"
    assert config.evals[0].num_fewshot == 0
    assert config.evals[0].metadata == {"max_seq_lengths": [4096, 8192]}


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
    assert step.config.engine_kwargs == {"max_model_len": 8192, "max_length": 8192}


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
