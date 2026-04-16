# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tier A/B: Harbor routing decisions — local vLLM vs external API.

No Harbor runtime imported here; all tests exercise the pure-Python helpers
that decide how `RunningModel` maps to Harbor's `model_name` + `agent_kwargs`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from marin.evaluation.api import HarborRun
from marin.evaluation.evaluators.harbor_evaluator import (
    _DEFAULT_HOSTED_VLLM_MODEL_INFO,
    _build_model_name_and_agent_kwargs,
    sanitize_hosted_vllm_canonical_name,
)
from marin.inference.model_launcher import LITELLM_PROVIDER_URL, OpenAIEndpoint, RunningModel


def _local_vllm_model(name: str = "served-model") -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(url="http://127.0.0.1:8000/v1", model=name),
        tokenizer_ref="hf/tok",
    )


def _external_api_model(name: str = "claude-opus-4") -> RunningModel:
    return RunningModel(
        endpoint=OpenAIEndpoint(url=LITELLM_PROVIDER_URL, model=name),
        tokenizer_ref="",
    )


def test_local_vllm_model_routes_via_hosted_vllm():
    model_name, kwargs = _build_model_name_and_agent_kwargs(_local_vllm_model(), agent_kwargs={})
    assert model_name == "hosted_vllm/served-model"
    assert kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    # Default `model_info` seeded only if caller didn't override.
    assert kwargs["model_info"] == _DEFAULT_HOSTED_VLLM_MODEL_INFO


def test_local_vllm_preserves_caller_api_base_override():
    model_name, kwargs = _build_model_name_and_agent_kwargs(
        _local_vllm_model(), agent_kwargs={"api_base": "http://override/v1"}
    )
    assert model_name == "hosted_vllm/served-model"
    assert kwargs["api_base"] == "http://override/v1"


def test_local_vllm_preserves_caller_model_info_override():
    custom_info = {"max_input_tokens": 4096}
    _, kwargs = _build_model_name_and_agent_kwargs(_local_vllm_model(), agent_kwargs={"model_info": custom_info})
    assert kwargs["model_info"] == custom_info


def test_external_api_model_routes_by_model_name_only():
    model_name, kwargs = _build_model_name_and_agent_kwargs(_external_api_model(), agent_kwargs={})
    assert model_name == "claude-opus-4"
    assert "api_base" not in kwargs
    # No `model_info` default seeded in external mode — LiteLLM knows the provider's limits.
    assert "model_info" not in kwargs


def test_external_api_model_drops_caller_api_base_to_avoid_misroute():
    # A caller that copy-pasted agent_kwargs with an api_base from a prior vLLM
    # run would misroute LiteLLM — route-via-name must win.
    model_name, kwargs = _build_model_name_and_agent_kwargs(
        _external_api_model(), agent_kwargs={"api_base": "http://stale-vllm/v1"}
    )
    assert model_name == "claude-opus-4"
    assert "api_base" not in kwargs


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("hello-world", "hello-world"),
        ("hello_world", "hello_world"),
        ("hello world", "hello_world"),
        ("x" * 64, "x" * 64),  # exactly 64 passes
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama_Llama-3.1-8B-Instruct"),
    ],
)
def test_sanitize_canonical_name_accepts_harbor_pattern(raw, expected):
    assert sanitize_hosted_vllm_canonical_name(raw) == expected


def test_sanitize_canonical_name_shortens_very_long_names():
    name = "x" * 200
    out = sanitize_hosted_vllm_canonical_name(name)
    assert len(out) <= 64
    # Different inputs must produce different outputs.
    assert sanitize_hosted_vllm_canonical_name("a" * 200) != out


def test_sanitize_canonical_name_fallback_for_empty():
    assert sanitize_hosted_vllm_canonical_name("") == "model"
    assert sanitize_hosted_vllm_canonical_name("___") == "model"


def test_harbor_run_dataclass_frozen():
    run = HarborRun(
        evals=[],
        output_path="/tmp/out",
        dataset="aime",
        version="1.0",
        agent="claude-code",
        n_concurrent=4,
    )
    with pytest.raises(FrozenInstanceError):
        run.dataset = "other"  # type: ignore[misc]


def test_harbor_run_default_env_is_local():
    run = HarborRun(
        evals=[],
        output_path="/tmp/out",
        dataset="aime",
        version="1.0",
        agent="claude-code",
        n_concurrent=4,
    )
    assert run.env == "local"
    assert run.agent_kwargs == {}
    assert run.model_info is None
