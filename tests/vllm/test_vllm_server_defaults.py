# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.simple_evaluator import SimpleEvaluator
from marin.inference.vllm import BrokeredVllmSystemConfig
from marin.inference.vllm_server import DEFAULT_TPU_MAX_NUM_BATCHED_TOKENS, _vllm_env, resolve_model_name_or_path


def test_resolve_model_name_applies_tpu_safe_batched_token_default(monkeypatch):
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    model = ModelConfig(name="test-model", path=None, engine_kwargs={"max_model_len": 8192})

    _, resolved = resolve_model_name_or_path(model)

    assert resolved.engine_kwargs["max_num_batched_tokens"] == DEFAULT_TPU_MAX_NUM_BATCHED_TOKENS


def test_resolve_model_name_respects_explicit_batched_token_override(monkeypatch):
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)
    model = ModelConfig(
        name="test-model",
        path=None,
        engine_kwargs={"max_model_len": 8192, "max_num_batched_tokens": 2048},
    )

    _, resolved = resolve_model_name_or_path(model)

    assert resolved.engine_kwargs["max_num_batched_tokens"] == 2048


def test_resolve_model_name_skips_tpu_default_for_non_tpu_target(monkeypatch):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cuda")
    model = ModelConfig(name="test-model", path=None, engine_kwargs={"max_model_len": 8192})

    _, resolved = resolve_model_name_or_path(model)

    assert "max_num_batched_tokens" not in resolved.engine_kwargs


def test_native_vllm_env_defaults_to_tpu_target(monkeypatch):
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)

    assert _vllm_env()["VLLM_TARGET_DEVICE"] == "tpu"


def test_brokered_vllm_workers_use_vllm_extra_without_general_tpu_extra():
    config = BrokeredVllmSystemConfig(model="test-model")

    assert config.worker_environment_extras == ("vllm",)


def test_simple_evaluator_passes_resolved_engine_kwargs_to_llm(monkeypatch, tmp_path):
    calls = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            calls["llm_kwargs"] = kwargs

        def generate(self, prompts, sampling_params):
            calls["prompts"] = prompts
            calls["sampling_params"] = sampling_params
            return []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling_kwargs"] = kwargs

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    monkeypatch.delenv("VLLM_TARGET_DEVICE", raising=False)

    SimpleEvaluator().evaluate(
        ModelConfig(
            name="test-model",
            path=None,
            engine_kwargs={"max_model_len": 8192, "enforce_eager": True},
        ),
        evals=[SimpleNamespace(name="quick")],
        output_path=str(tmp_path),
    )

    assert calls["llm_kwargs"] == {
        "model": "test-model",
        "enforce_eager": True,
        "trust_remote_code": True,
        "max_model_len": 8192,
        "max_num_batched_tokens": DEFAULT_TPU_MAX_NUM_BATCHED_TOKENS,
    }
    assert calls["prompts"] == SimpleEvaluator.QUICK_TEST_PLAN.prompts
