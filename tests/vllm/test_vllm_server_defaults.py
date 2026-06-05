# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.evaluation.evaluators.evaluator import ModelConfig
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
