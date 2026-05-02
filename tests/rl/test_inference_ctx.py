# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for InferenceContext utilities and chat template handling."""

import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from levanter.inference.openai import ChatMessage
from marin.rl.environments.inference_ctx import (
    MODEL_MAPPINGS,
    MODEL_TRANSPOSE_KEYS,
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    VLLMSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.inflight.worker import WorkerExtension
from marin.rl.environments.inference_ctx.openai_compat import OpenAICompatClient
from marin.rl.environments.inference_ctx.vllm import InferenceMode
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage
from transformers import AutoTokenizer

_LLAMA3_MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"


@dataclass
class DummyInferenceServer:
    """Minimal inference server for testing."""

    host: str = "localhost"
    port: int = 8000

    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def config(self):
        @dataclass
        class Config:
            model_name: str = "test-model"

        return Config()


@pytest.fixture(scope="session")
def llama3_tokenizer():
    """Llama 3 tokenizer — skips the test when HF is unreachable."""
    try:
        return AutoTokenizer.from_pretrained(_LLAMA3_MODEL_ID)
    except Exception:
        pytest.skip(f"Llama-3 tokenizer not accessible ({_LLAMA3_MODEL_ID})")


@pytest.fixture
def dummy_server():
    return DummyInferenceServer()


@pytest.fixture
def inference_ctx(llama3_tokenizer, dummy_server):
    return LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=llama3_tokenizer,
            stop_tokens=None,
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )


def create_choice_with_logprobs(tokenizer, response_text: str, logprobs_values: list[float] | None = None) -> Choice:
    """Create a Choice with proper BPE tokens and logprobs."""
    # Tokenize the response to get real token IDs
    token_ids = tokenizer.encode(response_text, add_special_tokens=False)

    if logprobs_values is None:
        logprobs_values = [-1.0] * len(token_ids)

    # Convert token IDs back to BPE tokens (preserves Ġ prefixes)
    logprobs_content = []
    for token_id, logprob in zip(token_ids, logprobs_values, strict=True):
        bpe_token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs_content.append(
            ChatCompletionTokenLogprob(
                token=bpe_token,
                logprob=logprob,
                bytes=list(bpe_token.encode("utf-8")),
                top_logprobs=[],
            )
        )

    return Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content=response_text),
        logprobs=ChoiceLogprobs(content=logprobs_content),
    )


def create_chat_completion(response_text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_text),
                logprobs=ChoiceLogprobs(content=[]),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=1, prompt_tokens=1, total_tokens=2),
    )


def test_apply_chat_template(llama3_tokenizer):
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Bigger: 87 or 3? Just the number:"),
    ]
    dict_messages = [msg.model_dump(exclude_none=True) for msg in messages]
    tokens = llama3_tokenizer.apply_chat_template(dict_messages, tokenize=True, add_generation_prompt=True)
    decoded = llama3_tokenizer.decode(tokens)

    assert "helpful assistant" in decoded
    assert "Bigger: 87 or 3?" in decoded
    assert "<|start_header_id|>assistant<|end_header_id|>" in decoded  # Llama 3's generation prompt marker


def test_bpe_round_trip_various_texts(llama3_tokenizer):
    """Validate BPE round-trip for diverse text patterns."""
    for text in ["!!}", "Hello world", "  spaces  ", "123", "\n\n"]:
        for token_id in llama3_tokenizer.encode(text, add_special_tokens=False):
            token_str = llama3_tokenizer.convert_ids_to_tokens(token_id)
            assert llama3_tokenizer.convert_tokens_to_ids(token_str) == token_id


def test_tokenize_prompt_adds_special_tokens(inference_ctx, llama3_tokenizer):
    """Test that tokenize_prompt uses chat template and adds special tokens."""
    prompt = "What is 2+2?"

    # InferenceContext uses chat template
    ctx_tokens = inference_ctx.tokenize_prompt(prompt)

    # Direct encode without template
    plain_tokens = llama3_tokenizer.encode(prompt, add_special_tokens=False)

    # Chat template should add tokens (system prompt markers, instruction markers, etc.)
    assert len(ctx_tokens) > len(plain_tokens)

    # Verify it's using the chat template by checking decoded output
    decoded = llama3_tokenizer.decode(ctx_tokens)
    assert "<|start_header_id|>user<|end_header_id|>" in decoded  # Llama 3 instruction markers
    assert prompt in decoded


def test_tokenize_prompt_supports_message_lists(inference_ctx, llama3_tokenizer):
    messages = [
        {"role": "system", "content": "You are careful."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    tokens = inference_ctx.tokenize_prompt(messages)
    decoded = llama3_tokenizer.decode(tokens)

    assert "You are careful." in decoded
    assert "What is 2+2?" in decoded
    assert "<|start_header_id|>assistant<|end_header_id|>" in decoded


def test_tokenize_prompt_fallback_no_template(gpt2_tokenizer, dummy_server):
    """Test fallback when tokenizer has no chat template."""
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=None,
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )

    prompt = "Test prompt"
    tokens = ctx.tokenize_prompt(prompt)

    # Should fallback to "user: Test prompt" format
    decoded = gpt2_tokenizer.decode(tokens)
    assert "user:" in decoded
    assert prompt in decoded


def test_tokenize_prompt_fallback_message_list_no_template(gpt2_tokenizer):
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=None,
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Test prompt"},
    ]
    tokens = ctx.tokenize_prompt(messages)

    decoded = gpt2_tokenizer.decode(tokens)
    assert "system: You are helpful." in decoded
    assert "user: Test prompt" in decoded


def test_response_tokens_from_choice(inference_ctx, llama3_tokenizer):
    """Test extracting token IDs from Choice using BPE round-trip."""
    response_text = "The answer is 42"
    choice = create_choice_with_logprobs(llama3_tokenizer, response_text)

    tokens = inference_ctx.response_tokens_from_choice(choice)

    # Should match tokenizer's encoding
    expected_tokens = llama3_tokenizer.encode(response_text, add_special_tokens=False)
    np.testing.assert_array_equal(tokens, expected_tokens)


def test_logprobs_from_choice(inference_ctx, llama3_tokenizer):
    """Test extracting logprobs array from Choice."""
    response_text = "The answer"
    choice = create_choice_with_logprobs(llama3_tokenizer, response_text)

    logprobs = inference_ctx.logprobs_from_choice(choice)

    # Should have same length as tokenized response
    expected_length = len(llama3_tokenizer.encode(response_text, add_special_tokens=False))
    assert logprobs.dtype == np.float32
    assert len(logprobs) == expected_length


def test_missing_logprobs_raises(inference_ctx):
    """Test that missing logprobs raises ValueError."""
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content="test"),
        logprobs=None,
    )

    with pytest.raises(ValueError, match="missing logprobs"):
        inference_ctx.response_tokens_from_choice(choice)

    with pytest.raises(ValueError, match="missing logprobs"):
        inference_ctx.logprobs_from_choice(choice)


def test_create_rollout_from_choice_end_to_end(inference_ctx, llama3_tokenizer):
    """Test full rollout construction from prompt and choice."""
    prompt = "What is 2+2?"
    response_text = "The answer is 4"
    logprobs_values = [-0.5, -1.0, -0.8, -0.3, -1.2]
    reward = 1.0

    choice = create_choice_with_logprobs(llama3_tokenizer, response_text, logprobs_values)

    rollout = inference_ctx.create_rollout_from_choice(
        prompt=prompt,
        choice=choice,
        env_name="math_env",
        env_example_id="ex_001",
        reward=reward,
        temperature=1.0,
    )

    # Verify metadata
    assert rollout.env_name == "math_env"
    assert rollout.env_example_id == "ex_001"
    assert rollout.episode_reward == reward

    # Verify prompt tokens use chat template (longer than plain encoding)
    plain_prompt_tokens = llama3_tokenizer.encode(prompt, add_special_tokens=False)
    assert len(rollout.prompt_tokens) > len(plain_prompt_tokens)

    # Verify response tokens match expected encoding
    expected_response_tokens = llama3_tokenizer.encode(response_text, add_special_tokens=False)
    np.testing.assert_array_equal(rollout.response_tokens, expected_response_tokens)

    # Verify logprobs
    np.testing.assert_array_almost_equal(rollout.response_logprobs, logprobs_values)

    # Verify token rewards
    assert len(rollout.token_rewards) == len(expected_response_tokens)
    np.testing.assert_array_equal(rollout.token_rewards, np.full(len(expected_response_tokens), reward))
    np.testing.assert_array_equal(rollout.response_loss_mask, np.ones(len(expected_response_tokens), dtype=np.float32))


def test_vllm_inference_context_uses_canonical_model_name(monkeypatch):
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: SimpleNamespace(get_vocab=lambda: {}),
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda model_name, _tokenizer: model_name),
    )

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=1024,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
        )
    )

    assert ctx.model_name == "gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f"
    assert ctx.canonical_model_name == "meta-llama/Llama-3.1-8B-Instruct"
    assert ctx.renderer == "meta-llama/Llama-3.1-8B-Instruct"


def test_vllm_sync_engine_receives_kv_cache_metrics_flag(monkeypatch):
    calls = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(vLLMInferenceContext, "_patch_tpu_inference_registry", staticmethod(lambda: None))
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.LLM", _FakeLLM)

    config = vLLMInferenceContextConfig(
        model_name="test-model",
        max_model_len=1024,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        sampling_params=VLLMSamplingConfig(),
        kv_cache_metrics=True,
    )

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["kv_cache_metrics"] is True
    assert calls["seed"] == 0


def test_vllm_sync_engine_receives_engine_seed(monkeypatch):
    calls = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(vLLMInferenceContext, "_patch_tpu_inference_registry", staticmethod(lambda: None))
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.LLM", _FakeLLM)

    config = vLLMInferenceContextConfig(
        model_name="test-model",
        max_model_len=1024,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        sampling_params=VLLMSamplingConfig(),
        seed=1234,
    )

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["seed"] == 1234


def test_vllm_async_engine_receives_kv_cache_metrics_flag(monkeypatch):
    calls = {}

    class _FakeSyncVLLMWrapper:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(vLLMInferenceContext, "_patch_tpu_inference_registry", staticmethod(lambda: None))
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SyncVLLMWrapper", _FakeSyncVLLMWrapper)

    config = vLLMInferenceContextConfig(
        model_name="test-model",
        max_model_len=1024,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        sampling_params=VLLMSamplingConfig(),
        mode=InferenceMode.ASYNC,
        kv_cache_metrics=True,
    )

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["kv_cache_metrics"] is True
    assert calls["seed"] == 0


def test_vllm_async_engine_receives_engine_seed(monkeypatch):
    calls = {}

    class _FakeSyncVLLMWrapper:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(vLLMInferenceContext, "_patch_tpu_inference_registry", staticmethod(lambda: None))
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SyncVLLMWrapper", _FakeSyncVLLMWrapper)

    config = vLLMInferenceContextConfig(
        model_name="test-model",
        max_model_len=1024,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        sampling_params=VLLMSamplingConfig(),
        mode=InferenceMode.ASYNC,
        seed=1234,
    )

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["seed"] == 1234


def test_levanter_openai_client_stays_native(inference_ctx, dummy_server):
    inference_ctx._inference_server = dummy_server

    client = inference_ctx.openai_client()

    assert isinstance(client, AsyncOpenAI)
    assert str(client.base_url).endswith("/v1/")


def test_vllm_openai_client_delegates_to_batch_completions(monkeypatch):
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: SimpleNamespace(get_vocab=lambda: {}),
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda model_name, _tokenizer: model_name),
    )

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="test-model",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(temperature=0.2, top_k=13),
        )
    )

    completion = create_chat_completion("hello")
    completion.choices[0].prompt_token_ids = [11, 12, 13]
    completion.choices[0].response_token_ids = [21, 22]
    captured = {}

    def _fake_batch_completions(*, prompts, temperature, n, max_tokens, top_k, stop, system_prompt):
        captured.update(
            prompts=prompts,
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            top_k=top_k,
            stop=stop,
            system_prompt=system_prompt,
        )
        return [completion]

    monkeypatch.setattr(ctx, "batch_completions", _fake_batch_completions)

    client = ctx.openai_client()
    result = asyncio.run(
        client.chat.completions.create(
            model="marin-model",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Solve 2+2"},
            ],
            temperature=0.7,
            n=2,
            max_completion_tokens=32,
            stop=["<stop>"],
            extra_body={"top_k": 17, "return_tokens_as_token_ids": True},
            logprobs=True,
            top_logprobs=1,
        )
    )

    assert isinstance(client, OpenAICompatClient)
    assert result is completion
    assert captured == {
        "prompts": [[{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Solve 2+2"}]],
        "temperature": 0.7,
        "n": 2,
        "max_tokens": 32,
        "top_k": 17,
        "stop": ["<stop>"],
        "system_prompt": None,
    }
    assert result.choices[0].prompt_token_ids == [11, 12, 13]
    assert result.choices[0].response_token_ids == [21, 22]


def test_vllm_openai_client_rejects_unsupported_kwargs():
    class _DummyCtx:
        def batch_completions(self, **_kwargs):
            raise AssertionError("batch_completions should not be called for unsupported kwargs")

    client = OpenAICompatClient(_DummyCtx())

    with pytest.raises(NotImplementedError, match="Tool-enabled verifier environments"):
        asyncio.run(
            client.chat.completions.create(
                model="marin-model",
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function"}],
            )
        )

    with pytest.raises(NotImplementedError, match="Unsupported OpenAI compatibility extra_body keys"):
        asyncio.run(
            client.chat.completions.create(
                model="marin-model",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"min_p": 0.1},
            )
        )

    with pytest.raises(NotImplementedError, match="Unsupported OpenAI compatibility kwargs"):
        asyncio.run(
            client.chat.completions.create(
                model="marin-model",
                messages=[{"role": "user", "content": "hello"}],
                response_format={"type": "json_object"},
            )
        )


def test_vllm_openai_client_completion_endpoint_is_not_supported():
    client = OpenAICompatClient(SimpleNamespace())

    with pytest.raises(NotImplementedError, match="Completion-format requests are not supported"):
        asyncio.run(client.completions.create(model="marin-model", prompt="hello"))


def test_worker_extension_uses_public_sync_weights():
    calls = {}

    class _FakeWorker:
        def sync_weights(self, new_state, *, mappings, transpose_keys, reshard_fn):
            calls["new_state"] = new_state
            calls["mappings"] = mappings
            calls["transpose_keys"] = transpose_keys
            calls["reshard_fn"] = reshard_fn

    serialized_state = {
        "model.layers.0.input_layernorm.weight": (
            np.zeros((2,), dtype=np.float32).tobytes(),
            "float32",
            (2,),
        ),
    }

    WorkerExtension.update_weight(_FakeWorker(), serialized_state, "meta-llama/Llama-3.1-8B-Instruct")

    assert hasattr(calls["new_state"], "flat_state")
    assert calls["mappings"] == MODEL_MAPPINGS["meta-llama/Llama-3.1-8B-Instruct"]
    assert calls["transpose_keys"] == MODEL_TRANSPOSE_KEYS["meta-llama/Llama-3.1-8B-Instruct"]
    assert calls["reshard_fn"] is None


def test_patch_tpu_inference_registry_registers_mistral_alias(monkeypatch):
    registry = {}

    def register_model(name, cls):
        registry[name] = cls

    tpu_inference_mod = ModuleType("tpu_inference")
    models_mod = ModuleType("tpu_inference.models")
    common_mod = ModuleType("tpu_inference.models.common")
    jax_mod = ModuleType("tpu_inference.models.jax")
    qwen2_mod = ModuleType("tpu_inference.models.jax.qwen2")
    llama3_mod = ModuleType("tpu_inference.models.jax.llama3")

    class FakeQwen2ForCausalLM:
        pass

    class FakeLlamaForCausalLM:
        pass

    common_mod.model_loader = SimpleNamespace(
        _MODEL_REGISTRY=registry,
        register_model=register_model,
    )
    qwen2_mod.Qwen2ForCausalLM = FakeQwen2ForCausalLM
    llama3_mod.LlamaForCausalLM = FakeLlamaForCausalLM

    monkeypatch.setitem(sys.modules, "tpu_inference", tpu_inference_mod)
    monkeypatch.setitem(sys.modules, "tpu_inference.models", models_mod)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.common", common_mod)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.jax", jax_mod)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.jax.qwen2", qwen2_mod)
    monkeypatch.setitem(sys.modules, "tpu_inference.models.jax.llama3", llama3_mod)

    vLLMInferenceContext._patch_tpu_inference_registry()

    assert registry["Qwen2ForCausalLM"] is FakeQwen2ForCausalLM
    assert registry["MistralForCausalLM"] is FakeLlamaForCausalLM
