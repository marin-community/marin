# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for InferenceContext utilities and chat template handling."""

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from types import ModuleType
from unittest.mock import AsyncMock

import numpy as np
import pytest
from levanter.inference.openai import ChatMessage
from marin.rl.decoding import DecodingConfig, DecodingStrategy
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from transformers import AutoTokenizer

from marin.rl.environments.inference_ctx import (
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    MODEL_MAPPINGS,
    MODEL_TRANSPOSE_KEYS,
    VLLMEngineConfig,
    VLLMFallbackSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.vllm import InferenceMode
from marin.rl.environments.inference_ctx.inflight.worker import WorkerExtension

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
        decoding=DecodingConfig(temperature=1.0),
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


def _chat_completion_from_choice(choice: Choice) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def _test_levanter_context(tokenizer, dummy_server, client) -> LevanterInferenceContext:
    class _TestLevanterInferenceContext(LevanterInferenceContext):
        def __init__(self):
            self.tokenizer = tokenizer
            self._stop_tokens = None
            self.max_tokens = 100
            self.mesh = None
            self.axis_mapping = {}
            self._inference_server = dummy_server

        def openai_client(self):
            return client

    return _TestLevanterInferenceContext()


def _test_vllm_engine_config(**overrides) -> VLLMEngineConfig:
    kwargs = {
        "model_name": "test-model",
        "max_model_len": 1024,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
    }
    kwargs.update(overrides)
    return VLLMEngineConfig(**kwargs)


def _test_vllm_inference_config(
    *,
    fallback_sampling: VLLMFallbackSamplingConfig | None = None,
    **engine_overrides,
) -> vLLMInferenceContextConfig:
    return vLLMInferenceContextConfig(
        engine=_test_vllm_engine_config(**engine_overrides),
        fallback_sampling=fallback_sampling or VLLMFallbackSamplingConfig(),
    )


def test_levanter_batch_completions_forwards_seed(gpt2_tokenizer, dummy_server):
    choice = create_choice_with_logprobs(gpt2_tokenizer, "hello")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_chat_completion_from_choice(choice))
    mock_client.close = AsyncMock()

    ctx = _test_levanter_context(gpt2_tokenizer, dummy_server, mock_client)
    completions = ctx.batch_completions(
        prompts=["prompt"],
        n=2,
        decoding=DecodingConfig(temperature=0.7, seed=123),
    )

    assert len(completions) == 1
    assert mock_client.chat.completions.create.await_args.kwargs["seed"] == 123


def test_levanter_batch_completions_maps_supported_decoding_fields(gpt2_tokenizer, dummy_server):
    choice = create_choice_with_logprobs(gpt2_tokenizer, "hello")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_chat_completion_from_choice(choice))
    mock_client.close = AsyncMock()

    ctx = _test_levanter_context(gpt2_tokenizer, dummy_server, mock_client)
    completions = ctx.batch_completions(
        prompts=["prompt"],
        n=2,
        decoding=DecodingConfig(
            strategy=DecodingStrategy.GREEDY,
            temperature=0.7,
            top_p=0.91,
            max_output_tokens=17,
            stop_strings=["<stop>"],
            seed=123,
        ),
    )

    assert len(completions) == 1
    assert mock_client.chat.completions.create.await_args.kwargs["temperature"] == 0.0
    assert mock_client.chat.completions.create.await_args.kwargs["top_p"] == 0.91
    assert mock_client.chat.completions.create.await_args.kwargs["max_tokens"] == 17
    assert mock_client.chat.completions.create.await_args.kwargs["stop"] == ["<stop>"]
    assert mock_client.chat.completions.create.await_args.kwargs["n"] == 2
    assert mock_client.chat.completions.create.await_args.kwargs["seed"] == 123


def test_levanter_batch_completions_converts_stop_token_ids_to_stop_strings(gpt2_tokenizer, dummy_server):
    choice = create_choice_with_logprobs(gpt2_tokenizer, "hello")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_chat_completion_from_choice(choice))
    mock_client.close = AsyncMock()

    stop_token_ids = gpt2_tokenizer.encode(" END", add_special_tokens=False)
    ctx = _test_levanter_context(gpt2_tokenizer, dummy_server, mock_client)
    ctx.batch_completions(
        prompts=["prompt"],
        n=1,
        decoding=DecodingConfig(temperature=1.0, stop_token_ids=stop_token_ids),
    )

    assert mock_client.chat.completions.create.await_args.kwargs["stop"] == [
        gpt2_tokenizer.decode([token_id]) for token_id in stop_token_ids
    ]


def test_levanter_batch_completions_uses_context_stop_token_fallback(gpt2_tokenizer, dummy_server):
    choice = create_choice_with_logprobs(gpt2_tokenizer, "hello")
    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=_chat_completion_from_choice(choice))
    mock_client.close = AsyncMock()

    stop_token_ids = gpt2_tokenizer.encode(" END", add_special_tokens=False)

    class _TestLevanterInferenceContext(LevanterInferenceContext):
        def __init__(self):
            self.tokenizer = gpt2_tokenizer
            self._stop_tokens = stop_token_ids
            self.max_tokens = 100
            self.mesh = None
            self.axis_mapping = {}
            self._inference_server = dummy_server

        def openai_client(self):
            return mock_client

    ctx = _TestLevanterInferenceContext()
    ctx.batch_completions(
        prompts=["prompt"],
        n=1,
        decoding=DecodingConfig(temperature=1.0),
    )

    assert mock_client.chat.completions.create.await_args.kwargs["stop"] == [
        gpt2_tokenizer.decode([token_id]) for token_id in stop_token_ids
    ]


def test_levanter_batch_completions_rejects_unsupported_decoding_fields(gpt2_tokenizer, dummy_server):
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    ctx = _test_levanter_context(gpt2_tokenizer, dummy_server, mock_client)

    with pytest.raises(ValueError, match=r"ignore_eos"):
        ctx.batch_completions(
            prompts=["prompt"],
            n=1,
            decoding=DecodingConfig(
                temperature=1.0,
                ignore_eos=True,
            ),
        )


def test_levanter_batch_completions_rejects_all_unsupported_decoding_fields(gpt2_tokenizer, dummy_server):
    mock_client = AsyncMock()
    mock_client.close = AsyncMock()
    ctx = _test_levanter_context(gpt2_tokenizer, dummy_server, mock_client)
    expected_match = "".join(
        [
            r"top_k, min_p, repetition_penalty, presence_penalty, ",
            r"frequency_penalty, min_output_tokens, ignore_eos",
        ]
    )

    with pytest.raises(
        ValueError,
        match=expected_match,
    ):
        ctx.batch_completions(
            prompts=["prompt"],
            n=1,
            decoding=DecodingConfig(
                temperature=1.0,
                top_k=8,
                min_p=0.1,
                repetition_penalty=1.1,
                presence_penalty=0.2,
                frequency_penalty=0.3,
                min_output_tokens=4,
                ignore_eos=True,
            ),
        )


def test_create_rollout_from_choice_records_resolved_levanter_stop_tokens(gpt2_tokenizer):
    ctx = LevanterInferenceContext(
        LevanterInferenceContextConfig(
            inference_server_config=None,
            tokenizer=gpt2_tokenizer,
            stop_tokens=[42],
            max_tokens=100,
            mesh=None,
            axis_mapping={},
        )
    )

    choice = create_choice_with_logprobs(gpt2_tokenizer, "hello")
    rollout = ctx.create_rollout_from_choice(
        prompt="test prompt",
        choice=choice,
        env_name="mock",
        env_example_id="example",
        reward=1.0,
        decoding=DecodingConfig(temperature=1.0),
    )

    assert rollout.decoding.stop_token_ids == (42,)


def test_vllm_resolve_decoding_includes_sampling_fallbacks(monkeypatch):
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
        _test_vllm_inference_config(
            fallback_sampling=VLLMFallbackSamplingConfig(top_k=16, stop_strings=["<stop>"]),
        )
    )

    resolved = ctx.resolve_decoding(DecodingConfig(temperature=1.0))

    assert resolved.top_k == 16
    assert resolved.stop_strings == ["<stop>"]


def test_vllm_sampling_params_maps_shared_decoding_fields(monkeypatch):
    calls = {}

    class FakeSamplingParams:
        def __init__(
            self,
            *,
            temperature,
            n,
            max_tokens,
            logprobs,
            include_stop_str_in_output,
            top_k=None,
            top_p=None,
            min_p=None,
            repetition_penalty=None,
            presence_penalty=None,
            frequency_penalty=None,
            min_tokens=None,
            stop=None,
            stop_token_ids=None,
            seed=None,
            ignore_eos=False,
            output_kind=None,
        ):
            calls["kwargs"] = {
                "temperature": temperature,
                "n": n,
                "max_tokens": max_tokens,
                "logprobs": logprobs,
                "include_stop_str_in_output": include_stop_str_in_output,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
                "min_tokens": min_tokens,
                "stop": stop,
                "stop_token_ids": stop_token_ids,
                "seed": seed,
                "ignore_eos": ignore_eos,
                "output_kind": output_kind,
            }
            self.max_tokens = max_tokens

    class FakeRenderer:
        def build_generation_prompt(self, _messages):
            return [1, 2, 3]

    class FakeLLM:
        def generate(self, prompts, sampling_params):
            calls["prompts"] = prompts
            calls["sampling_params"] = sampling_params
            return []

    class FakeTokensPrompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SamplingParams", FakeSamplingParams)
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.TokensPrompt", FakeTokensPrompt)
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: FakeLLM()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: SimpleNamespace(get_vocab=lambda: {}),
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda _model_name, _tokenizer: FakeRenderer()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.RequestOutputKind",
        SimpleNamespace(FINAL_ONLY="final-only"),
    )

    ctx = vLLMInferenceContext(
        _test_vllm_inference_config(
            fallback_sampling=VLLMFallbackSamplingConfig(include_stop_str_in_output=True),
            mode=InferenceMode.ASYNC,
        )
    )

    result = ctx.batch_completions(
        prompts=["hello"],
        n=2,
        decoding=DecodingConfig(
            strategy=DecodingStrategy.GREEDY,
            temperature=0.7,
            top_k=8,
            top_p=0.91,
            min_p=0.05,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            max_output_tokens=64,
            min_output_tokens=4,
            stop_strings=["<stop>"],
            ignore_eos=True,
            seed=123,
        ),
    )

    assert result == []
    assert [prompt.prompt_token_ids for prompt in calls["prompts"]] == [[1, 2, 3]]
    assert calls["kwargs"] == {
        "temperature": 0.0,
        "n": 2,
        "max_tokens": 64,
        "logprobs": 1,
        "include_stop_str_in_output": True,
        "top_k": 8,
        "top_p": 0.91,
        "min_p": 0.05,
        "repetition_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "min_tokens": 4,
        "stop": ["<stop>"],
        "stop_token_ids": None,
        "seed": 123,
        "ignore_eos": True,
        "output_kind": "final-only",
    }


def test_vllm_sampling_params_passes_stop_token_ids_when_supported(monkeypatch):
    calls = {}

    class FakeSamplingParams:
        def __init__(self, *, temperature, n, max_tokens, logprobs, include_stop_str_in_output, stop_token_ids=None):
            calls["stop_token_ids"] = stop_token_ids
            self.max_tokens = max_tokens

    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SamplingParams", FakeSamplingParams)
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
        staticmethod(lambda _model_name, _tokenizer: SimpleNamespace(build_generation_prompt=lambda _messages: [])),
    )

    ctx = vLLMInferenceContext(_test_vllm_inference_config())

    sampling_params = ctx._sampling_params_from_decoding(
        DecodingConfig(temperature=1.0, stop_token_ids=[7, 8]),
        n=1,
    )

    assert sampling_params.max_tokens == 512
    assert calls["stop_token_ids"] == [7, 8]


def test_vllm_sampling_params_rejects_stop_token_ids_when_unsupported(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, *, temperature, n, max_tokens, logprobs, include_stop_str_in_output):
            self.max_tokens = max_tokens

    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SamplingParams", FakeSamplingParams)
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
        staticmethod(lambda _model_name, _tokenizer: SimpleNamespace(build_generation_prompt=lambda _messages: [])),
    )

    ctx = vLLMInferenceContext(_test_vllm_inference_config())

    with pytest.raises(ValueError, match="does not support stop_token_ids"):
        ctx._sampling_params_from_decoding(DecodingConfig(temperature=1.0, stop_token_ids=[7]), n=1)


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
        _test_vllm_inference_config(
            model_name="gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
            canonical_model_name="meta-llama/Llama-3.1-8B-Instruct",
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

    config = _test_vllm_engine_config(tensor_parallel_size=2, kv_cache_metrics=True)

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["kv_cache_metrics"] is True


def test_vllm_async_engine_receives_kv_cache_metrics_flag(monkeypatch):
    calls = {}

    class _FakeSyncVLLMWrapper:
        def __init__(self, **kwargs):
            calls.update(kwargs)

    monkeypatch.setattr(vLLMInferenceContext, "_patch_tpu_inference_registry", staticmethod(lambda: None))
    monkeypatch.setattr("marin.rl.environments.inference_ctx.vllm.SyncVLLMWrapper", _FakeSyncVLLMWrapper)

    config = _test_vllm_engine_config(
        tensor_parallel_size=2,
        mode=InferenceMode.ASYNC,
        kv_cache_metrics=True,
    )

    vLLMInferenceContext._get_llm_engine(config)

    assert calls["kv_cache_metrics"] is True


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
