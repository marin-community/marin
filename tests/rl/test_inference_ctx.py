# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for InferenceContext utilities and chat template handling."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from flax import nnx
from levanter.inference.openai import ChatMessage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from transformers import AutoTokenizer

from marin.rl.environments.inference_ctx import LevanterInferenceContext, LevanterInferenceContextConfig
from marin.rl.environments.inference_ctx.vllm import (
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
    VllmSamplingConfig,
    _bootstrap_weights_into_engine,
    _patch_tpu_inference_llama_nnx_compat,
    coerce_vllm_sampling_config,
)
from marin.rl.weight_utils import (
    levanter_state_dict_to_nnx_state_on_cpu,
    torch_compatible_state_dict_to_levanter_state_dict,
)


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


@pytest.fixture
def llama3_tokenizer():
    """Llama 3 tokenizer with chat template (uses tiktoken, not sentencepiece)."""
    return AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")


@pytest.fixture
def gpt2_tokenizer():
    """GPT-2 tokenizer without chat template (for fallback testing)."""
    return AutoTokenizer.from_pretrained("gpt2")


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


def test_patch_tpu_inference_llama_nnx_compat_wraps_layers_in_nnx_list() -> None:
    llama3_module = SimpleNamespace()

    def make_layers():
        return 0, 2, ["layer-0", "layer-1"]

    class DummyLlamaModel:
        def __init__(self):
            _, _, self.layers = llama3_module.make_layers()

    llama3_module.LlamaModel = DummyLlamaModel
    llama3_module.make_layers = make_layers

    _patch_tpu_inference_llama_nnx_compat(llama3_module, nnx)
    patched_make_layers = llama3_module.make_layers

    model = llama3_module.LlamaModel()
    assert isinstance(model.layers, nnx.List)
    assert list(model.layers) == ["layer-0", "layer-1"]

    _patch_tpu_inference_llama_nnx_compat(llama3_module, nnx)
    assert llama3_module.make_layers is patched_make_layers


def test_coerce_vllm_sampling_config_fills_missing_legacy_fields() -> None:
    legacy_params = SimpleNamespace(
        temperature=0.7,
        n=4,
        max_tokens=128,
        top_k=256,
        include_stop_str_in_output=True,
        logprobs=2,
    )

    sampling_config = coerce_vllm_sampling_config(legacy_params)

    assert sampling_config == VllmSamplingConfig(
        temperature=0.7,
        n=4,
        max_tokens=128,
        top_k=256,
        stop=None,
        include_stop_str_in_output=True,
        logprobs=2,
        output_kind=None,
    )


def test_fast_bootstrap_requires_checkpoint_path() -> None:
    inference_config = vLLMInferenceContextConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=128,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.94,
        sampling_params=VllmSamplingConfig(),
        enable_fast_bootstrap=True,
        bootstrap_checkpoint_path=None,
    )

    with pytest.raises(ValueError, match="bootstrap_checkpoint_path is unset"):
        vLLMInferenceContext._build_llm_with_optional_fast_bootstrap(inference_config)


def test_fast_bootstrap_failure_does_not_fall_back_to_base_model(monkeypatch) -> None:
    inference_config = vLLMInferenceContextConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=128,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.94,
        sampling_params=VllmSamplingConfig(),
        enable_fast_bootstrap=True,
        bootstrap_checkpoint_path="gs://bucket/policy",
    )
    engine_calls: list[tuple[bool, str, str | None]] = []

    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm._is_object_store_path",
        lambda _: True,
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm._stage_bootstrap_metadata",
        lambda _: "/tmp/marin-vllm-bootstrap-test",
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.shutil.rmtree",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.AutoConfig.from_pretrained",
        lambda _: SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=2),
    )

    def _fake_get_llm_engine(config, model_source=None):
        engine_calls.append((config.enable_fast_bootstrap, config.load_format, model_source))
        return object()

    monkeypatch.setattr(vLLMInferenceContext, "_get_llm_engine", staticmethod(_fake_get_llm_engine))
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm._bootstrap_weights_into_engine",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("oom during sync_weights")),
    )

    with pytest.raises(RuntimeError, match="Fast bootstrap failed while loading policy weights"):
        vLLMInferenceContext._build_llm_with_optional_fast_bootstrap(inference_config)

    assert engine_calls == [(True, "dummy", "/tmp/marin-vllm-bootstrap-test")]


def test_checkpoint_native_fast_bootstrap_uses_levanter_checkpoint_loader(monkeypatch) -> None:
    inference_config = vLLMInferenceContextConfig(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        max_model_len=128,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.94,
        sampling_params=VllmSamplingConfig(),
        enable_fast_bootstrap=True,
        bootstrap_checkpoint_path="gs://bucket/levanter/step-1",
        bootstrap_checkpoint_format="levanter_checkpoint",
        bootstrap_levanter_model_config=SimpleNamespace(name="dummy-model-config"),
        bootstrap_tokenizer_name="meta-llama/Llama-3.1-8B-Instruct",
        bootstrap_vocab_size=32000,
    )
    engine_calls: list[tuple[bool, str, str | None]] = []
    bootstrap_calls: list[tuple[object, str, str, object, str, int | None]] = []

    def _fake_get_llm_engine(config, model_source=None):
        engine_calls.append((config.enable_fast_bootstrap, config.load_format, model_source))
        return object()

    monkeypatch.setattr(vLLMInferenceContext, "_get_llm_engine", staticmethod(_fake_get_llm_engine))
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm._bootstrap_levanter_checkpoint_into_engine",
        lambda llm, model_name, checkpoint_path, *, model_config, tokenizer_name, vocab_size: bootstrap_calls.append(
            (llm, model_name, checkpoint_path, model_config, tokenizer_name, vocab_size)
        ),
    )

    result = vLLMInferenceContext._build_llm_with_optional_fast_bootstrap(inference_config)

    assert result is bootstrap_calls[0][0]
    assert engine_calls == [(True, "dummy", "meta-llama/Llama-3.1-8B-Instruct")]
    assert bootstrap_calls == [
        (
            result,
            "meta-llama/Llama-3.1-8B-Instruct",
            "gs://bucket/levanter/step-1",
            inference_config.bootstrap_levanter_model_config,
            "meta-llama/Llama-3.1-8B-Instruct",
            32000,
        )
    ]


def test_torch_compatible_state_dict_round_trips_attention_tensors() -> None:
    hf_config = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    live_state_dict = {
        "model.layers.0.self_attn.q_proj.weight": np.arange(2 * 2 * 2 * 8, dtype=np.float32).reshape(2, 2, 2, 8),
        "model.layers.0.self_attn.q_proj.bias": np.arange(2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2),
        "model.layers.0.self_attn.k_proj.weight": np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8),
        "model.layers.0.self_attn.k_proj.bias": np.arange(2 * 2, dtype=np.float32).reshape(2, 2),
        "model.layers.0.self_attn.v_proj.weight": np.arange(2 * 2 * 8, dtype=np.float32).reshape(2, 2, 8) + 1000.0,
        "model.layers.0.self_attn.v_proj.bias": np.arange(2 * 2, dtype=np.float32).reshape(2, 2) + 100.0,
        "model.layers.0.self_attn.o_proj.weight": np.arange(8 * 4 * 2, dtype=np.float32).reshape(8, 4, 2),
    }
    exported_state_dict = {
        "model.layers.0.self_attn.q_proj.weight": (
            live_state_dict["model.layers.0.self_attn.q_proj.weight"].reshape(8, 8)
        ),
        "model.layers.0.self_attn.q_proj.bias": live_state_dict["model.layers.0.self_attn.q_proj.bias"].reshape(8),
        "model.layers.0.self_attn.k_proj.weight": (
            live_state_dict["model.layers.0.self_attn.k_proj.weight"].reshape(4, 8)
        ),
        "model.layers.0.self_attn.k_proj.bias": live_state_dict["model.layers.0.self_attn.k_proj.bias"].reshape(4),
        "model.layers.0.self_attn.v_proj.weight": (
            live_state_dict["model.layers.0.self_attn.v_proj.weight"].reshape(4, 8)
        ),
        "model.layers.0.self_attn.v_proj.bias": live_state_dict["model.layers.0.self_attn.v_proj.bias"].reshape(4),
        "model.layers.0.self_attn.o_proj.weight": (
            live_state_dict["model.layers.0.self_attn.o_proj.weight"].reshape(8, 8)
        ),
    }

    recovered_state_dict = torch_compatible_state_dict_to_levanter_state_dict(exported_state_dict, hf_config)

    for key, expected_value in live_state_dict.items():
        np.testing.assert_array_equal(np.asarray(recovered_state_dict[key]), expected_value)

    expected_nnx_state = levanter_state_dict_to_nnx_state_on_cpu(live_state_dict)
    recovered_nnx_state = levanter_state_dict_to_nnx_state_on_cpu(recovered_state_dict)

    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["q_proj"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["q_proj"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["q_proj_bias"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["q_proj_bias"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["k_proj"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["k_proj"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["k_proj_bias"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["k_proj_bias"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["v_proj"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["v_proj"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["v_proj_bias"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["v_proj_bias"].value),
    )
    np.testing.assert_array_equal(
        np.asarray(expected_nnx_state["model"]["layers"]["0"]["self_attn"]["o_proj"].value),
        np.asarray(recovered_nnx_state["model"]["layers"]["0"]["self_attn"]["o_proj"].value),
    )


def test_bootstrap_weights_reconstructs_grouped_attention_tensors_before_sync(monkeypatch) -> None:
    hf_config = SimpleNamespace(hidden_size=8, num_attention_heads=4, num_key_value_heads=2, head_dim=2)
    exported_state_dict = {
        "model.layers.0.self_attn.q_proj.weight": np.arange(8 * 8, dtype=np.float32).reshape(8, 8),
        "model.layers.0.self_attn.k_proj.weight": np.arange(4 * 8, dtype=np.float32).reshape(4, 8),
        "model.layers.0.self_attn.v_proj.weight": np.arange(4 * 8, dtype=np.float32).reshape(4, 8) + 1000.0,
        "model.layers.0.self_attn.o_proj.weight": np.arange(8 * 8, dtype=np.float32).reshape(8, 8) + 2000.0,
    }
    captured = {}

    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm._load_safetensors_from_remote",
        lambda _path: exported_state_dict,
    )

    class FakeDriverWorker:
        def sync_weights(self, nnx_state, mappings, transpose_keys, reshard_fn):
            del mappings, transpose_keys, reshard_fn
            self_attn = nnx_state["model"]["layers"]["0"]["self_attn"]
            captured["q_proj_shape"] = tuple(self_attn["q_proj"].value.shape)
            captured["k_proj_shape"] = tuple(self_attn["k_proj"].value.shape)
            captured["v_proj_shape"] = tuple(self_attn["v_proj"].value.shape)
            captured["o_proj_shape"] = tuple(self_attn["o_proj"].value.shape)

    class FakeLlmEngine:
        def __init__(self):
            self.model_executor = SimpleNamespace(driver_worker=FakeDriverWorker())
            self.reset_prefix_cache_calls = 0

        def reset_prefix_cache(self):
            self.reset_prefix_cache_calls += 1

    fake_llm_engine = FakeLlmEngine()
    fake_llm = SimpleNamespace(llm_engine=fake_llm_engine)

    _bootstrap_weights_into_engine(
        fake_llm,
        "meta-llama/Llama-3.1-8B-Instruct",
        "gs://bucket/policy",
        hf_config,
    )

    assert captured == {
        "q_proj_shape": (4, 128, 8),
        "k_proj_shape": (2, 128, 8),
        "v_proj_shape": (2, 128, 8),
        "o_proj_shape": (8, 4, 128),
    }
    assert fake_llm_engine.reset_prefix_cache_calls == 1
