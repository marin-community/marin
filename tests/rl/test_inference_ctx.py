# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for InferenceContext utilities and chat template handling."""

import json
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
from levanter.inference.openai import ChatMessage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletionTokenLogprob, Choice, ChoiceLogprobs
from transformers import AutoTokenizer

import marin.rl.environments.inference_ctx.vllm as vllm_module
from marin.rl.environments.inference_ctx import (
    LevanterInferenceContext,
    LevanterInferenceContextConfig,
    MODEL_MAPPINGS,
    MODEL_TRANSPOSE_KEYS,
    ToolCall,
    ToolSpec,
    VLLMSamplingConfig,
    vLLMInferenceContext,
    vLLMInferenceContextConfig,
)
from marin.rl.environments.inference_ctx.vllm import InferenceMode
from marin.rl.environments.inference_ctx.inflight.worker import WorkerExtension
from marin.rl.environments.inference_ctx.render import Qwen3Renderer

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


class RendererTestTokenizer:
    """Tokenizer stub that preserves text and treats <|im_end|> as one token."""

    _END_MESSAGE_TOKEN = 99999

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        tokens = []
        cursor = 0
        while cursor < len(text):
            if text.startswith("<|im_end|>", cursor):
                tokens.append(self._END_MESSAGE_TOKEN)
                cursor += len("<|im_end|>")
                continue
            tokens.append(ord(text[cursor]))
            cursor += 1
        return tokens

    def decode(self, tokens: list[int]) -> str:
        parts = []
        for token in tokens:
            if token == self._END_MESSAGE_TOKEN:
                parts.append("<|im_end|>")
            else:
                parts.append(chr(token))
        return "".join(parts)

    def get_vocab(self) -> dict[str, int]:
        return {"<|im_end|>": self._END_MESSAGE_TOKEN}


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


def test_qwen_renderer_renders_tool_schema_and_openreward_tool_payload():
    tokenizer = RendererTestTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    tool_call = ToolCall(
        id="call_submit",
        function=ToolCall.FunctionBody(
            name="submit_answer",
            arguments='{"confidence":1,"answer":"42"}',
        ),
    )
    tool_spec = ToolSpec(
        function=ToolSpec.FunctionBody(
            name="submit_answer",
            description="Submit the final answer.",
            parameters={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["answer"],
            },
        )
    )

    rendered = tokenizer.decode(
        renderer.build_generation_prompt(
            [
                {
                    "role": "assistant",
                    "content": "Solved it.",
                    "tool_calls": [tool_call],
                }
            ],
            tools=[tool_spec],
        )
    )

    assert "<tools>" in rendered
    assert (
        '{"type":"function","function":{"name":"submit_answer","description":"Submit the final answer.",'
        '"parameters":{"type":"object","properties":{"answer":{"type":"string"},"confidence":{"type":"number"}},'
        '"required":["answer"]}}}'
    ) in rendered
    assert (
        '<tool_call>\n{"name":"submit_answer","arguments":{"confidence":1,"answer":"42"},"id":"call_submit"}\n</tool_call>'
        in rendered
    )


def test_qwen_renderer_parses_newline_wrapped_canonical_tool_call():
    tokenizer = RendererTestTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    response = tokenizer.encode(
        '<tool_call>\n{"name":"submit_answer","arguments":{"answer":"42"},"id":"call_submit"}\n</tool_call><|im_end|>'
    )

    parse_result = renderer.parse_response(response)

    assert parse_result.parse_success is True
    assert parse_result.assistant_turn.content == ""
    assert parse_result.assistant_turn.tool_calls[0].function.name == "submit_answer"
    assert json.loads(parse_result.assistant_turn.tool_calls[0].function.arguments) == {"answer": "42"}
    assert parse_result.assistant_turn.tool_calls[0].id == "call_submit"


@pytest.mark.parametrize(
    ("payload", "expected_id"),
    [
        ('{"name":"submit_answer","arguments":{"answer":"42"},"id":"call_submit"}', "call_submit"),
        ('{"name":"submit_answer","args":{"answer":"42"}}', None),
    ],
)
def test_qwen_renderer_parses_openreward_and_legacy_tool_calls(payload: str, expected_id: str | None):
    tokenizer = RendererTestTokenizer()
    renderer = Qwen3Renderer(tokenizer)

    response = tokenizer.encode(f"<tool_call>{payload}</tool_call><|im_end|>")
    parse_result = renderer.parse_response(response)

    assert parse_result.parse_success is True
    assert parse_result.assistant_turn.content == ""
    assert parse_result.assistant_turn.tool_calls[0].function.name == "submit_answer"
    assert json.loads(parse_result.assistant_turn.tool_calls[0].function.arguments) == {"answer": "42"}
    assert parse_result.assistant_turn.tool_calls[0].id == expected_id


def test_qwen_renderer_strips_raw_tool_call_xml_from_assistant_content():
    tokenizer = RendererTestTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    response = tokenizer.encode(
        "I should submit the final answer now.\n"
        '<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>'
        "<|im_end|>"
    )

    parse_result = renderer.parse_response(response)

    assert parse_result.parse_success is True
    assert parse_result.assistant_turn.content == "I should submit the final answer now."
    assert "<tool_call>" not in parse_result.assistant_turn.content
    assert len(parse_result.assistant_turn.tool_calls) == 1


def test_qwen_renderer_strips_tool_calls_and_keeps_non_tool_text_with_multiple_calls():
    tokenizer = RendererTestTokenizer()
    renderer = Qwen3Renderer(tokenizer)
    response = tokenizer.encode(
        "Need two lookups before I answer.\n"
        '<tool_call>{"name":"search","arguments":{"query":"cats"}}</tool_call>\n'
        '<tool_call>{"name":"search","arguments":{"query":"dogs"}}</tool_call>\n'
        "Done."
        "<|im_end|>"
    )

    parse_result = renderer.parse_response(response)

    assert parse_result.parse_success is True
    assert parse_result.assistant_turn.content == "Need two lookups before I answer.\n\nDone."
    assert [tool_call.function.name for tool_call in parse_result.assistant_turn.tool_calls] == ["search", "search"]
    assert json.loads(parse_result.assistant_turn.tool_calls[0].function.arguments) == {"query": "cats"}
    assert json.loads(parse_result.assistant_turn.tool_calls[1].function.arguments) == {"query": "dogs"}


def test_vllm_batch_completions_passes_tools_to_renderer_and_stop_sequence(monkeypatch):
    calls = {}
    tool_spec = ToolSpec(
        function=ToolSpec.FunctionBody(
            name="submit_answer",
            description="Submit the final answer.",
            parameters={"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
        )
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            calls["sampling_kwargs"] = kwargs
            self.max_tokens = kwargs["max_tokens"]

    class FakeTokensPrompt:
        def __init__(self, *, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class FakeRenderer:
        def build_generation_prompt(self, messages, role="assistant", prefill=None, tools=None):
            del role, prefill
            calls["render_messages"] = messages
            calls["render_tools"] = tools
            return [11, 22, 33]

    class FakeLLM:
        def generate(self, prompts, sampling_params):
            calls["prompts"] = prompts
            calls["sampling_params"] = sampling_params
            return []

    monkeypatch.setattr(vllm_module, "SamplingParams", FakeSamplingParams)
    monkeypatch.setattr(vllm_module, "TokensPrompt", FakeTokensPrompt)
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

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="Qwen/Qwen3-8B",
            canonical_model_name="Qwen/Qwen3-8B",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
        )
    )

    completions = ctx.batch_completions(
        prompts=["Solve the problem and call submit_answer."],
        temperature=0.7,
        n=1,
        stop=["</tool_call>"],
        tools=[tool_spec],
    )

    assert completions == []
    assert calls["sampling_kwargs"]["stop"] == ["</tool_call>"]
    assert calls["render_messages"] == [{"role": "user", "content": "Solve the problem and call submit_answer."}]
    assert calls["render_tools"] == [tool_spec]


def test_vllm_assistant_turn_from_choice_parses_tool_calls(monkeypatch):
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: RendererTestTokenizer(),
    )

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="Qwen/Qwen3-8B",
            canonical_model_name="Qwen/Qwen3-8B",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
        )
    )

    choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content='<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>',
        ),
        logprobs=ChoiceLogprobs(content=[]),
    )
    choice.response_token_ids = RendererTestTokenizer().encode(
        '<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call><|im_end|>'
    )

    parse_result = ctx.assistant_turn_from_choice(choice)

    assert parse_result.parse_success is True
    assert parse_result.assistant_turn.content == ""
    assert parse_result.assistant_turn.tool_calls[0].function.name == "submit_answer"
    assert json.loads(parse_result.assistant_turn.tool_calls[0].function.arguments) == {"answer": "42"}


def test_vllm_convert_preserves_tool_call_token_ids_and_logprobs(monkeypatch):
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_llm_engine",
        staticmethod(lambda _config: object()),
    )
    monkeypatch.setattr(
        "marin.rl.environments.inference_ctx.vllm.load_tokenizer",
        lambda _path: SimpleNamespace(
            get_vocab=lambda: {"<tool_call>": 101, '{"answer":"42"}': 102, "</tool_call>": 103}
        ),
    )
    monkeypatch.setattr(
        vLLMInferenceContext,
        "_get_renderer",
        staticmethod(lambda _model_name, _tokenizer: object()),
    )

    ctx = vLLMInferenceContext(
        vLLMInferenceContextConfig(
            model_name="Qwen/Qwen3-8B",
            canonical_model_name="Qwen/Qwen3-8B",
            max_model_len=1024,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            sampling_params=VLLMSamplingConfig(),
        )
    )

    request_output = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                finish_reason="stop",
                text='<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>',
                token_ids=[101, 102, 103],
                logprobs=[
                    {
                        101: SimpleNamespace(rank=1, logprob=-0.1),
                        999: SimpleNamespace(rank=2, logprob=-1.5),
                    },
                    {102: SimpleNamespace(rank=1, logprob=-0.2)},
                    {103: SimpleNamespace(rank=1, logprob=-0.3)},
                ],
            )
        ],
    )

    completion = ctx._convert_vllm_to_openai(request_output)
    choice = completion.choices[0]

    assert choice.message.content == '<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>'
    assert choice.prompt_token_ids == [1, 2, 3]
    assert choice.response_token_ids == [101, 102, 103]
    np.testing.assert_array_equal(
        ctx.response_tokens_from_choice(choice),
        np.array([101, 102, 103], dtype=np.int32),
    )
    np.testing.assert_allclose(
        ctx.logprobs_from_choice(choice),
        np.array([-0.1, -0.2, -0.3], dtype=np.float32),
    )
    assert choice.logprobs.content[0].token == "<tool_call>"
    assert choice.logprobs.content[0].top_logprobs[1].token == "<id_999>"


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
