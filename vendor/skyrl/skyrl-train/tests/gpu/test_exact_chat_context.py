"""Pinned vLLM must validate the tokens that SkyRL actually serves."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.exceptions import VLLMValidationError

from skyrl_train.inference_engines.vllm.vllm_engine import SkyRLOpenAIServingChat, _exact_chat_prompt_token_ids

pytestmark = pytest.mark.manual


class RejectingRenderer:
    tokenizer = None

    async def render_chat_async(self, *args, **kwargs):
        raise ValueError("Reconstructed message text exceeds the prompt budget")


class ModelRegistry:
    def is_base_model(self, model):
        return model == "fixture"


class Engine:
    errored = False
    input_processor = None
    vllm_config = None

    def __init__(self):
        self.renderer = RejectingRenderer()
        self.model_config = SimpleNamespace(
            model="fixture",
            hf_config=SimpleNamespace(model_type="qwen3_5_moe"),
            max_model_len=65536,
            multimodal_config=None,
            enable_prompt_embeds=False,
            generation_config="vllm",
            get_diff_sampling_param=lambda: {},
        )

    def check_admission(self, n):
        pass


@pytest.mark.asyncio
async def test_exact_tokens_bypass_invalid_reconstruction_without_disabling_normal_validation():
    engine = Engine()
    renderer = OnlineRenderer(
        model_config=engine.model_config,
        renderer=engine.renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )
    serving = SkyRLOpenAIServingChat(
        engine_client=engine,
        models=ModelRegistry(),
        response_role="assistant",
        online_renderer=renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )
    request = ChatCompletionRequest(
        model="fixture",
        messages=[{"role": "user", "content": "large reconstructed history"}],
        max_completion_tokens=16384,
        kv_transfer_params={"kept": "metadata"},
    )
    with pytest.raises(ValueError, match="Reconstructed message"):
        await serving.render_chat_request(request)
    token = _exact_chat_prompt_token_ids.set([10, 11, 12])
    try:
        conversation, prompts = await serving.render_chat_request(request)
    finally:
        _exact_chat_prompt_token_ids.reset(token)
    assert conversation == []
    assert prompts[0]["prompt_token_ids"] == [10, 11, 12]
    assert request.kv_transfer_params == {"kept": "metadata"}
    token = _exact_chat_prompt_token_ids.set([10] * 49153)
    try:
        with pytest.raises(VLLMValidationError, match="maximum context length"):
            await serving.render_chat_request(request)
    finally:
        _exact_chat_prompt_token_ids.reset(token)
