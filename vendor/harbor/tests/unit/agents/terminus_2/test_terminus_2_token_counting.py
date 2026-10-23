import json

import httpx
import pytest
import tiktoken
import tiktoken.registry
from harbor.agents.terminus_2 import terminus_2
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.llms.base import LLMResponse
from harbor.llms.chat import Chat
from openai import AsyncOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("count_mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("model_name", "encoding_source"),
    [
        ("anthropic/claude-sonnet-4-5", "explicit"),
        ("gemini/gemini-2.5-pro", "explicit"),
        ("openrouter/anthropic/claude-sonnet-4-5", "explicit"),
        ("openai/gpt-4o", "model"),
    ],
)
async def test_proxy_without_tokenize_counts_with_explicit_encoding_and_generates(
    tmp_path, monkeypatch, model_name, count_mode, encoding_source
):
    # A local encoding keeps tokenizer data loading offline while using real BPE.
    encoding = tiktoken.Encoding(
        name="harbor-test-bytes",
        pat_str=r"(?s).",
        mergeable_ranks={bytes([value]): value for value in range(256)},
        special_tokens={},
    )
    monkeypatch.setitem(tiktoken.registry.ENCODINGS, encoding.name, encoding)
    monkeypatch.setitem(tiktoken.model.MODEL_TO_ENCODING, "gpt-4o", encoding.name)
    requests = []

    def proxy(request):
        requests.append(request)
        if request.url.path == "/tokenize":
            return httpx.Response(404)
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == model_name.split("/", 1)[1]
        content = body["messages"][-1]["content"]
        text = (
            content
            if isinstance(content, str)
            else "".join(part["text"] for part in content)
        )
        assert text == "Continue."
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-proxy",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(proxy)
    agent = Terminus2(
        logs_dir=tmp_path,
        model_name=model_name,
        api_base="http://proxy/v1",
        tokenizer_encoding=encoding.name if encoding_source == "explicit" else None,
        model_info={"max_input_tokens": 32768, "max_output_tokens": 1024},
    )
    chat = Chat(agent._llm)
    chat.append_response("Inspect the repository.", LLMResponse(content="Inspected."))
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            monkeypatch.setattr(
                terminus_2, "shared_async_tokenize_client", lambda: async_client
            )
            monkeypatch.setattr(
                terminus_2, "_get_shared_tokenize_client", lambda: sync_client
            )
            async with AsyncOpenAI(
                api_key="EMPTY", base_url="http://proxy/v1", http_client=async_client
            ) as model_client:
                agent._llm._openai_client = model_client
                extra = [{"role": "user", "content": "Count this summary request too."}]
                if count_mode == "sync":
                    base_count = agent._count_total_tokens(chat)
                    request_count = agent._count_total_tokens(chat, extra)
                else:
                    base_count = await agent._acount_total_tokens(chat)
                    request_count = await agent._acount_total_tokens(chat, extra)
                assert request_count > base_count > 0
                response = await chat.chat("Continue.")
                assert response.content == "Done."
    assert any(request.url.path == "/tokenize" for request in requests)
