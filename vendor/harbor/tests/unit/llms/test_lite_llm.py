import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from harbor.llms.base import (
    ContextBudgetExceededError,
    ContextLengthExceededError,
    LLMRequestTimeoutError,
    OpenAITransportConnectTimeoutError,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM


# ---------------------------------------------------------------------------
# Transport test doubles
#
# The completion transport is now openai.AsyncOpenAI. Tests inject fakes at the
# thin seams `_acreate_chat_raw` (returns the raw vLLM JSON dict) and
# `_acreate_response` (returns an openai Responses object), which is where the
# old tests used to monkeypatch `litellm.acompletion` / `litellm.aresponses`.
# ---------------------------------------------------------------------------


def _install_fake_chat(llm, *, response=None, capture=None, raise_exc=None):
    """Replace both chat transport seams with a fake capturing completion kwargs.

    The chat dispatch now routes (via ``_dispatch_chat``) to EITHER
    ``_acreate_chat_raw`` (AsyncOpenAI, OpenAI-compat) or ``_acreate_chat_litellm``
    (litellm fallback, native providers). Patching BOTH seams makes call()-level
    behavior tests transport-agnostic — the fake is hit whichever route the model
    resolves to. Route SELECTION itself is covered by the dedicated routing tests.
    """

    async def _fake(completion_kwargs):
        if capture is not None:
            capture.clear()
            capture.update(completion_kwargs)
        if raise_exc is not None:
            raise raise_exc
        return response

    llm._acreate_chat_raw = _fake  # type: ignore[assignment]
    llm._acreate_chat_litellm = _fake  # type: ignore[assignment]


def _install_fake_responses(llm, *, response=None, capture=None, raise_exc=None):
    async def _fake(responses_kwargs):
        if capture is not None:
            capture.clear()
            capture.update(responses_kwargs)
        if raise_exc is not None:
            raise raise_exc
        return response

    llm._acreate_response = _fake  # type: ignore[assignment]


def _install_fake_completion(llm, *, response=None, capture=None, raise_exc=None):
    async def _fake(completion_kwargs):
        if capture is not None:
            capture.clear()
            capture.update(completion_kwargs)
        if raise_exc is not None:
            raise raise_exc
        return response

    llm._acreate_completion_raw = _fake  # type: ignore[assignment]


def _bad_request(message: str, body: dict | None = None) -> openai.BadRequestError:
    req = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    resp = httpx.Response(400, request=req, json=body or {})
    return openai.BadRequestError(message, response=resp, body=body)


def _chat_response(content="hi", reasoning=None, finish_reason="stop", model=None):
    return {
        "model": model,
        "choices": [
            {
                "message": {"content": content, "reasoning_content": reasoning},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


@pytest.mark.asyncio
async def test_chat_response_preserves_provider_reasoning_field():
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
    )
    response = _chat_response(content="", finish_reason="length")
    response["choices"][0]["message"] = {
        "content": "",
        "reasoning": "reasoning-only partial completion",
    }
    _install_fake_chat(llm, response=response)

    with pytest.raises(OutputLengthExceededError) as exc_info:
        await llm.call(prompt="summarize")

    assert (
        exc_info.value.partial_response.reasoning_content
        == "reasoning-only partial completion"
    )


def test_openai_transport_has_no_read_deadline():
    llm = LiteLLM(model_name="openai/test-model", api_base="http://localhost/v1")

    client = llm._get_openai_client()

    assert client.timeout.connect == 20.0
    assert client.timeout.read is None


@pytest.mark.asyncio
async def test_128_concurrent_requests_receive_unique_correlation_ids():
    llm = LiteLLM(model_name="openai/test-model", api_base="http://localhost/v1")
    request_ids: list[str] = []

    async def fake_chat(completion_kwargs):
        request_ids.append(completion_kwargs["extra_headers"]["X-Request-Id"])
        await asyncio.sleep(0)
        return _chat_response()

    llm._acreate_chat_raw = fake_chat  # type: ignore[assignment]

    await asyncio.gather(*(llm.call(prompt=f"request {index}") for index in range(128)))

    assert len(request_ids) == 128
    assert len(set(request_ids)) == 128


@pytest.mark.asyncio
async def test_openai_timeout_is_typed_logged_and_not_retried(tmp_path, caplog):
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-17",
    )
    calls = 0
    episode_logs = []

    def capture_episode_log(path, request_kwargs, response):
        episode_logs.append((path, request_kwargs, response))

    llm._enqueue_episode_log = capture_episode_log  # type: ignore[method-assign]

    async def timeout(_completion_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        transport_error = httpx.ReadTimeout("read deadline", request=request)
        raise openai.APITimeoutError(request=request) from transport_error

    llm._acreate_chat_raw = timeout  # type: ignore[assignment]

    with pytest.raises(LLMRequestTimeoutError) as exc_info:
        await llm.call(
            prompt="slow request",
            logging_path=tmp_path / "debug.json",
            timeout=45.0,
        )

    assert calls == 1
    assert exc_info.value.request_id.startswith("harbor-")
    assert exc_info.value.session_id == "trial-17"
    assert exc_info.value.request_sequence == 1
    assert exc_info.value.timeout_phase == "read"
    assert exc_info.value.connect_timeout_sec == 45.0
    assert exc_info.value.read_timeout_sec == 45.0
    assert exc_info.value.write_timeout_sec == 45.0
    assert exc_info.value.pool_timeout_sec == 45.0
    assert isinstance(exc_info.value.__cause__, openai.APITimeoutError)
    assert "LLMRequestTimeoutError" in caplog.text
    assert exc_info.value.request_id in caplog.text
    assert "session_id=trial-17" in caplog.text
    assert len(episode_logs) == 1
    assert episode_logs[0][2]["error"]["exception_type"] == "LLMRequestTimeoutError"
    assert episode_logs[0][2]["error"]["timeout_phase"] == "read"
    assert episode_logs[0][2]["error"]["read_timeout_sec"] == 45.0


@pytest.mark.asyncio
async def test_responses_timeout_uses_same_typed_boundary():
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-23",
        use_responses_api=True,
    )
    calls = 0

    async def timeout(responses_kwargs):
        nonlocal calls
        calls += 1
        assert responses_kwargs["extra_headers"]["X-Request-Id"].startswith("harbor-")
        request = httpx.Request("POST", "http://localhost/v1/responses")
        transport_error = httpx.ReadTimeout("read deadline", request=request)
        raise openai.APITimeoutError(request=request) from transport_error

    llm._acreate_response = timeout  # type: ignore[assignment]

    with pytest.raises(LLMRequestTimeoutError) as exc_info:
        await llm.call(prompt="slow response", timeout=30.0)

    assert calls == 1
    assert exc_info.value.timeout_phase == "read"
    assert exc_info.value.read_timeout_sec == 30.0


# ===== Context-length error mapping (openai.BadRequestError -> harbor) =====


@pytest.mark.asyncio
async def test_litellm_raises_context_length_for_vllm_error():
    """vLLM-style context errors (400) trigger Harbor's fallback handling."""
    llm = LiteLLM(model_name="fake-provider/fake-model")
    _install_fake_chat(
        llm,
        raise_exc=_bad_request(
            "This model's maximum context length is 32768 tokens. "
            "However, your request has 33655 input tokens.",
            body={"error": {"code": "context_length_exceeded"}},
        ),
    )
    with pytest.raises(ContextLengthExceededError):
        await llm.call(prompt="hello", message_history=[])


@pytest.mark.asyncio
async def test_litellm_raises_context_length_for_anthropic_prompt_too_long():
    """Anthropic-proxy context errors trigger Harbor's fallback handling."""
    llm = LiteLLM(model_name="openai/anthropic-proxy-model")
    _install_fake_chat(
        llm,
        raise_exc=_bad_request(
            "anthropic error: prompt is too long: 205371 tokens > 200000 maximum"
        ),
    )
    with pytest.raises(ContextLengthExceededError):
        await llm.call(prompt="hello", message_history=[])


@pytest.mark.asyncio
async def test_litellm_raises_context_length_for_bedrock_input_too_long():
    """Bedrock-proxy context errors trigger Harbor's fallback handling."""
    llm = LiteLLM(model_name="openai/bedrock-proxy-model")
    _install_fake_chat(
        llm,
        raise_exc=_bad_request("bedrock error: Input is too long for requested model."),
    )
    with pytest.raises(ContextLengthExceededError):
        await llm.call(prompt="hello", message_history=[])


@pytest.mark.asyncio
async def test_litellm_translates_max_tokens_zero_to_context_signal():
    """vLLM rejects a request whose effective output allowance is zero with a
    structured ``max_tokens must be at least 1, got 0`` 400. Harbor translates it
    to ContextBudgetExceededError so it cannot be mistaken for input overflow.
    """
    llm = LiteLLM(model_name="fake-provider/some-model")
    payload = {
        "error": {
            "message": "max_tokens must be at least 1, got 0. "
            "(parameter=max_tokens, value=0)",
            "type": "BadRequestError",
            "param": "max_tokens",
            "code": 400,
        }
    }
    _install_fake_chat(llm, raise_exc=_bad_request(str(payload), body=payload))

    with pytest.raises(ContextBudgetExceededError) as exc_info:
        await llm.call(prompt="hello", message_history=[])

    assert isinstance(exc_info.value.__cause__, openai.BadRequestError)


@pytest.mark.asyncio
async def test_litellm_rejects_nonpositive_output_budget_before_dispatch():
    llm = LiteLLM(model_name="fake-provider/some-model")
    _install_fake_chat(llm, raise_exc=AssertionError("transport was called"))

    with pytest.raises(ContextBudgetExceededError, match="got 0"):
        await llm.call(prompt="hello", message_history=[], max_tokens=0)


@pytest.mark.asyncio
async def test_litellm_unrelated_bad_request_stays_bad_request():
    """A 400 that is NOT the context-budget-exhausted signal must remain a
    BadRequestError — the narrow mapping must not over-broaden and hide a
    genuine misconfiguration as infrastructure.
    """
    llm = LiteLLM(model_name="fake-provider/some-model")
    _install_fake_chat(
        llm,
        raise_exc=_bad_request(
            "Invalid value for 'temperature': must be between 0 and 2."
        ),
    )

    with pytest.raises(openai.BadRequestError):
        await llm.call(prompt="hello", message_history=[])


@pytest.mark.asyncio
async def test_litellm_authentication_error_is_not_retried_and_reraised():
    """openai.AuthenticationError is excluded from retry and re-raised as-is."""
    llm = LiteLLM(model_name="openai/some-model")
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    resp = httpx.Response(401, request=req, json={})
    _install_fake_chat(
        llm, raise_exc=openai.AuthenticationError("bad key", response=resp, body=None)
    )
    with pytest.raises(openai.AuthenticationError):
        await llm.call(prompt="hello", message_history=[])


# ===== get_model_context_limit / get_model_output_limit (registry) =====


def test_litellm_get_model_context_limit():
    max_input_tokens = 200000
    llm = LiteLLM(
        model_name="test-integration/context-limit-model",
        model_info={"max_input_tokens": max_input_tokens, "max_output_tokens": 8192},
    )
    assert llm.get_model_context_limit() == max_input_tokens


def test_litellm_get_model_context_limit_fallback_to_max_tokens():
    max_tokens = 100000
    llm = LiteLLM(
        model_name="test-integration/legacy-model",
        model_info={"max_tokens": max_tokens},
    )
    assert llm.get_model_context_limit() == max_tokens


def test_litellm_get_model_context_limit_ultimate_fallback(caplog):
    model_name = "test-integration/no-context-limit-model"
    llm = LiteLLM(model_name=model_name)
    with caplog.at_level("DEBUG"):
        assert llm.get_model_context_limit() == 1000000
    assert any(
        "Failed to retrieve model info" in record.message
        and model_name in record.message
        for record in caplog.records
    )


def test_litellm_get_model_output_limit():
    max_output_tokens = 8192
    llm = LiteLLM(
        model_name="test-integration/output-limit-model",
        model_info={"max_input_tokens": 200000, "max_output_tokens": max_output_tokens},
    )
    assert llm.get_model_output_limit() == max_output_tokens


def test_litellm_get_model_output_limit_gpt4():
    """GPT-4's output limit is retrieved from the (retained) litellm registry."""
    llm = LiteLLM(model_name="gpt-4")
    assert llm.get_model_output_limit() == 4096


def test_litellm_get_model_output_limit_not_available(caplog):
    model_name = "test-integration/no-output-limit-model"
    llm = LiteLLM(model_name=model_name, model_info={"max_input_tokens": 100000})
    assert llm.get_model_output_limit() is None
    assert any(
        "missing max_output_tokens field" in record.message
        and model_name in record.message
        for record in caplog.records
    )


def test_litellm_get_model_output_limit_no_model_info(caplog):
    model_name = "test-integration/unknown-model"
    llm = LiteLLM(model_name=model_name)
    assert llm.get_model_output_limit() is None
    assert any(
        "Failed to retrieve model info" in record.message
        and model_name in record.message
        for record in caplog.records
    )


# ===== Responses API Tests =====


def _make_responses_api_response(
    text="Hello, world!",
    response_id="resp_abc123",
    input_tokens=10,
    output_tokens=5,
    status="completed",
    incomplete_details=None,
    model="fake-model",
):
    content_part = SimpleNamespace(type="output_text", text=text)
    message_item = SimpleNamespace(type="message", content=[content_part])
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(
        id=response_id,
        output=[message_item],
        usage=usage,
        status=status,
        incomplete_details=incomplete_details,
        model=model,
    )


@pytest.mark.asyncio
async def test_litellm_responses_api_basic_call():
    """use_responses_api=True hits the responses seam, not the chat seam."""
    captured: dict = {}
    llm = LiteLLM(model_name="fake-provider/fake-model", use_responses_api=True)
    _install_fake_responses(
        llm, response=_make_responses_api_response(), capture=captured
    )

    chat_called = {"hit": False}

    async def _fake_chat(_):
        chat_called["hit"] = True
        return _chat_response()

    llm._acreate_chat_raw = _fake_chat  # type: ignore[assignment]

    response = await llm.call(prompt="hello", message_history=[])

    assert not chat_called["hit"]
    assert response.content == "Hello, world!"
    assert response.response_id == "resp_abc123"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert captured["input"] == [{"role": "user", "content": "hello"}]
    # Provider prefix is stripped: the endpoint receives the canonical name.
    assert captured["model"] == "fake-model"


@pytest.mark.asyncio
async def test_litellm_responses_api_with_previous_response_id():
    captured: dict = {}
    llm = LiteLLM(model_name="fake-provider/fake-model", use_responses_api=True)
    _install_fake_responses(
        llm,
        response=_make_responses_api_response(response_id="resp_def456"),
        capture=captured,
    )
    response = await llm.call(
        prompt="follow up", message_history=[], previous_response_id="resp_abc123"
    )
    assert captured["previous_response_id"] == "resp_abc123"
    assert captured["input"] == "follow up"
    assert response.response_id == "resp_def456"


@pytest.mark.asyncio
async def test_litellm_responses_api_with_message_history():
    captured: dict = {}
    llm = LiteLLM(model_name="fake-provider/fake-model", use_responses_api=True)
    _install_fake_responses(
        llm, response=_make_responses_api_response(), capture=captured
    )
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]
    await llm.call(prompt="second question", message_history=history)
    assert captured["input"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    assert "previous_response_id" not in captured


@pytest.mark.asyncio
async def test_litellm_responses_api_context_length_error():
    llm = LiteLLM(model_name="fake-provider/fake-model", use_responses_api=True)
    _install_fake_responses(
        llm,
        raise_exc=_bad_request("This model's maximum context length is 8192 tokens."),
    )
    with pytest.raises(ContextLengthExceededError):
        await llm.call(prompt="hello", message_history=[])


@pytest.mark.asyncio
async def test_litellm_responses_api_output_length_error():
    llm = LiteLLM(model_name="fake-provider/fake-model", use_responses_api=True)
    _install_fake_responses(
        llm,
        response=_make_responses_api_response(
            text="partial output...",
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        ),
    )
    with pytest.raises(OutputLengthExceededError) as exc_info:
        await llm.call(prompt="hello", message_history=[])
    assert exc_info.value.truncated_response == "partial output..."
    assert exc_info.value.partial_response is not None
    assert exc_info.value.partial_response.content == "partial output..."


@pytest.mark.asyncio
async def test_litellm_chat_output_length_error_preserves_rollout_evidence():
    llm = LiteLLM(model_name="openai/fake-model", collect_rollout_details=True)
    response = _chat_response(content="partial output...", finish_reason="length")
    response["prompt_token_ids"] = [10, 11]
    response["choices"][0]["provider_specific_fields"] = {
        "token_ids": [20, 21],
    }
    response["choices"][0]["logprobs"] = {
        "content": [
            {"token": "partial", "logprob": -0.1},
            {"token": " output", "logprob": -0.2},
        ]
    }
    _install_fake_chat(llm, response=response)

    with pytest.raises(OutputLengthExceededError) as exc_info:
        await llm.call(prompt="hello", message_history=[])

    assert exc_info.value.partial_response is not None
    assert exc_info.value.partial_response.content == "partial output..."
    assert exc_info.value.partial_response.prompt_token_ids == [10, 11]
    assert exc_info.value.partial_response.completion_token_ids == [20, 21]
    assert exc_info.value.partial_response.logprobs == [-0.1, -0.2]


@pytest.mark.asyncio
async def test_litellm_response_model_name_reflects_proxy_rewrite():
    """model_name in LLMResponse comes from the response, not the request config."""
    llm = LiteLLM(model_name="proxy/configured-model")
    _install_fake_chat(
        llm, response=_chat_response(content="hello", model="actual-model-from-proxy")
    )
    response = await llm.call(prompt="hi", message_history=[])
    assert response.model_name == "actual-model-from-proxy"


@pytest.mark.asyncio
async def test_litellm_default_temperature_is_omitted():
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model")
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert "temperature" not in captured


@pytest.mark.asyncio
async def test_litellm_temperature_is_forwarded():
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model", temperature=0.7)
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert captured["temperature"] == 0.7


@pytest.mark.asyncio
async def test_litellm_call_temperature_overrides_constructor():
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model", temperature=0.7)
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[], temperature=0.2)
    assert captured["temperature"] == 0.2


@pytest.mark.asyncio
async def test_litellm_session_id_is_forwarded_in_extra_body():
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model", session_id="test-session-12345")
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert captured["extra_body"]["session_id"] == "test-session-12345"


@pytest.mark.asyncio
async def test_litellm_session_id_preserves_existing_extra_body():
    captured: dict = {}
    llm = LiteLLM(
        model_name="fake/model",
        session_id="test-session-12345",
        extra_body={"existing": "value"},
    )
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert captured["extra_body"] == {
        "existing": "value",
        "session_id": "test-session-12345",
    }


@pytest.mark.asyncio
async def test_litellm_omits_session_id_when_not_configured():
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model")
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert "extra_body" not in captured


@pytest.mark.asyncio
async def test_litellm_model_name_is_canonical_on_request():
    """The served model sent to the endpoint is the provider-prefix-stripped name."""
    captured: dict = {}
    llm = LiteLLM(model_name="openrouter/anthropic/claude-3")
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[])
    assert captured["model"] == "anthropic/claude-3"


@pytest.mark.asyncio
async def test_litellm_previous_response_id_dropped_on_chat_path():
    """chat.chat always forwards previous_response_id; the chat path must drop it
    (it's Responses-API-only) so it never reaches the openai chat request."""
    captured: dict = {}
    llm = LiteLLM(model_name="fake/model")
    _install_fake_chat(llm, response=_chat_response(), capture=captured)
    await llm.call(prompt="hi", message_history=[], previous_response_id="resp_x")
    assert "previous_response_id" not in captured


# ===== Extra-body rejection retry (openai path) =====


@pytest.mark.asyncio
async def test_litellm_retries_without_rejected_extra_body_params():
    """When the provider rejects return_token_ids, retry without it (no token ids)."""
    calls: list[dict] = []

    import copy

    llm = LiteLLM(model_name="openai/gpt-x", collect_rollout_details=True)

    async def _fake(completion_kwargs):
        # Deep-copy: the 400-retry scrub mutates the SAME nested extra_body dict
        # in place, so a shallow capture would show the post-scrub state.
        calls.append(copy.deepcopy(completion_kwargs))
        if len(calls) == 1:
            raise _bad_request(
                "Unrecognized request argument supplied: return_token_ids"
            )
        return _chat_response()

    llm._acreate_chat_raw = _fake  # type: ignore[assignment]
    resp = await llm.call(prompt="hi", message_history=[])
    assert resp.content == "hi"
    assert len(calls) == 2  # first attempt failed, retried once
    # First attempt carried return_token_ids; the retry scrubbed it.
    assert calls[0]["extra_body"].get("return_token_ids") is True
    assert "return_token_ids" not in calls[1].get("extra_body", {})


# ===== TIS-critical: byte-identical rollout-detail capture from raw vLLM JSON =====

# A realistic raw vLLM chat-completion response: prompt_token_ids at top level,
# completion token ids under choice.provider_specific_fields.token_ids, and the
# per-token logprobs under choice.logprobs.content[].logprob.
_RAW_VLLM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "Qwen/Qwen3",
    "prompt_token_ids": [10, 11, 12],
    "choices": [
        {
            "index": 0,
            "message": {"content": "hello world", "reasoning_content": "because"},
            "finish_reason": "stop",
            "token_ids": [13, 14, 15],
            "provider_specific_fields": {
                "token_ids": [13, 14, 15],
                "routed_experts": [[0, 1], [2, 3], [4, 5]],
            },
            "logprobs": {
                "content": [
                    {"token": "he", "logprob": -0.1, "top_logprobs": []},
                    {"token": "llo", "logprob": -0.2, "top_logprobs": []},
                    {"token": " world", "logprob": -0.3, "top_logprobs": []},
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 3},
}


# A minimal token-level chat renderer. Assistant suffixes are configurable so
# the tests cover templates whose terminal and between-turn forms differ.
_IM_START, _IM_END, _NL = 900, 901, 902
_ROLE_IDS = {"system": 910, "user": 911, "assistant": 912}


def _fake_word_ids(text: str) -> list[int]:
    return [1000 + (hash(word) % 500) for word in text.split()]


def _fake_render(
    messages,
    add_generation_prompt: bool,
    *,
    terminal_assistant_suffix: tuple[int, ...],
    between_turn_assistant_suffix: tuple[int, ...],
) -> list[int]:
    ids: list[int] = []
    for index, message in enumerate(messages):
        ids += [_IM_START, _ROLE_IDS[message["role"]], _NL]
        ids += _fake_word_ids(message["content"])
        if message["role"] == "assistant":
            is_terminal = index == len(messages) - 1 and not add_generation_prompt
            ids += (
                terminal_assistant_suffix
                if is_terminal
                else between_turn_assistant_suffix
            )
        else:
            ids += [_IM_END, _NL]
    if add_generation_prompt:
        ids += [_IM_START, _ROLE_IDS["assistant"], _NL]
    return ids


def _install_fake_template(
    monkeypatch,
    *,
    terminal_assistant_suffix: tuple[int, ...],
    between_turn_assistant_suffix: tuple[int, ...],
) -> None:
    async def fake_tokenize_chat(*, messages, add_generation_prompt, **kwargs):
        return _fake_render(
            messages,
            add_generation_prompt,
            terminal_assistant_suffix=terminal_assistant_suffix,
            between_turn_assistant_suffix=between_turn_assistant_suffix,
        )

    monkeypatch.setattr("harbor.llms.lite_llm.tokenize_chat", fake_tokenize_chat)


def _token_path_llm(model_name: str = "hosted_vllm/Qwen/Qwen3") -> LiteLLM:
    return LiteLLM(
        model_name=model_name,
        api_base="http://localhost/v1",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_assistant_suffix", "between_turn_assistant_suffix"),
    [
        ((_IM_END, _NL), (_IM_END, _NL)),
        ((_IM_END,), (_IM_END,)),
        ((_IM_END, 903), (_IM_END,)),
    ],
    ids=["multi-token-suffix", "single-token-suffix", "terminal-suffix-differs"],
)
@pytest.mark.parametrize("completion_carries_turn_end", [True, False])
async def test_continuation_prompt_matches_the_chat_template(
    monkeypatch,
    terminal_assistant_suffix,
    between_turn_assistant_suffix,
    completion_carries_turn_end,
):
    """The assembled prompt must equal what the template itself would render."""
    _install_fake_template(
        monkeypatch,
        terminal_assistant_suffix=terminal_assistant_suffix,
        between_turn_assistant_suffix=between_turn_assistant_suffix,
    )
    llm = _token_path_llm()

    first_turn = [{"role": "user", "content": "do the thing"}]
    answer = "here is the answer"
    observation = "command output"

    previous_prompt_token_ids = _fake_render(
        first_turn,
        add_generation_prompt=True,
        terminal_assistant_suffix=terminal_assistant_suffix,
        between_turn_assistant_suffix=between_turn_assistant_suffix,
    )
    completion_token_ids = _fake_word_ids(answer) + (
        [_IM_END] if completion_carries_turn_end else []
    )
    prompt_ids = await llm.build_continuation_prompt_token_ids(
        prompt=observation,
        previous_prompt_token_ids=previous_prompt_token_ids,
        previous_completion_token_ids=completion_token_ids,
    )

    expected = _fake_render(
        first_turn
        + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": observation},
        ],
        add_generation_prompt=True,
        terminal_assistant_suffix=terminal_assistant_suffix,
        between_turn_assistant_suffix=between_turn_assistant_suffix,
    )
    assert prompt_ids == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("collect_rollout_details", [False, True])
async def test_chat_without_complete_tito_data_keeps_message_history_transport(
    monkeypatch, collect_rollout_details
):
    requests: list[dict] = []

    async def fake_chat(completion_kwargs):
        requests.append(completion_kwargs)
        return _chat_response(content=f"answer {len(requests)}")

    async def fail_completion(_):
        raise AssertionError("non-TITO chat used the completion transport")

    async def fail_tokenize_chat(**kwargs):
        raise AssertionError("non-TITO chat called the tokenization endpoint")

    llm = LiteLLM(
        model_name="hosted_vllm/test-model",
        api_base="http://localhost/v1",
        collect_rollout_details=collect_rollout_details,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    llm._acreate_chat_raw = fake_chat  # type: ignore[assignment]
    llm._acreate_completion_raw = fail_completion  # type: ignore[assignment]
    monkeypatch.setattr("harbor.llms.lite_llm.tokenize_chat", fail_tokenize_chat)
    chat = Chat(llm)

    await chat.chat("first question")
    await chat.chat("second question")

    assert requests[1]["messages"] == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "second question"},
    ]


@pytest.mark.asyncio
async def test_continuation_preserves_sampled_history_when_template_rewrites_it(
    monkeypatch,
):
    """Qwen-like history rewrites must not replace already sampled token IDs."""
    responses = iter(
        [
            [1, 2, 154827, 3],
            [1, 2, 999, 3],
            [1, 2, 154827, 3, 4, 5],
        ]
    )

    async def fake_tokenize_chat(**kwargs):
        return next(responses)

    monkeypatch.setattr("harbor.llms.lite_llm.tokenize_chat", fake_tokenize_chat)
    llm = LiteLLM(
        model_name="hosted_vllm/Qwen/Qwen3",
        api_base="http://localhost/v1",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )

    prompt_ids = await llm.build_continuation_prompt_token_ids(
        prompt="observation",
        previous_prompt_token_ids=[10, 11],
        previous_completion_token_ids=[10122, 90, 154827],
    )

    assert prompt_ids == [10, 11, 10122, 90, 154827, 3, 4, 5]


@pytest.mark.asyncio
async def test_exact_token_prompt_uses_completion_transport_and_processed_logprobs(
    monkeypatch,
):
    raw_response = {
        "id": "cmpl-1",
        "model": "Qwen/Qwen3",
        "choices": [
            {
                "index": 0,
                "text": "reasoning</think>answer",
                "finish_reason": "stop",
                "prompt_token_ids": [10, 11, 10122, 90],
                "token_ids": [20, 21],
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    capture = {}
    llm = LiteLLM(
        model_name="hosted_vllm/Qwen/Qwen3",
        api_base="http://localhost/v1",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    monkeypatch.setattr(llm, "get_model_context_limit", lambda: 8192)
    monkeypatch.setattr(llm, "get_model_output_limit", lambda: 2048)
    _install_fake_completion(llm, response=raw_response, capture=capture)

    response = await llm.call(
        prompt="unused text",
        prompt_token_ids=[10, 11, 10122, 90],
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
    )

    assert capture["prompt"] == [10, 11, 10122, 90]
    assert capture["logprobs"] == 0
    assert capture["extra_body"]["add_special_tokens"] is False
    assert capture["extra_body"]["return_token_ids"] is True
    assert response.prompt_token_ids == [10, 11, 10122, 90]
    assert response.completion_token_ids == [20, 21]
    assert response.logprobs == [-0.1, -0.2]
    assert response.reasoning_content == "reasoning"
    assert response.content == "answer"


@pytest.mark.parametrize(
    ("model_name", "text", "template_kwargs", "reasoning", "content"),
    [
        pytest.param(
            "hosted_vllm/snowball",
            '<|start_think|>look first<|end_think|>\n\n{"commands": []}',
            {},
            "look first",
            '\n\n{"commands": []}',
            id="special-token-markers",
        ),
        pytest.param(
            "hosted_vllm/snowball",
            '{"commands": []}',
            {},
            None,
            '{"commands": []}',
            id="marker-free",
        ),
        pytest.param(
            "hosted_vllm/snowball",
            "<|start_think|>cut mid",
            {},
            "cut mid",
            "",
            id="unclosed-marker",
        ),
        pytest.param(
            "hosted_vllm/Qwen/Qwen3",
            "still thinking",
            {},
            "still thinking",
            "",
            id="qwen-prompt-opened",
        ),
        pytest.param(
            "hosted_vllm/Qwen/Qwen3",
            "reasoning</think>answer",
            {},
            "reasoning",
            "answer",
            id="qwen-closing-marker",
        ),
        pytest.param(
            "hosted_vllm/snowball",
            "<|start_think|>x<|end_think|>y",
            {"enable_thinking": False},
            None,
            "<|start_think|>x<|end_think|>y",
            id="thinking-disabled",
        ),
        pytest.param(
            "hosted_vllm/snowball",
            '{"commands": [{"keystrokes": "echo <think>"}]}',
            {},
            None,
            '{"commands": [{"keystrokes": "echo <think>"}]}',
            id="marker-in-action-content",
        ),
    ],
)
@pytest.mark.asyncio
async def test_exact_token_prompt_splits_reasoning_markers(
    monkeypatch,
    model_name,
    text,
    template_kwargs,
    reasoning,
    content,
):
    raw_response = {
        "id": "cmpl-2",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
                "prompt_token_ids": [10, 11],
                "token_ids": [128002, 20, 128003, 21],
                "logprobs": {"token_logprobs": [-0.1, -0.2, -0.3, -0.4]},
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 4},
    }
    llm = _token_path_llm(model_name)
    monkeypatch.setattr(llm, "get_model_context_limit", lambda: 8192)
    monkeypatch.setattr(llm, "get_model_output_limit", lambda: 2048)
    _install_fake_completion(llm, response=raw_response, capture={})
    extra_body = {"skip_special_tokens": False}
    if template_kwargs:
        extra_body["chat_template_kwargs"] = template_kwargs

    response = await llm.call(
        prompt="unused text",
        prompt_token_ids=[10, 11],
        extra_body=extra_body,
    )

    assert response.reasoning_content == reasoning
    assert response.content == content
    assert response.completion_token_ids == [128002, 20, 128003, 21]


@pytest.mark.asyncio
async def test_rollout_capture_extracts_ids_and_logprobs_from_raw_vllm_json():
    """LLMResponse carries the exact vendor token ids + logprobs vLLM returned."""
    llm = LiteLLM(
        model_name="hosted_vllm/Qwen/Qwen3",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    _install_fake_chat(llm, response=_RAW_VLLM_RESPONSE)
    resp = await llm.call(prompt="hi", message_history=[])
    assert resp.prompt_token_ids == [10, 11, 12]
    assert resp.completion_token_ids == [13, 14, 15]
    assert resp.logprobs == [-0.1, -0.2, -0.3]
    # Non-token_ids provider fields are captured as `extra` (MoE routing etc.).
    assert resp.extra == {"routed_experts": [[0, 1], [2, 3], [4, 5]]}


@pytest.mark.asyncio
async def test_chat_response_preserves_native_tool_calls():
    tool_call = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "execute_commands", "arguments": '{"commands": []}'},
    }
    response = _chat_response(content="")
    response["choices"][0]["message"]["tool_calls"] = [tool_call]
    llm = LiteLLM(
        model_name="hosted_vllm/test-model",
        api_base="http://localhost/v1",
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    _install_fake_chat(llm, response=response)

    result = await llm.call(prompt="inspect")

    assert result.tool_calls == [tool_call]


@pytest.mark.asyncio
async def test_rollout_capture_feeds_byte_identical_rollout_details():
    """End-to-end via Chat: the captured signal becomes rollout_details with the
    same per-turn token ids + logprobs, length-aligned (the TIS invariant)."""
    llm = LiteLLM(
        model_name="hosted_vllm/Qwen/Qwen3",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    _install_fake_chat(llm, response=_RAW_VLLM_RESPONSE)
    chat = Chat(model=llm)
    await chat.chat("hi")
    details = chat.rollout_details
    assert len(details) == 1
    assert details[0]["prompt_token_ids"] == [[10, 11, 12]]
    assert details[0]["completion_token_ids"] == [[13, 14, 15]]
    # logprobs length-aligned 1:1 with completion_token_ids for this turn.
    assert details[0]["logprobs"] == [[-0.1, -0.2, -0.3]]
    assert details[0]["extra"]["routed_experts"] == [[[0, 1], [2, 3], [4, 5]]]


@pytest.mark.asyncio
async def test_rollout_capture_token_ids_only_on_choice_key():
    """Some vLLM builds surface token_ids directly on the choice (no
    provider_specific_fields) — still captured verbatim."""
    raw = {
        "model": "m",
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "message": {"content": "x", "reasoning_content": None},
                "finish_reason": "stop",
                "token_ids": [99, 100],
                "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.6}]},
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
    }
    llm = LiteLLM(
        model_name="hosted_vllm/m",
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 2048,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
    )
    _install_fake_chat(llm, response=raw)
    resp = await llm.call(prompt="hi", message_history=[])
    assert resp.completion_token_ids == [99, 100]
    assert resp.logprobs == [-0.5, -0.6]


# ===== _extract_provider_extra (raw-dict) Tests =====


def test_extract_provider_extra_with_extra_fields():
    llm = LiteLLM(model_name="fake/model", collect_rollout_details=True)
    response = {
        "choices": [
            {
                "provider_specific_fields": {
                    "token_ids": [1, 2, 3],
                    "routed_experts": [[0, 1], [2, 3]],
                    "router_logits": [0.5, 0.8],
                }
            }
        ]
    }
    result = llm._extract_provider_extra(response)
    assert result == {"routed_experts": [[0, 1], [2, 3]], "router_logits": [0.5, 0.8]}
    assert "token_ids" not in result


def test_extract_provider_extra_only_token_ids():
    llm = LiteLLM(model_name="fake/model", collect_rollout_details=True)
    response = {"choices": [{"provider_specific_fields": {"token_ids": [1, 2, 3]}}]}
    assert llm._extract_provider_extra(response) is None


def test_extract_provider_extra_no_provider_fields():
    llm = LiteLLM(model_name="fake/model", collect_rollout_details=True)
    response = {"choices": [{"message": {"content": "x"}}]}
    assert llm._extract_provider_extra(response) is None


def test_extract_provider_extra_empty_choices():
    llm = LiteLLM(model_name="fake/model", collect_rollout_details=True)
    assert llm._extract_provider_extra({"choices": []}) is None


# ---------------------------------------------------------------------------
# get_model_context_limit: vLLM max_model_len preference + global caching.
# ---------------------------------------------------------------------------


def test_get_model_context_limit_uses_vllm_when_registry_missing(monkeypatch):
    llm = LiteLLM(
        model_name="test-integration/unknown-hash-model",
        api_base="http://localhost:18000/v1",
    )
    monkeypatch.setattr(llm, "_get_vllm_max_model_len", lambda: 32768)
    # The live vLLM limit is shaved by _VLLM_CONTEXT_SAFETY_BUFFER (default 8)
    # so a +/-1 BPE boundary prompt can't render to max_model_len + 1.
    from harbor.llms.lite_llm import _VLLM_CONTEXT_SAFETY_BUFFER

    assert llm.get_model_context_limit() == 32768 - _VLLM_CONTEXT_SAFETY_BUFFER


def test_get_model_context_limit_takes_min_of_registry_and_vllm(monkeypatch):
    llm = LiteLLM(
        model_name="test-integration/dual-model",
        model_info={"max_input_tokens": 32000, "max_output_tokens": 8192},
    )
    monkeypatch.setattr(llm, "_get_vllm_max_model_len", lambda: 32768)
    assert llm.get_model_context_limit() == 32000


def test_get_vllm_max_model_len_none_without_api_base():
    llm = LiteLLM(model_name="test-integration/no-base")
    assert llm._get_vllm_max_model_len() is None


def test_get_vllm_max_model_len_parses_models_endpoint(monkeypatch):
    import httpx as _httpx

    captured = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "served-hash", "max_model_len": 32768}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            captured["url"] = url
            return _FakeResp()

    monkeypatch.setattr(_httpx, "Client", _FakeClient)
    llm = LiteLLM(
        model_name="test-integration/served",
        api_base="http://probe-host:8000/v1",
    )
    assert llm._get_vllm_max_model_len() == 32768
    assert captured["url"] == "http://probe-host:8000/v1/models"
    # cached on second call (no second request needed)
    assert llm._get_vllm_max_model_len() == 32768


def test_vllm_max_model_len_probe_is_global_cached(monkeypatch):
    """A SECOND, fresh LiteLLM instance pointing at the same endpoint reuses the
    process-global probe result and issues NO HTTP GET (the coordinator-loop
    probe-elimination fix)."""
    import httpx as _httpx

    probe_count = {"n": 0}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": [{"max_model_len": 40960}]}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            probe_count["n"] += 1
            return _FakeResp()

    monkeypatch.setattr(_httpx, "Client", _FakeClient)
    base = "http://global-cache-host:8000/v1"
    a = LiteLLM(model_name="test-integration/gc-a", api_base=base)
    b = LiteLLM(model_name="test-integration/gc-b", api_base=base)
    assert a._get_vllm_max_model_len() == 40960
    assert b._get_vllm_max_model_len() == 40960
    # Only ONE HTTP probe for the endpoint across both instances.
    assert probe_count["n"] == 1


# ---------------------------------------------------------------------------
# Transport router: AsyncOpenAI (OpenAI-compat) vs litellm.acompletion (native).
#
# The router is a STATIC, HTTP-free string check made once in __init__. These
# tests pin (a) the pure predicate, (b) that the right transport SEAM is hit per
# model/api_base, and (c) that computing the route issues NO network call.
# ---------------------------------------------------------------------------


def _install_route_recorders(llm):
    """Install distinct fakes on both seams; return a dict recording which fired."""
    hit = {"openai": 0, "litellm": 0}

    async def _fake_openai(_kwargs):
        hit["openai"] += 1
        return _chat_response()

    async def _fake_litellm(_kwargs):
        hit["litellm"] += 1
        return _chat_response()

    llm._acreate_chat_raw = _fake_openai  # type: ignore[assignment]
    llm._acreate_chat_litellm = _fake_litellm  # type: ignore[assignment]
    return hit


def test_route_predicate_openai_compat_and_native():
    """The pure predicate proves OpenAI-compat only for hosted_vllm/openai
    prefixes or an explicit /v1 api_base; everything else is litellm."""
    from harbor.llms.lite_llm import _is_openai_compatible_route

    # OpenAI-compatible.
    assert _is_openai_compatible_route("hosted_vllm", None) is True
    assert _is_openai_compatible_route("openai", None) is True
    assert _is_openai_compatible_route(None, "http://host:8000/v1") is True
    assert _is_openai_compatible_route("served", "http://host:8000/v1/") is True
    assert _is_openai_compatible_route("anthropic", "http://proxy:8000/v1") is True
    # Native / ambiguous -> litellm fallback (safe default).
    assert _is_openai_compatible_route("anthropic", None) is False
    assert _is_openai_compatible_route("gemini", None) is False
    assert _is_openai_compatible_route("vertex_ai", None) is False
    assert _is_openai_compatible_route("bedrock", None) is False
    assert _is_openai_compatible_route("some-unknown-provider", None) is False
    assert _is_openai_compatible_route(None, None) is False
    # An api_base that is NOT an OpenAI /v1 endpoint is not a proof of compat.
    assert (
        _is_openai_compatible_route("anthropic", "https://api.anthropic.com") is False
    )


_HOSTED_VLLM_INFO = {
    "max_input_tokens": 8192,
    "max_output_tokens": 2048,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
}


@pytest.mark.asyncio
async def test_route_hosted_vllm_uses_asyncopenai():
    llm = LiteLLM(model_name="hosted_vllm/Qwen/Qwen3", model_info=_HOSTED_VLLM_INFO)
    assert llm._use_openai_transport is True
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 1, "litellm": 0}


@pytest.mark.asyncio
async def test_route_openai_prefix_uses_asyncopenai():
    llm = LiteLLM(model_name="openai/gpt-4o")
    assert llm._use_openai_transport is True
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 1, "litellm": 0}


@pytest.mark.asyncio
async def test_route_vllm_v1_api_base_uses_asyncopenai():
    """A served-vLLM model with no known prefix but an explicit /v1 api_base is
    OpenAI-compatible and takes the AsyncOpenAI fast path."""
    llm = LiteLLM(
        model_name="served/my-checkpoint", api_base="http://127.0.0.1:8000/v1"
    )
    assert llm._use_openai_transport is True
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 1, "litellm": 0}


@pytest.mark.asyncio
async def test_route_native_anthropic_uses_litellm():
    llm = LiteLLM(model_name="anthropic/claude-sonnet-4-5-20250929")
    assert llm._use_openai_transport is False
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 0, "litellm": 1}


@pytest.mark.asyncio
async def test_route_native_gemini_uses_litellm():
    llm = LiteLLM(model_name="gemini/gemini-2.5-pro")
    assert llm._use_openai_transport is False
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 0, "litellm": 1}


@pytest.mark.asyncio
async def test_route_unknown_provider_defaults_to_litellm():
    """Ambiguity (unknown provider, no /v1 api_base) falls back to litellm — we
    never risk sending a possibly non-OpenAI request to AsyncOpenAI."""
    llm = LiteLLM(model_name="totally-unknown-provider/some-model")
    assert llm._use_openai_transport is False
    hit = _install_route_recorders(llm)
    await llm.call(prompt="hi", message_history=[])
    assert hit == {"openai": 0, "litellm": 1}


def test_routing_decision_issues_no_http(monkeypatch):
    """Computing the route (in __init__) must not touch the network — blow up any
    httpx client construction and assert the route still resolves."""
    import httpx as _httpx

    def _boom(*a, **k):
        raise AssertionError("routing must not issue any HTTP request")

    monkeypatch.setattr(_httpx, "Client", _boom)
    monkeypatch.setattr(_httpx, "AsyncClient", _boom)

    # Native and OpenAI-compat construction both resolve a route with zero HTTP.
    assert LiteLLM(model_name="anthropic/claude-x")._use_openai_transport is False
    assert (
        LiteLLM(
            model_name="hosted_vllm/m", model_info=_HOSTED_VLLM_INFO
        )._use_openai_transport
        is True
    )
    assert (
        LiteLLM(
            model_name="served/x", api_base="http://h:8000/v1"
        )._use_openai_transport
        is True
    )


def _truncated_completion_response(*, prompt_len: int, completion_len: int) -> dict:
    return {
        "id": "cmpl-1",
        "model": "Qwen/Qwen3",
        "choices": [
            {
                "index": 0,
                "text": "unfinished answer",
                "finish_reason": "length",
                "prompt_token_ids": list(range(prompt_len)),
                "token_ids": list(range(completion_len)),
                "logprobs": {"token_logprobs": [-0.1] * completion_len},
            }
        ],
        "usage": {"prompt_tokens": prompt_len, "completion_tokens": completion_len},
    }


@pytest.mark.parametrize(
    (
        "prompt_len",
        "completion_len",
        "requested_max_tokens",
        "effective_max_tokens",
        "error_type",
    ),
    [
        pytest.param(8000, 192, None, 192, ContextLengthExceededError, id="context"),
        pytest.param(100, 2048, None, 2048, OutputLengthExceededError, id="output"),
        pytest.param(6144, 2048, None, 2048, ContextLengthExceededError, id="tie"),
        pytest.param(100, 64, 64, 64, OutputLengthExceededError, id="caller-limit"),
    ],
)
@pytest.mark.asyncio
async def test_token_completion_length_stop_uses_the_exhausted_budget(
    monkeypatch,
    prompt_len,
    completion_len,
    requested_max_tokens,
    effective_max_tokens,
    error_type,
):
    llm = _token_path_llm()
    monkeypatch.setattr(llm, "get_model_context_limit", lambda: 8192)
    monkeypatch.setattr(llm, "get_model_output_limit", lambda: 2048)
    capture: dict = {}
    _install_fake_completion(
        llm,
        response=_truncated_completion_response(
            prompt_len=prompt_len,
            completion_len=completion_len,
        ),
        capture=capture,
    )
    call_kwargs = {}
    if requested_max_tokens is not None:
        call_kwargs["max_tokens"] = requested_max_tokens

    with pytest.raises(error_type) as exc_info:
        await llm.call(
            prompt="unused text",
            prompt_token_ids=list(range(prompt_len)),
            **call_kwargs,
        )

    assert capture["max_tokens"] == effective_max_tokens
    if error_type is OutputLengthExceededError:
        assert exc_info.value.partial_response.completion_token_ids == list(
            range(completion_len)
        )


class TestOpenAIConnectTimeoutEnv:
    """HARBOR_OPENAI_CONNECT_TIMEOUT_SEC configures the transport."""

    def test_default_when_unset(self):
        from harbor.llms.lite_llm import _openai_connect_timeout_sec

        assert _openai_connect_timeout_sec({}) == 20.0

    def test_env_override(self):
        from harbor.llms.lite_llm import _openai_transport_timeout

        timeout = _openai_transport_timeout({"HARBOR_OPENAI_CONNECT_TIMEOUT_SEC": "30"})

        assert timeout.connect == 30.0
        assert timeout.read is None
        assert timeout.write == 600.0
        assert timeout.pool == 600.0

    @pytest.mark.parametrize("bad", ["", "abc", "0", "-3", "nan", "inf"])
    def test_invalid_value_is_rejected(self, bad):
        from harbor.llms.lite_llm import _openai_connect_timeout_sec

        with pytest.raises(ValueError, match="must be a positive finite number"):
            _openai_connect_timeout_sec({"HARBOR_OPENAI_CONNECT_TIMEOUT_SEC": bad})

    def test_transport_timeout_uses_it(self):
        from harbor.llms.lite_llm import _OPENAI_TRANSPORT_TIMEOUT

        assert _OPENAI_TRANSPORT_TIMEOUT.connect == 20.0
        assert _OPENAI_TRANSPORT_TIMEOUT.read is None


# ---------------------------------------------------------------------------
# Connect-phase timeouts (Jupiter RL, 2026-09-05): the coordinator's event loop
# stalls for minutes and every pending connect deadline expires at once. Nothing
# reached the server, so the request is retried in-attempt instead of failing
# the whole trial.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_timeout_is_retried_in_attempt_and_then_succeeds(monkeypatch):
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-connect",
    )
    calls = 0
    response = {
        "id": "resp",
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }

    async def flaky_connect(_completion_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", "http://localhost/v1/chat/completions")
            transport_error = httpx.ConnectTimeout("connect deadline", request=request)
            raise openai.APITimeoutError(request=request) from transport_error
        return response

    llm._acreate_chat_raw = flaky_connect  # type: ignore[assignment]
    # tenacity's wait is a plain asyncio.sleep; skip the 4 s backoff.
    monkeypatch.setattr(
        "harbor.llms.lite_llm.LiteLLM.call.retry.sleep",
        lambda _seconds: asyncio.sleep(0),
        raising=False,
    )

    result = await llm.call(prompt="hello", timeout=45.0)

    assert calls == 2
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_connect_timeout_gives_up_after_the_bounded_retries(monkeypatch):
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-connect",
    )
    calls = 0

    async def always_connect_timeout(_completion_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        transport_error = httpx.ConnectTimeout("connect deadline", request=request)
        raise openai.APITimeoutError(request=request) from transport_error

    llm._acreate_chat_raw = always_connect_timeout  # type: ignore[assignment]
    monkeypatch.setattr(
        "harbor.llms.lite_llm.LiteLLM.call.retry.sleep",
        lambda _seconds: asyncio.sleep(0),
        raising=False,
    )

    with pytest.raises(OpenAITransportConnectTimeoutError) as exc_info:
        await llm.call(prompt="hello", timeout=45.0)

    assert calls == 3
    assert exc_info.value.timeout_phase == "connect"
    assert isinstance(exc_info.value, LLMRequestTimeoutError)


def test_timeout_phase_matches_a_vendored_httpx_connect_timeout():
    from harbor.llms.lite_llm import _timeout_phase

    connect_timeout = type(
        "ConnectTimeout", (Exception,), {"__module__": "httpx2._exceptions"}
    )
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    error = openai.APITimeoutError(request=request)
    error.__cause__ = connect_timeout()

    assert _timeout_phase(error) == "connect"


def test_timeout_phase_rejects_unrelated_same_named_exception():
    from harbor.llms.lite_llm import _timeout_phase

    class ConnectTimeout(Exception):
        pass

    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    error = openai.APITimeoutError(request=request)
    error.__cause__ = ConnectTimeout()

    assert _timeout_phase(error) == "unknown"


@pytest.mark.asyncio
async def test_unrelated_connect_timeout_name_is_not_retyped_or_retried():
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-connect",
    )
    calls = 0

    class ConnectTimeout(Exception):
        pass

    async def unrelated_timeout(_completion_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        raise openai.APITimeoutError(request=request) from ConnectTimeout()

    llm._acreate_chat_raw = unrelated_timeout  # type: ignore[assignment]

    with pytest.raises(LLMRequestTimeoutError) as exc_info:
        await llm.call(prompt="hello", timeout=45.0)

    assert type(exc_info.value) is LLMRequestTimeoutError
    assert exc_info.value.timeout_phase == "unknown"
    assert calls == 1


@pytest.mark.asyncio
async def test_generic_connect_phase_timeout_is_not_retried_by_string_label():
    llm = LiteLLM(
        model_name="openai/test-model",
        api_base="http://localhost/v1",
        session_id="trial-connect",
    )
    timeout_error = LLMRequestTimeoutError(
        request_id="request",
        session_id="trial-connect",
        request_sequence=1,
        model_name="openai/test-model",
        timeout_phase="connect",
        connect_timeout_sec=20.0,
        read_timeout_sec=None,
        write_timeout_sec=600.0,
        pool_timeout_sec=600.0,
    )
    calls = 0

    async def generic_timeout(_completion_kwargs):
        nonlocal calls
        calls += 1
        raise timeout_error

    llm._acreate_chat_raw = generic_timeout  # type: ignore[assignment]

    with pytest.raises(LLMRequestTimeoutError) as exc_info:
        await llm.call(prompt="hello")

    assert exc_info.value is timeout_error
    assert calls == 1
