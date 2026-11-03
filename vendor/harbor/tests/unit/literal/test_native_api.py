import asyncio
import json

import httpx
import httpx2
import pytest
from anthropic import AsyncAnthropic
from harbor.literal.native_api import (
    NativeAPILimits,
    RecordingCompletionTransport,
    anthropic_chat_request,
    responses_chat_request,
)
from harbor.literal.proxy import CORRELATION_ID_HEADER, TASK_AGENT_HEADER, RecordProxy
from openai import AsyncOpenAI


@pytest.mark.asyncio
async def test_record_proxy_releases_its_upstream_client(tmp_path, monkeypatch):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"count": 3})
        )
    )
    monkeypatch.setattr(
        "harbor.literal.proxy.httpx.AsyncClient", lambda **kwargs: client
    )
    proxy = RecordProxy("http://upstream", tmp_path / "literal.jsonl")
    try:
        status, body, _ = await proxy.forward("/tokenize", {"messages": []})
        assert (status, body) == (200, {"count": 3})
    finally:
        await proxy.aclose()
    assert client.is_closed


def native_request(protocol: str, *, followup: bool) -> dict:
    if protocol == "anthropic":
        messages = [{"role": "user", "content": "Run bash"}]
        if followup:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-one",
                                "name": "bash",
                                "input": {"command": "echo ok"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-one",
                                "content": "ok",
                            }
                        ],
                    },
                ]
            )
        return {
            "model": "bfcl-test",
            "messages": messages,
            "max_tokens": 8192,
            "tools": [
                {
                    "name": "bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
        }
    inputs = [{"role": "user", "content": "Run bash"}]
    if followup:
        inputs.extend(
            [
                {
                    "type": "function_call",
                    "call_id": "call-one",
                    "name": "bash",
                    "arguments": '{"command":"echo ok"}',
                },
                {"type": "function_call_output", "call_id": "call-one", "output": "ok"},
            ]
        )
    return {
        "model": "bfcl-test",
        "input": inputs,
        "max_output_tokens": 8192,
        "tools": [
            {
                "type": "function",
                "name": "bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
async def test_native_tool_history_uses_recorded_sdk_completions(tmp_path, protocol):
    received = []

    async def backend(request):
        body = json.loads(request.content)
        received.append(body)
        assert request.headers[CORRELATION_ID_HEADER] == "trial-one"
        assert request.headers[TASK_AGENT_HEADER] == protocol
        assert body["return_token_ids"] is True
        followup = len(received) == 2
        if followup:
            assert body["messages"][-1] == {
                "role": "tool",
                "content": "ok",
                "tool_call_id": "call-one",
            }
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(received)}",
                "object": "chat.completion",
                "created": 1,
                "model": "bfcl-test",
                "prompt_token_ids": [1, 2, 10, 90] if followup else [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" if followup else "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Done" if followup else "Calling bash",
                            "tool_calls": (
                                []
                                if followup
                                else [
                                    {
                                        "id": "call-one",
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"echo ok"}',
                                        },
                                    }
                                ]
                            ),
                        },
                        "provider_specific_fields": {
                            "token_ids": [11] if followup else [10]
                        },
                        "logprobs": {"content": [{"token": "Done", "logprob": -0.25}]},
                    }
                ],
                "usage": {
                    "prompt_tokens": 4 if followup else 2,
                    "completion_tokens": 1,
                    "total_tokens": 5 if followup else 3,
                },
            },
        )

    path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm", path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        async with AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://literal/v1",
            http_client=httpx.AsyncClient(
                transport=RecordingCompletionTransport(
                    proxy, trial_id="trial-one", task_agent=protocol
                )
            ),
            max_retries=0,
        ) as client:
            for followup in (False, True):
                body = native_request(protocol, followup=followup)
                chat = (
                    anthropic_chat_request(body)
                    if protocol == "anthropic"
                    else responses_chat_request(body)
                )
                chat["model"] = "bfcl-test"
                result = await client.chat.completions.create(**chat)
                if followup:
                    assert result.choices[0].message.content == "Done"
                else:
                    call = result.choices[0].message.tool_calls[0]
                    assert (
                        call.id,
                        call.function.name,
                        json.loads(call.function.arguments),
                    ) == ("call-one", "bash", {"command": "echo ok"})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["trial_id"] for r in records] == ["trial-one", "trial-one"]
    assert (
        records[0]["literal"]["prompt_token_ids"]
        + records[0]["literal"]["completion_token_ids"]
        == records[1]["literal"]["prompt_token_ids"][:-1]
    )
    assert [r["literal"]["completion_token_ids"] for r in records] == [[10], [11]]
    assert [r["literal"]["logprobs"] for r in records] == [[-0.25], [-0.25]]
    assert records[0]["literal"]["assistant_message"]["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command":"echo ok"}',
    }
    assert records[1]["literal"]["assistant_message"]["content"] == "Done"


@pytest.mark.asyncio
async def test_sdk_stream_yields_before_completion_and_retains_exact_tokens(tmp_path):
    first_received = asyncio.Event()
    closed = asyncio.Event()

    class BackendStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index, content in enumerate(("Hel", "lo")):
                if index:
                    await first_received.wait()
                chunk = {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "bfcl-test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": content,
                                "reasoning_content": "reason" if index == 0 else "ing",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        **(
                                            {"id": "call-one", "type": "function"}
                                            if index == 0
                                            else {}
                                        ),
                                        "function": {
                                            **({"name": "bash"} if index == 0 else {}),
                                            "arguments": (
                                                '{"command":'
                                                if index == 0
                                                else '"pwd"}'
                                            ),
                                        },
                                    }
                                ],
                                "provider_specific_fields": {"token_ids": [10 + index]},
                            },
                            "finish_reason": None,
                            "logprobs": {
                                "content": [{"token": content, "logprob": -0.25}]
                            },
                        }
                    ],
                }
                if index == 0:
                    chunk["prompt_token_ids"] = [1, 2]
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed.set()

    async def backend(request):
        assert request.headers[CORRELATION_ID_HEADER] == "stream-trial"
        assert request.headers[TASK_AGENT_HEADER] == "responses"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BackendStream()
        )

    path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm", path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        async with AsyncOpenAI(
            api_key="EMPTY",
            base_url="http://literal/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=RecordingCompletionTransport(
                    proxy,
                    trial_id="stream-trial",
                    task_agent="responses",
                )
            ),
        ) as client:
            stream = await client.chat.completions.create(
                model="bfcl-test",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )
            outputs = []
            async for chunk in stream:
                outputs.append(chunk.choices[0].delta.content)
                first_received.set()
    assert outputs == ["Hel", "lo"]
    assert closed.is_set()
    record = json.loads(path.read_text())
    assert record["trial_id"] == "stream-trial"
    assert record["literal"]["prompt_token_ids"] == [1, 2]
    assert record["literal"]["completion_token_ids"] == [10, 11]
    assert record["literal"]["logprobs"] == [-0.25, -0.25]

    assert record["literal"]["assistant_message"]["role"] == "assistant"
    assert record["literal"]["assistant_message"]["content"] == "Hello"
    assert record["literal"]["assistant_message"]["reasoning_content"] == "reasoning"
    assert record["literal"]["assistant_message"]["tool_calls"] == [
        {
            "id": "call-one",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("text_before_tool", [True, False])
async def test_native_stream_preserves_fragmented_tool_arguments(
    tmp_path, protocol, reasoning_field, text_before_tool
):
    deltas = [
        {reasoning_field: "Choose bash."},
        {"content": "Writing the answer"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call-one",
                    "type": "function",
                    "function": {"name": "bash", "arguments": ""},
                }
            ]
        },
        {},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"command": "'}}]},
        {},
        {
            "tool_calls": [
                {"index": 0, "function": {"arguments": "echo '[]' > /app/result.json"}}
            ]
        },
        {},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"}'}}]},
    ]

    if not text_before_tool:
        deltas.append(deltas.pop(1))

    class ToolStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index, delta in enumerate([*deltas, {}]):
                chunk = {
                    "id": "chatcmpl-tools",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "bfcl-test",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                **delta,
                                "provider_specific_fields": {"token_ids": [10 + index]},
                            },
                            "logprobs": {
                                "content": [{"token": "token", "logprob": -0.25}]
                            },
                            "finish_reason": (
                                "tool_calls" if index == len(deltas) else None
                            ),
                        }
                    ],
                }
                if index == 0:
                    chunk["prompt_token_ids"] = [1, 2]
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"

    async def backend(request):
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 2, "max_model_len": 8})
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=ToolStream()
        )

    proxy = RecordProxy("http://vllm", tmp_path / "literal.jsonl")
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        http_module = httpx2 if protocol == "anthropic" else httpx
        client_type = AsyncAnthropic if protocol == "anthropic" else AsyncOpenAI
        async with client_type(
            api_key="EMPTY",
            base_url=(
                "http://literal" if protocol == "anthropic" else "http://literal/v1"
            ),
            max_retries=0,
            http_client=http_module.AsyncClient(
                transport=http_module.ASGITransport(
                    app=proxy.app(NativeAPILimits("bfcl-test", 8, 8192))
                )
            ),
        ) as client:
            body = native_request(protocol, followup=False)
            if protocol == "anthropic":
                async with client.messages.stream(**body) as stream:
                    message = await stream.get_final_message()
                assert message.content[0].thinking == "Choose bash."
                assert message.usage.input_tokens == 2
                calls = [block for block in message.content if block.type == "tool_use"]
                assert len(calls) == 1
                result = (calls[0].id, calls[0].name, calls[0].input)
            else:
                async with client.responses.stream(**body) as stream:
                    response = await stream.get_final_response()
                assert response.output[0].summary[0].text == "Choose bash."
                assert response.usage.input_tokens == 2
                calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                assert len(calls) == 1
                result = (
                    calls[0].call_id,
                    calls[0].name,
                    json.loads(calls[0].arguments),
                )
    assert result == ("call-one", "bash", {"command": "echo '[]' > /app/result.json"})
    record = json.loads((tmp_path / "literal.jsonl").read_text())
    assert record["literal"]["completion_token_ids"] == list(
        range(10, 10 + len(deltas) + 1)
    )
    assert record["literal"]["assistant_message"]["tool_calls"][0]["function"][
        "arguments"
    ] == ('{"command": "echo \'[]\' > /app/result.json"}')


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("effort", [None, "high", "none"])
async def test_native_http_tool_roundtrip_preserves_recording_and_total_budget(
    tmp_path, protocol, effort
):
    calls = []

    async def backend(request):
        body = json.loads(request.content)
        if effort is not None:
            assert body["chat_template_kwargs"]["enable_thinking"] == (effort != "none")
        if request.url.path == "/tokenize":
            assert request.headers[TASK_AGENT_HEADER] == (
                "claude-code" if protocol == "anthropic" else "codex"
            )
            assert request.headers[CORRELATION_ID_HEADER] == "native-trial"
            return httpx.Response(
                200,
                json={
                    "count": 4 if len(body["messages"]) > 1 else 2,
                    "max_model_len": 8,
                },
            )
        calls.append(body)
        if effort is not None:
            assert body["reasoning_effort"] == effort
        assert "context_management" not in body
        followup = len(calls) == 2
        assert body["max_tokens"] == (4 if followup else 6)
        assert body["return_token_ids"] is True
        assert request.headers[CORRELATION_ID_HEADER] == "native-trial"
        if followup:
            assert body["messages"][-1]["role"] == "tool"
            assert body["messages"][-1]["content"] == "ok"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(calls)}",
                "object": "chat.completion",
                "created": 1,
                "model": "bfcl-test",
                "prompt_token_ids": [1, 2, 10, 90] if followup else [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop" if followup else "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Done" if followup else None,
                            "tool_calls": (
                                []
                                if followup
                                else [
                                    {
                                        "id": "call-one",
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"echo ok"}',
                                        },
                                    }
                                ]
                            ),
                        },
                        "provider_specific_fields": {
                            "token_ids": [11] if followup else [10]
                        },
                        "logprobs": {"content": [{"token": "token", "logprob": -0.25}]},
                    }
                ],
                "usage": {
                    "prompt_tokens": 4 if followup else 2,
                    "completion_tokens": 1,
                    "total_tokens": 5 if followup else 3,
                },
            },
        )

    path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm", path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        http_module = httpx2 if protocol == "anthropic" else httpx
        transport = http_module.ASGITransport(
            app=proxy.app(NativeAPILimits("bfcl-test", 8, 8192))
        )
        client_type = AsyncAnthropic if protocol == "anthropic" else AsyncOpenAI
        base_url = "http://literal" if protocol == "anthropic" else "http://literal/v1"
        async with client_type(
            api_key="EMPTY",
            base_url=base_url,
            max_retries=0,
            default_headers={CORRELATION_ID_HEADER: "native-trial"},
            http_client=http_module.AsyncClient(transport=transport),
        ) as client:
            for followup in (False, True):
                body = native_request(protocol, followup=followup)
                if effort is not None:
                    if protocol == "anthropic":
                        body["thinking"] = {
                            "type": "adaptive" if effort == "high" else "disabled"
                        }
                        body["extra_body"] = {
                            "output_config": {"effort": effort},
                            "context_management": {
                                "edits": [
                                    {"type": "clear_thinking_20251015", "keep": "all"}
                                ]
                            },
                        }
                    else:
                        body["reasoning"] = {"effort": effort}
                if protocol == "anthropic" and not followup:
                    count_request = dict(body)
                    count_request.pop("max_tokens")
                    counted = await client.messages.count_tokens(**count_request)
                    assert counted.input_tokens == 2
                output = (
                    await client.messages.create(**body)
                    if protocol == "anthropic"
                    else await client.responses.create(**body)
                )
                if followup:
                    text = (
                        output.content[0].text
                        if protocol == "anthropic"
                        else output.output_text
                    )
                    assert text == "Done"
                else:
                    call = (
                        output.content[0]
                        if protocol == "anthropic"
                        else next(
                            item
                            for item in output.output
                            if item.type == "function_call"
                        )
                    )
                    assert call.type == (
                        "tool_use" if protocol == "anthropic" else "function_call"
                    )
                    assert call.name == "bash"
                    arguments = (
                        call.input
                        if protocol == "anthropic"
                        else json.loads(call.arguments)
                    )
                    assert arguments == {"command": "echo ok"}
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert all(record["trial_id"] == "native-trial" for record in records)
    assert [record["literal"]["completion_token_ids"] for record in records] == [
        [10],
        [11],
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "responses"])
@pytest.mark.parametrize("tool_definitions", ["present", "empty", "omitted"])
async def test_native_http_stream_emits_sdk_events_and_seals_literal_record(
    tmp_path, protocol, tool_definitions
):
    closed = asyncio.Event()

    class NativeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index, value in enumerate(["Hel", "lo"]):
                chunk = {
                    "id": "chatcmpl-native",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "bfcl-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": None,
                            "delta": {
                                "role": "assistant",
                                "content": value,
                                "provider_specific_fields": {"token_ids": [10 + index]},
                            },
                            "logprobs": {
                                "content": [{"token": value, "logprob": -0.25}]
                            },
                        }
                    ],
                }
                if index == 0:
                    chunk["prompt_token_ids"] = [1, 2]
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b'data: {"id":"chatcmpl-native","object":"chat.completion.chunk","created":1,"model":"bfcl-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}\n\n'
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed.set()

    async def backend(request):
        translated = json.loads(request.content)
        if protocol == "responses":
            assert translated["messages"][0] == {
                "role": "system",
                "content": "Base instructions\n\nDeveloper policy",
            }
            assert [message["role"] for message in translated["messages"]] == [
                "system",
                "user",
            ]
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"count": 2, "max_model_len": 8})
        body = json.loads(request.content)
        if body.get("tools") == []:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "`tools` must not be an empty array.",
                        "type": "invalid_request_error",
                        "param": "tools",
                        "code": None,
                    }
                },
            )
        if not body.get("tools") and body.get("tool_choice") is not None:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "When using `tool_choice`, `tools` must be set.",
                        "type": "invalid_request_error",
                        "param": "tool_choice",
                        "code": None,
                    }
                },
            )
        if tool_definitions == "present":
            assert body["tools"][0]["function"]["name"] == "bash"
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=NativeStream()
        )

    path = tmp_path / "literal.jsonl"
    proxy = RecordProxy("http://vllm", path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(backend)) as upstream:
        proxy._client = upstream
        http_module = httpx2 if protocol == "anthropic" else httpx
        transport = http_module.ASGITransport(
            app=proxy.app(NativeAPILimits("bfcl-test", 8, 8192))
        )
        client_type = AsyncAnthropic if protocol == "anthropic" else AsyncOpenAI
        base_url = "http://literal" if protocol == "anthropic" else "http://literal/v1"
        async with client_type(
            api_key="EMPTY",
            base_url=base_url,
            max_retries=0,
            default_headers={CORRELATION_ID_HEADER: "native-stream"},
            http_client=http_module.AsyncClient(transport=transport),
        ) as client:
            body = {**native_request(protocol, followup=False), "stream": True}
            if protocol == "responses":
                body["instructions"] = "Base instructions"
                body["input"].insert(
                    0,
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "Developer policy"}],
                    },
                )
            if tool_definitions == "empty":
                body["tools"] = []
            elif tool_definitions == "omitted":
                body.pop("tools")
            if tool_definitions != "present":
                body["tool_choice"] = (
                    {"type": "auto"} if protocol == "anthropic" else "auto"
                )
            stream = (
                await client.messages.create(**body)
                if protocol == "anthropic"
                else await client.responses.create(**body)
            )
            outputs = []
            async for event in stream:
                if (
                    protocol == "anthropic"
                    and event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    outputs.append(event.delta.text)
                if (
                    protocol == "responses"
                    and event.type == "response.output_text.delta"
                ):
                    outputs.append(event.delta)
    assert "".join(outputs) == "Hello"
    assert closed.is_set()
    record = json.loads(path.read_text())
    assert record["trial_id"] == "native-stream"
    assert record["literal"]["prompt_token_ids"] == [1, 2]
    assert record["literal"]["completion_token_ids"] == [10, 11]
    assert record["literal"]["logprobs"] == [-0.25, -0.25]
