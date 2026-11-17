"""Native agent API translation through the literal completion recorder."""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

CHAT_ENDPOINT = "/v1/chat/completions"
CORRELATION_ID_HEADER = "x-ot-trial-id"


@dataclass(frozen=True)
class NativeAPILimits:
    served_model: str
    context_window: int
    max_output_tokens: int


class CompletionRecorder(Protocol):
    async def forward(
        self,
        path: str,
        body: dict[str, Any],
        method: str = "POST",
        *,
        trial_id: str | None = None,
        task_agent: str | None = None,
    ) -> tuple[int, dict[str, Any] | None, float]: ...

    async def open_upstream_stream(
        self,
        path: str,
        body: dict[str, Any],
        method: str = "POST",
        *,
        trial_id: str | None = None,
        task_agent: str | None = None,
    ) -> tuple[dict[str, Any], httpx.Response]: ...

    def relay_and_record(
        self,
        enriched: dict[str, Any],
        resp: httpx.Response,
        *,
        trial_id: str | None = None,
    ) -> AsyncGenerator[bytes, None]: ...


class RecordedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: AsyncGenerator[bytes, None]) -> None:
        self.chunks = chunks

    def __aiter__(self) -> AsyncGenerator[bytes, None]:
        return self.chunks

    async def aclose(self) -> None:
        await self.chunks.aclose()


class RecordingCompletionTransport(httpx.AsyncBaseTransport):
    """Keep translated SDK requests on the recorder's ordinary serving path."""

    def __init__(
        self,
        recorder: CompletionRecorder,
        *,
        trial_id: str | None,
        task_agent: str,
    ) -> None:
        self.recorder = recorder
        self.trial_id = trial_id
        self.task_agent = task_agent

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != CHAT_ENDPOINT:
            raise ValueError(
                f"Native agent translation requested an unsupported endpoint: {request.url.path}"
            )
        body = json.loads(request.content)
        if body.get("stream"):
            enriched, response = await self.recorder.open_upstream_stream(
                CHAT_ENDPOINT, body, trial_id=self.trial_id, task_agent=self.task_agent
            )
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=RecordedStream(
                    self.recorder.relay_and_record(
                        enriched, response, trial_id=self.trial_id
                    )
                ),
                request=request,
            )
        status, response, _ = await self.recorder.forward(
            CHAT_ENDPOINT, body, trial_id=self.trial_id, task_agent=self.task_agent
        )
        if response is None:
            raise ValueError(
                "Native agent completion upstream returned a non-JSON response"
            )
        return httpx.Response(status, json=response, request=request)


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(
        part.get("type") in ("text", "input_text", "output_text") for part in content
    ):
        return "".join(part["text"] for part in content)
    raise ValueError("This content requires text blocks")


def anthropic_chat_request(
    body: dict[str, Any],
) -> dict[str, Any]:
    """Translate Messages histories and function tools to chat completions."""
    context = body.get("context_management")
    if context is not None and context != {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }:
        raise ValueError("Native capture supports only retaining all thinking history")
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": _text_content(body["system"])})
    for message in body["messages"]:
        content = message["content"]
        if isinstance(content, str):
            messages.append({"role": message["role"], "content": content})
            continue
        translated: dict[str, Any] = {"role": message["role"], "content": []}
        for part in content:
            match part["type"]:
                case "text":
                    translated["content"].append({"type": "text", "text": part["text"]})
                case "image":
                    source = part["source"]
                    url = (
                        source["url"]
                        if source["type"] == "url"
                        else f"data:{source['media_type']};base64,{source['data']}"
                    )
                    translated["content"].append(
                        {"type": "image_url", "image_url": {"url": url}}
                    )
                case "thinking":
                    translated["reasoning_content"] = (
                        translated.get("reasoning_content", "") + part["thinking"]
                    )
                case "tool_use":
                    translated.setdefault("tool_calls", []).append(
                        {
                            "id": part["id"],
                            "type": "function",
                            "function": {
                                "name": part["name"],
                                "arguments": json.dumps(part["input"]),
                            },
                        }
                    )
                case "tool_result":
                    # Flush text before a tool result to preserve the source order.
                    if translated["content"]:
                        messages.append(translated)
                        translated = {"role": message["role"], "content": []}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": part["tool_use_id"],
                            "content": _text_content(part["content"]),
                        }
                    )
                case _:
                    raise ValueError(f"Unsupported Messages content: {part['type']}")
        if (
            translated["content"]
            or translated.get("tool_calls")
            or translated.get("reasoning_content")
        ):
            if not translated["content"]:
                translated["content"] = None
            messages.append(translated)
    chat = {
        key: body[key]
        for key in ("model", "max_tokens", "temperature", "top_p", "stream")
        if key in body
    }
    chat["messages"] = messages
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]
    if body.get("tools"):
        chat["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["input_schema"],
                },
            }
            for tool in body["tools"]
        ]
    choice = body.get("tool_choice")
    if choice:
        chat["tool_choice"] = (
            {"type": "function", "function": {"name": choice["name"]}}
            if choice["type"] == "tool"
            else {"any": "required"}.get(choice["type"], choice["type"])
        )
    effort = body.get("output_config", {}).get("effort")
    if body.get("thinking", {}).get("type") == "disabled":
        effort = "none"
    if effort:
        chat["reasoning_effort"] = effort
    return chat


def responses_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate complete Responses histories into chat-completion requests."""
    if body.get("previous_response_id"):
        raise ValueError(
            "Literal agent capture requires the complete Responses input history"
        )
    messages = []
    if body.get("instructions"):
        messages.append({"role": "system", "content": body["instructions"]})
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [{"role": "user", "content": inputs}]
    for item in inputs:
        match item.get("type", "message"):
            case "message":
                content = item["content"]
                if isinstance(content, list):
                    content = [
                        (
                            {"type": "text", "text": part["text"]}
                            if part["type"] in ("input_text", "output_text")
                            else (
                                {
                                    "type": "image_url",
                                    "image_url": {"url": part["image_url"]},
                                }
                                if part["type"] == "input_image"
                                else part
                            )
                        )
                        for part in content
                    ]
                messages.append(
                    {
                        "role": (
                            "system" if item["role"] == "developer" else item["role"]
                        ),
                        "content": content,
                    }
                )
            case "function_call":
                call = {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
                if messages and messages[-1]["role"] == "assistant":
                    messages[-1].setdefault("tool_calls", []).append(call)
                else:
                    messages.append(
                        {"role": "assistant", "content": None, "tool_calls": [call]}
                    )
            case "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": item["output"],
                    }
                )
            case "reasoning":
                # Encrypted provider state cannot be replayed to a local chat endpoint.
                if item.get("summary"):
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": _text_content(
                                [{**part, "type": "text"} for part in item["summary"]]
                            ),
                        }
                    )
            case _:
                raise ValueError(f"Unsupported Responses input: {item.get('type')}")
    prefix_end = next(
        (i for i, message in enumerate(messages) if message["role"] != "system"),
        len(messages),
    )
    if prefix_end:
        messages = [
            {
                "role": "system",
                "content": "\n\n".join(
                    _text_content(message["content"])
                    for message in messages[:prefix_end]
                ),
            },
            *messages[prefix_end:],
        ]
    chat = {
        key: body[key]
        for key in (
            "model",
            "temperature",
            "top_p",
            "stream",
            "tool_choice",
            "parallel_tool_calls",
        )
        if key in body
    }
    chat["messages"] = messages
    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    choice = chat.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") == "function":
        chat["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("text", {}).get("format"):
        format_config = body["text"]["format"]
        if format_config["type"] == "json_schema":
            chat["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    key: value for key, value in format_config.items() if key != "type"
                },
            }
        elif format_config["type"] == "json_object":
            chat["response_format"] = format_config
    if body.get("reasoning", {}).get("effort"):
        chat["reasoning_effort"] = body["reasoning"]["effort"]
    if body.get("tools"):
        if any(tool["type"] != "function" for tool in body["tools"]):
            raise ValueError("Native capture supports function tools")
        chat["tools"] = [
            {
                "type": "function",
                "function": {
                    key: value for key, value in tool.items() if key != "type"
                },
            }
            for tool in body["tools"]
        ]
    return chat
