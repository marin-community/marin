"""Native agent API translation through the literal completion recorder."""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol, cast

import httpx
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

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


def anthropic_chat_request(
    body: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Translate Messages requests while retaining native tool-name mappings."""
    body = dict(body)
    context_management = body.pop("context_management", None)
    if context_management is not None and context_management != {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }:
        raise ValueError("Native capture supports only retaining all thinking history")
    # Keeping all thinking requires no server-side history edits.
    request, names = (
        AnthropicAdapter().translate_completion_input_params_with_tool_mapping(body)
    )
    if request is None:
        raise ValueError("Anthropic Messages request could not be translated")
    thinking = body.get("thinking")
    if thinking is not None and thinking["type"] == "disabled":
        request["reasoning_effort"] = "none"
    return dict(cast(dict[str, Any], request)), names


def responses_chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate complete Responses histories into chat-completion requests."""
    if body.get("previous_response_id"):
        raise ValueError(
            "Literal agent capture requires the complete Responses input history"
        )
    chat = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=body["model"],
        input=body["input"],
        responses_api_request=cast(ResponsesAPIOptionalRequestParams, body),
        custom_llm_provider="openai",
        stream=body.get("stream", False),
    )
    messages = chat["messages"]
    prefix_end = next(
        (
            index
            for index, message in enumerate(messages)
            if message["role"] not in ("system", "developer")
        ),
        len(messages),
    )
    if prefix_end > 1 or (prefix_end == 1 and messages[0]["role"] == "developer"):
        instructions = []
        for message in messages[:prefix_end]:
            content = message["content"]
            if isinstance(content, str):
                instructions.append(content)
                continue
            if not isinstance(content, list) or any(
                part["type"] != "text" or not isinstance(part["text"], str)
                for part in content
            ):
                raise ValueError("Initial Responses instructions require text content")
            instructions.append("".join(part["text"] for part in content))
        # LiteLLM maps developer turns to system; system-only templates need one prefix.
        chat["messages"] = [
            {"role": "system", "content": "\n\n".join(instructions)},
            *messages[prefix_end:],
        ]
    if chat.get("tools") == []:
        chat.pop("tools")
    return chat
