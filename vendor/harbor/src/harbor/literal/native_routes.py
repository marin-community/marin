"""Messages and Responses endpoints backed by the literal completion recorder."""

from contextlib import AsyncExitStack
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI

from harbor.literal.native_api import (
    CORRELATION_ID_HEADER,
    CompletionRecorder,
    NativeAPILimits,
    RecordingCompletionTransport,
    anthropic_chat_request,
    responses_chat_request,
)
from harbor.literal.native_output import native_events, native_output
from harbor.llms.vllm_tokenization import (
    ContextBudgetExceeded,
    completion_budget_from_count,
)

LITERAL_API_BASE = "http://literal/v1"


async def native_prompt_count(
    recorder: CompletionRecorder,
    chat: dict[str, Any],
    request: Request,
    limits: NativeAPILimits,
    task_agent: str,
) -> tuple[int, int]:
    """Return prompt token count and the smaller of served and configured context limits."""
    if effort := chat.get("reasoning_effort"):
        extra_body = dict(chat.get("extra_body", {}))
        template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
        template_kwargs.setdefault("enable_thinking", effort != "none")
        extra_body["chat_template_kwargs"] = template_kwargs
        chat["extra_body"] = extra_body
    if chat["model"] != limits.served_model:
        raise HTTPException(400, "Native agent requested an unserved model")
    status, result, _ = await recorder.forward(
        "/tokenize",
        {
            "model": chat["model"],
            "messages": chat["messages"],
            "tools": chat.get("tools", []),
            "add_generation_prompt": True,
            "chat_template_kwargs": chat.get("extra_body", {}).get(
                "chat_template_kwargs", {}
            ),
        },
        trial_id=request.headers.get(CORRELATION_ID_HEADER),
        task_agent=task_agent,
    )
    if status != 200:
        raise HTTPException(status, result)
    if result is None:
        raise ValueError("Native tokenization returned a non-JSON response")
    count, window = result["count"], result["max_model_len"]
    if type(count) is not int or count < 0 or type(window) is not int or window < 1:
        raise ValueError("Native tokenization returned invalid count or context")
    return count, min(window, limits.context_window)


async def native_completion(
    recorder: CompletionRecorder,
    chat: dict[str, Any],
    request: Request,
    limits: NativeAPILimits,
    task_agent: str,
) -> tuple[Any, AsyncExitStack, int]:
    if not chat.get("tools") and chat.get("tool_choice") in (None, "auto", "none"):
        chat.pop("tool_choice", None)
    count, context = await native_prompt_count(
        recorder, chat, request, limits, task_agent
    )
    try:
        chat["max_tokens"] = completion_budget_from_count(
            prompt_tokens=count,
            context_window=context,
            max_output_tokens=limits.max_output_tokens,
            requested_max_tokens=chat.pop(
                "max_completion_tokens", chat.get("max_tokens")
            ),
        )
    except ContextBudgetExceeded as exc:
        raise HTTPException(400, str(exc)) from exc
    chat["model"] = limits.served_model
    resources = AsyncExitStack()
    try:
        client = await resources.enter_async_context(
            AsyncOpenAI(
                api_key="EMPTY",
                base_url=LITERAL_API_BASE,
                max_retries=0,
                http_client=httpx.AsyncClient(
                    transport=RecordingCompletionTransport(
                        recorder,
                        trial_id=request.headers.get(CORRELATION_ID_HEADER),
                        task_agent=task_agent,
                    )
                ),
            )
        )
        if chat.get("stream"):
            chat["stream_options"] = {"include_usage": True}
        result = await client.chat.completions.create(**chat)
    except BaseException:
        await resources.aclose()
        raise
    return result, resources, count


def register_native_routes(
    api: FastAPI, recorder: CompletionRecorder, limits: NativeAPILimits
) -> None:
    """Serve native tool protocols with the same recorder and context budget."""

    async def messages(request: Request):
        body = await request.json()
        try:
            chat = anthropic_chat_request(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        result, resources, prompt_tokens = await native_completion(
            recorder, chat, request, limits, "claude-code"
        )
        if body.get("stream"):
            return StreamingResponse(
                native_events(
                    result, resources, "anthropic", body["model"], prompt_tokens
                ),
                media_type="text/event-stream",
            )
        async with resources:
            return JSONResponse(
                native_output(result.model_dump(), "anthropic", body["model"])
            )

    async def count_tokens(request: Request):
        body = await request.json()
        try:
            chat = anthropic_chat_request(
                {**body, "max_tokens": limits.max_output_tokens}
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        count, _ = await native_prompt_count(
            recorder, chat, request, limits, "claude-code"
        )
        return JSONResponse({"input_tokens": count})

    async def responses(request: Request):
        body = await request.json()
        try:
            chat = responses_chat_request(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        result, resources, prompt_tokens = await native_completion(
            recorder, chat, request, limits, "codex"
        )
        if body.get("stream"):
            return StreamingResponse(
                native_events(
                    result, resources, "responses", body["model"], prompt_tokens
                ),
                media_type="text/event-stream",
            )
        async with resources:
            return JSONResponse(
                native_output(result.model_dump(), "responses", body["model"])
            )

    api.add_route("/v1/messages", messages, methods=["POST"])
    api.add_route("/v1/messages/count_tokens", count_tokens, methods=["POST"])
    api.add_route("/v1/responses", responses, methods=["POST"])
