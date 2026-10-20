import asyncio
import weakref
from typing import Any

import httpx


_ASYNC_TOKENIZE_HTTP_CLIENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def shared_async_tokenize_client(timeout: float = 10.0) -> httpx.AsyncClient:
    """Return the pooled vLLM tokenization client for the current event loop."""
    loop = asyncio.get_running_loop()
    client = _ASYNC_TOKENIZE_HTTP_CLIENTS.get(loop)
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
        _ASYNC_TOKENIZE_HTTP_CLIENTS[loop] = client
    return client


def tokenize_url(api_base: str) -> str:
    """Return vLLM's server-root tokenization endpoint for an OpenAI API base."""
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/tokenize"


class ContextBudgetExceeded(ValueError):
    """The rendered prompt leaves no room for a completion."""


def completion_budget_from_count(
    *,
    prompt_tokens: int,
    context_window: int,
    max_output_tokens: int,
    requested_max_tokens: int | None = None,
) -> int:
    """Cap one completion using a rendered prompt count and total context."""
    remaining = context_window - prompt_tokens
    if remaining < 1:
        raise ContextBudgetExceeded(
            f"Prompt has {prompt_tokens} tokens; context window is {context_window}"
        )
    requested = (
        max_output_tokens if requested_max_tokens is None else requested_max_tokens
    )
    return min(requested, max_output_tokens, remaining)


def completion_budget(
    client: httpx.Client,
    *,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    request_kwargs: dict[str, Any],
    context_window: int,
    max_output_tokens: int,
) -> int:
    """Count a vLLM prompt and cap the reply within the total context window."""
    response = client.post(
        tokenize_url(api_base),
        headers=request_kwargs.get("extra_headers"),
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "add_generation_prompt": True,
            "chat_template_kwargs": request_kwargs.get("extra_body", {}).get(
                "chat_template_kwargs", {}
            ),
        },
    )
    response.raise_for_status()
    tokenization = response.json()
    prompt_tokens = tokenization["count"]
    server_context = tokenization["max_model_len"]
    if type(prompt_tokens) is not int or prompt_tokens < 0:
        raise ValueError("vLLM /tokenize returned an invalid token count")
    if type(server_context) is not int or server_context < 1:
        raise ValueError("vLLM /tokenize returned an invalid context window")
    return completion_budget_from_count(
        prompt_tokens=prompt_tokens,
        context_window=min(context_window, server_context),
        max_output_tokens=max_output_tokens,
        requested_max_tokens=request_kwargs.get("max_tokens"),
    )


async def tokenize_chat(
    *,
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    add_generation_prompt: bool,
    template_options: dict[str, Any],
) -> list[int]:
    """Render chat messages to IDs with the tokenizer used by the live vLLM."""
    payload = {
        "model": model,
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        **template_options,
    }
    response = await shared_async_tokenize_client().post(
        tokenize_url(api_base), json=payload
    )
    response.raise_for_status()
    token_ids = response.json().get("tokens")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise ValueError("vLLM /tokenize did not return an integer token list")
    return token_ids
