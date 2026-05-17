# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Unified multi-backend LM caller for the diversity-gen pipeline.

Routes by model prefix:
  - `gpt-*`, `o*`           → OpenAI (env `OPENAI_API_KEY`)
  - `gemini-*`              → Google GenAI (env `GEMINI_API_KEY`)
  - `claude-*`              → Anthropic (env `ANTHROPIC_API_KEY`)
  - `grok-*`                → xAI via OpenAI-compatible API (env `XAI_API_KEY`)

All callers return the response *text content* as a string. Strict JSON
schemas are honored by every backend when `response_schema` is provided:
  - OpenAI: `response_format={"type":"json_schema","json_schema":<schema>}`
  - Gemini: `response_json_schema=<schema.schema>` on GenerateContentConfig
  - Anthropic: `output_config={"format":{"type":"json_schema","schema":<schema.schema>}}`
  - xAI: same as OpenAI

Project memory hard rules respected:
  - gpt-5.x → `reasoning_effort="none"` always
  - gemini-3.x Pro → `thinking_level="low"`, `temperature=0`
  - Claude → `thinking={"type":"disabled"}`

The helper wraps `RawAPILogger.call(...)` for raw-response persistence so
existing logging conventions are preserved.
"""

from __future__ import annotations

import os
import threading
from typing import Any

# Per-backend SDK clients (lazy-init, cached)
_clients: dict[str, Any] = {}
_clients_lock = threading.Lock()


def _model_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-") or m.startswith("o") and m[1:2].isdigit():
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("grok-"):
        return "xai"
    raise ValueError(f"unknown model prefix: {model!r}")


def _get_openai_client():
    from openai import OpenAI
    with _clients_lock:
        if "openai" not in _clients:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise SystemExit("OPENAI_API_KEY not set")
            _clients["openai"] = OpenAI(api_key=key)
    return _clients["openai"]


def _get_xai_client():
    from openai import OpenAI
    with _clients_lock:
        if "xai" not in _clients:
            key = os.environ.get("XAI_API_KEY")
            if not key:
                raise SystemExit("XAI_API_KEY not set — `source .env2`")
            _clients["xai"] = OpenAI(api_key=key, base_url="https://api.x.ai/v1")
    return _clients["xai"]


def _get_gemini_client():
    from google import genai  # type: ignore
    with _clients_lock:
        if "gemini" not in _clients:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise SystemExit("GEMINI_API_KEY not set — `source .env2`")
            _clients["gemini"] = genai.Client(api_key=key)
    return _clients["gemini"]


def _get_anthropic_client():
    import anthropic
    with _clients_lock:
        if "anthropic" not in _clients:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise SystemExit("ANTHROPIC_API_KEY not set — `source .env2`")
            _clients["anthropic"] = anthropic.Anthropic(api_key=key)
    return _clients["anthropic"]


# Conservative permissive safety settings for Gemini. Adversarial-content
# scenarios (slurs, hate, etc.) need this to not get auto-blocked at the API
# level; the model-spec evaluation pipeline is exactly the kind of work this
# is intended for.
def _gemini_safety_block_none() -> list[Any]:
    from google.genai import types  # type: ignore
    return [
        types.SafetySetting(category=cat, threshold="BLOCK_NONE")
        for cat in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        )
    ]


def call_lm(
    *,
    model: str,
    system: str,
    user: str,
    max_output_tokens: int = 4000,
    temperature: float | None = None,
    response_schema: dict | None = None,
) -> str:
    """Single LM call. Returns response text.

    Args:
      model: provider-prefixed model name (e.g., "gpt-5.1", "gemini-3.1-pro",
        "claude-sonnet-4-6", "grok-4").
      system: system prompt.
      user: user message content.
      max_output_tokens: cap on model output tokens.
      temperature: sampling temperature; ignored on backends that require fixed
        values (gemini-3.x Pro pins temp=0 per project rule).
      response_schema: optional strict JSON schema in the OpenAI-style
        `{"name", "strict": true, "schema": {...}}` format. Applied via the
        appropriate backend-specific surface.

    Raises on API errors; caller is responsible for retries.
    """
    provider = _model_provider(model)

    if provider in ("openai", "xai"):
        return _call_openai_compat(
            provider=provider, model=model, system=system, user=user,
            max_output_tokens=max_output_tokens, temperature=temperature,
            response_schema=response_schema,
        )
    if provider == "gemini":
        return _call_gemini(
            model=model, system=system, user=user,
            max_output_tokens=max_output_tokens,
            response_schema=response_schema,
        )
    if provider == "anthropic":
        return _call_anthropic(
            model=model, system=system, user=user,
            max_output_tokens=max_output_tokens, temperature=temperature,
            response_schema=response_schema,
        )
    raise AssertionError(f"unreachable provider: {provider}")


def _call_openai_compat(
    *, provider: str, model: str, system: str, user: str,
    max_output_tokens: int, temperature: float | None,
    response_schema: dict | None,
) -> str:
    client = _get_xai_client() if provider == "xai" else _get_openai_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_output_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    # gpt-5.x: hard rule reasoning_effort="none" (project memory).
    if provider == "openai" and (model.lower().startswith("gpt-5") or model.lower().startswith("o")):
        kwargs["reasoning_effort"] = "none"

    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": response_schema}
    else:
        kwargs["response_format"] = {"type": "json_object"}

    # xAI rejects `max_completion_tokens` in some versions — use `max_tokens` instead
    if provider == "xai":
        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

    resp = client.chat.completions.create(**kwargs)
    if not resp.choices:
        return ""
    return (resp.choices[0].message.content or "").strip()


def _call_gemini(
    *, model: str, system: str, user: str,
    max_output_tokens: int, response_schema: dict | None,
) -> str:
    from google.genai import types  # type: ignore

    gem = _get_gemini_client()

    # Strict json_schema mode (Appendix A.5) — Gemini takes the raw schema dict
    # WITHOUT the OpenAI {name, strict, schema} wrapper.
    response_mime_type = "application/json"
    response_json_schema = response_schema["schema"] if response_schema else None

    cfg_kwargs: dict[str, Any] = dict(
        system_instruction=system,
        max_output_tokens=max_output_tokens,
        temperature=0,  # hard rule per project memory for 3.x Pro
        thinking_config=types.ThinkingConfig(thinking_level="low"),  # hard rule
        response_mime_type=response_mime_type,
        safety_settings=_gemini_safety_block_none(),
    )
    if response_json_schema is not None:
        cfg_kwargs["response_json_schema"] = response_json_schema

    config = types.GenerateContentConfig(**cfg_kwargs)
    resp = gem.models.generate_content(model=model, contents=user, config=config)
    return (resp.text or "").strip() if hasattr(resp, "text") else ""


def _call_anthropic(
    *, model: str, system: str, user: str,
    max_output_tokens: int, temperature: float | None,
    response_schema: dict | None,
) -> str:
    cl = _get_anthropic_client()

    kwargs: dict[str, Any] = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_output_tokens,
        # Hard rule: disable thinking by default
        "thinking": {"type": "disabled"},
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Anthropic structured-outputs surface lives at output_config.format
    # (per Appendix A.5). Strip the OpenAI {name, strict, schema} wrapper.
    if response_schema is not None:
        kwargs["output_config"] = {
            "format": {
                "type": "json_schema",
                "schema": response_schema["schema"],
            }
        }

    resp = cl.messages.create(**kwargs)
    # Extract text from content blocks
    if not resp.content:
        return ""
    parts: list[str] = []
    for block in resp.content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()
