#!/usr/bin/env python3
# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off Gemini caller using the official Google Gen AI SDK.

This script uses the `google-genai` package, not raw HTTP.

Env:
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`

Examples:

```bash
export GEMINI_API_KEY=...
uv run --with google-genai python scripts/gemini_oneoff.py --list-models
uv run --with google-genai python scripts/gemini_oneoff.py --model gemini-3.1-pro-preview "Explain attention in one paragraph."
uv run --with google-genai python scripts/gemini_oneoff.py --system "You are a concise coding assistant." "Write a Python HTTP server."
printf 'Summarize this text' | uv run --with google-genai python scripts/gemini_oneoff.py
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY or GOOGLE_API_KEY in the environment.")
    return key


def _load_sdk():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise SystemExit(
            "Missing the official Google Gen AI SDK.\n"
            "Install it with one of:\n"
            "  uv add google-genai\n"
            "  uv run --with google-genai python scripts/gemini_oneoff.py ...\n"
        ) from exc
    return genai, types


def _client(api_key: str):
    genai, _types = _load_sdk()
    return genai.Client(api_key=api_key, vertexai=False)


def _list_models(*, api_key: str) -> int:
    client = _client(api_key)
    try:
        for model in client.models.list():
            name = getattr(model, "name", "")
            methods = getattr(model, "supported_generation_methods", None)
            if methods is None and isinstance(model, dict):
                methods = model.get("supported_generation_methods")
                name = model.get("name", name)
            if methods and "generateContent" in methods:
                print(str(name).removeprefix("models/"))
    except Exception as exc:
        raise SystemExit(f"Failed to list Gemini models: {exc}") from exc
    return 0


def _prompt_text(args: argparse.Namespace) -> str:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("No prompt provided. Pass text as arguments or via stdin.")
    return prompt


def _extract_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None)
    if not candidates and isinstance(response, dict):
        candidates = response.get("candidates")
    if not candidates:
        return None

    first = candidates[0]
    content = getattr(first, "content", None)
    if content is None and isinstance(first, dict):
        content = first.get("content")
    if not content:
        return None

    parts = getattr(content, "parts", None)
    if parts is None and isinstance(content, dict):
        parts = content.get("parts")
    if not parts:
        return None

    out: list[str] = []
    for part in parts:
        piece = getattr(part, "text", None)
        if piece is None and isinstance(part, dict):
            piece = part.get("text")
        if piece:
            out.append(str(piece))
    return "".join(out).strip() or None


def _response_to_json(response: Any) -> str:
    if hasattr(response, "model_dump_json"):
        return response.model_dump_json(indent=2)
    if hasattr(response, "model_dump"):
        return json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=True)
    if isinstance(response, dict):
        return json.dumps(response, indent=2, ensure_ascii=True)
    return json.dumps({"response": repr(response)}, indent=2, ensure_ascii=True)


def _generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    system: str | None,
    max_output_tokens: int | None,
    no_thinking: bool = False,
) -> Any:
    _genai, types = _load_sdk()
    client = _client(api_key)

    config_kwargs: dict[str, Any] = {}
    if system:
        config_kwargs["system_instruction"] = system
    if max_output_tokens is not None:
        config_kwargs["max_output_tokens"] = max_output_tokens
    if no_thinking:
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    try:
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as exc:
        raise SystemExit(f"Gemini generate_content failed for model {model!r}: {exc}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-off Gemini caller using the official SDK.")
    parser.add_argument("prompt", nargs="*", help="Prompt text. If omitted, reads stdin.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--system", help="Optional system instruction.")
    parser.add_argument("--max-output-tokens", type=int, help="Optional max output token cap.")
    parser.add_argument("--list-models", action="store_true", help="List available generateContent models.")
    parser.add_argument("--json", action="store_true", help="Print raw SDK response as JSON.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable thinking (set thinking_budget=0).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    api_key = _api_key()

    if args.list_models:
        return _list_models(api_key=api_key)

    prompt = _prompt_text(args)
    response = _generate(
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        system=args.system,
        max_output_tokens=args.max_output_tokens,
        no_thinking=args.no_thinking,
    )

    if args.json:
        print(_response_to_json(response))
        return 0

    text = _extract_text(response)
    if text is not None:
        print(text)
        return 0

    print(_response_to_json(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
