# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One native-budgeted completion, with no model-accessible tools or execution."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import APIStatusError, AsyncOpenAI
from taskcompendium.chat import chat_reply_status
from taskcompendium.request_budget import PromptBudget, TokenizationIdentity
from upath import UPath

from rolloutengine.models import ModelRequest, ModelResponseError, ModelTurn, TokenUsage

PROTECTED_FIELDS = frozenset(
    {
        "model",
        "messages",
        "tools",
        "tool_choice",
        "plugins",
        "max_tokens",
        "max_completion_tokens",
        "functions",
        "function_call",
        "web_search_options",
        "stream",
        "n",
    }
)


class PromptTokenizer(Protocol):
    def get_chat_template(self) -> str: ...

    def apply_chat_template(self, conversation: list[dict[str, Any]], **kwargs: Any) -> list[int]: ...


@dataclass(frozen=True)
class ToolFreeChatModel:
    """Validate supplied native budget evidence, retain raw traffic and return text only."""

    client: AsyncOpenAI
    model: str
    backend_identity_sha256: str
    tokenization: TokenizationIdentity
    tokenizer: PromptTokenizer
    budget: PromptBudget
    receipts: Path | UPath

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if request.continuation is not None or request.prompt_token_ids is not None:
            raise ValueError("Tool-free completion requires the unchanged source messages")
        options = dict(request.options)
        extra = options.pop("extra_body", {})
        if not isinstance(extra, Mapping) or PROTECTED_FIELDS.intersection(options.keys() | extra.keys()):
            raise ValueError("Sampling options cannot override the tool-free protocol or native budget")
        template = self.tokenizer.get_chat_template()
        if hashlib.sha256(template.encode()).hexdigest() != self.tokenization.chat_template_sha256:
            raise ValueError("Actual tokenizer chat template differs from the pinned budget identity")
        arguments = self.tokenization.template_arguments
        if arguments.keys() - {"enable_thinking", "reasoning_effort"}:
            raise ValueError("Template arguments may change only thinking controls, without truncating tokenization")
        if "chat_template_kwargs" in extra and extra["chat_template_kwargs"] != arguments:
            raise ValueError("Request template arguments differ from native token counting")
        tokens = self.tokenizer.apply_chat_template(
            list(request.messages),
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            **arguments,
        )
        capacity = self.budget.require_match(
            request.messages,
            model=self.model,
            endpoint=str(self.client.base_url),
            backend_identity_sha256=self.backend_identity_sha256,
            tokenization=self.tokenization,
            prompt_tokens=len(tokens),
        )
        body: dict[str, Any] = {
            **options,
            "model": self.model,
            "messages": list(request.messages),
            "max_tokens": capacity,
            "tools": [],
            "tool_choice": "none",
            "extra_body": {**extra, "chat_template_kwargs": arguments, "plugins": [{"id": "web", "enabled": False}]},
        }
        self.receipts.mkdir(parents=True, exist_ok=False)
        (self.receipts / "request.json").write_text(json.dumps(body, indent=2) + "\n")
        (self.receipts / "budget.json").write_text(self.budget.model_dump_json(indent=2) + "\n")
        try:
            response = await self.client.chat.completions.with_raw_response.create(stream=False, **body)
        except APIStatusError as error:
            (self.receipts / "wire_request.json").write_bytes(error.response.request.content)
            (self.receipts / "raw_response.json").write_bytes(error.response.content)
            (self.receipts / "response_status.json").write_text(json.dumps({"status": error.status_code}) + "\n")
            raise
        raw = response.content
        (self.receipts / "wire_request.json").write_bytes(response.http_request.content)
        (self.receipts / "raw_response.json").write_bytes(raw)
        (self.receipts / "response_status.json").write_text(json.dumps({"status": response.status_code}) + "\n")
        payload = json.loads(raw)
        choices = payload.get("choices", [])
        if len(choices) != 1 or not isinstance(payload.get("usage"), dict):
            raise ModelResponseError("Tool-free completion requires one choice and token usage", response=payload)
        usage = payload["usage"]
        if usage["prompt_tokens"] != len(tokens):
            raise ModelResponseError("Endpoint prompt usage differs from the exact native count", response=payload)
        if usage["completion_tokens"] > capacity:
            raise ModelResponseError("Endpoint completion exceeds the native output budget", response=payload)
        status = chat_reply_status(payload)
        content = choices[0]["message"].get("content") or ""
        if not isinstance(content, str):
            raise ModelResponseError("Tool-free completion requires text content", response=payload)
        (self.receipts / "answer_status.json").write_text(
            json.dumps(
                {
                    "origin": status.origin,
                    "finish_reason": status.finish_reason,
                    "truncated": status.truncated,
                    "provider_refusal": status.provider_refusal,
                    "tool_events": status.tool_events,
                },
                indent=2,
            )
            + "\n"
        )
        return ModelTurn(
            {"role": "assistant", "content": content},
            prompt_token_ids=tuple(tokens),
            response_token_ids=(),
            logprobs=None,
            stop_reason=choices[0]["finish_reason"],
            text=content,
            metadata={"native_budget": self.budget.model_dump(mode="json"), "traffic_receipts": str(self.receipts)},
            usage=TokenUsage(usage["prompt_tokens"], usage["completion_tokens"]),
            reply_status=status,
        )
