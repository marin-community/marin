# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The policy under training: a vLLM server with LoRA adapters that load at runtime, and its exact-token model.

``VllmPolicy`` is a ``rolloutengine`` model. It renders each request with the policy's own chat template, sends the
prompt token IDs to the server's completions endpoint, and returns the sampled token IDs and their log
probabilities, so that a trainer can learn from the rollout. A continuation keeps the served tokens unchanged and
appends only the template's boundary and the new messages. ``model`` names the adapter that serves the next request:
the trainer loads each new adapter and sets it, and requests in flight keep the adapter that they started with.
"""

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from rolloutengine.models import GenerationLimitReached, ModelRequest, ModelTurn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "base"
"""The served name of the base weights; an adapter is served under its own name."""
TOKEN_ID_PREFIX = "token_id:"
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
STARTUP_TIMEOUT = 1200.0
REQUEST_TIMEOUT = 1800.0
BOUNDARY_PROBE = "boundary"


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    max_tokens: int
    """The most tokens of one model turn."""


@dataclass
class VllmServer:
    """A running ``vllm serve`` process for one base model, with runtime LoRA loading."""

    base_url: str
    process: subprocess.Popen[bytes]

    @classmethod
    @contextmanager
    def start(
        cls,
        model: str,
        *,
        executable: str,
        max_model_len: int,
        max_lora_rank: int,
        gpu_memory_utilization: float,
    ) -> Iterator["VllmServer"]:
        """Start the server on a free port, wait until it is healthy, and stop it.

        The server uses the first GPU that ``CUDA_VISIBLE_DEVICES`` names. Pods on CoreWeave share the host's network,
        so a fixed port can belong to another task on the same node.
        """
        port = _free_port()
        command = [
            executable,
            "serve",
            model,
            "--served-model-name",
            BASE_MODEL_NAME,
            "--port",
            str(port),
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--max-loras",
            "4",
            "--max-cpu-loras",
            "8",
            "--disable-log-requests",
            "--disable-uvicorn-access-log",
        ]
        environment = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
        # The server writes its log to the task's log.
        process = subprocess.Popen(command, env=environment)
        server = cls(f"http://127.0.0.1:{port}", process)
        try:
            server._wait_healthy()
            yield server
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {self.process.returncode}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError("vllm serve did not become healthy")

    async def load_adapter(self, name: str, path: str) -> None:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=600) as client:
            response = await client.post("/v1/load_lora_adapter", json={"lora_name": name, "lora_path": path})
            response.raise_for_status()

    async def unload_adapter(self, name: str) -> None:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=600) as client:
            response = await client.post("/v1/unload_lora_adapter", json={"lora_name": name})
            response.raise_for_status()


@dataclass
class VllmPolicy:
    """A ``rolloutengine`` model over a ``VllmServer``, under ``TraceMode.EXACT_TOKENS``."""

    base_url: str
    tokenizer: PreTrainedTokenizerBase
    sampling: Sampling
    max_model_len: int
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    model: str = BASE_MODEL_NAME
    """The served name of the weights for the next request."""
    max_concurrent_requests: int = 256
    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _slots: asyncio.Semaphore | None = field(default=None, init=False, repr=False)

    @classmethod
    def for_model(cls, base_url: str, model: str, sampling: Sampling, max_model_len: int, **kwargs: Any) -> "VllmPolicy":
        return cls(base_url, AutoTokenizer.from_pretrained(model), sampling, max_model_len, **kwargs)

    async def __call__(self, request: ModelRequest) -> ModelTurn:
        if self._client is None:
            limits = httpx.Limits(max_connections=self.max_concurrent_requests, max_keepalive_connections=64)
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=REQUEST_TIMEOUT, limits=limits)
            self._slots = asyncio.Semaphore(self.max_concurrent_requests)
        assert self._slots is not None
        prompt = self.prompt_token_ids(request)
        remaining = self.max_model_len - len(prompt)
        if remaining <= 0:
            raise GenerationLimitReached(tuple(prompt))
        options = request.options
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": min(int(options.get("max_tokens", self.sampling.max_tokens)), remaining),
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "logprobs": 1,
            "return_tokens_as_token_ids": True,
            "skip_special_tokens": False,
        }
        async with self._slots:
            response = await self._client.post("/v1/completions", json=body)
        response.raise_for_status()
        choice = response.json()["choices"][0]
        tokens = tuple(int(token.removeprefix(TOKEN_ID_PREFIX)) for token in choice["logprobs"]["tokens"])
        logprobs = tuple(float(value) for value in choice["logprobs"]["token_logprobs"])
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        # A continuation of an assistant prefix is raw text, even when it closes a tool call of the prefix.
        tool_names = frozenset(tool["function"]["name"] for tool in request.options.get("tools") or ())
        message = (
            {"role": "assistant", "content": text} if _continues_final(request) else assistant_message(text, tool_names)
        )
        return ModelTurn(
            message,
            prompt_token_ids=tuple(prompt),
            response_token_ids=tokens,
            logprobs=logprobs,
            stop_reason=choice["finish_reason"],
            text=text,
            metadata={"model": self.model},
        )

    def prompt_token_ids(self, request: ModelRequest) -> list[int]:
        if request.prompt_token_ids is not None:
            return list(request.prompt_token_ids)
        messages = [_template_message(message) for message in request.messages]
        tools = request.options.get("tools")
        if request.continuation is not None:
            return self._continuation(
                messages, tools, request.continuation.served_token_ids, request.continuation.assistant_message_index
            )
        continue_final = _continues_final(request)
        return self._render(messages, tools, generation_prompt=not continue_final, continue_final=continue_final)

    def _render(
        self, messages: Sequence[dict[str, Any]], tools: Any, *, generation_prompt: bool, continue_final: bool = False
    ) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            list(messages),
            tools=tools,
            add_generation_prompt=generation_prompt,
            continue_final_message=continue_final,
            tokenize=True,
            **self.chat_template_kwargs,
        )
        # tokenize=True without return_dict renders the token IDs of one conversation.
        return [int(token) for token in rendered]

    def _continuation(
        self, messages: list[dict[str, Any]], tools: Any, served: Sequence[int], assistant_index: int
    ) -> list[int]:
        """The served tokens, the rest of the template's boundary after the assistant turn, and the new messages.

        The sampled tokens never tokenize again. The boundary is what the template writes after an assistant
        message, for example ``<|im_end|>\\n``; the served tokens can already end with a part of it.
        """
        history, prefix = messages[:assistant_index], messages[: assistant_index + 1]
        full = self._render(messages, tools, generation_prompt=True)
        prefix_ids = self._render(prefix, tools, generation_prompt=False)
        probe = [*history, {"role": "assistant", "content": BOUNDARY_PROBE}]
        probe_ids = self._render(probe, tools, generation_prompt=False)
        open_ids = self._render(probe, tools, generation_prompt=False, continue_final=True)
        if probe_ids[: len(open_ids)] != open_ids:
            raise ValueError("The chat template does not end an assistant message with a fixed boundary")
        boundary = probe_ids[len(open_ids) :]
        if full[: len(prefix_ids)] == prefix_ids:
            suffix = full[len(prefix_ids) :]
        else:
            # Templates that rewrite earlier assistant turns, for example by removing their reasoning: render the new
            # messages after the first prompt instead, and check that the full rendering ends with them.
            first = next(index for index, message in enumerate(messages) if message["role"] == "assistant")
            base = self._render(messages[:first], tools, generation_prompt=False)
            tail = self._render([*messages[:first], *messages[assistant_index + 1 :]], tools, generation_prompt=True)
            if tail[: len(base)] != base:
                raise ValueError("The chat template does not render the new messages as a suffix")
            suffix = tail[len(base) :]
            if full[-len(suffix) :] != suffix:
                raise ValueError("The chat template renders the new messages differently after the history")
        served = list(served)
        overlap = max(
            (count for count in range(1, min(len(served), len(boundary)) + 1) if served[-count:] == boundary[:count]),
            default=0,
        )
        return [*served, *boundary[overlap:], *suffix]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _continues_final(request: ModelRequest) -> bool:
    return bool(request.options.get("extra_body", {}).get("continue_final_message"))


def _template_message(message: dict[str, Any]) -> dict[str, Any]:
    """A chat message as chat templates read it: tool call arguments as objects, not JSON strings."""
    calls = message.get("tool_calls")
    if not calls:
        return message
    converted = []
    for call in calls:
        function = dict(call["function"])
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        converted.append({**call, "function": function})
    return {**message, "tool_calls": converted}


def assistant_message(text: str, tool_names: frozenset[str] = frozenset()) -> dict[str, Any]:
    """The assistant message of a reply in the Hermes tool-call format that Qwen chat templates use.

    A reply whose ``<tool_call>`` blocks hold JSON objects with a name calls those tools. Small policies often drop the
    opening tag, so a reply that is one JSON object with ``arguments`` and the name of one of ``tool_names``, the
    advertised tools, with or without the closing tag, is one call too; a JSON answer is not a call. Anything else
    stays text, so that the harness ends the attempt and the grader sees what the policy wrote.
    """
    blocks = TOOL_CALL.findall(text)
    prefix = text[: text.index("<tool_call>")] if blocks else ""
    if not blocks:
        bare = text.strip().removesuffix("</tool_call>").strip()
        try:
            candidate = json.loads(bare)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and candidate.get("name") in tool_names and "arguments" in candidate:
            blocks = [bare]
    calls = [_tool_call(block) for block in blocks]
    if not calls or any(call is None for call in calls):
        return {"role": "assistant", "content": text}
    return {"role": "assistant", "content": prefix.strip(), "tool_calls": calls}


def _tool_call(block: str) -> dict[str, Any] | None:
    try:
        call = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments", {}))},
    }
