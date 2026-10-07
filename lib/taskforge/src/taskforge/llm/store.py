# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed cache of GLM calls.

Each call lives at ``<root>/items/<stage>/<hash>/`` where ``hash`` is the ``taskforge.canonical.digest`` of
the request: model, messages, policy fields that change the output, the structured-output tool,
and the sample index. Callers that draw several independent samples of one request pass a distinct
``sample`` for each, so each sample has its own entry; ``sample`` defaults to 0 for a request drawn
once.
``request.json`` is written before the call, ``response.json`` (every ``Completion``, including
raw events) after it, and ``result.json`` last; a cached value is reused only when ``result.json``
exists and records the same hash. ``stall_timeout`` is excluded from the key because it does not
change what the model returns.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter

from taskforge.canonical import digest, pretty_json, write_atomic
from taskforge.llm.client import Completion, GlmClient
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import OutputT, StructuredResult, StructuredTool, complete_structured

STORE_VERSION = 1

_COMPLETIONS = TypeAdapter(tuple[Completion, ...])


def canonical_request(
    model: str, messages: Sequence[Message], policy: LLMPolicy, tool: StructuredTool | None, sample: int
) -> dict[str, object]:
    """The fields that determine a call's output, in JSON-ready form."""
    return {
        "version": STORE_VERSION,
        "model": model,
        "messages": list(messages),
        "max_tokens": policy.max_tokens,
        "sampling": policy.sampling_fields(),
        "max_continuations": policy.max_continuations,
        "tool": None if tool is None else tool.definition(),
        "sample": sample,
    }


def write_json(path: Path, value: object) -> None:
    write_atomic(path, (pretty_json(value) + "\n").encode())


class CallStore:
    """Caches ``GlmClient`` calls on disk under ``root``, one directory per distinct request."""

    def __init__(self, root: Path, client: GlmClient):
        self.root = root
        self.client = client

    def item_dir(self, stage: str, key: str) -> Path:
        return self.root / "items" / stage / key

    def _cached(self, work: Path, key: str) -> dict | None:
        result_path = work / "result.json"
        if not result_path.exists():
            return None
        result = json.loads(result_path.read_text())
        if result["request_hash"] != key:
            raise ValueError(f"{result_path} records hash {result['request_hash']}, expected {key}")
        return result

    def _begin(self, stage: str, request: Mapping[str, object]) -> tuple[Path, str]:
        key = digest(request)
        work = self.item_dir(stage, key)
        work.mkdir(parents=True, exist_ok=True)
        return work, key

    def _finish(self, work: Path, key: str, completions: Sequence[Completion], value: object) -> None:
        write_json(work / "response.json", _COMPLETIONS.dump_python(tuple(completions), mode="json"))
        write_json(work / "result.json", {"request_hash": key, "value": value})

    async def complete(
        self, stage: str, messages: Sequence[Message], policy: LLMPolicy, *, sample: int = 0
    ) -> Completion:
        """Return the cached completion for this exact request and ``sample``, or make the call and record it."""
        request = canonical_request(self.client.endpoint.model, messages, policy, None, sample)
        work, key = self._begin(stage, request)
        if self._cached(work, key) is not None:
            return _COMPLETIONS.validate_json((work / "response.json").read_text())[0]
        write_json(work / "request.json", {"request_hash": key, "stage": stage, "request": request})
        completion = await self.client.complete(messages, policy)
        self._finish(work, key, (completion,), {"content": completion.content})
        return completion

    async def structured(
        self,
        stage: str,
        messages: Sequence[Message],
        policy: LLMPolicy,
        tool: StructuredTool[OutputT],
        *,
        sample: int = 0,
    ) -> StructuredResult[OutputT]:
        """Return the cached validated value for this exact request and ``sample``, or make the call and record it."""
        request = canonical_request(self.client.endpoint.model, messages, policy, tool, sample)
        work, key = self._begin(stage, request)
        cached = self._cached(work, key)
        if cached is not None:
            completions = _COMPLETIONS.validate_json((work / "response.json").read_text())
            return StructuredResult(tool.output_type.model_validate(cached["value"]), completions)
        write_json(work / "request.json", {"request_hash": key, "stage": stage, "request": request})
        result = await complete_structured(self.client, messages, policy, tool)
        self._finish(work, key, result.completions, result.value.model_dump(mode="json"))
        return result
