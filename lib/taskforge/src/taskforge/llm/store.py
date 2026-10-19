# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed cache of model calls.

Each call lives at ``<root>/items/<stage>/<hash>/`` where ``hash`` is the ``taskforge.content_hash.digest`` of
the request: model, messages, the whole ``LLMPolicy``, the request fields (a decoding constraint)
or the structured output type's ``response_format``, and the sample index. Callers that draw
several independent samples of one request pass a distinct ``sample`` for each, so each sample has
its own entry; ``sample`` defaults to 0 for a request drawn once.
``request.json`` is written before the call, ``response.json`` (every ``Completion``, including
raw events) after it, and ``result.json`` last; a cached value is reused only when ``result.json``
exists and records the same hash.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, TypeAdapter
from rigging.filesystem.atomic import atomic_rename

from taskforge.content_hash import digest, pretty_json
from taskforge.llm.client import Completion, ModelClient, StructuredResult
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import response_format

STORE_VERSION = 4
"""Bumped when the request a key covers changes meaning.

4: structured requests carry the field instruction the library appends.
"""

_COMPLETIONS = TypeAdapter(tuple[Completion, ...])


def canonical_request(
    model: str, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object], sample: int
) -> dict[str, object]:
    """The fields that determine a call's output, in JSON-ready form."""
    return {
        "version": STORE_VERSION,
        "model": model,
        "messages": list(messages),
        "policy": policy,
        "request_fields": dict(request_fields),
        "sample": sample,
    }


def write_json(path: Path, value: object) -> None:
    with atomic_rename(str(path)) as temp:
        Path(temp).write_text(pretty_json(value) + "\n")


class CallStore:
    """Caches ``ModelClient`` calls on disk under ``root``, one directory per distinct request."""

    def __init__(self, root: Path, client: ModelClient):
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
        self,
        stage: str,
        messages: Sequence[Message],
        policy: LLMPolicy,
        *,
        request_fields: Mapping[str, object] = {},
        sample: int = 0,
    ) -> Completion:
        """Return the cached completion for this exact request and ``sample``, or make the call and record it."""
        request = canonical_request(self.client.model, messages, policy, request_fields, sample)
        work, key = self._begin(stage, request)
        if self._cached(work, key) is not None:
            return _COMPLETIONS.validate_json((work / "response.json").read_text())[0]
        write_json(work / "request.json", {"request_hash": key, "stage": stage, "request": request})
        completion = await self.client.complete(messages, policy, request_fields)
        self._finish(work, key, (completion,), {"content": completion.content})
        return completion

    async def structured[T: BaseModel](
        self,
        stage: str,
        messages: Sequence[Message],
        output_type: type[T],
        policy: LLMPolicy,
        *,
        sample: int = 0,
    ) -> StructuredResult[T]:
        """Return the cached validated value for this exact request and ``sample``, or make the call and record it."""
        fields = {"response_format": response_format(output_type)}
        request = canonical_request(self.client.model, messages, policy, fields, sample)
        work, key = self._begin(stage, request)
        cached = self._cached(work, key)
        if cached is not None:
            completions = _COMPLETIONS.validate_json((work / "response.json").read_text())
            return StructuredResult(output_type.model_validate(cached["value"]), completions)
        write_json(work / "request.json", {"request_hash": key, "stage": stage, "request": request})
        result = await self.client.complete_structured(messages, output_type, policy)
        self._finish(work, key, result.completions, result.value.model_dump(mode="json"))
        return result
