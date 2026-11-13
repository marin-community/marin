"""Persist installed-agent requests and exact model completions."""

import json
from pathlib import Path

from rolloutengine.harness.external import AgentCompletion
from rolloutengine.models import ModelTurn, Prompt

from skyrl_train.json_serialization import to_jsonable


def recorded_completion(complete: AgentCompletion, trace_path: Path) -> AgentCompletion:
    """Wrap a completion with paired JSONL request and response records."""
    request_index = 0

    async def recorded_complete(request: Prompt) -> ModelTurn:
        nonlocal request_index
        index = request_index
        request_index += 1
        with trace_path.open("a") as trace:
            trace.write(json.dumps({"schema_version": 1, "index": index, "request": to_jsonable(request)}) + "\n")
        turn = await complete(request)
        with trace_path.open("a") as trace:
            trace.write(json.dumps({"schema_version": 1, "index": index, "turn": to_jsonable(turn)}) + "\n")
        return turn

    return recorded_complete
