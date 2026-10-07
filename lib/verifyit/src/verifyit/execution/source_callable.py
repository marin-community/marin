# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Invoke an image-installed source scorer and write its reward as JSON."""

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def _function(reference: str, source_path: str | None = None):
    module, name = reference.rsplit(":", 1)
    if source_path is None:
        return getattr(importlib.import_module(module), name)
    specification = importlib.util.spec_from_file_location(module, source_path)
    if specification is None or specification.loader is None:
        raise ValueError(f"Cannot load source scorer at {source_path}")
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return getattr(loaded, name)


def _terminal_message(path: Path, answer: str, state_path: Path | None, input_format: str) -> dict[str, Any]:
    if state_path is not None and state_path.is_file():
        state = json.loads(state_path.read_text())
        captured = state.get("assistant_message")
        if isinstance(captured, Mapping):
            return {**captured, "content": answer}
    if input_format == "text":
        return {"role": "assistant", "content": answer, "tool_calls": []}
    event = json.loads(path.read_text())
    if event["type"] == "message" and event["role"] == "assistant":
        return {"content": event["content"], "tool_calls": []}
    if event["type"] == "assistant_tool_calls":
        return {
            "content": event["content"] or "",
            "tool_calls": [
                {"function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}
                for call in event["calls"]
            ],
        }
    raise ValueError("Source scorer requires a terminal assistant event")


def _value(reference: str, *, contract: dict, answer: str, message: dict) -> Any:
    if reference == "answer":
        return answer
    if reference == "terminal_message":
        return message
    if reference == "contract" or reference.startswith("contract."):
        value = contract
        for key in reference.split(".")[1:]:
            value = value[key]
        return value
    raise ValueError(f"Unsupported source scorer input: {reference}")


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return value


def main(descriptor_path: Path, config_path: Path, answer_path: Path, result_path: Path, state_path: Path | None = None):
    descriptor = json.loads(descriptor_path.read_text())
    if "source_sha256" in descriptor:
        source_path = Path(descriptor["source_path"])
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != descriptor["source_sha256"]:
            raise ValueError("Pinned source scorer has changed")
    config = json.loads(config_path.read_text())
    contract = config.get("contract", config)
    raw_answer = answer_path.read_text()
    input_format = descriptor.get("input_format", "text")
    answer = raw_answer
    if input_format == "event":
        event = json.loads(raw_answer)
        if event["type"] == "message" and event["role"] == "assistant":
            answer = event["content"]
        elif event["type"] == "assistant_tool_calls":
            answer = event["content"] or ""
        else:
            raise ValueError("Source scorer requires a terminal assistant event")
    extractor = descriptor.get("answer_extractor")
    if extractor is not None:
        answer = _function(extractor)(answer)
    needs_message = "terminal_message" in (*descriptor["args"], *descriptor.get("kwargs", {}).values())
    message = _terminal_message(answer_path, answer, state_path, input_format) if needs_message else {}
    if message and input_format == "event":
        message["content"] = answer
    references = {"contract": contract, "answer": answer, "message": message}
    args = [_value(item, **references) for item in descriptor["args"]]
    kwargs = {key: _value(item, **references) for key, item in descriptor.get("kwargs", {}).items()}
    result = _function(descriptor["function"], descriptor.get("source_path"))(*args, **kwargs)
    if "reward_key" in descriptor:
        reward = result[descriptor["reward_key"]]
        detail = result.get(descriptor.get("detail_key", "detail"), {})
    else:
        reward, detail = result
    result_path.write_text(json.dumps({"reward": reward, "detail": _json_value(detail)}, allow_nan=False))


if __name__ == "__main__":
    main(*(Path(argument) for argument in sys.argv[1:]))
