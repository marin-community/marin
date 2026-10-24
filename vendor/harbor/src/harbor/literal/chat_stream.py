"""Accumulate OpenAI chat deltas without discarding provider fields."""

from copy import deepcopy
from typing import Any


def merge_chat_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
    for key, value in delta.items():
        if value is None:
            continue
        if key == "tool_calls":
            calls = message.setdefault(key, [])
            for part in value:
                index = part["index"]
                while len(calls) <= index:
                    calls.append({})
                call = calls[index]
                for field in ("id", "type"):
                    if part.get(field):
                        call[field] = part[field]
                function = call.setdefault("function", {})
                for field, text in part.get("function", {}).items():
                    if text is not None:
                        function[field] = function.get(field, "") + text
        elif isinstance(value, str) and key != "role":
            message[key] = (message.get(key) or "") + value
        else:
            message[key] = deepcopy(value)
