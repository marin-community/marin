# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower recorded todo state operations to explicit Bash file operations."""

import json
import re
import shlex

import jsonschema

from experiments.tool_normalization.candidate import CALL_SCHEMA


def todo_commands(original: list[dict], path: str) -> dict[tuple[int, int], str] | None:
    """Return commands only when every todo result agrees with the recorded state."""
    parsed = {}
    for index, message in enumerate(original):
        if message["role"] != "assistant":
            continue
        try:
            parsed[index] = [
                json.loads(m[1]) for m in re.finditer(r"<tool_call>(.*?)</tool_call>", message["content"], re.DOTALL)
            ]
        except json.JSONDecodeError:
            return None
    if not any(isinstance(c, dict) and c.get("name") == "todoread" for calls in parsed.values() for c in calls):
        return {}
    if not any(m["role"] == "system" and "Use this tool to read your todo list" in m["content"] for m in original):
        return None
    commands = {}
    state = None
    for index, calls in parsed.items():
        if not any(isinstance(c, dict) and c.get("name") in {"todoread", "todowrite"} for c in calls):
            continue
        end = next((i for i in parsed if i > index), len(original))
        results = [m for m in original[index + 1 : end] if m["role"] in {"tool", "ipython"}]
        if len(calls) != len(results):
            return None
        for position, (call, result) in enumerate(zip(calls, results, strict=True)):
            if not isinstance(call, dict) or call.get("name") not in {"todoread", "todowrite"}:
                continue
            name = call["name"]
            args = call.get("arguments")
            match = re.fullmatch(
                r'\s*<tool_response name="([^"]+)">(.*)</tool_response>\s*', result["content"], re.DOTALL
            )
            if match is None or match[1] != name:
                return None
            try:
                wrapper = json.loads(match[2])
                if len(wrapper) != 1 or wrapper[0]["output"]["type"] != "text":
                    return None
                observed = json.loads(wrapper[0]["output"]["value"])
                if name == "todowrite":
                    jsonschema.validate(call, CALL_SCHEMA)
                    state = args["todos"]
                elif args != {} or state is None:
                    return None
            except (ValueError, KeyError, TypeError, jsonschema.ValidationError):
                return None
            if observed != state:
                return None
            if name == "todowrite":
                content = json.dumps(state, ensure_ascii=False, indent=2)
                code = (
                    f"from pathlib import Path; p = Path({path!r}); "
                    f"p.write_text({content!r}, encoding='utf-8'); print(p.read_text(encoding='utf-8'))"
                )
            else:
                code = f"from pathlib import Path; print(Path({path!r}).read_text(encoding='utf-8'))"
            commands[index, position] = "python3 -c " + shlex.quote(code)
    return commands
