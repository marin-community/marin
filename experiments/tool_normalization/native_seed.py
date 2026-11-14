# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map known source tools to the tested OpenCode interfaces before GLM review."""

import copy
import hashlib
import json
import re
import shlex
from collections.abc import Collection
from itertools import pairwise

import jsonschema

from experiments.tool_normalization.candidate import CALL_SCHEMA
from experiments.tool_normalization.kernel_setup import kernel_setup_calls
from experiments.tool_normalization.todo_state import todo_commands

KERNEL_CODE_ARGUMENTS = {
    "stateful_python_code_exec": "code",
    "python_interpreter": "script",
    "jupyter_kernel_run": "cell_content",
    "sandbox_exec": "snippet",
}

BINARY_EDITOR_SUFFIXES = {".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"}
NATIVE_READ_MAX_LINES = 2000
FINISH_DESCRIPTION = "Signals the completion of the current task or conversation."

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["convert", "hold", "environment_required"]},
        "rationale": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_message_index": {"type": "integer"},
                    "content": {"type": ["string", "null"]},
                    "reasoning_content": {"type": ["string", "null"]},
                    "calls": {"anyOf": [{"type": "null"}, {"type": "array", "items": CALL_SCHEMA}]},
                },
                "required": ["source_message_index", "content", "reasoning_content", "calls"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision", "rationale", "edits"],
    "additionalProperties": False,
}


def zero_based_read_offsets(original: list[dict]) -> bool:
    """Recognize the legacy read declaration rather than guessing from observed content."""
    decoder = json.JSONDecoder()
    for message in original:
        if message["role"] != "system":
            continue
        for block in re.findall(r"<tools>(.*?)</tools>", message["content"], re.DOTALL):
            remaining = block.strip()
            while remaining:
                declaration, end = decoder.raw_decode(remaining)
                remaining = remaining[end:].strip()
                function = declaration.get("function", {})
                if function.get("name") == "read":
                    offset = function.get("parameters", {}).get("properties", {}).get("offset", {})
                    return offset.get("description") == "The line number to start reading from (0-based)"
    return False


def native_seed(original: list[dict], available_packages: Collection[str]) -> dict | None:
    identity = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    todos = todo_commands(original, f"/tmp/marin-todos-{identity}.json")
    if todos is None:
        return None
    turns = []
    started = False
    kernel_started = False
    for index, message in enumerate(original):
        if message["role"] != "assistant":
            continue
        matches = list(re.finditer(r"<tool_call>(.*?)</tool_call>", message["content"], re.DOTALL))
        calls = []
        content = None
        if matches:
            if message["content"][matches[-1].end() :].strip():
                return None
            for previous, following in pairwise(matches):
                if message["content"][previous.end() : following.start()].strip():
                    return None
            content = message["content"][: matches[0].start()]
            for call_position, match in enumerate(matches):
                try:
                    action = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return None
                if not isinstance(action, dict):
                    return None
                name = action.get("name")
                arguments = action.get("arguments")
                if not isinstance(arguments, dict):
                    return None
                if (index, call_position) in todos:
                    calls.append({"name": "bash", "arguments": {"command": todos[index, call_position]}})
                elif name == "execute_bash" and set(arguments) == {"command"}:
                    if not isinstance(arguments["command"], str):
                        return None
                    calls.append({"name": "bash", "arguments": arguments.copy()})
                elif name in {"bash", "read", "write", "edit", "grep", "glob", "todowrite"}:
                    native_arguments = arguments.copy()
                    if name == "read" and "offset" in native_arguments and zero_based_read_offsets(original):
                        offset = native_arguments["offset"]
                        if type(offset) is not int or offset < 0:
                            return None
                        native_arguments["offset"] = offset + 1
                    if name in {"bash", "edit", "write"} and "description" in native_arguments:
                        if not isinstance(native_arguments.pop("description"), str):
                            return None
                    call = {"name": name, "arguments": native_arguments}
                    try:
                        jsonschema.validate(call, CALL_SCHEMA)
                    except jsonschema.ValidationError:
                        return None
                    calls.append(call)
                elif name == "str_replace_editor" and arguments.get("command") == "str_replace":
                    if set(arguments) != {"command", "path", "old_str", "new_str"} or not arguments["old_str"]:
                        return None
                    call = {
                        "name": "edit",
                        "arguments": {
                            "filePath": arguments["path"],
                            "oldString": arguments["old_str"],
                            "newString": arguments["new_str"],
                        },
                    }
                    try:
                        jsonschema.validate(call, CALL_SCHEMA)
                    except jsonschema.ValidationError:
                        return None
                    calls.append(call)
                elif name == "str_replace_editor" and arguments.get("command") == "view":
                    if set(arguments) != {"command", "path", "view_range"}:
                        return None
                    path, bounds = arguments["path"], arguments["view_range"]
                    if not isinstance(path, str) or any(
                        path.lower().endswith(suffix) for suffix in BINARY_EDITOR_SUFFIXES
                    ):
                        return None
                    if (
                        not isinstance(bounds, list)
                        or len(bounds) != 2
                        or any(type(value) is not int for value in bounds)
                    ):
                        return None
                    start, end = bounds
                    # Native read has a 2000-line default page; open-ended views need a separate contract.
                    if not 1 <= start <= end or end - start + 1 > NATIVE_READ_MAX_LINES:
                        return None
                    calls.append(
                        {"name": "read", "arguments": {"filePath": path, "offset": start, "limit": end - start + 1}}
                    )
                elif name == "str_replace_editor" and arguments.get("command") == "create":
                    if set(arguments) != {"command", "path", "file_text"}:
                        return None
                    path, text = arguments["path"], arguments["file_text"]
                    if not isinstance(path, str) or not isinstance(text, str):
                        return None
                    code = f"with open({path!r}, 'x', encoding='utf-8', newline='') as f: f.write({text!r})"
                    calls.append({"name": "bash", "arguments": {"command": "python3 -c " + shlex.quote(code)}})
                elif name == "finish" and index == len(original) - 1 and len(matches) == 1:
                    if set(arguments) != {"message"} or not isinstance(arguments["message"], str):
                        return None
                    if not any(m["role"] == "system" and FINISH_DESCRIPTION in m["content"] for m in original):
                        return None
                    # The declared completion tool delivers this final text; it performs no environment action.
                    content += arguments["message"]
                elif name == "evaluate_code" and set(arguments) == {"source_code"}:
                    if not isinstance(arguments["source_code"], str):
                        return None
                    if not any(
                        m["role"] == "system"
                        and "Evaluates the input string as Python code and returns the stdout." in m["content"]
                        for m in original
                    ):
                        return None
                    calls.append(
                        {
                            "name": "bash",
                            "arguments": {
                                "command": "python3 -c " + shlex.quote(arguments["source_code"]),
                                "timeout": 120000,
                            },
                        }
                    )
                elif name in KERNEL_CODE_ARGUMENTS and set(arguments) == {KERNEL_CODE_ARGUMENTS[name]}:
                    if name == "sandbox_exec" and not any(
                        m["role"] == "system" and "sandboxed Python container with notebook support" in m["content"]
                        for m in original
                    ):
                        return None
                    if "jupyter_client" not in available_packages:
                        return None
                    if not isinstance(arguments[KERNEL_CODE_ARGUMENTS[name]], str):
                        return None
                    code = arguments[KERNEL_CODE_ARGUMENTS[name]]
                    delimiter = "CELL_" + hashlib.sha256(code.encode()).hexdigest()
                    if delimiter in code.splitlines():
                        raise ValueError("Kernel heredoc delimiter collision")
                    if not kernel_started:
                        calls.extend(kernel_setup_calls())
                        kernel_started = True
                    command = f"python3 /workspace/cell.py /workspace/kernel.json <<'{delimiter}'\n{code}\n{delimiter}"
                    calls.append({"name": "bash", "arguments": {"command": command}})
                else:
                    return None
                started = True
        elif "<tool_call>" in message["content"]:
            return None
        turns.append({"source_message_index": index, "content": content, "reasoning_content": None, "calls": calls})
    if not started and any(
        message["role"] in {"tool", "ipython"}
        or re.search(r'"(?:commands|keystrokes|task_complete)"\s*:', message["content"])
        or "New Terminal Output:" in message["content"]
        or "Current Terminal Screen:" in message["content"]
        for message in original
        if message["role"] != "system"
    ):
        return None
    return {
        "decision": "convert",
        "rationale": "Exact known-tool mapping; pending GLM review" if started else "Unused tool-header normalization",
        "turns": turns,
    }


def apply_edits(seed: dict, proposal: dict) -> dict:
    candidate = copy.deepcopy(seed)
    candidate.update(decision=proposal["decision"], rationale=proposal["rationale"])
    turns = {turn["source_message_index"]: turn for turn in candidate["turns"]}
    seen = set()
    for edit in proposal["edits"]:
        index = edit["source_message_index"]
        if index not in turns or index in seen:
            raise ValueError("Edit targets an absent or repeated source turn")
        seen.add(index)
        for field in ("content", "reasoning_content", "calls"):
            if edit[field] is not None:
                turns[index][field] = edit[field]
    return candidate
