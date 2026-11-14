# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve source turns while replacing actions and binding real observations."""

import ast
import json
import re
from pathlib import Path

import jsonschema

from experiments.tool_normalization.scope import PYTHON_TOOLS
from experiments.tool_normalization.terminal_protocol import terminal_protocol

FORBIDDEN_TOOLS = PYTHON_TOOLS | {"execute_bash", "str_replace_editor"}
TARGET_TOOLS = frozenset({"bash", "edit", "read", "write", "grep", "glob", "todowrite"})
HEADER = re.compile(r"<\|start_header_id\|>(system|user|assistant|tool|ipython)<\|end_header_id\|>")
NATIVE_SCHEMAS = json.loads(Path(__file__).with_name("opencode-tools.json").read_text())
CALL_SCHEMA = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "name": {"enum": [tool["function"]["name"]]},
                "arguments": {**tool["function"]["parameters"], "additionalProperties": False},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        for tool in NATIVE_SCHEMAS
    ]
}
TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "source_message_index": {"type": "integer"},
        "content": {"type": ["string", "null"], "description": "Null preserves the original visible content exactly."},
        "reasoning_content": {"type": ["string", "null"], "description": "Null preserves original reasoning exactly."},
        "calls": {"type": "array", "items": CALL_SCHEMA},
    },
    "required": ["source_message_index", "content", "reasoning_content", "calls"],
    "additionalProperties": False,
}
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"enum": ["convert", "hold", "environment_required"]},
        "rationale": {"type": "string"},
        "turns": {"type": "array", "items": TURN_SCHEMA},
    },
    "required": ["decision", "rationale", "turns"],
    "additionalProperties": False,
}


def source_messages(text: str) -> list[dict]:
    headers = list(HEADER.finditer(text))
    if not headers or text[: headers[0].start()].strip() not in ("", "<|begin_of_text|>"):
        raise ValueError("Unrecognized source serialization")
    messages = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end() : end]
        match = re.fullmatch(r"(.*?)<\|(?:eot_id|eom_id)\|>\s*(?:<\|end_of_text\|>)?\s*", body, re.DOTALL)
        if match is None:
            raise ValueError("Source message lacks a complete boundary")
        message = {"role": header.group(1), "content": match.group(1)}
        if message["role"] == "assistant" and "<|start_think|>" in message["content"]:
            channels = re.fullmatch(r"\s*<\|start_think\|>(.*?)<\|end_think\|>(.*)", message["content"], re.DOTALL)
            if channels is None:
                raise ValueError("Malformed source reasoning boundary")
            message["reasoning_content"], message["content"] = channels.groups()
        messages.append(message)
    return messages


def resolved_turns(original: list[dict], candidate: dict) -> list[dict]:
    """Resolve explicit copy requests against the complete immutable source."""
    expected = [index for index, message in enumerate(original) if message["role"] == "assistant"]
    if [turn["source_message_index"] for turn in candidate["turns"]] != expected:
        raise ValueError("Candidate must cover every original assistant turn exactly once in order")
    return [
        {
            **turn,
            **{
                channel: (
                    original[turn["source_message_index"]].get(channel, "") if turn[channel] is None else turn[channel]
                )
                for channel in ("content", "reasoning_content")
            },
        }
        for turn in candidate["turns"]
    ]


def calls_for_candidate(original: list[dict], candidate: dict, schemas: list[dict]) -> list[dict]:
    if candidate["decision"] != "convert":
        raise ValueError("Candidate is held")
    turns = resolved_turns(original, candidate)
    terminal = terminal_protocol(original)
    available = {tool["function"]["name"]: tool["function"]["parameters"] for tool in schemas}
    calls = []
    for turn in turns:
        source = original[turn["source_message_index"]]
        if source.get("reasoning_content", "").strip() and not turn["reasoning_content"].strip():
            raise ValueError("Original reasoning channel was emptied or moved to visible content")
        if not turn["calls"] and source["content"].strip() and not turn["content"].strip():
            raise ValueError("Original visible answer was omitted")
        text = turn["content"] + turn["reasoning_content"]
        if any(name in text for name in FORBIDDEN_TOOLS) or "<tool_call>" in text or "<|" in text:
            raise ValueError("Legacy tools or serialized control markers in candidate text")
        if terminal and any(f'"{key}"' in turn["content"] for key in ("keystrokes", "task_complete")):
            raise ValueError("Terminus action/completion envelope remains in the native candidate")
        for call in turn["calls"]:
            if call["name"] not in TARGET_TOOLS & available.keys():
                raise ValueError("Tool is not available in the target harness")
            arguments = call["arguments"]
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            jsonschema.validate(arguments, available[call["name"]])
            calls.append({"name": call["name"], "arguments": arguments})
    return calls


def has_source_calls(original: list[dict]) -> bool:
    return any(message["role"] == "assistant" and "<tool_call>" in message["content"] for message in original)


def target_tool_instructions(message: dict, names: set[str]) -> dict:
    """Replace the dataset's explicit tool-list annotation without changing its task."""
    if message["role"] == "system" and message["content"].lstrip().startswith(
        "You are opencode, an interactive CLI tool"
    ):
        content = message["content"]
        for old, new in {
            "first use the WebFetch tool to gather information": "use the available tools to gather information",
            "- When doing file search, prefer to use the Task tool in order to reduce context usage.": (
                "- Use the available file-search tools for file searches."
            ),
            "[use the ls tool to list the files in the current directory": (
                "[use bash to list the files in the current directory"
            ),
        }.items():
            content = content.replace(old, new)
        content = re.sub(
            r"Skills provide specialized instructions and workflows for specific tasks\.\n"
            r"Use the skill tool to load a skill when a task matches its description\.\n"
            r"<available_skills>.*?</available_skills>\n?",
            "",
            content,
            flags=re.DOTALL,
        )
        return {**message, "content": content}
    if message["role"] != "user":
        return message.copy()

    def replacement(match: re.Match) -> str:
        declared = ast.literal_eval(match.group(1))
        if not isinstance(declared, list) or not all(isinstance(name, str) for name in declared):
            raise ValueError("Unrecognized source tool-list annotation")
        return "**Allowed Tools:** " + repr(sorted(names))

    return {
        **message,
        "content": re.sub(r"^\*\*Allowed Tools:\*\* (\[[^\n]*\])$", replacement, message["content"], flags=re.MULTILINE),
    }


def assembled_messages(original: list[dict], candidate: dict, observations: list[dict]) -> list[dict]:
    terminal = terminal_protocol(original)
    turns = {turn["source_message_index"]: turn for turn in resolved_turns(original, candidate)}
    output = []
    names = {call["name"] for turn in turns.values() for call in turn["calls"]}
    call_index = 0
    unanswered = 0
    for index, message in enumerate(original):
        if terminal and index in terminal.observation_indices:
            continue
        if terminal and index == terminal.task_index:
            output.append({"role": "user", "content": terminal.task})
            continue
        if message["role"] in ("tool", "ipython"):
            continue
        if message["role"] == "system" and index == 0 and message["content"].lstrip().startswith("Reasoning: "):
            # The Datakit auxiliary header is replaced by the pinned target tool definitions at export.
            instructions = message["content"].split("### Tools", 1)[0]
            instructions = re.sub(r"^\s*Reasoning: /(?:no)?think\s*", "", instructions)
            if instructions.strip():
                output.append({"role": "system", "content": instructions})
            continue
        if message["role"] != "assistant":
            output.append(target_tool_instructions(message, names))
            continue
        turn = turns[index]
        if not turn["calls"]:
            output.append(
                {
                    "role": "assistant",
                    "content": turn["content"],
                    "reasoning_content": turn["reasoning_content"],
                    "tool_calls": [],
                }
            )
        for within_turn, call in enumerate(turn["calls"]):
            observation = observations[call_index] if call_index < len(observations) else None
            if observation is None:
                source_calls = re.findall(r"<tool_call>.*?</tool_call>", message["content"], re.DOTALL)
                if index != len(original) - 1 or len(turn["calls"]) != 1 or len(source_calls) != 1:
                    raise ValueError("Missing observation for a nonterminal operation")
                unanswered += 1
            call_id = f"replay_{call_index}"
            if observation is not None and (
                observation["tool_call_id"] != call_id or observation["name"] != call["name"]
            ):
                raise ValueError("Observation pairing mismatch")
            native_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)},
            }
            # Match tested sequential execution; a parallel batch could reorder dependent edits.
            output.append(
                {
                    "role": "assistant",
                    "content": turn["content"] if within_turn == 0 else "",
                    "reasoning_content": turn["reasoning_content"] if within_turn == 0 else "",
                    "tool_calls": [native_call],
                }
            )
            if observation is not None:
                output.append(observation)
            call_index += 1
    if call_index != len(observations) + unanswered:
        raise ValueError("Unused execution observations")
    return output
