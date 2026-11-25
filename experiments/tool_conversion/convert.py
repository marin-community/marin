# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize this campaign's calls without executing source programs.

Read README.md, "Scope and limits — read before reuse", before invoking this module.
"""

import ast
import copy
import json
import re
import shlex
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import jsonschema

from experiments.tool_conversion.kernel import kernel_command

HEADER = re.compile(r"<\|start_header_id\|>(system|user|assistant|tool|ipython)<\|end_header_id\|>")
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
THOUGHT = re.compile(r"<\|start_think\|>(.*?)<\|end_think\|>", re.DOTALL)
RESULT = re.compile(r'(\s*<tool_response(?: name="([^"]+)")?>)(.*)(</tool_response>\s*)', re.DOTALL)
KERNEL_ARGUMENTS = {
    "stateful_python_code_exec": "code",
    "python_interpreter": "script",
    "jupyter_kernel_run": "cell_content",
    "sandbox_exec": "snippet",
}
PYTHON_TOOLS = frozenset(KERNEL_ARGUMENTS) | {"evaluate_code", "arc_python_executor"}
CONTROL_TOOLS = frozenset({"think", "submit", "finish"})
SCOPE_GUIDE = "experiments/tool_conversion/README.md: Scope and limits — read before reuse"
REAL_TOOLS = frozenset(
    {
        "bash",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "task",
        "webfetch",
        "todowrite",
        "todoread",
        "skill",
        "question",
        "invalid",
        "apply_patch",
        "exec_command",
        "write_stdin",
        "view_image",
        "str_replace_editor",
        "websearch",
        "web_search",
        "powershell",
        "find",
        "ls",
        "update_plan",
        "list_mcp_resources",
        "list_mcp_resource_templates",
        "read_mcp_resource",
    }
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the current working directory. Returns stdout and stderr.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
CAMPAIGN_TOOLS = (
    REAL_TOOLS
    | PYTHON_TOOLS
    | CONTROL_TOOLS
    | {"execute_bash", "web-search", "commit_solution", "explicit_solution_with_augmentations"}
)


class CampaignScopeError(RuntimeError):
    """An input lies outside the reviewed campaign boundary."""


class Disposition(StrEnum):
    EXCLUDED = "out_of_scope"
    NO_CALLS = "tool_free"
    UNCHANGED = "real_tools_only"
    VERIFIED = "rule_converted_adapter_checked"
    RESIDUAL = "mapping_unresolved"


@dataclass
class Turn:
    role: str
    header: str
    body: str
    ending: str


@dataclass
class Change:
    message: int
    call: int
    before: dict[str, Any]
    after: dict[str, Any] | None
    replacement: str | None = None
    removed_acknowledgement: int | None = None


@dataclass
class Conversion:
    disposition: Disposition
    reason: str
    text: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    changes: list[Change] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    candidate: dict[str, Any] = field(default_factory=dict)


def excluded_source(source: str) -> bool:
    parts = source.lower().split("/")
    return (
        "terminus" in source.lower()
        or source == "agenttrove"
        or ("nemotron" in source.lower() and any(p in {"interactive_agent", "tool_calling"} for p in parts))
    )


def parse_trace(text: str) -> tuple[str, list[Turn]]:
    """Parse complete serialized turns, preserving every body and boundary byte."""
    headers = list(HEADER.finditer(text))
    if not headers or text[: headers[0].start()].strip() not in {"", "<|begin_of_text|>"}:
        raise ValueError("Unrecognized conversation prefix")
    turns = []
    for i, header in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        match = re.fullmatch(
            r"(.*?)(<\|(?:eot_id|eom_id)\|>\s*(?:<\|end_of_text\|>)?\s*)", text[header.end() : end], re.DOTALL
        )
        if match is None:
            raise ValueError("Incomplete message boundary")
        turns.append(Turn(header[1], header[0], match[1], match[2]))
    return text[: headers[0].start()], turns


def visible_calls(body: str) -> list[re.Match[str]]:
    """Ignore call-like examples inside an explicit reasoning channel."""
    thoughts = list(THOUGHT.finditer(body))
    visible = THOUGHT.sub("", body)
    if "<|start_think|>" in visible or "<|end_think|>" in visible:
        raise ValueError("Unclosed reasoning channel")
    matches = [m for m in CALL.finditer(body) if not any(t.start() <= m.start() < t.end() for t in thoughts)]
    if visible.count("<tool_call>") != len(matches) or visible.count("</tool_call>") != len(matches):
        raise ValueError("Malformed call boundary")
    return matches


def tool_definitions(turns: list[Turn]) -> dict[str, dict]:
    definitions = {}
    decoder = json.JSONDecoder()
    for turn in turns:
        if turn.role != "system":
            continue
        for block in re.findall(r"<tools>(.*?)</tools>", turn.body, re.DOTALL):
            remaining = block.strip()
            while remaining:
                tool, end = decoder.raw_decode(remaining)
                if "function" not in tool:
                    tool = {"type": "function", "function": tool}
                name = tool["function"]["name"]
                if name in definitions and definitions[name] != tool:
                    raise ValueError("Conflicting tool definitions")
                definitions[name] = tool
                remaining = remaining[end:].strip()
    return definitions


def _replace(body: str, edits: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(edits, reverse=True):
        body = body[:start] + replacement + body[end:]
    return body


SHELL_PROGRAMS = frozenset(
    {
        "cat",
        "ls",
        "find",
        "grep",
        "rg",
        "sed",
        "awk",
        "head",
        "tail",
        "wc",
        "echo",
        "printf",
        "pwd",
        "true",
        "false",
        "git",
        "python",
        "python3",
        "pytest",
        "pip",
        "pip3",
        "uv",
        "node",
        "npm",
        "mkdir",
        "rm",
        "cp",
        "mv",
        "touch",
        "chmod",
        "curl",
        "wget",
    }
)


def self_contained_shell(command: str) -> bool:
    """Admit one foreground program, without shell syntax or state mutations."""
    if not command or re.search(r"[\n\r;$`|&<>(){}]", command):
        return False
    words = shlex.split(command)
    return bool(words) and words[0] in SHELL_PROGRAMS


def convert_trace(source: str, text: str, identity: str) -> Conversion:
    """Convert only fully supported traces; partial conversions never become verified.

    Verification concerns the adapter and preserved observations, not the truth
    of historical results. No input program is executed by this function.
    """
    if excluded_source(source):
        return Conversion(Disposition.EXCLUDED, "excluded dataset source", text)
    first_header = HEADER.search(text)
    if first_header is None or text[: first_header.start()].strip() not in {"", "<|begin_of_text|>"}:
        raise CampaignScopeError(f"Unsupported chat format; see {SCOPE_GUIDE}")
    try:
        return _convert(text, identity)
    except (ValueError, KeyError, TypeError, SyntaxError, jsonschema.ValidationError) as error:
        return Conversion(Disposition.RESIDUAL, str(error), text)


def _convert(text: str, identity: str) -> Conversion:
    prefix, original = parse_trace(text)
    if any(t.role == "assistant" and re.search(r'"keystrokes"\s*:', THOUGHT.sub("", t.body)) for t in original):
        return Conversion(Disposition.EXCLUDED, "Terminus protocol", text)
    calls = [
        (i, m, json.loads(m[1])) for i, t in enumerate(original) if t.role == "assistant" for m in visible_calls(t.body)
    ]
    if not calls:
        return Conversion(Disposition.NO_CALLS, "no native tool calls", text)
    names = {c["name"] for _, _, c in calls}
    definitions = tool_definitions(original)
    unknown = (names | definitions.keys()) - CAMPAIGN_TOOLS
    if unknown:
        raise CampaignScopeError(f"Unsupported tool set: {', '.join(sorted(unknown))}; see {SCOPE_GUIDE}")
    targets = names - REAL_TOOLS
    if not targets:
        return Conversion(Disposition.UNCHANGED, "only real harness tools", text)
    unsupported = targets - PYTHON_TOOLS - CONTROL_TOOLS - {"execute_bash"}
    if unsupported:
        return Conversion(Disposition.RESIDUAL, "no grounded mapping: " + ", ".join(sorted(unsupported)), text)
    turns = copy.deepcopy(original)
    changes = []
    rules = set()
    edits: dict[int, list[tuple[int, int, str]]] = {}
    mappings = {}
    removed_results = set()
    kernel_started = False
    reasoning_prefixes = {}
    result_ids = {}
    if len(names & KERNEL_ARGUMENTS.keys()) > 1:
        raise ValueError("Multiple notebook tools may own separate sessions")
    by_turn: dict[int, list[tuple[re.Match[str], dict]]] = {}
    for i, match, call in calls:
        by_turn.setdefault(i, []).append((match, call))
    for i, group in by_turn.items():
        if len(group) != 1:
            raise ValueError("Multiple native calls in one assistant turn are outside this mapping")
        next_assistant = next(
            (j for j in range(i + 1, len(original)) if original[j].role in {"assistant", "user"}), len(original)
        )
        observations = [j for j in range(i + 1, next_assistant) if original[j].role in {"tool", "ipython"}]
        if len(observations) != len(group) and not (i == len(original) - 1 and not observations):
            raise ValueError("Call/result count mismatch")
        for k, (match, call) in enumerate(group):
            name, args = call["name"], call["arguments"]
            if not isinstance(args, dict):
                raise ValueError("Tool arguments must be an object")
            observation = observations[k] if k < len(observations) else None
            result = RESULT.fullmatch(original[observation].body) if observation is not None else None
            if observation is not None and (result is None or result[2] not in (None, name)):
                raise ValueError("Tool response name does not match call")
            mapped = copy.deepcopy(call)
            if observation is not None:
                result_ids[observation] = call.get("id", f"source_{i}_{k}")
            if name in KERNEL_ARGUMENTS:
                key = KERNEL_ARGUMENTS[name]
                if set(args) != {key} or not isinstance(args[key], str):
                    raise ValueError("Unsupported Python arguments")
                description = definitions.get(name, {}).get("function", {}).get("description", "").lower()
                if not any(word in description for word in ("jupyter", "notebook", "ipython")):
                    raise ValueError("Missing notebook execution contract")
                ast.parse(args[key])
                if result and any(
                    s in result[3] for s in ("Traceback", "Timed out", "KeyboardInterrupt", "data:image/", "<image")
                ):
                    raise ValueError("Notebook error, timeout or rich output needs a separate mapping")
                mapped.update(
                    name="bash",
                    arguments={"command": kernel_command(args[key], identity, initialize=not kernel_started)},
                )
                kernel_started = True
                rules.add("persistent-ipython-v1")
            elif name == "evaluate_code":
                if set(args) != {"source_code"} or not isinstance(args["source_code"], str):
                    raise ValueError("Unsupported evaluate_code arguments")
                description = definitions.get(name, {}).get("function", {}).get("description", "")
                if "returns the stdout" not in description:
                    raise ValueError("Missing stateless stdout execution contract")
                # Only one Python operation is safe without an explicit reset contract.
                if sum(c["name"] in PYTHON_TOOLS for _, _, c in calls) != 1:
                    raise ValueError("Multiple evaluate_code calls may share interpreter state")
                ast.parse(args["source_code"])
                mapped.update(name="bash", arguments={"command": "python3 -c " + shlex.quote(args["source_code"])})
                rules.add("python-stdout-v1")
            elif name == "arc_python_executor":
                raise ValueError("ARC executor includes task-specific evaluation; bare Python is not equivalent")
            elif name == "execute_bash":
                if set(args) != {"command"} or not isinstance(args["command"], str):
                    raise ValueError("Unsupported execute_bash arguments")
                if any(
                    not self_contained_shell(c["arguments"].get("command", ""))
                    for _, _, c in calls
                    if c["name"] in {"execute_bash", "bash"}
                ):
                    raise ValueError("Persistent shell state requires explicit conversion")
                if result and any(
                    marker in result[3]
                    for marker in ("exit code -1", "Process exited with code -1", "[Command timed out")
                ):
                    raise ValueError("Running shell process requires an interactive session mapping")
                mapped["name"] = "bash"
                rules.add("execute-bash-v1")
            elif name in CONTROL_TOOLS:
                if len(group) != 1:
                    raise ValueError("Mixed control and execution calls need order-aware mapping")
                if name == "think":
                    if set(args) != {"thought"} or not isinstance(args["thought"], str):
                        raise ValueError("Unsupported think arguments")
                    if result and result[3].strip() not in {
                        "",
                        "Your thought has been logged.",
                        "Your thought has been logged",
                    }:
                        raise ValueError("Think result contains substantive information")
                    replacement = "<|start_think|>" + args["thought"] + "<|end_think|>"
                    if THOUGHT.search(original[i].body):
                        raise ValueError("Multiple reasoning segments need an explicit merge")
                else:
                    if next_assistant != len(original):
                        raise ValueError("Completion is not final")
                    if name == "finish" and set(args) == {"message"} and isinstance(args["message"], str):
                        replacement = args["message"]
                    elif name == "submit" and not args and not result:
                        replacement = ""
                    else:
                        raise ValueError("Completion carries environment effects or substantive result")
                    if result:
                        raise ValueError("Completion result must be retained outside assistant completion")
                if name == "think":
                    reasoning_prefixes[i] = replacement
                    edits.setdefault(i, []).append((match.start(), match.end(), ""))
                else:
                    edits.setdefault(i, []).append((match.start(), match.end(), replacement))
                if observation is not None:
                    removed_results.add(observation)
                changes.append(Change(i, k, call, None, replacement, observation))
                mappings[(i, k)] = None
                rules.add(name + "-to-assistant-v1")
                continue
            if mapped != call:
                edits.setdefault(i, []).append(
                    (
                        match.start(),
                        match.end(),
                        "<tool_call>\n" + json.dumps(mapped, ensure_ascii=False) + "\n</tool_call>",
                    )
                )
                if observation is not None and result is not None and result[2]:
                    turns[observation].body = (
                        result[1].replace('name="' + name + '"', 'name="' + mapped["name"] + '"') + result[3] + result[4]
                    )
                changes.append(Change(i, k, call, mapped))
            mappings[(i, k)] = mapped
    for i, replacements in edits.items():
        turns[i].body = _replace(turns[i].body, replacements)
        if i in reasoning_prefixes:
            body = turns[i].body
            leading = len(body) - len(body.lstrip())
            turns[i].body = body[:leading] + reasoning_prefixes[i] + body[leading:]
    used = {c["name"] for c in mappings.values() if c is not None}
    tools = _rewrite_declarations(turns, used, definitions, {change.before["name"] for change in changes}, rules)
    output = prefix + "".join(t.header + t.body + t.ending for i, t in enumerate(turns) if i not in removed_results)
    if output == text or not changes:
        raise ValueError("No actual tool change")
    messages, candidates = _structured_turns(turns, removed_results, result_ids)
    parse_trace(output)
    for change in changes:
        after = change.after
        if after is not None:
            jsonschema.validate(after["arguments"], BASH_TOOL["function"]["parameters"])
    return Conversion(
        Disposition.VERIFIED,
        "deterministic adapter checks; historical observations not reexecuted",
        output,
        messages,
        tools,
        changes,
        sorted(rules),
        {"decision": "convert", "turns": candidates},
    )


def _rewrite_declarations(
    turns: list[Turn], used: set[str], definitions: dict[str, dict], converted: set[str], rules: set[str]
) -> list[dict]:
    # Replace only explicit tool declarations, retaining all other system content.
    tools = [definitions[n] for n in sorted(definitions.keys() - converted - {"bash"})]
    if used - definitions.keys() - {"bash"}:
        raise ValueError("Missing declared schema for unchanged real tool")
    if "bash" in used or "bash" in definitions:
        tools.append(definitions.get("bash", BASH_TOOL))
    for turn in turns:
        if turn.role == "system":
            if "think-to-assistant-v1" in rules:
                turn.body = turn.body.replace("Reasoning: /nothink", "Reasoning: /think")
            turn.body = re.sub(
                r"<tools>.*?</tools>",
                lambda _: "<tools>\n" + "\n".join(json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>",
                turn.body,
                flags=re.DOTALL,
            )
    return tools


def _structured_turns(
    turns: list[Turn], removed_results: set[int], result_ids: dict[int, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = []
    candidates = []
    for i, turn in enumerate(turns):
        if i in removed_results:
            continue
        message: dict[str, Any] = {"role": turn.role, "content": turn.body}
        if turn.role == "assistant":
            matches = visible_calls(turn.body)
            reasoning = THOUGHT.findall(turn.body)
            if reasoning and (len(reasoning) != 1 or not turn.body.lstrip().startswith("<|start_think|>")):
                raise ValueError("Reasoning must precede assistant content")
            content = CALL.sub("", THOUGHT.sub("", turn.body))
            actions = [json.loads(m[1]) for m in matches]
            message.update(
                content=content,
                reasoning_content="\n".join(reasoning),
                tool_calls=[
                    {
                        "id": c.get("id", f"source_{i}_{k}"),
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for k, c in enumerate(actions)
                ],
            )
            candidates.append(
                {
                    "source_message_index": i,
                    "content": content,
                    "reasoning_content": message["reasoning_content"],
                    "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in actions],
                }
            )
        elif turn.role in {"tool", "ipython"}:
            result = RESULT.fullmatch(turn.body)
            if result is None:
                raise ValueError("Malformed result")
            message.update(role="tool", content=result[3], name=result[2], tool_call_id=result_ids[i])
        messages.append(message)
    return messages, candidates
