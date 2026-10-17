# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A pull-request reviewer for the agent harness: it reads the diff and the repository with its own tools.

The agent gets its instructions from the task's public context and owns its tool loop. Each model call goes through
the harness's ``complete``, so the recorder applies the limits and keeps the conversation. The tools only read: the
repository is the copy that the Iris job bundle holds, which is the pull request's head.
"""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rolloutengine.agent import AgentCompletion, AgentRun
from rolloutengine.contracts import Prompt
from rolloutengine.harness import HarnessContext
from rolloutengine.shell import task_prompt

from experiments.post_training.managed_agent.chat_model import CACHE_KEY_OPTION

REPOSITORY = Path(__file__).resolve().parents[3]
MAX_TOOL_OUTPUT = 20_000
MAX_READ_LINES = 400
MAX_SEARCH_MATCHES = 200
SEARCHED_SUFFIXES = (".py", ".md", ".toml", ".yaml")

SYSTEM_PROMPT = """You review a pull request in the Marin repository. Your tools list the changed files, show the diff \
of one file, read files of the repository at the pull request's head, and search the repository. Read the diff and \
the code before you judge them.

Report only problems that you can point to in the code: wrong behavior, a missed case, a test that does not test its \
claim, a doc sentence that the code contradicts, or code that can be simpler. Do not report style preferences.

When you are done, answer without a tool call. The answer is your review in Markdown: one short summary paragraph, \
then at most eight findings. Give each finding a severity (blocker, should fix, or nit), a file:line, the problem, \
and the fix."""

LAST_TURN_PROMPT = "One model turn is left. Write your final answer now, without a tool call."


@dataclass(frozen=True)
class PullRequest:
    """The pull request under review: its description and its unified diff from the base to the head."""

    number: int
    title: str
    body: str
    diff: str

    def file_diffs(self) -> dict[str, str]:
        """The diff of each changed file, by its path at the head."""
        files = {}
        for chunk in re.split(r"(?m)^(?=diff --git )", self.diff):
            if chunk.startswith("diff --git "):
                header = chunk.split("\n", 1)[0]
                files[header.rsplit(" b/", 1)[-1]] = chunk
        return files


def _function(name: str, description: str, properties: Mapping[str, Any], required: tuple[str, ...]) -> dict:
    parameters = {"type": "object", "properties": dict(properties), "required": list(required)}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {**parameters, "additionalProperties": False},
        },
    }


TOOLS = (
    _function("changed_files", "List the files that the pull request changes, with added and removed lines.", {}, ()),
    _function("file_diff", "Show the pull request's diff of one file.", {"path": {"type": "string"}}, ("path",)),
    _function(
        "read_file",
        f"Read lines of a repository file at the pull request's head, at most {MAX_READ_LINES} lines at a time.",
        {"path": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
        ("path",),
    ),
    _function(
        "search",
        "Search Python, Markdown, TOML, and YAML files below a repository directory for a regular expression.",
        {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Default: lib/rolloutengine"}},
        ("pattern",),
    ),
)


def _repository_path(path: str) -> Path:
    resolved = (REPOSITORY / path).resolve()
    if not resolved.is_relative_to(REPOSITORY):
        raise ValueError(f"{path} is outside the repository")
    return resolved


def _bounded(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text
    return text[:MAX_TOOL_OUTPUT] + f"\n[{len(text) - MAX_TOOL_OUTPUT} more characters not shown]"


def _changed_files(pull_request: PullRequest) -> str:
    lines = []
    for path, chunk in pull_request.file_diffs().items():
        body = [line for line in chunk.splitlines() if not line.startswith(("+++", "---"))]
        added = sum(line.startswith("+") for line in body)
        removed = sum(line.startswith("-") for line in body)
        lines.append(f"{path} +{added} -{removed}")
    return "\n".join(lines)


def _file_diff(pull_request: PullRequest, path: str) -> str:
    files = pull_request.file_diffs()
    if path not in files:
        return f"The pull request does not change {path}. Call changed_files for the list."
    return files[path]


def _read_file(path: str, start: int = 1, end: int | None = None) -> str:
    lines = _repository_path(path).read_text(errors="replace").splitlines()
    first = max(start, 1)
    last = first + MAX_READ_LINES - 1 if end is None else min(end, first + MAX_READ_LINES - 1)
    numbered = [f"{number}: {lines[number - 1]}" for number in range(first, min(last, len(lines)) + 1)]
    return "\n".join(numbered) + f"\n[{path} has {len(lines)} lines]"


def _search(pattern: str, path: str = "lib/rolloutengine") -> str:
    expression = re.compile(pattern)
    root = _repository_path(path)
    matches = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix not in SEARCHED_SUFFIXES or ".venv" in file.parts:
            continue
        for number, line in enumerate(file.read_text(errors="replace").splitlines(), start=1):
            if expression.search(line):
                matches.append(f"{file.relative_to(REPOSITORY)}:{number}: {line.strip()}")
                if len(matches) == MAX_SEARCH_MATCHES:
                    return "\n".join(matches) + f"\n[stopped at {MAX_SEARCH_MATCHES} matches]"
    return "\n".join(matches) or "No matches."


def run_tool(pull_request: PullRequest, name: str, arguments: str) -> str:
    """The observation for one tool call. A bad call gets an error message, so that the agent can correct it."""
    tools: dict[str, Callable[..., str]] = {
        "changed_files": lambda: _changed_files(pull_request),
        "file_diff": lambda path: _file_diff(pull_request, path),
        "read_file": _read_file,
        "search": _search,
    }
    try:
        parsed = json.loads(arguments or "{}")
        if name not in tools:
            return f"Unknown tool: {name}"
        return _bounded(tools[name](**parsed))
    except (TypeError, ValueError, OSError, re.error) as error:
        return f"Tool error: {error}"


def reviewer(pull_request: PullRequest, *, max_turns: int) -> AgentRun:
    """The agent program: it calls tools until it answers without one, and asks for the answer when one turn is left.

    ``max_turns`` is the attempt's ``RolloutLimits.max_turns``, so that the agent answers before the limit.
    """

    async def run(context: HarnessContext, complete: AgentCompletion) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *task_prompt(context.task, None, ()).messages,
        ]
        # The same options on every turn make each request a reply in the same conversation.
        options = {"tools": list(TOOLS), CACHE_KEY_OPTION: context.run_id}
        for remaining in range(max_turns - 1, -1, -1):
            turn = await complete(Prompt(tuple(messages), options))
            messages.append(turn.message)
            calls = turn.message.get("tool_calls") or ()
            if not calls or remaining == 0:
                return
            for call in calls:
                function = call["function"]
                observation = run_tool(pull_request, function["name"], function["arguments"])
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
            if remaining == 1:
                messages.append({"role": "user", "content": LAST_TURN_PROMPT})

    return run
