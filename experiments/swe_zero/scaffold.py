# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO execution-free agent scaffold.

Defines the tools, system prompt, and multi-turn conversation structure for
generating execution-free agentic rollouts in the style of SWE-ZERO.

The agent can:
  - Read files via str_replace_editor (view command)
  - Search for files/symbols via find_file / search
  - Edit files via str_replace_editor (str_replace / create / insert)
  - Think step-by-step via the think tool
  - Signal completion via finish

The agent CANNOT execute code — this is the key constraint of SWE-ZERO.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from experiments.swe_zero.data_loader import PRRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (Gemma 4 function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "A file editor tool for viewing and editing files. "
                "Commands: view (read file contents), str_replace (replace text), "
                "create (create new file), insert (insert text at line)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "str_replace", "create", "insert"],
                        "description": "The editor command to run.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "File content (for create command).",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Text to replace (for str_replace).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text (for str_replace/insert).",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number to insert at (for insert).",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Line range [start, end] to view.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "Search for files by name pattern in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "File name pattern to search for (glob-style).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in. Defaults to repo root.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for a text pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text/regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use this tool for extended reasoning about the problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Your reasoning about the problem.",
                    },
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that you have completed the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Summary of what was done.",
                    },
                },
                "required": ["message"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a highly skilled software engineer tasked with resolving a GitHub issue.

**IMPORTANT CONSTRAINTS:**
- The development environment is unavailable. You CANNOT RUN CODE for any purpose.
- You cannot use: python, pytest, node, npm, cargo, go, make, pip, apt, or any execution commands.
- You must resolve the issue purely through code reading, reasoning, and editing.

**Available tools:**
1. `str_replace_editor` — View, create, and edit files
2. `find_file` — Search for files by name pattern
3. `search` — Search for text patterns in files
4. `think` — Extended reasoning (use freely for complex analysis)
5. `finish` — Signal task completion

**Workflow phases:**
1. **Reading**: Understand the issue and identify relevant files
2. **Exploration**: Navigate the codebase to understand the architecture
3. **Analysis**: Determine the root cause and plan the fix
4. **Implementation**: Apply code changes using str_replace_editor
5. **Review**: Verify your changes are correct and complete, then finish

Work methodically. Read relevant files before editing. Make minimal, targeted changes.\
"""


def build_task_message(pr: PRRecord) -> str:
    """Build the initial user message describing the task from a PR record."""
    parts = [
        f"## Issue\n\n{pr.problem_statement}",
    ]
    if pr.interface:
        parts.append(f"\n## Relevant Interface\n\n{pr.interface}")
    parts.append(
        f"\n## Repository\n\n- **Repo**: {pr.repo}\n- **Language**: {pr.language}\n- **Base commit**: {pr.base_commit}"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Simulated environment: produces tool-call responses without execution
# ---------------------------------------------------------------------------


@dataclass
class RepoSnapshot:
    """
    A lightweight in-memory snapshot of the repo state at base_commit.

    For the SWE-ZERO approach we don't need the full repo — we only need
    the files that appear in the gold patch and test patch so that the
    agent's view/search calls return plausible content.

    In the MVP we populate this from the patch diffs themselves, which is
    enough for the model to produce meaningful tool-call traces.
    """

    files: dict[str, str] = field(default_factory=dict)
    repo_name: str = ""

    @classmethod
    def from_pr(cls, pr: PRRecord) -> RepoSnapshot:
        """Extract file paths and rough content hints from the PR patches."""
        snap = cls(repo_name=pr.repo)
        # Parse files mentioned in the patch
        for patch_text in [pr.patch, pr.test_patch]:
            if not patch_text:
                continue
            current_file = None
            lines: list[str] = []
            for line in patch_text.split("\n"):
                if line.startswith("diff --git"):
                    if current_file and lines:
                        snap.files[current_file] = "\n".join(lines)
                    # Extract b/ path
                    match = re.search(r"b/(.+)$", line)
                    current_file = "/" + match.group(1) if match else None
                    lines = []
                elif line.startswith("+++") or line.startswith("---"):
                    continue
                elif line.startswith("@@"):
                    continue
                elif current_file is not None:
                    # Keep context and added lines (strip the diff prefix)
                    if line.startswith(" ") or line.startswith("+"):
                        lines.append(line[1:])
                    elif line.startswith("-"):
                        # Lines that exist in the base (before the patch)
                        lines.append(line[1:])
            if current_file and lines:
                snap.files[current_file] = "\n".join(lines)
        return snap


def simulate_tool_response(
    tool_name: str,
    arguments: dict,
    snapshot: RepoSnapshot,
) -> str:
    """
    Produce a simulated tool-call response.

    Since SWE-ZERO is execution-free, we simulate file reads from the
    repo snapshot and acknowledge edits without actual execution.
    """
    if tool_name == "str_replace_editor":
        cmd = arguments.get("command", "")
        path = arguments.get("path", "")

        if cmd == "view":
            content = snapshot.files.get(path)
            if content is None:
                # Try partial match
                for fpath, fcontent in snapshot.files.items():
                    if fpath.endswith(path.lstrip("/")):
                        content = fcontent
                        break
            if content is None:
                return f"Error: File {path} not found in the repository."

            view_range = arguments.get("view_range")
            if view_range and len(view_range) == 2:
                lines = content.split("\n")
                start, end = max(0, view_range[0] - 1), min(len(lines), view_range[1])
                numbered = [f"{i + view_range[0]}:\t{l}" for i, l in enumerate(lines[start:end])]
                return "\n".join(numbered)

            lines = content.split("\n")
            numbered = [f"{i + 1}:\t{l}" for i, l in enumerate(lines)]
            return "\n".join(numbered)

        elif cmd == "str_replace":
            old_str = arguments.get("old_str", "")
            new_str = arguments.get("new_str", "")
            content = snapshot.files.get(path, "")
            if old_str not in content:
                return f"Error: old_str not found in {path}. No changes made."
            snapshot.files[path] = content.replace(old_str, new_str, 1)
            return f"Successfully replaced text in {path}."

        elif cmd == "create":
            file_text = arguments.get("file_text", "")
            snapshot.files[path] = file_text
            return f"Successfully created {path}."

        elif cmd == "insert":
            new_str = arguments.get("new_str", "")
            insert_line = arguments.get("insert_line", 0)
            content = snapshot.files.get(path, "")
            lines = content.split("\n")
            lines.insert(insert_line, new_str)
            snapshot.files[path] = "\n".join(lines)
            return f"Successfully inserted text at line {insert_line} in {path}."

        return f"Error: Unknown command '{cmd}'."

    elif tool_name == "find_file":
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", "/")
        matches = []
        for fpath in snapshot.files:
            if PurePosixPath(fpath).match(pattern):
                if fpath.startswith(search_path):
                    matches.append(fpath)
        if not matches:
            return f"No files matching '{pattern}' found."
        return "Found files:\n" + "\n".join(sorted(matches))

    elif tool_name == "search":
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", "/")
        results = []
        for fpath, content in snapshot.files.items():
            if not fpath.startswith(search_path):
                continue
            for i, line in enumerate(content.split("\n"), 1):
                if re.search(pattern, line):
                    results.append(f"{fpath}:{i}: {line.strip()}")
        if not results:
            return f"No matches for '{pattern}'."
        return "\n".join(results[:50])  # Limit output

    elif tool_name == "think":
        return "Your thought has been recorded."

    elif tool_name == "finish":
        return "Task marked as complete."

    return f"Error: Unknown tool '{tool_name}'."
