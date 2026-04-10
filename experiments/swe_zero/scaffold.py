# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO execution-free agent scaffold using mini-swe-agent v1 format.

mini-swe-agent v1 (https://github.com/SWE-agent/mini-SWE-agent/tree/v1.17.5)
uses bash-only interaction: the model outputs reasoning followed by a single
bash command in a ```bash block. The environment executes the command and
returns observations like "Observation: <output>".

Submission signal: the first line of the bash command output must be
"COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" (typically via `echo` or after a
successful command).

For SWE-ZERO (execution-free), we simulate bash command outputs from a
patch-derived repo snapshot instead of actually executing them.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field

from experiments.swe_zero.data_loader import PRRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (mini-swe-agent v1 style)
#
# This matches the default system_template from mini-swe-agent v1.17.5
# (src/minisweagent/config/default.yaml).
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that can interact with a computer to solve tasks.

You will be given a task and you will need to interact with the computer to solve it.
Every response must contain EXACTLY ONE bash code block (in triple backticks) with EXACTLY ONE command.
Include a THOUGHT section explaining your reasoning before the bash block.

Format:
THOUGHT: <your reasoning here>

```bash
<your bash command>
```

Notes:
- Directory or environment variable changes are NOT persistent. Every command runs in a new subshell.
- Commands can be chained with `&&` or `||`.
- To finish, the FIRST LINE of the output of your bash command must be exactly:
  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  e.g. `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`

Workflow:
1. Analyze the codebase by finding and reading relevant files (use `find`, `grep`, `cat`)
2. Understand the issue and identify the root cause
3. Edit source code to fix the issue (use `sed`, `cat <<EOF >`, etc.)
4. Verify your changes by reading the modified files
5. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\
"""


def build_task_message(pr: PRRecord) -> str:
    """Build the initial user message describing the task from a PR record."""
    parts = [f"Please solve this issue in the repository {pr.repo}.\n"]
    parts.append(f"## Issue\n\n{pr.problem_statement}")
    if pr.interface:
        parts.append(f"\n## Relevant Interface\n\n{pr.interface}")
    parts.append(
        f"\n## Repository Info\n\n"
        f"- **Language**: {pr.language}\n"
        f"- **Base commit**: {pr.base_commit}\n"
        f"- The repository is already cloned at `/repo`."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Repo snapshot from patch diffs
# ---------------------------------------------------------------------------


@dataclass
class RepoSnapshot:
    """
    Lightweight in-memory snapshot of the repo state at base_commit.

    Populated from the patch diffs — enough for the model to produce
    meaningful bash-based traces (cat, grep, find, sed).
    """

    files: dict[str, str] = field(default_factory=dict)
    repo_name: str = ""

    @classmethod
    def from_pr(cls, pr: PRRecord) -> RepoSnapshot:
        """Extract file paths and content from the PR patches."""
        snap = cls(repo_name=pr.repo)
        for patch_text in [pr.patch, pr.test_patch]:
            if not patch_text:
                continue
            current_file = None
            lines: list[str] = []
            for line in patch_text.split("\n"):
                if line.startswith("diff --git"):
                    if current_file and lines:
                        snap.files[current_file] = "\n".join(lines)
                    match = re.search(r"b/(.+)$", line)
                    current_file = "/repo/" + match.group(1) if match else None
                    lines = []
                elif line.startswith("+++") or line.startswith("---"):
                    continue
                elif line.startswith("@@"):
                    continue
                elif current_file is not None:
                    if line.startswith(" ") or line.startswith("+"):
                        lines.append(line[1:])
                    elif line.startswith("-"):
                        lines.append(line[1:])
            if current_file and lines:
                snap.files[current_file] = "\n".join(lines)
        return snap


# ---------------------------------------------------------------------------
# Bash command parser and simulator
# ---------------------------------------------------------------------------


def extract_bash_command(response: str) -> str | None:
    """Extract the bash command from a mini-swe-agent v1 response.

    v1 uses the regex `r"```bash\\s*\\n(.*?)\\n```"` to find exactly one
    triple-backtick bash block.
    """
    pattern = r"```bash\s*\n(.*?)\n```"
    matches = re.findall(pattern, response, re.DOTALL)
    if len(matches) >= 1:
        return matches[0].strip()
    return None


def simulate_bash(command: str, snapshot: RepoSnapshot) -> str:
    """
    Simulate a bash command against the repo snapshot.

    Supports common read-only commands (cat, find, grep, ls, head, tail)
    and write commands (sed, patch-like echo/tee). Since this is SWE-ZERO
    (execution-free), we don't actually execute anything.
    """
    command = command.strip()

    # Submit signal: simulate `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` so
    # the first line of the observation triggers the v1 has_finished check.
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
        return "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n"

    # cat with heredoc: write file (cat <<EOF > file or cat << 'EOF' > file)
    if "<<" in command and (">" in command or "cat" in command):
        # Extract target file from redirect
        redirect_match = re.search(r">\s*(\S+)", command)
        if redirect_match:
            path = redirect_match.group(1)
            # Extract content between heredoc markers
            heredoc_match = re.search(r"<<\s*'?(\w+)'?\s*\n?(.*)", command, re.DOTALL)
            if heredoc_match:
                marker = heredoc_match.group(1)
                rest = heredoc_match.group(2)
                # Find the end marker
                end_idx = rest.find(f"\n{marker}")
                if end_idx >= 0:
                    file_content = rest[:end_idx]
                else:
                    file_content = rest
                full_path = path if path.startswith("/") else f"/repo/{path}"
                snapshot.files[full_path] = file_content
                return ""
        return ""

    # cat: read file
    if command.startswith("cat "):
        path = _extract_first_path(command[4:])
        content = _resolve_file(path, snapshot)
        if content is not None:
            return content
        return f"cat: {path}: No such file or directory"

    # head: read first N lines
    if command.startswith("head "):
        parts = command.split()
        n_lines = 10
        path = ""
        for i, p in enumerate(parts[1:], 1):
            if p == "-n" and i + 1 < len(parts):
                try:
                    n_lines = int(parts[i + 1])
                except ValueError:
                    pass
            elif p.startswith("-") and p[1:].isdigit():
                n_lines = int(p[1:])
            elif not p.startswith("-"):
                path = p
        content = _resolve_file(path, snapshot)
        if content is not None:
            return "\n".join(content.split("\n")[:n_lines])
        return f"head: cannot open '{path}' for reading: No such file or directory"

    # find: list files
    if command.startswith("find "):
        parts = command.split()
        search_dir = "/repo"
        name_pattern = ""
        for i, p in enumerate(parts[1:], 1):
            if p == "-name" and i + 1 < len(parts):
                name_pattern = parts[i + 1].strip("'\"")
            elif not p.startswith("-") and i == 1:
                search_dir = p
        matches = []
        for fpath in sorted(snapshot.files):
            if not fpath.startswith(search_dir):
                continue
            if name_pattern:
                fname = fpath.rsplit("/", 1)[-1]
                if not _glob_match(fname, name_pattern):
                    continue
            matches.append(fpath)
        return "\n".join(matches) if matches else ""

    # grep: search in files
    if command.startswith("grep "):
        return _simulate_grep(command, snapshot)

    # ls: list directory
    if command.startswith("ls "):
        path = _extract_first_path(command[3:])
        if not path:
            path = "/repo"
        entries = set()
        for fpath in snapshot.files:
            if fpath.startswith(path.rstrip("/") + "/"):
                remainder = fpath[len(path.rstrip("/")) + 1 :]
                entry = remainder.split("/")[0]
                entries.add(entry)
        return "\n".join(sorted(entries)) if entries else f"ls: cannot access '{path}': No such file or directory"

    # sed: simulate inline edit
    if command.startswith("sed "):
        return _simulate_sed(command, snapshot)

    # echo/printf with redirect: write file
    if ">>" in command or (">" in command and not command.startswith("grep")):
        return ""  # Acknowledge write silently

    # tee: write to file
    if "tee " in command:
        return ""

    # wc: word/line count
    if command.startswith("wc "):
        path = _extract_first_path(command[3:])
        content = _resolve_file(path, snapshot)
        if content is not None:
            lines = content.split("\n")
            words = sum(len(l.split()) for l in lines)
            chars = len(content)
            return f"  {len(lines)}  {words} {chars} {path}"
        return f"wc: {path}: No such file or directory"

    # diff, patch, etc.: acknowledge
    if command.startswith("diff ") or command.startswith("patch "):
        return ""

    # pwd
    if command.strip() == "pwd":
        return "/repo"

    # Anything else: return empty (simulated no-op)
    return ""


def _extract_first_path(args: str) -> str:
    """Extract the first non-flag argument as a file path."""
    for part in args.split():
        if not part.startswith("-"):
            return part.strip("'\"")
    return ""


def _resolve_file(path: str, snapshot: RepoSnapshot) -> str | None:
    """Resolve a file path against the snapshot."""
    if path in snapshot.files:
        return snapshot.files[path]
    # Try with /repo prefix
    if not path.startswith("/repo") and f"/repo/{path.lstrip('/')}" in snapshot.files:
        return snapshot.files[f"/repo/{path.lstrip('/')}"]
    # Try partial match
    for fpath, content in snapshot.files.items():
        if fpath.endswith("/" + path.lstrip("/")):
            return content
    return None


def _glob_match(name: str, pattern: str) -> bool:
    """Simple glob matching for find -name."""
    regex = pattern.replace(".", r"\.").replace("*", ".*").replace("?", ".")
    return bool(re.match(f"^{regex}$", name))


def _simulate_grep(command: str, snapshot: RepoSnapshot) -> str:
    """Simulate grep against the snapshot."""
    parts = shlex.split(command)
    recursive = False
    line_numbers = False
    pattern = ""
    paths = []

    i = 1
    while i < len(parts):
        p = parts[i]
        if p in ("-r", "-R", "--recursive", "-rn", "-rn"):
            recursive = True
            if "n" in p:
                line_numbers = True
        elif p in ("-n", "--line-number"):
            line_numbers = True
        elif p in ("-l", "--files-with-matches"):
            pass  # simplified
        elif p.startswith("-"):
            pass
        elif not pattern:
            pattern = p
        else:
            paths.append(p)
        i += 1

    if not pattern:
        return ""

    search_files = {}
    if recursive or not paths:
        search_files = snapshot.files
    else:
        for p in paths:
            content = _resolve_file(p, snapshot)
            if content is not None:
                search_files[p] = content

    results = []
    try:
        regex = re.compile(pattern)
    except re.error:
        regex = re.compile(re.escape(pattern))

    for fpath, content in sorted(search_files.items()):
        if paths and not recursive:
            if not any(fpath.endswith(p.lstrip("/")) for p in paths):
                continue
        for lineno, line in enumerate(content.split("\n"), 1):
            if regex.search(line):
                if line_numbers:
                    results.append(f"{fpath}:{lineno}:{line}")
                else:
                    results.append(f"{fpath}:{line}")
    return "\n".join(results[:50])


def _simulate_sed(command: str, snapshot: RepoSnapshot) -> str:
    """Simulate sed -i for inline edits."""
    # Match: sed -i 's/old/new/g' file
    match = re.search(r"""sed\s+(?:-i\s*(?:''|"")?\s+)?'s/(.+?)/(.*?)/g?'\s+(\S+)""", command)
    if not match:
        match = re.search(r"""sed\s+(?:-i\s*(?:''|"")?\s+)?"s/(.+?)/(.*?)/g?"\s+(\S+)""", command)
    if match:
        old_pat, new_str, path = match.group(1), match.group(2), match.group(3)
        content = _resolve_file(path, snapshot)
        if content is not None:
            try:
                new_content = re.sub(old_pat, new_str, content)
                # Store back
                if path in snapshot.files:
                    snapshot.files[path] = new_content
                elif f"/repo/{path.lstrip('/')}" in snapshot.files:
                    snapshot.files[f"/repo/{path.lstrip('/')}"] = new_content
            except re.error:
                pass
        return ""
    return ""
