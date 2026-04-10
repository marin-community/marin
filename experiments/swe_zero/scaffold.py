# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
SWE-ZERO execution-free agent scaffold using mini-swe-agent v1 format.

mini-swe-agent v1 (https://github.com/SWE-agent/mini-SWE-agent/tree/v1.17.5)
uses bash-only interaction: the model outputs reasoning followed by a single
bash command in a ```bash block. The environment runs the command and returns
``Observation: <output>``.

Submission signal: the first line of the bash command output must be
``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` (typically via ``echo``).

For SWE-ZERO (execution-free) we run the agent's bash commands against a
**real** checkout of the repo at ``base_commit`` (with ``test_patch`` applied)
in a sandboxed subprocess that blocks language runtimes/build tools/network
access. The agent gets full read access to the source — what it cannot do is
execute the result and check whether it's correct. See ``safe_exec.py`` for
the blocklist and ``worktree.py`` for the per-rollout checkout management.
"""

from __future__ import annotations

import logging
import re

from experiments.swe_zero.data_loader import PRRecord
from experiments.swe_zero.safe_exec import ExecResult, safe_exec
from experiments.swe_zero.worktree import WorkTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (mini-swe-agent v1 style)
#
# This matches the default system_template from mini-swe-agent v1.17.5
# (src/minisweagent/config/default.yaml), tightened to spell out the SWE-ZERO
# constraint that no language runtimes are available.
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

CONSTRAINTS:
- Each command runs from the repository root as the working directory; use relative paths
  (e.g. `cat README.md`, `find . -name "*.py"`). The env var $REPO_PATH also points there.
- cd is NOT persistent across commands. Every command runs in a fresh subshell at the repo root.
- The development environment is unavailable. You CANNOT RUN CODE for any purpose.
  Blocked: python, pytest, pip, npm, node, cargo, go, make, gcc, java, ruby, perl, etc.
  Blocked: bash -c, sh -c, curl, wget, git fetch/pull/clone/push.
  Allowed: cat, head, tail, find, grep, ls, wc, awk, sed (including sed -i for in-place edits),
           cat <<EOF > file (heredoc), echo > file, tee, diff, git diff/log/show.
- Commands can be chained with && or ||.
- To finish, the FIRST LINE of the output of your bash command must be exactly:
  COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
  e.g.  echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Workflow:
1. Explore the repo with find/grep/cat to locate the relevant files.
2. Understand the issue and identify the root cause.
3. Edit source code to fix the issue (sed -i, cat <<EOF, etc.).
4. Verify your changes by re-reading the modified files.
5. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.\
"""


def build_task_message(pr: PRRecord) -> str:
    """Build the initial user message describing the task from a PR record."""
    parts = [f"Please solve this issue in the repository {pr.repo}.\n"]
    parts.append(f"## Issue\n\n{pr.problem_statement}")
    if pr.interface:
        parts.append(f"\n## Relevant Interface\n\n{pr.interface}")
    parts.append(
        f"\n## Repository Info\n\n"
        f"- Repo: {pr.repo}\n"
        f"- Language: {pr.language}\n"
        f"- Base commit: {pr.base_commit}\n"
        f"- The full repository is checked out as your current working directory; use relative paths."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Bash command parsing
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


# ---------------------------------------------------------------------------
# Bash execution against the worktree
# ---------------------------------------------------------------------------


def execute_in_worktree(command: str, worktree: WorkTree, *, timeout_seconds: float = 30.0) -> ExecResult:
    """Run ``command`` against the worktree's checkout via the sandbox.

    The command runs with ``cwd`` pinned to ``worktree.path``. We expose the
    same path under ``/repo`` via ``REPO_PATH`` so prompts can refer to it
    consistently across rollouts; the agent can use either the absolute
    ``worktree.path`` or the symbol-friendly ``$REPO_PATH``.
    """
    return safe_exec(
        command,
        cwd=str(worktree.path),
        timeout_seconds=timeout_seconds,
        extra_env={"REPO_PATH": str(worktree.path)},
    )
