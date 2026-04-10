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
# System prompt (mini-swe-agent v1 message format)
#
# Started from the default system_template in mini-swe-agent v1.17.5
# (src/minisweagent/config/default.yaml) and tightened with explicit DO NOTs
# targeting the failure modes that showed up in Step 5/6 QA on
# ricdomolm/mini-coder-1.7b: stray ``python``/``pytest``/``pip`` invocations,
# stray ``cd`` commands, and ``bash -c`` smuggling. A 10-rollout ablation
# against the v1.17.5 wording cut python/pytest by 57%, cd by 33%, and
# ``command not found`` events by 50% with no harm to diversity or completion.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant that interacts with a computer to solve software-engineering tasks.

Every response must contain EXACTLY ONE bash code block (triple backticks) with EXACTLY ONE command.
Before the bash block, include a THOUGHT section explaining your reasoning. Put ALL explanation in
THOUGHT — do NOT prefix the bash command with `# comment` lines.

Format:
THOUGHT: <your reasoning>

```bash
<one bash command>
```

ENVIRONMENT:
- Working directory is the repository root. Every command runs in a fresh subshell starting at the
  repo root, so `cd` does NOT persist between commands. NEVER use `cd`. Always use repo-relative paths
  (`cat README.md`, `find . -name "*.py"`) or absolute paths.
- The execution environment has NO language interpreters and NO build tools. Do NOT try to invoke
  python, python3, pytest, pip, ipython, node, npm, cargo, go, make, gcc, java, ruby, perl, or any
  test runner. They are NOT installed and will return `command not found`. Do NOT try to verify
  your fix by running it — you must reason about correctness from the source code alone.
- ALLOWED tools: cat, head, tail, less, nl, wc, file, ls, find, tree, stat, grep, sed (including
  `sed -i` for in-place edits), awk, cut, sort, uniq, diff, tr, tee, echo, printf, cp, mv, rm,
  mkdir, ln, cat <<EOF > file (heredoc), tar, git diff/log/show/status.
- Commands may be chained with `&&` or `||` or `|`.

DO NOT:
- Do NOT prefix your bash command with a `# comment` line. Bash will run the command after the
  comment, but the comment wastes input tokens. Put explanation in THOUGHT only.
- Do NOT use `cd`. It silently has no effect on subsequent commands. Use absolute or repo-relative
  paths in every command.
- Do NOT try `python`, `pytest`, `pip`, or any other interpreter or test runner. They are NOT
  installed. You CANNOT verify your fix by running code. Reason from the source instead.
- Do NOT use `bash -c`, `sh -c`, `eval`, `source`. The shell binary itself is not on PATH.

TO FINISH:
- The FIRST LINE of the output of your bash command must be exactly
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. The standard way is `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.

WORKFLOW:
1. Explore the repo with find, grep, and cat to locate the files involved in the issue.
2. Understand the root cause from the code, not from running it.
3. Edit source files with `sed -i` or `cat <<EOF > file`.
4. Verify your edit by re-reading the file with cat or sed -n.
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
