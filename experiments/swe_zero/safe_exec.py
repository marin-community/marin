# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Sandboxed bash executor for SWE-ZERO rollouts.

The agent runs real bash subprocesses against a cloned repo working tree, but
a regex blocklist rejects any command that would *execute* code, hit the
network, or modify the git history. This matches the SWE-ZERO constraint:
"You CANNOT RUN CODE for any purpose" while still letting the agent use
``cat``/``find``/``grep``/``sed`` against the actual repo.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Commands that would execute user code or invoke language runtimes/build tools.
# Matches the command name as a whole word, so ``mypython`` is fine but
# ``python``, ``python3``, ``python3.11`` are blocked. We also block common
# wrappers like ``coverage``, ``tox``, and the ``-c`` form of ``bash``/``sh``
# to make it harder for the model to escape via ``bash -c "python ..."``.
_EXEC_BLOCKLIST = re.compile(
    r"""
    (?<![\w./-])
    (?:
        python\d?(?:\.\d+)? |
        ipython\d? |
        pytest |
        coverage |
        tox |
        nosetests |
        unittest |
        pip\d? |
        pipx |
        uv |
        poetry |
        conda |
        npm | npx | yarn | pnpm | node | deno |
        bun |
        cargo | rustc |
        go(?:\s+(?:run|test|build|install|generate)) |
        java | javac | mvn | gradle | sbt |
        ruby | rails | rake | bundle |
        php |
        perl |
        Rscript |
        ghc | runhaskell |
        make | cmake | ninja |
        gcc | g\+\+ | clang | clang\+\+ |
        ld |
        dotnet
    )
    (?![\w-])
    """,
    re.VERBOSE,
)

# Bash/sh in -c form (which can be used to smuggle execution through other
# commands). We allow ``bash --version`` etc. but reject anything that runs a
# script or command.
_SHELL_EXEC_BLOCKLIST = re.compile(r"""(?<![\w./-])(?:bash|sh|zsh|fish|dash)\s+(?:-c|-ic|-i|-l|-s|<|<<|\S+\.sh)""")

# Network ingress/egress.
_NETWORK_BLOCKLIST = re.compile(
    r"""(?<![\w./-])(?:curl|wget|nc|ncat|netcat|telnet|ssh|scp|rsync|ftp|sftp|aria2c|httpie|http)(?![\w-])"""
)

# git operations that would mutate history or hit the network. Read-only
# operations like ``git diff``, ``git log``, ``git show`` remain allowed.
_GIT_BLOCKLIST = re.compile(
    r"""(?<![\w./-])git\s+
        (?:
            fetch | pull | clone | push |
            remote\s+(?:add|set-url) |
            reset\s+--hard |
            checkout\s+--detach |
            init | am
        )
        (?![\w-])
    """,
    re.VERBOSE,
)

# Anything that looks like a shebang or direct ./ invocation.
_DIRECT_EXEC = re.compile(r"""(?:^|[;|&]|\$\()\s*(?:\.{1,2}/|/)\S""")

_BLOCKLISTS: tuple[tuple[re.Pattern, str], ...] = (
    (_EXEC_BLOCKLIST, "running code or build tools"),
    (_SHELL_EXEC_BLOCKLIST, "spawning a child shell"),
    (_NETWORK_BLOCKLIST, "network access"),
    (_GIT_BLOCKLIST, "git history mutation or network operations"),
    (_DIRECT_EXEC, "direct file execution"),
)


@dataclass(frozen=True)
class ExecResult:
    """Outcome of running a single bash command in the sandbox."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool

    def as_observation(self, max_chars: int = 8000) -> str:
        """Format combined stdout+stderr as the agent's observation."""
        out = self.stdout
        if self.stderr:
            if out:
                out = out + "\n" + self.stderr
            else:
                out = self.stderr
        if len(out) > max_chars:
            head = out[: max_chars // 2]
            tail = out[-max_chars // 2 :]
            out = f"{head}\n\n... (output truncated, {len(out) - max_chars} chars elided) ...\n\n{tail}"
        if self.timed_out:
            out = (out + "\n\n[command timed out]").strip()
        return out


def check_blocklist(command: str) -> str | None:
    """Return a human-readable rejection reason if the command is blocked."""
    for pattern, reason in _BLOCKLISTS:
        if pattern.search(command):
            return f"command blocked: {reason}"
    return None


def safe_exec(
    command: str,
    cwd: str,
    *,
    timeout_seconds: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> ExecResult:
    """Run a bash command against ``cwd`` with the SWE-ZERO sandbox blocklist.

    The command runs through ``bash -c`` so pipes/redirects/heredocs work.
    Blocked commands return immediately with a synthetic stderr message; no
    subprocess is spawned in that case.
    """
    rejection = check_blocklist(command)
    if rejection is not None:
        return ExecResult(
            stdout="",
            stderr=f"{rejection}: this is the SWE-ZERO execution-free environment",
            returncode=126,
            timed_out=False,
        )

    env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": cwd,
        "SHELL": "/bin/bash",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PAGER": "cat",
        "TERM": "dumb",
    }
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return ExecResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return ExecResult(stdout=stdout, stderr=stderr, returncode=124, timed_out=True)
    except (FileNotFoundError, PermissionError) as e:
        return ExecResult(stdout="", stderr=f"sandbox error: {e}", returncode=127, timed_out=False)
