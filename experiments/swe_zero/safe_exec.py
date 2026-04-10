# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Sandboxed bash executor for SWE-ZERO rollouts.

The agent runs real bash subprocesses against a cloned repo working tree.
We restrict what the agent can run by setting ``PATH`` to a directory
containing symlinks **only** to a whitelist of allowed binaries (cat, grep,
sed, find, awk, ls, ...). Anything outside the whitelist - python, pytest,
pip, npm, curl, etc. - returns the standard ``command not found`` error from
bash with no special-case logic in our code. Heredoc bodies and string
literals containing the names of blocked commands no longer trigger false
positives because bash never tries to exec them.

This matches the SWE-ZERO constraint ("You CANNOT RUN CODE for any
purpose") while exposing the standard bash UX for syntax errors and missing
commands. Limitations:

- Absolute path invocations (``/usr/bin/python3 foo.py``) bypass the PATH
  restriction. We accept this for the MVP because (a) the model is unlikely
  to spontaneously construct absolute interpreter paths and (b) we can layer
  bubblewrap on top later for hermetic isolation.
- Shell builtins (``cd``, ``echo``, ``set``, ``source``, ``eval``, ...) do
  not go through PATH and are always available. None of them can execute
  external code on their own; ``source``/``eval`` running an external
  command still depends on PATH.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Whitelist of binaries the agent is allowed to run via bash. Bash builtins
# (cd, echo, set, source, eval, ...) work regardless of PATH and are not
# listed here. Adding a command to this list is the only way to expose a new
# binary to the agent.
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        # File reading
        "cat",
        "head",
        "tail",
        "less",
        "more",
        "nl",
        "wc",
        "od",
        "xxd",
        "file",
        "tac",
        "rev",
        "fold",
        "fmt",
        # Navigate / inspect
        "ls",
        "find",
        "tree",
        "pwd",
        "stat",
        "readlink",
        "realpath",
        "basename",
        "dirname",
        # Search
        "grep",
        "egrep",
        "fgrep",
        # Text processing
        "sed",
        "awk",
        "cut",
        "sort",
        "uniq",
        "tr",
        "tee",
        "echo",
        "printf",
        "diff",
        "cmp",
        "comm",
        "join",
        "paste",
        "split",
        "csplit",
        "expand",
        "unexpand",
        "yes",
        # File management (no-execute)
        "cp",
        "mv",
        "rm",
        "mkdir",
        "rmdir",
        "touch",
        "ln",
        "chmod",
        # Archives (read-mostly)
        "tar",
        "gzip",
        "gunzip",
        "zcat",
        "bzip2",
        "bunzip2",
        "bzcat",
        "xz",
        "unxz",
        "xzcat",
        # Hashes / checksums
        "md5sum",
        "sha1sum",
        "sha256sum",
        "sha512sum",
        "cksum",
        # Misc shell helpers
        "true",
        "false",
        "test",
        "[",
        "expr",
        "date",
        "env",
        "id",
        "whoami",
        "hostname",
        "uname",
        "which",
        "type",
        "sleep",
        # git: read-only operations are the most useful (diff, log, show,
        # status). The mutating ones (commit, push, fetch, ...) still work
        # because we whitelist the binary, not subcommands. They have no
        # effect outside the throwaway worktree though, so we accept this.
        "git",
    }
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


# Lazily-built directory of symlinks. One per process - threadsafe init via
# the lock so concurrent rollouts don't race to build it.
_path_dir_lock = threading.Lock()
_path_dir: Path | None = None
_SYSTEM_BIN_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _build_whitelist_path_dir() -> Path:
    """Create a tempdir of symlinks to every binary in ``ALLOWED_COMMANDS``.

    Resolves each binary via the *system* PATH (not the restricted PATH) so
    we always pick up the real coreutils. Missing binaries are silently
    skipped — the agent will see ``command not found`` if it tries them, same
    as any other unknown command.
    """
    d = Path(tempfile.mkdtemp(prefix="swe_zero_path_"))
    resolved = 0
    for cmd in ALLOWED_COMMANDS:
        src = shutil.which(cmd, path=_SYSTEM_BIN_PATH)
        if src is None:
            continue
        try:
            os.symlink(src, d / cmd)
            resolved += 1
        except FileExistsError:
            pass
    logger.info("safe_exec: built whitelist PATH dir at %s with %d/%d binaries", d, resolved, len(ALLOWED_COMMANDS))
    return d


def get_whitelist_path_dir() -> Path:
    """Return the lazily-initialized whitelist directory shared by this process."""
    global _path_dir
    with _path_dir_lock:
        if _path_dir is None:
            _path_dir = _build_whitelist_path_dir()
    return _path_dir


def safe_exec(
    command: str,
    cwd: str,
    *,
    timeout_seconds: float = 30.0,
    extra_env: dict[str, str] | None = None,
) -> ExecResult:
    """Run a bash command against ``cwd`` with a PATH-whitelisted sandbox.

    The command runs through ``bash -c`` so pipes/redirects/heredocs work
    normally. Bash will only successfully exec binaries that exist in the
    whitelist directory; everything else returns the usual ``command not
    found`` error with exit code 127.
    """
    path_dir = get_whitelist_path_dir()
    env = {
        "PATH": str(path_dir),
        "HOME": cwd,
        "SHELL": "/bin/bash",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PAGER": "cat",
        "TERM": "dumb",
    }
    if extra_env:
        env.update(extra_env)

    # Capture as bytes and decode with errors="replace" so non-UTF-8 output
    # (e.g. binaries the agent accidentally cats) does not crash the rollout.
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", command],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=False,
            timeout=timeout_seconds,
            check=False,
        )
        return ExecResult(
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            returncode=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        stdout_bytes = e.stdout if isinstance(e.stdout, bytes) else b""
        stderr_bytes = e.stderr if isinstance(e.stderr, bytes) else b""
        return ExecResult(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=124,
            timed_out=True,
        )
    except (FileNotFoundError, PermissionError) as e:
        return ExecResult(stdout="", stderr=f"sandbox error: {e}", returncode=127, timed_out=False)
