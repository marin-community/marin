# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright Prime Intellect. Licensed under Apache-2.0.
# Extracted without behavioral edits from research-environments b41d8759070715dc102000e06fba0d73ebce9955
# and verifiers 672d020c4dcabf5127a76f51aeb4adaf5b89d3fc.
# Annotation strings keep this guest module importable by the source images' Python 3.6.

import json
import re
from textwrap import dedent

SOLVER_SYSTEM_PROMPT = """\
You must not search for, open, fetch, view, or copy anything that could reveal this task's reference solution \
or upstream fix. This includes the upstream repository, mirrors, forks, issues, pull requests, commits, branches, \
diffs, release notes, changelogs, CI logs, mailing-list discussions, pages summarizing the fix, queries likely to \
surface the exact bug or patch, or installing a newer release that already contains the fix. Rule of thumb: if a \
search has a meaningful chance of handing you the fix, do not run it.

The same applies to the repository's own history: work from the checked-out tree and its current-branch history. \
Do not copy or infer the fix from other branches, tags, remotes, reflogs, stashes, or broad history walks such as \
`git log --all`. Ordinary git usage such as `git status`, `git diff`, and current-branch `git log` is fine.
"""


ENV = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
}


RESTORE = r"""
git checkout "$base" -- tests/ test/ Test/ Tests/ 2>/dev/null || true
git ls-tree -r --name-only "$base" 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      git checkout "$base" -- "$path" 2>/dev/null || true ;;
  esac
done
git ls-files --cached --others --exclude-standard 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    tests/*|test/*|Test/*|Tests/*|test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      if ! git cat-file -e "$base:$path" 2>/dev/null; then
        rm -f -- "$path"; git rm -q --cached -- "$path" 2>/dev/null || true
      fi ;;
  esac
done
"""


def _ids(raw: "str | list[str] | None") -> "list[str]":
    """F2P/P2P ids arrive as a list or a JSON-encoded string; normalize to list[str]."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return [raw] if raw else []
    if isinstance(parsed, list):
        return [str(t).strip() for t in parsed if t]
    return [parsed.strip()] if isinstance(parsed, str) and parsed else []


REPO_PATH = "/testbed"


LINK_VENV = (
    "for d in /opt/miniconda3/envs/testbed /opt/conda/envs/testbed; do "
    '[ -d "$d" ] && ln -sfn "$d" /root/.venv && break; done; true'
)


INSTALL_RG = (
    "command -v rg >/dev/null 2>&1 || { "
    "V=14.1.1; "
    "curl -sSL --max-time 100 "
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    "${V}/ripgrep-${V}-x86_64-unknown-linux-musl.tar.gz "
    "| tar xz -C /tmp "
    "&& install -m 755 /tmp/ripgrep-${V}-x86_64-unknown-linux-musl/rg /usr/local/bin/rg; "
    "}"
)


REVERT_TEST_FILES = r"""
git apply --numstat "$patch" | cut -f3- | while IFS= read -r path; do
  path=${path#\"}; path=${path%\"}
  if git cat-file -e "$base:$path" 2>/dev/null; then
    git checkout "$base" -- "$path" 2>/dev/null || true
  else
    rm -f -- "$path"
  fi
done
"""


def build_eval_script(test_cmd: str) -> str:
    """Wrap the row's canonical ``test_cmd`` (whole test-file paths plus the flags the
    repo's config expects: ``LANG=C.UTF-8``, ``-p no:cacheprovider``, sometimes
    ``--cov=pkg``) in conda activation. Exits 0 regardless of the pytest outcome —
    scoring reads the ``-rA`` summary, so a non-zero exit from here means infra, not a
    failing test."""
    return dedent(
        f"""\
        #!/bin/bash
        set -uo pipefail

        cd {REPO_PATH}

        set +u
        if [ -f /opt/miniconda3/bin/activate ]; then
            source /opt/miniconda3/bin/activate
        elif [ -f /opt/conda/bin/activate ]; then
            source /opt/conda/bin/activate
        fi
        conda activate testbed 2>/dev/null || true
        set -u

        set +e
        {test_cmd}
        PYTEST_EXIT=$?
        set -e

        echo ""
        echo "SWELEGO_PYTEST_EXIT=$PYTEST_EXIT"
        exit 0
        """
    )


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


PYTEST_OUTCOMES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")


def parse_pytest_outcomes(output: "str | None") -> "dict[str, str]":
    outcomes: dict[str, str] = {}
    for raw_line in ANSI_RE.sub("", output or "").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue

        outcome, test_id = parts
        if outcome not in PYTEST_OUTCOMES:
            continue

        # SKIPPED short-summary rows use "[N] file.py:line: reason", not a node id.
        if test_id.startswith("["):
            continue

        # These summary rows append " - <reason>"; passing node ids do not.
        if outcome in ("FAILED", "ERROR", "XFAIL", "XPASS") and " - " in test_id:
            parts = test_id.split(" - ")
            for index in range(1, len(parts)):
                node_id = " - ".join(parts[:index])
                if node_id.count("[") == node_id.count("]"):
                    test_id = node_id
                    break
        outcomes[test_id.rstrip()] = outcome
    return outcomes
