# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission guard for the scripted Atlas release pipeline (R40 gate).

Refuses any cluster submission whose stage sources are not maintained modules under
``release_stages/`` at a landed ``atlas-env-audit`` SHA, and any stage source that
carries release-identifying literals (32-hex load IDs, release/snapshot-shaped names).
Release identifiers travel only as typed manifest/artifact fields (see DESIGN.md).

The refusal is hard: there is no bypass flag. A failed guard means the release does
not submit.
"""

import ast
import dataclasses
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

ATLAS_ENV_AUDIT_BRANCH = "atlas-env-audit"
PACKAGE_DIR = Path(__file__).resolve().parent

# A 32-hex load ID anywhere in a string literal, bounded by non-hex characters.
_HEX_LOAD_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
# Release/snapshot-shaped names: "r39-...", "current26-...", at a name boundary.
_RELEASE_NAME = re.compile(r"(?<![0-9a-z])(r\d+[-_]|current\d+[-_])")


@dataclasses.dataclass(frozen=True)
class LiteralViolation:
    """One release-identifying literal found in stage source."""

    path: Path
    line: int
    value: str


def landing_log_path() -> Path:
    """The atlas-push landing log (``ATLAS_PUSH_LOG`` overrides the default)."""
    return Path(os.environ.get("ATLAS_PUSH_LOG", Path.home() / ".local/share/atlas-push/landings.jsonl"))


def scan_release_literals(paths: Iterable[Path]) -> list[LiteralViolation]:
    """Scan Python sources for release-identifying string literals."""
    violations: list[LiteralViolation] = []
    for path in sorted(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise ValueError(f"cannot parse stage source for the guard scan: {path}") from error
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if _HEX_LOAD_ID.search(node.value) or _RELEASE_NAME.search(node.value):
                violations.append(LiteralViolation(path=path, line=node.lineno, value=node.value))
    return violations


def assert_no_release_literals(package_dir: Path = PACKAGE_DIR) -> None:
    """Refuse stage sources that carry release-identifying literals."""
    violations = scan_release_literals(sorted(package_dir.rglob("*.py")))
    if violations:
        detail = "\n".join(f"  {v.path.relative_to(package_dir)}:{v.line}: {v.value!r}" for v in violations)
        raise ValueError(
            "release-identifying literals in stage source; load IDs, snapshot names, pins and"
            f" plan hashes must arrive as typed manifest/artifact fields:\n{detail}"
        )


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def repo_head(repo_root: Path) -> str:
    """The current checkout's HEAD sha."""
    return _git(repo_root, "rev-parse", "HEAD")


def atlas_landed_tip(repo_root: Path) -> str:
    """The current tip of the landed ``atlas-env-audit`` branch."""
    tip = _git(repo_root, "ls-remote", "origin", f"refs/heads/{ATLAS_ENV_AUDIT_BRANCH}")
    if not tip:
        raise ValueError(f"origin has no {ATLAS_ENV_AUDIT_BRANCH} branch; cannot verify a landed SHA")
    return tip.split()[0]


def sha_is_ancestor(head: str, tip: str, repo_root: Path) -> bool:
    """True when ``head`` equals or is an ancestor of ``tip``."""
    if head == tip:
        return True
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, tip],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def sha_in_landing_log(head: str, log_path: Path | None = None) -> bool:
    """True when the atlas-push landing log records ``head``."""
    path = log_path if log_path is not None else landing_log_path()
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("sha") == head:
            return True
    return False


def assert_stage_sources_landed(repo_root: Path, package_dir: Path = PACKAGE_DIR) -> None:
    """Refuse submission when stage sources are not at a landed atlas-env-audit SHA."""
    resolved = package_dir.resolve()
    expected_parent = (repo_root / "experiments/post_training/task_curation").resolve()
    if resolved != expected_parent / "release_stages":
        raise ValueError(
            "stage sources must live under experiments/post_training/task_curation/release_stages/:" f" {resolved}"
        )
    head = repo_head(repo_root)
    if sha_in_landing_log(head):
        return
    tip = atlas_landed_tip(repo_root)
    if not sha_is_ancestor(head, tip, repo_root):
        raise ValueError(
            f"checkout HEAD {head} is not landed on {ATLAS_ENV_AUDIT_BRANCH} (tip {tip});"
            " land the release pipeline via atlas-push before submitting stages"
        )


def assert_submission_allowed(repo_root: Path, package_dir: Path = PACKAGE_DIR) -> None:
    """The full submission gate: landed modules plus a clean literal scan.

    Preflight-receipt freshness joins here once ``preflight.py`` lands (DESIGN.md §6);
    the landed-SHA and literal checks are independent of it.
    """
    assert_stage_sources_landed(repo_root, package_dir)
    assert_no_release_literals(package_dir)
