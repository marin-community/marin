# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Path-filter CLI replacing dorny/paths-filter for Marin GitHub Actions workflows.

Run as: uv run python scripts/workflows/changes.py match ...
"""

import fnmatch
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class PathDecision:
    name: str
    matched: bool
    paths: tuple[str, ...]


def changed_paths(base_ref: str, head_ref: str, *, repo: Path) -> tuple[str, ...]:
    """Return paths changed between two refs, sorted and relative to repo root.

    Uses three-dot (merge-base) diff semantics: `git diff --name-only base...head`.
    This matches paths-filter behaviour for pull_request events where the diff
    is relative to the merge-base of the two branches rather than a direct
    two-dot range.

    The diff filter `ACMRTUX` excludes deleted files (D) so that a deletion
    alone does not trigger a group. Renames appear under the new path (R).

    Args:
        base_ref: Base commit SHA or ref (the older side of the diff).
        head_ref: Head commit SHA or ref (the newer side of the diff).
        repo: Absolute path to the repository root (.git must exist here).

    Returns:
        Sorted tuple of changed paths relative to repo root.

    Raises:
        ValueError: If either ref does not exist in the repository.
        subprocess.CalledProcessError: If the git invocation fails for any other reason.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMRTUX",
            f"{base_ref}...{head_ref}",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [p for p in result.stdout.splitlines() if p]
    return tuple(sorted(paths))


def _path_matches_pattern(path: str, pattern: str) -> bool:
    """Return True if path matches pattern using fnmatch glob semantics.

    Uses fnmatch.fnmatch which treats both * and ** as matching any sequence
    of characters including path separators. This gives the intuitive
    `lib/marin/**` behaviour: match any file at any depth under lib/marin/.
    Plain file patterns like `pyproject.toml` are matched literally.
    """
    return fnmatch.fnmatch(path, pattern)


def match_groups(paths: Iterable[str], groups: Mapping[str, Sequence[str]]) -> tuple[PathDecision, ...]:
    """Match changed paths against named glob groups.

    Patterns use fnmatch glob semantics with ** support.
    A pattern prefixed with `!` is a negation: a path matches a group if it
    matches at least one positive pattern and does not match any negative
    pattern. Positive patterns are evaluated first; negations are applied
    after, in declaration order.

    Args:
        paths: Iterable of changed file paths (relative to repo root).
        groups: Mapping of group name to list of glob patterns. Negated
            patterns start with `!`.

    Returns:
        Tuple of PathDecision, one per group, in the same order as groups.
    """
    path_list = list(paths)
    decisions: list[PathDecision] = []

    for name, patterns in groups.items():
        positive = [p for p in patterns if not p.startswith("!")]
        negative = [p[1:] for p in patterns if p.startswith("!")]

        matched_paths: list[str] = []
        for path in path_list:
            hits_positive = any(_path_matches_pattern(path, pat) for pat in positive)
            if not hits_positive:
                continue
            hits_negative = any(_path_matches_pattern(path, pat) for pat in negative)
            if not hits_negative:
                matched_paths.append(path)

        decisions.append(PathDecision(name=name, matched=bool(matched_paths), paths=tuple(matched_paths)))

    return tuple(decisions)


def _find_repo_root(start: Path) -> Path:
    """Walk up from start until a directory containing .git is found.

    Args:
        start: Directory to begin the search from.

    Returns:
        Path to the repository root.

    Raises:
        click.UsageError: If no .git directory is found before the filesystem root.
    """
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            raise click.UsageError(f"No git repository found above {start}")
        current = parent


def _write_github_output(decisions: tuple[PathDecision, ...]) -> None:
    """Write group results to $GITHUB_OUTPUT in the format key=value.

    If $GITHUB_OUTPUT is unset, this is a no-op so local users can pass
    --github-output without error.

    Args:
        decisions: Sequence of PathDecision results to emit.
    """
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a") as fh:
        for d in decisions:
            fh.write(f"{d.name}={'true' if d.matched else 'false'}\n")


def _parse_group_option(value: str) -> tuple[str, list[str]]:
    """Parse a single --group NAME=PAT1,PAT2 option value.

    Args:
        value: Raw string like ``marin=lib/marin/**,tests/**``.

    Returns:
        Tuple of (name, patterns) where patterns have surrounding whitespace
        stripped.

    Raises:
        click.BadParameter: If the value does not contain ``=``.
    """
    if "=" not in value:
        raise click.BadParameter(f"Expected NAME=PATTERNS, got: {value!r}")
    name, _, raw_patterns = value.partition("=")
    patterns = [p.strip() for p in raw_patterns.split(",") if p.strip()]
    return name.strip(), patterns


@click.group()
def cli() -> None:
    """Path-filter CLI for Marin GitHub Actions."""


@cli.command()
@click.option("--base", default=None, help="Base commit SHA for diff.")
@click.option("--head", default=None, help="Head commit SHA for diff.")
@click.option(
    "--group",
    "raw_groups",
    multiple=True,
    metavar="NAME=PATTERNS",
    help="Repeatable. NAME=comma-separated glob patterns. Prefix with ! to negate.",
)
@click.option("--always-match", is_flag=True, default=False, help="Match all groups unconditionally (no diff).")
@click.option("--github-output", is_flag=True, default=False, help="Append results to $GITHUB_OUTPUT if set.")
@click.option(
    "--repo",
    "repo_path",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Path to repository root. Defaults to nearest .git ancestor of CWD.",
)
def match(
    base: str | None,
    head: str | None,
    raw_groups: tuple[str, ...],
    always_match: bool,
    github_output: bool,
    repo_path: Path | None,
) -> None:
    """Evaluate which path groups are touched by a git diff.

    Use --always-match for workflow_dispatch and schedule triggers where every
    group should be considered changed regardless of actual file changes.
    """
    if always_match and (base is not None or head is not None):
        raise click.UsageError("--always-match is mutually exclusive with --base / --head")

    if not always_match and (base is None or head is None):
        raise click.UsageError("Both --base and --head are required unless --always-match is passed")

    repo = repo_path if repo_path is not None else _find_repo_root(Path.cwd())

    groups: dict[str, list[str]] = {}
    for raw in raw_groups:
        name, patterns = _parse_group_option(raw)
        groups[name] = patterns

    if always_match:
        decisions = tuple(PathDecision(name=name, matched=True, paths=()) for name in groups)
        reason = "manual-or-scheduled"
    else:
        # base and head are guaranteed non-None here due to the check above.
        diff = changed_paths(base, head, repo=repo)  # type: ignore[arg-type]
        decisions = match_groups(diff, groups)
        reason = "diff"

    output = {
        "groups": [{"name": d.name, "matched": d.matched, "paths": list(d.paths)} for d in decisions],
        "reason": reason,
    }
    click.echo(json.dumps(output))

    for d in decisions:
        label = "matched" if d.matched else "not matched"
        click.echo(f"{d.name}: {label} ({len(d.paths)} paths)", err=True)

    if github_output:
        _write_github_output(decisions)


if __name__ == "__main__":
    cli()
