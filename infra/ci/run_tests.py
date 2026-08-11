# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the safe unit tests affected by a local branch and working tree.

The default comparison includes committed, staged, unstaged, and untracked
changes from the branch point with main. Tests run once per affected package;
dedicated accelerator and browser suites remain delegated to CI.

Usage:
    uv run --no-project infra/ci/run_tests.py
    uv run --no-project infra/ci/run_tests.py --dry-run
    uv run --no-project infra/ci/run_tests.py --all
    uv run --no-project infra/ci/run_tests.py --base-ref upstream/main -- -x
"""

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infra.ci.select_tests import RUST_SETUP_TAG, SelectionResult, select_changed_tests

DEFAULT_BASE_REFS: tuple[str, ...] = ("origin/HEAD", "origin/main", "main")


class LocalTestError(RuntimeError):
    """A local test plan cannot be constructed or executed safely."""


@dataclass(frozen=True)
class WorktreeDiff:
    """Changed paths in the working tree relative to its branch point."""

    base_ref: str
    merge_base: str
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class PytestInvocation:
    """One local pytest process for an affected workspace package."""

    label: str
    package: str
    python: str
    extras: tuple[str, ...]
    pytest_args: tuple[str, ...]
    test_paths: tuple[str, ...]
    source_build: bool


def _run_git(args: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_output(args: list[str], repo_root: Path) -> str:
    result = _run_git(args, repo_root)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LocalTestError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def default_base_ref(repo_root: Path) -> str:
    """Return the first available conventional main-branch ref."""
    for ref in DEFAULT_BASE_REFS:
        result = _run_git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], repo_root)
        if result.returncode == 0:
            return ref
    refs = ", ".join(DEFAULT_BASE_REFS)
    raise LocalTestError(f"no default base ref found ({refs}); pass --base-ref explicitly")


def worktree_diff(base_ref: str, repo_root: Path) -> WorktreeDiff:
    """Find committed and local changes since ``HEAD`` diverged from ``base_ref``."""
    merge_base = _git_output(["merge-base", "HEAD", base_ref], repo_root)
    tracked = _git_output(["diff", "--name-only", "--no-renames", merge_base, "--"], repo_root).splitlines()
    untracked = _git_output(["ls-files", "--others", "--exclude-standard"], repo_root).splitlines()
    changed_files = tuple(dict.fromkeys(path for path in [*tracked, *untracked] if path))
    return WorktreeDiff(base_ref=base_ref, merge_base=merge_base, changed_files=changed_files)


def local_invocations(selection: SelectionResult) -> tuple[PytestInvocation, ...]:
    """Collapse CI shards into one local invocation per selected package."""
    grouped_paths: dict[str, list[str]] = {}
    metadata: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...], bool]] = {}

    for leg in selection.matrix:
        package = leg.package
        label = leg.label.split(maxsplit=1)[0]
        python = leg.python
        extras = tuple(shlex.split(leg.extras))
        pytest_args = tuple(shlex.split(leg.pytest_args))
        source_build = leg.setup == RUST_SETUP_TAG
        current = (label, python, extras, pytest_args, source_build)
        if package in metadata and metadata[package] != current:
            raise LocalTestError(f"inconsistent matrix metadata for {package}")
        metadata[package] = current
        grouped_paths.setdefault(package, []).extend(shlex.split(leg.test_paths))

    invocations = []
    for package, paths in grouped_paths.items():
        label, python, extras, pytest_args, source_build = metadata[package]
        invocations.append(
            PytestInvocation(
                label=label,
                package=package,
                python=python,
                extras=extras,
                pytest_args=pytest_args,
                test_paths=tuple(dict.fromkeys(paths)),
                source_build=source_build,
            )
        )
    return tuple(invocations)


def pytest_command(invocation: PytestInvocation, extra_args: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Build the local equivalent of one unified-unit pytest command."""
    return (
        "uv",
        "run",
        "--isolated",
        "--python",
        invocation.python,
        "--package",
        invocation.package,
        *invocation.extras,
        "--group",
        "test",
        "pytest",
        *invocation.pytest_args,
        *invocation.test_paths,
        *extra_args,
    )


def native_sources_enabled(repo_root: Path) -> bool:
    """Whether every native package is configured to build from this checkout."""
    result = subprocess.run(
        [sys.executable, "scripts/rust_mode.py", "status"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result.returncode == 0
        and "Rust build mode: dev" in result.stdout
        and "WARNING: mixed state" not in result.stdout
    )


def _print_plan(
    diff: WorktreeDiff,
    selection: SelectionResult,
    invocations: tuple[PytestInvocation, ...],
    extra_args: tuple[str, ...],
    *,
    show_commands: bool,
) -> None:
    print(
        f"Selected {sum(len(item.test_paths) for item in invocations)} test targets "
        f"in {len(invocations)} packages from {len(diff.changed_files)} changed files "
        f"({selection.reason}, base {diff.base_ref} at {diff.merge_base[:12]})."
    )
    for invocation in invocations:
        print(f"  {invocation.label}: {len(invocation.test_paths)} targets")
        if show_commands:
            print(f"    {shlex.join(pytest_command(invocation, extra_args))}")
    if selection.suites:
        print(f"Dedicated CI suites not run locally: {', '.join(selection.suites)}")


def run(argv: list[str] | None = None) -> int:
    """Run the selected local tests and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", help="branch or commit to diff against (default: origin's main branch)")
    parser.add_argument("--all", action="store_true", help="run every safe unit-test scope")
    parser.add_argument("--dry-run", action="store_true", help="print selected commands without running them")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra pytest arguments after --")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    try:
        base_ref = args.base_ref or default_base_ref(repo_root)
        diff = worktree_diff(base_ref, repo_root)
        selection = select_changed_tests(list(diff.changed_files), repo_root, run_all_tests=args.all)
        invocations = local_invocations(selection)
    except LocalTestError as error:
        parser.error(str(error))

    extra_args = tuple(args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args)
    _print_plan(diff, selection, invocations, extra_args, show_commands=args.dry_run)
    if not invocations or args.dry_run:
        return 0

    native_invocations = [item.label for item in invocations if item.source_build]
    if native_invocations and not native_sources_enabled(repo_root):
        labels = ", ".join(native_invocations)
        print(
            f"Native source changes selected ({labels}). Run `python3 scripts/rust_mode.py dev` "
            "and rerun so pytest exercises the local Rust code.",
            file=sys.stderr,
        )
        return 2

    failures: list[tuple[str, int]] = []
    for invocation in invocations:
        command = pytest_command(invocation, extra_args)
        print(f"\n==> {invocation.label} ({len(invocation.test_paths)} targets)", flush=True)
        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode == 5:
            print(f"{invocation.label}: no tests collected after the safe marker filter")
        elif result.returncode != 0:
            failures.append((invocation.label, result.returncode))

    if failures:
        summary = ", ".join(f"{label} (exit {status})" for label, status in failures)
        print(f"Failed test packages: {summary}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
