# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Run the safe unit tests affected by a local branch and working tree.

The default comparison includes committed, staged, unstaged, and untracked
changes from the branch point with main. Selected paths share one synced
workspace environment. When Haliax and other packages are selected together,
the runner gives Haliax one eight-device worker and runs the remaining workers
concurrently with the normal one-device topology. Dedicated accelerator and
browser suites remain delegated to CI.

Usage:
    uv run --no-project infra/ci/run_tests.py
    uv run --no-project infra/ci/run_tests.py --dry-run
    uv run --no-project infra/ci/run_tests.py --base-ref upstream/main -- -x
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from infra.ci.select_tests import BROAD_TRIGGER_REASON, RUST_SETUP_TAG, SelectionResult, select_local_tests

DEFAULT_BASE_REFS: tuple[str, ...] = ("origin/HEAD", "origin/main", "main")
LOCAL_SAFE_MARKERS = (
    "not slow and not integration and not data_integration and not cluster "
    "and not requires_cluster and not docker and not manual and not torch"
)
MAX_DISPLAYED_TEST_PATHS = 5
HALIAX_CPU_DEVICE_COUNT = 8
DEFAULT_WORKERS = max(2, os.cpu_count() or 2)
JAX_CPU_DEVICE_ENV = "JAX_NUM_CPU_DEVICES"


class LocalTestError(RuntimeError):
    """A local test plan cannot be constructed or executed safely."""


@dataclass(frozen=True)
class WorktreeDiff:
    """Changed paths in the working tree relative to its branch point."""

    base_ref: str
    merge_base: str
    changed_files: tuple[str, ...]


@dataclass(frozen=True)
class PackageSelection:
    """Selected pytest paths and native-build state for one workspace package."""

    label: str
    test_paths: tuple[str, ...]
    source_build: bool


@dataclass(frozen=True)
class PytestInvocation:
    """Affected workspace packages sharing one synced test environment."""

    python: str
    extras: tuple[str, ...]
    pytest_args: tuple[str, ...]
    packages: tuple[PackageSelection, ...]

    @property
    def test_paths(self) -> tuple[str, ...]:
        """Selected pytest paths across all packages, deduplicated in matrix order."""
        return tuple(dict.fromkeys(path for package in self.packages for path in package.test_paths))


@dataclass(frozen=True)
class PytestLane:
    """One concurrent pytest process with a uniform JAX device topology."""

    label: str
    invocation: PytestInvocation
    workers: int
    jax_cpu_devices: int


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


def _extra_names(extras: str) -> tuple[str, ...]:
    tokens = shlex.split(extras)
    if len(tokens) % 2 != 0 or any(flag != "--extra" for flag in tokens[::2]):
        raise LocalTestError(f"unsupported matrix extras: {extras}")
    return tuple(tokens[1::2])


def local_invocation(selection: SelectionResult) -> PytestInvocation | None:
    """Collapse CI shards into one workspace-wide local pytest invocation."""
    grouped_paths: dict[str, list[str]] = {}
    labels: dict[str, str] = {}
    source_builds: dict[str, bool] = {}
    python_versions: set[str] = set()
    pytest_arg_sets: set[tuple[str, ...]] = set()
    extras: list[str] = []

    for leg in selection.matrix:
        package = leg.package
        label = leg.label.split(maxsplit=1)[0]
        source_build = leg.setup == RUST_SETUP_TAG
        if package in labels and (labels[package], source_builds[package]) != (label, source_build):
            raise LocalTestError(f"inconsistent matrix metadata for {package}")
        labels[package] = label
        source_builds[package] = source_build
        grouped_paths.setdefault(package, []).extend(shlex.split(leg.test_paths))
        python_versions.add(leg.python)
        pytest_arg_sets.add(tuple(shlex.split(leg.pytest_args)))
        extras.extend(_extra_names(leg.extras))

    if not grouped_paths:
        return None
    if len(python_versions) != 1 or len(pytest_arg_sets) != 1:
        raise LocalTestError("inconsistent matrix metadata across workspace packages")

    packages = []
    for package, paths in grouped_paths.items():
        packages.append(
            PackageSelection(
                label=labels[package],
                test_paths=tuple(dict.fromkeys(paths)),
                source_build=source_builds[package],
            )
        )
    return PytestInvocation(
        python=python_versions.pop(),
        extras=tuple(dict.fromkeys(extras)),
        pytest_args=pytest_arg_sets.pop(),
        packages=tuple(packages),
    )


def pytest_lanes(invocation: PytestInvocation, workers: int) -> tuple[PytestLane, ...]:
    """Partition Haliax from ordinary tests while preserving one worker budget."""
    haliax = tuple(package for package in invocation.packages if package.label == "haliax")
    ordinary = tuple(package for package in invocation.packages if package.label != "haliax")
    if not haliax:
        return (PytestLane("workspace", invocation, workers, 1),)
    if not ordinary:
        return (PytestLane("haliax", invocation, workers, HALIAX_CPU_DEVICE_COUNT),)
    return (
        PytestLane("haliax", replace(invocation, packages=haliax), 1, HALIAX_CPU_DEVICE_COUNT),
        PytestLane("workspace", replace(invocation, packages=ordinary), workers - 1, 1),
    )


def uv_sync_command(invocation: PytestInvocation) -> tuple[str, ...]:
    """Build the shared workspace test-environment sync command."""
    return (
        "uv",
        "sync",
        "--python",
        invocation.python,
        "--all-packages",
        "--no-default-groups",
        *(value for extra in invocation.extras for value in ("--extra", extra)),
        "--group",
        "test",
    )


def pytest_command(
    lane: PytestLane,
    extra_args: tuple[str, ...] = (),
    *,
    no_sync: bool = False,
) -> tuple[str, ...]:
    """Build one root pytest command for a uniform-topology test lane."""
    invocation = lane.invocation
    return (
        "uv",
        "run",
        *(("--no-sync",) if no_sync else ()),
        "--python",
        invocation.python,
        "--all-packages",
        "--no-default-groups",
        *(value for extra in invocation.extras for value in ("--extra", extra)),
        "--group",
        "test",
        "pytest",
        *invocation.pytest_args,
        "--maxprocesses",
        str(lane.workers),
        "-m",
        LOCAL_SAFE_MARKERS,
        *invocation.test_paths,
        *extra_args,
    )


def pytest_environment(lane: PytestLane) -> dict[str, str]:
    """Build the process environment for one JAX device topology."""
    environment = os.environ.copy()
    if lane.jax_cpu_devices == 1:
        environment.pop(JAX_CPU_DEVICE_ENV, None)
    else:
        environment[JAX_CPU_DEVICE_ENV] = str(lane.jax_cpu_devices)
    return environment


def top_level_pytest_command(extra_args: tuple[str, ...] = ()) -> tuple[str, ...]:
    return ("uv", "run", "pytest", *extra_args)


def pytest_command_preview(
    lane: PytestLane,
    extra_args: tuple[str, ...] = (),
    *,
    no_sync: bool = False,
) -> str:
    """Render the command without flooding the terminal for a large selection."""
    invocation = lane.invocation
    command = pytest_command(lane, extra_args, no_sync=no_sync)
    environment_prefix = (
        f"env -u {JAX_CPU_DEVICE_ENV} " if lane.jax_cpu_devices == 1 else f"{JAX_CPU_DEVICE_ENV}={lane.jax_cpu_devices} "
    )
    if len(invocation.test_paths) <= MAX_DISPLAYED_TEST_PATHS:
        return f"{environment_prefix}{shlex.join(command)}"

    hidden = len(invocation.test_paths) - MAX_DISPLAYED_TEST_PATHS
    suffix_size = len(invocation.test_paths) + len(extra_args)
    command_prefix = command[:-suffix_size]
    visible_paths = invocation.test_paths[:MAX_DISPLAYED_TEST_PATHS]
    preview = shlex.join((*command_prefix, *visible_paths))
    if extra_args:
        return f"{environment_prefix}{preview} ... [{hidden} more test paths] {shlex.join(extra_args)}"
    return f"{environment_prefix}{preview} ... [{hidden} more test paths]"


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
    invocation: PytestInvocation | None,
    extra_args: tuple[str, ...],
    repo_root: Path,
    workers: int,
    *,
    show_commands: bool,
) -> None:
    packages = invocation.packages if invocation is not None else ()
    if selection.reason == BROAD_TRIGGER_REASON:
        print(
            f"The {len(diff.changed_files)} changed files include shared test-environment configuration "
            f"that cannot be narrowed (base {diff.base_ref} at {diff.merge_base[:12]})."
        )
        if show_commands:
            print(f"  {shlex.join(top_level_pytest_command(extra_args))}")
        if selection.suites:
            print(f"Dedicated CI suites not run locally: {', '.join(selection.suites)}")
        return

    print(
        f"Selected tests in {len(packages)} packages from {len(diff.changed_files)} changed files "
        f"({selection.reason}, base {diff.base_ref} at {diff.merge_base[:12]})."
    )
    for package in packages:
        directories = [path for path in package.test_paths if (repo_root / path).is_dir()]
        if directories:
            detail = f"full suite ({', '.join(directories)})"
        elif len(package.test_paths) <= 5:
            detail = ", ".join(package.test_paths)
        else:
            detail = f"{len(package.test_paths)} selected test files"
        print(f"  {package.label}: {detail}")
    if show_commands and invocation is not None:
        lanes = pytest_lanes(invocation, workers)
        if len(lanes) > 1:
            print(f"  {shlex.join(uv_sync_command(invocation))}")
        for lane in lanes:
            label = f"{lane.label}: " if len(lanes) > 1 else ""
            print(f"  {label}{pytest_command_preview(lane, extra_args, no_sync=len(lanes) > 1)}")
    if selection.suites:
        print(f"Dedicated CI suites not run locally: {', '.join(selection.suites)}")


def run(argv: list[str] | None = None) -> int:
    """Run the selected local tests and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", help="branch or commit to diff against (default: origin's main branch)")
    parser.add_argument("--dry-run", action="store_true", help="print selected commands without running them")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"total xdist worker budget (default: host CPU count, currently {DEFAULT_WORKERS})",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra pytest arguments after --")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    try:
        base_ref = args.base_ref or default_base_ref(repo_root)
        diff = worktree_diff(base_ref, repo_root)
        selection = select_local_tests(list(diff.changed_files), repo_root)
        invocation = local_invocation(selection)
    except LocalTestError as error:
        parser.error(str(error))

    if args.workers < 2:
        parser.error("--workers must be at least 2")

    extra_args = tuple(args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args)
    _print_plan(diff, selection, invocation, extra_args, repo_root, args.workers, show_commands=True)
    if invocation is None or args.dry_run:
        return 0

    if selection.reason == BROAD_TRIGGER_REASON:
        print("\n==> pytest (shared test environment changed)", flush=True)
        return subprocess.run(top_level_pytest_command(extra_args), cwd=repo_root, check=False).returncode

    native_packages = [package.label for package in invocation.packages if package.source_build]
    if native_packages and not native_sources_enabled(repo_root):
        labels = ", ".join(native_packages)
        print(
            f"Native source changes selected ({labels}). Run `python3 scripts/rust_mode.py dev` "
            "and rerun so pytest exercises the local Rust code.",
            file=sys.stderr,
        )
        return 2

    lanes = pytest_lanes(invocation, args.workers)
    if len(lanes) == 1:
        lane = lanes[0]
        labels = ", ".join(package.label for package in lane.invocation.packages)
        print(f"\n==> pytest ({labels}; {len(lane.invocation.test_paths)} selected paths)", flush=True)
        result = subprocess.run(
            pytest_command(lane, extra_args),
            cwd=repo_root,
            env=pytest_environment(lane),
            check=False,
        )
        return 0 if result.returncode == 5 else result.returncode

    print("\n==> syncing the shared test environment", flush=True)
    sync = subprocess.run(uv_sync_command(invocation), cwd=repo_root, check=False)
    if sync.returncode != 0:
        return sync.returncode

    print(f"\n==> pytest ({args.workers} workers across {len(lanes)} concurrent lanes)", flush=True)
    processes = [
        (
            lane,
            subprocess.Popen(
                pytest_command(lane, extra_args, no_sync=True),
                cwd=repo_root,
                env=pytest_environment(lane),
            ),
        )
        for lane in lanes
    ]
    failures = [(lane.label, status) for lane, process in processes if (status := process.wait()) not in (0, 5)]
    if failures:
        print(
            "Failed test lanes: " + ", ".join(f"{label} (exit {status})" for label, status in failures),
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
