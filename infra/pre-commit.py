#!/usr/bin/env -S uv run --script
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "pyyaml",
# ]
# ///
"""
Run pre-commits and lint checks for Marin.

This handles ruff/black/pyrefly type checking and license headers management for
the Marin mono-repo. Slightly different testing styles and license headers are
applied to different parts of the repo (e.g., levanter library code vs.
Marin application code).
"""

import ast
import fnmatch
import io
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import click
import tomllib
import yaml

ROOT_DIR = pathlib.Path(__file__).parent.parent
LEVANTER_LICENSE = ROOT_DIR / "lib/levanter/etc/license_header.txt"
HALIAX_LICENSE = ROOT_DIR / "lib/haliax/etc/license_header.txt"
MARIN_LICENSE = ROOT_DIR / "etc/license_header.txt"
LEVANTER_BLACK_CONFIG = ROOT_DIR / "lib/levanter/pyproject.toml"
HALIAX_BLACK_CONFIG = ROOT_DIR / "lib/haliax/pyproject.toml"

EXCLUDE_PATTERNS = [
    ".git/**",
    ".github/**",
    "tests/snapshots/**",
    # grpc generated files
    "**/*_connect.py",
    "**/*_pb2.py",
    "**/*.gz",
    "**/*.pb",
    "**/*.index",
    "**/*.ico",
    "**/*.npy",
    "**/*.lock",
    "**/*.png",
    "**/*.jpg",
    "**/*.html",
    "**/*.jpeg",
    "**/*.gif",
    "**/*.mov",
    "**/*.mp4",
    "**/*.data-*",
    "**/package-lock.json",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*-template.yaml",
]


@dataclass
class CheckResult:
    name: str
    exit_code: int
    output: str


# Collects all failure output to print at the end.
_check_results: list[CheckResult] = []


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT_DIR, capture_output=True, text=True, check=check)


def get_all_files(all_files: bool, file_args: list[str]) -> list[pathlib.Path]:
    """Get list of files to check, excluding deleted files."""
    if file_args:
        files = []
        for f in file_args:
            path = ROOT_DIR / f
            if not path.exists():
                click.echo(f"Warning: Skipping non-existent file: {f}")
                continue
            files.append(path)
        return files
    if all_files:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        # Use ACM filter to only get Added, Copied, Modified files (not Deleted)
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            check=True,
        )

    files = [ROOT_DIR / f for f in result.stdout.strip().split("\n") if f]
    # Filter to only include files that exist on disk
    return [f for f in files if f.exists()]


def matches_pattern(file_path: pathlib.Path, patterns: list[str]) -> bool:
    relative_path = str(file_path.relative_to(ROOT_DIR))
    for pattern in patterns:
        if fnmatch.fnmatch(relative_path, pattern):
            return True
    return False


def should_exclude(file_path: pathlib.Path) -> bool:
    return matches_pattern(file_path, EXCLUDE_PATTERNS)


def get_matching_files(
    patterns: list[str], all_files_list: list[pathlib.Path], exclude_patterns: list[str]
) -> list[pathlib.Path]:
    matched = []
    for file_path in all_files_list:
        if should_exclude(file_path):
            continue
        if matches_pattern(file_path, exclude_patterns):
            continue
        if matches_pattern(file_path, patterns):
            matched.append(file_path)
    return matched


def _record(name: str, exit_code: int, output: str = "") -> int:
    """Print a one-line status and stash failure details for the summary."""
    status = "ok" if exit_code == 0 else "FAIL"
    click.echo(f"  {name:.<40s} {status}")
    _check_results.append(CheckResult(name=name, exit_code=exit_code, output=output.rstrip()))
    return exit_code


def check_ruff(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    args = ["uvx", "ruff@0.14.3", "check"]
    if fix:
        args.extend(["--fix", "--exit-non-zero-on-fix"])

    file_args = [str(f.relative_to(ROOT_DIR)) for f in files]
    args.extend(file_args)

    result = run_cmd(args)
    output = (result.stdout + result.stderr).strip()
    return _record("Ruff linter", result.returncode, output)


def check_black(files: list[pathlib.Path], fix: bool, config: pathlib.Path | None = None) -> int:
    if not files:
        return 0

    args = ["uvx", "black@25.9.0", "--check"]
    if fix:
        args.append("--diff")
    if config:
        args.extend(["--config", str(config)])

    file_args = [str(f.relative_to(ROOT_DIR)) for f in files]
    args.extend(file_args)

    result = run_cmd(args)
    output = (result.stdout + result.stderr).strip()

    # If check failed and fix is requested, actually format them
    if result.returncode != 0 and fix:
        format_args = ["uvx", "black@25.9.0"]
        if config:
            format_args.extend(["--config", str(config)])
        format_args.extend(file_args)
        run_cmd(format_args)

    label = "Black formatter"
    if config:
        label += f" ({config.relative_to(ROOT_DIR)})"
    return _record(label, result.returncode, output)


def check_license_headers(files: list[pathlib.Path], fix: bool, license_file: pathlib.Path) -> int:
    if not files:
        return 0

    label = f"License headers ({license_file.relative_to(ROOT_DIR)})"

    if not license_file.exists():
        return _record(label, 0, f"Warning: License header file not found: {license_file}")

    with open(license_file) as f:
        license_template = f.read().strip()

    license_lines = [f"# {line}" if line else "#" for line in license_template.split("\n")]
    expected_header = "\n".join(license_lines) + "\n"

    files_without_header = []
    buf = io.StringIO()

    for file_path in files:
        with open(file_path) as f:
            content = f.read()

        lines = content.split("\n")

        comment_lines = []
        start_idx = 1 if content.startswith("#!") else 0

        for line in lines[start_idx:]:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                if len(stripped) > 1 and stripped[1] == " ":
                    comment_text = stripped[2:]
                else:
                    comment_text = stripped[1:]
                comment_lines.append(comment_text)
            elif stripped:
                break

        comment_block = "\n".join(comment_lines)
        if license_template not in comment_block:
            files_without_header.append(file_path)

            if fix:
                has_shebang = content.startswith("#!")
                if has_shebang:
                    shebang_line = lines[0]
                    rest_content = "\n".join(lines[1:])
                    new_content = f"{shebang_line}\n{expected_header}\n{rest_content}"
                else:
                    new_content = f"{expected_header}\n{content}"

                with open(file_path, "w") as f:
                    f.write(new_content)

    if files_without_header:
        buf.write(f"{len(files_without_header)} files missing license headers\n")
        for f in files_without_header:
            buf.write(f"  - {f.relative_to(ROOT_DIR)}\n")
        return _record(label, 1, buf.getvalue())

    return _record(label, 0)


def check_mypy(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    args = ["uvx", "mypy@1.19.1", "--ignore-missing-imports", "--python-version=3.11"]

    test_excluded = [f for f in files if not str(f.relative_to(ROOT_DIR)).startswith("tests/")]
    if not test_excluded:
        return _record("Mypy type checker", 0)

    file_args = [str(f.relative_to(ROOT_DIR)) for f in test_excluded]
    args.extend(file_args)

    result = run_cmd(args)
    output = (result.stdout + result.stderr).strip()
    return _record("Mypy type checker", result.returncode, output)


def check_large_files(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    max_size = 500 * 1024
    buf = io.StringIO()

    large_files = []
    for file_path in files:
        if file_path.stat().st_size > max_size:
            large_files.append((file_path, file_path.stat().st_size))

    if large_files:
        for path, size in large_files:
            buf.write(f"  - {path.relative_to(ROOT_DIR)} ({size / 1024:.1f} KB)\n")
        return _record("Large files", 1, buf.getvalue())

    return _record("Large files", 0)


def check_python_ast(files: list[pathlib.Path], fix: bool) -> int:
    py_files = [f for f in files if f.suffix == ".py"]
    if not py_files:
        return 0

    buf = io.StringIO()
    invalid_files = []

    for file_path in py_files:
        try:
            with open(file_path) as f:
                ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            invalid_files.append((file_path, str(e)))

    if invalid_files:
        for path, error in invalid_files:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}: {error}\n")
        return _record("Python AST", 1, buf.getvalue())

    return _record("Python AST", 0)


def check_merge_conflicts(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    conflict_markers = [b"<<<<<<<", b">>>>>>>"]
    files_with_conflicts = []
    buf = io.StringIO()

    for file_path in files:
        if "pre-commit" in str(file_path):
            continue
        try:
            with open(file_path, "rb") as f:
                content = f.read()
                if any(marker in content for marker in conflict_markers):
                    files_with_conflicts.append(file_path)
        except Exception:
            continue

    if files_with_conflicts:
        for path in files_with_conflicts:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}\n")
        return _record("Merge conflicts", 1, buf.getvalue())

    return _record("Merge conflicts", 0)


def check_toml_yaml(files: list[pathlib.Path], fix: bool) -> int:
    config_files = [f for f in files if f.suffix in [".toml", ".yaml", ".yml"]]
    if not config_files:
        return 0

    errors = []
    buf = io.StringIO()

    # levanter is weird
    def include_constructor(loader, node):
        filepath = loader.construct_scalar(node)
        base_dir = os.path.dirname(loader.name) if hasattr(loader, "name") else "."
        full_path = os.path.join(base_dir, filepath)

        with open(full_path, "r") as f:
            return yaml.safe_load(f)

    yaml.add_constructor("!include", include_constructor, Loader=yaml.SafeLoader)

    for file_path in config_files:
        if file_path.suffix == ".toml":
            try:
                with open(file_path, "rb") as f:
                    tomllib.load(f)
            except Exception as e:
                errors.append((file_path, str(e)))

        elif file_path.suffix in [".yaml", ".yml"]:
            try:
                with open(file_path) as f:
                    yaml.safe_load(f)
            except Exception as e:
                errors.append((file_path, str(e)))

    if errors:
        for path, error in errors:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}: {error}\n")
        return _record("TOML and YAML", 1, buf.getvalue())

    return _record("TOML and YAML", 0)


def check_trailing_whitespace(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    files_with_whitespace = []
    buf = io.StringIO()

    for file_path in files:
        try:
            with open(file_path) as f:
                lines = f.readlines()
        except Exception:
            continue

        has_trailing = any(line.rstrip("\n").endswith((" ", "\t")) for line in lines)

        if has_trailing:
            files_with_whitespace.append(file_path)

            if fix:
                file_ended_with_newline = lines[-1].endswith("\n") if lines else True

                with open(file_path, "w") as f:
                    for i, line in enumerate(lines):
                        is_last_line = i == len(lines) - 1
                        cleaned = line.rstrip()

                        if is_last_line and not file_ended_with_newline:
                            f.write(cleaned)
                        else:
                            f.write(cleaned + "\n")

    if files_with_whitespace:
        buf.write(f"{len(files_with_whitespace)} files with trailing whitespace\n")
        for f in files_with_whitespace:
            buf.write(f"  - {f.relative_to(ROOT_DIR)}\n")
        return _record("Trailing whitespace", 1, buf.getvalue())

    return _record("Trailing whitespace", 0)


def check_eof_newline(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    files_missing_newline = []
    buf = io.StringIO()

    for file_path in files:
        if file_path.stat().st_size == 0:
            continue

        try:
            with open(file_path, "rb") as f:
                content = f.read()
                if content and not content.endswith(b"\n"):
                    files_missing_newline.append(file_path)

                    if fix:
                        with open(file_path, "ab") as f:
                            f.write(b"\n")
        except Exception:
            pass

    if files_missing_newline:
        buf.write(f"{len(files_missing_newline)} files missing newline\n")
        for f in files_missing_newline:
            buf.write(f"  - {f.relative_to(ROOT_DIR)}\n")
        return _record("End-of-file newline", 1, buf.getvalue())

    return _record("End-of-file newline", 0)


def check_notebooks(files: list[pathlib.Path], fix: bool) -> int:
    """Check that Jupyter notebooks have cleared outputs and normalized formatting.

    TODO: Consider generating static HTML versions of notebooks (via nbconvert) and uploading
    to GCS, then recording the GCS path in the cleared notebook. This would preserve a trace
    of what the author saw at commit time while still keeping git diffs clean.
    """
    nb_files = [f for f in files if f.suffix == ".ipynb"]
    if not nb_files:
        return 0

    notebooks_needing_clean = []
    buf = io.StringIO()

    for nb_path in nb_files:
        try:
            with open(nb_path) as f:
                notebook = json.load(f)
        except Exception as e:
            buf.write(f"Error reading {nb_path.relative_to(ROOT_DIR)}: {e}\n")
            continue

        needs_cleaning = False

        if "cells" in notebook:
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    if cell.get("outputs") or cell.get("execution_count") is not None:
                        needs_cleaning = True
                        break

        if needs_cleaning:
            notebooks_needing_clean.append(nb_path)

            if fix:
                for cell in notebook.get("cells", []):
                    if cell.get("cell_type") == "code":
                        cell["outputs"] = []
                        cell["execution_count"] = None

                with open(nb_path, "w") as f:
                    json.dump(notebook, f, indent=1, ensure_ascii=False, sort_keys=True)
                    f.write("\n")

    if notebooks_needing_clean:
        buf.write(f"{len(notebooks_needing_clean)} notebooks with outputs or execution counts\n")
        for f in notebooks_needing_clean:
            buf.write(f"  - {f.relative_to(ROOT_DIR)}\n")
        return _record("Jupyter notebooks", 1, buf.getvalue())

    return _record("Jupyter notebooks", 0)


def check_pyrefly(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    args = ["uvx", "pyrefly@0.40.0", "check", "--baseline", ".pyrefly-baseline.json"]
    result = run_cmd(args)
    output = (result.stdout + result.stderr).strip()
    return _record("Pyrefly type checker", result.returncode, output)


@dataclass
class PrecommitConfig:
    patterns: list[str]
    checks: list[Callable[[list[pathlib.Path], bool], int]]
    exclude_patterns: list[str] = field(default_factory=list)


PRECOMMIT_CONFIGS = [
    PrecommitConfig(
        patterns=["lib/levanter/**/*.py"],
        checks=[
            check_ruff,
            lambda files, fix: check_black(files, fix, config=LEVANTER_BLACK_CONFIG),
            lambda files, fix: check_license_headers(files, fix, LEVANTER_LICENSE),
            # check_mypy,
        ],
    ),
    PrecommitConfig(
        patterns=["lib/haliax/**/*.py"],
        checks=[
            check_ruff,
            lambda files, fix: check_black(files, fix, config=HALIAX_BLACK_CONFIG),
            lambda files, fix: check_license_headers(files, fix, HALIAX_LICENSE),
        ],
    ),
    PrecommitConfig(
        patterns=["**/*.py"],
        exclude_patterns=["lib/levanter/**", "lib/haliax/**", "lib/**/vendor/**"],
        checks=[
            check_ruff,
            check_black,
            lambda files, fix: check_license_headers(files, fix, MARIN_LICENSE),
        ],
    ),
    PrecommitConfig(
        patterns=[
            "lib/marin/src/**/*.py",
            "lib/levanter/src/**/*.py",
            "lib/haliax/src/**/*.py",
            "lib/fray/src/**/*.py",
            "lib/iris/src/**/*.py",
            "lib/zephyr/src/**/*.py",
        ],
        checks=[
            check_pyrefly,
        ],
    ),
    PrecommitConfig(
        patterns=["**/*"],
        checks=[
            check_large_files,
            check_python_ast,
            check_merge_conflicts,
            check_toml_yaml,
            check_trailing_whitespace,
            check_eof_newline,
        ],
    ),
    PrecommitConfig(
        patterns=["**/*.ipynb"],
        checks=[
            check_notebooks,
        ],
    ),
]


@click.command()
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
@click.option("--all-files", is_flag=True, help="Run checks on all files, not just staged")
@click.argument("files", nargs=-1)
def main(fix: bool, all_files: bool, files: tuple[str, ...]):
    all_files_list = get_all_files(all_files, list(files))
    exit_codes = []

    for config in PRECOMMIT_CONFIGS:
        matched_files = get_matching_files(config.patterns, all_files_list, config.exclude_patterns)
        matched_files = [f for f in matched_files if f.exists()]
        if not matched_files:
            continue

        for check in config.checks:
            try:
                exit_code = check(matched_files, fix)
                exit_codes.append(exit_code)
            except Exception as e:
                click.echo(f"  Error running check {check.__name__}: {e}")
                exit_codes.append(1)

    # Print failure details at the end
    failures = [r for r in _check_results if r.exit_code != 0 and r.output]
    if failures:
        click.echo(f"\n{'=' * 60}")
        click.echo("Failure details:\n")
        for r in failures:
            click.echo(f"--- {r.name} ---")
            click.echo(r.output)
            click.echo()

    click.echo("=" * 60)
    if any(exit_codes):
        click.echo("FAILED")
        click.echo("=" * 60)
        sys.exit(1)
    else:
        click.echo("OK")
        click.echo("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
