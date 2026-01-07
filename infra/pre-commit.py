#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run pre-commits and lint checks for Marin.

This handles ruff/black/pyrefly type checking and license headers management for
the Marin mono-repo. Slightly different testing styles and license headers are
applied to different parts of the repo (e.g., levanter library code vs.
Marin application code).
"""

import ast
import fnmatch
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


def run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    click.echo(f"  $ {' '.join(cmd)[:200]}")
    return subprocess.run(cmd, cwd=ROOT_DIR, check=check)


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


def check_ruff(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nRuff linter:")
    args = ["uv", "run", "--all-packages", "ruff", "check"]
    if fix:
        args.extend(["--fix", "--exit-non-zero-on-fix"])

    file_args = [str(f.relative_to(ROOT_DIR)) for f in files]
    args.extend(file_args)

    return run_cmd(args).returncode


def check_black(files: list[pathlib.Path], fix: bool, config: pathlib.Path | None = None) -> int:
    if not files:
        return 0

    click.echo("\nBlack formatter:")
    args = ["uv", "run", "--all-packages", "black", "--check"]
    if fix:
        # When fixing, use --diff to show changes but still exit non-zero if files would be formatted
        args.append("--diff")
    if config:
        args.extend(["--config", str(config)])

    file_args = [str(f.relative_to(ROOT_DIR)) for f in files]
    args.extend(file_args)

    result = run_cmd(args)

    # If check failed (files need formatting) and fix is requested, format them
    if result.returncode != 0 and fix:
        format_args = ["uv", "run", "--all-packages", "black"]
        if config:
            format_args.extend(["--config", str(config)])
        format_args.extend(file_args)
        run_cmd(format_args)

    return result.returncode


def check_license_headers(files: list[pathlib.Path], fix: bool, license_file: pathlib.Path) -> int:
    if not files:
        return 0

    click.echo(f"\nLicense headers ({license_file.relative_to(ROOT_DIR)}):")

    if not license_file.exists():
        click.echo(f"  Warning: License header file not found: {license_file}")
        return 0

    with open(license_file) as f:
        license_template = f.read().strip()

    license_lines = [f"# {line}" if line else "#" for line in license_template.split("\n")]
    expected_header = "\n".join(license_lines) + "\n"

    files_without_header = []

    for file_path in files:
        with open(file_path) as f:
            content = f.read()

        lines = content.split("\n")

        # Scan forward until we find the first non-comment line
        comment_lines = []
        start_idx = 1 if content.startswith("#!") else 0  # Skip shebang

        for line in lines[start_idx:]:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                # Strip comment marker and the single space after it if present
                if len(stripped) > 1 and stripped[1] == " ":
                    comment_text = stripped[2:]
                else:
                    comment_text = stripped[1:]
                comment_lines.append(comment_text)
            elif stripped:
                # Found first non-comment line
                break

        # Check if license text appears in the comments
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
        if not fix:
            click.echo(f"  {len(files_without_header)} files missing license headers")
            for f in files_without_header:
                click.echo(f"    - {f.relative_to(ROOT_DIR)}")
        return 1

    click.echo("  All files have license headers")
    return 0


def check_mypy(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nMypy type checker:")
    args = ["uv", "run", "--all-packages", "mypy", "--ignore-missing-imports", "--python-version=3.11"]

    test_excluded = [f for f in files if not str(f.relative_to(ROOT_DIR)).startswith("tests/")]
    if not test_excluded:
        click.echo("  No files to check (all are tests)")
        return 0

    file_args = [str(f.relative_to(ROOT_DIR)) for f in test_excluded]
    args.extend(file_args)

    return run_cmd(args).returncode


def check_large_files(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nLarge files:")
    max_size = 500 * 1024

    large_files = []
    for file_path in files:
        if file_path.stat().st_size > max_size:
            large_files.append((file_path, file_path.stat().st_size))

    if large_files:
        click.echo("  Large files detected:")
        for path, size in large_files:
            click.echo(f"    - {path.relative_to(ROOT_DIR)} ({size / 1024:.1f} KB)")
        return 1

    click.echo("  No large files")
    return 0


def check_python_ast(files: list[pathlib.Path], fix: bool) -> int:
    py_files = [f for f in files if f.suffix == ".py"]
    if not py_files:
        return 0

    click.echo("\nPython AST:")
    invalid_files = []

    for file_path in py_files:
        try:
            with open(file_path) as f:
                ast.parse(f.read(), filename=str(file_path))
        except SyntaxError as e:
            invalid_files.append((file_path, str(e)))

    if invalid_files:
        click.echo("  Invalid Python syntax:")
        for path, error in invalid_files:
            click.echo(f"    - {path.relative_to(ROOT_DIR)}: {error}")
        return 1

    click.echo("  All Python files have valid syntax")
    return 0


def check_merge_conflicts(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nMerge conflicts:")
    conflict_markers = [b"<<<<<<<", b">>>>>>>"]
    files_with_conflicts = []

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
        click.echo("  Merge conflict markers found:")
        for path in files_with_conflicts:
            click.echo(f"    - {path.relative_to(ROOT_DIR)}")
        return 1

    click.echo("  No merge conflicts")
    return 0


def check_toml_yaml(files: list[pathlib.Path], fix: bool) -> int:
    config_files = [f for f in files if f.suffix in [".toml", ".yaml", ".yml"]]
    if not config_files:
        return 0

    click.echo("\nTOML and YAML:")
    errors = []

    # levanter is weird
    def include_constructor(loader, node):
        filepath = loader.construct_scalar(node)
        # Resolve relative to the current YAML file's directory
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
        click.echo("  Syntax errors:")
        for path, error in errors:
            click.echo(f"    - {path.relative_to(ROOT_DIR)}: {error}")
        return 1

    click.echo("  All files valid")
    return 0


def check_trailing_whitespace(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nTrailing whitespace:")
    files_with_whitespace = []

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
                # Check if original file ended with newline
                file_ended_with_newline = lines[-1].endswith("\n") if lines else True

                with open(file_path, "w") as f:
                    for i, line in enumerate(lines):
                        is_last_line = i == len(lines) - 1
                        cleaned = line.rstrip()

                        if is_last_line and not file_ended_with_newline:
                            # Last line didn't have newline, don't add one
                            f.write(cleaned)
                        else:
                            # Normal case: preserve the newline
                            f.write(cleaned + "\n")

    if files_with_whitespace:
        if not fix:
            click.echo(f"  {len(files_with_whitespace)} files with trailing whitespace")
            for f in files_with_whitespace:
                click.echo(f"    - {f.relative_to(ROOT_DIR)}")
        return 1

    click.echo("  No trailing whitespace")
    return 0


def check_eof_newline(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nEnd-of-file newline:")
    files_missing_newline = []

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
            click.echo(f"  Warning: Could not read file: {file_path}")

    if files_missing_newline:
        if not fix:
            click.echo(f"  {len(files_missing_newline)} files missing newline")
            for f in files_missing_newline:
                click.echo(f"    - {f.relative_to(ROOT_DIR)}")
        return 1

    click.echo("  All files have newlines")
    return 0


def check_pyrefly(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    click.echo("\nPyrefly type checker:")
    args = ["uv", "run", "--all-packages", "pyrefly", "check", "--baseline", ".pyrefly-baseline.json"]
    return run_cmd(args).returncode


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
        patterns=["lib/marin/src/**/*.py", "lib/levanter/src/**/*.py"],
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
        # Filter out non-existent files before running checks
        matched_files = [f for f in matched_files if f.exists()]
        if not matched_files:
            continue

        for check in config.checks:
            try:
                exit_code = check(matched_files, fix)
                exit_codes.append(exit_code)
            except Exception as e:
                click.echo(f"\nError running check {check.__name__}: {e}")
                exit_codes.append(1)

    click.echo("\n" + "=" * 60)
    if any(exit_codes):
        click.echo("FAILED: Some checks failed or files were modified")
        click.echo("=" * 60)
        sys.exit(1)
    else:
        click.echo("SUCCESS: All checks passed")
        click.echo("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
