#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
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
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import click
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from linter import LINT_REVIEW_AGENT_DEFAULT, run_lint_review

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


def get_staged_files() -> list[pathlib.Path]:
    """Get list of staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT_DIR / f for f in result.stdout.strip().split("\n") if f]


def get_unstaged_files() -> list[pathlib.Path]:
    """Get list of unstaged (tracked) changes."""
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACM"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT_DIR / f for f in result.stdout.strip().split("\n") if f]


def get_branch_files() -> list[pathlib.Path]:
    """Get list of branch-specific changes (compared to merge-base with origin/main)."""
    base_result = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    merge_base = base_result.stdout.strip()

    result = subprocess.run(
        ["git", "diff", f"{merge_base}...HEAD", "--name-only", "--diff-filter=ACM"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT_DIR / f for f in result.stdout.strip().split("\n") if f]


def get_changed_files() -> list[pathlib.Path]:
    """Get list of staged, unstaged (tracked), and branch-specific changes."""
    files: set[pathlib.Path] = set()
    files.update(get_staged_files())
    files.update(get_unstaged_files())
    files.update(get_branch_files())
    return [f for f in files if f.exists()]


def get_all_files() -> list[pathlib.Path]:
    """Get list of all tracked files in the repository."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    files = [ROOT_DIR / f for f in result.stdout.strip().split("\n") if f]
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
    assert license_file is not None, "license_file must be provided"

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
        has_current_header = license_template in comment_block
        if not has_current_header:
            files_without_header.append(file_path)

            if fix:
                has_shebang = content.startswith("#!")
                shebang_line = ""
                if has_shebang:
                    shebang_line = lines[0]
                    rest_content = "\n".join(lines[1:])
                else:
                    rest_content = content

                if not rest_content.startswith(expected_header):
                    new_rest_content = f"{expected_header}\n{rest_content}" if rest_content else expected_header
                else:
                    new_rest_content = rest_content

                if has_shebang:
                    new_content = f"{shebang_line}\n{new_rest_content}"
                else:
                    new_content = new_rest_content

                with open(file_path, "w") as f:
                    f.write(new_content)

    if files_without_header:
        buf.write(f"{len(files_without_header)} files missing license headers\n")
        for f in files_without_header:
            buf.write(f"  - {f.relative_to(ROOT_DIR)}\n")
        return _record(label, 1, buf.getvalue())

    return _record(label, 0)


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


def check_json(files: list[pathlib.Path], fix: bool) -> int:
    """Format JSON with a canonical 2-space indent and validate it parses.

    Object key order is preserved (Grafana panel arrays are order-sensitive and
    object keys carry no meaning, so reordering would only churn). Scoped by the
    caller to hand-edited dashboard JSON; a repo-wide sweep would rewrite fixtures.
    """
    json_files = [f for f in files if f.suffix == ".json"]
    if not json_files:
        return 0

    buf = io.StringIO()
    invalid = []
    reformatted = []
    for file_path in json_files:
        try:
            with open(file_path) as f:
                original = f.read()
            data = json.loads(original)
        except (OSError, json.JSONDecodeError) as e:
            invalid.append((file_path, str(e)))
            continue

        formatted = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        if formatted == original:
            continue
        reformatted.append(file_path)
        if fix:
            with open(file_path, "w") as f:
                f.write(formatted)

    if invalid:
        for path, error in invalid:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}: {error}\n")
        return _record("JSON formatter", 1, buf.getvalue())

    if reformatted:
        for path in reformatted:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}\n")
        return _record("JSON formatter", 1, buf.getvalue())

    return _record("JSON formatter", 0)


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


def check_markdown_precommit_invocation(files: list[pathlib.Path], fix: bool) -> int:
    md_files = [f for f in files if f.suffix == ".md"]
    if not md_files:
        return 0

    pattern = re.compile(r"\buv run(?:\s+python)?\s+(?:\./)?infra/pre-commit\.py\b")
    bad_refs: list[tuple[pathlib.Path, int, str]] = []

    for file_path in md_files:
        try:
            with open(file_path) as f:
                lines = f.readlines()
        except Exception:
            continue

        updated_lines = lines.copy()
        file_changed = False
        for i, line in enumerate(lines):
            if not pattern.search(line):
                continue
            bad_refs.append((file_path, i + 1, line.rstrip("\n")))
            if fix:
                new_line = pattern.sub("./infra/pre-commit.py", line)
                if new_line != line:
                    updated_lines[i] = new_line
                    file_changed = True

        if fix and file_changed:
            with open(file_path, "w") as f:
                f.writelines(updated_lines)

    if bad_refs:
        buf = io.StringIO()
        buf.write("Use ./infra/pre-commit.py directly in docs; do not prefix with uv/python:\n")
        for path, line_no, line in bad_refs:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}:{line_no}: {line}\n")
        return _record("Markdown pre-commit command", 1, buf.getvalue())

    return _record("Markdown pre-commit command", 0)


SKILL_REFERENCE_PATTERNS = [
    re.compile(r"`((?:\.agents|\.claude|lib|experiments|docs|scripts|infra|src|tests)/[^`\s,;:)]+)`"),
    re.compile(r"\]\(((?:\.agents|\.claude|lib|experiments|docs|scripts|infra|src|tests)/[^)]+)\)"),
]
SKILL_REFERENCE_PLACEHOLDERS = [
    "<",
    "YYYY",
    "...",
    "foo.md",
    "profile_summary.v1",
    "summary.md",
    "graphs.jsonl",
    "tasks.jsonl",
]
SKILL_REFERENCE_ALLOWLIST = {
    ".agents/ops/logs/",
}


def _is_skill_file(file_path: pathlib.Path) -> bool:
    relative_parts = file_path.relative_to(ROOT_DIR).parts
    return file_path.name == "SKILL.md" and len(relative_parts) == 4 and relative_parts[:2] == (".agents", "skills")


def _skill_reference_exists(skill_path: pathlib.Path, reference: str) -> bool:
    candidates = [skill_path.parent / reference, ROOT_DIR / reference]
    return any(candidate.exists() for candidate in candidates)


def _should_skip_skill_reference(reference: str) -> bool:
    return (
        reference in SKILL_REFERENCE_ALLOWLIST
        or any(token in reference for token in SKILL_REFERENCE_PLACEHOLDERS)
        or any(character in reference for character in "*{}")
    )


def check_skill_metadata(files: list[pathlib.Path], fix: bool) -> int:
    skill_files = [f for f in files if _is_skill_file(f)]
    if not skill_files:
        return 0

    all_skill_files = sorted((ROOT_DIR / ".agents" / "skills").glob("*/SKILL.md"))
    errors: list[tuple[pathlib.Path, str]] = []
    names: dict[str, list[pathlib.Path]] = {}

    for file_path in all_skill_files:
        try:
            text = file_path.read_text()
        except Exception as e:
            errors.append((file_path, f"could not read file: {e}"))
            continue

        if re.search(r"\.agents/project(?!s)", text):
            errors.append((file_path, "use .agents/projects/, not .agents/project/"))

        if "lib/finelog/src/finelog/proto/stats.proto" in text:
            errors.append(
                (
                    file_path,
                    "finelog stats proto path is lib/finelog/src/finelog/proto/finelog_stats.proto",
                )
            )

        for pattern in SKILL_REFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                reference = match.group(1).split("#", 1)[0]
                if _should_skip_skill_reference(reference):
                    continue
                if not _skill_reference_exists(file_path, reference):
                    errors.append((file_path, f"missing local reference: {reference}"))

        if not text.startswith("---\n"):
            errors.append((file_path, "missing opening frontmatter delimiter"))
            continue

        parts = text.split("---", 2)
        if len(parts) < 3:
            errors.append((file_path, "missing closing frontmatter delimiter"))
            continue

        try:
            metadata = yaml.safe_load(parts[1])
        except Exception as e:
            errors.append((file_path, f"invalid YAML frontmatter: {e}"))
            continue

        if not isinstance(metadata, dict):
            errors.append((file_path, f"frontmatter must be a YAML mapping, got {type(metadata).__name__}"))
            continue

        name = metadata.get("name")
        description = metadata.get("description")

        if not isinstance(name, str) or not name.strip():
            errors.append((file_path, f"frontmatter must include a non-empty string name, got {name!r}"))
        else:
            if name != file_path.parent.name:
                errors.append((file_path, f"name {name!r} must match directory name {file_path.parent.name!r}"))
            names.setdefault(name, []).append(file_path)

        if not isinstance(description, str) or not description.strip():
            errors.append((file_path, f"frontmatter must include a non-empty string description, got {description!r}"))
        elif "\n" in description:
            errors.append((file_path, "description must be a single-line string"))

        for key in ("schedule_cron", "schedule_tz"):
            value = metadata.get(key)
            if value is not None and not isinstance(value, str):
                errors.append((file_path, f"{key} must be a string, got {type(value).__name__}"))

        if ("schedule_cron" in metadata) != ("schedule_tz" in metadata):
            errors.append((file_path, "schedule_cron and schedule_tz must be specified together"))

        allowed_tools = metadata.get("allowed-tools")
        if allowed_tools is not None and not isinstance(allowed_tools, str):
            errors.append((file_path, f"allowed-tools must be a string, got {type(allowed_tools).__name__}"))

    for name, paths in names.items():
        if len(paths) <= 1:
            continue
        joined_paths = ", ".join(str(p.relative_to(ROOT_DIR)) for p in paths)
        for path in paths:
            errors.append((path, f"duplicate skill name {name!r}: {joined_paths}"))

    if errors:
        checked_paths = {path for path in skill_files}
        relevant_errors = [(path, error) for path, error in errors if path in checked_paths]
        if not relevant_errors:
            relevant_errors = errors

        buf = io.StringIO()
        for path, error in relevant_errors:
            buf.write(f"  - {path.relative_to(ROOT_DIR)}: {error}\n")
        return _record("Skill metadata", 1, buf.getvalue())

    return _record("Skill metadata", 0)


def _ensure_iris_protos() -> None:
    """Generate iris protobuf files if they are missing and npx is available.

    Pyrefly needs the generated *_pb2.py and *_connect.py files on disk to
    resolve imports from other modules, even when the generated files themselves
    are excluded from type-checking via project-excludes.
    """
    rpc_dir = ROOT_DIR / "lib" / "iris" / "src" / "iris" / "rpc"
    # Check if any pb2 file exists already
    if list(rpc_dir.glob("*_pb2.py")):
        return

    generate_script = ROOT_DIR / "lib" / "iris" / "scripts" / "generate_protos.py"
    if not generate_script.exists():
        return

    if shutil.which("npx") is None:
        print("  ⚠ Iris protobuf files are missing and npx is not installed; pyrefly may report false errors")
        return

    print("  Generating iris protobuf files for type checking...")
    result = subprocess.run(
        [sys.executable, str(generate_script)],
        cwd=ROOT_DIR / "lib" / "iris",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ⚠ Proto generation failed: {result.stderr.strip()}")


def check_pyrefly(files: list[pathlib.Path], fix: bool) -> int:
    if not files:
        return 0

    _ensure_iris_protos()

    args = [
        "uvx",
        "--from",
        "pyrefly>=1.0.0,<1.1.0",
        "pyrefly",
        "check",
        "--baseline",
        ".pyrefly-baseline.json",
    ]
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
            partial(check_black, config=LEVANTER_BLACK_CONFIG),
            partial(check_license_headers, license_file=LEVANTER_LICENSE),
        ],
    ),
    PrecommitConfig(
        patterns=["lib/haliax/**/*.py"],
        checks=[
            check_ruff,
            partial(check_black, config=HALIAX_BLACK_CONFIG),
            partial(check_license_headers, license_file=HALIAX_LICENSE),
        ],
    ),
    PrecommitConfig(
        patterns=["**/*.py"],
        exclude_patterns=["lib/levanter/**", "lib/haliax/**", "lib/**/vendor/**"],
        checks=[
            check_ruff,
            check_black,
            partial(check_license_headers, license_file=MARIN_LICENSE),
        ],
    ),
    PrecommitConfig(
        patterns=[
            "lib/marin/src/**/*.py",
            "lib/levanter/src/**/*.py",
            "lib/haliax/src/**/*.py",
            "lib/fray/src/**/*.py",
            "lib/iris/src/**/*.py",
            "lib/rigging/src/**/*.py",
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
        patterns=["infra/grafana/dashboards/*.json"],
        checks=[
            check_json,
        ],
    ),
    PrecommitConfig(
        patterns=["**/*.ipynb"],
        checks=[
            check_notebooks,
        ],
    ),
    PrecommitConfig(
        patterns=["**/*.md"],
        checks=[
            check_markdown_precommit_invocation,
        ],
    ),
    PrecommitConfig(
        patterns=[".agents/skills/*/SKILL.md"],
        checks=[
            check_skill_metadata,
        ],
    ),
]


def _get_check_name(check) -> str:
    if isinstance(check, partial):
        return check.func.__name__
    return check.__name__


@click.command()
@click.option("--fix", is_flag=True, help="Automatically fix issues where possible")
@click.option("--all-files", is_flag=True, help="Run checks on all files, not just changed")
@click.option(
    "--changed-files",
    "changed_files",
    is_flag=True,
    help="Run checks on uncommitted and branch-specific changes",
)
@click.option(
    "--pre-commit",
    is_flag=True,
    help="Run checks on staged changes only (for git pre-commit hook)",
)
@click.option(
    "--review",
    is_flag=True,
    help="Run the advisory infra/lint/ rule catalog over the branch diff via per-lane agents",
)
@click.option(
    "--agent-command",
    "agent_command",
    default=LINT_REVIEW_AGENT_DEFAULT,
    show_default=True,
    help=(
        "Headless agent invocation for --review. Agents calling this from a skill "
        "should pass their own command (e.g. 'claude -p', 'codex exec') so the "
        "sub-agent matches the calling environment."
    ),
)
@click.option(
    "--lint-lane",
    "lint_lanes",
    multiple=True,
    help="With --review, run only these lane(s): complexity, interfaces, robustness, cruft, prose, meta. Repeatable.",
)
@click.option(
    "--lint-compose/--no-lint-compose",
    "lint_compose",
    default=True,
    show_default=True,
    help="With --review, merge lanes via the composer agent (default) or a deterministic concat.",
)
@click.option("--files", "files_opt", multiple=True, help="Files to check (alias for positional args)")
@click.option("--skip", multiple=True, help="Skip specific checks by name (e.g. ruff, black)")
@click.option("--only", multiple=True, help="Run only specific checks by name (e.g. ruff, black)")
@click.argument("files", nargs=-1)
def main(
    fix: bool,
    all_files: bool,
    changed_files: bool,
    pre_commit: bool,
    review: bool,
    agent_command: str,
    lint_lanes: tuple[str, ...],
    lint_compose: bool,
    files_opt: tuple[str, ...],
    skip: tuple[str, ...],
    only: tuple[str, ...],
    files: tuple[str, ...],
):
    if review:
        sys.exit(run_lint_review(agent_command, lane_names=list(lint_lanes) or None, compose=lint_compose))

    if skip and only:
        click.echo("Error: --only and --skip are mutually exclusive.", err=True)
        sys.exit(1)

    known_check_names = {_get_check_name(c) for cfg in PRECOMMIT_CONFIGS for c in cfg.checks}
    known_short_names = sorted(name.removeprefix("check_") for name in known_check_names)
    for s in skip:
        if f"check_{s}" not in known_check_names:
            click.echo(f"Error: unknown --skip value '{s}'. Valid names: {', '.join(known_short_names)}", err=True)
            sys.exit(1)
    for o in only:
        if f"check_{o}" not in known_check_names:
            click.echo(f"Error: unknown --only value '{o}'. Valid names: {', '.join(known_short_names)}", err=True)
            sys.exit(1)

    all_files_set: set[pathlib.Path] = set()
    input_files = files_opt + files

    if all_files:
        all_files_set.update(get_all_files())
    elif pre_commit:
        all_files_set.update(get_staged_files())
    elif changed_files or not input_files:
        # This is the default behavior if no arguments are provided.
        all_files_set.update(get_changed_files())

    if input_files:
        for f in input_files:
            path = ROOT_DIR / f
            if path.exists():
                all_files_set.add(path)
            else:
                click.echo(f"Warning: Skipping non-existent file: {f}")

    all_files_list = sorted(list(all_files_set))
    exit_codes = []

    skip_set = set(f"check_{s}" for s in skip)
    only_set = set(f"check_{o}" for o in only)

    for config in PRECOMMIT_CONFIGS:
        matched_files = get_matching_files(config.patterns, all_files_list, config.exclude_patterns)
        matched_files = [f for f in matched_files if f.exists()]
        if not matched_files:
            continue

        for check in config.checks:
            check_name = _get_check_name(check)
            if only_set and check_name not in only_set:
                continue
            if check_name in skip_set:
                continue
            try:
                exit_code = check(matched_files, fix)
                exit_codes.append(exit_code)
            except Exception as e:
                click.echo(f"  Error running check {check_name}: {e}")
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
