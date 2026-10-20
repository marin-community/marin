# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent, checkout-local Evalchemy runtime and content provenance."""

import subprocess
from pathlib import Path

from iris.cluster.client.bundle import DEFAULT_EXCLUDE
from rigging.cache import combined_content_hash, file_content_hash
from rigging.config_discovery import find_project_root

EVALCHEMY_RUNTIME_PROJECT = "vendor/evalchemy"
EVALCHEMY_REQUIRED_EXTRAS = ("serve-eval",)
EVALCHEMY_PYTHON_VERSION = "3.12"

_EVALUATOR_DIRECTORIES = (
    "vendor/evalchemy/eval",
    "vendor/evalchemy/packages",
)
_EVALUATOR_FILES = (
    "vendor/evalchemy/pyproject.toml",
    "vendor/evalchemy/uv.lock",
)
_RUNTIME_DIRECTORIES = (*_EVALUATOR_DIRECTORIES, "lib/finestore/src", "lib/rigging/src")
_RUNTIME_FILES = (
    *_EVALUATOR_FILES,
    "lib/finestore/pyproject.toml",
    "lib/rigging/pyproject.toml",
    "lib/marin/src/marin/evaluation/evalchemy/client.py",
    "lib/marin/src/marin/evaluation/evalchemy/runner.py",
    "lib/marin/src/marin/evaluation/evalchemy/runtime.py",
)


def _source_hash(directories: tuple[str, ...], files: tuple[str, ...]) -> str:
    root = find_project_root(Path(__file__))
    if root is None:
        raise ValueError("Vendored Evalchemy requires a Marin source checkout")
    for directory in directories:
        source = root / directory
        if not source.is_dir():
            raise FileNotFoundError(f"Evalchemy runtime source directory is unavailable: {source}")
    for filename in files:
        if not (root / filename).is_file():
            raise FileNotFoundError(f"Evalchemy runtime source file is unavailable: {root / filename}")

    if (root / ".git").exists():
        # Match Iris bundle membership: changed tracked files and nonignored new
        # files both ship. Gitignored scratch files do not identify the runtime.
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *directories, *files],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        candidates = {root / filename for filename in result.stdout.split("\0") if filename}
    else:
        # Extracted Iris bundles contain no Git metadata. Their shipped files
        # already reflect Git membership; inspect only the declared source roots.
        candidates = {root / filename for filename in files}
        candidates.update(candidate for directory in directories for candidate in (root / directory).rglob("*"))

    components = []
    for candidate in sorted(candidates):
        relative = candidate.relative_to(root)
        # Optional benchmark packages generate build metadata inside source roots.
        if DEFAULT_EXCLUDE.search(relative.as_posix()) or any(
            part in {"build", "dist"} or part.endswith((".egg-info", ".dist-info")) for part in relative.parts
        ):
            continue
        if candidate.is_file():
            components.append(f"{relative.as_posix()}:{file_content_hash(candidate)}")
    return combined_content_hash(components)


def evalchemy_evaluator_source_hash() -> str:
    """Identify evaluator inputs used to cache immutable evaluation steps."""
    return _source_hash(_EVALUATOR_DIRECTORIES, _EVALUATOR_FILES)


def evalchemy_runtime_source_hash() -> str:
    """Identify evaluator, storage dependencies and Marin adapters for provenance."""
    return _source_hash(_RUNTIME_DIRECTORIES, _RUNTIME_FILES)


def evalchemy_runtime_descriptor(extras: tuple[str, ...], source_hash: str) -> str:
    selected = ",".join(sorted(set(extras)))
    return f"evalchemy[{selected}] @ {EVALCHEMY_RUNTIME_PROJECT}#sha256={source_hash}"
