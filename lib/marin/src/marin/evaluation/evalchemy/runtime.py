# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent, checkout-local Evalchemy runtime and content provenance."""

from pathlib import Path

from rigging.config_discovery import find_project_root

from marin.evaluation.runtime import source_content_hash

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
    "lib/marin/src/marin/evaluation/runtime.py",
    "lib/marin/src/marin/evaluation/evalchemy/client.py",
    "lib/marin/src/marin/evaluation/evalchemy/runner.py",
    "lib/marin/src/marin/evaluation/evalchemy/runtime.py",
)


def _source_hash(directories: tuple[str, ...], files: tuple[str, ...]) -> str:
    root = find_project_root(Path(__file__))
    if root is None:
        raise ValueError("Vendored Evalchemy requires a Marin source checkout")
    return source_content_hash(root, directories, files)


def evalchemy_evaluator_source_hash() -> str:
    """Identify evaluator inputs used to cache immutable evaluation steps."""
    return _source_hash(_EVALUATOR_DIRECTORIES, _EVALUATOR_FILES)


def evalchemy_runtime_source_hash() -> str:
    """Identify evaluator, storage dependencies and Marin adapters for provenance."""
    return _source_hash(_RUNTIME_DIRECTORIES, _RUNTIME_FILES)


def evalchemy_runtime_descriptor(extras: tuple[str, ...], source_hash: str) -> str:
    selected = ",".join(sorted(set(extras)))
    return f"evalchemy[{selected}] @ {EVALCHEMY_RUNTIME_PROJECT}#sha256={source_hash}"
