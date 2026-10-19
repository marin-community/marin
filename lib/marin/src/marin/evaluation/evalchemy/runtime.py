# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent, checkout-local Evalchemy runtime and content provenance."""

from pathlib import Path

from rigging.cache import combined_content_hash, file_content_hash
from rigging.config_discovery import find_project_root

EVALCHEMY_RUNTIME_PROJECT = "vendor/evalchemy"
EVALCHEMY_REQUIRED_EXTRAS = ("serve-eval",)
EVALCHEMY_PYTHON_VERSION = "3.12"

_RUNTIME_DIRECTORIES = (
    "vendor/evalchemy/eval",
    "vendor/evalchemy/packages/evalchemy-config/src/evalchemy_config",
    "lib/finestore/src",
    "lib/rigging/src",
)
_RUNTIME_FILES = (
    "vendor/evalchemy/pyproject.toml",
    "vendor/evalchemy/uv.lock",
    "lib/finestore/pyproject.toml",
    "lib/rigging/pyproject.toml",
    "lib/marin/src/marin/evaluation/evalchemy/client.py",
    "lib/marin/src/marin/evaluation/evalchemy/runner.py",
    "lib/marin/src/marin/evaluation/evalchemy/runtime.py",
)


def evalchemy_runtime_source_hash() -> str:
    """Identify evaluator code, its independent lock, and local package sources."""
    root = find_project_root(Path(__file__))
    if root is None:
        raise ValueError("Vendored Evalchemy requires a Marin source checkout")
    components = []
    for directory in _RUNTIME_DIRECTORIES:
        source = root / directory
        if not source.is_dir():
            raise FileNotFoundError(f"Evalchemy runtime source directory is unavailable: {source}")
        for candidate in sorted(source.rglob("*")):
            relative = candidate.relative_to(root)
            # Optional local benchmark packages generate build metadata while uv
            # resolves their dependencies. Only source inputs identify a runtime.
            if (
                any(
                    part in {"__pycache__", ".pytest_cache", ".venv", "build", "dist"}
                    or part.endswith((".egg-info", ".dist-info"))
                    for part in relative.parts
                )
                or candidate.suffix == ".pyc"
            ):
                continue
            if candidate.is_file():
                components.append(f"{relative.as_posix()}:{file_content_hash(candidate)}")
    components.extend(f"{path}:{file_content_hash(root / path)}" for path in _RUNTIME_FILES)
    return combined_content_hash(components)


def evalchemy_runtime_descriptor(extras: tuple[str, ...], source_hash: str) -> str:
    selected = ",".join(sorted(set(extras)))
    return f"evalchemy[{selected}] @ {EVALCHEMY_RUNTIME_PROJECT}#sha256={source_hash}"
