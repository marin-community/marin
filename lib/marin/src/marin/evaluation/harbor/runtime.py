# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaluator sources that identify a cached Harbor evaluation."""

from pathlib import Path

from rigging.config_discovery import find_project_root

from marin.evaluation.harbor.driver_config import HARBOR_RUNTIME_PROJECT
from marin.evaluation.runtime import source_content_hash

_RUNTIME_DIRECTORIES = (
    "vendor/harbor/src",
    "lib/finestore/src",
    "lib/rigging/src",
    "lib/shellbox/src",
    "lib/marin/src/marin/evaluation/harbor",
)
_RUNTIME_FILES = (
    "vendor/harbor/pyproject.toml",
    "vendor/harbor/uv.lock",
    "lib/finestore/pyproject.toml",
    "lib/rigging/pyproject.toml",
    "lib/shellbox/pyproject.toml",
    "lib/marin/src/marin/evaluation/runtime.py",
    "lib/marin/src/marin/evaluation/rollouts.py",
    "lib/marin/src/marin/evaluation/records.py",
    "lib/marin/src/marin/evaluation/runner.py",
    "lib/marin/src/marin/evaluation/eval_env.py",
)


def harbor_runtime_source_hash() -> str:
    """Identify evaluator, scoring, execution and storage sources used by Harbor."""
    root = find_project_root(Path(__file__))
    if root is None:
        raise ValueError("Vendored Harbor requires a Marin source checkout")
    return source_content_hash(root, _RUNTIME_DIRECTORIES, _RUNTIME_FILES)


def harbor_runtime_descriptor(source_hash: str) -> str:
    return f"harbor @ {HARBOR_RUNTIME_PROJECT}#sha256={source_hash}"
