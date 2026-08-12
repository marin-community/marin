# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared policy for generated dependency update pull requests."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class DependencyUpdate(StrEnum):
    EXTERNAL_RUNTIME = "external-runtime"
    NATIVE_PACKAGE = "native-package"


@dataclass(frozen=True)
class PullRequestPolicy:
    base_branch: str
    head_branch: str
    title: str
    allowed_files: frozenset[str]


EXTERNAL_RUNTIME_POLICY = PullRequestPolicy(
    base_branch="main",
    head_branch="automation/external-dependencies",
    title="[dependencies] Advance external runtimes",
    allowed_files=frozenset(
        {
            "config/external/MarinSkyRL/uv.lock",
            "config/external/evalchemy/uv.lock",
            "config/external/harbor/uv.lock",
            "lib/marin/src/marin/external_dependencies.py",
            "pyproject.toml",
            "uv.lock",
        }
    ),
)
NATIVE_PACKAGE_POLICY = PullRequestPolicy(
    base_branch="main",
    head_branch="automation/native-package-versions",
    title="[dependencies] Advance native package versions",
    allowed_files=frozenset(
        {
            "lib/dupekit/pyproject.toml",
            "lib/iris/pyproject.toml",
            "uv.lock",
        }
    ),
)
DEPENDENCY_UPDATE_POLICIES = MappingProxyType(
    {
        DependencyUpdate.EXTERNAL_RUNTIME: EXTERNAL_RUNTIME_POLICY,
        DependencyUpdate.NATIVE_PACKAGE: NATIVE_PACKAGE_POLICY,
    }
)

REQUIRED_CHECKS = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
GITHUB_ACTIONS_APP_ID = 15368
