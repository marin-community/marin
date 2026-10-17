# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared policy for generated dependency update pull requests."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class DependencyUpdate(StrEnum):
    EXTERNAL_RUNTIME = "external-runtime"
    NATIVE_PACKAGE = "native-package"


class ExternalRuntime(StrEnum):
    """Projects admitted to the external-runtime bot's file policy."""

    MARIN_SKYRL = "MarinSkyRL"
    EVALCHEMY = "evalchemy"
    HARBOR = "harbor"


@dataclass(frozen=True)
class PullRequestPolicy:
    base_branch: str
    head_branch: str
    title: str
    allowed_files: frozenset[str]


EXTERNAL_RUNTIME_POLICIES = MappingProxyType(
    {
        project: PullRequestPolicy(
            base_branch="main",
            head_branch=f"automation/external-dependencies-{project.value.lower()}",
            title=f"[dependencies] Advance {project.value}",
            allowed_files=frozenset(
                {
                    f"config/external/{project.value}/uv.lock",
                    "lib/marin/src/marin/external_dependencies.py",
                }
            ),
        )
        for project in ExternalRuntime
    }
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
REQUIRED_CHECKS = (
    "marin-docs",
    "marin-integration",
    "marin-lint",
    "rust-checks",
    "unit-tests",
)
GITHUB_ACTIONS_APP_ID = 15368
