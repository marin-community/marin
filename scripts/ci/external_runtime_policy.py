# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared policy for generated external-runtime update pull requests."""

EXPECTED_BASE_BRANCH = "main"
EXPECTED_HEAD_BRANCH = "automation/external-dependencies"
EXPECTED_TITLE = "[dependencies] Advance external runtimes"
EXPECTED_FILES = frozenset(
    {
        "config/external/MarinSkyRL/uv.lock",
        "config/external/evalchemy/uv.lock",
        "config/external/harbor/uv.lock",
        "lib/marin/src/marin/external_dependencies.py",
        "pyproject.toml",
        "uv.lock",
    }
)
REQUIRED_CHECKS = ("marin-integration", "marin-lint", "rust-checks", "unit-tests")
GITHUB_ACTIONS_APP_ID = 15368
